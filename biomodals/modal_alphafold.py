# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs AlphaFold2 or AF2-multimer predictions using ColabFold on Modal.

Limitations:
- It requires only one entry in a fasta file.
- If providing a complex, e.g., a binder and target pair,
  Provide the target first, then N binders after, separated by ":"
"""

import contextlib
import os
import zipfile
from pathlib import Path

import modal
from modal import App, Image

GPU = os.environ.get("GPU", "A100")
TIMEOUT = os.environ.get("TIMEOUT", 30)

image = (
    Image.micromamba(python_version="3.11")
    .apt_install("wget", "git")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold@a134f6a8f8de5c41c63cb874d07e1a334cb021bb"
    )
    .micromamba_install("kalign2=2.04", "hhsuite=3.3.0", channels=["conda-forge", "bioconda"])
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a10g",
    )
    # CRITICAL: Force downgrade numpy to <2.0 AFTER jax installation
    # This fixes the "np.sum(generator) is deprecated" error
    .run_commands('pip install --force-reinstall "numpy<2.0"')
    .pip_install("google-cloud-storage==2.14.0")
    .run_commands("python -m colabfold.download")
)

app = App("alphafold", image=image)


def score_af2m_binding(af2m_dict: dict, target_len: int, binders_len: list[int]) -> dict:
    """Calculates binding scores from AlphaFold2 multimer prediction results.

    The target is assumed to be the first part of the sequence, followed by one or more binders.

    Args:
        af2m_dict (dict): Dictionary loaded from an AlphaFold2 multimer JSON output file (usually contains 'plddt' and 'pae' keys).
        target_len (int): Length of the target protein sequence.
        binders_len (list[int]): List of lengths for each binder protein sequence.

    Returns:
        dict: A dictionary containing various scores:
            - "plddt_binder" (dict[int, float]): Average pLDDT for each binder, keyed by binder index (0-based).
            - "plddt_target" (float): Average pLDDT for the target.
            - "pae_binder" (dict[int, float]): Average PAE within each binder, keyed by binder index.
            - "pae_target" (float): Average PAE within the target.
            - "ipae" (dict[int, float]): Average interface PAE between the target and each binder, keyed by binder index.
            - "ipae_binder" (dict[int, list[float]]): Per-residue interface PAE scores for each binder interacting with the target, keyed by binder index.
    """

    import numpy as np

    plddt_array = np.array(af2m_dict["plddt"])
    pae_array = np.array(af2m_dict["pae"])

    assert len(plddt_array) == len(pae_array) == target_len + sum(binders_len)

    plddt_target = np.mean(plddt_array[:target_len])
    pae_target = np.mean(pae_array[:target_len, :target_len])

    plddt_binder = {}
    pae_binder = {}
    ipae = {}
    ipae_binder = {}

    current_pos = target_len
    for binder_n, binder_len in enumerate(binders_len):
        binder_start, binder_end = current_pos, current_pos + binder_len

        # --------------------------------------------------------------------------
        # pLDDT; binder
        #

        plddt_binder[binder_n] = np.mean(plddt_array[binder_start:binder_end])

        # --------------------------------------------------------------------------
        # PAE; binder vs itself; mean target<>binder; target<>binder separately
        #
        pae_binder[binder_n] = np.mean(pae_array[binder_start:binder_end, binder_start:binder_end])
        ipae[binder_n] = np.mean(
            [
                np.mean(pae_array[:target_len, binder_start:binder_end]),
                np.mean(pae_array[binder_start:binder_end, :target_len]),
            ]
        )

        ipae_binder[binder_n] = np.mean(
            [
                np.mean(pae_array[:target_len, binder_start:binder_end], axis=0),
                np.mean(pae_array[binder_start:binder_end, :target_len], axis=1),
            ],
            axis=0,
        )
        current_pos += binder_len

    return {
        "plddt_binder": {k: float(v) for k, v in plddt_binder.items()},
        "plddt_target": float(plddt_target),
        "pae_binder": {k: float(v) for k, v in pae_binder.items()},
        "pae_target": float(pae_target),
        "ipae": {k: float(v) for k, v in ipae.items()},
        "ipae_binder": {
            k: [float(ipae_b) for ipae_b in ipae_binder[k]] for k, v in ipae_binder.items()
        },
    }


# Maps CLI-friendly identifiers to the strings expected by ColabFold's run()
_MSA_MODE_MAP: dict[tuple[str, str], str] = {
    ("mmseqs2", "uniref_env"): "MMseqs2 (UniRef+Environmental)",
    ("mmseqs2", "uniref_only"): "MMseqs2 (UniRef only)",
    ("jackhmmer", "uniref_env"): "jackhmmer",
    ("jackhmmer", "uniref_only"): "jackhmmer",
}
_MODEL_TYPE_MAP: dict[str, str] = {
    "auto": "auto",
    "ptm": "AlphaFold2-ptm",
    "multimer_v1": "AlphaFold2-multimer-v1",
    "multimer_v2": "AlphaFold2-multimer-v2",
    "multimer_v3": "AlphaFold2-multimer-v3",
}
_PAIR_MODE_MAP: dict[str, str] = {
    "unpaired_paired": "unpaired+paired",
    "paired": "paired",
    "unpaired": "unpaired",
}


@app.function(
    image=image,
    gpu=GPU,
    timeout=int(TIMEOUT * 60),
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def alphafold(
    fasta_name: str = "input.fasta",
    fasta_str: str = "",
    fasta_gcs_uri: str = "",
    models: list[int] | str | None = None,
    num_recycles: int = 6,
    num_relax: int = 0,
    use_precomputed_msas: bool = False,
    return_all_files: bool = True,
    upload_to_gcs: bool = False,
    gcs_bucket: str = "dev-services",
    run_id: str | None = None,
    # Frontend-aligned params
    model_type: str = "auto",
    msa_tool: str = "mmseqs2",
    msa_databases: str = "uniref_env",
    template_mode: str = "none",
    pair_mode: str = "unpaired_paired",
    num_seeds: int = 3,
) -> dict:
    """Runs AlphaFold2/ColabFold prediction on Modal.

    Args:
        fasta_name (str): Name of the FASTA file (e.g., "protein.fasta").
        fasta_str (str): FASTA content. Separate chains with ":" for multimer mode —
                         ColabFold auto-detects complex prediction from this.
        fasta_gcs_uri (str): GCS URI to a FASTA file (gs://bucket/path/file.fa). If provided
                             and fasta_str is empty, the file is downloaded from GCS. When the
                             URI stem is used to derive fasta_name automatically.
        models (list[int] | str | None): Model numbers to run (1-5). Defaults to [1].
        num_recycles (int): Recycling iterations. Defaults to 6.
        num_relax (int): AMBER relaxation passes (0 = disabled, 1 = top model). Defaults to 0.
        use_precomputed_msas (bool): Reuse MSAs from a pre-mounted /msas directory. Defaults to False.
        return_all_files (bool): Return all output files; False returns only the ZIP. Defaults to True.
        model_type (str): "auto" | "ptm" | "multimer_v1" | "multimer_v2" | "multimer_v3".
                          "auto" selects ptm for monomers and multimer_v3 for complexes. Defaults to "auto".
        msa_tool (str): MSA algorithm — "mmseqs2" (fast) | "jackhmmer". Defaults to "mmseqs2".
        msa_databases (str): Database set — "uniref_env" (recommended) | "uniref_only". Defaults to "uniref_env".
        template_mode (str): Template lookup — "none" (disabled) | "pdb70". Defaults to "none".
        pair_mode (str): MSA pairing for complexes — "unpaired_paired" | "paired" | "unpaired".
                         Defaults to "unpaired_paired".
        num_seeds (int): Number of model seeds to run. Defaults to 3.

    Returns:
        dict: Structured output with output_files and metadata.
    """
    import glob
    import json
    import subprocess

    from colabfold.batch import get_queries, run
    from colabfold.download import default_data_dir

    # Parse models parameter (handle both string and list)
    if models is None:
        models_list: list[int] = [1]
    elif isinstance(models, str):
        # Parse comma-separated string: "1,2,3" -> [1, 2, 3]
        models_list = [int(m.strip()) for m in models.split(",") if m.strip()]
        if not models_list:
            models_list = [1]
    else:
        models_list = models

    in_dir = "/tmp/in_af"
    out_dir = "/tmp/out_af"
    Path(in_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Download FASTA from GCS if fasta_str not provided directly
    if not fasta_str and fasta_gcs_uri:
        import json as _json
        import os as _os

        import google.cloud.storage as _gcs

        print(f"⬇️  Downloading FASTA from GCS: {fasta_gcs_uri}")

        # Apply GCS credentials from Modal secret before creating client
        _creds_json = _os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if _creds_json:
            _creds_path = "/tmp/gcs_credentials.json"
            if not _os.path.exists(_creds_path):
                with open(_creds_path, "w") as _f:
                    _json.dump(_json.loads(_creds_json), _f)
            _os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _creds_path

        _parts = fasta_gcs_uri.replace("gs://", "").split("/", 1)
        _bucket_name, _obj_path = _parts[0], _parts[1]
        _gcs_client = _gcs.Client()
        _blob = _gcs_client.bucket(_bucket_name).blob(_obj_path)
        fasta_str = _blob.download_as_text()
        # Derive fasta_name from URI stem if still at default
        if fasta_name == "input.fasta":
            _uri_stem = fasta_gcs_uri.split("/")[-1].rsplit(".", 1)[0]
            fasta_name = f"{_uri_stem}.fasta"
        print(f"   ✓ Downloaded {len(fasta_str)} chars → {fasta_name}")

    # Validate FASTA input
    if not fasta_str or not fasta_str.strip():
        raise ValueError(
            "Empty FASTA input provided. AlphaFold requires valid FASTA sequences. "
            "Provide either fasta_str (content) or fasta_gcs_uri (GCS path)."
        )

    fasta_lines = fasta_str.splitlines()
    if not fasta_lines:
        raise ValueError(
            "FASTA input has no lines. AlphaFold requires valid FASTA sequences. "
            "Check that the upstream step produced valid output."
        )

    # saves the colabfold server, speeds things up
    if use_precomputed_msas:
        subprocess.run(f"cp -r /msas/* {out_dir}", shell=True)

    with open(Path(in_dir) / fasta_name, "w") as f:
        f.write(fasta_str)

    header = fasta_lines[0]
    fasta_seq = "".join(seq.strip() for seq in fasta_str.splitlines()[1:])
    if header[0] != ">" or any(aa not in "ACDEFGHIKLMNPQRSTVWY:" for aa in fasta_seq):
        raise AssertionError(f"invalid fasta:\n{fasta_str}")

    queries, is_complex = get_queries(in_dir)

    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # Map CLI-friendly identifiers to ColabFold API strings
    cf_msa_mode = _MSA_MODE_MAP.get((msa_tool, msa_databases), "MMseqs2 (UniRef+Environmental)")
    cf_model_type = _MODEL_TYPE_MAP.get(model_type, "auto")
    cf_pair_mode = _PAIR_MODE_MAP.get(pair_mode, "unpaired+paired")
    cf_use_templates = template_mode == "pdb70"

    run(
        queries=queries,
        result_dir=out_dir,
        use_templates=cf_use_templates,
        num_relax=num_relax,
        relax_max_iterations=200,
        msa_mode=cf_msa_mode,
        model_type=cf_model_type,
        num_models=len(models_list),
        num_recycles=num_recycles,
        num_seeds=num_seeds,
        model_order=models_list,
        is_complex=is_complex,
        data_dir=default_data_dir,
        keep_existing_results=False,
        rank_by="auto",
        pair_mode=cf_pair_mode,
        stop_at_score=100,
        zip_results=True,
        user_agent="colabfold/google-colab-batch",
    )

    # --------------------------------------------------------------------------
    # If binder_len is supplied, evaluate binder-target score using iPAE
    #
    af2m_scores = None
    if ":" in fasta_seq:  # then it is a multimer
        target_len = len(fasta_seq.split(":")[0])
        binders_len = [len(b_seq) for b_seq in fasta_seq.split(":")[1:]]

        results_zip = list(Path(out_dir).glob("**/*.zip"))
        assert len(results_zip) == 1, f"unexpected zip output: {results_zip}"

        with zipfile.ZipFile(results_zip[0], "a") as zip_ref:
            json_files = [f for f in zip_ref.namelist() if Path(f).suffix == ".json"]

            for json_file in json_files:
                json_data = json.loads(zip_ref.read(json_file))

                if "plddt" in json_data and "pae" in json_data:
                    prefix = Path(json_file).with_suffix("")
                    af2m_scores = score_af2m_binding(json_data, target_len, binders_len)
                    scores_json = json.dumps(af2m_scores, indent=2)
                    zip_ref.writestr(f"{prefix}.af2m_scores.json", scores_json)
                    break

    # Collect all output files
    all_files = glob.glob(f"{out_dir}/**/*", recursive=True)
    output_files = []

    for outfile in all_files:
        if not os.path.isfile(outfile):
            continue

        # Filter based on return_all_files flag
        if not return_all_files and Path(outfile).suffix != ".zip":
            continue

        file_path = Path(outfile)
        file_size = file_path.stat().st_size

        # Determine artifact type
        suffix = file_path.suffix.lower()
        if suffix == ".pdb":
            artifact_type = "pdb"
        elif suffix == ".json":
            artifact_type = "json"
        elif suffix == ".zip":
            artifact_type = "zip"
        elif suffix in [".a3m", ".sto"]:
            artifact_type = "msa"
        else:
            artifact_type = "file"

        # Determine if this is a ranked model
        is_ranked = "rank_" in file_path.name
        model_num = None
        if "model_" in file_path.name:
            with contextlib.suppress(ValueError, IndexError):
                model_num = int(file_path.name.split("model_")[1].split("_")[0])

        output_files.append(
            {
                "path": str(file_path.absolute()),
                "artifact_type": artifact_type,
                "filename": file_path.name,
                "size": file_size,
                "metadata": {
                    "is_ranked": is_ranked,
                    "model_num": model_num,
                    "is_complex": is_complex,
                    "num_recycles": num_recycles,
                },
            }
        )

    # Extract PDB files from zip files for downstream workflows
    print("\n📦 Extracting PDB structures from result zips...")
    extracted_pdbs = []

    for file_info in output_files:
        if file_info.get("artifact_type") == "zip":
            zip_path = file_info["path"]
            base_name = Path(
                str(file_info["filename"])
            ).stem  # e.g., "seq_50.result" from "seq_50.result.zip"

            try:
                with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
                    # Find the ranked PDB file (best model)
                    pdb_files = [
                        f
                        for f in zip_ref.namelist()
                        if f.endswith(
                            "_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
                        )
                        or f.endswith(
                            "_relaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
                        )
                    ]

                    if not pdb_files:
                        # Fallback: find any PDB file
                        pdb_files = [f for f in zip_ref.namelist() if f.endswith(".pdb")]

                    if pdb_files:
                        # Use the first (best) PDB
                        pdb_file = pdb_files[0]
                        extract_dir = Path(out_dir) / "structures"
                        extract_dir.mkdir(exist_ok=True)

                        # Create a clean filename
                        clean_name = f"{base_name}.pdb"
                        extracted_path = extract_dir / clean_name

                        extracted_path.write_bytes(zip_ref.read(pdb_file))

                        extracted_pdbs.append(
                            {
                                "path": str(extracted_path),
                                "artifact_type": "pdb",
                                "filename": clean_name,
                                "size": extracted_path.stat().st_size,
                                "metadata": {
                                    "extracted_from": file_info["filename"],
                                    "original_name": pdb_file,
                                    "is_complex": is_complex,
                                },
                            }
                        )
                        print(
                            f"    ✓ Extracted {clean_name} ({extracted_path.stat().st_size:,} bytes)"
                        )

            except Exception as e:
                print(f"    ⚠️  Failed to extract PDB from {file_info['filename']}: {e}")

    # Also extract metric JSON sidecars from each result zip:
    #   {stem}.colabfold_scores.json  — top-rank ColabFold JSON (has ptm, iptm)
    #   {stem}.af2m_scores.json       — custom binder-target interface metrics (multimer only)
    print("\n📊 Extracting score JSON sidecars...")
    extracted_scores: list[dict] = []
    for file_info in output_files:
        if file_info.get("artifact_type") != "zip":
            continue
        zip_path = file_info["path"]
        base_name = Path(str(file_info["filename"])).stem  # e.g. "binder_001_seq_1.result"
        try:
            with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
                all_names = zip_ref.namelist()
                scores_dir = Path(out_dir) / "scores"
                scores_dir.mkdir(exist_ok=True)

                # ColabFold top-rank scores JSON (has ptm, iptm, plddt per residue)
                cf_scores = [
                    f
                    for f in all_names
                    if f.endswith(".json") and "scores_rank_001" in f and "af2m_scores" not in f
                ]
                if cf_scores:
                    clean = f"{base_name}.colabfold_scores.json"
                    dest = scores_dir / clean
                    dest.write_bytes(zip_ref.read(cf_scores[0]))
                    extracted_scores.append(
                        {
                            "path": str(dest),
                            "artifact_type": "colabfold_scores",
                            "filename": clean,
                            "size": dest.stat().st_size,
                        }
                    )
                    print(f"    ✓ ColabFold scores → {clean}")

                # af2m_scores.json — written by score_af2m_binding (multimer only)
                af2m_json_files = [f for f in all_names if f.endswith(".af2m_scores.json")]
                if af2m_json_files:
                    clean = f"{base_name}.af2m_scores.json"
                    dest = scores_dir / clean
                    dest.write_bytes(zip_ref.read(af2m_json_files[0]))
                    extracted_scores.append(
                        {
                            "path": str(dest),
                            "artifact_type": "af2m_scores",
                            "filename": clean,
                            "size": dest.stat().st_size,
                        }
                    )
                    print(f"    ✓ AF2M scores → {clean}")
        except Exception as e:
            print(f"    ⚠️  Could not extract score sidecars from {file_info['filename']}: {e}")

    # Add extracted PDBs to output_files (these will be uploaded to GCS)
    if extracted_pdbs:
        output_files.extend(extracted_pdbs)
        print(f"  ✅ Total extracted: {len(extracted_pdbs)} PDB structures")
    if extracted_scores:
        output_files.extend(extracted_scores)
        print(f"  ✅ Total score sidecars: {len(extracted_scores)}")

    # Upload to GCS if requested
    gcs_files: list[dict[str, str]] = []
    if upload_to_gcs:
        from google.cloud import storage

        # Initialize GCS client with credentials from secret
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not credentials_json:
            return {
                "exit_code": 1,
                "error": "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                "output_files": output_files,
            }

        # Write credentials to temp file
        creds_path = "/tmp/gcs_credentials.json"
        with open(creds_path, "w") as f:
            json.dump(json.loads(credentials_json), f)

    if upload_to_gcs:
        from google.cloud import storage

        # Initialize GCS client with credentials from secret
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not credentials_json:
            return {
                "exit_code": 1,
                "error": "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                "output_files": output_files,
            }

        # Write credentials to temp file
        creds_path = "/tmp/gcs_credentials.json"
        with open(creds_path, "w") as f:
            f.write(credentials_json)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

        # Create GCS client
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)

        # Generate run_id if not provided
        if not run_id:
            from datetime import datetime

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Upload each file
        for file_info in output_files:
            local_path = file_info["path"]
            filename = file_info["filename"]

            # Construct GCS path
            blob_path = f"runs/{run_id}/alphafold/{filename}"
            blob = bucket.blob(blob_path)

            # Upload file
            blob.upload_from_filename(local_path)

            # Add GCS info
            gcs_files.append(
                {
                    "gcs_url": f"gs://{gcs_bucket}/{blob_path}",
                    "filename": str(filename),
                    "artifact_type": str(file_info.get("artifact_type", "")),
                    "size": str(file_info.get("size", 0)),
                    "metadata": str(file_info.get("metadata", {})),
                }
            )

        _common_meta = {
            "models": models_list,
            "model_type": cf_model_type,
            "msa_mode": cf_msa_mode,
            "template_mode": template_mode,
            "pair_mode": cf_pair_mode,
            "num_recycles": num_recycles,
            "num_seeds": num_seeds,
            "num_relax": num_relax,
            "is_complex": is_complex,
            "af2m_scores": af2m_scores,
            "run_id": run_id,
            "gcs_bucket": gcs_bucket,
        }
        return {
            "exit_code": 0,
            "output_files": gcs_files,
            "gcs_prefix": f"gs://{gcs_bucket}/runs/{run_id}/alphafold/",
            "message": f"AlphaFold completed. Uploaded {len(gcs_files)} files to GCS.",
            "stdout": f"AlphaFold completed successfully. Files uploaded to gs://{gcs_bucket}/runs/{run_id}/alphafold/",
            "metadata": _common_meta,
        }

    return {
        "exit_code": 0,
        "output_files": output_files,
        "message": f"AlphaFold completed. Generated {len(output_files)} output files.",
        "stdout": f"AlphaFold prediction completed with {len(models_list)} models and {num_recycles} recycles.",
        "metadata": {
            "models": models_list,
            "model_type": cf_model_type,
            "msa_mode": cf_msa_mode,
            "template_mode": template_mode,
            "pair_mode": cf_pair_mode,
            "num_recycles": num_recycles,
            "num_seeds": num_seeds,
            "num_relax": num_relax,
            "is_complex": is_complex,
            "af2m_scores": af2m_scores,
        },
    }


@app.local_entrypoint()
def main(
    input_fasta: str,
    models: str | None = None,
    num_recycles: int = 1,
    num_relax: int = 0,
    out_dir: str = "./out/alphafold",
    use_templates: bool = False,
    use_precomputed_msas: bool = False,
    return_all_files: bool = False,
    run_name: str | None = None,
) -> None:
    """Local entrypoint for running AlphaFold2 predictions.

    This function prepares the inputs, calls the remote `alphafold` Modal function,
    and saves the output files locally.

    Args:
        input_fasta (str): Path to the input FASTA file.
        models (list[int], optional): List of AlphaFold2 model numbers to run (1-5).
                                      Can be a comma-separated string if passed via CLI.
                                      Defaults to [1].
        num_recycles (int, optional): Number of recycles for the model. Defaults to 1.
        num_relax (int, optional): Number of Amber relaxation steps (0 for none, 1 for top model).
                                   Defaults to 0.
        out_dir (str, optional): Directory to save the output files. Defaults to ".".
        use_templates (bool, optional): Whether to use PDB templates. Defaults to False.
        use_precomputed_msas (bool, optional): Whether to use precomputed MSAs. Defaults to False.
        return_all_files (bool, optional): Whether to return all generated files from the remote
                                           function or just the primary zip. Defaults to False.

    Returns:
        None
    """
    from datetime import datetime

    fasta_str = open(input_fasta).read()
    models_int: list[int] = [int(m) for m in models.split(",")] if models else [1]

    result = alphafold.remote(
        fasta_name=Path(input_fasta).name,
        fasta_str=fasta_str,
        models=models_int,
        num_recycles=num_recycles,
        num_relax=num_relax,
        use_templates=use_templates,
        use_precomputed_msas=use_precomputed_msas,
        return_all_files=return_all_files,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    print(f"✓ AlphaFold completed: {result['message']}")
    print(f"✓ Generated {len(result['output_files'])} files")

    for file_info in result["output_files"]:
        # Read content from the file path returned by Modal
        filename = file_info["filename"]

        dest_path = Path(out_dir_full) / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Saving {filename} ({file_info['size']} bytes)")
        # Note: In practice, files would be in GCS and we'd download from there
        # For local testing, we'd need to fetch from Modal's output
