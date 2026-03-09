"""
RSO (Rapid Structure Optimization) - Binder design using ColabDesign/AlphaFold2

Adapted from https://github.com/coreyhowe999/RSO

This tool performs rapid binder design through iterative hallucination and optimization
using ColabDesign and AlphaFold2.

Example:
```
modal run modal_rso.py --input-pdb ABC1.pdb --run-name ABC1 --binder-len 60
```
"""

import os
from datetime import datetime  # Add this import
from pathlib import Path

import modal

app = modal.App("rso")

GPU = os.environ.get("MODAL_GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 180))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("wget", "git")
    .pip_install(
        "numpy",
        "pandas",
        "biopython",
    )
    .pip_install(
        "git+https://github.com/sokrypton/ColabDesign.git",
    )
    # Patch ColabDesign for modern JAX compatibility (jax.tree_map → jax.tree_util.tree_map)
    .run_commands(
        "find /usr/local/lib/python3.10/site-packages/colabdesign -name '*.py' -exec sed -i 's/jax\\.tree_map/jax.tree_util.tree_map/g' {} +"
    )
    # Install JAX with CUDA support (like AlphaFold does)
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]==0.4.38" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a10g",
    )
    .pip_install("google-cloud-storage==2.14.0")
    .run_commands(
        [
            "mkdir -p /root/params",
            "wget -P /root/params/ https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
            "tar -xvf /root/params/alphafold_params_2022-12-06.tar -C /root/params/",
            "rm /root/params/alphafold_params_2022-12-06.tar",
        ]
    )
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT * 60,
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def rso(
    pdb_name=None,
    pdb_str=None,
    pdb_gcs_uri=None,
    traj_iters=None,
    binder_len=None,
    chain=None,
    hotspot=None,
    thresholds=None,
    upload_to_gcs=False,
    gcs_bucket=None,
    run_id=None,
):
    """RSO (Rapid Structure Optimization) for binder design.

    Args:
        pdb_name: Name of the input PDB file (only needed with pdb_str)
        pdb_str: Content of the PDB file as a string (DEPRECATED - use pdb_gcs_uri)
        pdb_gcs_uri: GCS URI to target PDB file (e.g., gs://bucket/path/target.pdb). Preferred method.
        traj_iters: Number of trajectory iterations
        binder_len: Length of the binder to design
        chain: Target chain for binder design
        hotspot: Optional hotspot residues
        thresholds: Optional thresholds for filtering (rmsd, plddt, pae)
        upload_to_gcs: Whether to upload results to GCS
        gcs_bucket: GCS bucket name (e.g., "dev-services")
        run_id: Unique run identifier for organizing outputs

    Returns:
        dict: Structured output with exit_code, output_files, message, and metadata
    """
    # Import colabdesign modules here
    import tempfile

    import jax
    import jax.numpy as jnp
    import pandas as pd
    from colabdesign import clear_mem, mk_afdesign_model
    from colabdesign.af.alphafold.common import residue_constants
    from colabdesign.mpnn import mk_mpnn_model

    # Handle GCS URI input (preferred method)
    if pdb_gcs_uri:
        print(f"Downloading target PDB from GCS: {pdb_gcs_uri}")

        from google.cloud import storage

        # Initialize GCS client
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not credentials_json:
            return {
                "exit_code": 1,
                "output_files": [],
                "message": "Error: GOOGLE_APPLICATION_CREDENTIALS_JSON not found",
                "metadata": {},
            }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(credentials_json)
            credentials_file = f.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

        # Parse GCS URI
        if not pdb_gcs_uri.startswith("gs://"):
            return {
                "exit_code": 1,
                "output_files": [],
                "message": f"Error: Invalid GCS URI: {pdb_gcs_uri}",
                "metadata": {},
            }

        gcs_path = pdb_gcs_uri.replace("gs://", "")
        bucket_name, blob_path = gcs_path.split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download PDB content
        pdb_str = blob.download_as_text()
        pdb_name = Path(blob_path).name
        print(f"  ✓ Downloaded {pdb_name} ({len(pdb_str)} bytes)")

    if not pdb_str:
        return {
            "exit_code": 1,
            "output_files": [],
            "message": "Error: Either pdb_gcs_uri or pdb_str is required",
            "metadata": {},
        }

    pdb_path = Path("/tmp/in_rso") / pdb_name
    pdb_path.parent.mkdir(parents=True, exist_ok=True)

    if thresholds is None:
        # e.g. proper thresholds vs extremely permissive
        # thresholds = {"rmsd": 2, "plddt": 0.15, "pae": 0.4}
        thresholds = {"rmsd": 10, "plddt": 1, "pae": 1}

    pdb_path.write_text(pdb_str)

    def add_rg_loss(self, weight=0.1):
        """add radius of gyration loss"""

        def loss_fn(inputs, outputs):
            xyz = outputs["structure_module"]
            ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]

            ca = ca[-self._binder_len :]

            rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
            rg_th = 2.38 * ca.shape[0] ** 0.365
            rg = jax.nn.elu(rg - rg_th)
            return {"rg": rg}

        self._callbacks["model"]["loss"].append(loss_fn)
        self.opt["weights"]["rg"] = weight

    # Remove all PDB files with 'binder_design' in the file name
    for pdb_file in Path().glob("**/*binder_design*.pdb"):
        pdb_file.unlink()

    #
    # AFDesign steps
    #
    clear_mem()
    af_model = mk_afdesign_model(protocol="binder", data_dir="/root/params")
    add_rg_loss(af_model)
    af_model.prep_inputs(
        pdb_filename=str(pdb_path), chain=chain, hotspot=hotspot, binder_len=binder_len
    )

    #
    # Adjust as needed
    #
    af_model.restart(mode=["gumbel", "soft"])
    af_model.set_weights(helix=-0.2, plddt=0.1, pae=0.1, rg=0.5, i_pae=5.0, i_con=2.0)
    af_model.design_logits(traj_iters)
    af_model.save_pdb("backbone.pdb")

    ### SEQ DESIGN AND FILTER ####

    binder_model = mk_afdesign_model(
        protocol="binder", use_multimer=True, use_initial_guess=True, data_dir="/root/params"
    )
    monomer_model = mk_afdesign_model(protocol="fixbb", data_dir="/root/params")

    # binder_model.set_weights(i_pae=1.0)

    mpnn_model = mk_mpnn_model(weights="soluble")
    mpnn_model.prep_inputs(pdb_filename="backbone.pdb", chain="A,B", fix_pos="A", rm_aa="C")

    samples = mpnn_model.sample_parallel(8, temperature=0.01)
    monomer_model.prep_inputs(pdb_filename="backbone.pdb", chain="B")
    binder_model.prep_inputs(
        pdb_filename="backbone.pdb",
        chain="A",
        binder_chain="B",
        use_binder_template=True,
        rm_template_ic=True,
    )

    results_df = pd.DataFrame()

    # output results

    for j, seq in enumerate(samples["seq"]):
        print("Predicting binder only")
        monomer_model.predict(seq=seq[-binder_len:], num_recycles=3)
        if monomer_model.aux["losses"]["rmsd"] < thresholds["rmsd"]:
            print("Passed! Predicting binder with receptor using AF Multimer")
            binder_model.predict(seq=seq[-binder_len:], num_recycles=3)
            if (
                monomer_model.aux["losses"]["plddt"] < thresholds["plddt"]
                and monomer_model.aux["losses"]["pae"] < thresholds["pae"]
            ):
                binder_model.save_pdb(f"{Path(pdb_name).stem}_binder_design_{j}.pdb")
                results_df.loc[j, "pdb_id"] = f"{Path(pdb_name).stem}_binder_design_{j}.pdb"
                results_df.loc[j, "seq"] = seq[-binder_len:]
                for key in binder_model.aux["log"]:
                    results_df.loc[j, key] = binder_model.aux["log"][key]
                for weight in af_model.opt["weights"]:
                    results_df.loc[j, f"weights_{key}"] = weight
        else:
            print(f"Failed! RMSD: {monomer_model.aux['losses']['rmsd']} >= 2.0")

    results_df.to_csv("binder_design_scores.csv", index=False)

    # Collect output files
    output_files_list = []
    for out_file in Path().glob("**/*"):
        if out_file.is_file() and out_file.suffix != ".npz":
            suffix = out_file.suffix.lstrip(".")
            artifact_type = suffix if suffix else "file"

            output_files_list.append(
                {
                    "path": str(out_file.absolute()),
                    "filename": out_file.name,
                    "artifact_type": artifact_type,
                    "size": out_file.stat().st_size,
                    "metadata": {
                        "traj_iters": traj_iters,
                        "binder_len": binder_len,
                        "chain": chain,
                        "hotspot": hotspot,
                    },
                }
            )

    # GCS upload
    if upload_to_gcs and gcs_bucket and run_id:
        from google.cloud import storage

        # Initialize GCS client
        credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if credentials_json:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(credentials_json)
                credentials_file = f.name
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

        client = storage.Client()
        bucket = client.bucket(gcs_bucket)

        tool_name = "rso"
        gcs_prefix = f"runs/{run_id}/{tool_name}"

        output_files = []

        for file_info in output_files_list:
            local_path = Path(str(file_info["path"]))
            blob_path = f"{gcs_prefix}/{file_info['filename']}"

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(local_path))

            gcs_url = f"gs://{gcs_bucket}/{blob_path}"
            output_files.append(
                {
                    "gcs_url": gcs_url,
                    "filename": file_info["filename"],
                    "artifact_type": file_info["artifact_type"],
                    "size": file_info["size"],
                    "metadata": file_info["metadata"],
                }
            )

        return {
            "exit_code": 0,
            "output_files": output_files,
            "gcs_prefix": f"gs://{gcs_bucket}/{gcs_prefix}/",
            "message": f"RSO completed: {len(output_files)} files uploaded to GCS",
            "metadata": {
                "run_id": run_id,
                "tool_name": tool_name,
                "traj_iters": traj_iters,
                "binder_len": binder_len,
                "chain": chain,
                "num_designs": len(
                    [
                        f
                        for f in output_files
                        if "binder_design" in f["filename"] and f["artifact_type"] == "pdb"  # type: ignore[operator]
                    ]
                ),
            },
        }

    else:
        # Return without GCS upload (for local testing)
        return {
            "exit_code": 0,
            "output_files": output_files_list,
            "message": f"RSO completed: {len(output_files_list)} files generated",
            "metadata": {
                "traj_iters": traj_iters,
                "binder_len": binder_len,
                "chain": chain,
            },
        }


@app.local_entrypoint()
def main(
    input_pdb: str,
    num_designs: int = 1,
    traj_iters: int = 100,
    binder_len: int = 80,
    chain: str = "A",
    hotspot: str | None = None,
    thresholds: str | None = None,
    out_dir: str = "./out/rso",
    run_name: str | None = None,
) -> None:
    """Local entrypoint for RSO (Rapid Structure Optimization).

    Args:
        input_pdb: Path to input PDB file
        num_designs: Number of designs to generate (runs in parallel)
        traj_iters: Number of trajectory iterations for hallucination
        binder_len: Length of the binder to design
        chain: Target chain for binder design
        hotspot: Optional hotspot residues
        thresholds: Optional thresholds for filtering
        out_dir: Output directory for results
        run_name: Optional run name
    """
    pdb_str = open(input_pdb).read()
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Running RSO binder design...")
    print(f"  Input: {input_pdb}")
    print(f"  Binder length: {binder_len}")
    print(f"  Chain: {chain}")
    print(f"  Trajectory iterations: {traj_iters}")
    print(f"  Num designs: {num_designs}")

    # Run designs in parallel
    all_results = rso.starmap(
        [
            (
                Path(input_pdb).name,
                pdb_str,
                traj_iters,
                binder_len,
                chain,
                hotspot,
                thresholds,
                False,
                None,
                None,
            )
            for _ in range(num_designs)
        ]
    )

    # Process results
    total_files = 0
    total_binders = 0

    for design_num, result in enumerate(all_results):
        if result["exit_code"] != 0:
            print(f"Design {design_num} failed: {result.get('error', 'Unknown error')}")
            continue

        # Save outputs locally
        design_dir = Path(out_dir) / (run_name or today) / f"design_{design_num}"
        design_dir.mkdir(parents=True, exist_ok=True)

        for file_info in result["output_files"]:
            local_path = Path(str(file_info["path"]))
            dest_path = design_dir / file_info["filename"]

            if local_path.exists():
                import shutil

                shutil.copy2(local_path, dest_path)
                total_files += 1

                if "binder_design" in file_info["filename"] and file_info["artifact_type"] == "pdb":
                    total_binders += 1

    print("\nCompleted!")
    print(f"  Output directory: {Path(out_dir) / (run_name or today)}")
    print(f"  Total designs run: {num_designs}")
    print(f"  Total files generated: {total_files}")
    print(f"  Total binder PDBs: {total_binders}")
