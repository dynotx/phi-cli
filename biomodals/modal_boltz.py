# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs Boltz-1x for protein structure and complex prediction on Modal.

Boltz-1x: https://github.com/jwohlwend/boltz

## Example input, test_boltz.yaml:
```
sequences:
    - protein:
        id: A
        sequence: TDKLIFGKGTRVTVEP
```

## Example usage

Note yaml is highly recommended over fasta input

```
modal run modal_boltz.py --input-yaml test_boltz.yaml
```

Explicitly state the default params:
```
modal run modal_boltz.py --input-yaml test_boltz.yaml --params-str "--use_msa_server --seed 42"
```

Extra params (recommended for best performance):
```
modal run modal_boltz.py --input-yaml test_boltz.yaml --params-str "--use_msa_server --seed 42 --recycling_steps 10 --step_scale 1.0 --diffusion_samples 10"
```
"""

import os
from pathlib import Path
from typing import Any

import modal
from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))

BOLTZ_VOLUME_NAME = "boltz-models"
BOLTZ_MODEL_VOLUME = Volume.from_name(BOLTZ_VOLUME_NAME, create_if_missing=True)
CACHE_DIR = f"/{BOLTZ_VOLUME_NAME}"

ENTITY_TYPES = {"protein", "dna", "rna", "ccd", "smiles"}
ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_PARAMS = "--use_msa_server --seed 42"


def download_model() -> None:
    """Forces download of the Boltz-1 model by running it once.

    Args:
        None

    Returns:
        None
    """
    from boltz.main import download_boltz1, download_boltz2

    if not Path(f"{CACHE_DIR}/boltz1_conf.ckpt").exists():
        print("downloading boltz 1")
        download_boltz1(Path(CACHE_DIR))

    if not Path(f"{CACHE_DIR}/boltz2_conf.ckpt").exists():
        print("downloading boltz 2")
        download_boltz2(Path(CACHE_DIR))


image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git", "gcc", "g++")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold@acc0bf772f22feb7f887ad132b7313ff415c8a9f"
    )
    .micromamba_install("kalign2=2.04", "hhsuite=3.3.0", channels=["conda-forge", "bioconda"])
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a100",
    )
    .run_commands("python -m colabfold.download")
    .apt_install("build-essential")
    # Install CUDA toolkit via conda
    .pip_install(
        "boltz==2.2.1",
        "pyyaml",
        "pandas",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
        "google-cloud-storage==2.14.0",  # For GCS upload
    )
    .run_function(
        download_model,
        gpu="a10g",
        volumes={f"/{BOLTZ_VOLUME_NAME}": BOLTZ_MODEL_VOLUME},
    )
)

app = App(
    "boltz",
    image=image,
)


def fasta_iter(fasta_name):
    """Yields stripped sequence IDs and sequences from a FASTA file.

    Args:
        fasta_name (str): Path to the FASTA file.

    Yields:
        tuple[str, str]: Tuples of (sequence_id, sequence).
    """
    from itertools import groupby

    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line.startswith(">")))
        for header_line in faiter:
            header = next(header_line)[1:].strip()
            seq = "".join(s.strip() for s in next(faiter))
            yield header, seq


def _fasta_to_yaml(input_faa: str) -> str:
    """Converts Boltz FASTA to Boltz YAML format.

    Note: Only basic protein, rna, dna supported for now. Use YAML directly for more complex inputs.

    Args:
        input_faa (str): Path to the input FASTA file.

    Returns:
        str: A string containing the YAML representation of the FASTA content.

    Raises:
        NotImplementedError: If more than 26 chains are present or unsupported entity types/MSA are encountered.
    """
    import re

    import yaml

    chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    yaml_dict: dict[str, list] = {"sequences": []}

    rx = re.compile(r"^([A-Z])\|([^\|]+)\|?(.*)$")

    for n, (seq_id, seq) in enumerate(fasta_iter(input_faa)):
        if n >= len(chains):
            raise NotImplementedError(">26 chains not supported")

        s_info = rx.search(seq_id)
        if s_info is not None:
            entity_type = s_info.groups()[1].lower()
            if entity_type not in ["protein", "dna", "rna"]:
                raise NotImplementedError(f"Entity type {entity_type} not supported")
            chain_id = s_info.groups()[0].upper()
            if len(s_info.groups()) > 2 and s_info.groups()[2] not in ["", "empty"]:
                raise NotImplementedError("MSA not supported")
        else:
            entity_type = "protein"
            chain_id = chains[n]

        if entity_type == "protein":
            print(entity_type)
            assert all(aa.upper() in ALLOWED_AAS for aa in seq), f"not AAs: {seq}"

        entity = {entity_type: {"id": chain_id, "sequence": seq}}

        yaml_dict["sequences"].append(entity)

    return yaml.dump(yaml_dict, sort_keys=False)


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={f"/{BOLTZ_VOLUME_NAME}": BOLTZ_MODEL_VOLUME},
    secrets=[modal.Secret.from_name("cloudsql-credentials")],  # For GCS access
)
def boltz(
    input_str: str,
    params_str: str | None = None,
    model: str = "boltz-2",
    upload_to_gcs: bool = False,
    gcs_bucket: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Runs Boltz on a YAML or FASTA input string.

    The input can describe proteins, DNA, RNA, SMILES strings, or CCD identifiers.

    Args:
        input_str (str): Input content as a string, can be in FASTA or Boltz YAML format.
        params_str (str | None): Optional string of additional parameters to pass to the
                                 `boltz predict` command. Defaults to `DEFAULT_PARAMS`.
        model (str): Boltz model version to use. Valid values: "boltz-1", "boltz-2".
                     Default: "boltz-2" (recommended for better accuracy).
        upload_to_gcs (bool): Upload results to GCS. Default: False.
        gcs_bucket (str | None): GCS bucket name for uploads.
        run_id (str | None): Unique identifier for this run.

    Returns:
        dict: Standardized result with exit_code, output_files, message, metadata.
              Always returns dict format (not list of tuples).
    """
    import sys
    import tempfile
    from subprocess import CalledProcessError, run

    # Add project root to path for imports
    sys.path.insert(0, "/root")

    try:
        from biomodals.base import BiomodalBase

        use_base_class = True
    except ImportError:
        # Fallback if BiomodalBase not available
        use_base_class = False
        print("Warning: BiomodalBase not available, using fallback formatting")

    # Validate model parameter
    if model not in ["boltz-1", "boltz-2"]:
        return {
            "exit_code": 1,
            "output_files": [],
            "message": f"Error: Invalid model '{model}'. Must be 'boltz-1' or 'boltz-2'",
            "error": f"Invalid model: {model}",
            "metadata": {},
        }

    if params_str is None:
        params_str = DEFAULT_PARAMS

    # Add boltz version flag if not already specified
    # Convert "boltz-1" -> "boltz1", "boltz-2" -> "boltz2" for CLI
    model_cli = model.replace("-", "")
    if "--model" not in params_str:
        params_str = f"{params_str} --model {model_cli}"

    run_id = run_id or "boltz_run"

    if use_base_class:
        base = BiomodalBase(run_id, "boltz")

    try:
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            # Prepare input
            if input_str[0] == ">":
                in_faa = Path(in_dir) / "in.faa"
                in_faa.write_text(input_str)
                fixed_faa_str = _fasta_to_yaml(str(in_faa))
                fixed_yaml = Path(in_dir) / "fixed.yaml"
                fixed_yaml.write_text(fixed_faa_str)
            else:
                fixed_yaml = Path(in_dir) / "fixed.yaml"
                fixed_yaml.write_text(input_str)

            # Run Boltz
            cmd = (
                f'boltz predict "{fixed_yaml}"'
                f' --out_dir "{out_dir}"'
                f' --cache "{CACHE_DIR}"'
                f" {params_str}"
            )

            result = run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                if use_base_class:
                    return base.handle_error(
                        Exception(error_msg), context="Boltz prediction failed"
                    ).model_dump()
                else:
                    return {
                        "exit_code": result.returncode,
                        "output_files": [],
                        "message": f"Error: {error_msg}",
                        "error": error_msg,
                        "metadata": {},
                    }

            # Collect output files
            output_files_paths = [
                out_file for out_file in Path(out_dir).glob("**/*") if out_file.is_file()
            ]

            # Handle GCS upload if requested
            if upload_to_gcs and gcs_bucket:
                if use_base_class:
                    output_files = base.upload_to_gcs(
                        files=output_files_paths,
                        gcs_bucket=gcs_bucket,
                    )
                    return base.create_success_result(
                        output_files=output_files,
                        message=f"Boltz completed: {len(output_files)} files uploaded",
                        stdout=result.stdout,
                        num_output_files=len(output_files),
                        model_version=model,
                    ).model_dump()
                else:
                    # Fallback GCS upload
                    from google.cloud import storage

                    # Initialize GCS client with credentials from secret
                    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                    if not credentials_json:
                        return {
                            "exit_code": 1,
                            "output_files": [],
                            "message": "Error: GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                            "error": "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                            "metadata": {},
                        }

                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                        f.write(credentials_json)
                        credentials_file = f.name
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

                    client = storage.Client()
                    bucket = client.bucket(gcs_bucket)
                    gcs_prefix = f"runs/{run_id}/boltz"
                    output_files = []

                    for file_path in output_files_paths:
                        filename = file_path.name
                        blob_path = f"{gcs_prefix}/{filename}"
                        blob = bucket.blob(blob_path)
                        blob.upload_from_filename(str(file_path))

                        suffix = file_path.suffix.lstrip(".")
                        artifact_type = suffix if suffix else "file"

                        file_dict: dict[str, Any] = {
                            "gcs_url": f"gs://{gcs_bucket}/{blob_path}",
                            "filename": filename,
                            "artifact_type": artifact_type,
                            "path": str(file_path),
                            "size_bytes": file_path.stat().st_size,
                            "metadata": {},
                        }
                        output_files.append(file_dict)

                    return {
                        "exit_code": 0,
                        "output_files": output_files,
                        "gcs_prefix": f"gs://{gcs_bucket}/{gcs_prefix}/",
                        "message": f"Boltz completed: {len(output_files)} files uploaded",
                        "stdout": result.stdout,
                        "metadata": {"model_version": model},
                        "num_output_files": len(output_files),
                    }
            else:
                # No GCS upload - return local files in dict format
                if use_base_class:
                    return base.create_success_result(
                        output_files=output_files_paths,
                        message=f"Boltz completed: {len(output_files_paths)} files generated",
                        stdout=result.stdout,
                        num_output_files=len(output_files_paths),
                        model_version=model,
                    ).model_dump()
                else:
                    output_files = []
                    for file_path in output_files_paths:
                        suffix = file_path.suffix.lstrip(".")
                        file_info: dict[str, Any] = {
                            "filename": file_path.name,
                            "artifact_type": suffix if suffix else "file",
                            "path": str(file_path),
                            "size_bytes": file_path.stat().st_size,
                            "metadata": {},
                        }
                        output_files.append(file_info)

                    return {
                        "exit_code": 0,
                        "output_files": output_files,
                        "message": f"Boltz completed: {len(output_files)} files generated",
                        "stdout": result.stdout,
                        "metadata": {"model_version": model},
                        "num_output_files": len(output_files),
                    }

    except CalledProcessError as e:
        error_msg = f"Boltz command failed: {e.stderr or e.stdout}"
        if use_base_class:
            return base.handle_error(e, context="Boltz execution").model_dump()
        else:
            return {
                "exit_code": e.returncode,
                "output_files": [],
                "message": error_msg,
                "error": error_msg,
                "metadata": {},
            }
    except Exception as e:
        if use_base_class:
            return base.handle_error(e, context="Boltz execution").model_dump()
        else:
            return {
                "exit_code": 1,
                "output_files": [],
                "message": f"Error: {str(e)}",
                "error": str(e),
                "metadata": {},
            }


@app.local_entrypoint()
def main(
    input_faa: str | None = None,
    input_yaml: str | None = None,
    params_str: str | None = None,
    model: str = "boltz-2",
    run_name: str | None = None,
    out_dir: str = "./out/boltz",
) -> None:
    """Local entrypoint to run Boltz predictions using Modal.

    Args:
        input_faa (str | None): Path to an input FASTA file.
        input_yaml (str | None): Path to an input Boltz YAML file.
        params_str (str | None): Optional string of additional parameters for the `boltz predict` command.
        model (str): Boltz model version. Valid: "boltz-1", "boltz-2". Default: "boltz-2".
        run_name (str | None): Optional name for the run, used for organizing output files.
                               If None, a timestamp-based name is used.
        out_dir (str): Directory to save the output files. Defaults to "./out/boltz".

    Returns:
        None

    Raises:
        AssertionError: If neither `input_faa` nor `input_yaml` is provided.
    """
    from datetime import datetime

    assert input_faa or input_yaml, "input_faa or input_yaml required"

    input_file = input_yaml or input_faa
    if not input_file:
        raise ValueError("Either input_yaml or input_faa must be provided")
    input_str = open(input_file).read()

    # Call boltz - now returns dict
    result = boltz.remote(input_str, params_str=params_str, model=model)

    if result["exit_code"] != 0:
        print(f"❌ Boltz failed: {result.get('error', 'Unknown error')}")
        return

    print(f"✅ Boltz completed: {result['message']}")
    print(f"   Files: {result['num_output_files']}")

    # Save output files locally
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    for file_info in result["output_files"]:
        filename = file_info["filename"]
        output_path = out_dir_full / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If file has local path, copy it
        if "path" in file_info and Path(file_info["path"]).exists():
            import shutil

            shutil.copy2(file_info["path"], output_path)
            print(f"   Saved: {output_path}")
        else:
            print(f"   (GCS only): {file_info.get('gcs_url', filename)}")
