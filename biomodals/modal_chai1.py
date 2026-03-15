# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs Chai-1, a protein-ligand co-folding model, on Modal.

Chai-1r: https://github.com/chaidiscovery/chai-lab

Example fasta
```
>protein|name=insulin
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYSSTS
>ligand|name=caffeine
CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```
```
modal run modal_chai1.py --input-faa test_chai1.faa
```
"""

import os
from pathlib import Path

import modal
from modal import App, Image

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 30))


def download_models() -> None:
    """Downloads Chai-1 models by running a minimal inference.

    Args:
        None

    Returns:
        None
    """
    from tempfile import TemporaryDirectory

    import torch
    from chai_lab.chai1 import run_inference

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out:
        with open(f"{td_in}/tmp.faa", "w") as out:
            out.write(">protein|name=pro\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSL\n>ligand|name=lig\nCC\n")

        _ = run_inference(
            fasta_file=Path(f"{td_in}/tmp.faa"),
            output_dir=Path(f"{td_out}"),
            num_trunk_recycles=1,
            num_diffn_timesteps=10,
            seed=1,
            device=torch.device("cuda:0"),
            use_esm_embeddings=True,
        )


image = (
    Image.debian_slim()
    .apt_install("wget")
    .uv_pip_install("chai_lab==0.6.1")
    .uv_pip_install("google-cloud-storage==2.14.0")  # For GCS upload
    .run_function(download_models, gpu="a100")
)

app = App("chai1", image=image)


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def chai1(
    input_faa_str: str | None = None,
    input_faa_name: str | None = None,
    fasta_gcs_uri: str | None = None,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    seed: int = 42,
    use_esm_embeddings: bool = True,
    chai1_kwargs: dict | None = None,
    upload_to_gcs: bool = False,
    gcs_bucket: str = "dev-services",
    run_id: str | None = None,
) -> dict:
    """Runs Chai-1 on a FASTA file and returns the output files.

    Args:
        input_faa_str (str): Content of the input FASTA file as a string (DEPRECATED - use fasta_gcs_uri).
        input_faa_name (str): Original name of the input FASTA file (DEPRECATED - only needed with input_faa_str).
        fasta_gcs_uri (str): GCS URI to FASTA file (e.g., gs://bucket/path/input.faa). Preferred method.
        num_trunk_recycles (int): Number of trunk recycles for the model. Defaults to 3.
        num_diffn_timesteps (int): Number of diffusion timesteps. Defaults to 200.
        seed (int): Random seed for reproducibility. Defaults to 42.
        use_esm_embeddings (bool): Whether to use ESM embeddings. Defaults to True.
        chai1_kwargs (dict): Additional keyword arguments to pass to `run_inference`.
        upload_to_gcs (bool): Whether to upload results to GCS. Defaults to False.
        gcs_bucket (str): GCS bucket name. Defaults to "dev-services".
        run_id (str | None): Unique run identifier. If None, a timestamp will be generated.

    Returns:
        dict: Structured output with keys:
            - exit_code (int): 0 for success, 1 for failure
            - output_files (list[dict]): List of output file metadata
            - message (str): Status message
            - gcs_prefix (str): GCS prefix if upload_to_gcs=True
    """
    import tempfile
    from tempfile import TemporaryDirectory

    import torch
    from chai_lab.chai1 import run_inference

    if chai1_kwargs is None:
        chai1_kwargs = {}

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out:
        # Handle GCS URI input (preferred method)
        if fasta_gcs_uri:
            print(f"Downloading FASTA from GCS: {fasta_gcs_uri}")

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
            if not fasta_gcs_uri.startswith("gs://"):
                return {
                    "exit_code": 1,
                    "output_files": [],
                    "message": f"Error: Invalid GCS URI: {fasta_gcs_uri}",
                    "metadata": {},
                }

            gcs_path = fasta_gcs_uri.replace("gs://", "")
            bucket_name, blob_path = gcs_path.split("/", 1)

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            # Download FASTA file
            fasta_filename = Path(blob_path).name
            fasta_path = Path(td_in) / fasta_filename
            blob.download_to_filename(str(fasta_path))
            print(f"  ✓ Downloaded {fasta_filename}")

        # Fallback: Handle string input (legacy, deprecated)
        elif input_faa_str and input_faa_name:
            print("Using legacy input_faa_str input (deprecated - use fasta_gcs_uri)")
            fasta_path = Path(td_in) / input_faa_name
            fasta_path.write_text(input_faa_str)
        else:
            return {
                "exit_code": 1,
                "output_files": [],
                "message": "Error: Either fasta_gcs_uri OR (input_faa_str + input_faa_name) required",
                "metadata": {},
            }

        _ = run_inference(
            fasta_file=Path(fasta_path),
            output_dir=Path(td_out),
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffn_timesteps,
            seed=seed,
            device=torch.device("cuda:0"),
            use_esm_embeddings=use_esm_embeddings,
            **chai1_kwargs,
        )

        # Collect output files in structured format
        output_files = []
        for out_file in Path(td_out).glob("**/*"):
            if out_file.is_file():
                file_size = out_file.stat().st_size
                suffix = out_file.suffix.lstrip(".")

                # Determine artifact type
                artifact_type = (
                    "cif"
                    if suffix == "cif"
                    else "pdb"
                    if suffix == "pdb"
                    else "json"
                    if suffix == "json"
                    else "npz"
                    if suffix == "npz"
                    else "file"
                )

                output_files.append(
                    {
                        "path": str(out_file.absolute()),
                        "artifact_type": artifact_type,
                        "filename": out_file.name,
                        "size": file_size,
                        "metadata": {
                            "num_trunk_recycles": num_trunk_recycles,
                            "num_diffn_timesteps": num_diffn_timesteps,
                            "seed": seed,
                            "use_esm_embeddings": use_esm_embeddings,
                        },
                    }
                )

        # Upload to GCS if requested
        gcs_files = []
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
                blob_path = f"runs/{run_id}/chai1/{filename}"
                blob = bucket.blob(blob_path)

                # Upload file
                blob.upload_from_filename(local_path)

                # Add GCS info
                gcs_files.append(
                    {
                        "gcs_url": f"gs://{gcs_bucket}/{blob_path}",
                        "filename": filename,
                        "artifact_type": file_info["artifact_type"],
                        "size": file_info["size"],
                        "metadata": file_info["metadata"],
                    }
                )

            return {
                "exit_code": 0,
                "output_files": gcs_files,
                "gcs_prefix": f"gs://{gcs_bucket}/runs/{run_id}/chai1/",
                "message": f"Chai-1 completed. Uploaded {len(gcs_files)} files to GCS.",
                "stdout": f"Chai-1 completed successfully. Files uploaded to gs://{gcs_bucket}/runs/{run_id}/chai1/",
                "metadata": {
                    "num_trunk_recycles": num_trunk_recycles,
                    "num_diffn_timesteps": num_diffn_timesteps,
                    "seed": seed,
                    "use_esm_embeddings": use_esm_embeddings,
                    "run_id": run_id,
                    "gcs_bucket": gcs_bucket,
                },
            }

        return {
            "exit_code": 0,
            "output_files": output_files,
            "message": f"Chai-1 completed. Generated {len(output_files)} output files.",
            "stdout": "Chai-1 prediction completed successfully.",
            "metadata": {
                "num_trunk_recycles": num_trunk_recycles,
                "num_diffn_timesteps": num_diffn_timesteps,
                "seed": seed,
                "use_esm_embeddings": use_esm_embeddings,
            },
        }


@app.local_entrypoint()
def main(
    input_faa: str,
    out_dir: str = "./out/chai1",
    run_name: str | None = None,
    chai1_kwargs: str | None = None,
) -> None:
    """Local entrypoint for running Chai-1 predictions using Modal.

    Args:
        input_faa (str): Path to the input FASTA file.
        out_dir (str): Directory to save the output files. Defaults to "./out/chai1".
        run_name (str | None): Optional name for the run, used in the output directory structure.
        chai1_kwargs (str | None): Optional string representation of a dictionary for additional
                                   Chai-1 keyword arguments (e.g., '{"key": "value"}').

    Returns:
        None
    """
    from datetime import datetime

    input_faa_str = open(input_faa).read()

    result = chai1.remote(
        input_faa_str,
        input_faa_name=Path(input_faa).name,
        chai1_kwargs=dict(eval(chai1_kwargs)) if chai1_kwargs else {},
    )

    if result["exit_code"] != 0:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    print("Chai-1 completed successfully!")
    print(f"Generated {len(result['output_files'])} output files")

    # Save files locally
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    for file_info in result["output_files"]:
        local_path = file_info["path"]
        filename = file_info["filename"]

        # Copy file to output directory
        import shutil

        dest_path = out_dir_full / filename
        shutil.copy2(local_path, dest_path)
        print(f"  Saved: {dest_path}")

    print(f"\nOutput directory: {out_dir_full}")
