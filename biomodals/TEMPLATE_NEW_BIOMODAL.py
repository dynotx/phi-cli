# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""
Template for creating new biomodals following established patterns.

This template includes:
- Modal deployment setup
- GCS upload integration
- DB-ready structured output
- CLI entrypoint

Usage:
1. Copy this file to modal_YOURMODEL.py
2. Replace TEMPLATE placeholders with your model name
3. Customize the image with your dependencies
4. Implement the core computation logic
5. Deploy: uv run modal deploy biomodals/modal_YOURMODEL.py
6. Test: uv run modal run biomodals/modal_YOURMODEL.py --input-file test.txt

Example biomodals following this pattern:
- modal_esm2_predict_masked.py
- modal_boltz.py
- modal_chai1.py
- modal_boltzgen.py
"""

import os
from pathlib import Path
from typing import Any

import modal
from modal import App, Image

# ============================================================================
# CONFIGURATION
# ============================================================================

GPU = os.environ.get("GPU", None)  # None, "T4", "L40S", "A100", "H100"
TIMEOUT = int(os.environ.get("TIMEOUT", 60))  # Minutes

# Optional: Use Modal Volume for caching large models
# MODEL_VOLUME_NAME = "template-models"
# MODEL_VOLUME = Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
# CACHE_DIR = f"/{MODEL_VOLUME_NAME}"


# ============================================================================
# OPTIONAL: MODEL DOWNLOAD FUNCTION
# ============================================================================
# Run during image build to cache models/weights
# Uncomment if your model needs pre-downloading

# def download_models():
#     """Download model weights during image build to avoid runtime delays."""
#     import yourmodel
#
#     print("Downloading models...")
#     yourmodel.load_pretrained("model-name")
#     print("Model download complete")


# ============================================================================
# IMAGE DEFINITION
# ============================================================================

image = (
    Image.debian_slim(python_version="3.11")
    # OR: Image.micromamba(python_version="3.11") for conda packages
    # System dependencies
    .apt_install(["git", "wget"])
    # Python dependencies
    .pip_install(
        [
            "numpy",
            "pandas",
            "pydantic",
            # Add your model-specific dependencies here
        ]
    )
    # GCS upload dependency (required for all biomodals)
    .pip_install("google-cloud-storage==2.14.0")
    # Optional: Install from git repo
    # .run_commands(
    #     "git clone https://github.com/org/model /root/model",
    #     "cd /root/model && pip install -e .",
    # )
    # Optional: Download models during build
    # .run_function(
    #     download_models,
    #     gpu="a10g",
    #     volumes={CACHE_DIR: MODEL_VOLUME},
    # )
)

# ============================================================================
# APP DEFINITION
# ============================================================================

app = App(
    "template-biomodal",  # Change to your model name
    image=image,
    secrets=[modal.Secret.from_name("cloudsql-credentials")],  # For GCS access
)


# ============================================================================
# CORE COMPUTATION FUNCTION
# ============================================================================


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    # Optional: Add volume for model caching
    # volumes={CACHE_DIR: MODEL_VOLUME},
)
def run_template(
    # Core inputs - customize for your model
    input_data: str,
    parameters: dict[str, Any] | None = None,
    # GCS upload parameters (standard for all biomodals)
    upload_to_gcs: bool = False,
    gcs_bucket: str | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Run the biomodal computation.

    Args:
        input_data: Main input (customize type: str, bytes, dict, etc.)
        parameters: Model-specific parameters
        upload_to_gcs: Whether to upload results to GCS
        gcs_bucket: GCS bucket name (e.g., "dev-services")
        run_id: Unique run identifier for organizing outputs

    Returns:
        Dictionary with structured output:
        {
            "exit_code": 0,
            "output_files": [
                {
                    "gcs_url": "gs://bucket/path/file.ext",
                    "filename": "file.ext",
                    "artifact_type": "ext",
                    "size": 12345,
                    "metadata": {...}
                }
            ],
            "gcs_prefix": "gs://bucket/runs/run_id/tool/",
            "message": "Computation completed",
            "metadata": {...}
        }
    """
    import tempfile
    from tempfile import TemporaryDirectory

    # Set default parameters
    if parameters is None:
        parameters = {}

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # ====================================================================
        # IMPLEMENT YOUR COMPUTATION LOGIC HERE
        # ====================================================================

        # Example: Load your model
        # import yourmodel
        # model = yourmodel.load_pretrained("model-name")

        # Example: Process input
        # result = model.predict(input_data, **parameters)

        # Example: Save outputs
        # output_file = output_dir / "results.txt"
        # output_file.write_text(str(result))

        # For demonstration, create a dummy output file
        output_file = output_dir / "results.txt"
        output_file.write_text(f"Processed: {input_data[:100]}\n")

        # ====================================================================
        # COLLECT OUTPUT FILES (keep this structure)
        # ====================================================================

        output_files_list = []
        for out_file in output_dir.glob("**/*"):
            if out_file.is_file():
                # Determine artifact type from extension
                suffix = out_file.suffix.lstrip(".")
                artifact_type = suffix if suffix else "file"

                output_files_list.append(
                    {
                        "path": str(out_file.absolute()),
                        "filename": out_file.name,
                        "artifact_type": artifact_type,
                        "size": out_file.stat().st_size,
                        "metadata": {
                            # Add model-specific metadata
                            "model_version": "1.0.0",
                            "parameters": parameters,
                        },
                    }
                )

        # ====================================================================
        # GCS UPLOAD (standard pattern - keep as-is)
        # ====================================================================

        if upload_to_gcs and gcs_bucket and run_id:
            from google.cloud import storage

            # Initialize GCS client with credentials from secret
            credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if credentials_json:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    f.write(credentials_json)
                    credentials_file = f.name
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

            client = storage.Client()
            bucket = client.bucket(gcs_bucket)

            # Define GCS path structure
            tool_name = "template"  # Change to your tool name
            gcs_prefix = f"runs/{run_id}/{tool_name}"

            output_files = []

            # Upload each file
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
                "message": f"Template completed: {len(output_files)} files uploaded to GCS",
                "metadata": {
                    "run_id": run_id,
                    "tool_name": tool_name,
                    # Add any summary statistics
                },
            }

        else:
            # Return without GCS upload (for local testing)
            return {
                "exit_code": 0,
                "output_files": output_files_list,
                "message": f"Template completed: {len(output_files_list)} files generated",
                "metadata": {},
            }


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================


@app.local_entrypoint()
def main(
    input_file: str,
    param_key: str | None = None,  # Add your specific parameters
    out_dir: str = "./out/template",
    run_name: str | None = None,
) -> None:
    """
    Local CLI entrypoint for testing.

    Usage:
        modal run modal_template.py --input-file test.txt
        modal run modal_template.py --input-file test.txt --param-key value
    """
    from datetime import datetime

    # Read input
    input_path = Path(input_file)
    input_data = input_path.read_text()

    # Prepare parameters
    parameters = {}
    if param_key:
        parameters["param_key"] = param_key

    # Run computation
    print("Running template biomodal...")
    print(f"Input: {input_file}")

    result = run_template.remote(
        input_data=input_data,
        parameters=parameters,
    )

    if result["exit_code"] != 0:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    # Save outputs locally
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    for file_info in result["output_files"]:
        local_path = Path(str(file_info["path"]))
        dest_path = out_dir_full / file_info["filename"]

        import shutil

        shutil.copy2(local_path, dest_path)
        print(f"  Saved: {dest_path}")

    print(f"\nOutput directory: {out_dir_full}")
    print(f"Generated {len(result['output_files'])} files")


# ============================================================================
# NOTES FOR IMPLEMENTATION
# ============================================================================

"""
CHECKLIST FOR NEW BIOMODAL:

1. [ ] Rename file to modal_YOURMODEL.py
2. [ ] Update app name in App()
3. [ ] Add model-specific dependencies to image
4. [ ] Implement core computation logic in run_template()
5. [ ] Customize input parameters (type, validation)
6. [ ] Update artifact_type detection for your output files
7. [ ] Add model-specific metadata
8. [ ] Update CLI parameters in main()
9. [ ] Test locally: modal run biomodals/modal_YOURMODEL.py --input-file test.txt
10. [ ] Deploy: modal deploy biomodals/modal_YOURMODEL.py
11. [ ] Test with GCS: Add --upload-to-gcs --gcs-bucket dev-services --run-id test_001
12. [ ] Create database integration test (see test_chai1_with_db.py)

COMMON PATTERNS:

GPU Selection:
- None: CPU only (ESM2, simple models)
- "T4": Light GPU work (LigandMPNN)
- "L40S": Medium GPU (Boltz, BindCraft)
- "A100": Heavy GPU (AlphaFold, Chai-1)

Input Types:
- String: FASTA sequences, YAML configs
- Bytes: PDB files, binary data
- Dict: Complex structured inputs

Output Types:
- PDB/CIF: Structure files
- FASTA/FA: Sequence files
- TSV/CSV: Tabular results
- JSON: Metadata/scores
- PNG: Visualizations
- TXT/LOG: Text outputs

Timeouts:
- 15-30 min: Fast models (ESM2, LigandMPNN)
- 60-120 min: Medium models (Boltz, AlphaFold)
- 300+ min: Complex pipelines (BindCraft)

EXAMPLE DEPLOYMENTS:

See these for reference:
- biomodals/modal_esm2_predict_masked.py (Simple, CPU, fast)
- biomodals/modal_boltz.py (Medium, GPU, with volumes)
- biomodals/modal_chai1.py (Complex, A100, long timeout)
- biomodals/modal_boltzgen.py (Very complex, multi-step)
"""
