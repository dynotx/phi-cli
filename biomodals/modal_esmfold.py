# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""ESMFold on Modal using pre-built Docker image.

Uses biochunan/esmfold-image which has ESMFold, OpenFold, and models pre-installed.
This avoids the complex dependency issues with building from scratch.

Example usage:
    # Single sequence
    modal run modal/biomodals/modal_esmfold.py --input-fasta protein.fasta

    # With GCS upload
    modal run modal/biomodals/modal_esmfold.py \
        --input-fasta protein.fasta \
        --upload-to-gcs \
        --gcs-bucket dev-services \
        --run-id test_001
"""

import json
import os
from pathlib import Path

import modal
from modal import App, Image

# ============================================================================
# CONFIGURATION
# ============================================================================

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 20))  # Minutes

# Modal Volume for caching ESMFold weights (~3-4GB)
# This prevents re-downloading weights on every run
ESMFOLD_CACHE_VOLUME = modal.Volume.from_name("esmfold-weights", create_if_missing=True)
CACHE_DIR = "/cache"


# ============================================================================
# IMAGE DEFINITION - Using Pre-built Docker Image
# ============================================================================

# Use pre-built image with ESMFold already installed
# Source: https://hub.docker.com/r/biochunan/esmfold-image
# This image has: Python 3.9, PyTorch, ESM, OpenFold, and ESMFold weights pre-installed
image = (
    Image.from_registry(
        "biochunan/esmfold-image:nonroot-devel",
        setup_dockerfile_commands=[
            # Override entrypoint - the default entrypoint expects specific commands
            "ENTRYPOINT []",
        ],
    )
    # Install GCS upload dependency
    # Install to both Modal's Python and the conda environment
    .pip_install("google-cloud-storage==2.14.0")  # For Modal's Python (GCS upload)
    .run_commands(
        # Also install in conda env (in case subprocess needs it)
        "/home/vscode/.conda/envs/py39-esmfold/bin/pip install google-cloud-storage==2.14.0"
    )
)


# ============================================================================
# APP DEFINITION
# ============================================================================

app = App("esmfold", image=image)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _parse_fasta(fasta_str: str) -> list[tuple[str, str]]:
    """Parse FASTA string into list of (label, sequence) tuples.

    Args:
        fasta_str: FASTA formatted string

    Returns:
        List of (label, sequence) tuples

    Raises:
        AssertionError: If FASTA format is invalid
    """
    fasta_str = fasta_str.strip()
    assert fasta_str.startswith(">"), "FASTA must start with '>'"

    records = []
    seen_names = set()

    for i, entry in enumerate(fasta_str[1:].split("\n>"), 1):
        label, _, seq = entry.partition("\n")
        seq = "".join(seq.split()).upper()
        if not seq:
            continue

        # Strategy: Take first 2-3 words to ensure uniqueness
        # e.g., "T=0.1, sample=1, ..." -> "T_0.1_sample_1"
        words = label.strip().split()[:3]  # Take up to 3 words
        name = "_".join(words)

        # Sanitize for filenames: remove problematic characters
        name = (
            name.replace(",", "")
            .replace("=", "_")
            .replace("/", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )

        # Fallback: If name collision detected, append index
        if name in seen_names:
            name = f"{name}_{i}"

        seen_names.add(name)
        records.append((name, seq))

    return records


def _validate_sequence(seq: str) -> bool:
    """Validate that sequence contains only valid amino acids.

    Args:
        seq: Protein sequence string

    Returns:
        True if valid, False otherwise
    """
    VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
    return all(aa in VALID_AAS for aa in seq.upper())


# ============================================================================
# CORE COMPUTATION FUNCTION
# ============================================================================


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={CACHE_DIR: ESMFOLD_CACHE_VOLUME},
    secrets=[modal.Secret.from_name("cloudsql-credentials")],  # For GCS access
)
def esmfold(
    fasta_name: str = "input.fasta",
    fasta_str: str = "",
    fasta_gcs_uri: str = "",  # NEW: Support GCS URI input
    num_recycles: int = 3,
    extract_confidence: bool = True,
    upload_to_gcs: bool = False,
    gcs_bucket: str = "dev-services",
    run_id: str | None = None,
) -> dict:
    """Run ESMFold structure prediction on protein sequences.

    Args:
        fasta_name: Name of the input FASTA file (for output naming)
        fasta_str: FASTA formatted protein sequence(s)
        fasta_gcs_uri: GCS URI to FASTA file (alternative to fasta_str)
        num_recycles: Number of recycles for structure prediction
        extract_confidence: Whether to extract and save pLDDT confidence scores
        upload_to_gcs: Whether to upload results to GCS
        gcs_bucket: GCS bucket name
        run_id: Unique run identifier for organizing outputs

    Returns:
        Dictionary with structured output matching other biomodals
    """
    import os
    import subprocess
    import sys
    import tempfile
    from tempfile import TemporaryDirectory

    # Use the conda environment's Python to run ESMFold
    # This avoids Python path conflicts
    CONDA_PYTHON = "/home/vscode/.conda/envs/py39-esmfold/bin/python"

    # Validate inputs BEFORE any processing
    if not fasta_str and not fasta_gcs_uri:
        return {
            "exit_code": 1,
            "error": "esmfold requires either fasta_str or fasta_gcs_uri parameter",
            "output_files": [],
        }

    if fasta_str and isinstance(fasta_str, str) and not fasta_str.strip():
        print("❌ ERROR: fasta_str is empty string")
        return {
            "exit_code": 1,
            "error": "esmfold requires non-empty fasta_str parameter. Got: str with value: (empty)",
            "output_files": [],
        }

    # Download FASTA from GCS if URI provided
    if fasta_gcs_uri and not fasta_str:
        print(f"📥 Downloading FASTA from GCS: {fasta_gcs_uri}")
        try:
            from google.cloud import storage

            # Get credentials from Modal secret
            credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if credentials_json:
                creds_path = "/tmp/gcs_credentials.json"
                with open(creds_path, "w") as f:
                    f.write(credentials_json)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                print("  ✓ Loaded GCS credentials")

            # Parse GCS URI
            if not fasta_gcs_uri.startswith("gs://"):
                return {
                    "exit_code": 1,
                    "error": f"Invalid GCS URI format: {fasta_gcs_uri}",
                    "output_files": [],
                }

            parts = fasta_gcs_uri[5:].split("/", 1)
            if len(parts) != 2:
                return {
                    "exit_code": 1,
                    "error": f"Invalid GCS URI format: {fasta_gcs_uri}",
                    "output_files": [],
                }

            bucket_name, blob_path = parts

            # Download from GCS
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            fasta_str = blob.download_as_text()

            print(f"  ✓ Downloaded FASTA ({len(fasta_str)} bytes)")

            # Validate downloaded content
            if not fasta_str or not fasta_str.strip():
                print("❌ ERROR: Downloaded FASTA is empty")
                return {
                    "exit_code": 1,
                    "error": f"Downloaded FASTA from {fasta_gcs_uri} is empty",
                    "output_files": [],
                }

            # Log first 200 characters for debugging
            print("  📄 FASTA content preview (first 200 chars):")
            print(f"     {fasta_str[:200]}...")

            # Extract filename for naming
            if not fasta_name or fasta_name == "input.fasta":
                fasta_name = Path(blob_path).stem

        except Exception as e:
            print(f"❌ Failed to download from GCS: {e}")
            import traceback

            traceback.print_exc()
            return {
                "exit_code": 1,
                "error": f"Failed to download FASTA from GCS: {e}",
                "output_files": [],
            }

    # Parse and validate FASTA input
    try:
        records = _parse_fasta(fasta_str)
    except Exception as e:
        return {
            "exit_code": 1,
            "error": f"Failed to parse FASTA: {str(e)}",
            "output_files": [],
        }

    if not records:
        return {
            "exit_code": 1,
            "error": "No sequences found in FASTA input or fasta_gcs_uri required",
            "output_files": [],
        }

    # Validate sequences
    for label, seq in records:
        if not _validate_sequence(seq):
            # Find invalid characters for better error message
            VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")
            invalid_chars = [c for c in seq if c not in VALID_AAS]
            return {
                "exit_code": 1,
                "error": (
                    f"Invalid amino acids in sequence '{label}': {set(invalid_chars)}. "
                    f"Only standard 20 amino acids allowed. "
                    f"Note: '/' is a chain separator and should not be in sequences."
                ),
                "output_files": [],
            }
        if len(seq) > 1000:
            print(
                f"⚠️  Warning: Sequence '{label}' is {len(seq)} residues. "
                f"ESMFold may be slow or run out of memory for sequences >1000 residues."
            )

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        output_files_list = []

        # Get base name for output files
        fasta_stem = Path(fasta_name).stem

        # Create a Python script that will run in the conda environment
        script_path = Path(tmpdir) / "run_esmfold.py"
        script_content = f'''
import json
import os
import sys
from pathlib import Path

import esm
import torch

# Set cache directory for model weights (persisted in Modal Volume)
os.environ["TORCH_HOME"] = "{CACHE_DIR}"
os.environ["XDG_CACHE_HOME"] = "{CACHE_DIR}"

# Parse input sequences
records = {records!r}
output_dir = Path("{output_dir}")
num_recycles = {num_recycles}
extract_confidence = {extract_confidence}

# Load model (will use cached weights if available)
print("Loading ESMFold model...")
model = esm.pretrained.esmfold_v1()
model = model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    print(f"Using GPU: {{torch.cuda.get_device_name(0)}}")
else:
    print("Warning: Running on CPU")

results = []

# Process each sequence
for i, (name, seq) in enumerate(records):
    print(f"Processing {{i+1}}/{{len(records)}}: {{name}} ({{len(seq)}} residues)")
    
    try:
        # Generate PDB structure
        with torch.no_grad():
            pdb_str = model.infer_pdb(seq)
        
        # Save PDB
        pdb_path = output_dir / f"{{name}}.pdb"
        pdb_path.write_text(pdb_str)
        
        result_data = {{
            "name": name,
            "pdb_file": f"{{name}}.pdb",
            "seq_len": len(seq),
        }}
        
        # Extract confidence if requested
        if extract_confidence:
            with torch.no_grad():
                output = model.infer(seq, num_recycles=num_recycles)
                plddt = output["plddt"][0].cpu()
            
            conf_data = {{
                "sequence_name": name,
                "sequence_length": len(seq),
                "mean_plddt": float(plddt.mean()),
                "min_plddt": float(plddt.min()),
                "max_plddt": float(plddt.max()),
                "per_residue_plddt": plddt.tolist(),
            }}
            
            conf_path = output_dir / f"{{name}}_plddt.json"
            conf_path.write_text(json.dumps(conf_data, indent=2))
            
            result_data["plddt_file"] = f"{{name}}_plddt.json"
            result_data["mean_plddt"] = conf_data["mean_plddt"]
            
            print(f"  ✓ {{name}} (pLDDT: {{conf_data['mean_plddt']:.1f}})")
        else:
            print(f"  ✓ {{name}}")
        
        results.append(result_data)
        
    except Exception as e:
        print(f"  ✗ Failed: {{str(e)}}")
        results.append({{"name": name, "error": str(e)}})

# Save results summary
summary_path = output_dir / "results.json"
summary_path.write_text(json.dumps(results, indent=2))
print(f"Completed {{len(results)}} sequences")
'''

        script_path.write_text(script_content)

        # Run the script using conda environment's Python
        print("Running ESMFold...")
        result_proc = subprocess.run(
            [CONDA_PYTHON, str(script_path)],
            capture_output=True,
            text=True,
        )

        # Print stdout/stderr
        if result_proc.stdout:
            print(result_proc.stdout)
        if result_proc.stderr:
            print("Stderr:", result_proc.stderr, file=sys.stderr)

        if result_proc.returncode != 0:
            return {
                "exit_code": 1,
                "error": f"ESMFold script failed with exit code {result_proc.returncode}",
                "output_files": [],
                "stdout": result_proc.stdout,
                "stderr": result_proc.stderr,
            }

        # Collect output files with content (for local retrieval)
        for out_file in output_dir.glob("*"):
            if (
                out_file.is_file()
                and out_file.name != "run_esmfold.py"
                and out_file.name != "results.json"
            ):
                suffix = out_file.suffix.lstrip(".")
                artifact_type = suffix if suffix else "file"

                # Parse metadata from results.json if available
                file_metadata = {"filename": out_file.name}

                # Read file content to return it (Modal temporary files are destroyed after function returns)
                file_content = out_file.read_bytes()

                output_files_list.append(
                    {
                        "path": str(out_file.absolute()),  # For reference only
                        "content": file_content,  # Actual file content
                        "filename": out_file.name,
                        "artifact_type": artifact_type,
                        "size": len(file_content),
                        "metadata": file_metadata,
                    }
                )

        # Read results summary for metadata
        results_file = output_dir / "results.json"
        if results_file.exists():
            results_data = json.loads(results_file.read_text())

            # Add mean_plddt to PDB file metadata
            for result in results_data:
                if "mean_plddt" in result:
                    for file_info in output_files_list:
                        if file_info["filename"] == result.get("pdb_file"):
                            file_info["metadata"]["mean_plddt"] = result["mean_plddt"]  # type: ignore[index]
                            file_info["metadata"]["label"] = result["name"]  # type: ignore[index]
                            file_info["metadata"]["seq_len"] = result["seq_len"]  # type: ignore[index]

        num_successful = sum(1 for f in output_files_list if f["artifact_type"] == "pdb")

        # Create summary metadata
        metadata = {
            "fasta_name": fasta_name,
            "num_sequences": len(records),
            "num_structures_generated": num_successful,
            "gpu": GPU,
            "num_recycles": num_recycles,
            "extract_confidence": extract_confidence,
        }

        meta_filename = f"{fasta_stem}.metadata.json"
        meta_path = output_dir / meta_filename
        meta_content = json.dumps(metadata, indent=2)
        meta_path.write_text(meta_content)

        output_files_list.append(
            {
                "path": str(meta_path.absolute()),
                "content": meta_content.encode(),  # Add content for local retrieval
                "filename": meta_filename,
                "artifact_type": "json",
                "size": len(meta_content),
                "metadata": {"type": "run_metadata"},
            }
        )

        # Upload to GCS if requested
        if upload_to_gcs and gcs_bucket and run_id:
            from google.cloud import storage

            # Initialize GCS client with credentials from secret
            credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if not credentials_json:
                return {
                    "exit_code": 1,
                    "error": "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                    "output_files": output_files_list,
                }

            # Write credentials to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:  # type: ignore[assignment]
                f.write(credentials_json)
                credentials_file = f.name
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

            # Create GCS client
            client = storage.Client()
            bucket = client.bucket(gcs_bucket)

            # Define GCS path structure
            gcs_prefix = f"runs/{run_id}/esmfold"

            output_files = []

            # Upload each file from content
            for file_info in output_files_list:
                blob_path = f"{gcs_prefix}/{file_info['filename']}"

                blob = bucket.blob(blob_path)
                blob.upload_from_string(file_info["content"])

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
                "message": f"ESMFold completed: {num_successful} structures predicted",
                "stdout": f"ESMFold completed successfully. Files uploaded to gs://{gcs_bucket}/{gcs_prefix}/",
                "metadata": {
                    "run_id": run_id,
                    "gcs_bucket": gcs_bucket,
                    **metadata,
                },
            }

        else:
            # Return without GCS upload (for local testing)
            return {
                "exit_code": 0,
                "output_files": output_files_list,
                "message": f"ESMFold completed: {num_successful} structures generated",
                "stdout": f"ESMFold prediction completed successfully for {len(records)} sequences.",
                "metadata": metadata,
            }


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================


@app.local_entrypoint()
def main(
    input_fasta: str,
    num_recycles: int = 3,
    extract_confidence: bool = True,
    out_dir: str = "./out/esmfold",
    run_name: str | None = None,
    upload_to_gcs: bool = False,
    gcs_bucket: str = "dev-services",
    run_id: str | None = None,
) -> None:
    """Local CLI entrypoint for testing ESMFold.
    
    Usage:
        # Basic usage
        modal run modal/biomodals/modal_esmfold.py --input-fasta protein.fasta
        
        # Skip confidence extraction
        modal run modal/biomodals/modal_esmfold.py \
            --input-fasta protein.fasta \
            --extract-confidence=False
        
        # With GCS upload
        modal run modal/biomodals/modal_esmfold.py \
            --input-fasta protein.fasta \
            --upload-to-gcs \
            --gcs-bucket dev-services \
            --run-id test_001
    """
    from datetime import datetime

    # Read input FASTA
    fasta_str = Path(input_fasta).read_text()

    # Generate run_id if not provided and uploading to GCS
    if upload_to_gcs and not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Running ESMFold on {input_fasta}...")
    if upload_to_gcs:
        print(f"Will upload to gs://{gcs_bucket}/runs/{run_id}/esmfold/")

    # Run ESMFold
    result = esmfold.remote(
        fasta_name=Path(input_fasta).name,
        fasta_str=fasta_str,
        num_recycles=num_recycles,
        extract_confidence=extract_confidence,
        upload_to_gcs=upload_to_gcs,
        gcs_bucket=gcs_bucket,
        run_id=run_id,
    )

    # Check for errors
    if result["exit_code"] != 0:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        return

    # Print results
    print(f"✅ {result['message']}")
    print(f"✅ Generated {len(result['output_files'])} files")

    # If uploaded to GCS, just print locations
    if upload_to_gcs:
        print(f"\n📦 Files uploaded to: {result['gcs_prefix']}")
        for f in result["output_files"]:
            print(f"  • {f['artifact_type']}: {f['gcs_url']}")
        return

    # Otherwise, copy local temp outputs to out_dir
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Saving files to: {out_dir_full}")
    for f in result["output_files"]:
        dst = out_dir_full / f["filename"]
        # Write content from Modal response
        dst.write_bytes(f["content"])

        # Print with metadata if available
        if f["artifact_type"] == "pdb" and "mean_plddt" in f.get("metadata", {}):
            print(f"  • {f['filename']} (pLDDT: {f['metadata']['mean_plddt']:.1f})")
        else:
            print(f"  • {f['filename']}")

    print(f"\n✅ Output directory: {out_dir_full}")
