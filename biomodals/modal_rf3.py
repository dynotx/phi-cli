"""
RosettaFold3 (RF3) biomodal for structure prediction using Foundry.

RF3 is a state-of-the-art structure prediction model that can predict:
- Protein structures from sequences
- Protein-ligand complexes
- Protein-nucleic acid complexes
- Multi-chain complexes

Usage:
    modal run modal_rf3.py --fasta-content "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"
"""

import json
import os
import uuid
from pathlib import Path

import modal

# GPU configuration
GPU = "A10G"  # RF3 benefits from A10G or better
TIMEOUT = 60  # RF3 can take 30-60 minutes for large proteins

# Foundry image with RF3
# Use package install + manually download required config files
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git", "wget", "curl"])
    .pip_install(
        [
            "rc-foundry[rf3]",  # Install RF3 package
            "google-cloud-storage",
            "biotite",
        ]
    )
    .run_commands(
        # Download Foundry model checkpoints
        "foundry install base-models",
        # Put .project-root where rootutils will find it (searches up from site-packages/rf3/)
        "touch /usr/local/lib/python3.12/site-packages/.project-root",
        # Put configs where RF3 expects them (3 dirs up from rf3/inference.py = /usr/local/lib/python3.12/configs/)
        "mkdir -p /usr/local/lib/python3.12/configs",
        # Download config files from GitHub
        "cd /usr/local/lib/python3.12/configs"
        " && curl -sL -o inference.yaml https://raw.githubusercontent.com/RosettaCommons/foundry/refs/heads/production/models/rf3/configs/inference.yaml"
        " && curl -sL -o model.yaml https://raw.githubusercontent.com/RosettaCommons/foundry/refs/heads/production/models/rf3/configs/model.yaml"
        " && curl -sL -o dataset.yaml https://raw.githubusercontent.com/RosettaCommons/foundry/refs/heads/production/models/rf3/configs/dataset.yaml",
    )
)

app = modal.App(
    "rf3",
    image=image,
)

CHECKPOINT_DIR = Path("/root/.foundry/checkpoints")


@app.function(timeout=10 * 60, gpu=None)  # Simple diagnostics, no GPU needed
def foundry_diagnostics():
    """Check Foundry installation and available models."""
    import pkgutil
    import subprocess

    print("=" * 80)
    print("Foundry Diagnostics")
    print("=" * 80)

    # Check foundry CLI
    result = subprocess.run(["foundry", "list-available"], capture_output=True, text=True)

    print("\nAvailable Foundry models:")
    print(result.stdout)

    # Check checkpoint directory
    print(f"\nCheckpoint directory: {CHECKPOINT_DIR}")
    if CHECKPOINT_DIR.exists():
        print("✓ Directory exists")
        checkpoints = list(CHECKPOINT_DIR.glob("**/*.*"))
        print(f"✓ Found {len(checkpoints)} checkpoint files")
        for ckpt in checkpoints[:10]:  # Show first 10
            print(f"  - {ckpt.name} ({ckpt.stat().st_size / (1024**3):.2f} GB)")
    else:
        print("✗ Directory does not exist")

    # Check what's available in foundry package
    print("\nFoundry Python modules:")
    try:
        import foundry

        for _importer, modname, ispkg in pkgutil.iter_modules(foundry.__path__, prefix="foundry."):
            print(f"  - {modname} ({'package' if ispkg else 'module'})")
    except Exception as e:
        print(f"  Error listing foundry modules: {e}")

    # Check for RF3-related modules in site-packages
    print("\nSearching for RF3-related packages:")
    import sys

    for path in sys.path:
        if "site-packages" in path or "dist-packages" in path:
            try:
                import os

                if os.path.exists(path):
                    for item in os.listdir(path):
                        if "rf" in item.lower() and not item.startswith("."):
                            print(f"  - {item}")
            except Exception:
                pass

    return {
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "checkpoint_exists": CHECKPOINT_DIR.exists(),
        "num_checkpoints": len(list(CHECKPOINT_DIR.glob("**/*.*")))
        if CHECKPOINT_DIR.exists()
        else 0,
    }


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
)
def rf3_predict(
    # Input specification
    fasta_content: str | None = None,
    fasta_gcs_uri: str | None = None,
    sequences: list[str] | None = None,
    # Prediction params
    num_recycles: int = 3,
    num_models: int = 1,
    # Output
    upload_to_gcs: bool = False,
    gcs_bucket: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Predict protein structure using RosettaFold3.

    Args:
        fasta_content: FASTA file content with one or more sequences (DEPRECATED - use fasta_gcs_uri)
        fasta_gcs_uri: GCS URI to FASTA file (e.g., gs://bucket/path/input.faa). Preferred method.
        sequences: List of sequences (alternative to fasta_content)
        num_recycles: Number of recycling iterations (more = better but slower)
        num_models: Number of models to run (1-5)
        upload_to_gcs: Whether to upload outputs to GCS
        gcs_bucket: GCS bucket name
        run_id: Unique run identifier

    Returns:
        Dict with predicted structures and confidence metrics
    """
    import tempfile
    from tempfile import TemporaryDirectory

    run_id = run_id or f"rf3_{uuid.uuid4().hex[:8]}"

    # Handle GCS URI input (preferred method)
    if fasta_gcs_uri:
        print(f"Downloading FASTA from GCS: {fasta_gcs_uri}")

        from google.cloud import storage

        # Initialize GCS client
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not credentials_json:
            return {
                "success": False,
                "error": "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                "run_id": run_id,
            }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(credentials_json)
            credentials_file = f.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

        # Parse GCS URI
        if not fasta_gcs_uri.startswith("gs://"):
            return {
                "success": False,
                "error": f"Invalid GCS URI: {fasta_gcs_uri}",
                "run_id": run_id,
            }

        gcs_path = fasta_gcs_uri.replace("gs://", "")
        bucket_name, blob_path = gcs_path.split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download FASTA content
        fasta_content = blob.download_as_text()
        print(f"  ✓ Downloaded FASTA ({len(fasta_content)} bytes)")

    # Validate input
    if not fasta_content and not sequences:
        return {
            "success": False,
            "error": "Either fasta_content or sequences must be provided",
        }

    print("RF3 structure prediction:")
    print(f"  Run ID: {run_id}")
    print(f"  Num recycles: {num_recycles}")
    print(f"  Num models: {num_models}")

    with TemporaryDirectory() as work_dir, TemporaryDirectory() as out_dir:
        work_path = Path(work_dir)
        out_path = Path(out_dir)

        # Prepare input JSON for RF3
        # RF3 expects a JSON input file with sequence information
        import json

        # Parse sequences from FASTA
        sequences_list = []
        if fasta_content:
            lines = fasta_content.strip().split("\n")
            current_seq = ""
            current_name = ""
            for line in lines:
                if line.startswith(">"):
                    if current_seq:
                        sequences_list.append({"name": current_name, "sequence": current_seq})
                    current_name = line[1:].strip()
                    current_seq = ""
                else:
                    current_seq += line.strip()
            if current_seq:
                sequences_list.append({"name": current_name, "sequence": current_seq})
        elif sequences:
            for i, seq in enumerate(sequences):
                sequences_list.append({"name": f"sequence_{i + 1}", "sequence": seq})

        if not sequences_list:
            return {
                "success": False,
                "error": "No sequences found in input",
            }

        # Create JSON input for RF3
        # Format: array of items, each with "name" and "components"
        # Each component has "seq" and "chain_id"
        rf3_inputs = []
        for seq_info in sequences_list:
            rf3_inputs.append(
                {
                    "name": seq_info["name"],
                    "components": [
                        {
                            "seq": seq_info["sequence"],
                            "chain_id": "A",  # Single chain for now
                        }
                    ],
                }
            )

        # Write JSON file
        json_file = work_path / "input.json"
        json_file.write_text(json.dumps(rf3_inputs, indent=2))
        print(f"Wrote RF3 input JSON: {json_file}")

        # Set up checkpoint path
        checkpoint_file = CHECKPOINT_DIR / "rf3_foundry_01_24_latest_remapped.ckpt"
        if not checkpoint_file.exists():
            # Try alternative names
            alt_names = [
                "rf3_latest.ckpt",
                "rf3_foundry_01_24_latest.ckpt",
            ]
            for alt_name in alt_names:
                alt_file = CHECKPOINT_DIR / alt_name
                if alt_file.exists():
                    checkpoint_file = alt_file
                    break

        if checkpoint_file.exists():
            ckpt_path = str(checkpoint_file)
            print(f"Using checkpoint: {checkpoint_file.name}")
        else:
            return {
                "success": False,
                "error": f"RF3 checkpoint not found in {CHECKPOINT_DIR}",
            }

        try:
            # Call RF3 inference using python -m
            # .project-root is at /usr/local/lib/python3.12/site-packages/ (rootutils will find it)
            # configs are at /usr/local/lib/python3.12/configs/ (Hydra will find them)
            from subprocess import run

            cmd = [
                "python",
                "-m",
                "rf3.inference",
                f"inputs={json_file}",
                f"ckpt_path={ckpt_path}",
                f"+out_dir={out_path}",  # Use + prefix to append to config
            ]

            print("Running RF3 CLI command:")
            print(f"  {' '.join(cmd)}")
            print("⚠️  This may take 10-30 minutes depending on sequence length...")

            result = run(
                cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"✗ RF3 prediction failed with exit code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {
                    "success": False,
                    "error": f"RF3 exited with code {result.returncode}: {result.stderr}",
                }

            print("✓ RF3 prediction complete")
            print(f"STDOUT: {result.stdout}")

            # Process outputs
            # RF3 outputs: input_metrics.csv, input.score, input_model_0.cif.gz, etc.
            output_files = []
            structure_count = 0

            # Find all output files
            import glob
            import gzip

            all_files = glob.glob(f"{out_path}/**/*", recursive=True)

            # Parse metrics CSV if it exists
            metrics = {}
            metrics_file = None
            for filepath in all_files:
                if filepath.endswith("_metrics.csv"):
                    metrics_file = Path(filepath)
                    break

            if metrics_file and metrics_file.exists():
                import csv

                with open(metrics_file) as f:  # type: ignore[assignment]
                    reader = csv.DictReader(f)
                    for row in reader:
                        metrics = dict(row)
                        break  # Only first row
                print(
                    f"Parsed metrics: pTM={metrics.get('ptm', 'N/A')}, pLDDT={metrics.get('plddt', 'N/A')}"
                )

            # Process CIF files
            cif_files = sorted(
                [f for f in all_files if f.endswith(".cif.gz") or f.endswith(".cif")]
            )
            print(f"Found {len(cif_files)} CIF files")

            def extract_confidence_from_cif(cif_path: Path) -> dict:
                """Extract pLDDT scores from CIF file (stored in B-factor column)."""
                try:
                    # Read CIF file
                    if cif_path.suffix == ".gz":
                        with gzip.open(cif_path, "rt") as f:
                            content = f.read()
                    else:
                        with open(cif_path) as f:
                            content = f.read()

                    # Parse B-factor values from _atom_site.B_iso_or_equiv column
                    # CIF format has columns defined by _atom_site.xxx headers
                    b_factors = []
                    atom_names = []

                    lines = content.split("\n")
                    in_atom_site = False
                    b_factor_col = None
                    atom_name_col = None

                    for line in lines:
                        # Find column indices
                        if line.startswith("_atom_site.B_iso_or_equiv"):
                            in_atom_site = True
                            # Count the column index
                            b_factor_col = sum(
                                1 for ln in lines[: lines.index(line)] if ln.startswith("_atom_site.")
                            )
                        elif line.startswith("_atom_site.label_atom_id"):
                            atom_name_col = sum(
                                1 for ln in lines[: lines.index(line)] if ln.startswith("_atom_site.")
                            )
                        elif in_atom_site and line.startswith("ATOM "):
                            # Data line
                            parts = line.split()
                            if len(parts) > max(b_factor_col or 0, atom_name_col or 0):
                                atom_name = (
                                    parts[atom_name_col] if atom_name_col is not None else ""
                                )
                                # Only use CA atoms for per-residue confidence
                                if atom_name == "CA":
                                    try:
                                        b_factor = (
                                            float(parts[b_factor_col])
                                            if b_factor_col is not None
                                            else 0.0
                                        )
                                        b_factors.append(b_factor)
                                        atom_names.append(atom_name)
                                    except (ValueError, IndexError):
                                        pass

                    if b_factors:
                        import numpy as np

                        b_factors_arr = np.array(b_factors)
                        return {
                            "mean_plddt": float(b_factors_arr.mean()),
                            "min_plddt": float(b_factors_arr.min()),
                            "max_plddt": float(b_factors_arr.max()),
                            "num_residues": len(b_factors),
                        }
                except Exception as e:
                    print(f"  Warning: Could not parse confidence from {cif_path.name}: {e}")

                return {}

            for cif_file in cif_files:
                structure_count += 1
                cif_path = Path(cif_file)
                file_size = cif_path.stat().st_size if cif_path.exists() else 0

                # Extract per-structure confidence scores
                structure_metrics = extract_confidence_from_cif(cif_path)

                # If gzipped, decompress for easier viewing
                if cif_path.suffix == ".gz":
                    decompressed_path = out_path / cif_path.stem
                    with gzip.open(cif_path, "rb") as f_in, open(decompressed_path, "wb") as f_out:
                        f_out.write(f_in.read())
                    print(f"  ✓ Decompressed: {decompressed_path.name}")
                    if structure_metrics:
                        print(
                            f"    pLDDT: {structure_metrics['mean_plddt']:.1f} (range: {structure_metrics['min_plddt']:.1f}-{structure_metrics['max_plddt']:.1f})"
                        )

                    # Add both compressed and decompressed versions
                    output_files.append(
                        {
                            "path": str(cif_path.absolute()),
                            "artifact_type": "cif_gz",
                            "filename": cif_path.name,
                            "size": file_size,
                            "metadata": {
                                "model_idx": structure_count,
                                "num_recycles": num_recycles,
                                "compressed": True,
                                **metrics,
                                **structure_metrics,
                            },
                        }
                    )

                    output_files.append(
                        {
                            "path": str(decompressed_path.absolute()),
                            "artifact_type": "cif",
                            "filename": decompressed_path.name,
                            "size": decompressed_path.stat().st_size,
                            "metadata": {
                                "model_idx": structure_count,
                                "num_recycles": num_recycles,
                                "compressed": False,
                                **metrics,
                                **structure_metrics,
                            },
                        }
                    )
                else:
                    if structure_metrics:
                        print(
                            f"  Structure {structure_count}: pLDDT={structure_metrics['mean_plddt']:.1f}"
                        )

                    output_files.append(
                        {
                            "path": str(cif_path.absolute()),
                            "artifact_type": "cif",
                            "filename": cif_path.name,
                            "size": file_size,
                            "metadata": {
                                "model_idx": structure_count,
                                "num_recycles": num_recycles,
                                **metrics,
                                **structure_metrics,
                            },
                        }
                    )

            # Add metrics CSV if it exists
            if metrics_file and metrics_file.exists():
                output_files.append(
                    {
                        "path": str(metrics_file.absolute()),
                        "artifact_type": "csv",
                        "filename": metrics_file.name,
                        "size": metrics_file.stat().st_size,
                        "metadata": {
                            "is_metrics": True,
                        },
                    }
                )

            # Add score file if it exists
            score_files = [f for f in all_files if f.endswith(".score")]
            for score_file in score_files:
                score_path = Path(score_file)
                output_files.append(
                    {
                        "path": str(score_path.absolute()),
                        "artifact_type": "score",
                        "filename": score_path.name,
                        "size": score_path.stat().st_size,
                        "metadata": {
                            "is_score": True,
                        },
                    }
                )

            # Upload to GCS if requested
            gcs_prefix = None
            if upload_to_gcs and gcs_bucket:
                print(f"\nUploading to GCS bucket: {gcs_bucket}")

                credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                if not credentials_json:
                    print("❌ GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets")
                    return {
                        "success": False,
                        "error": "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                        "num_models": structure_count,
                        "output_files": output_files,
                    }

                # Write credentials to temp file
                creds_path = Path("/tmp/gcs_credentials.json")
                creds_path.write_text(credentials_json)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

                from google.cloud import storage

                client = storage.Client()
                bucket = client.bucket(gcs_bucket)

                for file_info in output_files:
                    gcs_path = f"runs/{run_id}/rf3/{file_info['filename']}"
                    blob = bucket.blob(gcs_path)
                    blob.upload_from_filename(file_info["path"])
                    file_info["gcs_url"] = f"gs://{gcs_bucket}/{gcs_path}"
                    print(f"  ✓ Uploaded: {file_info['filename']} -> {file_info['gcs_url']}")

                gcs_prefix = f"gs://{gcs_bucket}/runs/{run_id}/rf3/"

            if structure_count == 0:
                return {
                    "success": False,
                    "error": "No CIF files found in output directory",
                    "num_models": 0,
                }

            print("\n📊 Summary:")
            print(f"  Total models: {structure_count}")
            print(f"  Total files: {len(output_files)}")
            print(f"  Num recycles: {num_recycles}")
            if metrics:
                print(f"  pTM: {metrics.get('ptm', 'N/A')}")
                print(f"  pLDDT: {metrics.get('plddt', 'N/A')}")

            output_result = {
                "run_id": run_id,
                "num_models": structure_count,
                "num_output_files": len(output_files),
                "num_recycles": num_recycles,
                "metrics": metrics,
                "output_files": output_files,
                "success": True,
            }

            if gcs_prefix:
                output_result["gcs_prefix"] = gcs_prefix

            return output_result

        except Exception as e:
            print(f"✗ RF3 prediction failed: {e}")
            import traceback

            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "num_models": 0,
            }


@app.local_entrypoint()
def main(
    fasta_path: str | None = None,
    fasta_content: str | None = None,
    sequence: str | None = None,
    num_recycles: int = 3,
    num_models: int = 1,
    upload_to_gcs: bool = False,
    gcs_bucket: str = "dev-services",
    run_id: str | None = None,
    diagnostics: bool = False,
) -> None:
    """RF3 structure prediction CLI.

    Args:
        fasta_path: Path to FASTA file
        fasta_content: FASTA content as string
        sequence: Single sequence (alternative to FASTA)
        num_recycles: Number of recycling iterations
        num_models: Number of models to predict
        upload_to_gcs: Upload outputs to GCS
        gcs_bucket: GCS bucket name
        run_id: Unique run identifier
        diagnostics: Run diagnostics instead of prediction
    """

    if diagnostics:
        print("Running Foundry diagnostics...")
        result = foundry_diagnostics.remote()
        print(json.dumps(result, indent=2))
        return

    # Prepare input
    if fasta_path:
        fasta_content = Path(fasta_path).read_text()
    elif sequence:
        fasta_content = f">sequence\n{sequence}\n"
    elif not fasta_content:
        print("Error: Must provide --fasta-path, --fasta-content, or --sequence")
        return

    # Run prediction
    result = rf3_predict.remote(
        fasta_content=fasta_content,
        num_recycles=num_recycles,
        num_models=num_models,
        upload_to_gcs=upload_to_gcs,
        gcs_bucket=gcs_bucket,
        run_id=run_id,
    )

    if result.get("success"):
        print("\n✅ RF3 prediction completed successfully!")
        print(f"   Models generated: {result['num_models']}")
        print(f"   Total files: {result['num_output_files']}")
        print(f"   Run ID: {result['run_id']}")

        if upload_to_gcs:
            print(f"\n📦 Files uploaded to gs://{gcs_bucket}/rf3/{result['run_id']}/")
    else:
        print(f"\n❌ RF3 prediction failed: {result.get('error')}")
