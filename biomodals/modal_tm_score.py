# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""
TM-score calculation for protein structure comparison.

This biomodal wraps the TM-score binary from Zhang Lab for topology-based
structural similarity calculation.

TM-score is a metric for measuring structural similarity that is more sensitive
to global fold topology than RMSD.

Usage:
    modal run modal_tm_score.py --reference-pdb ref.pdb --structures pred1.pdb,pred2.pdb
"""

import json
import os
import tempfile
from pathlib import Path

import modal
from modal import App, Image

# ============================================================================
# CONFIGURATION
# ============================================================================

GPU = None  # CPU only
TIMEOUT = int(os.environ.get("TIMEOUT", 15))  # Minutes


# ============================================================================
# IMAGE DEFINITION
# ============================================================================

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(["wget", "g++"])
    .pip_install(
        [
            "biopython==1.83",
            "numpy==1.26.4",
        ]
    )
    # Download and compile TM-score binary
    .run_commands(
        "wget -qnc https://zhanggroup.org/TM-score/TMscore.cpp",
        "g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp",
        "mv TMscore /usr/local/bin/",
    )
    # GCS upload dependency
    .pip_install("google-cloud-storage==2.14.0")
)


# ============================================================================
# APP DEFINITION
# ============================================================================

app = App(
    "tm_score",
    image=image,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def download_from_gcs(gcs_uri: str, local_path: Path) -> None:
    """Download file from GCS URI to local path."""
    import google.cloud.storage as storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    uri_parts = gcs_uri[5:].split("/", 1)
    if len(uri_parts) != 2:
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}")

    bucket_name, object_path = uri_parts

    # Initialize GCS client
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_json:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(credentials_json)
            credentials_file = f.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)

    blob.download_to_filename(str(local_path))


def calculate_tmscore(ref_pdb_path: Path, pred_pdb_path: Path) -> dict[str, float]:
    """
    Calculate TM-score between two structures.

    Args:
        ref_pdb_path: Path to reference PDB
        pred_pdb_path: Path to predicted PDB

    Returns:
        Dict with TM-score metrics (tms, rms, gdt)
    """
    import subprocess

    # Run TMscore binary
    result = subprocess.run(
        ["TMscore", str(pred_pdb_path), str(ref_pdb_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"TMscore failed: {result.stderr}")

    # Parse output
    output = result.stdout

    def parse_float(line: str) -> float:
        """Parse float from TMscore output line."""
        return float(line.split("=")[1].split()[0])

    scores = {}
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("RMSD"):
            scores["rms"] = parse_float(line)
        elif line.startswith("TM-score"):
            scores["tms"] = parse_float(line)
        elif line.startswith("GDT-TS-score"):
            scores["gdt"] = parse_float(line)

    return scores


# ============================================================================
# CORE COMPUTATION FUNCTION
# ============================================================================


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def calculate_tm_scores(
    reference_pdb_gcs_uri: str,
    predicted_structures_gcs_uris: list[str],
    # GCS upload parameters
    upload_to_gcs: bool = False,
    gcs_bucket: str | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Calculate TM-scores for multiple predicted structures.

    Args:
        reference_pdb_gcs_uri: GCS URI to reference PDB
        predicted_structures_gcs_uris: List of GCS URIs to predicted structures
        upload_to_gcs: Whether to upload results to GCS
        gcs_bucket: GCS bucket name
        run_id: Unique run identifier

    Returns:
        Dictionary with TM-score metrics for all structures
    """
    from tempfile import TemporaryDirectory

    print(f"📐 Calculating TM-scores for {len(predicted_structures_gcs_uris)} structures")
    print(f"   Reference: {reference_pdb_gcs_uri}")

    with TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # Download reference structure
        ref_pdb_path = work_dir / "reference.pdb"
        print("\n📥 Downloading reference PDB...")
        download_from_gcs(reference_pdb_gcs_uri, ref_pdb_path)
        print(f"   ✓ Downloaded ({ref_pdb_path.stat().st_size:,} bytes)")

        # Process each predicted structure
        tm_scores = []
        for idx, pred_gcs_uri in enumerate(predicted_structures_gcs_uris):
            print(
                f"\n[{idx + 1}/{len(predicted_structures_gcs_uris)}] Processing {Path(pred_gcs_uri).name}"
            )

            try:
                # Download predicted structure
                pred_pdb_path = work_dir / f"predicted_{idx}.pdb"
                download_from_gcs(pred_gcs_uri, pred_pdb_path)

                # Calculate TM-score
                scores = calculate_tmscore(ref_pdb_path, pred_pdb_path)

                # Extract sequence ID from filename
                sequence_id = Path(pred_gcs_uri).stem

                tm_score_result = {
                    "structure_uri": pred_gcs_uri,
                    "sequence_id": sequence_id,
                    "tm_score": scores.get("tms"),
                    "rmsd": scores.get("rms"),
                    "gdt_ts": scores.get("gdt"),
                }

                tm_scores.append(tm_score_result)

                # Log key metrics
                print(f"   ✓ TM-score: {scores.get('tms', 0):.3f}")
                print(f"     RMSD: {scores.get('rms', 0):.2f} Å")
                print(f"     GDT-TS: {scores.get('gdt', 0):.3f}")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                tm_scores.append(
                    {
                        "structure_uri": pred_gcs_uri,
                        "sequence_id": Path(pred_gcs_uri).stem,
                        "error": str(e),
                        "tm_score": None,
                        "rmsd": None,
                        "gdt_ts": None,
                    }
                )

        # Save TM-scores to JSON
        output_dir = work_dir / "output"
        output_dir.mkdir(exist_ok=True)

        tm_scores_json_path = output_dir / "tm_scores.json"
        tm_scores_json_path.write_text(json.dumps(tm_scores, indent=2))

        print("\n📊 Summary:")
        print(f"   Total structures: {len(tm_scores)}")
        successful = sum(1 for s in tm_scores if "error" not in s)
        print(f"   Successfully scored: {successful}")
        failed = len(tm_scores) - successful
        if failed > 0:
            print(f"   Failed: {failed}")

        # Calculate summary statistics
        successful_scores = [s for s in tm_scores if "error" not in s and s["tm_score"] is not None]
        if successful_scores:
            import statistics

            # Filter for numeric TM scores - already validated by isinstance check
            tm_values: list[float] = []
            for s in successful_scores:
                score = s.get("tm_score")
                if score is not None and isinstance(score, (int, float)):
                    tm_values.append(float(score))

            summary_stats = {
                "num_structures": len(tm_scores),
                "num_successful": len(successful_scores),
                "num_failed": failed,
                "tm_score_mean": statistics.mean(tm_values) if tm_values else 0.0,
                "tm_score_min": min(tm_values) if tm_values else 0.0,
                "tm_score_max": max(tm_values) if tm_values else 0.0,
                "tm_score_median": statistics.median(tm_values) if tm_values else 0.0,
            }

            print(
                f"\n   TM-score: {summary_stats['tm_score_mean']:.3f} (min: {summary_stats['tm_score_min']:.3f}, max: {summary_stats['tm_score_max']:.3f})"
            )
        else:
            summary_stats = {
                "num_structures": len(tm_scores),
                "num_successful": 0,
                "num_failed": failed,
            }

        # Collect output files
        output_files_list = [
            {
                "path": str(tm_scores_json_path.absolute()),
                "filename": tm_scores_json_path.name,
                "artifact_type": "json",
                "size": tm_scores_json_path.stat().st_size,
                "metadata": {
                    "reference_pdb": reference_pdb_gcs_uri,
                    "num_structures": len(tm_scores),
                },
            }
        ]

        # GCS upload
        if upload_to_gcs and gcs_bucket and run_id:
            import google.cloud.storage as storage

            # Initialize GCS client
            credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if credentials_json:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                    f.write(credentials_json)
                    credentials_file = f.name
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

            client = storage.Client()
            bucket = client.bucket(gcs_bucket)

            tool_name = "tm_score"
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
                "message": f"Calculated TM-scores for {len(successful_scores)}/{len(tm_scores)} structures",
                "metadata": {
                    "run_id": run_id,
                    "tool_name": tool_name,
                    "tm_scores": tm_scores,
                    "summary": summary_stats,
                },
            }

        else:
            # Return without GCS upload
            return {
                "exit_code": 0,
                "output_files": output_files_list,
                "message": f"Calculated TM-scores for {len(successful_scores)}/{len(tm_scores)} structures",
                "metadata": {
                    "tm_scores": tm_scores,
                    "summary": summary_stats,
                },
            }


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================


@app.local_entrypoint()
def main(
    reference_pdb: str,
    structures: str,  # Comma-separated GCS URIs or local paths
    out_dir: str = "./out/tm_score",
    run_name: str | None = None,
) -> None:
    """
    Local CLI entrypoint for testing TM-score calculation.

    Usage:
        modal run modal_tm_score.py --reference-pdb ref.pdb --structures pred1.pdb,pred2.pdb
    """
    from datetime import datetime

    # Parse structure list
    structure_uris = [s.strip() for s in structures.split(",") if s.strip()]

    print("📐 TM-score Calculation Tool")
    print(f"   Reference: {reference_pdb}")
    print(f"   Structures: {len(structure_uris)}")

    # Run TM-score calculation
    result = calculate_tm_scores.remote(
        reference_pdb_gcs_uri=reference_pdb,
        predicted_structures_gcs_uris=structure_uris,
    )

    if result["exit_code"] != 0:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        return

    # Save outputs locally
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    for file_info in result["output_files"]:
        if "path" in file_info:
            local_path = Path(str(file_info["path"]))
            if local_path.exists():
                dest_path = out_dir_full / file_info["filename"]

                import shutil

                shutil.copy2(local_path, dest_path)
                print(f"\n✓ Saved: {dest_path}")

    print("\n✅ TM-score calculation complete!")
    print(f"   Output directory: {out_dir_full}")
    print(f"   {result['message']}")
