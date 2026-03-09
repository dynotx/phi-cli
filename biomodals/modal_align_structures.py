# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""
Structural alignment and RMSD calculation for protein design workflows.

This biomodal aligns predicted structures against a reference structure and calculates:
- RMSD (overall and per-region)
- pLDDT confidence scores
- Structural quality metrics

Runs on CPU (no GPU needed) using BioPython for fast alignment.

Usage:
    modal run modal_align_structures.py --reference-pdb reference.pdb --structures structure1.pdb,structure2.pdb
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import modal
from modal import App, Image
from typing_extensions import TypedDict

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
            "pydantic==2.9.2",
        ]
    )
    # GCS upload dependency (required for all biomodals)
    .pip_install("google-cloud-storage==2.14.0")
    # TM-score binary for topology comparison (optional)
    .run_commands(
        "wget -qnc https://zhanggroup.org/TM-score/TMscore.cpp",
        "g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp",
        "mv TMscore /usr/local/bin/",
    )
)


# ============================================================================
# APP DEFINITION
# ============================================================================

app = App(
    "align_structures",
    image=image,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def download_from_gcs(gcs_uri: str, local_path: Path) -> None:
    """Download file from GCS URI to local path."""
    from google.cloud import storage

    # Parse GCS URI: gs://bucket/path/to/file -> (bucket, path/to/file)
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    uri_parts = gcs_uri[5:].split("/", 1)  # Remove "gs://"
    if len(uri_parts) != 2:
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}")

    bucket_name, object_path = uri_parts

    # Initialize GCS client with credentials from secret
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not credentials_json:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets. "
            "Ensure Modal secret 'cloudsql-credentials' contains GCS credentials."
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(credentials_json)
        credentials_file = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_path)

    blob.download_to_filename(str(local_path))


class PlddtStats(TypedDict):
    """Type definition for pLDDT statistics."""

    mean_plddt: float | None
    min_plddt: float | None
    max_plddt: float | None
    plddt_scores: list[float]


def extract_plddt_from_pdb(structure: Any) -> PlddtStats:
    """
    Extract pLDDT scores from PDB B-factor column.

    Args:
        structure: BioPython Structure object

    Returns:
        Dict with pLDDT statistics
    """
    plddt_scores = []

    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip hetero atoms
                if residue.id[0] != " ":
                    continue

                # B-factor stores pLDDT in AlphaFold/ESMFold outputs
                for atom in residue:
                    if atom.name == "CA":  # Use CA atom
                        plddt_scores.append(atom.bfactor)
                        break

    if not plddt_scores:
        return PlddtStats(
            mean_plddt=None,
            min_plddt=None,
            max_plddt=None,
            plddt_scores=[],
        )

    import numpy as np

    return PlddtStats(
        mean_plddt=float(np.mean(plddt_scores)),
        min_plddt=float(np.min(plddt_scores)),
        max_plddt=float(np.max(plddt_scores)),
        plddt_scores=[float(s) for s in plddt_scores],
    )


def extract_motif_residues(chain: Any, motif_residues: list[str]) -> list[Any]:
    """
    Extract CA atoms for specified motif residues.

    Args:
        chain: BioPython Chain object
        motif_residues: List of residue IDs (e.g., ["A45", "A67"])

    Returns:
        List of CA atoms for motif residues
    """
    motif_atoms = []

    # Parse motif residues (e.g., "A45" -> chain A, residue 45)
    motif_positions: dict[str, list[int]] = {}
    for res_id in motif_residues:
        if not res_id:
            continue
        # Assume format: CHAINNUM (e.g., "A45")
        chain_id = res_id[0]
        res_num = int(res_id[1:])
        if chain_id not in motif_positions:
            motif_positions[chain_id] = []
        motif_positions[chain_id].append(res_num)

    # Extract atoms for this chain if it matches
    chain_id = chain.id
    if chain_id in motif_positions:
        for residue in chain:
            if residue.id[0] != " ":  # Skip hetero atoms
                continue
            res_num = residue.id[1]
            if res_num in motif_positions[chain_id]:
                for atom in residue:
                    if atom.name == "CA":
                        motif_atoms.append(atom)
                        break

    return motif_atoms


def calculate_tmscore(ref_pdb_path: Path, pred_pdb_path: Path) -> dict[str, float] | None:
    """
    Calculate TM-score between two structures.

    Args:
        ref_pdb_path: Path to reference PDB
        pred_pdb_path: Path to predicted PDB

    Returns:
        Dict with TM-score metrics (tm_score, gdt_ts) or None if TM-score not available
    """
    import subprocess

    try:
        # Run TMscore binary
        result = subprocess.run(
            ["TMscore", str(pred_pdb_path), str(ref_pdb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return None

        # Parse output
        output = result.stdout

        def parse_float(line: str) -> float:
            return float(line.split("=")[1].split()[0])

        scores = {}
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("TM-score"):
                scores["tm_score"] = parse_float(line)
            elif line.startswith("GDT-TS-score"):
                scores["gdt_ts"] = parse_float(line)

        return scores if scores else None

    except Exception:
        return None


def align_structures(
    reference_pdb_path: Path,
    predicted_pdb_path: Path,
    motif_residues: list[str] | None = None,
    calculate_tm_score: bool = False,
) -> dict[str, Any]:
    """
    Align predicted structure to reference and calculate RMSD.

    Args:
        reference_pdb_path: Path to reference PDB
        predicted_pdb_path: Path to predicted PDB
        motif_residues: Optional list of motif residues (e.g., ["A45", "A67"])
        calculate_tm_score: Whether to calculate TM-score (optional, slower)

    Returns:
        Dict with alignment metrics
    """
    from Bio.PDB import PDBParser, Superimposer

    parser = PDBParser(QUIET=True)

    # Parse structures
    ref_structure = parser.get_structure("reference", reference_pdb_path)
    pred_structure = parser.get_structure("predicted", predicted_pdb_path)

    # Extract CA atoms
    ref_atoms = []
    pred_atoms = []

    for ref_chain, pred_chain in zip(ref_structure[0], pred_structure[0]):
        for ref_res, pred_res in zip(ref_chain, pred_chain):
            # Skip hetero atoms
            if ref_res.id[0] != " " or pred_res.id[0] != " ":
                continue

            # Get CA atoms
            if "CA" in ref_res and "CA" in pred_res:
                ref_atoms.append(ref_res["CA"])
                pred_atoms.append(pred_res["CA"])

    if len(ref_atoms) == 0 or len(pred_atoms) == 0:
        raise ValueError("No matching CA atoms found between structures")

    # Align structures using Superimposer
    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, pred_atoms)

    rmsd_overall = float(super_imposer.rms) if super_imposer.rms is not None else 0.0

    # Calculate motif RMSD if specified
    rmsd_motif = None
    if motif_residues:
        ref_motif_atoms = []
        pred_motif_atoms = []

        for ref_chain, pred_chain in zip(ref_structure[0], pred_structure[0]):
            ref_motif_atoms.extend(extract_motif_residues(ref_chain, motif_residues))
            pred_motif_atoms.extend(extract_motif_residues(pred_chain, motif_residues))

        if len(ref_motif_atoms) > 0 and len(ref_motif_atoms) == len(pred_motif_atoms):
            motif_super_imposer = Superimposer()
            motif_super_imposer.set_atoms(ref_motif_atoms, pred_motif_atoms)
            rmsd_motif = (
                float(motif_super_imposer.rms) if motif_super_imposer.rms is not None else 0.0
            )

    # Extract pLDDT scores
    plddt_stats = extract_plddt_from_pdb(pred_structure)

    # Calculate motif pLDDT if motif residues specified
    min_plddt_motif = None
    mean_plddt_motif = None
    if motif_residues and plddt_stats["plddt_scores"]:
        motif_plddt_scores = []
        for res_id in motif_residues:
            if not res_id:
                continue
            # Extract position index (simplified - assumes single chain)
            try:
                res_num = int(res_id[1:])
                # Get pLDDT for this residue (1-indexed)
                if 0 <= res_num - 1 < len(plddt_stats["plddt_scores"]):
                    motif_plddt_scores.append(plddt_stats["plddt_scores"][res_num - 1])
            except (ValueError, IndexError):
                continue

        if motif_plddt_scores:
            import numpy as np

            min_plddt_motif = float(np.min(motif_plddt_scores))
            mean_plddt_motif = float(np.mean(motif_plddt_scores))

    # Calculate TM-score if requested
    tm_score = None
    gdt_ts = None
    if calculate_tm_score:
        tm_scores = calculate_tmscore(reference_pdb_path, predicted_pdb_path)
        if tm_scores:
            tm_score = tm_scores.get("tm_score")
            gdt_ts = tm_scores.get("gdt_ts")

    return {
        "rmsd_overall": rmsd_overall,
        "rmsd_motif": rmsd_motif,
        "num_aligned_residues": len(ref_atoms),
        "mean_plddt": plddt_stats["mean_plddt"],
        "min_plddt": plddt_stats["min_plddt"],
        "max_plddt": plddt_stats["max_plddt"],
        "min_plddt_motif": min_plddt_motif,
        "mean_plddt_motif": mean_plddt_motif,
        "tm_score": tm_score,
        "gdt_ts": gdt_ts,
    }


# ============================================================================
# CORE COMPUTATION FUNCTION
# ============================================================================


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def align_structures_batch(
    reference_pdb_gcs_uri: str,
    predicted_structures_gcs_uris: list[str],
    motif_residues: str | None = None,
    calculate_tm_score: bool = False,
    # GCS upload parameters (standard for all biomodals)
    upload_to_gcs: bool = False,
    gcs_bucket: str | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Align multiple predicted structures to a reference structure.

    Args:
        reference_pdb_gcs_uri: GCS URI to reference PDB (gs://bucket/path/ref.pdb)
        predicted_structures_gcs_uris: List of GCS URIs to predicted structures
        motif_residues: Optional comma-separated motif residues (e.g., "A45,A67,A89")
        upload_to_gcs: Whether to upload results to GCS
        gcs_bucket: GCS bucket name (e.g., "dev-services")
        run_id: Unique run identifier for organizing outputs

    Returns:
        Dictionary with alignment metrics for all structures
    """
    from tempfile import TemporaryDirectory

    print(f"🔬 Aligning {len(predicted_structures_gcs_uris)} structures to reference")
    print(f"   Reference: {reference_pdb_gcs_uri}")

    # Parse motif residues if provided
    motif_residues_list = None
    if motif_residues:
        motif_residues_list = [r.strip() for r in motif_residues.split(",") if r.strip()]
        print(f"   Motif residues: {motif_residues_list}")

    with TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # Download reference structure
        ref_pdb_path = work_dir / "reference.pdb"
        print("\n📥 Downloading reference PDB...")
        download_from_gcs(reference_pdb_gcs_uri, ref_pdb_path)
        print(f"   ✓ Downloaded ({ref_pdb_path.stat().st_size:,} bytes)")

        # Process each predicted structure
        alignments = []
        for idx, pred_gcs_uri in enumerate(predicted_structures_gcs_uris):
            print(
                f"\n[{idx + 1}/{len(predicted_structures_gcs_uris)}] Processing {Path(pred_gcs_uri).name}"
            )

            try:
                # Download predicted structure
                pred_pdb_path = work_dir / f"predicted_{idx}.pdb"
                download_from_gcs(pred_gcs_uri, pred_pdb_path)

                # Align and calculate metrics
                metrics = align_structures(
                    ref_pdb_path,
                    pred_pdb_path,
                    motif_residues_list,
                    calculate_tm_score,
                )

                # Extract sequence ID from filename (e.g., "seq_001.pdb" -> "seq_001")
                sequence_id = Path(pred_gcs_uri).stem

                alignment_result = {
                    "structure_uri": pred_gcs_uri,
                    "sequence_id": sequence_id,
                    **metrics,
                }

                alignments.append(alignment_result)

                # Log key metrics
                print(f"   ✓ RMSD: {metrics['rmsd_overall']:.2f} Å")
                if metrics["rmsd_motif"] is not None:
                    print(f"     Motif RMSD: {metrics['rmsd_motif']:.2f} Å")
                if metrics["mean_plddt"] is not None:
                    print(f"     Mean pLDDT: {metrics['mean_plddt']:.1f}")
                if metrics.get("tm_score") is not None:
                    print(f"     TM-score: {metrics['tm_score']:.3f}")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                # Include failed alignment in results
                alignments.append(
                    {
                        "structure_uri": pred_gcs_uri,
                        "sequence_id": Path(pred_gcs_uri).stem,
                        "error": str(e),
                        "rmsd_overall": None,
                        "rmsd_motif": None,
                        "mean_plddt": None,
                    }
                )

        # Save alignments to JSON
        output_dir = work_dir / "output"
        output_dir.mkdir(exist_ok=True)

        alignments_json_path = output_dir / "alignment_metrics.json"
        alignments_json_path.write_text(json.dumps(alignments, indent=2))

        print("\n📊 Summary:")
        print(f"   Total structures: {len(alignments)}")
        successful = sum(1 for a in alignments if "error" not in a)
        print(f"   Successfully aligned: {successful}")
        failed = len(alignments) - successful
        if failed > 0:
            print(f"   Failed: {failed}")

        # Calculate summary statistics
        successful_alignments = [
            a for a in alignments if "error" not in a and a["rmsd_overall"] is not None
        ]
        if successful_alignments:
            import numpy as np

            rmsd_values = [a["rmsd_overall"] for a in successful_alignments]
            plddt_values = [
                a["mean_plddt"] for a in successful_alignments if a["mean_plddt"] is not None
            ]

            summary_stats = {
                "num_structures": len(alignments),
                "num_successful": len(successful_alignments),
                "num_failed": failed,
                "rmsd_mean": float(np.mean(rmsd_values)),
                "rmsd_min": float(np.min(rmsd_values)),
                "rmsd_max": float(np.max(rmsd_values)),
            }

            if plddt_values:
                summary_stats.update(
                    {
                        "plddt_mean": float(np.mean(plddt_values)),
                        "plddt_min": float(np.min(plddt_values)),
                        "plddt_max": float(np.max(plddt_values)),
                    }
                )

            print(
                f"\n   RMSD: {summary_stats['rmsd_mean']:.2f} Å (min: {summary_stats['rmsd_min']:.2f}, max: {summary_stats['rmsd_max']:.2f})"
            )
            if plddt_values:
                print(
                    f"   pLDDT: {summary_stats['plddt_mean']:.1f} (min: {summary_stats['plddt_min']:.1f}, max: {summary_stats['plddt_max']:.1f})"
                )
        else:
            summary_stats = {
                "num_structures": len(alignments),
                "num_successful": 0,
                "num_failed": failed,
            }

        # Collect output files
        output_files_list = [
            {
                "path": str(alignments_json_path.absolute()),
                "filename": alignments_json_path.name,
                "artifact_type": "json",
                "size": alignments_json_path.stat().st_size,
                "metadata": {
                    "reference_pdb": reference_pdb_gcs_uri,
                    "num_structures": len(alignments),
                    "motif_residues": motif_residues,
                },
            }
        ]

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

            tool_name = "align_structures"
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
                "message": f"Aligned {len(successful_alignments)}/{len(alignments)} structures successfully. Results uploaded to GCS.",
                "metadata": {
                    "run_id": run_id,
                    "tool_name": tool_name,
                    "alignments": alignments,
                    "summary": summary_stats,
                },
            }

        else:
            # Return without GCS upload
            return {
                "exit_code": 0,
                "output_files": output_files_list,
                "message": f"Aligned {len(successful_alignments)}/{len(alignments)} structures successfully",
                "metadata": {
                    "alignments": alignments,
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
    motif_residues: str | None = None,
    calculate_tm_score: bool = False,
    out_dir: str = "./out/align_structures",
    run_name: str | None = None,
) -> None:
    """
    Local CLI entrypoint for testing alignment.

    Usage:
        modal run modal_align_structures.py --reference-pdb ref.pdb --structures pred1.pdb,pred2.pdb
        modal run modal_align_structures.py --reference-pdb gs://bucket/ref.pdb --structures gs://bucket/pred1.pdb,gs://bucket/pred2.pdb --motif-residues A45,A67
    """
    from datetime import datetime

    # Parse structure list
    structure_uris = [s.strip() for s in structures.split(",") if s.strip()]

    print("🔬 Structural Alignment Tool")
    print(f"   Reference: {reference_pdb}")
    print(f"   Structures: {len(structure_uris)}")
    if motif_residues:
        print(f"   Motif residues: {motif_residues}")

    # Run alignment
    result = align_structures_batch.remote(
        reference_pdb_gcs_uri=reference_pdb,
        predicted_structures_gcs_uris=structure_uris,
        motif_residues=motif_residues,
        calculate_tm_score=calculate_tm_score,
    )

    if result["exit_code"] != 0:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        return

    # Save outputs locally
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    for file_info in result["output_files"]:
        # Check if path exists (for local testing without GCS)
        if "path" in file_info:
            local_path = Path(str(file_info["path"]))
            if local_path.exists():
                dest_path = out_dir_full / file_info["filename"]

                import shutil

                shutil.copy2(local_path, dest_path)
                print(f"\n✓ Saved: {dest_path}")

    print("\n✅ Alignment complete!")
    print(f"   Output directory: {out_dir_full}")
    print(f"   {result['message']}")
