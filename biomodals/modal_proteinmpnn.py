# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""ProteinMPNN via Foundry for sequence design on Modal.

ProteinMPNN designs protein sequences for given backbone structures.
This implementation uses the Foundry package from Baker Lab.

## Example: Design sequences for a backbone

```bash
modal run modal_proteinmpnn.py \
  --pdb-path backbone.pdb \
  --num-sequences 10 \
  --temperature 0.1
```

## Example: Design with fixed positions

```bash
modal run modal_proteinmpnn.py \
  --pdb-path backbone.pdb \
  --num-sequences 10 \
  --fixed-positions "A10-20,A45-50" \
  --temperature 0.1
```
"""

import os
import uuid
from pathlib import Path

import modal

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 30))

# ProteinMPNN image using standalone repository (similar to LigandMPNN approach)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "wget"])
    .pip_install(
        [
            "torch==2.0.1",
            "numpy==1.23.5",
            "biopython==1.79",
            "biotite>=0.41.0",  # For CIF → PDB conversion
            "google-cloud-storage>=2.18.0",
        ]
    )
    .run_commands(
        # Clone ProteinMPNN repository
        "git clone https://github.com/dauparas/ProteinMPNN.git /ProteinMPNN",
        # Create weights directory and download model weights
        "mkdir -p /ProteinMPNN/vanilla_model_weights",
        "cd /ProteinMPNN/vanilla_model_weights"
        " && wget -O v_48_020.pt https://files.ipd.uw.edu/pub/ProteinMPNN/model_weights/v_48_020.pt || echo 'Weight download failed, will try at runtime'",
    )
)

app = modal.App("proteinmpnn", image=image)

CHECKPOINT_PATH = "/ProteinMPNN/vanilla_model_weights/v_48_020.pt"


def parse_fixed_positions(fixed_positions_str: str) -> dict[str, list[int]]:
    """
    Parse comma-separated position string into dict by chain.

    Args:
        fixed_positions_str: Comma-separated positions (e.g., "A52,A56,A63" or "A10,B25,C30")

    Returns:
        Dict mapping chain ID to list of residue numbers (e.g., {"A": [52, 56, 63]})

    Example:
        parse_fixed_positions("A52,A56,A63") -> {"A": [52, 56, 63]}
        parse_fixed_positions("A10,B25,C30") -> {"A": [10], "B": [25], "C": [30]}
    """
    positions_by_chain: dict[str, list[int]] = {}
    for pos in fixed_positions_str.split(","):
        pos = pos.strip()
        if not pos:
            continue
        # Extract chain (first character) and residue number (rest)
        chain = pos[0].upper()
        try:
            resnum = int(pos[1:])
            if chain not in positions_by_chain:
                positions_by_chain[chain] = []
            positions_by_chain[chain].append(resnum)
        except ValueError:
            print(f"Warning: Could not parse position '{pos}', skipping")
            continue
    return positions_by_chain


def create_fixed_positions_jsonl(
    positions_by_chain: dict[str, list[int]], output_path: Path
) -> Path:
    """
    Create JSONL file for ProteinMPNN fixed positions.

    Args:
        positions_by_chain: Dict mapping chain ID to list of residue numbers
        output_path: Path where JSONL file should be written

    Returns:
        Path to created JSONL file

    The JSONL file contains a single JSON object with chain IDs as keys
    and lists of residue numbers as values.
    """
    import json

    # Sort residue numbers for each chain
    sorted_positions = {chain: sorted(resnums) for chain, resnums in positions_by_chain.items()}

    # ProteinMPNN expects: {pdb_name: {chain: [positions]}}
    # The PDB file is saved as "input.pdb", so pdb_name is "input"
    pdb_name = "input"
    fixed_positions_dict = {pdb_name: sorted_positions}

    # Write JSONL file (single line with JSON object)
    with open(output_path, "w") as f:
        json.dump(fixed_positions_dict, f)

    print(f"Created fixed positions JSONL: {output_path}")
    print(f"  PDB name: {pdb_name}")
    print(f"  Positions by chain: {sorted_positions}")
    print(f"  Full JSONL: {json.dumps(fixed_positions_dict)}")
    return output_path


@app.function(
    gpu=GPU,
    timeout=TIMEOUT * 60,
    secrets=[
        modal.Secret.from_name("cloudsql-credentials")
    ],  # For GCS access via GOOGLE_APPLICATION_CREDENTIALS_JSON
)
def proteinmpnn_design(
    pdb_content: str | None = None,
    pdb_gcs_uri: str | None = None,
    num_sequences: int = 10,
    temperature: float = 0.1,
    fixed_positions: str | None = None,
    design_only_chain: str | None = None,
    upload_to_gcs: bool = False,
    gcs_bucket: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Design protein sequences using ProteinMPNN.

    Args:
        pdb_content: PDB file content as string (either this OR pdb_gcs_uri required)
        pdb_gcs_uri: GCS URI (gs://bucket/path) or signed URL (https://...) - will be downloaded
        num_sequences: Number of sequences to design
        temperature: Sampling temperature (lower = more conservative)
        fixed_positions: Positions to keep fixed (e.g., "A10-20,A45-50")
        design_only_chain: Single chain to design (e.g., "A"). Other chains will be fixed. Useful for binder design.
        upload_to_gcs: Whether to upload outputs to GCS
        gcs_bucket: GCS bucket name
        run_id: Unique run identifier

    Returns:
        Dict with designed sequences and metadata
    """
    from subprocess import run
    from tempfile import TemporaryDirectory

    run_id = run_id or f"proteinmpnn_{uuid.uuid4().hex[:8]}"

    print("=" * 60)
    print("ProteinMPNN sequence design:")
    print(f"  Num sequences: {num_sequences}")
    print(f"  Temperature: {temperature}")
    if fixed_positions:
        print(f"  Fixed positions: {fixed_positions}")
    if design_only_chain:
        print(f"  Design only chain: {design_only_chain} (others will be fixed)")
    print(f"  pdb_content provided: {pdb_content is not None}")
    print(f"  pdb_gcs_uri provided: {pdb_gcs_uri is not None}")
    if pdb_gcs_uri:
        print(f"  pdb_gcs_uri: {pdb_gcs_uri[:100]}...")
    print(f"  upload_to_gcs: {upload_to_gcs}")
    print(f"  gcs_bucket: {gcs_bucket}")
    print(f"  run_id: {run_id}")
    print("=" * 60)

    # Validate input
    if not pdb_content and not pdb_gcs_uri:
        print("❌ ERROR: Neither pdb_content nor pdb_gcs_uri provided")
        return {
            "success": False,
            "exit_code": 1,
            "error": "Either pdb_content or pdb_gcs_uri must be provided",
            "sequences": [],
            "output_files": [],
        }

    try:
        with TemporaryDirectory() as work_dir:
            work_path = Path(work_dir)
            out_path = work_path / "output"
            out_path.mkdir()
            print(f"Working directory: {work_path}")

            # Get PDB content - either from parameter or download from GCS
            if pdb_gcs_uri:
                print(f"📥 Downloading PDB from: {pdb_gcs_uri[:100]}...")

                if pdb_gcs_uri.startswith("gs://"):
                    # Download from GCS
                    try:
                        import json
                        import os

                        from google.cloud import storage

                        print("  Initializing GCS client...")

                        # Get credentials from Modal secret
                        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                        if credentials_json:
                            # Write credentials to temp file
                            creds_path = "/tmp/gcs_credentials.json"
                            with open(creds_path, "w") as f:
                                f.write(credentials_json)
                            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                            print("  ✓ Loaded GCS credentials from secret")
                        else:
                            print(
                                "  ⚠️ GOOGLE_APPLICATION_CREDENTIALS_JSON not found, trying default credentials..."
                            )

                        # Extract bucket and blob path
                        parts = pdb_gcs_uri[5:].split("/", 1)  # Remove "gs://" prefix
                        if len(parts) != 2:
                            print(f"❌ Invalid GCS URI format: {pdb_gcs_uri}")
                            return {
                                "success": False,
                                "exit_code": 1,
                                "error": f"Invalid GCS URI format: {pdb_gcs_uri}",
                                "sequences": [],
                                "output_files": [],
                            }

                        bucket_name, blob_path = parts
                        print(f"  Bucket: {bucket_name}, Path: {blob_path}")

                        # Download from GCS
                        client = storage.Client()
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob(blob_path)

                        print("  Downloading blob...")
                        pdb_content = blob.download_as_text()
                        print(f"✓ Downloaded PDB from GCS: {len(pdb_content)} characters")
                    except Exception as e:
                        print(f"❌ Failed to download from GCS: {e}")
                        import traceback

                        traceback.print_exc()
                        return {
                            "success": False,
                            "exit_code": 1,
                            "error": f"Failed to download PDB from GCS: {e}",
                            "sequences": [],
                            "output_files": [],
                        }

                elif pdb_gcs_uri.startswith("https://") or pdb_gcs_uri.startswith("http://"):
                    # Download from signed URL
                    try:
                        import urllib.request

                        print("  Downloading from HTTP/HTTPS URL...")
                        with urllib.request.urlopen(pdb_gcs_uri) as response:
                            pdb_content = response.read().decode("utf-8")
                        print(f"✓ Downloaded PDB from signed URL: {len(pdb_content)} characters")
                    except Exception as e:
                        print(f"❌ Failed to download from URL: {e}")
                        return {
                            "success": False,
                            "exit_code": 1,
                            "error": f"Failed to download PDB from URL: {e}",
                            "sequences": [],
                            "output_files": [],
                        }
                else:
                    print(f"❌ Unsupported URI format: {pdb_gcs_uri}")
                    return {
                        "success": False,
                        "exit_code": 1,
                        "error": f"Unsupported URI format: {pdb_gcs_uri}",
                        "sequences": [],
                        "output_files": [],
                    }

            # Write input PDB
            if not pdb_content:
                print("❌ ERROR: No PDB content available after processing")
                return {
                    "success": False,
                    "exit_code": 1,
                    "error": "No PDB content available after processing URI",
                    "sequences": [],
                    "output_files": [],
                }

            # Check if content is CIF format and convert to PDB if needed
            # ProteinMPNN parser only supports PDB format
            if pdb_content.strip().startswith("data_") or ".cif" in (pdb_gcs_uri or ""):
                print("🔄 Detected CIF format, converting to PDB...")
                try:
                    from biotite.structure.io.pdb import PDBFile
                    from biotite.structure.io.pdbx import PDBxFile, get_structure

                    # Write CIF to temp file
                    cif_file = work_path / "input.cif"
                    cif_file.write_text(pdb_content)

                    # Load from CIF using correct biotite API
                    pdbx_file = PDBxFile.read(str(cif_file))
                    structure = get_structure(pdbx_file, model=1)  # Get first model

                    # Write as PDB
                    pdb_file = work_path / "input.pdb"
                    out_pdb = PDBFile()
                    out_pdb.set_structure(structure)
                    out_pdb.write(str(pdb_file))

                    print(f"✓ Converted CIF → PDB: {pdb_file}")
                except Exception as e:
                    print(f"❌ CIF conversion failed: {e}")
                    import traceback

                    traceback.print_exc()
                    return {
                        "success": False,
                        "exit_code": 1,
                        "error": f"CIF to PDB conversion failed: {e}",
                        "sequences": [],
                        "output_files": [],
                    }
            else:
                # Already PDB format
                pdb_file = work_path / "input.pdb"
                pdb_file.write_text(pdb_content)
                print(f"✓ Wrote input PDB: {pdb_file} ({len(pdb_content)} chars)")

            # Download weights if not present
            weights_dir = Path("/ProteinMPNN/vanilla_model_weights")
            weights_file = weights_dir / "v_48_020.pt"

            if not weights_file.exists() or weights_file.stat().st_size == 0:
                print(f"Downloading ProteinMPNN weights to {weights_file}...")
                weights_dir.mkdir(parents=True, exist_ok=True)

                # Try downloading from GitHub releases or alternative sources
                import urllib.request

                # Try the GitHub release URL
                weight_url = "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt"
                try:
                    urllib.request.urlretrieve(weight_url, str(weights_file))
                    print(f"✓ Downloaded weights: {weights_file}")
                except Exception as e:
                    return {
                        "success": False,
                        "exit_code": 1,
                        "error": f"Failed to download ProteinMPNN weights: {e}",
                        "sequences": [],
                        "output_files": [],
                    }

            # Build ProteinMPNN command
            cmd_parts = [
                "python",
                "/ProteinMPNN/protein_mpnn_run.py",
                "--pdb_path",
                str(pdb_file),
                "--out_folder",
                str(out_path),
                "--num_seq_per_target",
                str(num_sequences),
                "--sampling_temp",
                str(temperature),
                "--seed",
                "37",
                "--batch_size",
                "1",
                "--path_to_model_weights",
                str(weights_dir),
            ]

            # Add design_only_chain if specified using --chain_id_jsonl
            if design_only_chain and design_only_chain.strip():
                import json

                chain_jsonl = out_path / "chain_id.jsonl"
                # Key must be basename without extension (e.g., "input" not "input.pdb")
                pdb_basename = pdb_file.stem
                chain_dict = {pdb_basename: design_only_chain.strip()}
                with open(chain_jsonl, "w") as f:
                    json.dump(chain_dict, f)

                print(f"✓ Chain ID JSONL: {chain_dict}")
                cmd_parts.extend(["--chain_id_jsonl", str(chain_jsonl)])
                print(f"✓ Designing only chain: {design_only_chain} (other chains fixed)")

            # Add fixed positions if specified
            if fixed_positions and fixed_positions.strip():
                # Parse comma-separated string to dict by chain
                positions_by_chain = parse_fixed_positions(fixed_positions)
                if positions_by_chain:
                    # Create JSONL file for ProteinMPNN
                    fixed_jsonl = out_path / "fixed_positions.jsonl"
                    create_fixed_positions_jsonl(positions_by_chain, fixed_jsonl)
                    # Add to command
                    cmd_parts.extend(["--fixed_positions_jsonl", str(fixed_jsonl)])
                    print(f"✓ Added fixed positions: {fixed_positions}")
                else:
                    print(f"Warning: Could not parse fixed_positions '{fixed_positions}', skipping")

            cmd = " ".join(cmd_parts)
            print(f"Running: {cmd}")

            # Run ProteinMPNN
            result = run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"✗ ProteinMPNN failed with exit code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {
                    "success": False,
                    "exit_code": 1,
                    "error": f"ProteinMPNN exited with code {result.returncode}: {result.stderr}",
                    "sequences": [],
                    "output_files": [],
                }

            print("✓ ProteinMPNN design complete")

            # Parse output FASTA files
            sequences = []
            output_files = []

            # ProteinMPNN outputs go to seqs/ subdirectory
            seqs_dir = out_path / "seqs"
            if not seqs_dir.exists():
                print(f"✗ No sequences directory found: {seqs_dir}")
                return {
                    "success": False,
                    "exit_code": 1,
                    "error": f"No output sequences found in {seqs_dir}",
                    "sequences": [],
                    "output_files": [],
                }

            # Find FASTA files
            fasta_files = list(seqs_dir.glob("*.fa"))
            print(f"Found {len(fasta_files)} FASTA files in {seqs_dir}")
            if not fasta_files:
                # Also check for .fasta extension
                fasta_files = list(seqs_dir.glob("*.fasta"))
                print(f"Found {len(fasta_files)} .fasta files")
            if not fasta_files:
                # List all files in seqs_dir for debugging
                all_files = list(seqs_dir.glob("*"))
                print(f"All files in {seqs_dir}: {[f.name for f in all_files]}")

            for fasta_file in sorted(fasta_files):
                # Read sequences from FASTA
                with open(fasta_file) as f:
                    content = f.read()

                    # Parse FASTA format with score extraction
                    lines = content.strip().split("\n")
                    for i in range(0, len(lines), 2):
                        if i + 1 < len(lines):
                            header = lines[i].lstrip(">")
                            sequence = lines[i + 1]

                            # Skip the input sequence (ProteinMPNN outputs the original backbone as "input")
                            # This fixes the off-by-one bug where asking for N sequences produces N+1
                            if header.startswith("input"):
                                print(f"  Skipping input sequence: {header[:50]}...")
                                continue

                            # CRITICAL FIX: Sanitize header and extract ONLY the binder sequence
                            # RFDiffusion3 outputs binder/target complexes where the sequence is concatenated
                            # Format: "DESIGNED_BINDER_SEQ/ORIGINAL_TARGET_SEQ"
                            # We must extract ONLY the binder portion for ESMFold

                            # 1. Sanitize header (remove '/' characters)
                            if "/" in header:
                                header = header.replace("/", "_")
                                print(f"  Sanitized header (removed '/'): {header[:50]}...")

                            # 2. CRITICAL: Extract only the binder chain (before '/')
                            # RFDiffusion3 outputs binder/target, but ESMFold needs just the binder
                            if "/" in sequence:
                                binder_seq, target_seq = sequence.split("/", 1)
                                print(
                                    f"  ⚠️  SPLIT sequence: binder={len(binder_seq)}aa, target={len(target_seq)}aa"
                                )
                                print("      Keeping ONLY binder sequence (before '/')")
                                sequence = binder_seq  # Use only the designed binder

                            # 3. Validate the sequence (no special characters)
                            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                            invalid_chars = set(sequence) - valid_aa
                            if invalid_chars:
                                print(
                                    f"  ❌ ERROR: Invalid characters in sequence: {invalid_chars}"
                                )
                                print("      Skipping this sequence")
                                continue

                            # Extract scores from header (e.g., "T=0.1, sample=1, score=0.8567, global_score=0.8677, seq_recovery=0.3969")
                            score = None
                            global_score = None
                            seq_recovery = None

                            import re

                            if "score=" in header:
                                score_match = re.search(r"score=([\d.]+)", header)
                                if score_match:
                                    score = float(score_match.group(1))

                            if "global_score=" in header:
                                global_match = re.search(r"global_score=([\d.]+)", header)
                                if global_match:
                                    global_score = float(global_match.group(1))

                            if "seq_recovery=" in header:
                                recovery_match = re.search(r"seq_recovery=([\d.]+)", header)
                                if recovery_match:
                                    seq_recovery = float(recovery_match.group(1))

                            sequences.append(
                                {
                                    "name": header,
                                    "sequence": sequence,
                                    "score": score,
                                    "global_score": global_score,
                                    "seq_recovery": seq_recovery,
                                    "length": len(sequence),
                                }
                            )

                # Add to output files
                file_size = fasta_file.stat().st_size
                # Derive a per-design stem so parallel map items don't collide in GCS
                import pathlib as _pathlib

                if pdb_gcs_uri:
                    design_stem = _pathlib.Path(pdb_gcs_uri.rstrip("/").split("?")[0]).stem
                elif run_id:
                    design_stem = run_id[:8]
                else:
                    design_stem = "design"

                output_filename = f"{design_stem}_sequences.fa"
                output_files.append(
                    {
                        "path": str(fasta_file.absolute()),
                        "artifact_type": "fasta",
                        "filename": output_filename,  # Use descriptive name instead of input.fa
                        "size": file_size,
                        "metadata": {
                            "num_sequences": len(sequences),
                        },
                    }
                )

                # Create individual FASTA files for each sequence
                print(f"  Creating {len(sequences)} individual FASTA files...")
                individual_files = []
                for idx, seq_data in enumerate(sequences):
                    # Include design_stem so parallel map items don't overwrite each other in GCS
                    individual_filename = f"{design_stem}_seq_{idx}.fa"
                    individual_path = out_path / individual_filename

                    # Write single-sequence FASTA
                    with open(individual_path, "w") as f:
                        f.write(f">{seq_data['name']}\n{seq_data['sequence']}\n")

                    individual_files.append(
                        {
                            "path": str(individual_path.absolute()),
                            "artifact_type": "fasta_individual",
                            "filename": individual_filename,
                            "size": individual_path.stat().st_size,
                            "metadata": {
                                "sequence_index": idx,
                                "sequence_id": seq_data["name"],
                                "sequence_length": seq_data["length"],
                            },
                        }
                    )
                    print(f"    ✓ Created {individual_filename} ({seq_data['length']} residues)")

                output_files.extend(individual_files)
                print(f"  ✓ Created {len(individual_files)} individual FASTA files")

            print(f"  Parsed {len(sequences)} sequences")
            print(f"  Total output_files before GCS upload: {len(output_files)}")

            # Upload to GCS if requested
            if upload_to_gcs and gcs_bucket:
                print(f"\n📤 Uploading {len(output_files)} files to GCS bucket: {gcs_bucket}")

                import os

                credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                print("  Checking for GCS credentials...")
                if not credentials_json:
                    print("❌ GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets")
                    print("   Files will be returned with local paths only (not uploaded to GCS)")
                    # Don't fail - just return files without GCS URLs
                    return {
                        "success": True,
                        "exit_code": 0,
                        "run_id": run_id,
                        "num_sequences": len(sequences),
                        "num_output_files": len(output_files),
                        "temperature": temperature,
                        "fixed_positions": fixed_positions,
                        "sequences": sequences,
                        "output_files": output_files,  # Local paths only
                        "message": f"Generated {len(sequences)} sequences. GCS upload skipped (credentials not available).",
                    }

                # Write credentials to temp file (if not already done during download)
                creds_path = "/tmp/gcs_credentials.json"
                if (
                    not os.path.exists(creds_path)
                    or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") != creds_path
                ):
                    with open(creds_path, "w") as f:
                        f.write(credentials_json)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                    print("  ✓ Loaded GCS credentials from secret")

                from google.cloud import storage

                client = storage.Client()
                bucket = client.bucket(gcs_bucket)

                for file_info in output_files:
                    # Use tool-specific subdirectory for consistency with other tools (e.g., esmfold/)
                    gcs_path = f"runs/{run_id}/proteinmpnn/{file_info['filename']}"
                    print(
                        f"  Uploading {file_info['filename']} ({file_info.get('size', 0)} bytes) to gs://{gcs_bucket}/{gcs_path}"
                    )
                    blob = bucket.blob(gcs_path)
                    blob.upload_from_filename(file_info["path"])
                    file_info["gcs_url"] = f"gs://{gcs_bucket}/{gcs_path}"
                    print(f"  ✓ Uploaded: {file_info['filename']} -> {file_info['gcs_url']}")

            print("\n📊 Summary:")
            print(f"  Total sequences: {len(sequences)}")
            print(f"  Total files: {len(output_files)}")
            if sequences:
                lengths = [int(str(s.get("length", 0))) for s in sequences if s.get("length")]
                avg_len = sum(lengths) / len(lengths) if lengths else 0
                print(f"  Average length: {avg_len:.1f} residues")

            # Build individual_sequence_files with scores
            individual_sequence_files = []
            for file_info in output_files:
                if file_info.get("artifact_type") == "fasta_individual":
                    metadata = file_info.get("metadata", {})
                    if not isinstance(metadata, dict):
                        continue
                    seq_idx = metadata.get("sequence_index", 0)
                    seq_data = sequences[seq_idx] if seq_idx < len(sequences) else {}

                    individual_sequence_files.append(
                        {
                            "sequence_id": metadata.get("sequence_id", ""),
                            "sequence_index": seq_idx,
                            "filename": file_info["filename"],
                            "gcs_uri": file_info.get(
                                "gcs_url", file_info.get("gcs_uri")
                            ),  # Set after GCS upload
                            "sequence": seq_data.get("sequence", ""),
                            "length": seq_data.get("length", 0),
                            # ProteinMPNN scores (extracted from FASTA header if available)
                            "score": seq_data.get("score"),
                            "global_score": seq_data.get("global_score"),
                            "seq_recovery": seq_data.get("seq_recovery"),
                        }
                    )

            return {
                "success": True,
                "exit_code": 0,
                "run_id": run_id,
                "num_sequences": len(sequences),
                "num_output_files": len(output_files),
                "temperature": temperature,
                "fixed_positions": fixed_positions,
                "sequences": sequences,
                "output_files": output_files,
                "individual_sequence_files": individual_sequence_files,
            }

    except Exception as e:
        print(f"✗ ProteinMPNN design failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "exit_code": 1,
            "error": str(e),
            "sequences": [],
            "output_files": [],  # Always include output_files
        }


@app.local_entrypoint()
def main(
    pdb_path: str,
    num_sequences: int = 10,
    temperature: float = 0.1,
    fixed_positions: str | None = None,
    upload_to_gcs: bool = False,
    gcs_bucket: str = "dev-services",
) -> None:
    """CLI entrypoint for ProteinMPNN sequence design.

    Args:
        pdb_path: Path to input PDB file
        num_sequences: Number of sequences to design
        temperature: Sampling temperature
        fixed_positions: Positions to keep fixed (e.g., "A10-20,A45-50")
        upload_to_gcs: Whether to upload to GCS
        gcs_bucket: GCS bucket name
    """
    # Read PDB file
    pdb_file = Path(pdb_path)
    if not pdb_file.exists():
        print(f"❌ PDB file not found: {pdb_path}")
        return

    pdb_content = pdb_file.read_text()

    # Run design
    result = proteinmpnn_design.remote(
        pdb_content=pdb_content,
        num_sequences=num_sequences,
        temperature=temperature,
        fixed_positions=fixed_positions,
        upload_to_gcs=upload_to_gcs,
        gcs_bucket=gcs_bucket if upload_to_gcs else None,
    )

    if result["success"]:
        print("\n✅ ProteinMPNN design completed successfully!")
        print(f"   Sequences generated: {result['num_sequences']}")
        print(f"   Output files: {result['num_output_files']}")

        # Print first few sequences
        print("\n📝 Designed sequences:")
        for seq in result["sequences"][:3]:
            print(
                f"   {seq['name']}: {seq['sequence'][:50]}{'...' if len(seq['sequence']) > 50 else ''}"
            )

        if len(result["sequences"]) > 3:
            print(f"   ... and {len(result['sequences']) - 3} more")
    else:
        print(f"\n❌ ProteinMPNN design failed: {result.get('error', 'Unknown error')}")
