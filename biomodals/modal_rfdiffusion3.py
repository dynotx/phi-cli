"""RFDiffusion3 (RFD3) - All-atom generative protein design
Baker Lab Foundry: https://github.com/RosettaCommons/Foundry

RFDiffusion3 is an all-atom generative model capable of designing protein
structures under complex constraints including:
- De novo backbone generation
- Binder design with hotspot specification
- Motif scaffolding
- Symmetric oligomers

Example usage:
```bash
# De novo generation
modal run modal_rfdiffusion3.py --length 100 --num-designs 10

# Binder design
modal run modal_rfdiffusion3.py --target-pdb target.pdb --target-chain A \
  --hotspots A45,A67,A89 --length 70 --num-designs 50

# Motif scaffolding
modal run modal_rfdiffusion3.py --motif-pdb motif.pdb --motif-residues 10-20,45-55 \
  --length 150 --num-designs 25

# Symmetric oligomer
modal run modal_rfdiffusion3.py --length 80 --symmetry C3 --num-designs 20
```

QC Thresholds (follow up with validation):
- pLDDT > 85 (backbone confidence)
- pAE < 10 Å (domain arrangement)
- Follow with ProteinMPNN for sequence design
"""

import os
from pathlib import Path

import modal
from modal import App, Image

GPU = os.environ.get("GPU", "H100")  # H100 by default - medium proteins already use 89% of A100
TIMEOUT = int(os.environ.get("TIMEOUT", 180))  # 3 hours default

# GPU memory capacities (approximate usable VRAM after system overhead)
GPU_MEMORY_GB = {
    "A10G": 22,  # 24GB total, ~22GB usable
    "A100": 38,  # 40GB total, ~38GB usable
    "H100": 78,  # 80GB total, ~78GB usable
}


def download_rfd3_checkpoints():
    """Download RFD3 checkpoints during image build."""
    import subprocess

    print("Downloading RFD3 checkpoints...")
    subprocess.run(
        ["foundry", "install", "rfd3", "--checkpoint-dir", "/root/.foundry/checkpoints"],
        check=True,
    )
    print("RFD3 checkpoint download complete")


image = (
    Image.debian_slim(python_version="3.12")  # Foundry requires Python 3.12+
    .apt_install("git", "wget", "build-essential")
    .pip_install("rc-foundry[all]")  # Installs RFD3, AtomWorks, dependencies
    .pip_install("google-cloud-storage==2.14.0")  # For GCS upload
    .run_function(
        download_rfd3_checkpoints,
        gpu="a10g",  # Need GPU for checkpoint download
    )
)

app = App(
    "rfdiffusion3",
    image=image,
)


def clean_pdb_protein_only(pdb_content: str) -> str:
    """Remove non-protein atoms (ligands, ions, water) from PDB content.

    This is critical for RFDiffusion3 which expects clean protein-only PDB files.
    Many PDB files contain HETATM records (ligands, ions, water) that cause
    atomworks.io to create multi-character chain IDs (AA, AB, etc.) which break
    the contig parser.

    Args:
        pdb_content: Raw PDB file content

    Returns:
        Cleaned PDB content with only protein ATOM records
    """
    import tempfile
    from pathlib import Path

    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    # Write to temp file, load, filter, write back
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(pdb_content)
        temp_path = Path(f.name)

    try:
        # Load structure
        pdb_file = PDBFile.read(temp_path)
        structure = pdb_file.get_structure()[0]

        # Filter to protein atoms only (removes HETATM, water, ligands, ions)
        protein_mask = struc.filter_amino_acids(structure)
        protein_structure = structure[protein_mask]

        # CRITICAL: Also filter out HETATM records (hetero flag)
        # filter_amino_acids() only checks residue names (GLU, ALA, etc.)
        # but HETATM records with amino acid names will pass!
        # Example: HETATM GLU A1301 looks like protein but is actually a ligand/artifact
        non_hetero_mask = ~protein_structure.hetero
        protein_structure = protein_structure[non_hetero_mask]

        # Write cleaned structure
        clean_pdb_file = PDBFile()
        clean_pdb_file.set_structure(protein_structure)
        clean_pdb_file.write(temp_path)

        # Read cleaned content
        return temp_path.read_text()
    finally:
        # Cleanup temp file
        temp_path.unlink(missing_ok=True)


@app.function(timeout=10 * 60, gpu=None)  # Simple diagnostics, no GPU needed
def foundry_diagnostics():
    """Diagnostic function to explore Foundry package structure."""
    import subprocess
    import sys

    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}\n")

    # Check package installation location
    result = subprocess.run(["pip", "show", "rc-foundry"], capture_output=True, text=True)
    print("rc-foundry package info:")
    print(result.stdout)
    print()

    # Import foundry
    try:
        import foundry

        print("✓ Successfully imported foundry")
        print(f"  Location: {getattr(foundry, '__file__', 'unknown')}\n")

        # List top-level items
        print("Top-level foundry items:")
        for item in sorted(dir(foundry)):
            if not item.startswith("_"):
                obj = getattr(foundry, item, None)
                print(f"  - {item}: {type(obj).__name__}")
        print()

        # Try to find models
        import_attempts = [
            "foundry.models",
            "foundry.models.rfd3",
            "foundry.rfd3",
            "foundry.RFD3",
        ]

        for import_path in import_attempts:
            try:
                parts = import_path.split(".")
                mod = __import__(parts[0])
                for part in parts[1:]:
                    mod = getattr(mod, part)

                print(f"✓ Successfully imported {import_path}")
                print(f"  Type: {type(mod).__name__}")
                if hasattr(mod, "__file__"):
                    print(f"  Location: {mod.__file__}")
                print("  Available items:")
                for item in sorted(dir(mod)):
                    if not item.startswith("_"):
                        obj = getattr(mod, item, None)
                        print(f"    - {item}: {type(obj).__name__}")
                print()

            except (ImportError, AttributeError) as e:
                print(f"✗ Failed to import {import_path}: {e}")
                continue

        # Search for RFD3, RF3, MPNN packages
        print("\nSearching for model packages...")
        import subprocess

        # Check for packages containing rfd, rf3, or mpnn
        result = subprocess.run(["pip", "list"], capture_output=True, text=True)
        print("Packages with 'rf' or 'mpnn':")
        for line in result.stdout.split("\n"):
            if any(x in line.lower() for x in ["rf", "mpnn", "protein"]):
                print(f"  {line}")
        print()

        # Search site-packages for rfd3/rf3/mpnn directories
        result = subprocess.run(
            [
                "find",
                "/usr/local/lib/python3.12/site-packages",
                "-maxdepth",
                "1",
                "-type",
                "d",
                "-iname",
                "*rf*",
            ],
            capture_output=True,
            text=True,
        )
        print("Directories with 'rf' in site-packages:")
        for line in result.stdout.strip().split("\n"):
            if line:
                print(f"  {line}")
        print()

        # Try importing different variants
        model_attempts = [
            "rfd3",
            "rfdiffusion3",
            "rfdiffusion",
            "rf3",
            "mpnn",
        ]

        for mod_name in model_attempts:
            try:
                mod = __import__(mod_name)
                print(f"✓ Successfully imported {mod_name}")
                print(f"  Location: {getattr(mod, '__file__', 'unknown')}")
                print(f"  Available: {[x for x in dir(mod) if not x.startswith('_')][:10]}")
                print()
            except ImportError as e:
                print(f"✗ Cannot import {mod_name}: {e}")

        # Deep dive into rfd3
        print("\n" + "=" * 80)
        print("DEEP DIVE: rfd3 package")
        print("=" * 80)
        try:
            import rfd3

            print("✓ Imported rfd3")
            print("\nAll attributes:")
            for item in sorted(dir(rfd3)):
                if not item.startswith("_"):
                    obj = getattr(rfd3, item, None)
                    print(f"  {item}: {type(obj).__name__} = {repr(obj)[:100]}")

            # List all Python files in rfd3
            print("\nrfd3 package structure:")
            result = subprocess.run(
                [
                    "find",
                    "/usr/local/lib/python3.12/site-packages/rfd3",
                    "-name",
                    "*.py",
                    "-type",
                    "f",
                ],
                capture_output=True,
                text=True,
            )
            for line in sorted(result.stdout.strip().split("\n")):
                print(f"  {line}")

            # Try to find main inference classes/functions
            print("\nLooking for inference engines...")
            result = subprocess.run(
                [
                    "find",
                    "/usr/local/lib/python3.12/site-packages/rfd3",
                    "-name",
                    "*infer*",
                    "-o",
                    "-name",
                    "*engine*",
                    "-o",
                    "-name",
                    "*model*",
                ],
                capture_output=True,
                text=True,
            )
            for line in sorted(result.stdout.strip().split("\n")):
                if line:
                    print(f"  {line}")

            # Check foundry CLI
            print("\n" + "=" * 80)
            print("Foundry CLI commands")
            print("=" * 80)
            result = subprocess.run(["foundry", "--help"], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            print("\n" + "=" * 80)
            print("RFD3 CLI/Engine exploration")
            print("=" * 80)

            # Try importing the engine
            try:
                from rfd3 import engine

                print("✓ Imported rfd3.engine")
                print(f"  Available: {[x for x in dir(engine) if not x.startswith('_')]}")
            except Exception as e:
                print(f"✗ Cannot import rfd3.engine: {e}")

            # Try importing run_inference
            try:
                from rfd3 import run_inference

                print("✓ Imported rfd3.run_inference")
                print(f"  Available: {[x for x in dir(run_inference) if not x.startswith('_')]}")
            except Exception as e:
                print(f"✗ Cannot import rfd3.run_inference: {e}")

            # Explore RFD3InferenceEngine
            print("\n" + "=" * 80)
            print("RFD3InferenceEngine signature")
            print("=" * 80)
            try:
                import inspect

                from rfd3.engine import RFD3InferenceEngine

                # Check __init__
                sig = inspect.signature(RFD3InferenceEngine.__init__)
                print("RFD3InferenceEngine.__init__ signature:")
                print(f"  {sig}")

                # Check run method
                if hasattr(RFD3InferenceEngine, "run"):
                    print("\nRFD3InferenceEngine.run signature:")
                    print(f"  {inspect.signature(RFD3InferenceEngine.run)}")

                # Check all public methods
                print("\nPublic methods:")
                for name in dir(RFD3InferenceEngine):
                    if not name.startswith("_"):
                        obj = getattr(RFD3InferenceEngine, name)
                        if callable(obj):
                            sig = inspect.signature(obj)
                            print(f"  - {name}{sig}")
            except Exception as e:
                print(f"✗ Error exploring RFD3InferenceEngine: {e}")
                import traceback

                traceback.print_exc()

            # Explore RFD3InferenceConfig
            print("\n" + "=" * 80)
            print("RFD3InferenceConfig signature")
            print("=" * 80)
            try:
                import inspect

                from rfd3.engine import RFD3InferenceConfig

                sig = inspect.signature(RFD3InferenceConfig.__init__)
                print("RFD3InferenceConfig.__init__ signature:")
                print(f"  {sig}")
                print("\nParameters:")
                for param_name, param in sig.parameters.items():
                    if param_name != "self":
                        print(
                            f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}"
                        )
                        if param.default != inspect.Parameter.empty:
                            print(f"      default={param.default}")

                # Check if it's a dataclass
                import dataclasses

                if dataclasses.is_dataclass(RFD3InferenceConfig):
                    print("\n✓ RFD3InferenceConfig is a dataclass")
                    print("Fields:")
                    for field in dataclasses.fields(RFD3InferenceConfig):
                        print(f"  - {field.name}: {field.type}")
                        if field.default != dataclasses.MISSING:
                            print(f"      default={field.default}")
                        elif field.default_factory != dataclasses.MISSING:
                            print(f"      default_factory={field.default_factory}")
            except Exception as e:
                print(f"✗ Error exploring RFD3InferenceConfig: {e}")
                import traceback

                traceback.print_exc()

        except Exception as e:
            print(f"✗ Error exploring rfd3: {e}")
            import traceback

            traceback.print_exc()

        return {"success": True, "module": "rfd3"}

    except ImportError as e:
        print(f"✗ Failed to import foundry: {e}")
        return {"success": False, "error": str(e)}

    # If all imports failed, list site-packages
    print("\nListing site-packages for 'foundry' or 'rc':")  # type: ignore[unreachable]
    result = subprocess.run(
        [
            "find",
            "/usr/local/lib/python3.12/site-packages",
            "-maxdepth",
            "2",
            "-name",
            "*foundry*",
            "-o",
            "-name",
            "rc_*",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    return {"success": False, "error": "Could not import any variant"}


# Main function with configurable GPU (deployment time)
@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def rfdiffusion3_generate(
    # De novo generation
    length: int | None = None,
    # Binder design
    target_pdb: str | None = None,
    target_pdb_gcs_uri: str | None = None,
    target_chain: str | None = None,
    hotspots: list[str] | None = None,  # ["A45", "A67", "A89"]
    # Motif scaffolding
    motif_pdb: str | None = None,
    motif_pdb_gcs_uri: str | None = None,
    motif_residues: list[str] | None = None,
    # Generation params
    num_designs: int = 10,
    inference_steps: int = 50,
    # Constraints
    contigs: str | None = None,  # "A1-150/0 70-100"
    symmetry: str | None = None,  # "C3", "D2", etc.
    # Output
    upload_to_gcs: bool = False,
    gcs_bucket: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Generate protein structures using RFDiffusion3.

    Args:
        length: Backbone length for de novo generation
        target_pdb: Target PDB content for binder design (DEPRECATED - use target_pdb_gcs_uri)
        target_pdb_gcs_uri: GCS URI to target PDB file (e.g., gs://bucket/path/target.pdb). Preferred method.
        target_chain: Target chain ID for binder design
        hotspots: List of hotspot residues (e.g., ["A45", "A67"])
        motif_pdb: Motif PDB content for scaffolding (DEPRECATED - use motif_pdb_gcs_uri)
        motif_pdb_gcs_uri: GCS URI to motif PDB file (e.g., gs://bucket/path/motif.pdb). Preferred method.
        motif_residues: List of motif residue ranges (e.g., ["10-20", "45-55"])
        num_designs: Number of designs to generate
        inference_steps: Number of diffusion inference steps
        contigs: Contig specification string
        symmetry: Symmetry specification (C3, D2, etc.)
        upload_to_gcs: Whether to upload outputs to GCS
        gcs_bucket: GCS bucket name
        run_id: Unique run identifier

    Returns:
        Dict with output_files, metadata, and optional GCS info
    """
    import tempfile
    from tempfile import TemporaryDirectory

    # Handle GCS URI inputs (preferred method)
    if target_pdb_gcs_uri:
        print(f"Downloading target PDB from GCS: {target_pdb_gcs_uri}")

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
        if not target_pdb_gcs_uri.startswith("gs://"):
            return {
                "success": False,
                "error": f"Invalid GCS URI: {target_pdb_gcs_uri}",
                "run_id": run_id,
            }

        gcs_path = target_pdb_gcs_uri.replace("gs://", "")
        bucket_name, blob_path = gcs_path.split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download target PDB content
        target_pdb = blob.download_as_text()
        print(f"  ✓ Downloaded target PDB ({len(target_pdb)} bytes)")

    if motif_pdb_gcs_uri:
        print(f"Downloading motif PDB from GCS: {motif_pdb_gcs_uri}")

        from google.cloud import storage

        # Initialize GCS client if not already done
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
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
        if not motif_pdb_gcs_uri.startswith("gs://"):
            return {
                "success": False,
                "error": f"Invalid GCS URI: {motif_pdb_gcs_uri}",
                "run_id": run_id,
            }

        gcs_path = motif_pdb_gcs_uri.replace("gs://", "")
        bucket_name, blob_path = gcs_path.split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download motif PDB content
        motif_pdb = blob.download_as_text()
        print(f"  ✓ Downloaded motif PDB ({len(motif_pdb)} bytes)")

    # Determine design type
    # Check both direct content and GCS URI parameters
    if target_pdb or target_pdb_gcs_uri:
        design_type = "binder"
        # Verify target_pdb is set (should be set by download above)
        if not target_pdb:
            return {
                "success": False,
                "error": "Binder design requires target PDB but target_pdb is not set",
                "run_id": run_id,
            }
    elif motif_pdb or motif_pdb_gcs_uri:
        design_type = "motif_scaffold"
        # Verify motif_pdb is set (should be set by download above)
        if not motif_pdb:
            return {
                "success": False,
                "error": "Motif scaffolding requires motif PDB but motif_pdb is not set",
                "run_id": run_id,
            }
    else:
        design_type = "de_novo"

    print(f"RFDiffusion3 {design_type} generation:")
    print(f"  Designs: {num_designs}")
    print(f"  Inference steps: {inference_steps}")
    if length:
        print(f"  Length: {length}")
    if target_chain:
        print(f"  Target chain: {target_chain}")
    if hotspots:
        print(f"  Hotspots: {hotspots}")
    if symmetry:
        print(f"  Symmetry: {symmetry}")

    # ========================================
    # Set PyTorch memory management environment variables
    # ========================================
    import os as _os

    # Reduce fragmentation for large allocations (helps with OOM errors)
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
    # Enable cudnn benchmarking for faster convolutions
    _os.environ["CUDNN_BENCHMARK"] = "1"

    # Import RFD3 API at function scope
    try:
        import json
        from pathlib import Path

        import torch
        from atomworks.io.utils.io_utils import to_cif_file
        from rfd3.engine import DesignInputSpecification, RFD3InferenceConfig, RFD3InferenceEngine

        print("✓ Successfully imported RFD3 inference engine")

        # Enable memory efficient settings
        if torch.cuda.is_available():
            # Clear any cached memory from previous runs
            torch.cuda.empty_cache()
            # Enable TF32 for faster computation on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    except ImportError as e:
        raise RuntimeError(f"Failed to import RFD3: {e}")

    with TemporaryDirectory() as work_dir, TemporaryDirectory() as out_dir:
        work_path = Path(work_dir)
        out_path = Path(out_dir)

        # Write input files if provided
        target_file = None
        if target_pdb:
            target_file = work_path / "target.pdb"
            # Clean PDB to remove heteroatoms (ligands, ions, water) that break RFDiffusion3
            cleaned_target_pdb = clean_pdb_protein_only(target_pdb)
            target_file.write_text(cleaned_target_pdb)
            print(f"Wrote cleaned target PDB: {target_file}")

        motif_file = None
        if motif_pdb:
            motif_file = work_path / "motif.pdb"
            # Clean PDB to remove heteroatoms
            cleaned_motif_pdb = clean_pdb_protein_only(motif_pdb)
            motif_file.write_text(cleaned_motif_pdb)
            print(f"Wrote cleaned motif PDB: {motif_file}")

        # Set up checkpoint path
        # ckpt_path can be either a path to a .ckpt file or a registry key like "rfd3"
        # The default "rfd3" looks in the foundry checkpoint registry
        checkpoint_dir = Path("/root/.foundry/checkpoints")
        checkpoint_file = checkpoint_dir / "rfd3_latest.ckpt"

        if checkpoint_file.exists():
            ckpt_path = str(checkpoint_file)
            print(f"Using checkpoint file: {ckpt_path}")
        else:
            # Fall back to registry key
            ckpt_path = "rfd3"
            print(f"Using checkpoint registry key: {ckpt_path}")

        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == "cpu":
            print("⚠️  Warning: Running on CPU, this will be slow!")

        # Create inference configuration
        try:
            # Prepare design input specification
            if design_type == "de_novo":
                if not length:
                    raise ValueError("De novo generation requires --length parameter")

                print(f"Generating {num_designs} de novo backbones of length {length}...")
                print(f"Inference steps: {inference_steps}")

                # For de novo generation, specification is a simple dict
                specification = {
                    "length": length,
                }

                # Add extra specifications if provided (symmetry, contigs, etc.)
                if contigs or symmetry:
                    specification["extra"] = {}  # type: ignore[assignment]
                    if contigs:
                        specification["extra"]["contigs"] = contigs  # type: ignore[index]
                    if symmetry:
                        specification["extra"]["symmetry"] = symmetry  # type: ignore[index]

                print(f"Specification: {json.dumps(specification, indent=2)}")

                # Configure diffusion sampler (number of inference steps)
                sampler_config: dict[str, int | float] = {
                    "num_steps": inference_steps,
                }

                # Calculate n_batches: each batch generates diffusion_batch_size structures
                # To get num_designs total, we need: n_batches = ceil(num_designs / diffusion_batch_size)
                batch_size = min(num_designs, 4)  # Limit batch size for memory
                n_batches = (num_designs + batch_size - 1) // batch_size

                print(
                    f"Batch configuration: {batch_size} designs/batch × {n_batches} batches = {batch_size * n_batches} total designs"
                )

                # Create inference config
                config = RFD3InferenceConfig(
                    ckpt_path=ckpt_path,
                    specification=specification,
                    inference_sampler=sampler_config,
                    diffusion_batch_size=batch_size,
                    verbose=True,
                )

                # Initialize engine and run generation
                print("Initializing RFD3 inference engine...")
                engine = RFD3InferenceEngine(**config.__dict__)

                print("Running RFD3 inference...")
                # IMPORTANT: out_dir=None returns structures in memory
                # If out_dir is set to a path, RFD3 writes files but returns EMPTY dict!
                outputs = engine.run(
                    inputs=None,  # None for unconditional de novo generation
                    out_dir=None,  # None = return structures in memory (don't write files)
                    n_batches=n_batches,  # Number of batches to run
                )

                print("✓ RFD3 inference complete")
                print(f"Generated {len(outputs)} batches with output keys: {list(outputs.keys())}")

                # Collect structures and save them manually
                # outputs is a dict: batch_idx -> list[RFD3Output]
                # Each RFD3Output has: .atom_array, .summary_confidences, .confidences
                output_files = []
                structure_count = 0

                for batch_idx, batch_outputs in outputs.items():
                    print(f"\nBatch {batch_idx}: {len(batch_outputs)} structures")

                    for _design_idx, rfd3_output in enumerate(batch_outputs):
                        structure_count += 1
                        design_name = f"design_{structure_count:03d}"

                        # Save structure as CIF file (AtomWorks provides to_cif_file, not to_pdb_file)
                        cif_path = out_path / f"{design_name}.cif"
                        to_cif_file(rfd3_output.atom_array, str(cif_path))
                        print(f"  ✓ Saved {design_name}.cif ({len(rfd3_output.atom_array)} atoms)")

                        # Get file size
                        file_size = cif_path.stat().st_size if cif_path.exists() else 0

                        output_files.append(
                            {
                                "path": str(cif_path.absolute()),
                                "artifact_type": "cif",  # Changed from "type" to "artifact_type"
                                "filename": f"{design_name}.cif",
                                "size": file_size,
                                "metadata": {
                                    "design_idx": structure_count,
                                    "num_atoms": len(rfd3_output.atom_array),
                                    "batch_idx": batch_idx,
                                },
                            }
                        )

                        # Save confidence scores as JSON
                        if (
                            hasattr(rfd3_output, "summary_confidences")
                            and rfd3_output.summary_confidences
                        ):
                            conf_path = out_path / f"{design_name}_confidence.json"
                            conf_path.write_text(
                                json.dumps(rfd3_output.summary_confidences, indent=2)
                            )
                            print("    - Saved confidence scores")

                            conf_size = conf_path.stat().st_size if conf_path.exists() else 0

                            output_files.append(
                                {
                                    "path": str(conf_path.absolute()),
                                    "artifact_type": "json",  # Changed from "type" to "artifact_type"
                                    "filename": f"{design_name}_confidence.json",
                                    "size": conf_size,
                                    "metadata": {
                                        "design_idx": structure_count,
                                        "is_confidence_scores": True,
                                    },
                                }
                            )

                # Upload to GCS if requested
                if upload_to_gcs and gcs_bucket:
                    print(f"\nUploading to GCS bucket: {gcs_bucket}")

                    # Initialize GCS client with credentials from secret
                    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                    if not credentials_json:
                        print("❌ GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets")
                        return {
                            "run_id": run_id,
                            "design_type": design_type,
                            "num_designs": structure_count,
                            "num_output_files": len(output_files),
                            "length": length,
                            "inference_steps": inference_steps,
                            "output_files": output_files,
                            "success": False,
                            "error": "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                        }

                    # Write credentials to temp file
                    creds_path = Path("/tmp/gcs_credentials.json")
                    creds_path.write_text(credentials_json)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

                    # Create GCS client
                    from google.cloud import storage

                    client = storage.Client()
                    bucket = client.bucket(gcs_bucket)

                    for file_info in output_files:
                        gcs_path = f"runs/{run_id}/rfdiffusion3/{file_info['filename']}"
                        blob = bucket.blob(gcs_path)
                        blob.upload_from_filename(file_info["path"])
                        file_info["gcs_url"] = f"gs://{gcs_bucket}/{gcs_path}"
                        print(f"  ✓ Uploaded: {file_info['filename']} -> {file_info['gcs_url']}")

                print("\n📊 Summary:")
                print(f"  Total structures: {structure_count}")
                print(f"  Total files: {len(output_files)}")
                print(
                    f"  CIF structure files: {sum(1 for f in output_files if f['artifact_type'] == 'cif')}"
                )
                print(
                    f"  JSON metadata files: {sum(1 for f in output_files if f['artifact_type'] == 'json')}"
                )

                return {
                    "run_id": run_id,
                    "design_type": design_type,
                    "num_designs": structure_count,
                    "num_output_files": len(output_files),
                    "length": length,
                    "inference_steps": inference_steps,
                    "output_files": output_files,
                    "success": True,
                }

            elif design_type == "binder" and target_pdb:
                # Binder design: generate a protein binder for a target structure
                if not length:
                    raise ValueError("Binder design requires --length parameter for binder size")

                print(f"Generating {num_designs} binders of length {length} for target...")
                print(f"Target chain: {target_chain or 'A'}")
                if hotspots:
                    print(f"Hotspots: {', '.join(hotspots)}")
                print(f"Inference steps: {inference_steps}")

                # Write target PDB to file (cleaned to remove heteroatoms)
                target_pdb_file = work_path / "target.pdb"
                cleaned_target_pdb = clean_pdb_protein_only(target_pdb)
                target_pdb_file.write_text(cleaned_target_pdb)

                # Load PDB to get actual residue range for the target chain
                import biotite.structure as struc
                from biotite.structure.io.pdb import PDBFile

                pdb_file = PDBFile.read(target_pdb_file)
                structure = pdb_file.get_structure()[0]

                chain_id = target_chain or "A"
                chain_mask = structure.chain_id == chain_id
                chain_structure = structure[chain_mask]

                # Filter to only amino acid residues (exclude water, ligands, etc.)
                protein_mask = struc.filter_amino_acids(chain_structure)
                protein_structure = chain_structure[protein_mask]

                # CRITICAL: Also filter out HETATM records (hetero flag)
                # filter_amino_acids() only checks residue names (GLU, ALA, etc.)
                # but HETATM GLU residues will pass that filter!
                # We must also exclude hetero atoms to avoid referencing non-polymer residues
                non_hetero_mask = ~protein_structure.hetero
                protein_structure = protein_structure[non_hetero_mask]

                # Get residue IDs and build contig that handles gaps
                residue_ids = sorted(set(protein_structure.res_id))
                min_res = int(residue_ids[0])
                max_res = int(residue_ids[-1])

                print(
                    f"Target: Chain {chain_id}, {len(residue_ids)} residues ({min_res}-{max_res})"
                )

                # Check target size for memory constraints based on GPU type
                # Memory estimates are rough - actual usage depends on many factors
                # Check actual GPU at runtime (important for H100 function calling this)
                import subprocess as _gpu_check

                try:
                    nvidia_smi = _gpu_check.check_output(
                        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
                    )
                    if "H100" in nvidia_smi:
                        gpu_type = "H100"
                    elif "A100" in nvidia_smi:
                        gpu_type = "A100"
                    elif "A10G" in nvidia_smi:
                        gpu_type = "A10G"
                    else:
                        gpu_type = GPU  # Fallback to environment variable
                except Exception:
                    gpu_type = GPU  # Fallback to environment variable

                gpu_capacity = GPU_MEMORY_GB.get(gpu_type, 38)  # Default to A100

                # Rough memory estimation (very approximate)
                # RFD3 memory usage is roughly: base_model + (residues * 50MB) + (binder_length * 30MB)
                estimated_memory_gb = 10 + (len(residue_ids) * 0.05) + (length * 0.03)

                print(
                    f"💾 Estimated GPU memory: ~{estimated_memory_gb:.1f} GB (GPU capacity: {gpu_capacity} GB)"
                )

                # Memory-based informational logging only - let it run and see if it OOMs
                if estimated_memory_gb > gpu_capacity * 0.8:
                    print(
                        f"⚠️  Memory usage may be high ({estimated_memory_gb:.1f}GB / {gpu_capacity}GB)"
                    )
                    print("   If OOM occurs, switch to H100 or use hotspots to reduce memory")
                else:
                    print(
                        f"✓ Memory estimate looks reasonable ({estimated_memory_gb:.1f}GB / {gpu_capacity}GB)"
                    )

                # Legacy check for no-hotspots case
                MAX_TARGET_RESIDUES_WITHOUT_HOTSPOTS = 300
                if len(residue_ids) > MAX_TARGET_RESIDUES_WITHOUT_HOTSPOTS and not hotspots:
                    print(
                        f"ℹ️  Large target ({len(residue_ids)} residues) without hotspots may use more memory.\n"
                        f"   Hotspots help RFD3 focus on the binding region and reduce memory usage."
                    )

                # Build contig for binder design
                # Format: "<binder_length>,/0,<chain><start>-<end>"
                # Example: "40-120,/0,E6-155" means binder of 40-120 residues, chain break, target E6-155
                if contigs:
                    contig_str = contigs
                else:
                    # Build a contig that includes continuous ranges, handling gaps
                    # Find continuous ranges in residue_ids
                    ranges = []
                    start = residue_ids[0]
                    prev = residue_ids[0]

                    for res_id in residue_ids[1:]:
                        if res_id != prev + 1:
                            # Gap found - close current range
                            if start == prev:
                                ranges.append(f"{chain_id}{start}")
                            else:
                                ranges.append(f"{chain_id}{start}-{prev}")
                            start = res_id
                        prev = res_id

                    # Add the last range
                    if start == prev:
                        ranges.append(f"{chain_id}{start}")
                    else:
                        ranges.append(f"{chain_id}{start}-{prev}")

                    # Build contig: binder + chain break + target ranges
                    target_spec = ",".join(ranges)
                    contig_str = f"{length},/0,{target_spec}"

                print(f"Contig: {contig_str}")

                # Build specification kwargs
                spec_kwargs = {
                    "input": str(target_pdb_file),
                    "contig": contig_str,
                    "infer_ori_strategy": "hotspots" if hotspots else "com",
                    "is_non_loopy": True,  # Recommended for binder design
                    "dialect": 2,  # Use new parser
                }

                # Add hotspots if specified
                # Format: {"E64": "CD2,CZ", "E88": "CG,CZ"}
                if hotspots:
                    # Convert hotspots from ["A45", "A50"] to {"A45": "ALL", "A50": "ALL"}
                    # or if user provides atom details: {"A45": "CD2,CZ"}
                    hotspot_dict = {}
                    for hs in hotspots:
                        if ":" in hs:
                            # Format: "A45:CD2,CZ"
                            res, atoms = hs.split(":", 1)
                            hotspot_dict[res] = atoms
                        else:
                            # Format: "A45" - select all heavy atoms
                            hotspot_dict[hs] = "ALL"

                    spec_kwargs["select_hotspots"] = hotspot_dict  # type: ignore[assignment]
                    print(f"Hotspot specification: {hotspot_dict}")

                print(f"Creating specification dict with: {list(spec_kwargs.keys())}")
                # Don't pre-validate the specification - let RFD3 handle it
                # This avoids the "Both 'input' and 'atom_array_input' provided" error
                specification_dict = spec_kwargs

                # Configure diffusion sampler with recommended binder design settings
                sampler_config = {
                    "num_steps": inference_steps,
                    "step_scale": 3.0,  # Higher than default (1.5) for better designability
                    "gamma_0": 0.2,  # Lower than default (0.6) for lower-temperature designs
                }

                print("Sampler config: step_scale=3.0, gamma_0=0.2 (optimized for binder design)")

                # Calculate batch configuration
                batch_size = min(num_designs, 4)
                n_batches = (num_designs + batch_size - 1) // batch_size

                print(
                    f"Batch configuration: {batch_size} designs/batch × {n_batches} batches = {batch_size * n_batches} total designs"
                )

                # Create inference config with spec dict instead of pre-validated object
                config = RFD3InferenceConfig(
                    ckpt_path=ckpt_path,
                    specification=specification_dict,
                    inference_sampler=sampler_config,
                    diffusion_batch_size=batch_size,
                    verbose=True,
                )

                # Initialize engine and run generation
                print("Initializing RFD3 inference engine...")
                engine = RFD3InferenceEngine(**config.__dict__)

                print("Running RFD3 binder design inference...")
                outputs = engine.run(
                    inputs=None,
                    out_dir=None,
                    n_batches=n_batches,
                )

                print("✓ RFD3 binder design complete")
                print(f"Generated {len(outputs)} batches with output keys: {list(outputs.keys())}")

                # Process outputs
                output_files = []
                structure_count = 0

                for batch_idx, batch_outputs in outputs.items():
                    print(f"\nBatch {batch_idx}: {len(batch_outputs)} structures")

                    for _design_idx, rfd3_output in enumerate(batch_outputs):
                        structure_count += 1
                        design_name = f"binder_{structure_count:03d}"

                        # Save structure as CIF file
                        cif_path = out_path / f"{design_name}.cif"
                        to_cif_file(rfd3_output.atom_array, str(cif_path))
                        print(f"  ✓ Saved {design_name}.cif ({len(rfd3_output.atom_array)} atoms)")

                        file_size = cif_path.stat().st_size if cif_path.exists() else 0

                        output_files.append(
                            {
                                "path": str(cif_path.absolute()),
                                "artifact_type": "cif",
                                "filename": f"{design_name}.cif",
                                "size": file_size,
                                "metadata": {
                                    "design_idx": structure_count,
                                    "num_atoms": len(rfd3_output.atom_array),
                                    "batch_idx": batch_idx,
                                    "target_chain": target_chain,
                                    "binder_length": length,
                                },
                            }
                        )

                        # Save confidence scores
                        if (
                            hasattr(rfd3_output, "summary_confidences")
                            and rfd3_output.summary_confidences
                        ):
                            conf_path = out_path / f"{design_name}_confidence.json"
                            conf_path.write_text(
                                json.dumps(rfd3_output.summary_confidences, indent=2)
                            )
                            print("    - Saved confidence scores")

                            conf_size = conf_path.stat().st_size if conf_path.exists() else 0

                            output_files.append(
                                {
                                    "path": str(conf_path.absolute()),
                                    "artifact_type": "json",
                                    "filename": f"{design_name}_confidence.json",
                                    "size": conf_size,
                                    "metadata": {
                                        "design_idx": structure_count,
                                        "is_confidence_scores": True,
                                    },
                                }
                            )

                # Upload to GCS if requested
                if upload_to_gcs and gcs_bucket:
                    print(f"\nUploading to GCS bucket: {gcs_bucket}")

                    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                    if not credentials_json:
                        print("❌ GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets")
                        return {
                            "run_id": run_id,
                            "design_type": design_type,
                            "num_designs": structure_count,
                            "num_output_files": len(output_files),
                            "length": length,
                            "inference_steps": inference_steps,
                            "output_files": output_files,
                            "success": False,
                            "error": "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                        }

                    # Write credentials to temp file
                    creds_path = Path("/tmp/gcs_credentials.json")
                    creds_path.write_text(credentials_json)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

                    from google.cloud import storage

                    client = storage.Client()
                    bucket = client.bucket(gcs_bucket)

                    for file_info in output_files:
                        gcs_path = f"runs/{run_id}/rfdiffusion3/{file_info['filename']}"
                        blob = bucket.blob(gcs_path)
                        blob.upload_from_filename(file_info["path"])
                        file_info["gcs_url"] = f"gs://{gcs_bucket}/{gcs_path}"
                        print(f"  ✓ Uploaded: {file_info['filename']} -> {file_info['gcs_url']}")

                print("\n📊 Summary:")
                print(f"  Total binders: {structure_count}")
                print(f"  Total files: {len(output_files)}")
                print(
                    f"  CIF structure files: {sum(1 for f in output_files if f['artifact_type'] == 'cif')}"
                )
                print(
                    f"  JSON metadata files: {sum(1 for f in output_files if f['artifact_type'] == 'json')}"
                )

                return {
                    "run_id": run_id,
                    "design_type": design_type,
                    "num_designs": structure_count,
                    "num_output_files": len(output_files),
                    "length": length,
                    "inference_steps": inference_steps,
                    "target_chain": target_chain,
                    "hotspots": hotspots,
                    "output_files": output_files,
                    "success": True,
                }

            elif False:
                # Binder design with DesignInputSpecification
                if not length:  # type: ignore[unreachable]
                    raise ValueError("Binder design requires --length parameter for binder size")

                print(f"Generating {num_designs} binders of length {length} for target...")
                print(f"Target chain: {target_chain or 'A'}")
                if hotspots:
                    print(f"Hotspots: {', '.join(hotspots)}")
                print(f"Inference steps: {inference_steps}")

                # Write target PDB to file so we can load it (cleaned to remove heteroatoms)
                target_pdb_file = work_path / "target.pdb"
                cleaned_target_pdb = clean_pdb_protein_only(target_pdb)
                target_pdb_file.write_text(cleaned_target_pdb)

                # For binder design, contigs must specify residues in format 'ChainIDResID'
                # Example: 'A19-184' means residues 19-184 from chain A of the input
                # We need to load the PDB to get the actual residue range

                import biotite.structure as struc
                from biotite.structure.io.pdb import PDBFile

                # Load PDB to get actual residue range
                pdb_file = PDBFile.read(target_pdb_file)
                structure = pdb_file.get_structure()[0]  # Get first model

                chain_id = target_chain or "A"
                chain_mask = structure.chain_id == chain_id
                chain_structure = structure[chain_mask]

                # Filter to only standard amino acids (not HETATM)
                protein_mask = struc.filter_amino_acids(chain_structure)
                chain_structure = chain_structure[protein_mask]

                # CRITICAL: Also filter out HETATM records
                non_hetero_mask = ~chain_structure.hetero
                chain_structure = chain_structure[non_hetero_mask]

                # Get residue IDs
                residue_ids = chain_structure.res_id
                min_res = int(residue_ids.min())
                max_res = int(residue_ids.max())
                num_residues = max_res - min_res + 1

                print(
                    f"Target PDB: Chain {chain_id}, residues {min_res}-{max_res} ({num_residues} total)"
                )

                # Check target size for memory constraints
                MAX_TARGET_RESIDUES_WITHOUT_HOTSPOTS = 300
                if num_residues > MAX_TARGET_RESIDUES_WITHOUT_HOTSPOTS and not hotspots:
                    raise ValueError(
                        f"Target protein is too large ({num_residues} residues) for binder design without hotspot specification. "
                        f"GPU memory is limited. Please either:\n"
                        f"  1. Specify hotspot residues to focus on the binding region (e.g., hotspot_residues='A45,A67,A89')\n"
                        f"  2. Provide a smaller target PDB containing only the binding domain (~{MAX_TARGET_RESIDUES_WITHOUT_HOTSPOTS} residues max)\n"
                        f"  3. Use a structure prediction tool to identify the binding site first\n"
                        f"For reference: RFDiffusion3 on A100-40GB can handle ~300 target residues without hotspots."
                    )

                # Build contig that references the actual residues from the input
                if contigs:
                    contig_str = contigs
                else:
                    # Use actual residue range from PDB
                    contig_str = f"{chain_id}{min_res}-{max_res}"

                print(f"Contig: {contig_str}, Binder length: {length}")

                # Create DesignInputSpecification
                # For now, try without contig - just input and length
                # The contig format is proving too complex with missing residues and hotspots
                spec_kwargs = {
                    "input": str(target_pdb_file),
                    "length": length,  # Binder length
                }

                # TODO: Add contig and hotspots support once we understand the correct format
                # For now, let's try the simplest approach
                if hotspots:
                    print(f"Warning: Hotspots not yet supported in binder design: {hotspots}")

                print(f"Creating DesignInputSpecification with: {list(spec_kwargs.keys())}")
                print("Note: Using simple input + length specification (no contig/hotspots yet)")
                specification = DesignInputSpecification(**spec_kwargs)

                # Configure diffusion sampler
                sampler_config = {
                    "num_steps": inference_steps,
                }

                # Calculate batch configuration
                batch_size = min(num_designs, 4)
                n_batches = (num_designs + batch_size - 1) // batch_size

                print(
                    f"Batch configuration: {batch_size} designs/batch × {n_batches} batches = {batch_size * n_batches} total designs"
                )

                # Create inference config
                config = RFD3InferenceConfig(
                    ckpt_path=ckpt_path,
                    specification=specification,
                    inference_sampler=sampler_config,
                    diffusion_batch_size=batch_size,
                    verbose=True,
                )

                # Initialize engine and run generation
                print("Initializing RFD3 inference engine...")
                engine = RFD3InferenceEngine(**config.__dict__)

                print("Running RFD3 binder design inference...")
                outputs = engine.run(
                    inputs=None,
                    out_dir=None,
                    n_batches=n_batches,
                )

                print("✓ RFD3 binder design complete")
                print(f"Generated {len(outputs)} batches with output keys: {list(outputs.keys())}")

                # Process outputs (same as de novo)
                output_files = []
                structure_count = 0

                for batch_idx, batch_outputs in outputs.items():
                    print(f"\nBatch {batch_idx}: {len(batch_outputs)} structures")

                    for _design_idx, rfd3_output in enumerate(batch_outputs):
                        structure_count += 1
                        design_name = f"binder_{structure_count:03d}"

                        # Save structure as CIF file
                        cif_path = out_path / f"{design_name}.cif"
                        to_cif_file(rfd3_output.atom_array, str(cif_path))
                        print(f"  ✓ Saved {design_name}.cif ({len(rfd3_output.atom_array)} atoms)")

                        file_size = cif_path.stat().st_size if cif_path.exists() else 0

                        output_files.append(
                            {
                                "path": str(cif_path.absolute()),
                                "artifact_type": "cif",
                                "filename": f"{design_name}.cif",
                                "size": file_size,
                                "metadata": {
                                    "design_idx": structure_count,
                                    "num_atoms": len(rfd3_output.atom_array),
                                    "batch_idx": batch_idx,
                                    "target_chain": target_chain,
                                    "binder_length": length,
                                },
                            }
                        )

                        # Save confidence scores
                        if (
                            hasattr(rfd3_output, "summary_confidences")
                            and rfd3_output.summary_confidences
                        ):
                            conf_path = out_path / f"{design_name}_confidence.json"
                            conf_path.write_text(
                                json.dumps(rfd3_output.summary_confidences, indent=2)
                            )
                            print("    - Saved confidence scores")

                            conf_size = conf_path.stat().st_size if conf_path.exists() else 0

                            output_files.append(
                                {
                                    "path": str(conf_path.absolute()),
                                    "artifact_type": "json",
                                    "filename": f"{design_name}_confidence.json",
                                    "size": conf_size,
                                    "metadata": {
                                        "design_idx": structure_count,
                                        "is_confidence_scores": True,
                                    },
                                }
                            )

                # Upload to GCS if requested
                if upload_to_gcs and gcs_bucket:
                    print(f"\nUploading to GCS bucket: {gcs_bucket}")

                    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                    if not credentials_json:
                        print("❌ GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets")
                        return {
                            "run_id": run_id,
                            "design_type": design_type,
                            "num_designs": structure_count,
                            "num_output_files": len(output_files),
                            "length": length,
                            "inference_steps": inference_steps,
                            "output_files": output_files,
                            "success": False,
                            "error": "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                        }

                    # Write credentials to temp file
                    creds_path = Path("/tmp/gcs_credentials.json")
                    creds_path.write_text(credentials_json)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

                    from google.cloud import storage

                    client = storage.Client()
                    bucket = client.bucket(gcs_bucket)

                    for file_info in output_files:
                        gcs_path = f"runs/{run_id}/rfdiffusion3/{file_info['filename']}"
                        blob = bucket.blob(gcs_path)
                        blob.upload_from_filename(file_info["path"])
                        file_info["gcs_url"] = f"gs://{gcs_bucket}/{gcs_path}"
                        print(f"  ✓ Uploaded: {file_info['filename']} -> {file_info['gcs_url']}")

                print("\n📊 Summary:")
                print(f"  Total binders: {structure_count}")
                print(f"  Total files: {len(output_files)}")
                print(
                    f"  CIF structure files: {sum(1 for f in output_files if f['artifact_type'] == 'cif')}"
                )
                print(
                    f"  JSON metadata files: {sum(1 for f in output_files if f['artifact_type'] == 'json')}"
                )

                # Extract structure GCS URLs for downstream processing (ProteinMPNN expects this)
                structures = [
                    f["gcs_url"]
                    for f in output_files
                    if f["artifact_type"] == "cif" and "gcs_url" in f
                ]

                return {
                    "run_id": run_id,
                    "design_type": design_type,
                    "num_designs": structure_count,
                    "num_output_files": len(output_files),
                    "length": length,
                    "inference_steps": inference_steps,
                    "target_chain": target_chain,
                    "hotspots": hotspots,
                    "output_files": output_files,
                    "structures": structures,  # List of GCS URLs for CIF files
                    "success": True,
                }

            elif design_type == "motif_scaffold":
                # Motif scaffolding: generate a protein scaffold around a structural motif
                if not length:
                    raise ValueError(
                        "Motif scaffolding requires --length parameter for scaffold size"
                    )

                # Verify motif_pdb is present (should be guaranteed by earlier check)
                assert motif_pdb is not None, "motif_pdb must be set for motif_scaffold design"

                print(f"Generating {num_designs} scaffolds with motif...")
                print(f"Scaffold length: {length}")
                if motif_residues:
                    print(f"Motif residues: {', '.join(motif_residues)}")
                print(f"Inference steps: {inference_steps}")

                # Write motif PDB to file
                motif_pdb_file = work_path / "motif.pdb"
                motif_pdb_file.write_text(motif_pdb)

                # Parse motif structure to determine which residues to keep
                pdb_content = motif_pdb_file.read_text()
                lines = pdb_content.split("\n")
                motif_chain = None
                motif_res_ids = []

                for line in lines:
                    if line.startswith("ATOM"):
                        chain_id = line[21:22].strip()
                        res_id = int(line[22:26].strip())
                        if not motif_chain:
                            motif_chain = chain_id
                        if chain_id == motif_chain and res_id not in motif_res_ids:
                            motif_res_ids.append(res_id)

                # Default: keep all residues from the motif if not specified
                motif_start: int | None = None
                motif_end: int | None = None
                if not motif_residues:
                    motif_start = min(motif_res_ids)
                    motif_end = max(motif_res_ids)
                    motif_length = len(motif_res_ids)
                    print(
                        f"Auto-detected motif: {motif_chain}{motif_start}-{motif_chain}{motif_end} ({motif_length} residues)"
                    )
                else:
                    # Parse user-specified motif residues (e.g., ["A45-A50"])
                    for res_range in motif_residues:
                        if "-" in res_range:
                            start_str, end_str = res_range.split("-")
                            # Extract chain and residue number (e.g., "A45" -> chain="A", res=45)
                            if start_str[0].isalpha():
                                motif_chain = start_str[0]
                                motif_start = int(start_str[1:])
                                motif_end = int(end_str[1:])
                            else:
                                motif_start = int(start_str)
                                motif_end = int(end_str)

                    if motif_start and motif_end:
                        motif_length = motif_end - motif_start + 1
                        print(
                            f"Using specified motif: {motif_chain}{motif_start}-{motif_chain}{motif_end} ({motif_length} residues)"
                        )

                # Motif scaffolding uses 'unindex' (not 'contig'):
                # - unindex: marks motif tokens whose sequence placement is unknown
                # - length: total scaffold length (motif + designed regions)
                # - select_fixed_atoms: which atoms of the motif to fix in 3D space

                # Build unindex string (e.g., "A45-A50" or "A431,A572-573")
                if not motif_residues:
                    unindex_str = f"{motif_chain}{motif_start}-{motif_chain}{motif_end}"
                    print(f"Using unindex: {unindex_str} (motif length: {motif_length})")
                else:
                    # Convert motif_residues list to unindex string
                    unindex_parts = []
                    for res_range in motif_residues:
                        if "-" in res_range:
                            # Keep as-is if it has chain ID, otherwise add it
                            if res_range[0].isalpha():
                                unindex_parts.append(res_range)
                            else:
                                start, end = res_range.split("-")
                                unindex_parts.append(f"{motif_chain}{start}-{motif_chain}{end}")
                        else:
                            # Single residue
                            if res_range[0].isalpha():
                                unindex_parts.append(res_range)
                            else:
                                unindex_parts.append(f"{motif_chain}{res_range}")
                    unindex_str = ",".join(unindex_parts)
                    print(f"Using unindex: {unindex_str}")

                # Specification for motif scaffolding
                spec_kwargs = {
                    "input": str(motif_pdb_file),
                    "unindex": unindex_str,  # Which residues are the motif
                    "length": length,  # Total scaffold length
                    "select_fixed_atoms": {unindex_str: "ALL"},  # type: ignore[dict-item]  # Fix all motif atoms
                    "dialect": 2,
                }

                print("Motif scaffolding specification:")
                print(f"  Input: {motif_pdb_file.name}")
                print(f"  Unindex (motif): {unindex_str}")
                print(f"  Total length: {length}")
                print("  Fixed atoms: ALL")

                # Pass specification as dictionary (not pre-created object)
                # This avoids conflicts with atom_array_input being added internally
                specification_dict = spec_kwargs

                # Configure diffusion sampler
                sampler_config = {
                    "num_steps": inference_steps,
                }

                # Calculate batch configuration
                batch_size = min(num_designs, 4)
                n_batches = (num_designs + batch_size - 1) // batch_size

                print(
                    f"Batch configuration: {batch_size} designs/batch × {n_batches} batches = {batch_size * n_batches} total designs"
                )

                # Create inference config with spec dict
                config = RFD3InferenceConfig(
                    ckpt_path=ckpt_path,
                    specification=specification_dict,
                    inference_sampler=sampler_config,
                    diffusion_batch_size=batch_size,
                    verbose=True,
                )

                # Initialize engine and run generation
                print("Initializing RFD3 inference engine...")
                engine = RFD3InferenceEngine(**config.__dict__)

                print("Running RFD3 motif scaffolding inference...")
                outputs = engine.run(
                    inputs=None,
                    out_dir=None,
                    n_batches=n_batches,
                )

                print("✓ RFD3 motif scaffolding complete")
                print(f"Generated {len(outputs)} batches with output keys: {list(outputs.keys())}")

                # Process outputs (same pattern as de novo)
                output_files = []
                structure_count = 0

                for batch_idx, batch_outputs in outputs.items():
                    print(f"\nBatch {batch_idx}: {len(batch_outputs)} structures")

                    for _design_idx, rfd3_output in enumerate(batch_outputs):
                        structure_count += 1
                        design_name = f"scaffold_{structure_count:03d}"

                        # Save structure as CIF file
                        cif_path = out_path / f"{design_name}.cif"
                        to_cif_file(rfd3_output.atom_array, str(cif_path))
                        print(f"  ✓ Saved {design_name}.cif ({len(rfd3_output.atom_array)} atoms)")

                        file_size = cif_path.stat().st_size if cif_path.exists() else 0

                        output_files.append(
                            {
                                "path": str(cif_path.absolute()),
                                "artifact_type": "cif",
                                "filename": f"{design_name}.cif",
                                "size": file_size,
                                "metadata": {
                                    "design_idx": structure_count,
                                    "num_atoms": len(rfd3_output.atom_array),
                                    "batch_idx": batch_idx,
                                    "scaffold_length": length,
                                },
                            }
                        )

                        # Save confidence scores
                        if (
                            hasattr(rfd3_output, "summary_confidences")
                            and rfd3_output.summary_confidences
                        ):
                            conf_path = out_path / f"{design_name}_confidence.json"
                            conf_path.write_text(
                                json.dumps(rfd3_output.summary_confidences, indent=2)
                            )
                            print("    - Saved confidence scores")

                            conf_size = conf_path.stat().st_size if conf_path.exists() else 0

                            output_files.append(
                                {
                                    "path": str(conf_path.absolute()),
                                    "artifact_type": "json",
                                    "filename": f"{design_name}_confidence.json",
                                    "size": conf_size,
                                    "metadata": {
                                        "design_idx": structure_count,
                                        "is_confidence_scores": True,
                                    },
                                }
                            )

                # Upload to GCS if requested
                if upload_to_gcs and gcs_bucket:
                    print(f"\nUploading to GCS bucket: {gcs_bucket}")

                    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                    if not credentials_json:
                        print("❌ GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets")
                        return {
                            "run_id": run_id,
                            "design_type": design_type,
                            "num_designs": structure_count,
                            "num_output_files": len(output_files),
                            "length": length,
                            "inference_steps": inference_steps,
                            "output_files": output_files,
                            "success": False,
                            "error": "GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets",
                        }

                    # Write credentials to temp file
                    creds_path = Path("/tmp/gcs_credentials.json")
                    creds_path.write_text(credentials_json)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

                    from google.cloud import storage

                    client = storage.Client()
                    bucket = client.bucket(gcs_bucket)

                    gcs_prefix = f"runs/{run_id}/rfdiffusion3/"
                    for file_info in output_files:
                        gcs_path = f"{gcs_prefix}{file_info['filename']}"
                        blob = bucket.blob(gcs_path)
                        blob.upload_from_filename(file_info["path"])
                        file_info["gcs_url"] = f"gs://{gcs_bucket}/{gcs_path}"
                        print(f"  ✓ Uploaded: {file_info['filename']} -> {file_info['gcs_url']}")

                print("\n📊 Summary:")
                print(f"  Total scaffolds: {structure_count}")
                print(f"  Total files: {len(output_files)}")
                print(
                    f"  CIF structure files: {sum(1 for f in output_files if f['artifact_type'] == 'cif')}"
                )
                print(
                    f"  JSON metadata files: {sum(1 for f in output_files if f['artifact_type'] == 'json')}"
                )

                return {
                    "run_id": run_id,
                    "design_type": design_type,
                    "num_designs": structure_count,
                    "num_output_files": len(output_files),
                    "length": length,
                    "inference_steps": inference_steps,
                    "motif_residues": motif_residues,
                    "output_files": output_files,
                    "gcs_prefix": f"gs://{gcs_bucket}/{gcs_prefix}",
                    "success": True,
                }

            else:
                raise ValueError(
                    f"Invalid design configuration: "
                    f"design_type={design_type}, "
                    f"target_pdb={bool(target_pdb)}, "
                    f"motif_pdb={bool(motif_pdb)}"
                )

        except Exception as e:
            print(f"✗ RFD3 inference failed: {e}")
            import traceback

            traceback.print_exc()
            raise

        # Collect output files
        output_files_list = [  # type: ignore[unreachable]
            (out_file.relative_to(out_path), out_file.read_bytes())
            for out_file in out_path.rglob("*")
            if out_file.is_file()
        ]

        print(f"\nGenerated {len(output_files_list)} output files")

        # Handle GCS upload if requested
        if upload_to_gcs and gcs_bucket and run_id:
            import tempfile

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

            gcs_prefix = f"runs/{run_id}/rfdiffusion3"
            output_files = []

            # Upload each file to GCS
            for relative_path, content in output_files_list:
                filename = str(relative_path)
                blob_path = f"{gcs_prefix}/{filename}"
                blob = bucket.blob(blob_path)
                blob.upload_from_string(content)

                # Determine artifact type from file extension
                suffix = Path(filename).suffix.lstrip(".")
                artifact_type = suffix if suffix else "file"

                gcs_url = f"gs://{gcs_bucket}/{blob_path}"
                output_files.append(
                    {
                        "gcs_url": gcs_url,
                        "filename": filename,
                        "artifact_type": artifact_type,
                        "size": len(content),
                        "content": content,  # Include content for DB storage
                        "metadata": {
                            "design_type": design_type,
                            "num_designs": num_designs,
                            "inference_steps": inference_steps,
                        },
                    }
                )

            return {
                "exit_code": 0,
                "output_files": output_files,
                "gcs_prefix": f"gs://{gcs_bucket}/{gcs_prefix}/",
                "message": f"RFD3 completed: {len(output_files)} files uploaded to GCS",
                "tool": "rfdiffusion3",
                "run_id": run_id,
                "metadata": {
                    "num_designs": num_designs,
                    "inference_steps": inference_steps,
                    "design_type": design_type,
                    "target_chain": target_chain if target_chain else None,
                    "hotspots": hotspots if hotspots else None,
                    "symmetry": symmetry if symmetry else None,
                    "model_version": "rfd3",
                },
            }
        else:
            # Return files without GCS upload
            output_files = []
            for relative_path, content in output_files_list:
                filename = str(relative_path)
                suffix = Path(filename).suffix.lstrip(".")
                artifact_type = suffix if suffix else "file"

                output_files.append(
                    {
                        "filename": filename,
                        "artifact_type": artifact_type,
                        "size": len(content),
                        "content": content,
                        "metadata": {
                            "design_type": design_type,
                            "num_designs": num_designs,
                            "inference_steps": inference_steps,
                        },
                    }
                )

            return {
                "exit_code": 0,
                "output_files": output_files,
                "message": f"RFD3 completed: {len(output_files)} files generated",
                "tool": "rfdiffusion3",
                "run_id": run_id,
                "metadata": {
                    "num_designs": num_designs,
                    "inference_steps": inference_steps,
                    "design_type": design_type,
                    "target_chain": target_chain if target_chain else None,
                    "hotspots": hotspots if hotspots else None,
                    "symmetry": symmetry if symmetry else None,
                    "model_version": "rfd3",
                },
            }


# H100-specific endpoint for large proteins
# Note: H100-specific function removed - main function now uses H100 by default


@app.local_entrypoint()
def main(
    # De novo
    length: int | None = None,
    # Binder design
    target_pdb: str | None = None,
    target_chain: str | None = None,
    hotspots: str | None = None,  # Comma-separated: "A45,A67,A89"
    # Motif scaffolding
    motif_pdb: str | None = None,
    motif_residues: str | None = None,  # Comma-separated: "10-20,45-55"
    # Generation params
    num_designs: int = 10,
    inference_steps: int = 50,
    # Constraints
    contigs: str | None = None,
    symmetry: str | None = None,
    # Output
    out_dir: str = "./out/rfdiffusion3",
    run_name: str | None = None,
) -> None:
    """Run RFDiffusion3 locally with results saved to out_dir.

    Args:
        length: Backbone length for generation
        target_pdb: Path to target PDB file (for binder design)
        target_chain: Target chain ID (e.g., "A")
        hotspots: Comma-separated hotspot residues (e.g., "A45,A67,A89")
        motif_pdb: Path to motif PDB file (for scaffolding)
        motif_residues: Comma-separated motif residue ranges (e.g., "10-20,45-55")
        num_designs: Number of designs to generate
        inference_steps: Number of diffusion inference steps
        contigs: Contig specification string
        symmetry: Symmetry specification (C3, D2, etc.)
        out_dir: Local output directory
        run_name: Optional run name (defaults to timestamp)
    """
    from datetime import datetime

    # Read input files if provided
    target_pdb_content = None
    if target_pdb:
        target_pdb_content = Path(target_pdb).read_text()

    motif_pdb_content = None
    if motif_pdb:
        motif_pdb_content = Path(motif_pdb).read_text()

    # Parse comma-separated lists
    hotspots_list = None
    if hotspots:
        hotspots_list = [h.strip() for h in hotspots.split(",")]

    motif_residues_list = None
    if motif_residues:
        motif_residues_list = [r.strip() for r in motif_residues.split(",")]

    # Run generation
    result = rfdiffusion3_generate.remote(
        length=length,
        target_pdb=target_pdb_content,
        target_chain=target_chain,
        hotspots=hotspots_list,
        motif_pdb=motif_pdb_content,
        motif_residues=motif_residues_list,
        num_designs=num_designs,
        inference_steps=inference_steps,
        contigs=contigs,
        symmetry=symmetry,
    )

    # Save outputs locally
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    for file_info in result["output_files"]:
        output_path = out_dir_full / file_info["filename"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(file_info["content"])

    print(f"\nResults saved to: {out_dir_full}")
    print(f"Generated {len(result['output_files'])} files")
    print(f"Design type: {result['metadata']['design_type']}")


# Note: H100-specific endpoint removed - main function now uses H100 by default
