"""BoltzGen https://github.com/HannesStark/boltzgen

Example yaml file:
```yaml
entities:
  - protein:
      id: B
      sequence: 80..140
  - file:
      path: 6m1u.cif
      include:
        - chain:
            id: A
```

Example usage:
```bash
wget https://raw.githubusercontent.com/HannesStark/boltzgen/refs/heads/main/example/vanilla_protein/1g13.cif
wget https://raw.githubusercontent.com/HannesStark/boltzgen/refs/heads/main/example/vanilla_protein/1g13prot.yaml
modal run modal_boltzgen.py --input-yaml 1g13prot.yaml --protocol protein-anything --num-designs 2
```

Available protocols: protein-anything, peptide-anything, protein-small_molecule, nanobody-anything

Other useful options:
  --steps design inverse_folding folding    # Run specific steps only
  --cache /path/to/cache                    # Custom cache directory
  --devices 2                               # Number of GPUs to use
"""

import os
import subprocess
from pathlib import Path

import modal
from modal import App, Image

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 120))


def download_boltzgen_models():
    """Download all boltzgen models during image build to avoid runtime timeouts."""

    # Download all artifacts to default cache location (~/.cache)
    print("Downloading boltzgen models...")
    subprocess.run(
        ["boltzgen", "download", "all"],
        check=True,
    )
    print("Model download complete")


image = (
    Image.debian_slim()
    .apt_install("git", "wget", "build-essential")
    .pip_install("torch>=2.4.1")
    .run_commands(
        "git clone https://github.com/HannesStark/boltzgen /root/boltzgen",
        "cd /root/boltzgen && git checkout 247b9bbd8b68a60aba854c2968d6a0cddd21ad6d && pip install -e .",  # Dec 18 2025 - includes weights_only fix
        gpu="a10g",
    )
    .pip_install("google-cloud-storage==2.14.0")  # For GCS upload
    .run_function(
        download_boltzgen_models,
        gpu="a10g",
    )
)

app = App(
    "boltzgen",
    image=image,
)


@app.function(
    timeout=10 * 60,
    gpu="T4",  # BoltzGen checks GPU availability even for configure
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def boltzgen_configure(
    yaml_str: str,
    yaml_name: str,
    additional_files: dict[str, bytes],
    protocol: str = "protein-anything",
    num_designs: int = 200,
    budget: int | None = None,
    extra_args: str | None = None,
    gcs_bucket: str = "dev-services",
    run_id: str | None = None,
) -> dict:
    """Generate BoltzGen configuration and upload to GCS (for parallel workflows).

    This is the first step in: configure → execute (parallel) → merge
    Config is stored in GCS so multiple execute nodes can access it.

    Args:
        yaml_str: YAML design specification
        yaml_name: Name of YAML file
        additional_files: Referenced files (e.g., CIF structures)
        protocol: Design protocol
        num_designs: Total designs to configure for
        budget: Final filtered designs (defaults to num_designs // 10)
        extra_args: Additional CLI arguments
        gcs_bucket: GCS bucket name (required for parallel workflows)
        run_id: Run identifier (required)

    Returns:
        Dict with GCS config URI
    """
    import json
    import os
    from subprocess import run
    from tempfile import TemporaryDirectory

    from google.cloud import storage
    from google.oauth2 import service_account

    if not run_id:
        raise ValueError("run_id required for configure step")

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        # Write YAML and files
        yaml_path = Path(in_dir) / yaml_name
        yaml_path.write_text(yaml_str)

        for rel_path, content in additional_files.items():
            file_path = Path(in_dir) / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)

        # Build configure command
        cmd = [
            "boltzgen",
            "configure",
            str(yaml_path),
            "--output",
            out_dir,
            "--protocol",
            protocol,
            "--num_designs",
            str(num_designs),
        ]

        if budget:
            cmd.extend(["--budget", str(budget)])
        if extra_args:
            cmd.extend(extra_args.split())

        print(f"Configuring: {' '.join(cmd)}")
        run(cmd, check=True)

        # Upload everything to GCS
        credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not credentials_json:
            raise ValueError("GCS credentials required for configure step")

        credentials = service_account.Credentials.from_service_account_info(
            json.loads(credentials_json)
        )
        client = storage.Client(credentials=credentials)
        bucket_obj = client.bucket(gcs_bucket)

        gcs_prefix = f"runs/{run_id}/boltzgen_config"

        # Upload config files
        for path in Path(out_dir).rglob("*"):
            if path.is_file():
                rel_path_str = str(path.relative_to(out_dir))
                blob_path = f"{gcs_prefix}/config/{rel_path_str}"
                blob = bucket_obj.blob(blob_path)
                blob.upload_from_filename(str(path))

        # Upload original YAML and additional files with metadata about original paths
        blob = bucket_obj.blob(f"{gcs_prefix}/input/{yaml_name}")
        blob.metadata = {"original_path": str(yaml_path)}
        blob.upload_from_string(yaml_str)

        for rel_path, content in additional_files.items():
            blob = bucket_obj.blob(f"{gcs_prefix}/input/{rel_path}")
            src_path = Path(in_dir) / rel_path
            blob.metadata = {"original_path": str(src_path)}
            blob.upload_from_string(content)

        # Store the output directory path as metadata for execute step
        blob = bucket_obj.blob(f"{gcs_prefix}/metadata.json")
        metadata = {
            "output_dir": str(out_dir),
            "yaml_path": str(yaml_path),
        }
        blob.upload_from_string(json.dumps(metadata))

        config_uri = f"gs://{gcs_bucket}/{gcs_prefix}/"

        return {
            "exit_code": 0,
            "config_uri": config_uri,
            "gcs_bucket": gcs_bucket,
            "gcs_prefix": gcs_prefix,
            "output_dir": str(out_dir),
            "message": f"Config uploaded to {config_uri}",
        }


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def boltzgen_execute(
    config_gcs_uri: str,
    gcs_bucket: str,
    run_id: str,
    batch_id: str,
    steps: str = "design inverse_folding folding analysis",
    num_designs_per_batch: int = 10,
    protocol: str = "protein-anything",
) -> dict:
    """Execute BoltzGen batch.

    NOTE: Currently runs full pipeline per batch due to BoltzGen's state management.
    See docs/BOLTZGEN_PARALLEL_TODO.md for details on parallelization challenges.

    This is the middle step in: configure → execute (parallel) → merge
    Each batch runs independently with its own num_designs.

    Args:
        config_gcs_uri: GCS URI from configure step (for YAML file)
        gcs_bucket: GCS bucket name
        run_id: Run identifier
        batch_id: Unique ID for this batch
        steps: Specific steps to run
        num_designs_per_batch: Number of designs for this batch
        protocol: Design protocol

    Returns:
        Dict with GCS output URI
    """
    import json
    import os
    from subprocess import run
    from tempfile import TemporaryDirectory

    from google.cloud import storage
    from google.oauth2 import service_account

    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not credentials_json:
        raise ValueError("GCS credentials required")

    credentials = service_account.Credentials.from_service_account_info(
        json.loads(credentials_json)
    )
    client = storage.Client(credentials=credentials)
    bucket_obj = client.bucket(gcs_bucket)

    # Parse GCS prefix from URI
    config_prefix = config_gcs_uri.replace(f"gs://{gcs_bucket}/", "").rstrip("/")

    with TemporaryDirectory() as work_dir:
        # Download YAML from config
        blobs = list(bucket_obj.list_blobs(prefix=f"{config_prefix}/input/"))
        yaml_blobs = [b for b in blobs if b.name.endswith(".yaml")]

        if not yaml_blobs:
            raise FileNotFoundError(f"No YAML found in {config_prefix}/input/")

        yaml_blob = yaml_blobs[0]
        yaml_path = Path(work_dir) / "input.yaml"
        yaml_blob.download_to_filename(str(yaml_path))
        print(f"✓ Downloaded YAML: {yaml_blob.name}")

        # Download any additional files (CIFs, etc.)
        for blob in blobs:
            if not blob.name.endswith(".yaml"):
                rel_path = blob.name.replace(f"{config_prefix}/input/", "")
                dest_path = Path(work_dir) / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(dest_path))
                print(f"✓ Downloaded: {rel_path}")

        # Run BoltzGen with fresh output directory
        output_dir = Path(work_dir) / "output"
        output_dir.mkdir()

        cmd = [
            "boltzgen",
            "run",
            str(yaml_path),
            "--output",
            str(output_dir),
            "--protocol",
            protocol,
            "--num_designs",
            str(num_designs_per_batch),
            "--steps",
        ] + steps.split()

        print(f"\nRunning batch {batch_id}: {' '.join(cmd)}")
        run(cmd, check=True)

        # Upload results to GCS
        output_prefix = f"runs/{run_id}/boltzgen_execute_{batch_id}"

        print("\nUploading results...")
        uploaded_count = 0
        for path in output_dir.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(output_dir)
                blob_path = f"{output_prefix}/{rel_path}"
                blob = bucket_obj.blob(blob_path)
                blob.upload_from_filename(str(path))
                uploaded_count += 1

        print(f"✓ Uploaded {uploaded_count} files")

        output_uri = f"gs://{gcs_bucket}/{output_prefix}/"

        return {
            "exit_code": 0,
            "output_uri": output_uri,
            "gcs_bucket": gcs_bucket,
            "gcs_prefix": output_prefix,
            "batch_id": batch_id,
            "num_files": uploaded_count,
            "message": f"Batch {batch_id} complete: {uploaded_count} files at {output_uri}",
        }


@app.function(
    timeout=30 * 60,
    gpu="T4",  # Light GPU for filtering only
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def boltzgen_merge(
    execute_gcs_uris: list[str],
    config_gcs_uri: str,
    gcs_bucket: str,
    run_id: str,
    protocol: str = "protein-anything",
    budget: int = 100,
    alpha: float = 0.01,
    filter_args: str | None = None,
) -> dict:
    """Merge parallel BoltzGen runs from GCS and re-filter combined results.

    This is the final step in: configure → execute (parallel) → merge
    Reads results from multiple GCS URIs, merges, filters, and uploads final results.

    Args:
        execute_gcs_uris: List of GCS URIs from execute steps
        config_gcs_uri: GCS URI from configure step (for original YAML)
        gcs_bucket: GCS bucket name
        run_id: Run identifier
        protocol: Design protocol
        budget: Final number of filtered designs
        alpha: Diversity vs quality tradeoff (0.0=quality, 1.0=diversity)
        filter_args: Additional filtering arguments

    Returns:
        Dict with final results GCS URI
    """
    import json
    import os
    from subprocess import run
    from tempfile import TemporaryDirectory

    from google.cloud import storage
    from google.oauth2 import service_account

    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not credentials_json:
        raise ValueError("GCS credentials required")

    credentials = service_account.Credentials.from_service_account_info(
        json.loads(credentials_json)
    )
    client = storage.Client(credentials=credentials)
    bucket_obj = client.bucket(gcs_bucket)

    with TemporaryDirectory() as work_dir:
        # Download each batch to separate directories (required for boltzgen merge)
        print(f"Downloading {len(execute_gcs_uris)} batches...")
        source_dirs = []
        total_design_count = 0

        for i, execute_uri in enumerate(execute_gcs_uris):
            source_dir = Path(work_dir) / f"source_{i}"
            source_dir.mkdir()
            source_dirs.append(str(source_dir))

            print(f"  Batch {i}: {execute_uri}")
            prefix = execute_uri.replace(f"gs://{gcs_bucket}/", "").rstrip("/")

            blobs = list(bucket_obj.list_blobs(prefix=prefix))
            print(f"    Downloading {len(blobs)} files...")

            if len(blobs) == 0:
                print("    WARNING: No files found!")
                continue

            batch_designs = 0
            for blob in blobs:
                if blob.name.endswith("/"):
                    continue
                rel_path = blob.name.replace(f"{prefix}/", "")

                dest_path = source_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(dest_path))

                if "intermediate_designs/" in rel_path and rel_path.endswith(".cif"):
                    batch_designs += 1

            print(f"    ✓ {batch_designs} designs in this batch")
            total_design_count += batch_designs

        print(f"\n✓ Downloaded {total_design_count} total designs from {len(source_dirs)} batches")

        # Download original YAML from config
        config_prefix = config_gcs_uri.replace(f"gs://{gcs_bucket}/", "").rstrip("/")
        input_blobs = list(bucket_obj.list_blobs(prefix=f"{config_prefix}/input/"))

        print(f"\nLooking for YAML in {config_prefix}/input/")
        print(f"Found {len(input_blobs)} blobs in config input:")
        for blob in input_blobs:
            print(f"  - {blob.name}")

        yaml_blobs = [b for b in input_blobs if b.name.endswith(".yaml")]
        if not yaml_blobs:
            raise FileNotFoundError(
                f"No YAML file found in {config_prefix}/input/. "
                f"Found {len(input_blobs)} blobs total. "
                f"This likely means the configure step didn't upload the input YAML properly."
            )

        yaml_blob = yaml_blobs[0]
        yaml_path = Path(work_dir) / "input.yaml"
        yaml_blob.download_to_filename(str(yaml_path))
        print(f"✓ Downloaded YAML: {yaml_blob.name}")

        # Use boltzgen's built-in merge command for parallel runs
        # This properly combines analyzed results from multiple batches
        print(f"\nMerging {len(source_dirs)} batches using BoltzGen...")

        final_merged = Path(work_dir) / "final_merged"
        merge_cmd = ["boltzgen", "merge"] + source_dirs + ["--output", str(final_merged)]

        print(f"Merge command: {' '.join(merge_cmd)}")
        from subprocess import run

        run(merge_cmd, check=True)

        # Now run filtering on merged results
        filter_cmd = [
            "boltzgen",
            "run",
            str(yaml_path),
            "--steps",
            "filtering",
            "--output",
            str(final_merged),
            "--protocol",
            protocol,
            "--budget",
            str(budget),
            "--alpha",
            str(alpha),
        ]

        if filter_args:
            filter_cmd.extend(filter_args.split())

        print(f"\nFiltering merged results: {' '.join(filter_cmd)}")
        run(filter_cmd, check=True)

        print("✓ Merge and filter completed successfully")

        # Upload final results to GCS
        final_prefix = f"runs/{run_id}/boltzgen_final"

        print("\nUploading final results to GCS...")
        uploaded_count = 0
        for path in final_merged.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(final_merged)
                blob_path = f"{final_prefix}/{rel_path}"
                blob = bucket_obj.blob(blob_path)
                blob.upload_from_filename(str(path))
                uploaded_count += 1

        # Count final designs
        final_designs_dir = final_merged / "final_ranked_designs"
        final_designs = list(final_designs_dir.glob("*.cif")) if final_designs_dir.exists() else []

        print(f"✓ Uploaded {uploaded_count} files ({len(final_designs)} final designs)")

        output_uri = f"gs://{gcs_bucket}/{final_prefix}/"

        return {
            "exit_code": 0,
            "output_uri": output_uri,
            "gcs_bucket": gcs_bucket,
            "gcs_prefix": final_prefix,
            "num_final_designs": len(final_designs),
            "message": f"Merge complete: {len(final_designs)} final designs at {output_uri}",
        }


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def boltzgen_run(
    yaml_str: str | None = None,
    yaml_name: str | None = None,
    yaml_gcs_uri: str | None = None,
    structure_file_gcs_uri: str | None = None,
    additional_files: dict[str, bytes] | None = None,
    protocol: str = "protein-anything",
    num_designs: int = 10,
    steps: str | None = None,
    cache: str | None = None,
    devices: int | None = None,
    extra_args: str | None = None,
    upload_to_gcs: bool = False,
    gcs_bucket: str | None = None,
    run_id: str | None = None,
) -> list | dict:
    """Run BoltzGen on a yaml specification.

    Args:
        yaml_str: YAML design specification as string (deprecated, use yaml_gcs_uri)
        yaml_name: Name of the yaml file (deprecated, use yaml_gcs_uri)
        yaml_gcs_uri: GCS URI to YAML file (e.g., gs://bucket/path/design.yaml)
        structure_file_gcs_uri: GCS URI for structure file referenced in YAML
                                (e.g., "gs://bucket/structures/1abc.pdb")
                                File is downloaded with its original basename (e.g., "1abc.pdb")
                                YAML must reference file by this basename (path: 1abc.pdb)
        additional_files: Dict of relative_path -> file_content for referenced files (deprecated)
        protocol: Design protocol (protein-anything, peptide-anything, etc.)
        num_designs: Number of designs to generate
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        cache: Custom cache directory path
        devices: Number of GPUs to use
        extra_args: Additional CLI arguments as string

    Returns:
        Dict with exit_code, output_files, gcs_prefix, message
    """
    import tempfile
    from subprocess import run
    from tempfile import TemporaryDirectory

    from google.cloud import storage

    # Initialize GCS client
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
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

    client = storage.Client()

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        # Handle GCS URI input (preferred method)
        if yaml_gcs_uri:
            print(f"Downloading YAML from GCS: {yaml_gcs_uri}")

            # Parse GCS URI: gs://bucket/path/to/file.yaml
            if not yaml_gcs_uri.startswith("gs://"):
                return {
                    "exit_code": 1,
                    "output_files": [],
                    "message": f"Error: Invalid GCS URI: {yaml_gcs_uri}",
                    "metadata": {},
                }

            gcs_path = yaml_gcs_uri.replace("gs://", "")
            bucket_name, blob_path = gcs_path.split("/", 1)
            bucket = client.bucket(bucket_name)

            # Download YAML file
            yaml_filename = Path(blob_path).name
            yaml_path = Path(in_dir) / yaml_filename
            blob = bucket.blob(blob_path)
            blob.download_to_filename(str(yaml_path))
            print(f"  ✓ Downloaded {yaml_filename}")

            # Download structure file specified in structure_file_gcs_uri
            # File is downloaded with its original basename to same directory as YAML
            if structure_file_gcs_uri:
                print("  Downloading structure file:")

                if not structure_file_gcs_uri.startswith("gs://"):
                    return {
                        "exit_code": 1,
                        "output_files": [],
                        "message": f"Error: Invalid GCS URI: {structure_file_gcs_uri}",
                        "metadata": {},
                    }

                # Parse GCS URI and get basename
                file_gcs_path = structure_file_gcs_uri.replace("gs://", "")
                file_bucket_name, file_blob_path = file_gcs_path.split("/", 1)
                file_bucket = client.bucket(file_bucket_name)
                file_blob = file_bucket.blob(file_blob_path)

                # Download with original basename
                basename = Path(file_blob_path).name
                local_path = Path(in_dir) / basename

                print(f"    {basename} <- {structure_file_gcs_uri}")

                if file_blob.exists():
                    file_blob.download_to_filename(str(local_path))
                    print(f"      ✓ Downloaded {basename} ({local_path.stat().st_size} bytes)")
                else:
                    return {
                        "exit_code": 1,
                        "output_files": [],
                        "message": f"Error: File not found at {structure_file_gcs_uri}",
                        "metadata": {},
                    }

        # Fallback: Handle string input (legacy, deprecated)
        elif yaml_str and yaml_name:
            print("Using legacy yaml_str input (deprecated, use yaml_gcs_uri)")
            yaml_path = Path(in_dir) / yaml_name
            yaml_path.write_text(yaml_str)

            # Write any additional files
            if additional_files:
                for rel_path, content in additional_files.items():
                    file_path = Path(in_dir) / rel_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_bytes(content)
        else:
            return {
                "exit_code": 1,
                "output_files": [],
                "message": "Error: Either yaml_gcs_uri or (yaml_str + yaml_name) required",
                "metadata": {},
            }

        # Build command
        cmd = [
            "boltzgen",
            "run",
            str(yaml_path),
            "--output",
            out_dir,
            "--protocol",
            protocol,
            "--num_designs",
            str(num_designs),
        ]

        if steps:
            cmd.extend(["--steps"] + steps.split())
        if cache:
            cmd.extend(["--cache", cache])
        if devices:
            cmd.extend(["--devices", str(devices)])
        if extra_args:
            cmd.extend(extra_args.split())

        print(f"Running: {' '.join(cmd)}")
        run(cmd, check=True)

        # Collect all output files
        output_files_list = [
            (out_file.relative_to(out_dir), out_file.read_bytes())
            for out_file in Path(out_dir).rglob("*")
            if out_file.is_file()
        ]

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

            gcs_prefix = f"runs/{run_id}/boltzgen"
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
                        "metadata": {
                            "protocol": protocol,
                            "num_designs": num_designs,
                        },
                    }
                )

            return {
                "exit_code": 0,
                "output_files": output_files,
                "gcs_prefix": f"gs://{gcs_bucket}/{gcs_prefix}/",
                "message": f"BoltzGen completed: {len(output_files)} files uploaded to GCS",
                "metadata": {
                    "protocol": protocol,
                    "num_designs": num_designs,
                },
            }
        else:
            # Return files as list of tuples (original behavior)
            return output_files_list


@app.local_entrypoint()
def test_merge_only():
    """Test merge step only using existing GCS data (for debugging).

    IMPORTANT: Get the run_id from the output of test_boltzgen_parallel.py
    Look for lines like:
        Run ID: boltzgen_test_20260203_110644
        ✓ Batch 0 complete
          URI: gs://dev-services/runs/boltzgen_test_20260203_110644/boltzgen_execute_batch_0/

    Usage:
        1. Update run_id below
        2. Run: modal run modal/biomodals/modal_boltzgen.py::test_merge_only
    """
    # UPDATE THIS with your actual run_id from a successful test run:
    run_id = "boltzgen_test_20260203_110644"
    gcs_bucket = "dev-services"

    config_gcs_uri = f"gs://{gcs_bucket}/runs/{run_id}/boltzgen_config/"
    execute_gcs_uris = [
        f"gs://{gcs_bucket}/runs/{run_id}/boltzgen_execute_batch_0/",
        f"gs://{gcs_bucket}/runs/{run_id}/boltzgen_execute_batch_1/",
    ]

    print("=" * 60)
    print("Testing MERGE step only (reusing GCS data)")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"GCS Bucket: {gcs_bucket}")
    print(f"Config URI: {config_gcs_uri}")
    print(f"\nExecute URIs ({len(execute_gcs_uris)} batches):")
    for uri in execute_gcs_uris:
        print(f"  - {uri}")
    print()

    final_result = boltzgen_merge.remote(
        execute_gcs_uris=execute_gcs_uris,
        config_gcs_uri=config_gcs_uri,
        gcs_bucket=gcs_bucket,
        run_id=f"{run_id}_merge_test",
        protocol="protein-anything",
        budget=10,
        alpha=0.01,
    )

    print("\n" + "=" * 60)
    print("✓ Merge complete!")
    print("=" * 60)
    print(f"Final designs: {final_result['num_final_designs']}")
    print(f"Output URI: {final_result['output_uri']}")


@app.local_entrypoint()
def main(
    input_yaml: str,
    protocol: str = "protein-anything",
    num_designs: int = 10,
    steps: str | None = None,
    cache: str | None = None,
    devices: int | None = None,
    extra_args: str | None = None,
    out_dir: str = "./out/boltzgen",
    run_name: str | None = None,
) -> None:
    """Run BoltzGen locally with results saved to out_dir.

    Args:
        input_yaml: Path to YAML design specification file
        protocol: Design protocol (protein-anything, peptide-anything, protein-small_molecule, nanobody-anything)
        num_designs: Number of designs to generate
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        cache: Custom cache directory path
        devices: Number of GPUs to use
        extra_args: Additional CLI arguments as string
        out_dir: Local output directory
        run_name: Optional run name (defaults to timestamp)
    """
    import re
    from datetime import datetime

    yaml_path = Path(input_yaml)
    yaml_str = yaml_path.read_text()
    yaml_dir = yaml_path.parent

    # Find any file references in the yaml (path: something.cif)
    # File paths in yaml are relative to the yaml file location
    additional_files = {}
    for match in re.finditer(r"path:\s*([^\s\n]+)", yaml_str):
        ref_file = match.group(1)
        ref_path = yaml_dir / ref_file
        if ref_path.exists():
            additional_files[ref_file] = ref_path.read_bytes()
            print(f"Including referenced file: {ref_file}")

    outputs = boltzgen_run.remote(
        yaml_str=yaml_str,
        yaml_name=yaml_path.name,
        additional_files=additional_files,
        protocol=protocol,
        num_designs=num_designs,
        steps=steps,
        cache=cache,
        devices=devices,
        extra_args=extra_args,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        output_path = out_dir_full / out_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(out_content)

    print(f"\nResults saved to: {out_dir_full}")
