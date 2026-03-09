# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""ESM2 predict masked amino acid.

Input a fasta with format:
>1
MA<mask>GMT

Returns a tsv file of most probably amino acids.
"""

import os
from pathlib import Path

import modal
from modal import App, Image

GPU = os.environ.get("GPU", None)
TIMEOUT = int(os.environ.get("TIMEOUT", 15))


def download_model():
    import esm

    _model, _alphabet = esm.pretrained.esm2_t33_650M_UR50D()


image = (
    Image.micromamba(python_version="3.10")
    .apt_install(["git", "wget", "gcc", "g++", "libffi-dev"])
    .pip_install(["torch==1.13.1+cu117"], index_url="https://download.pytorch.org/whl/cu117")
    .pip_install(["numpy<2.0"])  # Pin NumPy to 1.x for ESM compatibility
    .pip_install(["fair-esm"])
    .pip_install(["pandas", "matplotlib"])
    .pip_install(["google-cloud-storage==2.14.0"])  # For GCS upload
    .run_function(download_model, gpu=GPU)
)

app = App("esm2", image=image)


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def esm2(
    fasta_name: str,
    fasta_str: str,
    make_figures: bool = False,
    upload_to_gcs: bool = False,
    gcs_bucket: str = "dev-services",
    run_id: str | None = None,
) -> dict:
    from tempfile import TemporaryDirectory

    import esm
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    with TemporaryDirectory() as td_out:
        # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        assert fasta_str.startswith(">"), f"{fasta_name} is not a fasta file"

        data = []
        for entry in fasta_str[1:].split("\n>"):
            label, _, seq = entry.partition("\n")
            seq = seq.replace("\n", "").strip()
            data.append((label, seq))

        _batch_labels, _batch_strs, batch_tokens = batch_converter(data)

        results_list = []
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        for i, (label, seq) in enumerate(data):
            # Find ALL mask positions for this sequence (not just the first one!)
            mask_positions = (batch_tokens[i] == alphabet.mask_idx).nonzero(as_tuple=True)[0]

            if len(mask_positions) == 0:
                print(f"Warning: No mask tokens found in sequence '{label}'")
                continue

            print(
                f"\nSequence '{label}' has {len(mask_positions)} mask(s) at positions: {mask_positions.tolist()}"
            )

            # Process each masked position independently
            for mask_idx, mask_position in enumerate(mask_positions.tolist()):
                # Get logits for this masked position
                logits = results["logits"][i, mask_position]

                # Convert logits to probabilities
                probs = torch.nn.functional.softmax(logits, dim=0)

                # Get the top 5 predictions for logging
                top_probs, top_indices = probs.topk(5)
                best_prediction = alphabet.get_tok(top_indices[0])
                best_probability = top_probs[0].item()

                print(
                    f"  Mask #{mask_idx + 1} (position {mask_position}): {best_prediction} ({best_probability:.4f})"
                )
                print(
                    f"    Top 5: {[(alphabet.get_tok(top_indices[j]), f'{top_probs[j].item():.4f}') for j in range(5)]}"
                )

                # Store ALL predictions (sorted by probability)
                # Filter out special tokens (keep only standard amino acids)
                all_probs, all_indices = probs.sort(descending=True)
                for prob, idx in zip(all_probs, all_indices, strict=False):
                    aa = alphabet.get_tok(idx)
                    # Skip special tokens for cleaner output
                    if aa not in ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>", "<null_1>"]:
                        results_list.append(
                            (i, label, mask_idx + 1, mask_position, aa, round(float(prob), 4))
                        )

            if make_figures:
                # Visualize the contact map
                plt.figure(figsize=(10, 10))
                plt.matshow(results["contacts"][i].cpu())
                plt.title(f"Contact Map for {label}")
                plt.colorbar()
                plt.savefig(f"{td_out}/{fasta_name}.contact_map_{label}.png")
                plt.close()

        df = pd.DataFrame(
            results_list, columns=["seq_n", "label", "mask_num", "mask_pos", "aa", "prob"]
        )
        df.to_csv(Path(td_out) / f"{fasta_name}.results.tsv", sep="\t", index=None)

        print(results_list)

        # Collect output files in structured format
        output_files = []
        for out_file in Path(td_out).glob("**/*.*"):
            if out_file.is_file():
                file_size = out_file.stat().st_size
                suffix = out_file.suffix.lstrip(".")

                # Determine artifact type
                artifact_type = (
                    "tsv"
                    if suffix == "tsv"
                    else "png"
                    if suffix == "png"
                    else "csv"
                    if suffix == "csv"
                    else "file"
                )

                output_files.append(
                    {
                        "path": str(out_file.absolute()),
                        "artifact_type": artifact_type,
                        "filename": out_file.name,
                        "size": file_size,
                        "metadata": {
                            "make_figures": make_figures,
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
                blob_path = f"runs/{run_id}/esm2/{filename}"
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
                "gcs_prefix": f"gs://{gcs_bucket}/runs/{run_id}/esm2/",
                "message": f"ESM2 completed. Uploaded {len(gcs_files)} files to GCS.",
                "stdout": f"ESM2 completed successfully. Files uploaded to gs://{gcs_bucket}/runs/{run_id}/esm2/",
                "metadata": {
                    "make_figures": make_figures,
                    "run_id": run_id,
                    "gcs_bucket": gcs_bucket,
                },
            }

        return {
            "exit_code": 0,
            "output_files": output_files,
            "message": f"ESM2 completed. Generated {len(output_files)} output files.",
            "stdout": "ESM2 prediction completed successfully.",
            "metadata": {
                "make_figures": make_figures,
            },
        }


@app.local_entrypoint()
def main(
    input_faa: str,
    make_figures: bool = False,
    out_dir: str = "./out/esm2_predict_masked",
    run_name: str | None = None,
) -> None:
    from datetime import datetime

    fasta_str = open(input_faa).read()

    result = esm2.remote(Path(input_faa).name, fasta_str, make_figures)

    if result["exit_code"] != 0:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return

    print("ESM2 completed successfully!")
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
