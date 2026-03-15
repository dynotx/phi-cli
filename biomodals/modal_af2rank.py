# /// script
# requires-python = ">=3.9,<3.11"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Ranks protein structures using AF2Rank/ColabDesign.

This script provides a Modal app to run AF2Rank, a method for ranking protein structures
based on AlphaFold2 predictions. It can take a PDB file as input, run the AF2Rank protocol
using specified models and chains, and return various scores and the ranked structure.

e.g.,
```
wget https://files.rcsb.org/download/4KRL.pdb
modal run modal_af2rank.py --input-pdb 4RKL.pdb
```

using AF2 multimer instead, and use both chains
```
modal run modal_af2rank.py --input-pdb 4KRL.pdb --model-name "model_1_multimer_v3" --chains "A,B"
```

Note: AF2Rank uses standard AlphaFold model names (model_1, model_2, etc.) not PTM variants.

"""

import os
from pathlib import Path

import modal
from modal import App, Image

app = App("af2rank")

GPU = os.environ.get("MODAL_GPU", "L40S")
TIMEOUT = os.environ.get("MODAL_TIMEOUT", 20 * 60)

image = (
    Image.micromamba(python_version="3.10")
    .apt_install("wget", "curl", "git", "g++")
    .pip_install("git+https://github.com/sokrypton/ColabDesign.git@v1.1.2")
    # Patch ColabDesign for modern JAX compatibility (jax.tree_map → jax.tree_util.tree_map)
    .run_commands(
        "find /opt/conda/lib/python3.10/site-packages/colabdesign -name '*.py' -exec sed -i 's/jax\\.tree_map/jax.tree_util.tree_map/g' {} +"
    )
    # Install JAX with CUDA support (like AlphaFold does)
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]==0.4.38" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="l40s",
    )
    .run_commands(
        "ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign",
        "mkdir params",
        "curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C params",
        "mv params /root/",
        "wget -qnc https://zhanggroup.org/TM-score/TMscore.cpp",
        "g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp",
        "cp TMscore /root/",
    )
    .pip_install("ipython")
    .pip_install("google-cloud-storage==2.14.0")
)


with image.imports():
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr

    def tmscore(x, y):
        """Calculates the TMscore between two protein structures.

        Args:
            x (list): A list of coordinates for the first protein structure.
            y (list): A list of coordinates for the second protein structure.

        Returns:
            dict: A dictionary containing 'rms', 'tms', and 'gdt' scores.
        """
        # save to dumpy pdb files
        for n, z in enumerate([x, y]):
            out = open(f"{n}.pdb", "w")
            for k, c in enumerate(z):
                out.write(
                    "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f   d%4.2f\n"
                    % (k + 1, "CA", "ALA", "A", k + 1, c[0], c[1], c[2], 1, 0)
                )
            out.close()

        # pass to TMscore
        output = os.popen("./TMscore 0.pdb 1.pdb")

        # parse outputs
        def parse_float(x_str):
            """Parses a float from a TMscore output line."""
            return float(x_str.split("=")[1].split()[0])

        o = {}
        for line in output:
            line = line.rstrip()
            if line.startswith("RMSD"):
                o["rms"] = parse_float(line)
            if line.startswith("TM-score"):
                o["tms"] = parse_float(line)
            if line.startswith("GDT-TS-score"):
                o["gdt"] = parse_float(line)

        return o

    def plot_me(
        scores,
        x="tm_i",
        y="composite",
        title=None,
        diag=False,
        scale_axis=True,
        dpi=100,
        **kwargs,
    ):
        """Plots scores, such as TMscore vs. composite scores.

        Args:
            scores (list[dict]): A list of dictionaries, where each dictionary contains scoring data.
            x (str): The key for the x-axis values in the scores.
            y (str): The key for the y-axis values in the scores.
            title (str | None): Optional title for the plot.
            diag (bool): Whether to draw a diagonal line on the plot.
            scale_axis (bool): Whether to scale axes from -0.1 to 1.1 if x or y are known score types.
            dpi (int): Dots per inch for the plot.
            **kwargs: Additional keyword arguments for `plt.scatter`.

        Returns:
            None
        """

        def rescale(a, amin=None, amax=None):
            """Rescales an array to the range [0, 1] based on provided min/max or array's own min/max."""
            a = np.copy(a)
            if amin is None:
                amin = a.min()
            if amax is None:
                amax = a.max()
            a[a < amin] = amin
            a[a > amax] = amax
            return (a - amin) / (amax - amin)

        plt.figure(figsize=(5, 5), dpi=dpi)
        if title is not None:
            plt.title(title)
        x_vals = np.array([k[x] for k in scores])
        y_vals = np.array([k[y] for k in scores])
        c = rescale(np.array([k["plddt"] for k in scores]), 0.5, 0.9)
        plt.scatter(
            x_vals,
            y_vals,
            c=c * 0.75,
            s=5,
            vmin=0,
            vmax=1,
            cmap="gist_rainbow",
            **kwargs,
        )
        if diag:
            plt.plot([0, 1], [0, 1], color="black")

        labels = {
            "tm_i": "TMscore of Input",
            "tm_o": "TMscore of Output",
            "tm_io": "TMscore between Input and Output",
            "ptm": "Predicted TMscore (pTM)",
            "i_ptm": "Predicted interface TMscore (ipTM)",
            "plddt": "Predicted LDDT (pLDDT)",
            "composite": "Composite",
        }

        plt.xlabel(labels.get(x, x))
        plt.ylabel(labels.get(y, y))
        if scale_axis:
            if x in labels:
                plt.xlim(-0.1, 1.1)
            if y in labels:
                plt.ylim(-0.1, 1.1)

        print(spearmanr(x_vals, y_vals).correlation)

    class af2rank:
        """A class to perform AF2Rank predictions using ColabDesign."""

        def __init__(self, pdb, chain=None, model_name="model_1_ptm", model_names=None):
            """Initializes the af2rank class.

            Args:
                pdb (str): Path to the PDB file.
                chain (str | None): Specific chain(s) to use from the PDB file.
                model_name (str): Name of the AlphaFold2 model to use.
                model_names (list[str] | None): Specific model names if not using a default set.
            """
            self.args = {
                "pdb": pdb,
                "chain": chain,
                "use_multimer": ("multimer" in model_name),
                "model_name": model_name,
                "model_names": model_names,
            }
            self.reset()

        def reset(self):
            """Resets and initializes the ColabDesign model."""
            from colabdesign import mk_af_model
            from colabdesign.shared.utils import copy_dict

            self.model = mk_af_model(
                protocol="fixbb",
                use_templates=True,
                use_multimer=self.args["use_multimer"],
                debug=False,
                model_names=self.args["model_names"],
                data_dir="/root/params",  # Tell ColabDesign where AlphaFold params are!
            )

            self.model.prep_inputs(self.args["pdb"], chain=self.args["chain"])
            self.model.set_seq(mode="wildtype")
            self.wt_batch = copy_dict(self.model._inputs["batch"])
            self.wt = self.model._wt_aatype

        def set_pdb(self, pdb, chain=None):
            """Sets the PDB file and chain for the model.

            Args:
                pdb (str): Path to the PDB file.
                chain (str | None): Specific chain(s) to use from the PDB file.
            """
            if chain is None:
                chain = self.args["chain"]
            self.model.prep_inputs(pdb, chain=chain)
            self.model.set_seq(mode="wildtype")
            self.wt = self.model._wt_aatype

        def set_seq(self, seq):
            """Sets the sequence for the model.

            Args:
                seq (str): Amino acid sequence.
            """
            self.model.set_seq(seq=seq)
            self.wt = self.model._params["seq"][0].argmax(-1)

        def _get_score(self):
            """Calculates and returns various scores from the model's auxiliary output.

            Returns:
                dict: A dictionary containing scores such as 'plddt', 'pae', 'ptm', 'iptm',
                      'rmsd_io' (RMSD between input and output), 'tm_i' (TMscore to input if reference provided),
                      'tm_o' (TMscore to output if reference provided), 'tm_io' (TMscore between input and output),
                      and 'composite' (ptm * plddt * tm_io).
            """
            from colabdesign.shared.utils import copy_dict

            score = copy_dict(self.model.aux["log"])

            score["plddt"] = score["plddt"]
            score["pae"] = 31.0 * score["pae"]
            score["rmsd_io"] = score.pop("rmsd", None)

            i_xyz = self.model._inputs["batch"]["all_atom_positions"][:, 1]
            o_xyz = np.array(self.model.aux["atom_positions"][:, 1])

            # TMscore to input/output
            if hasattr(self, "wt_batch"):
                n_xyz = self.wt_batch["all_atom_positions"][:, 1]
                score["tm_i"] = tmscore(n_xyz, i_xyz)["tms"]
                score["tm_o"] = tmscore(n_xyz, o_xyz)["tms"]

            # TMscore between input and output
            score["tm_io"] = tmscore(i_xyz, o_xyz)["tms"]

            # composite score
            score["composite"] = score["ptm"] * score["plddt"] * score["tm_io"]
            return score

        def predict(
            self,
            pdb=None,
            seq=None,
            chain=None,
            input_template=True,
            model_name=None,
            rm_seq=True,
            rm_sc=True,
            rm_ic=False,
            recycles=1,
            iterations=1,
            output_pdb=None,
            extras=None,
            verbose=True,
        ):
            """Runs the AF2Rank prediction.

            Args:
                pdb (str | None): Path to a new PDB file to use for this prediction.
                seq (str | None): Amino acid sequence to use for this prediction.
                chain (str | None): Specific chain(s) to use from the PDB file.
                input_template (bool): Whether to use the input structure as a template.
                model_name (str | None): Specific AlphaFold2 model name for this prediction.
                rm_seq (bool): Whether to remove the sequence from the template.
                rm_sc (bool): Whether to remove sidechain information from the template.
                rm_ic (bool): Whether to remove interchain information from the template (for multimers).
                recycles (int): Number of recycles for the AlphaFold2 model.
                iterations (int): Number of "manual" recycles using templates.
                output_pdb (str | None): If provided, saves the predicted structure to this path.
                extras (dict | None): Additional items to add to the score dictionary.
                verbose (bool): Whether to print score summaries.

            Returns:
                dict: The score dictionary produced by `_get_score`, potentially updated with `extras`.
            """
            if model_name is not None:
                self.args["model_name"] = model_name
                if "multimer" in model_name:
                    if not self.args["use_multimer"]:
                        self.args["use_multimer"] = True
                        self.reset()
                else:
                    if self.args["use_multimer"]:
                        self.args["use_multimer"] = False
                        self.reset()

            if pdb is not None:
                self.set_pdb(pdb, chain)
            if seq is not None:
                self.set_seq(seq)

            # set template sequence
            self.model._inputs["batch"]["aatype"] = self.wt

            # set other options
            self.model.set_opt(template={"rm_ic": rm_ic}, num_recycles=recycles)
            self.model._inputs["rm_template"][:] = not input_template
            self.model._inputs["rm_template_sc"][:] = rm_sc
            self.model._inputs["rm_template_seq"][:] = rm_seq

            # "manual" recycles using templates
            ini_atoms = self.model._inputs["batch"]["all_atom_positions"].copy()
            for i in range(iterations):
                # ColabDesign's predict() expects models=None or a list of indices [0,1,2,...]
                # The model_name was already set during initialization, so we just use None here
                self.model.predict(models=None, verbose=False)
                if i < iterations - 1:
                    self.model._inputs["batch"]["all_atom_positions"] = self.model.aux[
                        "atom_positions"
                    ]
                else:
                    self.model._inputs["batch"]["all_atom_positions"] = ini_atoms

            score = self._get_score()
            if extras is not None:
                score.update(extras)

            if output_pdb is not None:
                self.model.save_pdb(output_pdb)

            if verbose:
                print_list = [
                    "tm_i",
                    "tm_o",
                    "tm_io",
                    "composite",
                    "ptm",
                    "i_ptm",
                    "plddt",
                    "fitness",
                    "id",
                ]

                def print_score(k):
                    return (
                        f"{k} {score[k]:.4f}" if isinstance(score[k], float) else f"{k} {score[k]}"
                    )

                print(*[print_score(k) for k in print_list if k in score])

            return score


@app.function(
    image=image,
    gpu=GPU,
    timeout=int(TIMEOUT),
    secrets=[modal.Secret.from_name("cloudsql-credentials")],
)
def run_af2rank(
    pdb_str: str,
    pdb_name: str | None = None,
    chains: str = "A",
    model_name: str | None = None,
    num_recycles: int = 1,
    num_iterations: int = 1,
    mask_sequence: bool = False,
    mask_sidechains: bool = False,
    mask_interchain: bool = False,
    upload_to_gcs: bool = False,
    gcs_bucket: str | None = None,
    run_id: str | None = None,
):
    """Modal function for running AF2Rank.

    Args:
        pdb_str (str): The content of the PDB file as a string.
        pdb_name (str | None): Optional name for the PDB file (used for output naming).
        chains (str): Comma-separated string of chain IDs to use (e.g., "A" or "A,B").
        model_name (str | None): Name of the AlphaFold2 model to use (e.g., "model_1", "model_2"). If None, defaults to "model_1".
        num_recycles (int): Number of recycles for the AlphaFold2 model.
        num_iterations (int): Number of "manual" recycles using templates.
        mask_sequence (bool): Whether to remove the sequence from the template.
        mask_sidechains (bool): Whether to remove sidechain information from the template.
        mask_interchain (bool): Whether to remove interchain information from the template (for multimers).
        upload_to_gcs (bool): Whether to upload results to GCS.
        gcs_bucket (str | None): GCS bucket name (e.g., "dev-services").
        run_id (str | None): Unique run identifier for organizing outputs.

    Returns:
        dict: Structured output with exit_code, output_files, message, and metadata.
    """
    import json
    import tempfile

    if pdb_name is None:
        pdb_name = "af2rank.pdb"

    # AF2Rank uses standard AlphaFold model names (model_1, model_2, etc.)
    # not PTM variants (model_1_ptm)
    if model_name is None:
        model_name = "model_1"

    Path(in_pdb := "/tmp/in_af2rank/input.pdb").parent.mkdir(parents=True, exist_ok=True)
    Path(in_pdb).write_text(pdb_str)

    Path(out_dir := "/tmp/out_af2rank").mkdir(parents=True, exist_ok=True)

    SETTINGS = {
        "rm_seq": mask_sequence,
        "rm_sc": mask_sidechains,
        "rm_ic": mask_interchain,
        "recycles": num_recycles,
        "iterations": num_iterations,
        "model_name": model_name,
    }
    print("settings:", SETTINGS)
    af = af2rank(in_pdb, chains, model_name=SETTINGS["model_name"])
    score = af.predict(pdb=in_pdb, **SETTINGS, extras={"id": in_pdb})

    results = SETTINGS | {"score": score, "chains": chains}
    open(Path(out_dir) / "results.json", "w").write(json.dumps(results))
    open(Path(out_dir) / f"{Path(pdb_name).stem}_af2rank.pdb", "w").write(pdb_str)

    # Collect output files
    output_files_list = []
    for out_file in Path(out_dir).glob("**/*"):
        if out_file.is_file():
            suffix = out_file.suffix.lstrip(".")
            artifact_type = suffix if suffix else "file"

            output_files_list.append(
                {
                    "path": str(out_file.absolute()),
                    "filename": out_file.name,
                    "artifact_type": artifact_type,
                    "size": out_file.stat().st_size,
                    "metadata": {
                        "model_name": model_name,
                        "chains": chains,
                        "num_recycles": num_recycles,
                        "num_iterations": num_iterations,
                    },
                }
            )

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

        tool_name = "af2rank"
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
            "message": f"AF2Rank completed: {len(output_files)} files uploaded to GCS",
            "metadata": {
                "run_id": run_id,
                "tool_name": tool_name,
                "model_name": model_name,
                "chains": chains,
                "score": score,
            },
        }

    else:
        # Return without GCS upload (for local testing)
        return {
            "exit_code": 0,
            "output_files": output_files_list,
            "message": f"AF2Rank completed: {len(output_files_list)} files generated",
            "metadata": {
                "model_name": model_name,
                "chains": chains,
                "score": score,
            },
        }


@app.local_entrypoint()
def main(
    input_pdb: str,
    chains: str = "A",
    model_name: str | None = None,
    num_recycles: int = 1,
    num_iterations: int = 1,
    mask_sequence: bool = False,
    mask_sidechains: bool = False,
    mask_interchain: bool = False,
    out_dir: str = "./out/af2rank",
    run_name: str | None = None,
) -> None:
    """Local entrypoint for the Modal app to run AF2Rank.

    This function handles fetching the PDB data, invoking the remote Modal function
    `run_af2rank`, and saving the results locally.

    Args:
        input_pdb (str): Path to the input PDB file.
        chains (str): Comma-separated string of chain IDs to use (e.g., "A" or "A,B").
        model_name (str | None): Name of the AlphaFold2 model (e.g., "model_1", "model_2"). If None, defaults to "model_1".
        num_recycles (int): Number of recycles for the AlphaFold2 model.
        num_iterations (int): Number of "manual" recycles using templates.
        mask_sequence (bool): Whether to remove the sequence from the template.
        mask_sidechains (bool): Whether to remove sidechain information from the template.
        mask_interchain (bool): Whether to remove interchain information from the template (for multimers).
        out_dir (str): Directory to save the output files.
        run_name (str | None): Optional name for the run, used to create a subdirectory in `out_dir`.
                               If None, a timestamp-based name is used.

    Returns:
        None
    """
    from datetime import datetime

    pdb_str = open(input_pdb).read()

    # AF2Rank uses: model_1, model_2, model_3, model_4, model_5
    # Or for multimers: model_1_multimer_v3, model_2_multimer_v3, etc.
    if model_name is None:
        model_name = "model_1"

    print("Running AF2Rank...")
    print(f"  Input: {input_pdb}")
    print(f"  Chains: {chains}")
    print(f"  Model: {model_name}")

    result = run_af2rank.remote(
        pdb_str=pdb_str,
        pdb_name=Path(input_pdb).name,
        chains=chains,
        model_name=model_name,
        num_recycles=num_recycles,
        num_iterations=num_iterations,
        mask_sequence=mask_sequence,
        mask_sidechains=mask_sidechains,
        mask_interchain=mask_interchain,
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

        if local_path.exists():
            import shutil

            shutil.copy2(local_path, dest_path)
            print(f"  Saved: {dest_path}")

    print(f"\nOutput directory: {out_dir_full}")
    print(f"Generated {len(result['output_files'])} files")

    # Print scores if available
    if "metadata" in result and "score" in result["metadata"]:
        score = result["metadata"]["score"]
        print("\nScores:")
        for key in ["composite", "ptm", "plddt", "tm_io"]:
            if key in score:
                print(f"  {key}: {score[key]:.4f}")
