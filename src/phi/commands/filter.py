import argparse

from phi.api import _request, _require_key, _submit
from phi.config import _FILTER_PRESETS, _resolve_dataset_id, _save_state
from phi.display import _C_SAND, _print_filter_done, _print_status, _print_submission, console
from phi.download import _download_job, _load_scores_csv
from phi.polling import _poll
from phi.types import PhiApiError

_THRESHOLD_KEYS = [
    "plddt_threshold",
    "iptm_threshold",
    "ipae_threshold",
    "ptm_threshold",
    "rmsd_threshold",
]


def _build_filter_params(args: argparse.Namespace, preset_name: str | None) -> dict:
    base = dict(_FILTER_PRESETS[preset_name]) if preset_name else {}
    defaults = _FILTER_PRESETS["default"]

    params: dict = {
        key: getattr(args, key) if getattr(args, key) is not None else base.get(key, defaults[key])
        for key in _THRESHOLD_KEYS
    }
    params["num_sequences"] = args.num_sequences
    params["num_recycles"] = args.num_recycles
    params["msa_tool"] = args.msa_tool
    return params


def cmd_filter(args: argparse.Namespace) -> None:
    dataset_id = _resolve_dataset_id(args)
    preset_name: str | None = args.preset

    if preset_name:
        p = _FILTER_PRESETS[preset_name]
        console.print(
            f"[dim]Using [{_C_SAND}]{preset_name}[/{_C_SAND}] filter preset  "
            f"pLDDT≥{p['plddt_threshold']}  "
            f"pTM≥{p['ptm_threshold']}  "
            f"ipTM≥{p['iptm_threshold']}  "
            f"iPAE≤{p['ipae_threshold']}Å  "
            f"RMSD≤{p['rmsd_threshold']}Å  "
            f"[italic](override with explicit flags)[/italic][/]"
        )

    params = _build_filter_params(args, preset_name)

    if params["msa_tool"] == "single_sequence":
        console.print(
            "  [dim]AF2 running in single-sequence mode (no MSA) — faster, "
            "better-calibrated for novel designed binders[/]"
        )

    result = _submit("design_pipeline", params, run_id=None, dataset_id=dataset_id)
    job_id = _require_key(result, "job_id", "POST /jobs (filter)")
    _save_state({"last_job_id": job_id})
    _print_submission(result)

    if args.wait or args.out:
        final = _poll(job_id)
        _print_status(final)
        if final.get("status") == "completed":
            results: dict = {}
            try:
                results = _request("GET", f"/jobs/{job_id}/scores")
                final["_results"] = results
            except PhiApiError:
                pass
            csv_content = _load_scores_csv(results.get("artifact_files") or [])
            _print_filter_done(job_id, final, thresholds=params, csv_content=csv_content)
        if args.out:
            _download_job(final, args.out, all_files=args.all)
