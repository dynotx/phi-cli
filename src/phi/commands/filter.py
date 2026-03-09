import argparse

from phi.api import _request, _require_key, _submit
from phi.config import _FILTER_PRESETS, _resolve_dataset_id, _save_state
from phi.display import _C_SAND, _print_filter_done, _print_status, _print_submission, console
from phi.download import _download_job, _load_scores_csv
from phi.polling import _poll
from phi.types import PhiApiError


def cmd_filter(args: argparse.Namespace) -> None:
    dataset_id = _resolve_dataset_id(args)

    preset_name: str | None = getattr(args, "preset", None)
    base: dict[str, float] = dict(_FILTER_PRESETS[preset_name]) if preset_name else {}

    def _resolve_threshold(name: str) -> float:
        explicit: float | None = getattr(args, name, None)
        if explicit is not None:
            return explicit
        return base.get(name, _FILTER_PRESETS["default"][name])

    if preset_name:
        p = base
        console.print(
            f"[dim]Using [{_C_SAND}]{preset_name}[/{_C_SAND}] filter preset  "
            f"pLDDT≥{p['plddt_threshold']}  "
            f"pTM≥{p['ptm_threshold']}  "
            f"ipTM≥{p['iptm_threshold']}  "
            f"iPAE≤{p['ipae_threshold']}  "
            f"RMSD≤{p['rmsd_threshold']}Å  "
            f"[italic](override with explicit flags)[/italic][/]"
        )

    params: dict = {
        "num_sequences": args.num_sequences,
        "plddt_threshold": _resolve_threshold("plddt_threshold"),
        "iptm_threshold": _resolve_threshold("iptm_threshold"),
        "ipae_threshold": _resolve_threshold("ipae_threshold"),
        "ptm_threshold": _resolve_threshold("ptm_threshold"),
        "rmsd_threshold": _resolve_threshold("rmsd_threshold"),
        "num_recycles": args.num_recycles,
    }
    msa_tool: str = getattr(args, "msa_tool", "mmseqs2")
    params["msa_tool"] = msa_tool
    if msa_tool == "single_sequence":
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
            run_id_final = final.get("run_id")
            results: dict = {}
            if run_id_final:
                try:
                    results = _request("GET", f"/runs/{run_id_final}/results")
                    final["_results"] = results
                except PhiApiError:
                    pass
            csv_content = _load_scores_csv(results.get("artifact_files") or [])
            _print_filter_done(job_id, final, thresholds=params, csv_content=csv_content)
        if args.out:
            _download_job(final, args.out, all_files=getattr(args, "all", False))
