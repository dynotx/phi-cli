import argparse
import json
from pathlib import Path

from rich import box as rich_box
from rich.table import Table

from phi.api import _request, _status
from phi.config import _base_url, _require_api_key, _resolve_job_id
from phi.display import (
    _C_BLUE,
    _C_SAND,
    _STATUS_COLOR,
    _die,
    _print_status,
    _render_per_model_table,
    _render_scores_table,
    console,
)
from phi.download import _download_job, _load_scores_csv
from phi.types import PhiApiError


def cmd_status(args: argparse.Namespace) -> None:
    s = _status(args.job_id)
    if args.json:
        print(json.dumps(s, indent=2))
    else:
        _print_status(s)


def cmd_jobs(args: argparse.Namespace) -> None:
    params: dict = {"page_size": args.limit}
    if args.status:
        params["status"] = args.status
    if args.job_type:
        params["job_type"] = args.job_type
    query = "&".join(f"{k}={v}" for k, v in params.items())
    result = _request("GET", f"/jobs/?{query}")
    if args.json:
        print(json.dumps(result, indent=2))
        return
    jobs = result.get("jobs", [])
    if not jobs:
        console.print("[dim]No jobs found.[/]")
        return

    table = Table(box=rich_box.SIMPLE_HEAVY, show_header=True, header_style="bold dim")
    table.add_column("JOB ID", style=f"dim {_C_BLUE}", no_wrap=True)
    table.add_column("TYPE", style="bold")
    table.add_column("STATUS", no_wrap=True)
    table.add_column("CREATED", style="dim")

    for j in jobs:
        status = j.get("status", "?")
        color = _STATUS_COLOR.get(status, "")
        styled_status = f"[{color}]{status}[/]" if color else status
        table.add_row(
            j.get("job_id", "?"),
            j.get("job_type", "?"),
            styled_status,
            (j.get("created_at") or "?")[:19],
        )

    console.print(table)
    total = result.get("total_count", len(jobs))
    running = result.get("total_running", 0)
    pending = result.get("total_pending", 0)
    console.print(f"[dim]{total} total  ({running} running, {pending} pending)[/]")


def cmd_logs(args: argparse.Namespace) -> None:
    url = f"{_base_url()}/v1/phi/jobs/{args.job_id}/logs/stream"
    console.print(f"Streaming logs from [{_C_BLUE}]{url}[/]")
    console.print("(Note: EventSource auth requires token as query param on this endpoint)")
    console.print(f"  [dim]curl -N '{url}?x_api_key={_require_api_key()[:8]}...'[/]")


def cmd_cancel(args: argparse.Namespace) -> None:
    result = _request("DELETE", f"/jobs/{args.job_id}")
    console.print(f"[bold {_C_SAND}]✓[/] Cancel requested: {result.get('message', 'ok')}")


def cmd_scores(args: argparse.Namespace) -> None:
    args.job_id = _resolve_job_id(args)
    s = _status(args.job_id)
    if s.get("status") not in ("completed", "failed"):
        _die(f"Job is '{s.get('status')}' — scores are only available after completion")

    try:
        results = _request("GET", f"/jobs/{args.job_id}/scores")
    except PhiApiError as e:
        _die(f"Could not fetch results: {e}")

    scores_content = _load_scores_csv(results.get("artifact_files") or [])

    if not scores_content:
        console.print(f"[dim]No scores/metrics found for job {args.job_id}.[/]")
        console.print("[dim]The pipeline may not have produced a report yet.[/]")
        return

    console.print(f"\n[bold {_C_SAND}]Scores[/]  [dim](job {args.job_id[:8]}…)[/]\n")
    params = s.get("params") or {}
    threshold_keys = {"plddt_threshold", "ptm_threshold", "iptm_threshold", "ipae_threshold", "rmsd_threshold"}
    thresholds = {k: v for k, v in params.items() if k in threshold_keys} or None
    _render_scores_table(scores_content, thresholds)
    _render_per_model_table(scores_content)

    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(scores_content)
        console.print(f"\n[{_C_SAND}]✓[/] Saved → [bold]{dest}[/bold]")


def cmd_download(args: argparse.Namespace) -> None:
    args.job_id = _resolve_job_id(args)
    s = _status(args.job_id)
    if s.get("status") != "completed":
        _die(f"Job is '{s.get('status')}' — can only download completed jobs")
    try:
        results = _request("GET", f"/jobs/{args.job_id}/scores")
        s["_results"] = results
    except PhiApiError:
        pass
    _download_job(s, args.out, all_files=getattr(args, "all", False))
