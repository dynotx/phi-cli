import argparse
import csv
import io
import json
import urllib.request

from rich import box as rich_box
from rich.table import Table

from phi.api import _request, _status
from phi.config import _base_url, _resolve_job_id, _ssl_context
from phi.display import _C_BLUE, _C_SAND, _STATUS_COLOR, _die, _print_status, console
from phi.download import _download_job
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
    from phi.api import _api_key

    url = f"{_base_url()}/api/v1/jobs/{args.job_id}/logs/stream"
    console.print(f"Streaming logs from [{_C_BLUE}]{url}[/]")
    console.print("(Note: EventSource auth requires token as query param on this endpoint)")
    console.print(f"  [dim]curl -N '{url}?x_api_key={_api_key()[:8]}...'[/]")


def cmd_cancel(args: argparse.Namespace) -> None:
    result = _request("DELETE", f"/jobs/{args.job_id}")
    console.print(f"[bold {_C_SAND}]✓[/] Cancel requested: {result.get('message', 'ok')}")


def cmd_scores(args: argparse.Namespace) -> None:
    args.job_id = _resolve_job_id(args)
    s = _status(args.job_id)
    if s.get("status") not in ("completed", "failed"):
        _die(f"Job is '{s.get('status')}' — scores are only available after completion")

    run_id = s.get("run_id")
    if not run_id:
        _die("No run_id found for this job")

    try:
        results = _request("GET", f"/runs/{run_id}/results")
    except PhiApiError as e:
        _die(f"Could not fetch results: {e}")

    workflow_artifacts = results.get("workflow_artifacts", {})
    artifact_files = results.get("artifact_files", [])

    scores_content: str | None = None
    scores_source: str = ""

    for key, val in workflow_artifacts.items():
        if isinstance(val, str) and key in ("scores_csv", "metrics_csv", "scores"):
            scores_content = val
            scores_source = f"workflow artifact '{key}'"
            break

    scores_artifact: dict | None = None
    if not scores_content:
        for af in artifact_files:
            name = af.get("name") or af.get("filename") or ""
            if name.endswith((".csv", ".parquet")) and any(
                kw in name.lower() for kw in ("score", "metric", "report")
            ):
                scores_artifact = af
                scores_source = f"artifact file '{name}'"
                break

    if not scores_content and scores_artifact:
        artifact_id = scores_artifact.get("artifact_id")
        url = scores_artifact.get("download_url") or scores_artifact.get("url")
        if not url and artifact_id:
            try:
                dl_resp = _request("GET", f"/artifacts/{artifact_id}/download")
                url = dl_resp.get("download_url")
            except PhiApiError:
                pass
        if url and not url.startswith("gs://"):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "phi-cli/1.0"}, method="GET"
                )
                with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
                    scores_content = resp.read().decode("utf-8", errors="replace")
            except Exception as exc:
                console.print(f"  [yellow]⚠ Could not fetch scores file: {exc}[/]")

    if not scores_content:
        console.print(f"[dim]No scores/metrics found for job {args.job_id}.[/]")
        console.print("[dim]The pipeline may not have produced a report yet.[/]")
        return

    console.print(
        f"\n[bold {_C_SAND}]Scores[/] from {scores_source}  [dim](job {args.job_id[:8]}…)[/]\n"
    )

    try:
        reader = csv.DictReader(io.StringIO(scores_content))
        rows = list(reader)
        if not rows:
            console.print("[dim]Score file is empty.[/]")
        else:
            fieldnames = list(rows[0].keys()) if rows else []
            sort_col = next(
                (c for c in ("iptm", "complex_iptm", "binder_plddt", "plddt") if c in fieldnames),
                None,
            )
            if sort_col:
                rows.sort(key=lambda r: float(r.get(sort_col, 0) or 0), reverse=True)

            display_rows = rows[: args.top]
            table = Table(
                box=rich_box.SIMPLE_HEAVY,
                show_header=True,
                header_style=f"bold {_C_SAND}",
            )
            for col in fieldnames:
                table.add_column(col, no_wrap=True)
            for row in display_rows:
                table.add_row(*[row.get(c, "") for c in fieldnames])
            console.print(table)
            if len(rows) > args.top:
                console.print(
                    f"[dim]Showing top {args.top} of {len(rows)} candidates"
                    f"{f' (sorted by {sort_col})' if sort_col else ''}.[/]"
                )
    except Exception as exc:
        console.print(scores_content[:2000])
        if len(scores_content) > 2000:
            console.print(f"[dim]… ({len(scores_content)} chars total)[/]")
        console.print(f"[dim]Note: Could not parse as CSV table: {exc}[/]")

    if args.out:
        from pathlib import Path

        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(scores_content)
        console.print(f"\n[{_C_SAND}]✓[/] Saved → [bold]{dest}[/bold]")


def cmd_download(args: argparse.Namespace) -> None:
    args.job_id = _resolve_job_id(args)
    s = _status(args.job_id)
    if s.get("status") != "completed":
        _die(f"Job is '{s.get('status')}' — can only download completed jobs")
    run_id = s.get("run_id")
    if run_id:
        try:
            results = _request("GET", f"/runs/{run_id}/results")
            s["_results"] = results
        except PhiApiError:
            pass
    _download_job(s, args.out, all_files=getattr(args, "all", False))
