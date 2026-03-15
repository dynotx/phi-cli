import argparse
import json

from rich import box as rich_box
from rich.table import Table

from phi.api import _request, _resolve_identity
from phi.config import POLL_INTERVAL, _save_state
from phi.display import _C_BLUE, _C_ROSE, _C_SAND, _die, _print_dataset_ready, console
from phi.polling import _ingest_poll
from phi.upload import (
    _collect_files,
    _create_ingest_session,
    _request_signed_urls,
    _upload_all_parallel,
)


def cmd_upload(args: argparse.Namespace) -> None:
    _resolve_identity()

    if getattr(args, "gcs", None):
        console.print("  External cloud storage import is not yet available.")
        console.print("  The backend import worker is planned — see backend-api-gaps.md §9.")
        console.print(f"  When available: phi upload --gcs {args.gcs}")
        return

    files = _collect_files(args)
    console.print(f"[bold {_C_SAND}]✓[/] Found {len(files)} file(s) to upload")

    session_id = _create_ingest_session(files, args)
    url_map = _request_signed_urls(session_id, files)

    failures = _upload_all_parallel(files, url_map)
    if failures:
        console.print(f"\n[bold {_C_ROSE}]error:[/] {len(failures)} upload(s) failed:")
        for msg in failures[:10]:
            console.print(f"  [dim]{msg}[/]")
        _die("Upload incomplete — fix errors and retry.")

    console.print(f"  [bold {_C_SAND}]✓[/] All {len(files)} file(s) uploaded successfully")
    console.print("  Finalizing ingest session …")
    _request("POST", f"/ingest_sessions/{session_id}/finalize", {})

    if args.wait:
        console.print(
            f"\n[dim]Ingesting and validating files (polling every {POLL_INTERVAL}s) …[/]"
        )
        result = _ingest_poll(session_id)
        status = result.get("status")
        if status == "READY":
            dataset_id = result.get("dataset_id")
            if dataset_id:
                _save_state({"dataset_id": dataset_id})
            _print_dataset_ready(dataset_id, result.get("artifact_count", len(files)))
        else:
            _die(f"Ingest failed ({status}): {result.get('error', 'unknown error')}")
    else:
        console.print(f"\n[bold {_C_SAND}]✓[/] Session finalized — ingestion running in background")
        console.print(f"  [dim]session_id[/]: {session_id}")
        console.print(f"  Check status: [{_C_BLUE}]phi ingest-session {session_id}[/]")


def cmd_ingest_session(args: argparse.Namespace) -> None:
    result = _request("GET", f"/ingest_sessions/{args.session_id}")
    if args.json:
        print(json.dumps(result, indent=2))
        return
    console.print(f"[dim]session_id    [/] : {result.get('session_id') or result.get('id', '?')}")
    console.print(f"[dim]status        [/] : {result.get('status', '?')}")
    console.print(f"[dim]expected_files[/] : {result.get('expected_files', '?')}")
    console.print(f"[dim]uploaded_files[/] : {result.get('uploaded_files', '?')}")
    console.print(f"[dim]artifact_count[/] : {result.get('artifact_count', '?')}")
    if result.get("dataset_id"):
        console.print(f"[dim]dataset_id    [/] : {result.get('dataset_id')}")
        console.print(f"\n  [{_C_BLUE}]phi esmfold     --dataset-id {result.get('dataset_id')}[/]")
        console.print(f"  [{_C_BLUE}]phi alphafold   --dataset-id {result.get('dataset_id')}[/]")


def cmd_datasets(args: argparse.Namespace) -> None:
    params = f"?page_size={args.limit}"
    result = _request("GET", f"/datasets{params}")
    if args.json:
        print(json.dumps(result, indent=2))
        return
    datasets = result.get("datasets", [])
    if not datasets:
        console.print("[dim]No datasets found.[/]")
        return

    table = Table(box=rich_box.SIMPLE_HEAVY, show_header=True, header_style="bold dim")
    table.add_column("DATASET ID", style=_C_BLUE, no_wrap=True)
    table.add_column("ARTIFACTS", justify="right")
    table.add_column("STATUS")
    table.add_column("CREATED", style="dim")

    for d in datasets:
        status = d.get("status", "?")
        s_color = _C_SAND if status == "READY" else (_C_ROSE if status == "FAILED" else "dim")
        table.add_row(
            d.get("dataset_id", "?"),
            str(d.get("artifact_count", "?")),
            f"[{s_color}]{status}[/]",
            str(d.get("created_at", "?"))[:19],
        )

    console.print(table)
    total = result.get("total") or result.get("total_count") or len(datasets)
    console.print(f"[dim]{total} total dataset(s)[/]")


def cmd_dataset(args: argparse.Namespace) -> None:
    result = _request("GET", f"/datasets/{args.dataset_id}")
    if args.json:
        print(json.dumps(result, indent=2))
        return
    console.print(f"[dim]dataset_id    [/] : [{_C_BLUE}]{result.get('dataset_id')}[/]")
    console.print(f"[dim]status        [/] : {result.get('status')}")
    console.print(f"[dim]artifact_count[/] : {result.get('artifact_count')}")
    console.print(f"[dim]version       [/] : {result.get('version', 1)}")
    console.print(f"[dim]created_at    [/] : {str(result.get('created_at', ''))[:19]}")
    manifest = result.get("manifest_uri")
    if manifest:
        console.print(f"[dim]manifest_uri  [/] : {manifest}")
    files = result.get("files") or result.get("sample_artifacts") or []
    if files:
        console.print(f"\n[bold]Sample files[/] (first {len(files)}):")
        for f in files:
            fname = f.get("filename") or f.get("source_filename", "?")
            size = f.get("size_bytes") or f.get("size", 0)
            console.print(f"  [{_C_BLUE}]{fname:<40}[/]  [dim]{size:>10} B[/]")
    console.print("\n[bold]Run a job:[/]")
    console.print(f"  [{_C_SAND}]phi esmfold   --dataset-id {result.get('dataset_id')}[/]")
    console.print(f"  [{_C_SAND}]phi alphafold --dataset-id {result.get('dataset_id')}[/]")


def cmd_use(args: argparse.Namespace) -> None:
    _save_state({"dataset_id": args.dataset_id})
    console.print(
        f"[bold {_C_SAND}]✓[/] Active dataset set to [{_C_BLUE}]{args.dataset_id}[/]\n"
        f"  [dim]Saved to .phi-state.json — all commands will use this dataset by default.[/]"
    )
