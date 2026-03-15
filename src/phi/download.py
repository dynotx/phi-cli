import argparse
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import cast

from phi.api import _request
from phi.config import _ssl_context
from phi.display import (
    _C_BLUE,
    _C_ROSE,
    _C_SAND,
    _die,
    _fmt_size,
    _make_upload_progress,
    _print_download_result,
    console,
    err_console,
)
from phi.types import PhiApiError

_DOWNLOAD_KEY_TYPES = {"pdb", "csv", "colabfold_scores", "af2m_scores", "json"}

_DOWNLOAD_SUBDIR: dict[str, str] = {
    "pdb": "structures",
    "csv": "scores",
    "colabfold_scores": "scores/raw",
    "af2m_scores": "scores/raw",
    "json": "scores/raw",
}


def _read_fasta(args: argparse.Namespace) -> str:
    if args.fasta_str:
        return str(args.fasta_str)
    if args.fasta:
        return Path(args.fasta).read_text()
    _die("Provide --fasta FILE or --fasta-str '>name\\nSEQUENCE'")


def _fetch_url_to_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "phi-cli/1.0"}, method="GET")
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:
        dest.write_bytes(resp.read())


def _fetch_url_to_str(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "phi-cli/1.0"}, method="GET")
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:
        return cast(bytes, resp.read()).decode("utf-8")


def _get_artifact_url(af: dict) -> str | None:
    url: object = af.get("download_url") or af.get("url")
    if not url:
        artifact_id = af.get("artifact_id")
        if artifact_id:
            try:
                dl_resp = _request("GET", f"/artifacts/{artifact_id}/download")
                url = dl_resp.get("download_url")
            except PhiApiError:
                pass
    if url and isinstance(url, str):
        return str(url)
    return None


def _load_scores_csv(artifact_files: list[dict]) -> str | None:
    scores_artifact: dict | None = None
    for af in artifact_files:
        name = af.get("name") or af.get("filename") or ""
        atype = af.get("artifact_type") or ""
        if atype == "csv" or name.endswith(".csv"):
            scores_artifact = af
            break

    if not scores_artifact:
        return None

    url = _get_artifact_url(scores_artifact)
    if not url:
        err_console.print("  [dim]Scores table unavailable: no download URL for scores.csv[/]")
        return None

    try:
        return _fetch_url_to_str(url)
    except Exception as exc:
        err_console.print(f"  [dim]Could not fetch scores table: {exc}[/]")
        return None


def _categorize_artifacts(
    artifact_files: list[dict], all_files: bool
) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for af in artifact_files:
        atype = af.get("artifact_type") or "file"
        groups[atype].append(af)

    selected_types = set(groups.keys()) if all_files else (_DOWNLOAD_KEY_TYPES & set(groups.keys()))

    to_download = [
        af for af in artifact_files if (af.get("artifact_type") or "file") in selected_types
    ]
    to_skip = [
        af for af in artifact_files if (af.get("artifact_type") or "file") not in selected_types
    ]
    return to_download, to_skip


def _print_artifact_summary(artifact_files: list[dict], selected_types: set[str]) -> None:
    from rich import box as rich_box
    from rich.table import Table

    groups: dict[str, list[dict]] = defaultdict(list)
    for af in artifact_files:
        groups[af.get("artifact_type") or "file"].append(af)

    table = Table(
        box=rich_box.SIMPLE,
        show_header=True,
        header_style="bold dim",
        show_edge=False,
        padding=(0, 1),
    )
    table.add_column("Type", style=f"{_C_BLUE}", no_wrap=True)
    table.add_column("Files", justify="right", style="dim")
    table.add_column("Size", justify="right", style="dim")
    table.add_column("", style="dim")

    for atype in sorted(groups.keys(), key=lambda t: (t not in selected_types, t)):
        items = groups[atype]
        total_bytes = sum(af.get("size_bytes") or 0 for af in items)
        will_dl = atype in selected_types
        action = (
            f"→ {_DOWNLOAD_SUBDIR.get(atype, '.')}/" if will_dl else "[dim]skip (use --all)[/dim]"
        )
        style = "" if will_dl else "dim"
        table.add_row(
            f"[{style}]{atype}[/{style}]" if style else atype,
            str(len(items)),
            _fmt_size(total_bytes),
            action,
        )
    console.print(table)


def _download_artifacts(to_download: list[dict], out_dir: Path) -> tuple[int, int, dict[str, int]]:
    downloaded, errors = 0, 0
    counts_by_subdir: dict[str, int] = defaultdict(int)

    with _make_upload_progress() as progress:
        task = progress.add_task(
            f"[{_C_BLUE}]Downloading {len(to_download)} file(s)…",
            total=len(to_download),
        )
        for af in to_download:
            name = af.get("name") or af.get("filename") or af.get("artifact_id", "artifact")
            atype = af.get("artifact_type") or "file"
            subdir = _DOWNLOAD_SUBDIR.get(atype, ".")
            dest_dir = out_dir / subdir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(name).name

            url = _get_artifact_url(af)
            if url:
                try:
                    _fetch_url_to_file(url, dest)
                    downloaded += 1
                    counts_by_subdir[subdir] += 1
                except Exception as exc:
                    console.print(f"  [{_C_ROSE}]⚠ {Path(name).name}: {exc}[/]")
                    errors += 1
            else:
                console.print(
                    f"  [{_C_ROSE}]⚠ {Path(name).name}: server could not generate a download URL[/]"
                )
                errors += 1
            progress.advance(task)

    return downloaded, errors, dict(counts_by_subdir)


def _write_manifest(out_dir: Path, data: object) -> None:
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(data, indent=2))
    console.print(f"  [dim]manifest → {manifest}[/]")


def _download_artifact_files(
    artifact_files: list[dict], out: Path, out_dir: str, all_files: bool
) -> None:
    to_download, to_skip = _categorize_artifacts(artifact_files, all_files)
    selected_types = {af.get("artifact_type") or "file" for af in to_download}
    _print_artifact_summary(artifact_files, selected_types)

    if not all_files and to_skip:
        console.print(
            f"  [dim]{len(to_skip)} file(s) skipped"
            f" (msa/zip/scripts) — pass [bold]--all[/bold] to include[/]"
        )

    if not to_download:
        console.print("  [dim]Nothing to download.[/]")
    else:
        downloaded, errors, counts_by_subdir = _download_artifacts(to_download, out)
        _print_download_result(out_dir, downloaded, errors, counts_by_subdir)

    _write_manifest(out, artifact_files)


def _download_workflow_artifacts(workflow_artifacts: dict, out: Path) -> None:
    console.print("\n[bold]Workflow artifacts:[/]")
    for key, val in workflow_artifacts.items():
        console.print(f"  [{_C_BLUE}]{key}[/]: {val}")
    _write_manifest(out, workflow_artifacts)


def _download_output_files(output_files: list[dict], out: Path) -> None:
    console.print(f"\n[bold]Output files[/] ({len(output_files)}) — stored in cloud:")
    for f in output_files:
        name = f.get("name", "")
        val = f.get("value", "")
        if isinstance(val, list):
            for v in val:
                console.print(f"  [{_C_BLUE}]{v}[/]")
        else:
            console.print(f"  [{_C_BLUE}]{name}[/]: {val}")
    _write_manifest(out, output_files)


def _download_job(status: dict, out_dir: str, all_files: bool = False) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = status.get("_results") or {}
    artifact_files: list[dict] = results.get("artifact_files") or []
    workflow_artifacts: dict = results.get("workflow_artifacts") or {}
    output_files: list[dict] = status.get("output_files") or []

    if artifact_files:
        _download_artifact_files(artifact_files, out, out_dir, all_files)
    elif workflow_artifacts:
        _download_workflow_artifacts(workflow_artifacts, out)
    elif output_files:
        _download_output_files(output_files, out)
    else:
        console.print(f"  [dim]No output files found for job {status.get('job_id')}[/]")
        console.print(f"  [dim]run_id: {status.get('run_id')}[/]")
        console.print(f"  Check: [{_C_SAND}]phi status {status.get('job_id')} --json[/]")
