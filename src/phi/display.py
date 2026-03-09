import csv
import io
import sys
from datetime import datetime
from typing import NoReturn

from rich import box as rich_box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()
err_console = Console(stderr=True)

_C_SAND = "#B5A58F"
_C_ROSE = "#D4A5B8"
_C_BLUE = "#8FA5B8"

_STATUS_COLOR = {
    "completed": _C_SAND,
    "failed": _C_ROSE,
    "cancelled": _C_ROSE,
    "running": _C_BLUE,
    "pending": "dim",
    "submitted": "dim",
}
_STATUS_ICON = {
    "completed": "✓",
    "failed": "✗",
    "cancelled": "⊘",
}


def _die(msg: str) -> NoReturn:
    err_console.print(f"[bold {_C_ROSE}]error:[/] {msg}")
    sys.exit(1)


def _fmt_size(total_bytes: int) -> str:
    if total_bytes >= 1_048_576:
        return f"{total_bytes / 1_048_576:.1f} MB"
    if total_bytes >= 1024:
        return f"{total_bytes / 1024:.0f} KB"
    return f"{total_bytes} B"


def _duration_str(started_at: str, completed_at: str) -> str:
    try:
        t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        return f"{int((t1 - t0).total_seconds())}s"
    except Exception:
        return ""


def _print_submission(result: dict) -> None:
    console.print(f"\n[bold {_C_SAND}]✓[/] Job submitted")
    console.print(f"  [dim]job_id[/]  {result.get('job_id')}")
    console.print(f"  [dim]run_id[/]  {result.get('run_id')}")
    console.print(f"  [dim]status[/]  [dim]{result.get('status')}[/]")
    if result.get("message"):
        console.print(f"  [dim]message[/] {result['message']}")


def _print_status(s: dict) -> None:
    status = s.get("status", "?")
    icon = _STATUS_ICON.get(status, "·")
    color = _STATUS_COLOR.get(status, "white")
    job_id = s.get("job_id", "?")

    duration = ""
    if s.get("started_at") and s.get("completed_at"):
        d = _duration_str(s["started_at"], s["completed_at"])
        if d:
            duration = f"  [dim]{d}[/]"

    console.print(f"\n[{color}]{icon}[/] [bold]{job_id}[/]  [[{color}]{status}[/]]{duration}")

    if s.get("error"):
        console.print(f"  [{_C_ROSE}]error[/] : {s['error']}")

    p = s.get("progress") or {}
    if p.get("current_step"):
        console.print(f"  [dim]step[/]  : {p['current_step']}")

    if s.get("started_at") and s.get("completed_at"):
        t0 = s["started_at"][:19].replace("T", " ")
        t1 = s["completed_at"][:19].replace("T", " ")
        console.print(f"  [dim]timing[/] : {t0} → {t1}")

    files = s.get("output_files") or []
    if files:
        table = Table(
            box=rich_box.SIMPLE,
            padding=(0, 1),
            show_header=True,
            header_style="dim",
            show_edge=False,
        )
        table.add_column("filename", style=_C_BLUE, no_wrap=True)
        table.add_column("type", style="dim")
        for f in files[:10]:
            fname = f.get("filename") or f.get("gcs_url", "?")
            ftype = f.get("artifact_type", "")
            table.add_row(fname, ftype)
        if len(files) > 10:
            table.add_row(f"[dim]… {len(files) - 10} more[/]", "")
        console.print(table)


def _print_dataset_ready(dataset_id: str | None, artifact_count: int) -> None:
    content = Text()
    content.append("dataset_id      ", style="dim")
    content.append(f"{dataset_id}\n", style=f"bold {_C_BLUE}")
    content.append("artifact_count  ", style="dim")
    content.append(f"{artifact_count}\n\n")
    content.append("Run a job against this dataset:\n", style="bold")
    content.append(f"  phi folding          --dataset-id {dataset_id}\n", style=_C_SAND)
    content.append(f"  phi complex_folding  --dataset-id {dataset_id}\n", style=_C_SAND)
    content.append(f"  phi inverse_folding  --dataset-id {dataset_id}\n", style=_C_SAND)
    content.append(
        f"  phi filter           --dataset-id {dataset_id} --preset default --wait",
        style=_C_SAND,
    )
    console.print(
        Panel(
            content,
            title=f"[bold {_C_SAND}]✓ Dataset ready[/]",
            border_style=_C_SAND,
            padding=(1, 2),
        )
    )


_SCORE_DISPLAY_COLS = [
    ("design_index", "#", 4, False),
    ("esmfold_plddt", "pLDDT", 6, False),
    ("af2_ptm", "pTM", 6, False),
    ("af2_iptm", "ipTM", 6, False),
    ("af2_ipae", "iPAE", 6, False),
    ("rmsd", "RMSD", 7, True),
    ("fail_reasons", "Failures", 30, False),
]


def _render_scores_table(csv_content: str, thresholds: dict | None = None) -> None:
    try:
        rows = list(csv.DictReader(io.StringIO(csv_content)))
    except Exception as exc:
        err_console.print(f"  [dim]Could not parse scores CSV: {exc}[/]")
        return

    if not rows:
        return

    table = Table(
        box=rich_box.SIMPLE,
        show_header=True,
        header_style=f"bold {_C_SAND}",
        show_edge=False,
        padding=(0, 1),
        title="[dim]Per-design scores[/]",
        title_justify="left",
    )

    available_cols = [c for c in _SCORE_DISPLAY_COLS if c[0] in rows[0]]
    for _field, label, width, _is_ang in available_cols:
        table.add_column(label, min_width=width, no_wrap=True)

    for row in rows:
        passed = row.get("passed", "").lower() in ("true", "1", "yes")
        row_style = _C_SAND if passed else ""
        cells = []
        for field, label, _width, is_ang in available_cols:
            val = row.get(field, "")
            if val in ("", "None", "none"):
                cell = Text("—", style="dim")
            elif field == "fail_reasons":
                cell = Text(
                    val[:40] + ("…" if len(val) > 40 else ""),
                    style=f"dim {_C_ROSE}" if val else "dim",
                )
            elif field == "design_index":
                cell = Text(val, style="bold")
            else:
                try:
                    fval = float(val)
                    formatted = f"{fval:.3f}" + (" Å" if is_ang else "")
                    fail_str = row.get("fail_reasons", "")
                    label_short = label.lower().replace(" ", "")
                    is_failing = any(
                        label_short in reason.lower() for reason in fail_str.split(";")
                    )
                    cell = Text(formatted, style=_C_ROSE if is_failing else "")
                except (ValueError, TypeError):
                    cell = Text(val, style="dim")
            cells.append(cell)

        icon = Text("✓" if passed else "✗", style=_C_SAND if passed else _C_ROSE)
        if "fail_reasons" not in [c[0] for c in available_cols]:
            cells.append(icon)
        table.add_row(*cells, style=row_style if passed else "")

    console.print(table)

    if thresholds:
        parts = []
        if "plddt_threshold" in thresholds:
            parts.append(f"pLDDT≥{thresholds['plddt_threshold']}")
        if "ptm_threshold" in thresholds:
            parts.append(f"pTM≥{thresholds['ptm_threshold']}")
        if "iptm_threshold" in thresholds:
            parts.append(f"ipTM≥{thresholds['iptm_threshold']}")
        if "ipae_threshold" in thresholds:
            parts.append(f"iPAE≤{thresholds['ipae_threshold']}")
        if "rmsd_threshold" in thresholds:
            parts.append(f"RMSD≤{thresholds['rmsd_threshold']}Å")
        if parts:
            console.print(f"  [dim]thresholds: {' | '.join(parts)}[/]")


def _print_filter_done(
    job_id: str,
    final: dict,
    thresholds: dict | None = None,
    csv_content: str | None = None,
) -> None:
    artifacts = final.get("final_state", {}).get("artifacts", final.get("artifacts", {}))

    passed = artifacts.get("designs_passed", 0)
    failed = artifacts.get("designs_failed", 0)
    total = passed + failed
    summary = artifacts.get("scores_summary", f"{passed}/{total} designs passed")
    csv_uri = artifacts.get("scores_csv_gcs_uri", "")

    if csv_content:
        _render_scores_table(csv_content, thresholds)

    content = Text()
    pass_style = _C_SAND if passed > 0 else _C_ROSE
    content.append(f"{passed}/{total} designs passed  ", style=f"bold {pass_style}")
    if failed > 0:
        content.append(f"({failed} failed)\n\n", style=f"dim {_C_ROSE}")
    else:
        content.append("\n\n")

    content.append(summary + "\n\n", style="dim")

    if csv_uri:
        content.append("scores CSV      ", style="dim")
        content.append(f"{csv_uri}\n\n", style=f"{_C_BLUE}")

    content.append("Download results:\n", style="bold")
    content.append(f"  phi download {job_id} --out ./results\n\n", style=_C_SAND)

    if passed > 0:
        content.append("Run next design iteration:\n", style="bold")
        content.append(
            "  phi filter      --dataset-id <next-dataset-id> --preset relaxed --wait",
            style=_C_SAND,
        )

    console.print(
        Panel(
            content,
            title=f"[bold {_C_SAND}]✓ Filter complete[/]",
            border_style=_C_SAND,
            padding=(1, 2),
        )
    )


def _print_download_result(
    out_dir: str,
    downloaded: int,
    errors: int,
    counts_by_subdir: dict[str, int],
) -> None:
    result_content = Text()
    result_content.append(f"{downloaded} file(s) saved to ", style="bold")
    result_content.append(f"{out_dir}/\n\n", style=f"bold {_C_BLUE}")
    for subdir, count in sorted(counts_by_subdir.items()):
        result_content.append(f"  {subdir}/", style=f"{_C_BLUE}")
        result_content.append(f"   {count} file(s)\n", style="dim")
    if errors:
        result_content.append(
            f"\n  {errors} file(s) could not be downloaded", style=f"dim {_C_ROSE}"
        )
    console.print(
        Panel(
            result_content,
            title=f"[bold {_C_SAND}]✓ Download complete[/]",
            border_style=_C_SAND,
            padding=(1, 2),
        )
    )


def _make_upload_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
