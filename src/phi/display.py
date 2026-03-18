from __future__ import annotations

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

_DESIGN_NAME_MAX_LEN = 20

_SCORE_DISPLAY_COLS = [
    ("design_name", "Design", 16, False),
    ("esmfold_plddt", "pLDDT", 6, False),
    ("af2_ptm", "pTM", 6, False),
    ("af2_iptm", "ipTM", 6, False),
    ("af2_ipae", "iPAE", 6, False),
    ("rmsd", "RMSD", 7, True),
]

# Shown in a secondary table only when per-model sidecar data is present.
# M1/M2 = model_1_seed_000 / model_2_seed_000; primary iPAE uses rank_001
# (best across all seeds), so values may differ slightly.
_PER_MODEL_COLS = [
    ("af2_model1_iptm", "M1 ipTM", 7, False),
    ("af2_model1_ipae", "M1 iPAE", 7, False),
    ("af2_model2_iptm", "M2 ipTM", 7, False),
    ("af2_model2_ipae", "M2 iPAE", 7, False),
]

_THRESHOLD_LABELS = [
    ("plddt_threshold", "pLDDT≥{}"),
    ("ptm_threshold", "pTM≥{}"),
    ("iptm_threshold", "ipTM≥{}"),
    ("ipae_threshold", "iPAE≤{}Å"),
    ("rmsd_threshold", "RMSD≤{}Å"),
]


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


def _truncate_name(val: str) -> str:
    return (val[:_DESIGN_NAME_MAX_LEN] + "…") if len(val) > _DESIGN_NAME_MAX_LEN else val


def _threshold_summary(thresholds: dict) -> str:
    parts = [fmt.format(thresholds[key]) for key, fmt in _THRESHOLD_LABELS if key in thresholds]
    return ("  [dim]" + "  ·  ".join(parts) + "[/]") if parts else ""


def _is_cell_failing(label: str, fail_reasons: str) -> bool:
    label_short = label.lower().replace(" ", "")
    return any(label_short in r.lower() for r in fail_reasons.split(";"))


def _format_score_cell(field: str, val: str, is_ang: bool, is_failing: bool = False) -> Text:
    if val in ("", "None", "none"):
        return Text("—", style="dim")
    if field == "design_name":
        return Text(_truncate_name(val), style="bold")
    try:
        fval = float(val)
        formatted = f"{fval:.3f}" + (" Å" if is_ang else "")
        return Text(formatted, style=_C_ROSE if is_failing else "")
    except (ValueError, TypeError):
        return Text(val, style="dim")


def _has_per_model_data(rows: list[dict]) -> bool:
    return any(
        row.get(field, "").strip() not in ("", "None", "none")
        for row in rows
        for field, *_ in _PER_MODEL_COLS
        if field in row
    )


def _make_scores_table(rows: list[dict], thresholds: dict | None) -> Table:
    title = "[dim]Per-design scores[/]"
    if thresholds:
        title += _threshold_summary(thresholds)

    table = Table(
        box=rich_box.SIMPLE,
        show_header=True,
        header_style=f"bold {_C_SAND}",
        show_edge=False,
        padding=(0, 1),
        title=title,
        title_justify="left",
    )

    available_cols = [c for c in _SCORE_DISPLAY_COLS if c[0] in rows[0]]
    for _field, label, width, _is_ang in available_cols:
        table.add_column(label, min_width=width, no_wrap=True)
    table.add_column("", min_width=2, no_wrap=True)

    for row in rows:
        passed = row.get("passed", "").lower() in ("true", "1", "yes")
        fail_reasons = row.get("fail_reasons", "")
        cells = [
            _format_score_cell(
                field, row.get(field, ""), is_ang, _is_cell_failing(label, fail_reasons)
            )
            for field, label, _width, is_ang in available_cols
        ]
        cells.append(
            Text("✓", style=f"bold {_C_SAND}") if passed else Text("✗", style=f"dim {_C_ROSE}")
        )
        table.add_row(*cells, style="" if passed else "dim")

    return table


def _make_per_model_table(rows: list[dict]) -> Table:
    table = Table(
        box=rich_box.SIMPLE,
        show_header=True,
        header_style=f"bold {_C_SAND}",
        show_edge=False,
        padding=(0, 1),
        title="[dim]Per-model scores  (M1 = model_1_seed_000, M2 = model_2_seed_000)[/]",
        title_justify="left",
    )
    table.add_column("Design", min_width=16, no_wrap=True)
    per_model_avail = [c for c in _PER_MODEL_COLS if c[0] in rows[0]]
    for _field, label, width, _is_ang in per_model_avail:
        table.add_column(label, min_width=width, no_wrap=True)

    for row in rows:
        passed = row.get("passed", "").lower() in ("true", "1", "yes")
        design_name = row.get("design_name", row.get("design_index", ""))
        cells: list[Text] = [Text(_truncate_name(str(design_name)), style="bold")]
        cells += [
            _format_score_cell(field, row.get(field, ""), is_ang)
            for field, _label, _width, is_ang in per_model_avail
        ]
        table.add_row(*cells, style="" if passed else "dim")

    return table


def _parse_scores_csv(csv_content: str) -> list[dict] | None:
    try:
        rows = list(csv.DictReader(io.StringIO(csv_content)))
        return rows if rows else None
    except Exception as exc:
        err_console.print(f"  [dim]Could not parse scores CSV: {exc}[/]")
        return None


def _render_scores_table(csv_content: str, thresholds: dict | None = None) -> None:
    rows = _parse_scores_csv(csv_content)
    if rows is None:
        return
    console.print(_make_scores_table(rows, thresholds))


def _render_per_model_table(csv_content: str) -> None:
    rows = _parse_scores_csv(csv_content)
    if rows is None or not _has_per_model_data(rows):
        return
    console.print(_make_per_model_table(rows))


def _count_from_csv(csv_content: str) -> tuple[int, int]:
    rows = _parse_scores_csv(csv_content)
    if not rows:
        return 0, 0
    passed = sum(1 for r in rows if r.get("passed", "").lower() in ("true", "1", "yes"))
    return passed, len(rows) - passed


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
            raw = f.get("filename") or f.get("gcs_url", "?")
            fname = raw.split("/")[-1] if raw.startswith("gs://") else raw
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
    content.append("Next step:\n", style="bold")
    content.append("  phi filter\n\n", style=f"bold {_C_SAND}")
    content.append("Run a specific job:\n", style="bold dim")
    content.append(f"  phi folding          --dataset-id {dataset_id}\n", style="dim")
    content.append(f"  phi complex_folding  --dataset-id {dataset_id}\n", style="dim")
    content.append(f"  phi inverse_folding  --dataset-id {dataset_id}\n", style="dim")
    content.append(
        f"  phi filter           --dataset-id {dataset_id} --preset default --wait",
        style="dim",
    )
    console.print(
        Panel(
            content,
            title=f"[bold {_C_SAND}]✓ Dataset ready[/]",
            border_style=_C_SAND,
            padding=(1, 2),
        )
    )


def _print_filter_done(
    job_id: str,
    final: dict,
    thresholds: dict | None = None,
    csv_content: str | None = None,
) -> None:
    _results = final.get("_results", {})
    workflow_artifacts: dict = _results.get("workflow_artifacts") or {}

    passed = workflow_artifacts.get("designs_passed")
    failed = workflow_artifacts.get("designs_failed")

    if (passed is None or failed is None) and csv_content:
        passed, failed = _count_from_csv(csv_content)

    if passed is None or failed is None:
        # Fall back to counting output_files by artifact_type
        output_files: list[dict] = final.get("output_files") or []
        if output_files:
            passed = (
                passed
                if passed is not None
                else sum(1 for f in output_files if f.get("artifact_type") == "passed_designs")
            )
            failed = (
                failed
                if failed is not None
                else sum(1 for f in output_files if f.get("artifact_type") == "failed_designs")
            )

    passed = 0 if passed is None else passed
    failed = 0 if failed is None else failed

    total = passed + failed

    if csv_content:
        _render_scores_table(csv_content, thresholds)

    content = Text()
    content.append(f"{passed} passed", style=f"bold {_C_SAND}")
    content.append("  ·  ", style="dim")
    content.append(f"{failed} failed", style="dim")
    content.append("  ·  ", style="dim")
    content.append(f"{total} evaluated\n\n", style="dim")

    content.append("View scores inline:\n", style="bold")
    content.append(f"  phi scores {job_id}\n\n", style=_C_SAND)

    content.append("Download results:\n", style="bold")
    content.append(f"  phi download {job_id} --out ./results\n", style=_C_SAND)

    if passed > 0:
        content.append("\n")
        content.append("Run next design iteration:\n", style="bold")
        content.append(
            "  phi filter --dataset-id <next-dataset-id> --preset relaxed --wait",
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
