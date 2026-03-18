from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel

from phi.api import _request
from phi.display import _C_BLUE, _C_ROSE, _C_SAND, _die, console
from phi.research import _stream_research
from phi.types import PhiApiError

_ERROR_REPORT_PREFIXES = ("# Research Failed", "# Error", "Error: Command failed")


def _is_error_report(report: str) -> bool:
    stripped = report.strip()
    return any(stripped.startswith(prefix) for prefix in _ERROR_REPORT_PREFIXES)


def cmd_research(args: argparse.Namespace) -> None:
    notes_file: Path | None = None if args.no_save else Path(args.notes_file)
    dataset_id: str | None = args.dataset_id

    stream_url = os.environ.get("DYNO_RESEARCH_STREAM_URL") or args.url

    context: str | None = None
    if args.context_file:
        try:
            prior = Path(args.context_file).read_text(encoding="utf-8")
            existing = args.context or ""
            context = (prior + "\n\n" + existing).strip() if existing else prior
        except Exception as exc:
            console.print(f"[{_C_ROSE}]Warning: could not read context file:[/] {exc}")
    elif args.context:
        context = args.context

    _stream_research(
        question=args.question,
        base_url=stream_url,
        notes_file=notes_file,
        dataset_id=dataset_id,
        context=context,
    )


def cmd_notes(args: argparse.Namespace) -> None:
    dataset_id: str = args.dataset_id

    try:
        data = _request("GET", f"/datasets/{dataset_id}/research-notes")
    except PhiApiError as exc:
        _die(str(exc))

    if not data.get("exists"):
        console.print(f"[{_C_ROSE}]No research notes found[/] for dataset {dataset_id}")
        return

    content: str = data.get("content") or ""
    data.get("gcs_url")
    data.get("gcs_uri")

    if args.out:
        out = Path(args.out)
        if out.suffix.lower() == ".md":
            dest = out
        else:
            out.mkdir(parents=True, exist_ok=True)
            dest = out / "research.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        console.print(f"[{_C_SAND}]Notes saved[/] → {dest}")
        return

    if args.json:
        print(json.dumps(data, indent=2))
        return

    console.print()
    console.print(
        Panel(
            Markdown(content),
            title=f"[{_C_BLUE}]Research Notes[/] — {dataset_id[:8]}…",
            border_style=_C_BLUE,
            padding=(1, 2),
        )
    )
