import argparse
import json
from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel

import phi.config as config
from phi.api import _request, _submit
from phi.display import _C_BLUE, _C_ROSE, _C_SAND, _die, _print_status, _print_submission, console
from phi.download import _download_job
from phi.polling import _poll
from phi.research import _append_research_notes, _stream_research
from phi.types import PhiApiError


def cmd_research(args: argparse.Namespace) -> None:
    notes_file: Path | None = None if args.no_save else Path(args.notes_file)
    dataset_id: str | None = args.dataset_id

    if args.stream:
        _stream_research(
            question=args.question,
            base_url=args.stream_url,
            notes_file=notes_file,
            dataset_id=dataset_id,
        )
        return

    params: dict = {
        "question": args.question,
        "databases": [d.strip() for d in args.databases.split(",") if d.strip()],
        "max_papers": args.max_papers,
        "include_structures": args.structures,
    }
    if args.target:
        params["target"] = args.target

    if args.context_file:
        try:
            prior = Path(args.context_file).read_text(encoding="utf-8")
            existing = args.context or ""
            params["context"] = (prior + "\n\n" + existing).strip() if existing else prior
        except Exception as exc:
            console.print(f"[{_C_ROSE}]Warning: could not read context file:[/] {exc}")
    elif args.context:
        params["context"] = args.context

    result = _submit("research", params, run_id=args.run_id)
    _print_submission(result)
    if args.wait:
        console.print(f"\n[dim]Polling every {config.POLL_INTERVAL}s …[/]")
        final = _poll(result["job_id"])
        _print_status(final)

        report = (final.get("outputs") or {}).get("report_md")
        citations_raw = (final.get("outputs") or {}).get("citations")

        if report:
            if notes_file is not None:
                _append_research_notes(
                    notes_file=notes_file,
                    question=args.question,
                    report=report,
                    citations_raw=citations_raw,
                    dataset_id=dataset_id,
                )

            if args.out:
                out = Path(args.out)
                out.mkdir(parents=True, exist_ok=True)
                (out / "research_report.md").write_text(report)
                console.print(f"\n[{_C_SAND}]Report written[/] → {out}/research_report.md")
                if citations_raw:
                    try:
                        citations = json.loads(citations_raw)
                        (out / "citations.json").write_text(json.dumps(citations, indent=2))
                        console.print(f"[{_C_SAND}]Citations written[/] → {out}/citations.json")
                    except Exception:
                        pass
            else:
                console.print("\n" + "─" * 60)
                console.print(report)
                if citations_raw:
                    try:
                        citations = json.loads(citations_raw)
                        console.print(f"\n[{_C_BLUE}]Sources ({len(citations)})[/]")
                        for i, src in enumerate(citations[:10], 1):
                            title = src.get("title") or src.get("pmid") or str(src)
                            console.print(f"  [{_C_BLUE}]{i}.[/] {title}")
                        if len(citations) > 10:
                            console.print(f"  [dim]… and {len(citations) - 10} more[/]")
                    except Exception:
                        pass
        elif args.out and final.get("status") == "completed":
            _download_job(final, args.out)


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
    gcs_url: str | None = data.get("gcs_url")
    gcs_uri: str | None = data.get("gcs_uri")

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
        if gcs_url:
            console.print(f"[dim]Download URL:[/] {gcs_url}")
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
    if gcs_uri:
        console.print(f"[dim]Storage URI:[/] {gcs_uri}")
