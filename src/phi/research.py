import json
import ssl
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from phi.api import _request
from phi.display import _C_BLUE, _C_ROSE, _C_SAND, console

_TOOL_ICONS: dict[str, str] = {
    "pubmed_search": "📚",
    "pdb_fetch": "🧬",
    "uniprot_fetch": "🔍",
    "uniprot_search": "🔍",
    "alphafold_fetch": "🔮",
    "interpro_scan": "🗺 ",
    "string_interactions": "🕸 ",
    "string_query": "🕸 ",
    "ensembl_lookup": "🧫",
    "ensembl_orthologs": "🌿",
}


def _tool_icon(tool_name: str) -> str:
    short = tool_name.split("__")[-1].replace("_tool", "")
    return _TOOL_ICONS.get(short, _TOOL_ICONS.get(tool_name, "🔧"))


def _build_research_section(
    question: str,
    report: str,
    citations_raw: str | None = None,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        f"\n---\n## {timestamp} — {question}\n",
        report.strip(),
    ]
    if citations_raw:
        try:
            citations = json.loads(citations_raw)
            lines = [f"\n**Sources ({len(citations)})**"]
            for i, src in enumerate(citations[:20], 1):
                title = src.get("title") or src.get("pmid") or str(src)
                lines.append(f"{i}. {title}")
            parts.append("\n".join(lines))
        except (json.JSONDecodeError, AttributeError) as exc:
            console.print(f"[dim]Warning: could not parse citations ({exc})[/]")
    return "\n\n".join(parts) + "\n"


def _save_research_notes_locally(notes_file: Path, section: str) -> None:
    try:
        notes_file.parent.mkdir(parents=True, exist_ok=True)
        with notes_file.open("a", encoding="utf-8") as fh:
            fh.write(section)
        console.print(f"[{_C_SAND}]Notes appended[/] → {notes_file}")
    except Exception as exc:
        console.print(f"[{_C_ROSE}]Warning: could not write notes file:[/] {exc}")


def _sync_research_notes_to_gcs(dataset_id: str, section: str) -> None:
    try:
        _request(
            "POST",
            f"/datasets/{dataset_id}/research-notes",
            {"content": section},
        )
        console.print(f"[{_C_SAND}]Notes synced to cloud[/] (dataset {dataset_id[:8]}…)")
    except Exception as exc:
        console.print(f"[{_C_ROSE}]Warning: cloud sync failed (notes saved locally):[/] {exc}")


def _append_research_notes(
    notes_file: Path,
    question: str,
    report: str,
    citations_raw: str | None = None,
    dataset_id: str | None = None,
) -> None:
    section = _build_research_section(question, report, citations_raw)
    _save_research_notes_locally(notes_file, section)
    if dataset_id:
        _sync_research_notes_to_gcs(dataset_id, section)


class _ResearchStreamState:
    def __init__(self) -> None:
        self.tool_calls: list[str] = []
        self.report_parts: list[str] = []
        self.turn: int = 0
        self.done: bool = False
        self.errored: bool = False


def _parse_sse_line(line: bytes) -> tuple[str, str] | None:
    decoded = line.decode("utf-8", errors="replace").rstrip("\n")
    if decoded.startswith("event:"):
        return ("_event", decoded[6:].strip())
    if decoded.startswith("data:"):
        return ("_data", decoded[5:].strip())
    return None


def _dispatch_sse_event(event_type: str, data: dict, state: _ResearchStreamState) -> None:
    if event_type == "tool_call":
        state.tool_calls.append(data.get("tool", "unknown"))
    elif event_type == "message":
        text = data.get("content", "")
        if text:
            state.report_parts.append(text)
    elif event_type == "complete":
        state.turn = data.get("turns", 0)
        state.done = True
    elif event_type == "error":
        state.errored = True


def _render_research_event(event_type: str, data: dict, state: _ResearchStreamState) -> None:
    if event_type == "start":
        console.print(
            f"[{_C_BLUE}]▶ Research started[/]  [dim]max_turns={data.get('max_turns', '?')}[/]"
        )
    elif event_type == "tool_call":
        tool = data.get("tool", "unknown")
        icon = _tool_icon(tool)
        inp = data.get("input", {})
        hint = next((str(v)[:60] for v in inp.values() if v), "") if isinstance(inp, dict) else ""
        console.print(f"  {icon} [{_C_BLUE}]{tool}[/]" + (f"  [dim]{hint}[/]" if hint else ""))
    elif event_type == "complete":
        model = data.get("model_used", "?")
        console.print(
            f"\n[{_C_SAND}]✓ Complete[/]  "
            f"[dim]{state.turn} turns · {len(state.tool_calls)} tool calls · {model}[/]"
        )
    elif event_type == "error":
        console.print(f"\n[{_C_ROSE}]✗ Stream error:[/] {data.get('error', 'unknown error')}")
    elif event_type == "retry":
        reason = data.get("reason", "")
        next_m = data.get("next_model", "")
        console.print(f"\n[{_C_ROSE}]↺ Retrying[/] ({reason})" + (f" → {next_m}" if next_m else ""))


def _stream_research(
    question: str,
    base_url: str,
    max_turns: int = 15,
    notes_file: Path | None = None,
    dataset_id: str | None = None,
) -> None:
    url = f"{base_url.rstrip('/')}/research/stream?" + urllib.parse.urlencode(
        {"question": question, "max_turns": max_turns}
    )

    console.print(f"\n[{_C_BLUE}]Connecting to research stream…[/]")
    console.print(f"[dim]{url}[/]\n")

    state = _ResearchStreamState()
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})

    try:
        try:
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl._create_unverified_context()

        with urllib.request.urlopen(req, context=ctx, timeout=600) as resp:
            event_type = "message"
            buffer: list[str] = []

            for raw_line in resp:
                parsed = _parse_sse_line(raw_line)
                if parsed is None:
                    if buffer:
                        data_str = "\n".join(buffer)
                        buffer = []
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            data = {"raw": data_str}

                        _dispatch_sse_event(event_type, data, state)
                        _render_research_event(event_type, data, state)

                        if state.errored:
                            return

                    event_type = "message"
                    continue

                key, val = parsed
                if key == "_event":
                    event_type = val
                elif key == "_data":
                    buffer.append(val)

    except KeyboardInterrupt:
        console.print(f"\n[{_C_ROSE}]Interrupted.[/]")
        return
    except Exception as e:
        console.print(f"\n[{_C_ROSE}]Stream failed:[/] {e}")
        return

    if state.report_parts:
        full_report = "\n".join(state.report_parts)
        console.print("\n" + "─" * 60)
        console.print(full_report)
        if notes_file is not None:
            _append_research_notes(
                notes_file=notes_file,
                question=question,
                report=full_report,
                citations_raw=None,
                dataset_id=dataset_id,
            )
    elif not state.done:
        console.print(f"\n[{_C_ROSE}]No report received.[/]")
