import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phi.research import (
    _build_research_section,
    _dispatch_sse_event,
    _parse_sse_line,
    _ResearchStreamState,
)


def test_parse_sse_line_data() -> None:
    result = _parse_sse_line(b"data: {\"event\": \"start\"}\n")
    assert result == ("_data", '{"event": "start"}')


def test_parse_sse_line_event() -> None:
    result = _parse_sse_line(b"event: tool_call\n")
    assert result == ("_event", "tool_call")


def test_parse_sse_line_blank_returns_none() -> None:
    assert _parse_sse_line(b"\n") is None
    assert _parse_sse_line(b"") is None


def test_parse_sse_line_comment_returns_none() -> None:
    assert _parse_sse_line(b": keep-alive\n") is None


def test_dispatch_tool_call_increments_count() -> None:
    state = _ResearchStreamState()
    _dispatch_sse_event("tool_call", {"tool": "pubmed_search"}, state)
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0] == "pubmed_search"


def test_dispatch_tool_call_multiple() -> None:
    state = _ResearchStreamState()
    _dispatch_sse_event("tool_call", {"tool": "pdb_fetch"}, state)
    _dispatch_sse_event("tool_call", {"tool": "uniprot_fetch"}, state)
    assert len(state.tool_calls) == 2


def test_dispatch_message_appends_to_parts() -> None:
    state = _ResearchStreamState()
    _dispatch_sse_event("message", {"content": "Hello"}, state)
    _dispatch_sse_event("message", {"content": " world"}, state)
    assert state.report_parts == ["Hello", " world"]


def test_dispatch_complete_sets_done_flag() -> None:
    state = _ResearchStreamState()
    _dispatch_sse_event("complete", {"turns": 5}, state)
    assert state.done is True
    assert state.turn == 5


def test_dispatch_error_sets_errored_flag() -> None:
    state = _ResearchStreamState()
    _dispatch_sse_event("error", {"error": "timeout"}, state)
    assert state.errored is True


def test_build_research_section_contains_question() -> None:
    section = _build_research_section("What binds PD-L1?", "Some findings.")
    assert "What binds PD-L1?" in section
    assert "Some findings." in section


def test_build_research_section_contains_date_header() -> None:
    section = _build_research_section("test question", "test report")
    assert "##" in section
    assert "—" in section


def test_build_research_section_includes_citations() -> None:
    import json

    citations = [{"title": "Paper A"}, {"title": "Paper B"}]
    section = _build_research_section("query", "report", json.dumps(citations))
    assert "Paper A" in section
    assert "Sources (2)" in section


def test_build_research_section_handles_invalid_citations() -> None:
    section = _build_research_section("query", "report", "not-valid-json")
    assert "query" in section
    assert "report" in section
