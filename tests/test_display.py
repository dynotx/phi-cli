import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phi.display import _duration_str, _fmt_size


def test_fmt_size_bytes() -> None:
    assert _fmt_size(0) == "0 B"
    assert _fmt_size(512) == "512 B"
    assert _fmt_size(1023) == "1023 B"


def test_fmt_size_kilobytes() -> None:
    assert _fmt_size(1024) == "1 KB"
    assert _fmt_size(2048) == "2 KB"
    assert _fmt_size(1_047_552) == "1023 KB"


def test_fmt_size_megabytes() -> None:
    assert _fmt_size(1_048_576) == "1.0 MB"
    assert _fmt_size(5_242_880) == "5.0 MB"
    assert _fmt_size(10_485_760) == "10.0 MB"


def test_duration_str_normal() -> None:
    result = _duration_str("2026-01-01T10:00:00", "2026-01-01T10:01:14")
    assert result == "74s"


def test_duration_str_missing_timestamps() -> None:
    assert _duration_str("", "") == ""


def test_duration_str_invalid_format() -> None:
    assert _duration_str("not-a-date", "also-not") == ""


def test_duration_str_very_long_job() -> None:
    result = _duration_str("2026-01-01T00:00:00", "2026-01-01T02:00:00")
    assert result == "7200s"


def test_duration_str_with_timezone() -> None:
    result = _duration_str("2026-01-01T10:00:00Z", "2026-01-01T10:01:30Z")
    assert result == "90s"
