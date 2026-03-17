import time
from collections.abc import Callable

import phi.config as config
from phi.api import _request, _status
from phi.config import INGEST_TERMINAL, POLL_TIMEOUT, TERMINAL_STATUSES
from phi.display import _C_BLUE, _C_ROSE, _C_SAND, _STATUS_COLOR, console
from phi.types import PhiApiError

_POLL_MAX_CONSECUTIVE_ERRORS = 3


def _poll_loop(
    job_id: str,
    interval: float,
    on_tick: Callable[[dict, str, int, int, str], None],
) -> dict:
    start = time.time()
    consecutive_errors = 0
    last_s: dict = {}
    while time.time() - start < POLL_TIMEOUT:
        try:
            last_s = _status(job_id)
            consecutive_errors = 0
        except PhiApiError as e:
            consecutive_errors += 1
            if consecutive_errors > _POLL_MAX_CONSECUTIVE_ERRORS:
                raise PhiApiError(
                    f"Status check failed {consecutive_errors} times in a row: {e}\n"
                    f"  Job {job_id} may still be running — check with: phi status {job_id}"
                ) from e
            time.sleep(interval)
            continue
        status = last_s.get("status", "unknown")
        progress = last_s.get("progress") or {}
        pct = int(progress.get("percent_complete", 0))
        step = str(progress.get("current_step", ""))
        elapsed = int(time.time() - start)
        on_tick(last_s, status, pct, elapsed, step)
        if status in TERMINAL_STATUSES:
            return last_s
        time.sleep(interval)
    raise PhiApiError(f"Timed out after {POLL_TIMEOUT}s waiting for job {job_id}")


def _ingest_poll_loop(
    session_id: str,
    interval: float,
    on_tick: Callable[[dict, str, int, object, int], None],
) -> dict:
    start = time.time()
    consecutive_errors = 0
    while time.time() - start < POLL_TIMEOUT:
        try:
            s = _request("GET", f"/ingest_sessions/{session_id}")
            consecutive_errors = 0
        except PhiApiError as e:
            consecutive_errors += 1
            if consecutive_errors > _POLL_MAX_CONSECUTIVE_ERRORS:
                raise PhiApiError(
                    f"Status check failed {consecutive_errors} times in a row: {e}"
                ) from e
            time.sleep(interval)
            continue
        status = s.get("status", "UNKNOWN")
        uploaded = int(s.get("uploaded_files", 0))
        expected = s.get("expected_files", "?")
        elapsed = int(time.time() - start)
        on_tick(s, status, uploaded, expected, elapsed)
        if status in INGEST_TERMINAL:
            return s
        time.sleep(interval)
    raise PhiApiError(f"Timed out after {POLL_TIMEOUT}s waiting for ingest session {session_id}")


def _poll(job_id: str, quiet: bool = False, poll_interval: float | None = None) -> dict:
    interval = poll_interval if poll_interval is not None else config.POLL_INTERVAL

    if quiet:
        return _poll_loop(job_id, interval, lambda *_: None)

    if console.is_terminal:
        with console.status("", spinner="dots") as live:

            def _on_tick_terminal(s: dict, status: str, pct: int, elapsed: int, step: str) -> None:
                color = _STATUS_COLOR.get(status, "white")
                msg = f"[{color}]{status}[/]  [dim]{pct:>3}%  {elapsed:>4}s[/]"
                if step:
                    msg += f"  [dim]{step[:70]}[/]"
                live.update(msg)

            return _poll_loop(job_id, interval, _on_tick_terminal)

    def _on_tick_plain(s: dict, status: str, pct: int, elapsed: int, step: str) -> None:
        parts = [f"[{elapsed:>4}s]", f"{status:<12}", f"{pct:>3}%"]
        if step:
            parts.append(step[:60])
        print("  " + "  ".join(parts))

    return _poll_loop(job_id, interval, _on_tick_plain)


def _ingest_poll(session_id: str) -> dict:
    if console.is_terminal:
        with console.status("", spinner="dots") as live:

            def _on_tick_terminal(
                s: dict, status: str, uploaded: int, expected: object, elapsed: int
            ) -> None:
                color = (
                    _C_SAND if status == "READY" else (_C_ROSE if status == "FAILED" else _C_BLUE)
                )
                live.update(
                    f"[{color}]{status}[/]  [dim]{uploaded}/{expected} files indexed  {elapsed}s[/]"
                )

            return _ingest_poll_loop(session_id, config.POLL_INTERVAL, _on_tick_terminal)

    def _on_tick_plain(s: dict, status: str, uploaded: int, expected: object, elapsed: int) -> None:
        print(f"  [{elapsed:>4}s]  {status:<14}  {uploaded}/{expected} files indexed")

    return _ingest_poll_loop(session_id, config.POLL_INTERVAL, _on_tick_plain)
