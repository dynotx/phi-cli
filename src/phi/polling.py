import time

from phi.api import _request, _status
from phi.config import INGEST_TERMINAL, POLL_TIMEOUT, TERMINAL_STATUSES
from phi.display import _C_BLUE, _C_ROSE, _C_SAND, _STATUS_COLOR, console
from phi.types import PhiApiError


def _poll(job_id: str, quiet: bool = False, poll_interval: float | None = None) -> dict:
    from phi.config import POLL_INTERVAL

    interval = poll_interval if poll_interval is not None else POLL_INTERVAL
    start = time.time()

    def _fetch() -> tuple[dict, str, int, int, str]:
        s = _status(job_id)
        status = s.get("status", "unknown")
        progress = s.get("progress") or {}
        pct = int(progress.get("percent_complete", 0))
        step = str(progress.get("current_step", ""))
        return s, status, pct, int(time.time() - start), step

    if quiet:
        while time.time() - start < POLL_TIMEOUT:
            s, status, _, _, _ = _fetch()
            if status in TERMINAL_STATUSES:
                return s
            time.sleep(interval)
        raise PhiApiError(f"Timed out after {POLL_TIMEOUT}s waiting for job {job_id}")

    if console.is_terminal:
        with console.status("", spinner="dots") as live:
            while time.time() - start < POLL_TIMEOUT:
                s, status, pct, elapsed, step = _fetch()
                color = _STATUS_COLOR.get(status, "white")
                msg = f"[{color}]{status}[/]  [dim]{pct:>3}%  {elapsed:>4}s[/]"
                if step:
                    msg += f"  [dim]{step[:70]}[/]"
                live.update(msg)
                if status in TERMINAL_STATUSES:
                    return s
                time.sleep(interval)
        raise PhiApiError(f"Timed out after {POLL_TIMEOUT}s waiting for job {job_id}")

    while time.time() - start < POLL_TIMEOUT:
        s, status, pct, elapsed, step = _fetch()
        parts = [f"[{elapsed:>4}s]", f"{status:<12}", f"{pct:>3}%"]
        if step:
            parts.append(step[:60])
        print("  " + "  ".join(parts))
        if status in TERMINAL_STATUSES:
            return s
        time.sleep(interval)
    raise PhiApiError(f"Timed out after {POLL_TIMEOUT}s waiting for job {job_id}")


def _ingest_poll(session_id: str) -> dict:
    from phi.config import POLL_INTERVAL

    start = time.time()

    def _fetch() -> tuple[dict, str, int, object]:
        s = _request("GET", f"/ingest_sessions/{session_id}")
        status = s.get("status", "UNKNOWN")
        uploaded = int(s.get("uploaded_files", 0))
        expected = s.get("expected_files", "?")
        return s, status, uploaded, expected

    if console.is_terminal:
        with console.status("", spinner="dots") as live:
            while time.time() - start < POLL_TIMEOUT:
                s, status, uploaded, expected = _fetch()
                elapsed = int(time.time() - start)
                color = (
                    _C_SAND if status == "READY" else (_C_ROSE if status == "FAILED" else _C_BLUE)
                )
                live.update(
                    f"[{color}]{status}[/]  [dim]{uploaded}/{expected} files indexed  {elapsed}s[/]"
                )
                if status in INGEST_TERMINAL:
                    return s
                time.sleep(POLL_INTERVAL)
        raise PhiApiError(
            f"Timed out after {POLL_TIMEOUT}s waiting for ingest session {session_id}"
        )

    while time.time() - start < POLL_TIMEOUT:
        s, status, uploaded, expected = _fetch()
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>4}s]  {status:<14}  {uploaded}/{expected} files indexed")
        if status in INGEST_TERMINAL:
            return s
        time.sleep(POLL_INTERVAL)
    raise PhiApiError(f"Timed out after {POLL_TIMEOUT}s waiting for ingest session {session_id}")
