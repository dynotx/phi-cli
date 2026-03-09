#!/usr/bin/env python3
"""
phi — Dyno Phi protein design platform CLI.

Submit scoring jobs, manage datasets, and download results from
the Dyno Phi API at https://design.dynotx.com/api/v1.

Authentication:
  export DYNO_API_KEY=your_key   # from Settings → API keys
  export DYNO_API_BASE_URL=...   # optional override

Quick try (single sequence / structure):
  phi esmfold     --fasta sequences.fasta
  phi alphafold   --fasta complex.fasta
  phi proteinmpnn --pdb design.pdb  --num-sequences 20
  phi esm2        --fasta sequences.fasta
  phi boltz       --fasta complex.fasta

Batch workflow (100–50,000 files):
  phi upload --dir ./designs/ --file-type pdb --run-id pdl1_batch
  # → prints dataset_id: dataset_abc

  phi esmfold     --dataset-id dataset_abc --out ./screen
  phi alphafold   --dataset-id dataset_abc --out ./validation
  phi proteinmpnn --dataset-id dataset_abc --num-sequences 20

Dataset management:
  phi datasets               # list your datasets
  phi dataset DATASET_ID     # show dataset details

Authentication:
  phi login                  # verify API key + print connection and identity

Research:
  phi research --question "What are known PD-L1 binding hotspots?"

Job management:
  phi status   JOB_ID
  phi jobs     [--limit 20] [--status running]
  phi cancel   JOB_ID
  phi download JOB_ID [--out ./results]
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import NoReturn, TypedDict, cast

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

from phi._version import __version__  # noqa: F401  (exported for `phi --version`)

# ─────────────────────────── rich console ────────────────────────────────────

console = Console()
err_console = Console(stderr=True)

# ─────────────────────────── colour palette ──────────────────────────────────
# Three muted tones that feel cohesive and don't strain the eye.
_C_SAND = "#B5A58F"  # warm sand  — success, ✓ marks, completed
_C_ROSE = "#D4A5B8"  # dusty rose — errors, failed, cancelled, warnings
_C_BLUE = "#8FA5B8"  # steel blue — info, filenames, links, running

# ─────────────────────────── ASCII logo ──────────────────────────────────────


# ─────────────────────────── API response shapes ─────────────────────────────


class IngestSessionResponse(TypedDict, total=False):
    session_id: str
    id: str  # alternative key some backends use
    status: str
    expected_files: int
    uploaded_files: int
    dataset_id: str
    artifact_count: int
    error: str


class SignedUrlEntry(TypedDict):
    file: str
    url: str


class SignedUrlsResponse(TypedDict, total=False):
    urls: list[SignedUrlEntry]


class JobSubmitResponse(TypedDict, total=False):
    job_id: str
    run_id: str
    status: str
    message: str


def _require_key(d: dict, key: str, context: str) -> str:
    """Return d[key] or die with a clear message if missing."""
    value = d.get(key)
    if not value:
        _die(f"API response for {context} is missing required field '{key}': {d}")
    return value  # type: ignore[no-any-return]  # _die() is NoReturn; value is truthy str


# ─────────────────────────── configuration ───────────────────────────────────

DEFAULT_BASE_URL = "https://design.dynotx.com"
POLL_INTERVAL = 5  # seconds between status checks
POLL_TIMEOUT = 7200  # 2-hour maximum poll window
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
# "submitted" kept for backward compat with jobs created before the 2026-03-07 backend fix.
# New jobs return "pending" immediately on submission.
NON_TERMINAL_STATUSES = {"pending", "submitted", "running"}
INGEST_TERMINAL = {"READY", "FAILED"}
UPLOAD_BATCH_SIZE = 50  # filenames per signed-URL request
UPLOAD_WORKERS = 8  # parallel upload threads
STATE_FILE = Path(".phi-state.json")  # project-local state, gitignored


# ─────────────────────────── local state ─────────────────────────────────────


def _load_state() -> dict[str, object]:
    try:
        return dict(json.loads(STATE_FILE.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(updates: dict) -> None:
    state = _load_state()
    state.update(updates)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def _resolve_cached_id(
    args: argparse.Namespace,
    attr: str,
    state_key: str,
    missing_hint: str,
) -> str:
    """Return the named arg if provided, else fall back to the cached state value, else die."""
    explicit: str | None = getattr(args, attr, None)
    if explicit:
        return explicit
    cached = _load_state().get(state_key)
    if cached:
        console.print(
            f"[dim]Using cached {state_key} [{_C_BLUE}]{cached}[/] (from .phi-state.json)[/]"
        )
        return str(cached)
    _die(missing_hint)
    raise SystemExit(1)  # unreachable — _die exits


def _resolve_dataset_id(args: argparse.Namespace) -> str:
    return _resolve_cached_id(
        args,
        attr="dataset_id",
        state_key="dataset_id",
        missing_hint=(
            "No --dataset-id provided and none cached.\n"
            "  Upload a dataset first:  phi upload ./designs/\n"
            "  Or set one explicitly:   phi use <dataset-id>"
        ),
    )


def _resolve_job_id(args: argparse.Namespace) -> str:
    return _resolve_cached_id(
        args,
        attr="job_id",
        state_key="last_job_id",
        missing_hint=(
            "No job_id provided and none cached.\n"
            "  Run 'phi filter --wait' to submit and cache a job, or pass the job_id explicitly."
        ),
    )


def _base_url() -> str:
    return os.environ.get("DYNO_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context that works on macOS where system certs may be missing."""
    ctx = ssl.create_default_context()
    # Honour explicit override first (e.g. SSL_CERT_FILE=/opt/homebrew/etc/openssl@3/cert.pem)
    env_cafile = os.environ.get("SSL_CERT_FILE")
    if env_cafile and Path(env_cafile).exists():
        ctx.load_verify_locations(env_cafile)
        return ctx
    # Try common macOS Homebrew / Linux system locations
    for cafile in [
        "/opt/homebrew/etc/openssl@3/cert.pem",
        "/opt/homebrew/etc/openssl@1.1/cert.pem",
        "/etc/ssl/cert.pem",
        "/usr/local/etc/openssl/cert.pem",
    ]:
        if Path(cafile).exists():
            ctx.load_verify_locations(cafile)
            return ctx
    return ctx  # fall back to default (works fine on Linux CI)


def _api_key() -> str:
    key = os.environ.get("DYNO_API_KEY")
    if not key:
        for candidate in [Path(".env"), Path.home() / ".dyno" / ".env"]:
            if candidate.exists():
                for line in candidate.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("DYNO_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip("\"'")
                        if key:
                            break
    if not key:
        _die(
            "DYNO_API_KEY is not set.\n"
            "  1. Open https://design.dynotx.com/dashboard/settings\n"
            "  2. Create an API key under 'API keys'\n"
            "  3. Run: export DYNO_API_KEY=your_key"
        )
    return key  # _die() is NoReturn, so type-checker knows this is never None


class PhiApiError(Exception):
    """Raised by _request() on HTTP or network errors."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


def _die(msg: str) -> NoReturn:
    err_console.print(f"[bold {_C_ROSE}]error:[/] {msg}")
    sys.exit(1)


# ─────────────────────────── HTTP helpers ────────────────────────────────────


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{_base_url()}/api/v1{path}"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": _api_key(),
        "X-Organization-ID": os.environ.get("DYNO_ORG_ID", "default-org"),
        "X-User-ID": os.environ.get("DYNO_USER_ID", "default-user"),
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
            return cast(dict, json.loads(resp.read()))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read())
        except Exception:
            detail = e.reason
        raise PhiApiError(f"HTTP {e.code} — {detail}") from e
    except urllib.error.URLError as e:
        raise PhiApiError(f"Network error — {e.reason}\n  URL: {url}") from e


_UPLOAD_RETRIES = 3
_UPLOAD_RETRY_BASE = 2.0  # seconds; doubles on each retry


def _put_file(signed_url: str, path: Path) -> None:
    """Upload a single file via signed URL (no auth header — direct-to-GCS).

    Retries up to _UPLOAD_RETRIES times with exponential backoff on transient
    errors (HTTP 429, 5xx, and network timeouts).
    """
    data = path.read_bytes()
    last_exc: Exception | None = None

    for attempt in range(_UPLOAD_RETRIES):
        req = urllib.request.Request(
            signed_url,
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            method="PUT",
        )
        try:
            with urllib.request.urlopen(req, timeout=300, context=_ssl_context()) as resp:
                _ = resp.read()
            return  # success
        except urllib.error.HTTPError as e:
            if e.code in {429, 500, 502, 503, 504} and attempt < _UPLOAD_RETRIES - 1:
                wait = _UPLOAD_RETRY_BASE**attempt
                time.sleep(wait)
                last_exc = e
                continue
            raise RuntimeError(f"Upload failed for {path.name}: HTTP {e.code}") from e
        except urllib.error.URLError as e:
            if attempt < _UPLOAD_RETRIES - 1:
                wait = _UPLOAD_RETRY_BASE**attempt
                time.sleep(wait)
                last_exc = e
                continue
            raise RuntimeError(f"Upload failed for {path.name}: {e.reason}") from e

    raise RuntimeError(
        f"Upload failed for {path.name} after {_UPLOAD_RETRIES} attempts"
    ) from last_exc


def _resolve_identity() -> None:
    """Populate DYNO_USER_ID + DYNO_ORG_ID from GET /auth/me if not already set.

    Upload endpoints require X-User-ID and X-Organization-ID headers.
    Silently falls back to env-var defaults when /auth/me returns 404
    (staging environments where Clerk is not yet wired up).
    Run `phi login` to see current identity and connection status.
    """
    if os.environ.get("DYNO_USER_ID") and os.environ.get("DYNO_ORG_ID"):
        return
    try:
        me = _request("GET", "/auth/me")
        if not os.environ.get("DYNO_USER_ID"):
            os.environ["DYNO_USER_ID"] = me.get("user_id") or ""
        if not os.environ.get("DYNO_ORG_ID"):
            os.environ["DYNO_ORG_ID"] = me.get("org_id") or ""
    except PhiApiError:
        pass  # staging / local: static key, no /auth/me — use env defaults


def _submit(
    job_type: str,
    params: dict,
    run_id: str | None = None,
    context: dict | None = None,
    dataset_id: str | None = None,
) -> dict:
    body: dict = {"job_type": job_type, "params": params}
    if dataset_id:
        body["dataset_id"] = dataset_id
    if run_id:
        body["run_id"] = run_id
    if context:
        body["context"] = context
    return _request("POST", "/jobs/", body)


def _status(job_id: str) -> dict:
    return _request("GET", f"/jobs/{job_id}/status")


# ─────────────────────────── status colors ───────────────────────────────────

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


# ─────────────────────────── polling ─────────────────────────────────────────


def _poll(job_id: str, quiet: bool = False) -> dict:
    """Poll until terminal status and return final status object.

    On a TTY: shows a live animated spinner that updates in place.
    On non-TTY (CI/pipe) or quiet=True: falls back to scrolling plain-text lines.
    """
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
            time.sleep(POLL_INTERVAL)
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
                time.sleep(POLL_INTERVAL)
        raise PhiApiError(f"Timed out after {POLL_TIMEOUT}s waiting for job {job_id}")

    # non-TTY fallback: scrolling lines
    while time.time() - start < POLL_TIMEOUT:
        s, status, pct, elapsed, step = _fetch()
        parts = [f"[{elapsed:>4}s]", f"{status:<12}", f"{pct:>3}%"]
        if step:
            parts.append(step[:60])
        print("  " + "  ".join(parts))
        if status in TERMINAL_STATUSES:
            return s
        time.sleep(POLL_INTERVAL)
    raise PhiApiError(f"Timed out after {POLL_TIMEOUT}s waiting for job {job_id}")


# ─────────────────────────── pretty output ───────────────────────────────────


def _print_submission(result: dict) -> None:
    console.print(f"\n[bold {_C_SAND}]✓[/] Job submitted")
    console.print(f"  [dim]job_id[/]  {result.get('job_id')}")
    console.print(f"  [dim]run_id[/]  {result.get('run_id')}")
    console.print(f"  [dim]status[/]  [dim]{result.get('status')}[/]")
    if result.get("message"):
        console.print(f"  [dim]message[/] {result['message']}")


def _duration_str(started_at: str, completed_at: str) -> str:
    """Return human-readable elapsed like '74s', or '' on parse failure."""
    try:
        t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        return f"{int((t1 - t0).total_seconds())}s"
    except Exception:
        return ""


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


# ─────────────────────────── ingest helpers ──────────────────────────────────


def _collect_files(args: argparse.Namespace) -> list[Path]:
    """Return a deduplicated sorted list of files to upload.

    Raises SystemExit early if any filenames collide (same basename from
    different directories), because the signed-URL map is keyed by filename.
    """
    paths: list[Path] = []

    # Positional file arguments — directories are expanded automatically
    for f in getattr(args, "files", None) or []:
        p = Path(f)
        if not p.exists():
            _die(f"File not found: {f}")
        if p.is_dir():
            ext = f".{args.file_type}" if getattr(args, "file_type", None) else None
            found = (
                sorted(q for q in p.iterdir() if q.is_file())
                if not ext
                else sorted(p.glob(f"*{ext}"))
            )
            if not found:
                _die(f"No {'*' + ext if ext else ''} files found in {p}")
            paths.extend(found)
        else:
            paths.append(p)

    # --dir directory scan
    if getattr(args, "dir", None):
        d = Path(args.dir)
        if not d.is_dir():
            _die(f"Not a directory: {args.dir}")
        ext = f".{args.file_type}" if getattr(args, "file_type", None) else None
        found = (
            sorted(p for p in d.iterdir() if p.is_file()) if not ext else sorted(d.glob(f"*{ext}"))
        )
        if not found:
            _die(f"No {'*' + ext if ext else ''} files found in {d}")
        paths.extend(found)

    if not paths:
        _die(
            "No files specified. Use positional arguments, --dir, or --gcs.\n"
            "  phi upload --dir ./designs/ --file-type pdb\n"
            "  phi upload binder_001.pdb binder_002.pdb\n"
            "  phi upload sequences.fasta"
        )

    # Deduplicate by resolved path (same inode from two relative references)
    seen_resolved: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen_resolved:
            seen_resolved.add(rp)
            unique.append(p)

    # Fail fast on filename collisions — signed-URL map is keyed by p.name
    names: list[str] = [p.name for p in unique]
    seen_names: set[str] = set()
    collisions: list[str] = []
    for name in names:
        if name in seen_names:
            collisions.append(name)
        seen_names.add(name)
    if collisions:
        _die(
            f"Filename collision(s) detected — the upload API keys signed URLs by "
            f"filename, so all files must have unique basenames.\n"
            f"  Colliding names: {', '.join(sorted(set(collisions)))}\n"
            f"  Rename the duplicates before uploading."
        )

    return unique


def _ingest_poll(session_id: str) -> dict:
    """Poll an ingest session until READY or FAILED.

    On a TTY: shows a live animated spinner.
    On non-TTY: scrolling plain-text lines.
    """
    start = time.time()

    def _fetch() -> tuple[dict, str, int, int]:
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

    # non-TTY fallback
    while time.time() - start < POLL_TIMEOUT:
        s, status, uploaded, expected = _fetch()
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>4}s]  {status:<14}  {uploaded}/{expected} files indexed")
        if status in INGEST_TERMINAL:
            return s
        time.sleep(POLL_INTERVAL)
    raise PhiApiError(f"Timed out after {POLL_TIMEOUT}s waiting for ingest session {session_id}")


# ─────────────────────────── subcommands ─────────────────────────────────────


def _run_model_job(job_type: str, params: dict, args: argparse.Namespace) -> None:
    """Shared logic for all model job commands — handles dataset-id or inline mode."""
    dataset_id = getattr(args, "dataset_id", None)
    result = _submit(job_type, params, run_id=args.run_id, dataset_id=dataset_id)
    job_id = _require_key(result, "job_id", f"POST /jobs ({job_type})")
    _print_submission(result)
    if args.wait:
        console.print(f"\n[dim]Polling every {POLL_INTERVAL}s …[/]")
        final = _poll(job_id)
        _print_status(final)
        if args.out and final.get("status") == "completed":
            _download_job(final, args.out)


def _create_ingest_session(files: list[Path], args: argparse.Namespace) -> str:
    """Create an ingest session and return its ID."""
    body: dict = {"expected_files": len(files)}
    if args.run_id:
        body["run_id"] = args.run_id
    if getattr(args, "file_type", None):
        body["file_type"] = args.file_type
    elif files:
        # Auto-infer file type from the first file's extension
        body["file_type"] = files[0].suffix.lstrip(".")
    session = _request("POST", "/ingest_sessions/", body)
    # Accept either "session_id" or "id" as the session identifier
    session_id = session.get("session_id") or session.get("id")
    if not session_id:
        _die(f"POST /ingest_sessions response missing 'session_id': {session}")
    console.print(f"  [dim]session_id[/] : {session_id}")
    return session_id  # type: ignore[no-any-return]  # _die() is NoReturn; session_id is truthy str


def _request_signed_urls(session_id: str, files: list[Path]) -> dict[str, str]:
    """Request signed upload URLs for all files; returns filename → URL mapping."""
    total = len(files)
    batches = [files[i : i + UPLOAD_BATCH_SIZE] for i in range(0, total, UPLOAD_BATCH_SIZE)]
    console.print(f"  Requesting signed URLs ({len(batches)} batch(es) of ≤{UPLOAD_BATCH_SIZE}) …")
    url_map: dict[str, str] = {}
    for batch in batches:
        resp = _request(
            "POST",
            f"/ingest_sessions/{session_id}/upload_urls",
            {"files": [p.name for p in batch]},
        )
        for entry in resp.get("urls", []):
            url_map[entry["file"]] = entry["url"]
    if len(url_map) != total:
        _die(
            f"Expected {total} signed URLs but received {len(url_map)}. "
            "Check the ingest_sessions endpoint."
        )
    return url_map


def _upload_all_parallel(files: list[Path], url_map: dict[str, str]) -> list[str]:
    """Upload all files in parallel. Returns a list of failure messages (empty = success)."""
    total = len(files)
    failures: list[str] = []

    def _upload_one(path: Path) -> tuple[str, bool, str]:
        signed_url = url_map.get(path.name)
        if not signed_url:
            return path.name, False, "No signed URL received"
        try:
            _put_file(signed_url, path)
            return path.name, True, ""
        except RuntimeError as exc:
            return path.name, False, str(exc)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            f"  Uploading {total} file(s) with {UPLOAD_WORKERS} workers",
            total=total,
        )
        with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
            futures = {pool.submit(_upload_one, p): p for p in files}
            for future in as_completed(futures):
                name, ok, err = future.result()
                progress.advance(task)
                if not ok:
                    failures.append(f"{name}: {err}")
                    console.print(f"  [bold {_C_ROSE}]✗[/] [dim]{name}[/]: {err}")

    return failures


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


def _fetch_and_show_scores(artifact_files: list[dict], thresholds: dict | None = None) -> None:
    """Fetch scores.csv from artifact_files and display a per-design rich table."""
    import csv
    import io

    scores_artifact: dict | None = None
    for af in artifact_files:
        name = af.get("name") or af.get("filename") or ""
        atype = af.get("artifact_type") or ""
        if atype == "csv" or name.endswith(".csv"):
            scores_artifact = af
            break

    if not scores_artifact:
        return

    url = _get_artifact_url(scores_artifact)
    if not url:
        err_console.print("  [dim]Scores table unavailable: no download URL for scores.csv[/]")
        return

    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        _fetch_url_to_file(url, tmp_path)
        csv_content = tmp_path.read_text()
    except Exception as exc:
        err_console.print(f"  [dim]Could not fetch scores table: {exc}[/]")
        return

    try:
        rows = list(csv.DictReader(io.StringIO(csv_content)))
    except Exception as exc:
        err_console.print(f"  [dim]Could not parse scores CSV: {exc}[/]")
        return

    if not rows:
        return

    # Key display columns (subset of the full CSV)
    score_cols = [
        ("design_index", "#", 4, False),
        ("esmfold_plddt", "pLDDT", 6, False),
        ("af2_ptm", "pTM", 6, False),
        ("af2_iptm", "ipTM", 6, False),
        ("af2_ipae", "iPAE", 6, False),
        ("rmsd", "RMSD", 7, True),  # suffix "Å"
        ("fail_reasons", "Failures", 30, False),
    ]

    table = Table(
        box=rich_box.SIMPLE,
        show_header=True,
        header_style=f"bold {_C_SAND}",
        show_edge=False,
        padding=(0, 1),
        title="[dim]Per-design scores[/]",
        title_justify="left",
    )

    available_cols = [c for c in score_cols if c[0] in rows[0]]
    for _field, label, width, _is_ang in available_cols:
        table.add_column(label, min_width=width, no_wrap=True)

    for row in rows:
        passed = row.get("passed", "").lower() in ("true", "1", "yes")
        row_style = _C_SAND if passed else ""
        cells = []
        for field, _label, _width, is_ang in available_cols:
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
                # Numeric: check if this metric caused a failure
                try:
                    fval = float(val)
                    formatted = f"{fval:.3f}" + (" Å" if is_ang else "")
                    # Highlight failing metric value
                    fail_str = row.get("fail_reasons", "")
                    label_short = _label.lower().replace(" ", "")
                    is_failing = any(
                        label_short in reason.lower() for reason in fail_str.split(";")
                    )
                    cell = Text(formatted, style=_C_ROSE if is_failing else "")
                except (ValueError, TypeError):
                    cell = Text(val, style="dim")
            cells.append(cell)

        icon = Text("✓" if passed else "✗", style=_C_SAND if passed else _C_ROSE)
        # Insert pass icon as the last column
        if "fail_reasons" not in [c[0] for c in available_cols]:
            cells.append(icon)
        table.add_row(*cells, style=row_style if passed else "")

    console.print(table)

    # Thresholds footer
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


def _print_filter_done(job_id: str, final: dict, thresholds: dict | None = None) -> None:
    """Print post-filter summary panel with scores and next steps."""
    artifacts = final.get("final_state", {}).get("artifacts", final.get("artifacts", {}))

    passed = artifacts.get("designs_passed", 0)
    failed = artifacts.get("designs_failed", 0)
    total = passed + failed
    summary = artifacts.get("scores_summary", f"{passed}/{total} designs passed")
    csv_uri = artifacts.get("scores_csv_gcs_uri", "")

    # Show per-design scores table if we have downloaded results
    results = final.get("_results") or {}
    artifact_files = results.get("artifact_files") or []
    if artifact_files:
        _fetch_and_show_scores(artifact_files, thresholds)

    content = Text()

    # Pass/fail counts with colour coding
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


def _ensure_authenticated() -> None:
    """Probe the API; on 401, exit with a clear message. Use before upload/job commands."""
    try:
        _request("GET", "/jobs/?page_size=1")
    except PhiApiError as e:
        if "401" in str(e):
            _die("Not authenticated. Run 'phi login' to verify your API key and endpoint.")
        raise


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload local files → staged ingest → versioned dataset."""
    _resolve_identity()  # populate X-User-ID + X-Organization-ID from /auth/me or env
    _ensure_authenticated()

    if getattr(args, "gcs", None):
        console.print("  External cloud storage import is not yet available.")
        console.print("  The backend import worker is planned — see backend-api-gaps.md §9.")
        console.print(f"  When available: phi upload --gcs {args.gcs}")
        return

    files = _collect_files(args)
    console.print(f"[bold {_C_SAND}]✓[/] Found {len(files)} file(s) to upload")

    session_id = _create_ingest_session(files, args)
    url_map = _request_signed_urls(session_id, files)

    failures = _upload_all_parallel(files, url_map)
    if failures:
        console.print(f"\n[bold {_C_ROSE}]error:[/] {len(failures)} upload(s) failed:")
        for msg in failures[:10]:
            console.print(f"  [dim]{msg}[/]")
        _die("Upload incomplete — fix errors and retry.")

    console.print(f"  [bold {_C_SAND}]✓[/] All {len(files)} file(s) uploaded successfully")

    console.print("  Finalizing ingest session …")
    _request("POST", f"/ingest_sessions/{session_id}/finalize", {})

    if args.wait:
        console.print(
            f"\n[dim]Ingesting and validating files (polling every {POLL_INTERVAL}s) …[/]"
        )
        result = _ingest_poll(session_id)
        status = result.get("status")
        if status == "READY":
            dataset_id = result.get("dataset_id")
            if dataset_id:
                _save_state({"dataset_id": dataset_id})
            _print_dataset_ready(dataset_id, result.get("artifact_count", len(files)))
        else:
            _die(f"Ingest failed ({status}): {result.get('error', 'unknown error')}")
    else:
        console.print(f"\n[bold {_C_SAND}]✓[/] Session finalized — ingestion running in background")
        console.print(f"  [dim]session_id[/]: {session_id}")
        console.print(f"  Check status: [{_C_BLUE}]phi ingest-session {session_id}[/]")


def cmd_ingest_session(args: argparse.Namespace) -> None:
    """Show status of an ingest session (after upload + finalize)."""
    result = _request("GET", f"/ingest_sessions/{args.session_id}")
    if args.json:
        print(json.dumps(result, indent=2))
        return
    console.print(f"[dim]session_id    [/] : {result.get('session_id') or result.get('id', '?')}")
    console.print(f"[dim]status        [/] : {result.get('status', '?')}")
    console.print(f"[dim]expected_files[/] : {result.get('expected_files', '?')}")
    console.print(f"[dim]uploaded_files[/] : {result.get('uploaded_files', '?')}")
    console.print(f"[dim]artifact_count[/] : {result.get('artifact_count', '?')}")
    if result.get("dataset_id"):
        console.print(f"[dim]dataset_id    [/] : {result.get('dataset_id')}")
        console.print(f"\n  [{_C_BLUE}]phi esmfold     --dataset-id {result.get('dataset_id')}[/]")
        console.print(f"  [{_C_BLUE}]phi alphafold   --dataset-id {result.get('dataset_id')}[/]")


def cmd_datasets(args: argparse.Namespace) -> None:
    """List datasets owned by the current user."""
    params = f"?page_size={args.limit}"
    result = _request("GET", f"/datasets{params}")
    if args.json:
        print(json.dumps(result, indent=2))
        return
    datasets = result.get("datasets", [])
    if not datasets:
        console.print("[dim]No datasets found.[/]")
        return

    table = Table(box=rich_box.SIMPLE_HEAVY, show_header=True, header_style="bold dim")
    table.add_column("DATASET ID", style=_C_BLUE, no_wrap=True)
    table.add_column("ARTIFACTS", justify="right")
    table.add_column("STATUS")
    table.add_column("CREATED", style="dim")

    for d in datasets:
        status = d.get("status", "?")
        s_color = _C_SAND if status == "READY" else (_C_ROSE if status == "FAILED" else "dim")
        table.add_row(
            d.get("dataset_id", "?"),
            str(d.get("artifact_count", "?")),
            f"[{s_color}]{status}[/]",
            str(d.get("created_at", "?"))[:19],
        )

    console.print(table)
    total = result.get("total") or result.get("total_count") or len(datasets)
    console.print(f"[dim]{total} total dataset(s)[/]")


def cmd_dataset(args: argparse.Namespace) -> None:
    """Show details for a specific dataset."""
    result = _request("GET", f"/datasets/{args.dataset_id}")
    if args.json:
        print(json.dumps(result, indent=2))
        return
    console.print(f"[dim]dataset_id    [/] : [{_C_BLUE}]{result.get('dataset_id')}[/]")
    console.print(f"[dim]status        [/] : {result.get('status')}")
    console.print(f"[dim]artifact_count[/] : {result.get('artifact_count')}")
    console.print(f"[dim]version       [/] : {result.get('version', 1)}")
    console.print(f"[dim]created_at    [/] : {str(result.get('created_at', ''))[:19]}")
    manifest = result.get("manifest_uri")
    if manifest:
        console.print(f"[dim]manifest_uri  [/] : {manifest}")
    # Backend returns "files" (sample list); older shape used "sample_artifacts"
    files = result.get("files") or result.get("sample_artifacts") or []
    if files:
        console.print(f"\n[bold]Sample files[/] (first {len(files)}):")
        for f in files:
            fname = f.get("filename") or f.get("source_filename", "?")
            size = f.get("size_bytes") or f.get("size", 0)
            console.print(f"  [{_C_BLUE}]{fname:<40}[/]  [dim]{size:>10} B[/]")
    console.print("\n[bold]Run a job:[/]")
    console.print(f"  [{_C_SAND}]phi esmfold   --dataset-id {result.get('dataset_id')}[/]")
    console.print(f"  [{_C_SAND}]phi alphafold --dataset-id {result.get('dataset_id')}[/]")


def cmd_esmfold(args: argparse.Namespace) -> None:
    """Fast structure prediction from sequence (~1 min/sequence on A100)."""
    params: dict = {
        "num_recycles": args.recycles,
        "extract_confidence": not args.no_confidence,
    }
    if not getattr(args, "dataset_id", None):
        params["fasta_str"] = _read_fasta(args)
        if args.fasta_name:
            params["fasta_name"] = args.fasta_name
    _run_model_job("esmfold", params, args)


def cmd_alphafold(args: argparse.Namespace) -> None:
    """Structure prediction — monomer or multimer (separate chains with ':')."""
    models = [int(m) for m in args.models.split(",") if m.strip()]
    params: dict = {
        "models": models,
        "model_type": args.model_type,
        "msa_tool": args.msa_tool,
        "msa_databases": args.msa_databases,
        "template_mode": args.template_mode,
        "pair_mode": args.pair_mode,
        "num_recycles": args.recycles,
        "num_seeds": args.num_seeds,
        # --amber flag takes precedence; fall back to legacy --relax int
        "num_relax": 1 if args.amber else args.relax,
    }
    if not getattr(args, "dataset_id", None):
        fasta_str = _read_fasta(args)
        # Modal auto-detects multimer from ":" in the FASTA sequence
        params["fasta_str"] = fasta_str
        if ":" in fasta_str:
            console.print(f"  [dim]multimer mode ({fasta_str.count(':') + 1} chains)[/]")
    _run_model_job("alphafold", params, args)


def cmd_proteinmpnn(args: argparse.Namespace) -> None:
    """Sequence design via inverse folding (1–2 min for 10 sequences)."""
    params: dict = {
        "num_sequences": args.num_sequences,
        "temperature": args.temperature,
    }
    if not getattr(args, "dataset_id", None):
        if args.pdb:
            params["pdb_content"] = Path(args.pdb).read_text()
        elif args.pdb_gcs:
            params["pdb_gcs_uri"] = args.pdb_gcs
        else:
            _die("Provide --pdb FILE, --pdb-gcs GCS_URI, or --dataset-id DATASET_ID")
    if args.fixed:
        params["fixed_positions"] = args.fixed
    _run_model_job("proteinmpnn", params, args)


def cmd_esm2(args: argparse.Namespace) -> None:
    """Protein language model scoring: per-position log-likelihood and perplexity."""
    params: dict = {}
    if not getattr(args, "dataset_id", None):
        params["fasta_str"] = _read_fasta(args)
    if args.mask:
        params["mask_positions"] = args.mask
    _run_model_job("esm2", params, args)


def cmd_boltz(args: argparse.Namespace) -> None:
    """Biomolecular structure prediction for complexes, DNA, RNA, and small molecules."""
    params: dict = {
        "num_recycles": args.recycles,
        "use_msa": not args.no_msa,
    }
    if not getattr(args, "dataset_id", None):
        params["fasta_str"] = _read_fasta(args)
    _run_model_job("boltz", params, args)


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
    """Get icon for a tool, stripping MCP prefixes like mcp__biology__."""
    # Strip MCP namespace prefix (e.g. "mcp__biology__pubmed_search" → "pubmed_search")
    short = tool_name.split("__")[-1].replace("_tool", "")
    return _TOOL_ICONS.get(short, _TOOL_ICONS.get(tool_name, "🔧"))


def _build_research_section(
    question: str,
    report: str,
    citations_raw: str | None = None,
) -> str:
    """Return a dated markdown section for one research entry.

    Pure function — no I/O, no side effects.
    """
    from datetime import datetime as _dt

    timestamp = _dt.now().strftime("%Y-%m-%d %H:%M")
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


def _save_research_notes_locally(notes_file: "Path", section: str) -> None:
    """Append a section to the local notes file, creating it if necessary."""
    try:
        notes_file.parent.mkdir(parents=True, exist_ok=True)
        with notes_file.open("a", encoding="utf-8") as fh:
            fh.write(section)
        console.print(f"[{_C_SAND}]Notes appended[/] → {notes_file}")
    except Exception as exc:
        console.print(f"[{_C_ROSE}]Warning: could not write notes file:[/] {exc}")


def _sync_research_notes_to_gcs(dataset_id: str, section: str) -> None:
    """POST a research notes section to the cloud-backed API endpoint."""
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
    notes_file: "Path",
    question: str,
    report: str,
    citations_raw: str | None = None,
    dataset_id: str | None = None,
) -> None:
    """Build a research section, persist it locally, and optionally sync to GCS."""
    section = _build_research_section(question, report, citations_raw)
    _save_research_notes_locally(notes_file, section)
    if dataset_id:
        _sync_research_notes_to_gcs(dataset_id, section)


class _ResearchStreamState:
    """Mutable state accumulated while consuming an SSE research stream."""

    def __init__(self) -> None:
        self.tool_calls: list[str] = []
        self.report_parts: list[str] = []
        self.turn: int = 0
        self.done: bool = False
        self.errored: bool = False


def _render_research_event(event_type: str, data: dict, state: "_ResearchStreamState") -> None:
    """Print progress for a single parsed SSE event. No state mutations."""
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


def _dispatch_sse_event(event_type: str, data: dict, state: "_ResearchStreamState") -> None:
    """Update stream state for a single parsed SSE event. No I/O."""
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


def _parse_sse_line(line: bytes) -> tuple[str, str] | None:
    """Parse one raw SSE line into (field, value) or None for blank/irrelevant lines."""
    decoded = line.decode("utf-8", errors="replace").rstrip("\n")
    if decoded.startswith("event:"):
        return ("_event", decoded[6:].strip())
    if decoded.startswith("data:"):
        return ("_data", decoded[5:].strip())
    return None


def _stream_research(
    question: str,
    base_url: str,
    max_turns: int = 15,
    notes_file: "Path | None" = None,
    dataset_id: str | None = None,
) -> None:
    """Stream research progress live from the Modal SSE endpoint."""
    import urllib.parse

    url = f"{base_url.rstrip('/')}/research/stream?" + urllib.parse.urlencode(
        {"question": question, "max_turns": max_turns}
    )

    console.print(f"\n[{_C_BLUE}]Connecting to research stream…[/]")
    console.print(f"[dim]{url}[/]\n")

    state = _ResearchStreamState()
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})

    try:
        # Use certifi bundle when available, fall back to unverified for Modal endpoints.
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
                    # Blank line: dispatch the accumulated buffer as one event.
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

                    event_type = "message"  # reset for next event block
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

    # Display the assembled report.
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


def cmd_research(args: argparse.Namespace) -> None:
    """Submit a biological research query and return a structured report with citations."""
    # Resolve notes file (None when --no-save is set)
    notes_file: Path | None = None
    if not getattr(args, "no_save", False):
        raw_notes = getattr(args, "notes_file", "./research.md")
        notes_file = Path(raw_notes)

    dataset_id: str | None = getattr(args, "dataset_id", None)

    # ── streaming mode: hit Modal SSE endpoint directly ───────────────────────
    if getattr(args, "stream", False):
        _stream_research(
            question=args.question,
            base_url=getattr(
                args, "stream_url", "https://dynotx--research-agent-streaming-fastapi-app.modal.run"
            ),
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

    # Load prior context from a file if provided
    context_file = getattr(args, "context_file", None)
    if context_file:
        try:
            prior = Path(context_file).read_text(encoding="utf-8")
            existing = args.context or ""
            params["context"] = (prior + "\n\n" + existing).strip() if existing else prior
        except Exception as exc:
            console.print(f"[{_C_ROSE}]Warning: could not read context file:[/] {exc}")
    elif args.context:
        params["context"] = args.context

    result = _submit("research", params, run_id=args.run_id)
    _print_submission(result)
    if args.wait:
        console.print(f"\n[dim]Polling every {POLL_INTERVAL}s …[/]")
        final = _poll(result["job_id"])
        _print_status(final)

        # Try inline outputs first (report_md stored directly in job params)
        report = (final.get("outputs") or {}).get("report_md")
        citations_raw = (final.get("outputs") or {}).get("citations")

        if report:
            # Append to notes file (local + cloud sync if dataset_id set)
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
                    import json as _json

                    try:
                        citations = _json.loads(citations_raw)
                        (out / "citations.json").write_text(_json.dumps(citations, indent=2))
                        console.print(f"[{_C_SAND}]Citations written[/] → {out}/citations.json")
                    except Exception:
                        pass
            else:
                console.print("\n" + "─" * 60)
                console.print(report)
                if citations_raw:
                    import json as _json

                    try:
                        citations = _json.loads(citations_raw)
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
    """Download and display accumulated research notes for a dataset."""
    from rich.markdown import Markdown
    from rich.panel import Panel

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

    out_path = getattr(args, "out", None)
    if out_path:
        out = Path(out_path)
        # Accept either a direct .md path or a directory (saves as DIR/research.md)
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

    # Rich display
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


def cmd_login(args: argparse.Namespace) -> None:
    """Verify the configured API key and print connection details.

    Tries GET /auth/me first. If that endpoint is not yet deployed (404),
    falls back to a lightweight GET /jobs/ probe to confirm the key is accepted.
    """
    key = _api_key()
    masked = key[:8] + "…" if len(key) > 8 else key
    base = _base_url()

    # Primary: full identity from Clerk
    try:
        me = _request("GET", "/auth/me")
        if args.json:
            print(json.dumps(me, indent=2))
            return

        content = Text()
        content.append("✓ Logged in\n\n", style=f"bold {_C_SAND}")
        content.append("endpoint  ", style="dim")
        content.append(f"{base}\n")
        content.append("API key   ", style="dim")
        content.append(f"{masked}\n\n")
        content.append("Identity\n", style="bold")
        for label, key_name in [
            ("user_id     ", "user_id"),
            ("email       ", "email"),
            ("display_name", "display_name"),
            ("org_id      ", "org_id"),
            ("org_name    ", "org_name"),
        ]:
            val = me.get(key_name) or "—"
            content.append(f"  {label}  ", style="dim")
            content.append(f"{val}\n")
        content.append("\n")
        content.append("Tip: ", style="bold dim")
        content.append("cache these to skip /auth/me on uploads:\n", style="dim")
        content.append(
            f"  export DYNO_USER_ID={me.get('user_id', 'YOUR_USER_ID')}\n",
            style=f"dim {_C_BLUE}",
        )
        content.append(
            f"  export DYNO_ORG_ID={me.get('org_id', 'YOUR_ORG_ID')}",
            style=f"dim {_C_BLUE}",
        )
        console.print(
            Panel(content, title=f"[{_C_BLUE}]dyno phi[/]", border_style=_C_BLUE, padding=(1, 2))
        )
        return

    except PhiApiError as exc:
        if "404" not in str(exc):
            _die(str(exc))
        # 404 → endpoint not yet deployed; fall through to probe

    # Fallback: probe the jobs list endpoint to confirm the key is valid
    try:
        _request("GET", "/jobs/?page_size=1")
        if args.json:
            print(json.dumps({"status": "connected", "auth_me": "not_deployed"}, indent=2))
            return

        content = Text()
        content.append("✓ Logged in\n\n", style=f"bold {_C_SAND}")
        content.append("endpoint  ", style="dim")
        content.append(f"{base}\n")
        content.append("API key   ", style="dim")
        content.append(f"{masked}\n\n")
        content.append("Note: ", style="bold dim")
        content.append(
            "User identity will appear here once GET /auth/me is deployed on this environment.",
            style="dim",
        )
        console.print(
            Panel(content, title=f"[{_C_BLUE}]dyno phi[/]", border_style=_C_BLUE, padding=(1, 2))
        )

    except PhiApiError as probe_exc:
        msg = f"Authentication failed — {probe_exc}"
        if "401" in str(probe_exc) and key.startswith("ak_"):
            msg += (
                "\n  This endpoint may not yet accept Clerk API keys (ak_…). "
                "Check backend config or use the API that is wired to Clerk."
            )
        _die(msg)


def cmd_scores(args: argparse.Namespace) -> None:
    """Fetch and display the metrics table for a completed job."""
    import csv
    import io

    args.job_id = _resolve_job_id(args)
    s = _status(args.job_id)
    if s.get("status") not in ("completed", "failed"):
        _die(f"Job is '{s.get('status')}' — scores are only available after completion")

    run_id = s.get("run_id")
    if not run_id:
        _die("No run_id found for this job")

    # Fetch workflow artifacts to find scores/metrics files
    try:
        results = _request("GET", f"/runs/{run_id}/results")
    except PhiApiError as e:
        _die(f"Could not fetch results: {e}")

    workflow_artifacts = results.get("workflow_artifacts", {})
    artifact_files = results.get("artifact_files", [])

    # Look for a scores CSV or metrics artifact in workflow artifacts first
    scores_content: str | None = None
    scores_source: str = ""

    for key, val in workflow_artifacts.items():
        if isinstance(val, str) and key in ("scores_csv", "metrics_csv", "scores"):
            scores_content = val
            scores_source = f"workflow artifact '{key}'"
            break

    # If not found inline, look for a downloadable scores file in artifact_files
    scores_artifact: dict | None = None
    if not scores_content:
        for af in artifact_files:
            name = af.get("name") or af.get("filename") or ""
            if name.endswith((".csv", ".parquet")) and any(
                kw in name.lower() for kw in ("score", "metric", "report")
            ):
                scores_artifact = af
                scores_source = f"artifact file '{name}'"
                break

    if not scores_content and scores_artifact:
        # Fetch download URL and get the file bytes
        artifact_id = scores_artifact.get("artifact_id")
        url = scores_artifact.get("download_url") or scores_artifact.get("url")
        if not url and artifact_id:
            try:
                dl_resp = _request("GET", f"/artifacts/{artifact_id}/download")
                url = dl_resp.get("download_url")
            except PhiApiError:
                pass
        if url and not url.startswith("gs://"):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "phi-cli/1.0"}, method="GET"
                )
                with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
                    scores_content = resp.read().decode("utf-8", errors="replace")
            except Exception as exc:
                console.print(f"  [yellow]⚠ Could not fetch scores file: {exc}[/]")

    if not scores_content:
        console.print(f"[dim]No scores/metrics found for job {args.job_id}.[/]")
        console.print("[dim]The pipeline may not have produced a report yet.[/]")
        return

    console.print(
        f"\n[bold {_C_SAND}]Scores[/] from {scores_source}  [dim](job {args.job_id[:8]}…)[/]\n"
    )

    # Parse CSV and display as a rich table
    try:
        reader = csv.DictReader(io.StringIO(scores_content))
        rows = list(reader)
        if not rows:
            console.print("[dim]Score file is empty.[/]")
        else:
            fieldnames = list(rows[0].keys()) if rows else []
            # Sort by iptm or binder_plddt descending if present
            sort_col = next(
                (c for c in ("iptm", "complex_iptm", "binder_plddt", "plddt") if c in fieldnames),
                None,
            )
            if sort_col:
                rows.sort(key=lambda r: float(r.get(sort_col, 0) or 0), reverse=True)

            # Limit display to top-N rows
            display_rows = rows[: args.top]
            table = Table(
                box=rich_box.SIMPLE_HEAVY,
                show_header=True,
                header_style=f"bold {_C_SAND}",
            )
            for col in fieldnames:
                table.add_column(col, no_wrap=True)
            for row in display_rows:
                table.add_row(*[row.get(c, "") for c in fieldnames])
            console.print(table)
            if len(rows) > args.top:
                console.print(
                    f"[dim]Showing top {args.top} of {len(rows)} candidates"
                    f"{f' (sorted by {sort_col})' if sort_col else ''}.[/]"
                )
    except Exception as exc:
        # Fall back to raw print if CSV parsing fails
        console.print(scores_content[:2000])
        if len(scores_content) > 2000:
            console.print(f"[dim]… ({len(scores_content)} chars total)[/]")
        console.print(f"[dim]Note: Could not parse as CSV table: {exc}[/]")

    # Optionally save to file
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(scores_content)
        console.print(f"\n[{_C_SAND}]✓[/] Saved → [bold]{dest}[/bold]")


def cmd_status(args: argparse.Namespace) -> None:
    s = _status(args.job_id)
    if args.json:
        print(json.dumps(s, indent=2))
    else:
        _print_status(s)


def cmd_jobs(args: argparse.Namespace) -> None:
    params: dict = {"page_size": args.limit}
    if args.status:
        params["status"] = args.status
    if args.job_type:
        params["job_type"] = args.job_type
    query = "&".join(f"{k}={v}" for k, v in params.items())
    result = _request("GET", f"/jobs/?{query}")
    if args.json:
        print(json.dumps(result, indent=2))
        return
    jobs = result.get("jobs", [])
    if not jobs:
        console.print("[dim]No jobs found.[/]")
        return

    table = Table(box=rich_box.SIMPLE_HEAVY, show_header=True, header_style="bold dim")
    table.add_column("JOB ID", style=f"dim {_C_BLUE}", no_wrap=True)
    table.add_column("TYPE", style="bold")
    table.add_column("STATUS", no_wrap=True)
    table.add_column("CREATED", style="dim")

    for j in jobs:
        status = j.get("status", "?")
        color = _STATUS_COLOR.get(status, "")
        styled_status = f"[{color}]{status}[/]" if color else status
        table.add_row(
            j.get("job_id", "?"),
            j.get("job_type", "?"),
            styled_status,
            (j.get("created_at") or "?")[:19],
        )

    console.print(table)
    total = result.get("total_count", len(jobs))
    running = result.get("total_running", 0)
    pending = result.get("total_pending", 0)
    console.print(f"[dim]{total} total  ({running} running, {pending} pending)[/]")


def cmd_logs(args: argparse.Namespace) -> None:
    """Stream job logs (prints available log lines; --follow polls for new lines)."""
    url = f"{_base_url()}/api/v1/jobs/{args.job_id}/logs/stream"
    console.print(f"Streaming logs from [{_C_BLUE}]{url}[/]")
    console.print("(Note: EventSource auth requires token as query param on this endpoint)")
    console.print(f"  [dim]curl -N '{url}?x_api_key={_api_key()[:8]}...'[/]")


def cmd_cancel(args: argparse.Namespace) -> None:
    result = _request("DELETE", f"/jobs/{args.job_id}")
    console.print(f"[bold {_C_SAND}]✓[/] Cancel requested: {result.get('message', 'ok')}")


def cmd_use(args: argparse.Namespace) -> None:
    """Set the active dataset ID cached in .phi-state.json."""
    _save_state({"dataset_id": args.dataset_id})
    console.print(
        f"[bold {_C_SAND}]✓[/] Active dataset set to [{_C_BLUE}]{args.dataset_id}[/]\n"
        f"  [dim]Saved to .phi-state.json — all commands will use this dataset by default.[/]"
    )


# Metrics available from the current pipeline: ESMFold pLDDT, AF2 pTM/ipTM/iPAE, Binder RMSD.
# Metrics that require PyRosetta/FoldX (not yet generated) are excluded:
#   Binder_Energy_Score, Surface_Hydrophobicity, ShapeComplementarity, dG, dSASA,
#   n_InterfaceResidues, n_InterfaceHbonds, n_InterfaceUnsatHbonds, Binder_Loop%, InterfaceAAs,
#   Hotspot_RMSD.
_FILTER_PRESETS: dict[str, dict[str, float]] = {
    "default": {
        # ESMFold binder pLDDT (lower bound)
        "plddt_threshold": 0.80,
        # AlphaFold2 complex pTM (lower bound)
        "ptm_threshold": 0.55,
        # AlphaFold2 interface pTM (lower bound)
        "iptm_threshold": 0.50,
        # AlphaFold2 interface PAE (upper bound — lower is better)
        "ipae_threshold": 0.35,
        # Binder backbone RMSD vs design (upper bound)
        "rmsd_threshold": 3.5,
    },
    "relaxed": {
        "plddt_threshold": 0.80,
        "ptm_threshold": 0.45,
        "iptm_threshold": 0.50,
        "ipae_threshold": 0.40,
        "rmsd_threshold": 4.5,
    },
}


def cmd_filter(args: argparse.Namespace) -> None:
    """Submit binder scoring and filter pipeline: inverse folding → folding → complex folding → score."""
    dataset_id = _resolve_dataset_id(args)

    # Start from preset values if --preset is given, then let explicit flags win.
    # Falls back to _FILTER_PRESETS["default"] so there is exactly one source of truth
    # for threshold defaults — updating the preset automatically updates the fallback.
    preset_name: str | None = getattr(args, "preset", None)
    base: dict[str, float] = dict(_FILTER_PRESETS[preset_name]) if preset_name else {}

    def _resolve_threshold(name: str) -> float:
        explicit: float | None = getattr(args, name, None)
        if explicit is not None:
            return explicit
        return base.get(name, _FILTER_PRESETS["default"][name])

    if preset_name:
        p = base
        console.print(
            f"[dim]Using [{_C_SAND}]{preset_name}[/{_C_SAND}] filter preset  "
            f"pLDDT≥{p['plddt_threshold']}  "
            f"pTM≥{p['ptm_threshold']}  "
            f"ipTM≥{p['iptm_threshold']}  "
            f"iPAE≤{p['ipae_threshold']}  "
            f"RMSD≤{p['rmsd_threshold']}Å  "
            f"[italic](override with explicit flags)[/italic][/]"
        )

    params: dict = {
        "num_sequences": args.num_sequences,
        "plddt_threshold": _resolve_threshold("plddt_threshold"),
        "iptm_threshold": _resolve_threshold("iptm_threshold"),
        "ipae_threshold": _resolve_threshold("ipae_threshold"),
        "ptm_threshold": _resolve_threshold("ptm_threshold"),
        "rmsd_threshold": _resolve_threshold("rmsd_threshold"),
        "num_recycles": args.num_recycles,
    }
    msa_tool: str = getattr(args, "msa_tool", "mmseqs2")
    params["msa_tool"] = msa_tool
    if msa_tool == "single_sequence":
        console.print(
            "  [dim]AF2 running in single-sequence mode (no MSA) — faster, "
            "better-calibrated for novel designed binders[/]"
        )

    result = _submit("design_pipeline", params, run_id=None, dataset_id=dataset_id)
    job_id = _require_key(result, "job_id", "POST /jobs (filter)")
    _save_state({"last_job_id": job_id})
    _print_submission(result)

    if args.wait or args.out:
        final = _poll(job_id)
        _print_status(final)
        if final.get("status") == "completed":
            run_id_final = final.get("run_id")
            if run_id_final:
                try:
                    results = _request("GET", f"/runs/{run_id_final}/results")
                    final["_results"] = results
                except PhiApiError:
                    pass
            _print_filter_done(job_id, final, thresholds=params)
        if args.out:
            _download_job(final, args.out, all_files=getattr(args, "all", False))


def cmd_download(args: argparse.Namespace) -> None:
    args.job_id = _resolve_job_id(args)
    s = _status(args.job_id)
    if s.get("status") != "completed":
        _die(f"Job is '{s.get('status')}' — can only download completed jobs")
    # Fetch richer results (includes artifact_files and workflow_artifacts)
    run_id = s.get("run_id")
    if run_id:
        try:
            results = _request("GET", f"/runs/{run_id}/results")
            s["_results"] = results
        except PhiApiError:
            pass  # Fall back to output_files from status
    _download_job(s, args.out, all_files=getattr(args, "all", False))


# ─────────────────────────── helpers ─────────────────────────────────────────


def _read_fasta(args: argparse.Namespace) -> str:
    if hasattr(args, "fasta_str") and args.fasta_str:
        return str(args.fasta_str)
    if hasattr(args, "fasta") and args.fasta:
        return Path(args.fasta).read_text()
    _die("Provide --fasta FILE or --fasta-str '>name\\nSEQUENCE'")


def _fetch_url_to_file(url: str, dest: Path) -> None:
    """Download bytes from a URL (e.g. a signed GCS URL) to a local file."""
    req = urllib.request.Request(url, headers={"User-Agent": "phi-cli/1.0"}, method="GET")
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:
        dest.write_bytes(resp.read())


# Artifact types that are downloaded by default (useful outputs only).
# MSA files, zip archives, and shell scripts are skipped unless --all is passed.
_DOWNLOAD_KEY_TYPES = {"pdb", "csv", "colabfold_scores", "af2m_scores", "json"}

# Map artifact type → subdirectory within the output folder.
_DOWNLOAD_SUBDIR: dict[str, str] = {
    "pdb": "structures",
    "csv": "scores",
    "colabfold_scores": "scores/raw",
    "af2m_scores": "scores/raw",
    "json": "scores/raw",
}


def _fmt_size(total_bytes: int) -> str:
    """Format a byte count as a human-readable string (B / KB / MB)."""
    if total_bytes >= 1_048_576:
        return f"{total_bytes / 1_048_576:.1f} MB"
    if total_bytes >= 1024:
        return f"{total_bytes / 1024:.0f} KB"
    return f"{total_bytes} B"


def _get_artifact_url(af: dict) -> str | None:
    """
    Return a signed HTTPS download URL for an artifact dict.

    Tries the following in order:
      1. Pre-signed HTTPS URL already present in the dict.
      2. Call GET /artifacts/{artifact_id}/download to have the server
         generate and return a signed URL.

    Returns None if no URL can be obtained; callers surface this as an
    error to the user.
    """
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


def _download_job(status: dict, out_dir: str, all_files: bool = False) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Prefer artifact_files from /runs/{run_id}/results over output_files from status
    results = status.get("_results") or {}
    artifact_files: list[dict] = results.get("artifact_files") or []
    workflow_artifacts: dict = results.get("workflow_artifacts") or {}
    output_files: list[dict] = status.get("output_files") or []

    if artifact_files:
        # ── 1. Group by type for summary ──────────────────────────────────
        groups: dict[str, list[dict]] = defaultdict(list)
        for af in artifact_files:
            atype = af.get("artifact_type") or "file"
            groups[atype].append(af)

        # Decide which types to download
        if all_files:
            selected_types = set(groups.keys())
        else:
            selected_types = _DOWNLOAD_KEY_TYPES & set(groups.keys())

        to_download = [
            af for af in artifact_files if (af.get("artifact_type") or "file") in selected_types
        ]
        to_skip = [
            af for af in artifact_files if (af.get("artifact_type") or "file") not in selected_types
        ]

        # ── 2. Summary table ──────────────────────────────────────────────
        summary_table = Table(
            box=rich_box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            show_edge=False,
            padding=(0, 1),
        )
        summary_table.add_column("Type", style=f"{_C_BLUE}", no_wrap=True)
        summary_table.add_column("Files", justify="right", style="dim")
        summary_table.add_column("Size", justify="right", style="dim")
        summary_table.add_column("", style="dim")

        for atype in sorted(groups.keys(), key=lambda t: (t not in selected_types, t)):
            items = groups[atype]
            total_bytes = sum(af.get("size_bytes") or 0 for af in items)
            will_dl = atype in selected_types
            action = (
                f"→ {_DOWNLOAD_SUBDIR.get(atype, '.')}/"
                if will_dl
                else "[dim]skip (use --all)[/dim]"
            )
            style = "" if will_dl else "dim"
            summary_table.add_row(
                f"[{style}]{atype}[/{style}]" if style else atype,
                str(len(items)),
                _fmt_size(total_bytes),
                action,
            )

        console.print(summary_table)

        if not all_files and to_skip:
            console.print(
                f"  [dim]{len(to_skip)} file(s) skipped"
                f" (msa/zip/scripts) — pass [bold]--all[/bold] to include[/]"
            )

        if not to_download:
            console.print("  [dim]Nothing to download.[/]")
        else:
            # ── 3. Download selected files with progress ───────────────────
            downloaded, errors = 0, 0
            counts_by_subdir: dict[str, int] = defaultdict(int)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                task = progress.add_task(
                    f"[{_C_BLUE}]Downloading {len(to_download)} file(s)…",
                    total=len(to_download),
                )
                for af in to_download:
                    name = af.get("name") or af.get("filename") or af.get("artifact_id", "artifact")
                    atype = af.get("artifact_type") or "file"
                    subdir = _DOWNLOAD_SUBDIR.get(atype, ".")
                    dest_dir = out / subdir
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
                            f"  [{_C_ROSE}]⚠ {Path(name).name}: "
                            f"server could not generate a download URL[/]"
                        )
                        errors += 1
                    progress.advance(task)

            # ── 4. Result summary panel ────────────────────────────────────
            result_content = Text()
            result_content.append(f"{downloaded} file(s) saved to ", style="bold")
            result_content.append(f"{out}/\n\n", style=f"bold {_C_BLUE}")
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

        # Always write manifest for reference
        manifest = out / "manifest.json"
        manifest.write_text(json.dumps(artifact_files, indent=2))
        console.print(f"  [dim]manifest → {manifest}[/]")

    elif workflow_artifacts:
        console.print("\n[bold]Workflow artifacts:[/]")
        for key, val in workflow_artifacts.items():
            console.print(f"  [{_C_BLUE}]{key}[/]: {val}")
        manifest = out / "manifest.json"
        manifest.write_text(json.dumps(workflow_artifacts, indent=2))
        console.print(f"  [dim]manifest written → {manifest}[/]")

    elif output_files:
        console.print(f"\n[bold]Output files[/] ({len(output_files)}) — stored in cloud:")
        for f in output_files:
            name = f.get("name", "")
            val = f.get("value", "")
            if isinstance(val, list):
                for v in val:
                    console.print(f"  [{_C_BLUE}]{v}[/]")
            else:
                console.print(f"  [{_C_BLUE}]{name}[/]: {val}")
        manifest = out / "manifest.json"
        manifest.write_text(json.dumps(output_files, indent=2))
        console.print(f"  [dim]manifest written → {manifest}[/]")

    else:
        console.print(f"  [dim]No output files found for job {status.get('job_id')}[/]")
        console.print(f"  [dim]run_id: {status.get('run_id')}[/]")
        console.print(f"  Check: [{_C_BLUE}]phi status {status.get('job_id')} --json[/]")


# ─────────────────────────── argument parser ─────────────────────────────────


def _add_fasta_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fasta", metavar="FILE", help="FASTA file to submit")
    g.add_argument("--fasta-str", metavar="FASTA", help="FASTA content as a string (for scripting)")
    g.add_argument(
        "--dataset-id",
        metavar="DATASET_ID",
        help="Pre-ingested dataset ID (for batch runs of 100–50,000 files)",
    )


def _add_pdb_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pdb", metavar="FILE", help="PDB structure file")
    g.add_argument("--pdb-gcs", metavar="URI", help="Cloud storage URI to PDB (gs://…)")
    g.add_argument(
        "--dataset-id",
        metavar="DATASET_ID",
        help="Pre-ingested dataset ID (for batch runs of 100–50,000 files)",
    )


def _add_job_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-id", metavar="ID", help="Optional run label")
    p.add_argument(
        "--wait", action="store_true", default=True, help="Poll until job completes (default: on)"
    )
    p.add_argument(
        "--no-wait", action="store_false", dest="wait", help="Return immediately after submission"
    )
    p.add_argument("--out", metavar="DIR", help="Download results to DIR when done")
    p.add_argument("--json", action="store_true", help="Output raw JSON")


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="phi",
        description="Dyno Phi protein design platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    root.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    root.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        metavar="S",
        help=f"Seconds between status-poll requests (default: {POLL_INTERVAL})",
    )
    sub = root.add_subparsers(dest="command", required=True)

    # ── login ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("login", help="Verify API key and print connection + identity details")
    p.add_argument("--json", action="store_true")

    # ── upload ────────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "upload",
        help="Upload files → ingest → dataset (batch workflow entry point)",
    )
    p.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="Files to upload (positional). Use --dir for directories.",
    )
    p.add_argument("--dir", metavar="DIR", help="Upload all matching files in this directory")
    p.add_argument(
        "--file-type",
        metavar="TYPE",
        help="Override auto-detected file type (pdb, cif, fasta, csv). "
        "When omitted, type is inferred from file extensions.",
    )
    p.add_argument(
        "--gcs",
        metavar="URI",
        help="[Future] Import from external cloud storage (gs://bucket/prefix/)",
    )
    p.add_argument("--run-id", metavar="ID", help="Label for this ingest session")
    p.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Poll until dataset is READY (default: on)",
    )
    p.add_argument(
        "--no-wait",
        action="store_false",
        dest="wait",
        help="Return after finalizing without polling",
    )

    # ── datasets ──────────────────────────────────────────────────────────────
    p = sub.add_parser("datasets", help="List your datasets")
    p.add_argument("--limit", type=int, default=20, metavar="N")
    p.add_argument("--json", action="store_true")

    # ── dataset ───────────────────────────────────────────────────────────────
    p = sub.add_parser("dataset", help="Show details for a dataset")
    p.add_argument("dataset_id")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("ingest-session", help="Show status of an ingest session")
    p.add_argument("session_id", metavar="SESSION_ID")
    p.add_argument("--json", action="store_true")

    # ── esmfold ───────────────────────────────────────────────────────────────
    p = sub.add_parser("esmfold", aliases=["folding"], help="Fast structure prediction (~1 min)")
    _add_fasta_args(p)
    p.add_argument(
        "--recycles", type=int, default=3, metavar="N", help="Recycling iterations (default: 3)"
    )
    p.add_argument("--no-confidence", action="store_true", help="Skip per-residue pLDDT extraction")
    p.add_argument(
        "--fasta-name",
        metavar="NAME",
        help="Name label for output files (single-sequence mode only)",
    )
    _add_job_args(p)

    # ── alphafold ─────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "alphafold",
        aliases=["complex_folding"],
        help="Structure prediction — monomer or multimer (8–15 min)",
    )
    _add_fasta_args(p)
    p.add_argument(
        "--models", default="1,2,3", metavar="1,2,3", help="Model numbers to run (default: 1,2,3)"
    )
    p.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "ptm", "multimer_v1", "multimer_v2", "multimer_v3"],
        help="Model type — auto picks ptm for monomers, multimer_v3 for complexes (default: auto)",
    )
    p.add_argument(
        "--msa-tool",
        default="mmseqs2",
        choices=["mmseqs2", "jackhmmer"],
        help="MSA algorithm (default: mmseqs2)",
    )
    p.add_argument(
        "--msa-databases",
        default="uniref_env",
        choices=["uniref_env", "uniref_only"],
        help="Database set searched for MSA (default: uniref_env)",
    )
    p.add_argument(
        "--template-mode",
        default="none",
        choices=["none", "pdb70"],
        help="Template structure lookup mode (default: none)",
    )
    p.add_argument(
        "--pair-mode",
        default="unpaired_paired",
        choices=["unpaired_paired", "paired", "unpaired"],
        help="MSA pairing strategy for complexes (default: unpaired+paired)",
    )
    p.add_argument(
        "--recycles", type=int, default=6, metavar="N", help="Recycling iterations (default: 6)"
    )
    p.add_argument(
        "--num-seeds", type=int, default=3, metavar="N", help="Number of model seeds (default: 3)"
    )
    p.add_argument(
        "--amber",
        action="store_true",
        help="Run AMBER force-field relaxation (removes stereochemical violations)",
    )
    p.add_argument(
        "--relax",
        type=int,
        default=0,
        metavar="N",
        help="Amber relaxation passes as int (legacy; prefer --amber)",
    )
    _add_job_args(p)

    # ── proteinmpnn ───────────────────────────────────────────────────────────
    p = sub.add_parser(
        "proteinmpnn",
        aliases=["inverse_folding"],
        help="Sequence design via inverse folding (1–2 min)",
    )
    _add_pdb_args(p)
    p.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        metavar="N",
        help="Sequences to design (default: 10)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        metavar="T",
        help="Sampling temperature 0–1 (default: 0.1)",
    )
    p.add_argument("--fixed", metavar="A52,A56", help="Fixed residue positions e.g. A52,A56,A63")
    _add_job_args(p)

    # ── esm2 ──────────────────────────────────────────────────────────────────
    p = sub.add_parser("esm2", help="Language model scoring: log-likelihood and perplexity")
    _add_fasta_args(p)
    p.add_argument(
        "--mask", metavar="5,10,15", help="Comma-separated positions to mask for scoring"
    )
    _add_job_args(p)

    # ── boltz ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("boltz", help="Biomolecular complex prediction — proteins, DNA, RNA")
    _add_fasta_args(p)
    p.add_argument("--recycles", type=int, default=3, metavar="N")
    p.add_argument("--no-msa", action="store_true", help="Disable MSA (faster, lower accuracy)")
    _add_job_args(p)

    # ── research ──────────────────────────────────────────────────────────────
    p = sub.add_parser("research", help="Biological research query with citations (2–5 min)")
    p.add_argument(
        "--question",
        required=True,
        metavar="QUESTION",
        help="Research question, e.g. 'What are known binding hotspots for PD-L1?'",
    )
    p.add_argument(
        "--target",
        metavar="TARGET",
        help="Protein or gene name to focus the search (e.g. PD-L1, KRAS, EGFR)",
    )
    p.add_argument(
        "--databases",
        default="pubmed,uniprot,pdb",
        metavar="pubmed,uniprot,pdb",
        help="Comma-separated databases to query (default: pubmed,uniprot,pdb)",
    )
    p.add_argument(
        "--max-papers",
        type=int,
        default=20,
        metavar="N",
        help="Maximum papers to retrieve from PubMed (default: 20)",
    )
    p.add_argument(
        "--structures", action="store_true", help="Include related PDB structures in report"
    )
    p.add_argument("--context", metavar="TEXT", help="Additional context for the research query")
    p.add_argument(
        "--context-file",
        metavar="FILE",
        help="Path to a prior research.md file — prepended as context for this query",
    )
    p.add_argument(
        "--dataset-id",
        metavar="ID",
        help="Associate notes with a dataset and sync to cloud storage",
    )
    p.add_argument(
        "--notes-file",
        metavar="FILE",
        default="./research.md",
        help="Local append-only notes file (default: ./research.md)",
    )
    p.add_argument(
        "--no-save", action="store_true", help="Skip saving the report to the local notes file"
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help="Stream results live from Modal SSE endpoint (skips job tracking)",
    )
    p.add_argument(
        "--stream-url",
        metavar="URL",
        default="https://dynotx--research-agent-streaming-fastapi-app.modal.run",
        help="Override the SSE base URL",
    )
    _add_job_args(p)

    # ── notes ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("notes", help="View accumulated research notes for a dataset")
    p.add_argument(
        "dataset_id", metavar="DATASET_ID", help="Dataset ID to retrieve research notes for"
    )
    p.add_argument(
        "--out",
        metavar="PATH",
        help="Save notes to PATH (.md file) or PATH/research.md (directory) instead of printing",
    )
    p.add_argument("--json", action="store_true", help="Output raw JSON")

    # ── job management ────────────────────────────────────────────────────────
    p = sub.add_parser("status", help="Get job status")
    p.add_argument("job_id")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("jobs", help="List recent jobs")
    p.add_argument("--limit", type=int, default=20, metavar="N")
    p.add_argument(
        "--status",
        metavar="STATUS",
        choices=["pending", "running", "completed", "failed", "cancelled"],
    )
    p.add_argument("--job-type", metavar="TYPE")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("logs", help="Print log stream URL for a job")
    p.add_argument("job_id")
    p.add_argument("--follow", action="store_true")

    p = sub.add_parser("cancel", help="Cancel a running job")
    p.add_argument("job_id")

    p = sub.add_parser(
        "use",
        help="Set the active dataset ID (cached in .phi-state.json, used as default for filter etc.)",
    )
    p.add_argument("dataset_id", metavar="DATASET_ID")

    p = sub.add_parser("download", help="Download output files for a completed job")
    p.add_argument("job_id", nargs="?", default=None, help="Job ID (default: last cached job)")
    p.add_argument("--out", default="./results", metavar="DIR")
    p.add_argument(
        "--all",
        action="store_true",
        help="Download all artifact types including MSA files, zip archives, and scripts",
    )

    p = sub.add_parser(
        "filter",
        help="Filter and score binder designs (inverse folding → folding → complex folding → score)",
    )
    p.add_argument(
        "--dataset-id",
        default=None,
        metavar="ID",
        help="Dataset of PDB/CIF designs (default: cached from last upload or `phi use`)",
    )
    p.add_argument(
        "--preset",
        choices=list(_FILTER_PRESETS),
        metavar="NAME",
        help=f"Named filter preset: {', '.join(_FILTER_PRESETS)} "
        f"(individual flags override preset values)",
    )
    p.add_argument(
        "--num-sequences",
        type=int,
        default=4,
        metavar="N",
        help="ProteinMPNN sequences per design (default: 4)",
    )
    p.add_argument(
        "--plddt-threshold",
        type=float,
        default=None,
        metavar="F",
        help="ESMFold binder pLDDT lower-bound cutoff (default preset: 0.80)",
    )
    p.add_argument(
        "--iptm-threshold",
        type=float,
        default=None,
        metavar="F",
        help="AlphaFold2 interface pTM lower-bound cutoff (default preset: 0.50)",
    )
    p.add_argument(
        "--ipae-threshold",
        type=float,
        default=None,
        metavar="F",
        help="AlphaFold2 interface PAE upper-bound cutoff (default preset: 0.35)",
    )
    p.add_argument(
        "--ptm-threshold",
        type=float,
        default=None,
        metavar="F",
        help="AlphaFold2 complex pTM lower-bound cutoff (default preset: 0.55)",
    )
    p.add_argument(
        "--rmsd-threshold",
        type=float,
        default=None,
        metavar="F",
        help="Binder backbone RMSD upper-bound cutoff in Å (default preset: 3.5)",
    )
    p.add_argument(
        "--num-recycles",
        type=int,
        default=3,
        metavar="N",
        help="AlphaFold2 recycle iterations (default: 3)",
    )
    p.add_argument("--run-id", metavar="ID", help="Optional custom run ID")
    p.add_argument("--wait", action="store_true", help="Poll until pipeline completes")
    p.add_argument("--out", metavar="DIR", help="Download results on completion")
    p.add_argument(
        "--all",
        action="store_true",
        help="When --out is set, download all artifact types including MSA files and archives",
    )
    p.add_argument(
        "--msa-tool",
        default="mmseqs2",
        choices=["mmseqs2", "jackhmmer", "single_sequence"],
        metavar="TOOL",
        help=(
            "MSA algorithm for AF2 complex prediction: mmseqs2 (default), jackhmmer, or "
            "single_sequence (skips MSA — ~4x faster, better-calibrated for novel designed "
            "binders that have no natural sequence homologs). "
            "Matches the --msa-tool option on 'phi alphafold'."
        ),
    )

    p = sub.add_parser("scores", help="Display scoring metrics table for a completed job")
    p.add_argument("job_id", nargs="?", default=None, help="Job ID (default: last cached job)")
    p.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Show top-N candidates (default: 20)",
    )
    p.add_argument("--out", metavar="FILE", help="Save scores CSV to file")
    p.add_argument("--json", action="store_true", help="Output raw JSON")

    return root


COMMANDS = {
    "login": cmd_login,
    "upload": cmd_upload,
    "ingest-session": cmd_ingest_session,
    "datasets": cmd_datasets,
    "dataset": cmd_dataset,
    "esmfold": cmd_esmfold,
    "folding": cmd_esmfold,  # alias
    "alphafold": cmd_alphafold,
    "complex_folding": cmd_alphafold,  # alias
    "proteinmpnn": cmd_proteinmpnn,
    "inverse_folding": cmd_proteinmpnn,  # alias
    "esm2": cmd_esm2,
    "boltz": cmd_boltz,
    "research": cmd_research,
    "notes": cmd_notes,
    "status": cmd_status,
    "jobs": cmd_jobs,
    "logs": cmd_logs,
    "cancel": cmd_cancel,
    "use": cmd_use,
    "download": cmd_download,
    "filter": cmd_filter,
    "scores": cmd_scores,
}


def _print_state_footer() -> None:
    """Print a one-line reminder of what is cached in .phi-state.json.

    Shown after every successful command so users always know what dataset
    and job they are working with, and what commands are available next.
    Only printed when at least one value is cached.
    """
    state = _load_state()
    dataset_id = state.get("dataset_id")
    job_id = state.get("last_job_id")
    if not dataset_id and not job_id:
        return

    parts: list[str] = []
    if dataset_id:
        parts.append(f"dataset [{_C_BLUE}]{dataset_id}[/{_C_BLUE}]")
    if job_id:
        parts.append(f"job [{_C_BLUE}]{job_id}[/{_C_BLUE}]")

    console.print(f"[dim]Active: {' · '.join(parts)}[/dim]")


def main() -> None:
    global POLL_INTERVAL
    parser = build_parser()
    args = parser.parse_args()
    if args.poll_interval is not None:
        POLL_INTERVAL = args.poll_interval
    try:
        COMMANDS[args.command](args)
        _print_state_footer()
    except PhiApiError as exc:
        _die(str(exc))


if __name__ == "__main__":
    main()
