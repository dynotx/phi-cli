from __future__ import annotations

import argparse
import json
import os
import ssl
from pathlib import Path

from phi.display import _C_BLUE, _die, console

DEFAULT_BASE_URL = "https://api.dyno-agents.app"
POLL_INTERVAL: float = 5
POLL_TIMEOUT = 7200
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
# "submitted" kept for backward compat with jobs created before the 2026-03-07 backend fix.
NON_TERMINAL_STATUSES = {"pending", "submitted", "running"}
INGEST_TERMINAL = {"READY", "FAILED"}
UPLOAD_BATCH_SIZE = 50
UPLOAD_WORKERS = 8
STATE_DIR = Path(".phi")
STATE_FILE = STATE_DIR / "state.json"

_UPLOAD_RETRIES = 3
_UPLOAD_RETRY_BASE = 2.0

_MACOS_CERT_PATHS = [
    "/opt/homebrew/etc/openssl@3/cert.pem",
    "/opt/homebrew/etc/openssl@1.1/cert.pem",
    "/etc/ssl/cert.pem",
    "/usr/local/etc/openssl/cert.pem",
]


def _load_state() -> dict[str, object]:
    try:
        return dict(json.loads(STATE_FILE.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_state(updates: dict) -> None:
    state = _load_state()
    state.update(updates)
    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def _resolve_cached_id(
    args: argparse.Namespace,
    attr: str,
    state_key: str,
    missing_hint: str,
) -> str:
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
    ctx = ssl.create_default_context()
    env_cafile = os.environ.get("SSL_CERT_FILE")
    if env_cafile and Path(env_cafile).exists():
        ctx.load_verify_locations(env_cafile)
        return ctx
    for cafile in _MACOS_CERT_PATHS:
        if Path(cafile).exists():
            ctx.load_verify_locations(cafile)
            return ctx
    return ctx


_FILTER_PRESETS: dict[str, dict] = {
    "default": {
        "plddt_threshold": 0.80,
        "ptm_threshold": 0.55,
        "iptm_threshold": 0.50,
        # iPAE is reported in Å (AF2 scale 0–31.75 Å).
        # Threshold derived from BindCraft normalized convention: ipAE_norm = ipAE_Å / 31
        # 0.35 (BindCraft default) × 31 = 10.85 Å ≈ ~11 Å (standard RFdiffusion cutoff)
        "ipae_threshold": 10.85,
        "rmsd_threshold": 3.5,
        # Designed binders have no natural homologs — single_sequence avoids MSA artifacts.
        "msa_tool": "single_sequence",
    },
    "relaxed": {
        "plddt_threshold": 0.80,
        "ptm_threshold": 0.45,
        "iptm_threshold": 0.50,
        # 0.40 (relaxed normalized) × 31 = 12.4 Å
        "ipae_threshold": 12.4,
        "rmsd_threshold": 4.5,
        "msa_tool": "single_sequence",
    },
}


_CACHED_API_KEY: str | None = None


def _require_api_key() -> str:
    global _CACHED_API_KEY
    if _CACHED_API_KEY:
        return _CACHED_API_KEY
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
        # Check state.json as a final fallback
        key = str(_load_state().get("api_key", "")) or None
    if not key:
        _die(
            "DYNO_API_KEY is not set.\n"
            "  1. Open https://design.dynotx.com/dashboard/settings → API keys\n"
            "  2. Create and copy an API key\n"
            "  3. Run: export DYNO_API_KEY=your_key"
        )
    _CACHED_API_KEY = key
    _save_state({"api_key": key})
    return key
