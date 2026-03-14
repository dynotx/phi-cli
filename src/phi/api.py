import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

from phi.config import (
    _UPLOAD_RETRIES,
    _UPLOAD_RETRY_BASE,
    _base_url,
    _require_api_key,
    _ssl_context,
)
from phi.display import _die
from phi.types import PhiApiError


def _require_key(d: dict, key: str, context: str) -> str:
    value = d.get(key)
    if not value:
        _die(f"API response for {context} is missing required field '{key}': {d}")
    assert isinstance(value, str)
    return value


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{_base_url()}/v1/phi{path}"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": _require_api_key(),
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


def _put_file(signed_url: str, data: bytes | Path) -> None:
    payload = data if isinstance(data, bytes) else data.read_bytes()
    label = "<bytes>" if isinstance(data, bytes) else data.name
    last_exc: Exception | None = None

    for attempt in range(_UPLOAD_RETRIES):
        req = urllib.request.Request(
            signed_url,
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            method="PUT",
        )
        try:
            with urllib.request.urlopen(req, timeout=300, context=_ssl_context()) as resp:
                _ = resp.read()
            return
        except urllib.error.HTTPError as e:
            if e.code in {429, 500, 502, 503, 504} and attempt < _UPLOAD_RETRIES - 1:
                time.sleep(_UPLOAD_RETRY_BASE**attempt)
                last_exc = e
                continue
            raise RuntimeError(f"Upload failed for {label}: HTTP {e.code}") from e
        except urllib.error.URLError as e:
            if attempt < _UPLOAD_RETRIES - 1:
                time.sleep(_UPLOAD_RETRY_BASE**attempt)
                last_exc = e
                continue
            raise RuntimeError(f"Upload failed for {label}: {e.reason}") from e

    raise RuntimeError(
        f"Upload failed for {label} after {_UPLOAD_RETRIES} attempts"
    ) from last_exc


def _resolve_identity() -> None:
    if os.environ.get("DYNO_USER_ID") and os.environ.get("DYNO_ORG_ID"):
        return
    try:
        me = _request("GET", "/auth/me")
        if not os.environ.get("DYNO_USER_ID"):
            os.environ["DYNO_USER_ID"] = me.get("user_id") or ""
        if not os.environ.get("DYNO_ORG_ID"):
            os.environ["DYNO_ORG_ID"] = me.get("org_id") or ""
    except PhiApiError:
        pass


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


def _ensure_authenticated() -> None:
    try:
        _request("GET", "/jobs/?page_size=1")
    except PhiApiError as e:
        if "401" in str(e):
            _die("Not authenticated. Run 'phi login' to verify your API key and endpoint.")
        raise
