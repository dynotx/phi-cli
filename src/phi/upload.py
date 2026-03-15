import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from phi.api import _put_file, _request
from phi.config import UPLOAD_BATCH_SIZE, UPLOAD_WORKERS
from phi.display import _C_ROSE, _die, _make_upload_progress, console


def _collect_files(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []

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

    seen_resolved: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen_resolved:
            seen_resolved.add(rp)
            unique.append(p)

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


def _create_ingest_session(files: list[Path], args: argparse.Namespace) -> str:
    body: dict = {"expected_files": len(files)}
    if args.run_id:
        body["run_id"] = args.run_id
    if getattr(args, "file_type", None):
        body["file_type"] = args.file_type
    elif files:
        body["file_type"] = files[0].suffix.lstrip(".")
    session = _request("POST", "/ingest_sessions/", body)
    session_id = session.get("session_id") or session.get("id")
    if not session_id:
        _die(f"POST /ingest_sessions response missing 'session_id': {session}")
    assert isinstance(session_id, str)
    console.print(f"  [dim]session_id[/] : {session_id}")
    return session_id


def _request_signed_urls(session_id: str, files: list[Path]) -> dict[str, str]:
    total = len(files)
    console.print(f"  Requesting signed URLs ({total} file(s)) …")
    url_map: dict[str, str] = {}
    # Batch requests to stay within the 50-file-per-call limit.
    for i in range(0, total, UPLOAD_BATCH_SIZE):
        batch = files[i : i + UPLOAD_BATCH_SIZE]
        resp = _request(
            "POST",
            f"/ingest_sessions/{session_id}/upload_urls",
            {"files": [f.name for f in batch]},
        )
        for entry in resp.get("urls", []):
            filename = entry.get("file")
            url = entry.get("url")
            if filename and url:
                url_map[str(filename)] = str(url)
    if len(url_map) != total:
        _die(
            f"Expected {total} signed URLs but received {len(url_map)}. "
            "Check the /ingest_sessions/{session_id}/upload_urls endpoint."
        )
    return url_map


def _upload_all_parallel(files: list[Path], url_map: dict[str, str]) -> list[str]:
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

    with _make_upload_progress() as progress:
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
