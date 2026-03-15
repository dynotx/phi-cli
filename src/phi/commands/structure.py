import argparse
from pathlib import Path

from phi.api import _ensure_authenticated, _put_file, _request, _resolve_identity
from phi.config import _save_state
from phi.display import _C_SAND, _die, console
from phi.polling import _ingest_poll
from phi.structure import (
    _afdb_pdb_url,
    _count_residues,
    _extract_chain,
    _fetch_pdb_afdb,
    _fetch_pdb_rcsb,
    _trim_low_confidence,
    _trim_residues,
)


def _fetch_pdb(args: argparse.Namespace) -> tuple[str, str]:
    if args.pdb:
        source_id = args.pdb.upper()
        console.print(f"  Downloading [bold]{source_id}[/bold] from RCSB PDB …")
        try:
            return _fetch_pdb_rcsb(source_id), f"https://files.rcsb.org/download/{source_id}.pdb"
        except ValueError as exc:
            _die(str(exc))

    source_id = args.uniprot.upper()
    console.print(f"  Downloading [bold]{source_id}[/bold] from AlphaFold DB …")
    try:
        return _fetch_pdb_afdb(source_id), _afdb_pdb_url(source_id)
    except ValueError as exc:
        _die(str(exc))


def _apply_crops(pdb: str, args: argparse.Namespace) -> str:
    if args.chain:
        console.print(f"  Extracting chain [bold]{args.chain.upper()}[/bold] …")
        try:
            pdb = _extract_chain(pdb, args.chain)
        except ValueError as exc:
            _die(str(exc))

    if args.residues:
        try:
            start_s, end_s = args.residues.split("-", 1)
            start, end = int(start_s.strip()), int(end_s.strip())
        except ValueError:
            _die(f"--residues must be in START-END format (e.g., 56-290), got: {args.residues}")
        console.print(f"  Trimming to residues [bold]{start}–{end}[/bold] …")
        pdb = _trim_residues(pdb, start, end)

    if args.trim_low_confidence is not None:
        if not args.uniprot:
            console.print(
                "  [dim]Warning: --trim-low-confidence is designed for AlphaFold DB structures "
                "(pLDDT in B-factor). Applying anyway.[/dim]"
            )
        console.print(f"  Trimming residues with pLDDT < [bold]{args.trim_low_confidence}[/bold] …")
        pdb = _trim_low_confidence(pdb, args.trim_low_confidence)

    return pdb


def _save_pdb(pdb: str, source_id: str, chain: str | None, out: str | None) -> Path:
    suffix = f"_{chain.upper()}" if chain else ""
    filename = out or f"{source_id}{suffix}.pdb"
    path = Path(filename)
    path.write_text(pdb)
    return path


def _create_upload_session(filename: str, name: str | None) -> str:
    body: dict = {"expected_files": 1, "file_type": "pdb"}
    if name:
        body["run_id"] = name
    session = _request("POST", "/ingest_sessions/", body)
    session_id = session.get("session_id") or session.get("id")
    if not session_id:
        _die(f"POST /ingest_sessions response missing 'session_id': {session}")
    assert isinstance(session_id, str)
    return session_id


def _get_signed_url(session_id: str, filename: str) -> str:
    resp = _request(
        "POST",
        f"/ingest_sessions/{session_id}/upload_urls",
        {"files": [filename]},
    )
    urls = resp.get("urls", [])
    url = urls[0].get("url") if urls else None
    if not url:
        _die(f"No signed URL returned for structure upload. Response: {resp}")
    assert isinstance(url, str)
    return url


def _resolve_gcs_uri(dataset_id: str, filename: str) -> str:
    dataset = _request("GET", f"/datasets/{dataset_id}")
    for f in dataset.get("sample_files", []):
        if f.get("filename") == filename:
            gcs_uri = str(f.get("gcs_uri", ""))
            if gcs_uri:
                return gcs_uri
    sample = dataset.get("sample_files", [{}])
    return str(sample[0].get("gcs_uri", "")) if sample else ""


def _upload_structure(pdb_content: str, filename: str, name: str | None) -> tuple[str, str]:
    session_id = _create_upload_session(filename, name)
    signed_url = _get_signed_url(session_id, filename)
    _put_file(signed_url, pdb_content.encode())
    _request("POST", f"/ingest_sessions/{session_id}/finalize", {})
    result = _ingest_poll(session_id)
    dataset_id = result.get("dataset_id", "")
    if not dataset_id:
        _die("Ingest session completed but returned no dataset_id")
    assert isinstance(dataset_id, str)
    gcs_uri = _resolve_gcs_uri(dataset_id, filename)
    return dataset_id, gcs_uri


def _print_fetch_next_steps(source_url: str, out_path: Path, raw_id: str) -> None:
    console.print()
    console.print(f"  [dim]Source: {source_url}[/dim]")
    console.print()
    console.print("  [dim]Next steps:[/dim]")
    console.print(
        f"  [dim]  phi design --target-pdb {out_path} --hotspots <A45,A67> --num-designs 50[/dim]"
    )
    console.print(
        f"  [dim]  phi fetch --pdb {raw_id} --upload  # upload to cloud for GCS URI[/dim]"
    )


def _fetch_and_upload(
    pdb_content: str,
    out_path: Path,
    source_url: str,
    name: str | None,
) -> None:
    _resolve_identity()
    _ensure_authenticated()
    console.print("\n  Uploading to cloud storage …")
    dataset_id, gcs_uri = _upload_structure(pdb_content, out_path.name, name)
    _save_state({"target_gcs_uri": gcs_uri, "target_dataset_id": dataset_id})
    console.print()
    console.rule(f"[{_C_SAND}]Structure ready[/]")
    for label, value in [
        ("Source  ", source_url),
        ("File    ", str(out_path)),
        ("Dataset ", dataset_id),
        ("GCS URI ", gcs_uri),
    ]:
        console.print(f"  [bold]{label}[/bold] {value}")
    console.print()
    console.print("  [dim]Next steps:[/dim]")
    console.print(
        f"  [dim]  phi design --target-pdb-gcs {gcs_uri} --hotspots <A45,A67> --num-designs 50[/dim]"
    )
    console.rule()


def cmd_fetch(args: argparse.Namespace) -> None:
    if not args.pdb and not args.uniprot:
        _die("Provide --pdb PDB_ID or --uniprot UNIPROT_ID")
    if args.pdb and args.uniprot:
        _die("--pdb and --uniprot are mutually exclusive")

    pdb_content, source_url = _fetch_pdb(args)

    counts_before = _count_residues(pdb_content)
    total_before = sum(counts_before.values())
    console.print(
        f"  [dim]Downloaded: {total_before} residues across chains "
        f"{', '.join(sorted(counts_before))}[/dim]"
    )

    source_id = (args.pdb or args.uniprot).upper()
    pdb_content = _apply_crops(pdb_content, args)

    total_after = sum(_count_residues(pdb_content).values())
    if total_after != total_before:
        console.print(f"  [dim]After cropping: {total_after} residues[/dim]")
    if total_after == 0:
        _die("No residues remain after cropping. Check --chain, --residues, and --trim-low-confidence.")

    out_path = _save_pdb(pdb_content, source_id, args.chain, args.out)
    console.print(f"  [{_C_SAND}]Saved[/] → [bold]{out_path}[/bold]  ({total_after} residues)")

    if args.upload:
        _fetch_and_upload(pdb_content, out_path, source_url, args.name)
    else:
        _print_fetch_next_steps(source_url, out_path, args.pdb or args.uniprot)
