import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import cast

_PDB_CHAIN_COL = 21
_PDB_RESNUM_COLS = slice(22, 26)
_PDB_BFACTOR_COLS = slice(60, 66)
_STRUCTURAL_RECORDS = frozenset({"ATOM", "HETATM"})


def _iter_pdb_lines(pdb: str) -> Iterator[tuple[str, str]]:
    for line in pdb.splitlines(keepends=True):
        yield line[:6].rstrip(), line


def _filter_pdb(pdb: str, keep_atom: Callable[[str], bool]) -> str:
    kept: list[str] = []
    for record, line in _iter_pdb_lines(pdb):
        if record in _STRUCTURAL_RECORDS:
            if keep_atom(line):
                kept.append(line)
        else:
            kept.append(line)
    return "".join(kept)


def _fetch_pdb_rcsb(pdb_id: str) -> str:
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        with urllib.request.urlopen(url) as resp:
            return cast(bytes, resp.read()).decode("utf-8")
    except urllib.error.HTTPError as e:
        raise ValueError(f"PDB ID '{pdb_id}' not found in RCSB (HTTP {e.code})") from e


def _afdb_pdb_url(uniprot_id: str) -> str:
    return f"https://alphafold.ebi.ac.uk/entry/{uniprot_id.upper()}"


def _fetch_pdb_afdb(uniprot_id: str) -> str:
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id.upper()}"
    try:
        with urllib.request.urlopen(api_url) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise ValueError(f"UniProt ID '{uniprot_id}' not found in AlphaFold DB (HTTP {e.code})") from e
    if not data:
        raise ValueError(f"AlphaFold DB returned no predictions for '{uniprot_id}'")
    pdb_url: str = data[0]["pdbUrl"]
    with urllib.request.urlopen(pdb_url) as resp:
        return cast(bytes, resp.read()).decode("utf-8")


def _extract_chain(pdb_content: str, chain: str) -> str:
    target = chain.upper()
    found_atoms = False

    def _keep(line: str) -> bool:
        nonlocal found_atoms
        if len(line) > _PDB_CHAIN_COL and line[_PDB_CHAIN_COL].upper() == target:
            found_atoms = True
            return True
        return False

    result = _filter_pdb(pdb_content, _keep)
    if not found_atoms:
        raise ValueError(f"Chain '{chain}' not found in structure")
    return result


def _trim_residues(pdb_content: str, start: int, end: int) -> str:
    def _keep(line: str) -> bool:
        try:
            return start <= int(line[_PDB_RESNUM_COLS]) <= end
        except (ValueError, IndexError):
            return True

    return _filter_pdb(pdb_content, _keep)


def _trim_low_confidence(pdb_content: str, min_plddt: float) -> str:
    def _keep(line: str) -> bool:
        try:
            return float(line[_PDB_BFACTOR_COLS]) >= min_plddt
        except (ValueError, IndexError):
            return True

    return _filter_pdb(pdb_content, _keep)


def _count_residues(pdb_content: str) -> dict[str, int]:
    seen: set[tuple[str, str]] = set()
    for record, line in _iter_pdb_lines(pdb_content):
        if record not in _STRUCTURAL_RECORDS:
            continue
        if len(line) < 27:
            continue
        seen.add((line[_PDB_CHAIN_COL], line[_PDB_RESNUM_COLS].strip()))
    counts: dict[str, int] = {}
    for chain, _ in seen:
        counts[chain] = counts.get(chain, 0) + 1
    return counts
