import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phi.structure import (
    _count_residues,
    _extract_chain,
    _trim_low_confidence,
    _trim_residues,
)

_CHAIN_A_LINES = [
    "ATOM      1  N   MET A   1      11.000  12.000  13.000  1.00 85.00           N  \n",
    "ATOM      2  CA  MET A   1      11.500  12.500  13.500  1.00 85.00           C  \n",
    "ATOM      3  N   ALA A   2      14.000  15.000  16.000  1.00 90.00           N  \n",
]

_CHAIN_B_LINES = [
    "ATOM      4  N   GLY B   1      21.000  22.000  23.000  1.00 40.00           N  \n",
    "ATOM      5  CA  GLY B   2      24.000  25.000  26.000  1.00 35.00           C  \n",
]

_REMARK_LINE = "REMARK   2 RESOLUTION.    2.00 ANGSTROMS.\n"
_END_LINE = "END\n"


def _make_pdb(*extra_lines: str) -> str:
    return "".join([_REMARK_LINE] + list(extra_lines) + [_END_LINE])


def _multi_chain_pdb() -> str:
    return _make_pdb(*_CHAIN_A_LINES, *_CHAIN_B_LINES)


class TestExtractChain:
    def test_extracts_correct_chain(self) -> None:
        pdb = _multi_chain_pdb()
        result = _extract_chain(pdb, "A")
        assert "ATOM      1" in result
        assert "ATOM      2" in result
        assert "ATOM      3" in result
        assert "ATOM      4" not in result
        assert "ATOM      5" not in result

    def test_preserves_header_records(self) -> None:
        pdb = _multi_chain_pdb()
        result = _extract_chain(pdb, "A")
        assert "REMARK" in result
        assert "END" in result

    def test_case_insensitive_chain(self) -> None:
        pdb = _multi_chain_pdb()
        result_lower = _extract_chain(pdb, "a")
        result_upper = _extract_chain(pdb, "A")
        assert result_lower == result_upper

    def test_raises_on_missing_chain(self) -> None:
        pdb = _multi_chain_pdb()
        with pytest.raises(ValueError, match="Chain 'Z' not found"):
            _extract_chain(pdb, "Z")

    def test_extracts_chain_b(self) -> None:
        pdb = _multi_chain_pdb()
        result = _extract_chain(pdb, "B")
        assert "GLY B   1" in result
        assert "GLY B   2" in result
        assert "MET A" not in result


class TestTrimResidues:
    def test_keeps_residues_in_range(self) -> None:
        pdb = _make_pdb(*_CHAIN_A_LINES)
        result = _trim_residues(pdb, 1, 1)
        assert "MET A   1" in result
        assert "ALA A   2" not in result

    def test_keeps_full_range(self) -> None:
        pdb = _make_pdb(*_CHAIN_A_LINES)
        result = _trim_residues(pdb, 1, 2)
        assert "MET A   1" in result
        assert "ALA A   2" in result

    def test_empty_range_removes_all_atoms(self) -> None:
        pdb = _make_pdb(*_CHAIN_A_LINES)
        result = _trim_residues(pdb, 99, 100)
        assert "ATOM" not in result

    def test_preserves_non_atom_lines(self) -> None:
        pdb = _make_pdb(*_CHAIN_A_LINES)
        result = _trim_residues(pdb, 99, 100)
        assert "REMARK" in result
        assert "END" in result


class TestTrimLowConfidence:
    def test_removes_low_plddt_residues(self) -> None:
        pdb = _multi_chain_pdb()
        result = _trim_low_confidence(pdb, 70.0)
        assert "MET A   1" in result
        assert "ALA A   2" in result
        assert "GLY B   1" not in result
        assert "GLY B   2" not in result

    def test_keeps_all_above_threshold(self) -> None:
        pdb = _multi_chain_pdb()
        result = _trim_low_confidence(pdb, 30.0)
        assert "MET A   1" in result
        assert "GLY B   1" in result

    def test_removes_all_below_threshold(self) -> None:
        pdb = _multi_chain_pdb()
        result = _trim_low_confidence(pdb, 95.0)
        assert "ATOM" not in result

    def test_preserves_non_atom_lines(self) -> None:
        pdb = _multi_chain_pdb()
        result = _trim_low_confidence(pdb, 99.0)
        assert "REMARK" in result
        assert "END" in result

    def test_exact_threshold_is_kept(self) -> None:
        line = "ATOM      1  N   MET A   1      11.000  12.000  13.000  1.00 70.00           N  \n"
        pdb = _make_pdb(line)
        result = _trim_low_confidence(pdb, 70.0)
        assert "MET A   1" in result


class TestCountResidues:
    def test_counts_single_chain(self) -> None:
        pdb = _make_pdb(*_CHAIN_A_LINES)
        counts = _count_residues(pdb)
        assert counts == {"A": 2}

    def test_counts_multiple_chains(self) -> None:
        pdb = _multi_chain_pdb()
        counts = _count_residues(pdb)
        assert counts == {"A": 2, "B": 2}

    def test_empty_pdb_returns_empty(self) -> None:
        counts = _count_residues("END\n")
        assert counts == {}

    def test_multiple_atoms_same_residue_counted_once(self) -> None:
        pdb = _make_pdb(*_CHAIN_A_LINES)
        counts = _count_residues(pdb)
        assert counts["A"] == 2
