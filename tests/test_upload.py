import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phi.upload import _collect_files


def _args(**kwargs):  # type: ignore[return]
    import argparse

    ns = argparse.Namespace(files=[], dir=None, file_type=None)
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def test_collect_files_single_file(tmp_path: Path) -> None:
    f = tmp_path / "binder.pdb"
    f.write_text("ATOM ...")
    args = _args(files=[str(f)])
    result = _collect_files(args)
    assert result == [f]


def test_collect_files_directory_expansion(tmp_path: Path) -> None:
    (tmp_path / "a.pdb").write_text("ATOM ...")
    (tmp_path / "b.pdb").write_text("ATOM ...")
    args = _args(files=[str(tmp_path)], file_type="pdb")
    result = _collect_files(args)
    assert len(result) == 2
    assert all(p.suffix == ".pdb" for p in result)


def test_collect_files_dir_flag(tmp_path: Path) -> None:
    (tmp_path / "c.pdb").write_text("ATOM ...")
    (tmp_path / "d.pdb").write_text("ATOM ...")
    args = _args(dir=str(tmp_path), file_type="pdb")
    result = _collect_files(args)
    assert len(result) == 2


def test_collect_files_deduplicates(tmp_path: Path) -> None:
    f = tmp_path / "binder.pdb"
    f.write_text("ATOM ...")
    args = _args(files=[str(f), str(f)])
    result = _collect_files(args)
    assert len(result) == 1


def test_collect_files_raises_on_collision(tmp_path: Path) -> None:
    dir1 = tmp_path / "group1"
    dir2 = tmp_path / "group2"
    dir1.mkdir()
    dir2.mkdir()
    (dir1 / "binder.pdb").write_text("ATOM ...")
    (dir2 / "binder.pdb").write_text("ATOM ...")
    args = _args(files=[str(dir1 / "binder.pdb"), str(dir2 / "binder.pdb")])
    with pytest.raises(SystemExit):
        _collect_files(args)


def test_collect_files_raises_on_no_files(tmp_path: Path) -> None:
    args = _args()
    with pytest.raises(SystemExit):
        _collect_files(args)


def test_collect_files_raises_on_missing_file(tmp_path: Path) -> None:
    args = _args(files=[str(tmp_path / "nonexistent.pdb")])
    with pytest.raises(SystemExit):
        _collect_files(args)
