import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import phi.config as config
from phi.config import _load_state, _resolve_cached_id, _save_state


def test_load_state_returns_empty_on_missing_file(tmp_path: Path) -> None:
    original = config.STATE_FILE
    config.STATE_FILE = tmp_path / "nonexistent.json"
    try:
        assert _load_state() == {}
    finally:
        config.STATE_FILE = original


def test_load_state_returns_empty_on_invalid_json(tmp_path: Path) -> None:
    state_file = tmp_path / ".phi-state.json"
    state_file.write_text("not valid json")
    original = config.STATE_FILE
    config.STATE_FILE = state_file
    try:
        assert _load_state() == {}
    finally:
        config.STATE_FILE = original


def test_load_state_parses_valid_json(tmp_path: Path) -> None:
    state_file = tmp_path / ".phi-state.json"
    state_file.write_text(json.dumps({"dataset_id": "abc-123"}))
    original = config.STATE_FILE
    config.STATE_FILE = state_file
    try:
        assert _load_state() == {"dataset_id": "abc-123"}
    finally:
        config.STATE_FILE = original


def test_save_state_creates_file(tmp_path: Path) -> None:
    state_file = tmp_path / ".phi-state.json"
    original = config.STATE_FILE
    config.STATE_FILE = state_file
    try:
        _save_state({"dataset_id": "xyz-999"})
        assert state_file.exists()
        saved = json.loads(state_file.read_text())
        assert saved["dataset_id"] == "xyz-999"
    finally:
        config.STATE_FILE = original


def test_save_state_merges_without_overwriting(tmp_path: Path) -> None:
    state_file = tmp_path / ".phi-state.json"
    state_file.write_text(json.dumps({"dataset_id": "original", "last_job_id": "job-1"}))
    original = config.STATE_FILE
    config.STATE_FILE = state_file
    try:
        _save_state({"last_job_id": "job-2"})
        saved = json.loads(state_file.read_text())
        assert saved["dataset_id"] == "original"
        assert saved["last_job_id"] == "job-2"
    finally:
        config.STATE_FILE = original


def test_resolve_cached_id_returns_explicit_arg(tmp_path: Path) -> None:
    import argparse

    args = argparse.Namespace(dataset_id="explicit-id")
    original = config.STATE_FILE
    config.STATE_FILE = tmp_path / "state.json"
    try:
        result = _resolve_cached_id(args, "dataset_id", "dataset_id", "no hint")
        assert result == "explicit-id"
    finally:
        config.STATE_FILE = original


def test_resolve_cached_id_falls_back_to_state(tmp_path: Path) -> None:
    import argparse

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"dataset_id": "cached-id"}))
    args = argparse.Namespace(dataset_id=None)
    original = config.STATE_FILE
    config.STATE_FILE = state_file
    try:
        result = _resolve_cached_id(args, "dataset_id", "dataset_id", "no hint")
        assert result == "cached-id"
    finally:
        config.STATE_FILE = original


def test_resolve_cached_id_dies_when_neither_present(tmp_path: Path) -> None:
    import argparse

    args = argparse.Namespace(dataset_id=None)
    original = config.STATE_FILE
    config.STATE_FILE = tmp_path / "state.json"
    try:
        with pytest.raises(SystemExit):
            _resolve_cached_id(args, "dataset_id", "dataset_id", "missing hint")
    finally:
        config.STATE_FILE = original
