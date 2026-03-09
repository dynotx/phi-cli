import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phi.download import _categorize_artifacts, _get_artifact_url
from phi.types import PhiApiError


def test_get_artifact_url_returns_presigned_url() -> None:
    af = {"download_url": "https://storage.example.com/signed?token=abc"}
    assert _get_artifact_url(af) == "https://storage.example.com/signed?token=abc"


def test_get_artifact_url_falls_back_to_url_key() -> None:
    af = {"url": "https://storage.example.com/signed?token=xyz"}
    assert _get_artifact_url(af) is not None


@patch("phi.download._request")
def test_get_artifact_url_calls_api_when_missing(mock_request: MagicMock) -> None:
    mock_request.return_value = {"download_url": "https://storage.example.com/fresh"}
    af = {"artifact_id": "art-123"}
    result = _get_artifact_url(af)
    assert result == "https://storage.example.com/fresh"
    mock_request.assert_called_once_with("GET", "/artifacts/art-123/download")


@patch("phi.download._request")
def test_get_artifact_url_returns_none_when_api_fails(mock_request: MagicMock) -> None:
    mock_request.side_effect = PhiApiError("404")
    af = {"artifact_id": "art-missing"}
    result = _get_artifact_url(af)
    assert result is None


def test_get_artifact_url_returns_none_when_no_id() -> None:
    af = {"filename": "binder.pdb"}
    result = _get_artifact_url(af)
    assert result is None


def test_categorize_artifacts_splits_by_type() -> None:
    artifacts = [
        {"artifact_type": "pdb", "filename": "binder.pdb"},
        {"artifact_type": "csv", "filename": "scores.csv"},
        {"artifact_type": "msa", "filename": "binder.a3m"},
        {"artifact_type": "zip", "filename": "results.zip"},
    ]
    to_download, to_skip = _categorize_artifacts(artifacts, all_files=False)
    download_types = {af["artifact_type"] for af in to_download}
    skip_types = {af["artifact_type"] for af in to_skip}
    assert "pdb" in download_types
    assert "csv" in download_types
    assert "msa" in skip_types
    assert "zip" in skip_types


def test_categorize_artifacts_all_files_flag() -> None:
    artifacts = [
        {"artifact_type": "pdb", "filename": "binder.pdb"},
        {"artifact_type": "msa", "filename": "binder.a3m"},
        {"artifact_type": "zip", "filename": "results.zip"},
    ]
    to_download, to_skip = _categorize_artifacts(artifacts, all_files=True)
    assert len(to_download) == 3
    assert len(to_skip) == 0


def test_categorize_artifacts_unknown_type_skipped_by_default() -> None:
    artifacts = [{"artifact_type": "unknown_binary", "filename": "data.bin"}]
    to_download, to_skip = _categorize_artifacts(artifacts, all_files=False)
    assert len(to_skip) == 1
    assert len(to_download) == 0
