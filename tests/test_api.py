import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phi.api import _put_file, _request
from phi.types import PhiApiError


def _mock_response(data: dict, status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.read.return_value = json.dumps(data).encode()
    mock.status = status
    return mock


@patch("phi.api._require_api_key", return_value="test-key")
@patch("phi.api._base_url", return_value="http://localhost:8000")
@patch("phi.api._ssl_context")
@patch("urllib.request.urlopen")
def test_request_sends_auth_header(
    mock_urlopen: MagicMock,
    mock_ssl: MagicMock,
    mock_base: MagicMock,
    mock_key: MagicMock,
) -> None:
    mock_urlopen.return_value = _mock_response({"ok": True})
    result = _request("GET", "/test")
    assert result == {"ok": True}
    call_args = mock_urlopen.call_args
    req = call_args[0][0]
    assert req.get_header("X-api-key") == "test-key"


@patch("phi.api._require_api_key", return_value="test-key")
@patch("phi.api._base_url", return_value="http://localhost:8000")
@patch("phi.api._ssl_context")
@patch("urllib.request.urlopen")
def test_request_raises_on_4xx(
    mock_urlopen: MagicMock,
    mock_ssl: MagicMock,
    mock_base: MagicMock,
    mock_key: MagicMock,
) -> None:
    err = urllib.error.HTTPError(
        url="http://localhost:8000/test",
        code=401,
        msg="Unauthorized",
        hdrs=MagicMock(),
        fp=MagicMock(read=lambda: b'{"detail": "Unauthorized"}'),
    )
    mock_urlopen.side_effect = err
    with pytest.raises(PhiApiError, match="HTTP 401"):
        _request("GET", "/test")


@patch("phi.api._require_api_key", return_value="test-key")
@patch("phi.api._base_url", return_value="http://localhost:8000")
@patch("phi.api._ssl_context")
@patch("urllib.request.urlopen")
def test_request_raises_on_network_error(
    mock_urlopen: MagicMock,
    mock_ssl: MagicMock,
    mock_base: MagicMock,
    mock_key: MagicMock,
) -> None:
    mock_urlopen.side_effect = urllib.error.URLError("connection refused")
    with pytest.raises(PhiApiError, match="Network error"):
        _request("GET", "/test")


@patch("phi.api._ssl_context")
@patch("urllib.request.urlopen")
def test_put_file_retries_on_5xx(
    mock_urlopen: MagicMock,
    mock_ssl: MagicMock,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "test.pdb"
    test_file.write_bytes(b"ATOM ...")

    http_err = urllib.error.HTTPError(
        url="http://storage.example.com/signed",
        code=503,
        msg="Service Unavailable",
        hdrs=MagicMock(),
        fp=MagicMock(),
    )
    success = _mock_response({})
    mock_urlopen.side_effect = [http_err, http_err, success]

    with patch("phi.api.time.sleep"):
        _put_file("http://storage.example.com/signed", test_file)

    assert mock_urlopen.call_count == 3


@patch("phi.api._ssl_context")
@patch("urllib.request.urlopen")
def test_put_file_raises_after_max_retries(
    mock_urlopen: MagicMock,
    mock_ssl: MagicMock,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "test.pdb"
    test_file.write_bytes(b"ATOM ...")

    http_err = urllib.error.HTTPError(
        url="http://storage.example.com/signed",
        code=500,
        msg="Internal Server Error",
        hdrs=MagicMock(),
        fp=MagicMock(),
    )
    mock_urlopen.side_effect = [http_err, http_err, http_err]

    with patch("phi.api.time.sleep"), pytest.raises(RuntimeError, match="Upload failed"):
        _put_file("http://storage.example.com/signed", test_file)
