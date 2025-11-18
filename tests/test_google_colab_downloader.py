# tests/test_google_colab_downloader.py

import os
from pathlib import Path

import pytest

from task_grader.docs.google_colab import (
    extract_drive_file_id,
    GoogleColabDownloader,
)


# -------------------------------------------------------------------
# Helpers: fake HTTP layer
# -------------------------------------------------------------------


class FakeResponse:
    def __init__(
        self,
        chunks: list[bytes],
        cookies: dict[str, str] | None = None,
        ok: bool = True,
        status_code: int = 200,
        text: str = "",
    ):
        self._chunks = chunks
        self.cookies = cookies or {}
        self.ok = ok
        self.status_code = status_code
        self.text = text

    def iter_content(self, _chunk_size: int):
        # Ignore chunk_size in the fake; just yield what we have
        for chunk in self._chunks:
            yield chunk


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self._responses = responses
        self.calls: list[tuple[str, dict | None, bool]] = []
        self._idx = 0

    def get(self, url: str, params=None, stream: bool = False, **_kwargs):
        self.calls.append((url, params, stream))
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


# -------------------------------------------------------------------
# extract_drive_file_id tests
# -------------------------------------------------------------------


def test_extract_drive_file_id_colab_drive_path():
    url = "https://colab.research.google.com/drive/1AbCdEfGhIjK_lMnOp"
    file_id = extract_drive_file_id(url)
    assert file_id == "1AbCdEfGhIjK_lMnOp"


def test_extract_drive_file_id_open_id_param():
    url = "https://drive.google.com/open?id=1AbCdEfGhIjK_lMnOp"
    file_id = extract_drive_file_id(url)
    assert file_id == "1AbCdEfGhIjK_lMnOp"


def test_extract_drive_file_id_file_d_view():
    url = "https://drive.google.com/file/d/1AbCdEfGhIjK_lMnOp/view?usp=sharing"
    file_id = extract_drive_file_id(url)
    assert file_id == "1AbCdEfGhIjK_lMnOp"


def test_extract_drive_file_id_returns_none_for_invalid_url():
    url = "https://example.com/not-a-drive-url"
    assert extract_drive_file_id(url) is None


# -------------------------------------------------------------------
# GoogleColabDownloader.download_as tests
# -------------------------------------------------------------------


def test_download_as_success_without_confirm_token(tmp_path: Path):
    # Fake response with no confirm cookie
    content_chunks = [b"hello", b" ", b"world"]
    fake_resp = FakeResponse(chunks=content_chunks, cookies={})
    fake_session = FakeSession(responses=[fake_resp])

    downloader = GoogleColabDownloader(session=None)
    # Override the underlying session with our fake
    downloader._session = fake_session  # type: ignore[attr-defined]

    url = "https://colab.research.google.com/drive/FILEID123"
    dest_dir = str(tmp_path)

    filepath = downloader.download_as(
        doc_url=url,
        dest_dir=dest_dir,
        filename="notebook",
        as_format="ipynb",
    )

    # File path & content
    assert filepath.endswith("notebook.ipynb")
    with open(filepath, "rb") as f:
        data = f.read()
    assert data == b"hello world"

    # Session was called once with the correct URL and params
    assert len(fake_session.calls) == 1
    url_called, params, stream = fake_session.calls[0]
    assert "uc?export=download" in url_called
    assert params == {"id": "FILEID123"}
    assert stream is True


def test_download_as_success_with_confirm_token(tmp_path: Path):
    # First response has confirm token cookie, second has final content
    first_resp = FakeResponse(
        chunks=[b"ignored"],
        cookies={"download_warning_123": "TOKEN123"},
    )
    second_resp = FakeResponse(
        chunks=[b"final-content"],
        cookies={},
    )

    fake_session = FakeSession(responses=[first_resp, second_resp])

    downloader = GoogleColabDownloader(session=None)
    downloader._session = fake_session  # type: ignore[attr-defined]

    url = "https://drive.google.com/file/d/FILEID456/view"
    dest_dir = str(tmp_path)

    filepath = downloader.download_as(
        doc_url=url,
        dest_dir=dest_dir,
        filename=None,  # will use file_id as filename
        as_format="ipynb",
    )

    # Filename should default to extracted file_id
    expected_name = os.path.join(dest_dir, "FILEID456.ipynb")
    assert filepath == expected_name

    with open(filepath, "rb") as f:
        data = f.read()
    assert data == b"final-content"

    # Two calls: first without confirm, second with confirm token
    assert len(fake_session.calls) == 2
    _, params1, _ = fake_session.calls[0]
    _, params2, _ = fake_session.calls[1]
    assert params1 == {"id": "FILEID456"}
    assert params2 == {"id": "FILEID456", "confirm": "TOKEN123"}


def test_download_as_raises_for_invalid_url(tmp_path: Path):
    fake_session = FakeSession(responses=[])
    downloader = GoogleColabDownloader(session=None)
    downloader._session = fake_session  # type: ignore[attr-defined]

    with pytest.raises(ValueError) as excinfo:
        downloader.download_as(
            doc_url="https://example.com/not-drive",
            dest_dir=str(tmp_path),
        )

    msg = str(excinfo.value)
    assert "Could not extract document ID" in msg


def test_download_as_raises_on_http_error(tmp_path: Path):
    # Response with ok=False
    error_resp = FakeResponse(
        chunks=[b"error body"],
        cookies={},
        ok=False,
        status_code=403,
        text="Forbidden",
    )
    fake_session = FakeSession(responses=[error_resp])

    downloader = GoogleColabDownloader(session=None)
    downloader._session = fake_session  # type: ignore[attr-defined]

    url = "https://colab.research.google.com/drive/FILEID999"

    with pytest.raises(RuntimeError) as excinfo:
        downloader.download_as(
            doc_url=url,
            dest_dir=str(tmp_path),
        )

    msg = str(excinfo.value)
    assert "Failed to download" in msg
    assert "403" in msg
