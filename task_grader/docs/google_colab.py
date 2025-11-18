import os
import re
from pathlib import Path
from requests import Response, Session
from urllib.parse import urlparse, parse_qs
from .generic import SubmissionDownloader


def extract_drive_file_id(url: str) -> str | None:
    """
    Extract a Google Drive file ID from a Colab or Drive URL.

    Supports:
    - Colab URLs containing `/drive/<ID>`
    - Drive URLs like `...?id=<ID>`
    - Drive URLs like `/file/d/<ID>/view`

    Returns the file ID as a string, or None if not found.
    """
    # Case 1: Colab URL with /drive/<ID>
    match = re.search(r"/drive/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    # Case 2: URLs like .../open?id=<ID>
    if "id" in qs and qs["id"]:
        return qs["id"][0]

    # Case 3: URLs like .../file/d/<ID>/view
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    return None


class GoogleColabDownloader(SubmissionDownloader):
    """
    Download a Google Colab notebook from its URL.

    Expects the notebook to be publicly accessible
    (e.g. 'Anyone with the link can view').

    The file is downloaded in its original format from Google Drive;
    the `as_format` parameter only controls the local filename extension.
    """

    _DOWNLOAD_URL = "https://drive.google.com/uc?export=download"

    def __init__(self, session: Session | None = None) -> None:
        super().__init__(session)

    @classmethod
    def get_description(cls) -> str | None:
        return cls.__doc__ or None

    def download_as(
        self,
        doc_url: str,
        dest_dir: str,
        filename: str | None = None,
        as_format: str = "ipynb",
    ) -> str:
        """
        Download a Colab/Drive file given its URL and save it to dest_dir.

        - `filename` is the desired name for the file on disk (without extension).
          If None is provided, the extracted file ID is used.
        - `as_format` controls only the file extension used on disk
          (the content is downloaded as-is from Google Drive).
        - Returns the absolute filepath to the downloaded file.
        """
        file_id = extract_drive_file_id(doc_url)
        if not file_id:
            raise ValueError(f"Could not extract document ID from URL: {doc_url}")

        response = self._session.get(
            self._DOWNLOAD_URL,
            params={"id": file_id},
            stream=True,
            timeout=30,
        )
        token = self._get_confirm_token(response)

        if token:
            response = self._session.get(
                self._DOWNLOAD_URL,
                params={"id": file_id, "confirm": token},
                stream=True,
                timeout=30,
            )

        if not response.ok:
            raise RuntimeError(
                f"Failed to download Google Drive file (status {response.status_code}). "
                f"URL: {self._DOWNLOAD_URL}?id={file_id}.\nResponse text:\n{response.text}"
            )

        if not filename:
            filename = file_id

        filepath = os.path.join(dest_dir, f"{filename}.{as_format}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self._save_response_content(response, filepath)

        return str(filepath)

    @staticmethod
    def _get_confirm_token(response: Response) -> str | None:
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    @staticmethod
    def _save_response_content(
        response: Response,
        filepath: str | Path,
        chunk_size: int = 32768,
    ) -> None:
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
