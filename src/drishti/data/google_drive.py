"""Chunked, temporary ingestion for public Google Drive video assets.

The dataset is never copied into the repository. Each video is downloaded in
chunks to an operating-system temporary directory, processed, and deleted when
the context manager exits. This is deliberately called temporary staging rather
than zero-copy streaming: OpenCV needs a seekable file for reliable MKV decode.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterator, List, Tuple

import requests


_DRIVE_ID = re.compile(r"/file/d/([A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class DriveAsset:
    asset_id: str
    drive_file_id: str
    source_url: str
    filename: str
    expected_bytes: int | None = None
    source_label: str | None = None


def extract_drive_file_id(value: str) -> str:
    match = _DRIVE_ID.search(value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", value):
        return value
    raise ValueError(f"Not a supported Google Drive file URL or ID: {value!r}")


def load_drive_manifest(path: str | Path) -> List[DriveAsset]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"Manifest has no assets: {manifest_path}")

    result: List[DriveAsset] = []
    for raw in assets:
        drive_id = raw.get("drive_file_id") or extract_drive_file_id(raw["source_url"])
        result.append(
            DriveAsset(
                asset_id=str(raw["asset_id"]),
                drive_file_id=str(drive_id),
                source_url=str(raw["source_url"]),
                filename=str(raw["filename"]),
                expected_bytes=int(raw["expected_bytes"]) if raw.get("expected_bytes") else None,
                source_label=raw.get("source_label"),
            )
        )
    return result


@contextmanager
def materialize_drive_asset(
    asset: DriveAsset,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
    timeout: Tuple[float, float] = (15.0, 180.0),
    temp_root: str | Path | None = None,
) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """Stage one Drive video in a temporary directory and report provenance."""

    download_url = (
        "https://drive.google.com/uc?export=download&id=" + asset.drive_file_id
    )
    digest = sha256()
    bytes_written = 0

    staging_root = Path(temp_root).resolve() if temp_root is not None else None
    if staging_root is not None:
        staging_root.mkdir(parents=True, exist_ok=True)
        cached_file = staging_root / asset.filename
        if cached_file.exists() and (asset.expected_bytes is None or cached_file.stat().st_size == asset.expected_bytes):
            provenance = {
                "asset_id": asset.asset_id,
                "source_url": asset.source_url,
                "drive_file_id": asset.drive_file_id,
                "filename": asset.filename,
                "bytes": cached_file.stat().st_size,
                "staging": "local_persistent_cache",
                "persisted_in_repository": True,
            }
            yield cached_file, provenance
            return

    target = (staging_root / asset.filename) if staging_root is not None else Path(tempfile.gettempdir()) / asset.filename
    try:
        import gdown
        gdown.download(id=asset.drive_file_id, output=str(target), quiet=False)
        if target.exists():
            bytes_written = target.stat().st_size
    except Exception:
        pass

    if bytes_written == 0:
        session = requests.Session()
        download_url = f"https://drive.usercontent.google.com/download?id={asset.drive_file_id}&export=download&confirm=t"
        with session.get(
            download_url,
            stream=True,
            timeout=timeout,
            headers={"User-Agent": "NETRA-4D verified dataset runner"},
        ) as response:
            if response.status_code != 200 or "text/html" in response.headers.get("content-type", "").lower():
                fallback_url = f"https://drive.google.com/uc?export=download&id={asset.drive_file_id}&confirm=t"
                response = session.get(fallback_url, stream=True, timeout=timeout)

            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_bytes):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)

        if bytes_written == 0:
            raise RuntimeError(f"Downloaded zero bytes for {asset.asset_id}")
        if asset.expected_bytes is not None and bytes_written != asset.expected_bytes:
            raise RuntimeError(
                f"Size mismatch for {asset.asset_id}: expected {asset.expected_bytes}, "
                f"received {bytes_written}"
            )

        provenance = {
            "asset_id": asset.asset_id,
            "source_url": asset.source_url,
            "drive_file_id": asset.drive_file_id,
            "filename": asset.filename,
            "bytes": bytes_written,
            "sha256": digest.hexdigest(),
            "staging": "chunked_download_to_os_temporary_directory",
            "persisted_in_repository": False,
        }
        yield target, provenance
