"""Recursively upload NETRA-4D output folders to the requested Google Drive folder.

The uploader uses Google's supported Drive API, preserves the local directory
tree, and performs resumable uploads. Existing files with the same name in the
same Drive folder are updated instead of duplicated.

Authentication options:
  * Colab: ``--auth colab`` (or ``auto`` while running in Colab)
  * Desktop OAuth: ``--auth oauth --client-secrets credentials.json``
  * Service account: ``--auth service-account --service-account key.json``

OAuth tokens and service-account keys are credentials, not project data. Keep
them outside Git (the repository .gitignore contains defensive rules).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import mimetypes
from pathlib import Path
import sys
from typing import Any, Iterable


DEFAULT_DESTINATION_FOLDER_ID = "1vEfaikd1G9_dDr0-fbDm01w20Mk0TI18"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def _google_modules() -> dict[str, Any]:
    """Import optional Google dependencies only when the CLI is executed."""

    try:
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Google Drive dependencies are missing. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return {
        "service_account": service_account,
        "Credentials": Credentials,
        "InstalledAppFlow": InstalledAppFlow,
        "build": build,
        "MediaFileUpload": MediaFileUpload,
    }


def _running_in_colab() -> bool:
    return "google.colab" in sys.modules


def authenticate(args: argparse.Namespace, modules: dict[str, Any]) -> Any:
    auth_mode = args.auth
    if auth_mode == "auto":
        auth_mode = "colab" if _running_in_colab() else "oauth"

    if auth_mode == "colab":
        try:
            from google.colab import auth as colab_auth
            import google.auth
        except ImportError as exc:
            raise RuntimeError("--auth colab must be run inside Google Colab") from exc
        colab_auth.authenticate_user()
        credentials, _ = google.auth.default(scopes=[DRIVE_SCOPE])
        return credentials

    if auth_mode == "service-account":
        if not args.service_account:
            raise ValueError("--service-account is required for service-account auth")
        return modules["service_account"].Credentials.from_service_account_file(
            str(args.service_account), scopes=[DRIVE_SCOPE]
        )

    token_path: Path = args.token
    credentials = None
    if token_path.exists():
        credentials = modules["Credentials"].from_authorized_user_file(
            str(token_path), [DRIVE_SCOPE]
        )
    if credentials and credentials.expired and credentials.refresh_token:
        from google.auth.transport.requests import Request

        credentials.refresh(Request())
    elif not credentials or not credentials.valid:
        if not args.client_secrets.exists():
            raise FileNotFoundError(
                f"OAuth client file not found: {args.client_secrets}. "
                "Download a Desktop OAuth client JSON from Google Cloud Console."
            )
        flow = modules["InstalledAppFlow"].from_client_secrets_file(
            str(args.client_secrets), [DRIVE_SCOPE]
        )
        credentials = flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def escape_drive_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def iter_local_files(source: Path) -> Iterable[Path]:
    return (path for path in sorted(source.rglob("*")) if path.is_file())


class DriveTreeUploader:
    def __init__(self, service: Any, media_file_upload: Any, *, chunk_mb: int = 16):
        self.service = service
        self.media_file_upload = media_file_upload
        self.chunk_bytes = chunk_mb * 1024 * 1024
        self.folder_cache: dict[tuple[str, str], str] = {}

    def _find_child(self, parent_id: str, name: str, *, folder: bool) -> list[dict[str, Any]]:
        mime_clause = (
            f"mimeType = '{FOLDER_MIME_TYPE}'"
            if folder
            else f"mimeType != '{FOLDER_MIME_TYPE}'"
        )
        query = (
            f"'{escape_drive_query(parent_id)}' in parents and "
            f"name = '{escape_drive_query(name)}' and {mime_clause} and trashed = false"
        )
        response = self.service.files().list(
            q=query,
            spaces="drive",
            fields="files(id,name,mimeType,modifiedTime,webViewLink)",
            orderBy="createdTime",
            pageSize=100,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        return response.get("files", [])

    def ensure_folder(self, parent_id: str, name: str) -> str:
        cache_key = (parent_id, name)
        if cache_key in self.folder_cache:
            return self.folder_cache[cache_key]
        matches = self._find_child(parent_id, name, folder=True)
        if matches:
            folder_id = matches[0]["id"]
        else:
            created = self.service.files().create(
                body={"name": name, "mimeType": FOLDER_MIME_TYPE, "parents": [parent_id]},
                fields="id,name,webViewLink",
                supportsAllDrives=True,
            ).execute()
            folder_id = created["id"]
            print(f"Created Drive folder: {name} ({folder_id})")
        self.folder_cache[cache_key] = folder_id
        return folder_id

    def upload_file(self, local_path: Path, parent_id: str) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        media = self.media_file_upload(
            str(local_path), mimetype=mime_type, chunksize=self.chunk_bytes, resumable=True
        )
        matches = self._find_child(parent_id, local_path.name, folder=False)
        if matches:
            action = "updated"
            request = self.service.files().update(
                fileId=matches[0]["id"],
                media_body=media,
                fields="id,name,size,md5Checksum,modifiedTime,webViewLink",
                supportsAllDrives=True,
            )
        else:
            action = "created"
            request = self.service.files().create(
                body={"name": local_path.name, "parents": [parent_id]},
                media_body=media,
                fields="id,name,size,md5Checksum,modifiedTime,webViewLink",
                supportsAllDrives=True,
            )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"  {local_path.name}: {status.progress() * 100:5.1f}%")
        response["action"] = action
        response["local_path"] = str(local_path)
        response["local_bytes"] = local_path.stat().st_size
        print(f"{action.title()}: {local_path} -> {response['id']}")
        return response

    def upload_tree(self, source: Path, destination_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for local_file in iter_local_files(source):
            parent_id = destination_id
            for part in local_file.relative_to(source).parts[:-1]:
                parent_id = self.ensure_folder(parent_id, part)
            record = self.upload_file(local_file, parent_id)
            record["relative_path"] = local_file.relative_to(source).as_posix()
            records.append(record)
        return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively upload a NETRA-4D output directory to Google Drive."
    )
    parser.add_argument("source", type=Path, help="Local output directory to upload")
    parser.add_argument(
        "--destination-folder-id",
        default=DEFAULT_DESTINATION_FOLDER_ID,
        help="Drive destination folder ID (defaults to the user-supplied folder)",
    )
    parser.add_argument(
        "--root-name",
        help="Optional Drive folder created below the destination before uploading",
    )
    parser.add_argument("--auth", choices=("auto", "colab", "oauth", "service-account"), default="auto")
    parser.add_argument("--client-secrets", type=Path, default=Path("credentials.json"))
    parser.add_argument("--token", type=Path, default=Path(".secrets/google_drive_token.json"))
    parser.add_argument("--service-account", type=Path)
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("outputs/upload_receipts/google_drive_upload.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Output directory does not exist: {source}")
    if args.chunk_mb <= 0:
        raise ValueError("--chunk-mb must be positive")
    files = list(iter_local_files(source))
    if not files:
        raise RuntimeError(f"No output files found under: {source}")

    modules = _google_modules()
    credentials = authenticate(args, modules)
    service = modules["build"]("drive", "v3", credentials=credentials, cache_discovery=False)
    uploader = DriveTreeUploader(service, modules["MediaFileUpload"], chunk_mb=args.chunk_mb)

    destination_id = args.destination_folder_id
    if args.root_name:
        destination_id = uploader.ensure_folder(destination_id, args.root_name)
    records = uploader.upload_tree(source, destination_id)

    receipt = {
        "schema_version": "1.0",
        "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination_folder_id": args.destination_folder_id,
        "upload_root_folder_id": destination_id,
        "file_count": len(records),
        "files": records,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"Upload complete: {len(records)} files")
    print(f"Receipt: {args.receipt.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
