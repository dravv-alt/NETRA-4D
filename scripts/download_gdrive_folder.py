"""Resumably download a public Google Drive folder and record every outcome."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import gdown


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(item, root: Path, retries: int) -> dict:
    target = root / Path(item.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return {"id": item.id, "path": item.path, "status": "existing", "bytes": target.stat().st_size}
    error = None
    for attempt in range(1, retries + 1):
        try:
            result = gdown.download(id=item.id, output=str(target), quiet=True, resume=True)
            if result and target.is_file() and target.stat().st_size > 0:
                return {"id": item.id, "path": item.path, "status": "downloaded", "bytes": target.stat().st_size}
            error = "download returned no completed file"
        except Exception as exc:  # individual Drive failures must not abort the folder
            error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(2 ** attempt, 8))
    return {"id": item.id, "path": item.path, "status": "failed", "error": error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1:
        raise SystemExit("--workers and --retries must be positive")

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    items = gdown.download_folder(id=args.folder_id, output=str(root), quiet=True, skip_download=True)
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="drive-download") as pool:
        futures = {pool.submit(download_one, item, root, args.retries): item for item in items}
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if index % 50 == 0 or record["status"] == "failed":
                print(f"{index}/{len(items)} {record['status']}: {record['path']}", flush=True)

    for record in records:
        path = root / Path(record["path"])
        if record["status"] != "failed" and path.is_file():
            record["sha256"] = sha256(path)
    records.sort(key=lambda item: item["path"])
    counts = {status: sum(item["status"] == status for item in records) for status in ("existing", "downloaded", "failed")}
    manifest = {
        "schema_version": "1.0.0",
        "folder_id": args.folder_id,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root),
        "listed_files": len(items),
        "counts": counts,
        "files": records,
    }
    manifest_path = (args.manifest or root / "drive_download_manifest.json").resolve()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "listed_files": len(items), **counts}, indent=2))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
