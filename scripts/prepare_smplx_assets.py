"""Safely extract and validate licensed SMPL-X assets for Multi-HMR.

This script never fabricates a body model. It accepts the user-supplied SMPL-X
archive and extracts only after confirming all ZIP members stay inside the
requested destination. The Multi-HMR 2024 demo requires SMPLX_NEUTRAL.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = ROOT / "models_smplx_v1_1.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as payload:
        members = payload.infolist()
        for member in members:
            target = (root / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member rejected: {member.filename}")
        payload.extractall(root)
    neutral = root / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not neutral.is_file() or neutral.stat().st_size == 0:
        raise FileNotFoundError(f"SMPLX_NEUTRAL.npz was not extracted: {neutral}")
    return neutral


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--destination", type=Path, default=ROOT / "third_party" / "smplx_assets")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"SMPL-X archive not found: {archive}")
    with zipfile.ZipFile(archive) as payload:
        names = set(payload.namelist())
    expected = "models/smplx/SMPLX_NEUTRAL.npz"
    if expected not in names:
        raise ValueError(f"Archive does not contain required {expected}")
    record = {"archive": str(archive), "archive_sha256": sha256(archive), "required_member": expected}
    if args.verify_only:
        print(json.dumps(record, indent=2))
        return 0
    neutral = safe_extract(archive, args.destination.resolve())
    record.update({"neutral_model": str(neutral), "neutral_model_sha256": sha256(neutral)})
    (args.destination.resolve() / "smplx_asset_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
