"""Cache browser-playable previews of verified Google Drive source videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drishti.data.google_drive import load_drive_manifest, materialize_drive_asset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--manifest", type=Path, default=Path("drishti4d/configs/datasets.json"))
    parser.add_argument("--verified-run", type=Path, default=Path("OUTPUT/netra4d_final_20260823/run.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("OUTPUT/showcase_inputs"))
    parser.add_argument("--annotated-output-dir", type=Path, default=Path("OUTPUT/showcase_outputs"))
    args = parser.parse_args()

    run = json.loads((ROOT / args.verified_run).read_text(encoding="utf-8"))
    verified = {item["asset_id"]: item for item in run["assets"]}
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_output_dir = (ROOT / args.annotated_output_dir).resolve()
    annotated_output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for asset in load_drive_manifest(ROOT / args.manifest)[: args.limit]:
        expected = verified[asset.asset_id]["source"]
        target = output_dir / f"{asset.asset_id}_source_preview.mp4"
        if not target.is_file():
            with materialize_drive_asset(asset) as (source, provenance):
                if provenance["sha256"] != expected["sha256"]:
                    raise RuntimeError(f"Source hash mismatch for {asset.asset_id}")
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
                        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(target),
                    ],
                    check=True,
                )
        annotated_source = (ROOT / args.verified_run).resolve().parent / asset.asset_id / verified[asset.asset_id]["outputs"]["annotated_video"]
        annotated_preview = annotated_output_dir / f"{asset.asset_id}_annotated_preview.mp4"
        if not annotated_preview.is_file():
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(annotated_source),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
                    "-an", "-movflags", "+faststart", str(annotated_preview),
                ],
                check=True,
            )
        records.append({
            "asset_id": asset.asset_id, "preview_file": target.name,
            "original_filename": asset.filename, "source_url": asset.source_url,
            "verified_source_sha256": expected["sha256"], "preview_bytes": target.stat().st_size,
            "annotated_preview_file": annotated_preview.name,
            "annotated_preview_bytes": annotated_preview.stat().st_size,
            "meaning": "Browser-compatible transcode of the hash-verified original source; no annotations.",
        })

    (output_dir / "manifest.json").write_text(json.dumps({"assets": records}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "assets": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
