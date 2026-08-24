"""
Drishti-4D Dataset Forensic Audit CLI.
Enumerates videos, checks container health, extracts resolution/FPS distributions,
and generates dataset manifest.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from drishti.video.reader import FileVideoSource

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def main():
    parser = argparse.ArgumentParser(description="Drishti-4D Dataset Forensic Audit")
    parser.add_argument("--root", type=str, default=".", help="Root directory containing dataset videos")
    parser.add_argument("--output", type=str, default="dataset_audit_report.json", help="Output JSON path")
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    print("=" * 80)
    print(f"[*] DRISHTI-4D FORENSIC AUDIT ON: {root_dir}")
    print("=" * 80)

    video_paths = sorted([p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])
    print(f"[*] Found {len(video_paths)} candidate video files.\n")

    records = []
    for idx, vp in enumerate(video_paths, 1):
        try:
            src = FileVideoSource(str(vp))
            meta = src.get_metadata()
            health = src.check_health()
            src.close()

            rec = {
                "video_id": f"VID_{idx:02d}",
                "filename": meta.filename,
                "path": str(vp),
                "extension": Path(vp).suffix.lower(),
                "width": meta.width,
                "height": meta.height,
                "resolution": f"{meta.width}x{meta.height}",
                "fps": meta.fps,
                "frame_count": meta.frame_count,
                "duration_sec": meta.duration_sec,
                "duration_min": round(meta.duration_sec / 60.0, 2),
                "size_mb": round(vp.stat().st_size / (1024 * 1024), 2),
                "codec": meta.codec,
                "health_status": health["status"],
                "decode_success_rate": health["success_rate"]
            }
            records.append(rec)
            print(f"[{idx}/{len(video_paths)}] {meta.filename}: {rec['resolution']} @ {rec['fps']:.1f} FPS | {rec['duration_min']} min | Health: {rec['health_status']}")
        except Exception as e:
            print(f"[{idx}/{len(video_paths)}] [!] Error reading {vp.name}: {e}")

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    csv_file = out_file.with_suffix(".csv")
    pd.DataFrame(records).to_csv(csv_file, index=False)

    print("\n" + "=" * 80)
    print(f"[✔] AUDIT COMPLETE! Reports saved to:\n    - {out_file}\n    - {csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
