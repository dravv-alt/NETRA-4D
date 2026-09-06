#!/usr/bin/env python3
"""
Watcher Batch Processor for Full-Length Videos:
Processes video_01.mp4, video_04.mp4, and video_05.mp4 covering all 5304 frames.
Generates full overlay videos, temporal feature logs, alert snapshots, and SQLite database records.
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import subprocess
from pathlib import Path

# Paths
script_dir = Path(__file__).parent
watcher_root = script_dir.parent
root_dir = watcher_root.parent

run_script = watcher_root / "scripts" / "run_watcher.py"

videos = [
    {
        "name": "video 01",
        "file": "data/videos/video 01.mkv",
        "output_video": "outputs/overlays/video_01_overlay.avi",
        "output_json": "outputs/logs/video_01_features.json"
    },
    {
        "name": "video 02",
        "file": "data/videos/video 02.mkv",
        "output_video": "outputs/overlays/video_02_overlay.avi",
        "output_json": "outputs/logs/video_02_features.json"
    },
    {
        "name": "video 03",
        "file": "data/videos/video 03.mkv",
        "output_video": "outputs/overlays/video_03_overlay.avi",
        "output_json": "outputs/logs/video_03_features.json"
    },
    {
        "name": "video 04",
        "file": "data/videos/video 04.mkv",
        "output_video": "outputs/overlays/video_04_overlay.avi",
        "output_json": "outputs/logs/video_04_features.json"
    },
    {
        "name": "video 05",
        "file": "data/videos/video 05.mkv",
        "output_video": "outputs/overlays/video_05_overlay.avi",
        "output_json": "outputs/logs/video_05_features.json"
    }
]

def main():
    print("=" * 70)
    print("WATCHER BATCH FULL VIDEO PROCESSING FOR ALL MKV FILES (PHASES 1-10)")
    print("=" * 70)
    
    total_start = time.time()
    
    for idx, v in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] STARTING PROCESSING: {v['name']} ({v['file']})")
        print(f"Overlay Output: {v['output_video']}")
        print(f"Log Output:     {v['output_json']}")
        print("-" * 70)
        
        cmd = [
            sys.executable,
            str(run_script),
            "--source", f"file://{v['file']}",
            "--output", v["output_video"],
            "--output-json", v["output_json"],
            "--db-path", "watcher.db",
            "--frames", "0",
            "--stride", "2"
        ]
        
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(watcher_root))
        t1 = time.time()
        
        if result.returncode == 0:
            print(f"\n[OK] {v['name']} completed successfully in {(t1 - t0):.1f}s ({((t1 - t0)/60):.2f} min)!")
        else:
            print(f"\n[ERROR] {v['name']} failed with return code {result.returncode}")
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"ALL BATCH PROCESSING COMPLETE in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)!")
    print("=" * 70)

if __name__ == "__main__":
    main()
