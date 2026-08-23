"""Interactively create genuine YOLO chit labels from the real Drive videos.

No label is written unless a human explicitly draws a box or marks the frame
negative. Keys: B=draw chit boxes, N=reviewed negative, S=skip, Q=quit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drishti.data.google_drive import load_drive_manifest, materialize_drive_asset


def yolo_line(box, width: int, height: int) -> str:
    x, y, w, h = (float(value) for value in box)
    return f"0 {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, help="One VIDEO_XX asset per labeling session")
    parser.add_argument("--manifest", type=Path, default=Path("drishti4d/configs/datasets.json"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/chit_reviewed"))
    parser.add_argument("--sample-every-sec", type=float, default=2.0)
    parser.add_argument("--maximum-samples", type=int, default=0)
    args = parser.parse_args()
    if args.sample_every_sec <= 0:
        raise SystemExit("--sample-every-sec must be positive")

    assets = {asset.asset_id: asset for asset in load_drive_manifest(ROOT / args.manifest)}
    if args.asset not in assets:
        raise SystemExit(f"Unknown asset {args.asset}; choose one of {sorted(assets)}")
    dataset = (ROOT / args.dataset_dir).resolve()
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    reviewed = 0
    provenance = None
    with materialize_drive_asset(assets[args.asset], temp_root=dataset / ".staging") as (video_path, provenance):
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        stride = max(1, round(fps * args.sample_every_sec))
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if frame_index % stride:
                    frame_index += 1
                    continue
                preview = frame.copy()
                cv2.putText(preview, "B=boxes N=negative S=skip Q=quit", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("NETRA-4D real chit labeling", preview)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
                boxes = []
                if key == ord("b"):
                    boxes = list(cv2.selectROIs("Draw every visible chit; ENTER when done", frame, False, False))
                    cv2.destroyWindow("Draw every visible chit; ENTER when done")
                    boxes = [box for box in boxes if box[2] > 1 and box[3] > 1]
                    if not boxes:
                        frame_index += 1
                        continue
                elif key != ord("n"):
                    frame_index += 1
                    continue
                split = "val" if reviewed % 5 == 4 else "train"
                stem = f"{args.asset}_frame_{frame_index:07d}"
                image_path = dataset / "images" / split / f"{stem}.jpg"
                label_path = dataset / "labels" / split / f"{stem}.txt"
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"Unable to save {image_path}")
                height, width = frame.shape[:2]
                label_path.write_text("\n".join(yolo_line(box, width, height) for box in boxes), encoding="utf-8")
                reviewed += 1
                if args.maximum_samples and reviewed >= args.maximum_samples:
                    break
                frame_index += 1
        finally:
            capture.release()
            cv2.destroyAllWindows()

    (dataset / "dataset.yaml").write_text(yaml.safe_dump({
        "path": str(dataset), "train": "images/train", "val": "images/val", "names": {0: "chit"}
    }, sort_keys=False), encoding="utf-8")
    receipt = {
        "asset": args.asset,
        "human_reviewed_examples_added": reviewed,
        "source_sha256": provenance["sha256"] if provenance else None,
        "important": "Empty label files are explicit human-reviewed negative frames; skipped frames are not saved.",
    }
    (dataset / f"labeling_receipt_{args.asset}.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
