"""Render a calibration-only preview without running or fabricating detections."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drishti.cbt import CBTBehaviouralReasoner, draw_cbt_context, load_scene_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("workstation_map", type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = load_scene_map(ROOT / args.workstation_map)
    capture = cv2.VideoCapture(str((ROOT / args.source).resolve()))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to decode frame {args.frame_index}")
    scene.validate_frame_geometry(frame.shape[1], frame.shape[0])
    preview = draw_cbt_context(frame, CBTBehaviouralReasoner(scene))
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), preview):
        raise RuntimeError(f"Unable to write {output}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
