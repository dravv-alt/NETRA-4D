"""Human-calibrate semantic CBT workstation regions on one real source frame.

No default/fixed boxes are generated. Required regions are drawn by a human;
cancelled optional regions remain absent in the validated map.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import sys

import cv2
from PIL import Image, ImageTk
import tkinter as tk
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drishti.cbt.scene import CBTSceneMap
from drishti.data.google_drive import load_drive_manifest, materialize_drive_asset


def rectangle(box, width: int, height: int):
    x, y, w, h = (float(value) for value in box)
    if w <= 1 or h <= 1:
        return None
    return {"points": [
        (x / width, y / height), ((x + w) / width, y / height),
        ((x + w) / width, (y + h) / height), (x / width, (y + h) / height),
    ]}


def select_region(frame, label: str, width: int, height: int, *, required: bool):
    while True:
        try:
            box = cv2.selectROI(f"Draw {label}; ENTER accepts, C cancels", frame, False, False)
            cv2.destroyWindow(f"Draw {label}; ENTER accepts, C cancels")
        except cv2.error:
            box = select_region_tk(frame, label)
        region = rectangle(box, width, height)
        if region or not required:
            return region
        print(f"{label} is required for every workstation.")


def select_region_tk(frame, label: str):
    """Tkinter fallback for environments where OpenCV HighGUI is headless."""
    height, width = frame.shape[:2]
    scale = min(1.0, 1400 / width, 820 / height)
    display_width, display_height = round(width * scale), round(height * scale)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize((display_width, display_height), Image.Resampling.LANCZOS)
    root = tk.Tk()
    root.title(f"Draw {label}; ENTER accepts, ESC skips optional region")
    canvas = tk.Canvas(root, width=display_width, height=display_height, cursor="crosshair")
    canvas.pack()
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    state = {"start": None, "rect": None, "box": (0, 0, 0, 0)}

    def press(event):
        state["start"] = (event.x, event.y)
        if state["rect"]:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00ff66", width=2)

    def drag(event):
        if state["start"] and state["rect"]:
            canvas.coords(state["rect"], state["start"][0], state["start"][1], event.x, event.y)

    def release(event):
        if not state["start"]:
            return
        x1, y1 = state["start"]
        x2, y2 = event.x, event.y
        left, top = min(x1, x2), min(y1, y2)
        state["box"] = (left / scale, top / scale, abs(x2 - x1) / scale, abs(y2 - y1) / scale)

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", release)
    root.bind("<Return>", lambda _event: root.destroy())
    root.bind("<Escape>", lambda _event: (state.update(box=(0, 0, 0, 0)), root.destroy()))
    root.mainloop()
    return state["box"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="Local original or evidence video")
    source.add_argument("--asset", help="VIDEO_XX from the Drive manifest")
    parser.add_argument("--manifest", type=Path, default=Path("drishti4d/configs/datasets.json"))
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--calibrated-by", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--auto-horizontal-neighbours", action="store_true")
    args = parser.parse_args()

    video_id = args.video_id or args.asset or args.camera_id
    output = (ROOT / (args.output or Path(f"drishti4d/configs/workstations/{video_id}.yaml"))).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.asset:
        assets = {asset.asset_id: asset for asset in load_drive_manifest(ROOT / args.manifest)}
        if args.asset not in assets:
            raise SystemExit(f"Unknown asset {args.asset}; choose one of {sorted(assets)}")
        asset = assets[args.asset]
        context = materialize_drive_asset(asset, temp_root=ROOT / "outputs" / ".calibration_staging")
        source_label = asset.source_url
    else:
        path = args.source.resolve()
        if not path.is_file():
            raise SystemExit(f"Video does not exist: {path}")
        context = nullcontext((path, {"sha256": None}))
        source_label = str(path)

    with context as (video_path, _):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to decode frame {args.frame_index}")

    height, width = frame.shape[:2]
    encoded_ok, encoded = cv2.imencode(".png", frame)
    if not encoded_ok:
        raise RuntimeError("Unable to encode calibration frame for provenance")
    frame_hash = sha256(encoded.tobytes()).hexdigest()
    workstations = []
    while True:
        workstation_id = input("Workstation ID (blank when finished): ").strip()
        if not workstation_id:
            break
        item = {
            "workstation_id": workstation_id,
            "seat": select_region(frame, f"{workstation_id} seat", width, height, required=True),
            "own_monitor": select_region(frame, f"{workstation_id} own monitor", width, height, required=True),
        }
        for field_name, label in (
            ("candidate_body", "candidate body envelope"), ("keyboard", "keyboard"),
            ("mouse", "mouse"), ("desk", "desk surface"), ("under_desk", "under-desk region"),
        ):
            region = select_region(frame, f"{workstation_id} optional {label}", width, height, required=False)
            if region:
                item[field_name] = region
        workstations.append(item)
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    if not workstations:
        raise SystemExit("No workstation was calibrated; no file was written")
    if args.auto_horizontal_neighbours:
        ordered = sorted(workstations, key=lambda item: sum(point[0] for point in item["own_monitor"]["points"]) / 4)
        for index, item in enumerate(ordered):
            if index:
                item["left_neighbour_id"] = ordered[index - 1]["workstation_id"]
            if index + 1 < len(ordered):
                item["right_neighbour_id"] = ordered[index + 1]["workstation_id"]

    payload = {
        "schema_version": "1.0.0", "calibration_status": "VALIDATED",
        "camera_id": args.camera_id, "video_id": video_id,
        "reference_width": width, "reference_height": height, "workstations": workstations,
        "provenance": {
            "source_video": source_label, "source_frame_index": args.frame_index,
            "source_frame_sha256": frame_hash, "calibrated_by": args.calibrated_by,
            "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "human_roi_calibration",
        },
    }
    validated = CBTSceneMap.model_validate(payload)
    output.write_text(yaml.safe_dump(validated.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    print(f"Validated {len(validated.workstations)} workstations -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
