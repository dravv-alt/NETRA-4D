"""Download official Ultralytics YOLO weights without fabricating assets.

Only official model names accepted by Ultralytics are used. Existing files are
left untouched unless their SHA-256 is recorded by the caller separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = ("yolo26n.pt", "yolo26n-pose.pt", "yolo11n.pt")


def fetch(model_name: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / model_name
    if target.is_file():
        print(f"Already present: {target}")
        return target
    model = YOLO(model_name)
    downloaded = Path(getattr(model, "ckpt_path", model_name))
    if not downloaded.is_file():
        downloaded = Path(model_name)
    if not downloaded.is_file():
        raise RuntimeError(f"Official Ultralytics weight was not materialized: {model_name}")
    shutil.copy2(downloaded, target)
    print(f"Downloaded official weight: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=DEFAULT_MODELS)
    parser.add_argument("--destination", type=Path, default=ROOT / "models" / "weights")
    args = parser.parse_args()
    for model_name in args.model or DEFAULT_MODELS:
        fetch(model_name, args.destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
