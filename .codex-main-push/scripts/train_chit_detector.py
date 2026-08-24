"""Train a real custom YOLO chit/paper detector from reviewed annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml


ALLOWED_NAMES = {"chit", "paper chit", "cheat sheet", "piece of paper", "note paper"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_yaml", type=Path)
    parser.add_argument("--base-model", default="models/weights/yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--project", default="outputs/training/chit_detector")
    args = parser.parse_args()

    if not args.dataset_yaml.is_file():
        raise SystemExit(f"Dataset YAML does not exist: {args.dataset_yaml}")
    config = yaml.safe_load(args.dataset_yaml.read_text(encoding="utf-8")) or {}
    names = config.get("names", {})
    values = names.values() if isinstance(names, dict) else names
    normalized = {str(value).strip().lower().replace("_", " ") for value in values or []}
    if not normalized.intersection(ALLOWED_NAMES):
        raise SystemExit(f"Dataset must contain an explicit chit/paper class; found {sorted(normalized)}")
    train_path, validation_path = config.get("train"), config.get("val")
    if not train_path or not validation_path:
        raise SystemExit("Dataset YAML must define reviewed train and val splits")

    dataset_root = Path(config.get("path", args.dataset_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (args.dataset_yaml.parent / dataset_root).resolve()
    positive_labels = {}
    for split_name, image_value in (("train", train_path), ("val", validation_path)):
        image_dir = Path(image_value)
        if not image_dir.is_absolute():
            image_dir = dataset_root / image_dir
        parts = list(image_dir.parts)
        if "images" not in parts:
            raise SystemExit(f"{split_name} path must use YOLO images/<split> layout: {image_dir}")
        parts[parts.index("images")] = "labels"
        label_dir = Path(*parts)
        label_files = list(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        positive = sum(bool(path.read_text(encoding="utf-8").strip()) for path in label_files)
        if not image_dir.is_dir() or not label_files or positive == 0:
            raise SystemExit(
                f"{split_name} must contain human-reviewed images, labels, and at least one positive chit box"
            )
        positive_labels[split_name] = positive

    from ultralytics import YOLO
    model = YOLO(args.base_model)
    model.train(
        data=str(args.dataset_yaml.resolve()), epochs=args.epochs, imgsz=args.image_size,
        device=args.device, project=args.project, name="yolo26n_chit", exist_ok=False,
    )
    print({"training_started_with_positive_labels": positive_labels})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
