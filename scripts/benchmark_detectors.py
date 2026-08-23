"""Measure YOLO11n versus YOLO26n detection on the same real video frames.

This is a runtime/count comparison, not an accuracy benchmark: no reviewed
ground truth is supplied with the Drive videos. Each model sees identical
decoded frames from one source asset and writes a JSON evidence report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drishti.data.google_drive import load_drive_manifest, materialize_drive_asset
from drishti.providers.object_provider import ObjectDisambiguationProvider
from scripts.run_verified_inference import file_sha256, model_record, resolve_device


def percentile(values: list[float], fraction: float) -> float | None:
    return float(np.percentile(values, fraction)) if values else None


def benchmark_model(
    model_path: Path, frames: list[tuple[int, float, np.ndarray]], device: str, confidence: float,
    warmup_frames: int,
) -> dict[str, Any]:
    provider = ObjectDisambiguationProvider(str(model_path), conf_thresh=confidence, device=device)
    provider.start()
    latencies: list[float] = []
    class_counts: dict[str, int] = {}
    phone_count = 0
    try:
        for frame_index, timestamp, frame in frames[:warmup_frames]:
            provider.process_frame(frame_index, timestamp, frame, video_id="benchmark_warmup")
        for frame_index, timestamp, frame in frames:
            started = perf_counter()
            _, objects = provider.process_frame(frame_index, timestamp, frame, video_id="benchmark")
            latencies.append((perf_counter() - started) * 1000.0)
            for item in objects:
                raw_class = str(item.metadata.get("raw_model_class", item.object_class))
                class_counts[raw_class] = class_counts.get(raw_class, 0) + 1
                phone_count += int(raw_class == "cell phone")
    finally:
        provider.close()
    return {
        "model": model_record(model_path),
        "frames": len(frames),
        "warmup_frames_excluded": min(warmup_frames, len(frames)),
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "raw_object_count": sum(class_counts.values()),
        "raw_cell_phone_count": phone_count,
        "raw_class_counts": dict(sorted(class_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="drishti4d/configs/datasets.json")
    parser.add_argument("--asset", default="VIDEO_01")
    parser.add_argument("--model", action="append", default=["models/weights/yolo11n.pt", "models/weights/yolo26n.pt"])
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--warmup-frames", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_frames < 1 or args.sample_every < 1 or args.warmup_frames < 0:
        raise ValueError("--max-frames and --sample-every must be positive; --warmup-frames cannot be negative")
    device = resolve_device(args.device, args.require_cuda)
    manifest = load_drive_manifest((ROOT / args.manifest).resolve())
    asset = next((item for item in manifest if item.asset_id == args.asset), None)
    if asset is None:
        raise ValueError(f"Unknown asset: {args.asset}")
    frames: list[tuple[int, float, np.ndarray]] = []
    with materialize_drive_asset(asset, temp_root=ROOT / "outputs" / ".benchmark_staging") as (source, provenance):
        capture = cv2.VideoCapture(str(source))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            raise RuntimeError(f"Invalid FPS for {asset.asset_id}")
        frame_index = 0
        while len(frames) < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.sample_every == 0:
                frames.append((frame_index, frame_index / fps, frame.copy()))
            frame_index += 1
        capture.release()
    if not frames:
        raise RuntimeError("No real frames were decoded for benchmark")
    model_paths = [(ROOT / value).resolve() for value in args.model]
    for model_path in model_paths:
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model weight: {model_path}")
    report = {
        "run_kind": "real_detector_runtime_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accuracy_claim": "not evaluated: no reviewed ground-truth labels supplied",
        "asset": {"id": asset.asset_id, "source": provenance, "sampled_frames": [item[0] for item in frames]},
        "device": device,
        "confidence": args.confidence,
        "results": [
            benchmark_model(path, frames, device, args.confidence, args.warmup_frames)
            for path in model_paths
        ],
    }
    output = args.output or ROOT / "outputs" / "benchmarks" / f"{asset.asset_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "models": len(model_paths), "frames": len(frames)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
