"""Run the official Multi-HMR SMPL-X demo on sampled real Drive video frames.

Requires a separately installed official `naver/multi-hmr` checkout in a
Python/CUDA environment compatible with its pinned dependencies. It extracts
real frames, invokes `demo.py`, and records provenance. It never synthesizes
3-D joints or calls a 2-D pose result a mesh reconstruction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from drishti.data.google_drive import load_drive_manifest, materialize_drive_asset
from scripts.prepare_smplx_assets import safe_extract, sha256


def prepare_repository_smplx(repo: Path, archive: Path) -> Path:
    source = safe_extract(archive, repo / ".netra_smplx_extract")
    target = repo / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        shutil.copy2(source, target)
    return target


def extract_frames(source: Path, destination: Path, every: int, maximum: int) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    written: list[Path] = []
    index = 0
    try:
        while len(written) < maximum:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every == 0:
                output = destination / f"frame_{index:06d}.jpg"
                if not cv2.imwrite(str(output), frame):
                    raise RuntimeError(f"Failed to write sampled frame: {output}")
                written.append(output)
            index += 1
    finally:
        capture.release()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="drishti4d/configs/datasets.json")
    parser.add_argument("--asset", default="VIDEO_01")
    parser.add_argument("--repo", type=Path, required=True, help="Official naver/multi-hmr checkout")
    parser.add_argument("--smplx-archive", type=Path, default=ROOT / "models_smplx_v1_1.zip")
    parser.add_argument("--model-name", default="multiHMR_896_L_synth")
    parser.add_argument("--sample-every", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "multihmr_smplx")
    parser.add_argument("--python", default=sys.executable)
    # Multi-HMR's official demo is CUDA-only. Keep this runner strict so a
    # missing driver cannot silently produce a misleading CPU run.
    parser.add_argument("--require-cuda", action="store_true", help="Deprecated compatibility flag; CUDA is always required")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-HMR requires CUDA; NVIDIA driver/PyTorch CUDA is unavailable. No CPU fallback is permitted.")
    torch.cuda.set_device(0)
    if args.sample_every < 1 or args.max_frames < 1:
        raise ValueError("--sample-every and --max-frames must be positive")
    repo, archive = args.repo.resolve(), args.smplx_archive.resolve()
    demo = repo / "demo.py"
    if not demo.is_file():
        raise FileNotFoundError(f"Official Multi-HMR demo.py not found: {demo}")
    if not archive.is_file():
        raise FileNotFoundError(f"SMPL-X archive not found: {archive}")
    smplx_neutral = prepare_repository_smplx(repo, archive)
    manifest = load_drive_manifest((ROOT / args.manifest).resolve())
    asset = next((item for item in manifest if item.asset_id == args.asset), None)
    if asset is None:
        raise ValueError(f"Unknown asset: {args.asset}")
    run_id = datetime.now(timezone.utc).strftime("multihmr_smplx_%Y%m%dT%H%M%SZ")
    output = args.output_dir.resolve() / run_id
    frames = output / "input_frames"
    rendered = output / "rendered"
    with materialize_drive_asset(asset, temp_root=output / ".staging") as (source, provenance):
        sampled = extract_frames(source, frames, args.sample_every, args.max_frames)
    if not sampled:
        raise RuntimeError("No real video frames could be sampled for Multi-HMR")
    command = [
        args.python, str(demo), "--img_folder", str(frames), "--out_folder", str(rendered),
        "--model_name", args.model_name,
    ]
    subprocess.run(command, cwd=repo, check=True)
    record: dict[str, Any] = {
        "run_kind": "official_multihmr_smplx_frame_reconstruction",
        "model_name": args.model_name,
        "source": provenance,
        "sampled_frames": [path.name for path in sampled],
        "smplx_neutral": str(smplx_neutral),
        "smplx_neutral_sha256": sha256(smplx_neutral),
        "command": command,
        "rendered_directory": str(rendered),
        "claims": "Multi-HMR output only; not a cheating classifier or validated gaze estimate.",
    }
    (output / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "sampled_frames": len(sampled)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
