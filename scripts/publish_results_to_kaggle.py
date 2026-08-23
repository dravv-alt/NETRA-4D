"""Publish verified NETRA-4D results and create a private Kaggle notebook.

The user explicitly authorizes reading ``username`` and ``key`` from the given
``kaggle.json``. The key is passed to official Kaggle CLI child processes only
through their environment; it is never printed, copied into a bundle, written
to metadata, or included in subprocess arguments.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "outputs" / "full_video_runs" / "20260822_full_video_cpu"
DEFAULT_KEY = ROOT / "kaggle.json"
DEFAULT_KERNEL_BUNDLE = ROOT / "outputs" / "kaggle_netra4d_notebook"


def load_kaggle_credentials(path: Path) -> tuple[str, str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    username = payload.get("username")
    key = payload.get("key")
    if not isinstance(username, str) or not username.strip():
        raise ValueError("kaggle.json has no valid username")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("kaggle.json has no valid key")
    return username.strip(), key.strip()


def validate_results(results: Path) -> dict[str, Any]:
    run_path = results / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"Missing verified run manifest: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    assets = run.get("assets", [])
    if len(assets) != 5:
        raise RuntimeError(f"Expected 5 completed videos, found {len(assets)}")
    for asset in assets:
        asset_dir = results / asset["asset_id"]
        if not (asset_dir / "COMPLETE.json").is_file():
            raise RuntimeError(f"Incomplete video folder: {asset_dir}")
        if (asset_dir / "INCOMPLETE.txt").exists():
            raise RuntimeError(f"Incomplete marker present: {asset_dir}")
        video = asset_dir / asset["outputs"]["annotated_video"]
        if not video.is_file() or video.stat().st_size != asset["integrity"]["annotated_video_bytes"]:
            raise RuntimeError(f"Annotated video size mismatch: {video}")
    return run


def kaggle_environment(username: str, key: str) -> dict[str, str]:
    environment = os.environ.copy()
    config_dir = ROOT / ".secrets" / "kaggle_runtime"
    config_dir.mkdir(parents=True, exist_ok=True)
    environment["KAGGLE_CONFIG_DIR"] = str(config_dir)
    environment["KAGGLE_USERNAME"] = username
    environment["KAGGLE_KEY"] = key
    return environment


def run_cli(arguments: list[str], environment: dict[str, str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("kaggle")
    command = [executable, *arguments] if executable else [sys.executable, "-m", "kaggle", *arguments]
    completed = subprocess.run(
        command,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Kaggle CLI failed with exit code {completed.returncode}:\n{completed.stdout.strip()}"
        )
    return completed


def write_dataset_metadata(results: Path, username: str, dataset_slug: str) -> Path:
    metadata = {
        "title": "NETRA-4D Verified Full Video Results",
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
        "description": (
            "Actual YOLO pose and object inference outputs for five supplied videos. "
            "Includes annotated MP4 files, frame-level detections, hashes, and completion manifests."
        ),
    }
    path = results / "dataset-metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def publish_dataset(
    results: Path,
    username: str,
    dataset_slug: str,
    environment: dict[str, str],
) -> str:
    reference = f"{username}/{dataset_slug}"
    status = run_cli(["datasets", "status", reference], environment, check=False)
    if status.returncode == 0:
        action = [
            "datasets", "version", "-p", str(results),
            "-m", "Refresh verified full-video outputs", "-r", "zip",
        ]
    else:
        # Kaggle creates datasets privately by default; do not pass --public.
        action = ["datasets", "create", "-p", str(results), "-r", "zip"]
    result = run_cli(action, environment)
    print(result.stdout.strip())
    return reference


def create_notebook_bundle(bundle: Path, username: str, dataset_reference: str, kernel_slug: str) -> Path:
    bundle.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# NETRA-4D verified full-video results\n",
                    "This notebook exposes measured model outputs only. It does not fabricate detections or metrics."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    f"RESULTS = Path('/kaggle/input/{dataset_reference.split('/', 1)[1]}')\n",
                    "run_files = list(RESULTS.rglob('run.json'))\n",
                    "assert run_files, f'run.json not found under {RESULTS}'\n",
                    "run = json.loads(run_files[0].read_text())\n",
                    "print(json.dumps(run['summary'], indent=2))\n",
                    "for asset in run['assets']:\n",
                    "    print(asset['asset_id'], asset['outputs']['annotated_video'], asset['counts'])\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (bundle / "netra_4d.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": "netra-4d",
        "code_file": "netra_4d.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_reference],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (bundle / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dataset-slug", default="netra-4d-verified-results")
    parser.add_argument("--kernel-slug", default="netra-4d")
    parser.add_argument("--kernel-bundle", type=Path, default=DEFAULT_KERNEL_BUNDLE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key_path = args.key.resolve()
    results = args.results.resolve()
    run = validate_results(results)
    username, key = load_kaggle_credentials(key_path)
    dataset_reference = f"{username}/{args.dataset_slug}"
    create_notebook_bundle(args.kernel_bundle.resolve(), username, dataset_reference, args.kernel_slug)
    print(f"Validated results: {run['summary']}")
    print(f"Kaggle account: {username}")
    print("Kaggle API key: loaded securely (value not displayed)")
    if args.dry_run:
        return 0

    write_dataset_metadata(results, username, args.dataset_slug)
    environment = kaggle_environment(username, key)
    publish_dataset(results, username, args.dataset_slug, environment)
    pushed = run_cli(["kernels", "push", "-p", str(args.kernel_bundle.resolve())], environment)
    print(pushed.stdout.strip())
    print(f"Dataset: https://www.kaggle.com/datasets/{dataset_reference}")
    print(f"Notebook: https://www.kaggle.com/code/{username}/{args.kernel_slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
