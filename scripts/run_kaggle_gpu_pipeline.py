"""Repeatable NETRA-4D Kaggle GPU pipeline.

Submits a CUDA-required kernel, waits for its terminal state, downloads the
actual kernel output, validates every per-video completion manifest, and
publishes the result as a private Kaggle dataset version. API secrets are
passed only to the existing publisher and are never printed or bundled.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.publish_results_to_kaggle import (
    DEFAULT_KEY,
    kaggle_environment,
    load_kaggle_credentials,
    publish_dataset,
    run_cli,
    validate_results,
    write_dataset_metadata,
)
from scripts.submit_kaggle_gpu_job import DEFAULT_BUNDLE, build_bundle

def kernel_status(reference: str, environment: dict[str, str]) -> str:
    result = run_cli(["kernels", "status", reference], environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Unable to read Kaggle kernel status:\n{result.stdout.strip()}")
    return result.stdout.strip()


def wait_for_kernel(reference: str, environment: dict[str, str], poll_seconds: int) -> None:
    while True:
        status = kernel_status(reference, environment)
        normalized = status.lower()
        print(f"Kaggle kernel status: {status}", flush=True)
        if "complete" in normalized:
            return
        if any(token in normalized for token in ("error", "failed", "cancelled")):
            raise RuntimeError(f"Kaggle GPU kernel did not complete successfully: {status}")
        time.sleep(poll_seconds)


def completed_run_root(download_dir: Path, expected_run_id: str) -> Path:
    candidates = [path.parent for path in download_dir.rglob("run.json")]
    expected = [path for path in candidates if path.name == expected_run_id]
    if len(expected) != 1:
        raise RuntimeError(
            f"Expected exactly one downloaded run named {expected_run_id!r}; found {expected}"
        )
    return expected[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--slug", default="netra-4d-full-video-real-model-inference")
    parser.add_argument("--title", help="Kaggle notebook title; use the existing notebook title when reusing it")
    parser.add_argument("--dataset-slug", default="netra-4d-verified-results")
    parser.add_argument("--run-id")
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--download-dir", type=Path, default=ROOT / "outputs" / "kaggle_gpu_downloads")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--submit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least 10")
    username, key = load_kaggle_credentials(args.key.resolve())
    run_id = args.run_id or datetime.now(timezone.utc).strftime("kaggle_gpu_%Y%m%dT%H%M%SZ")
    reference = f"{username}/{args.slug}"
    environment = kaggle_environment(username, key)
    bundle = build_bundle(
        args.bundle.resolve(), username, args.slug,
        run_id=run_id, assets=args.asset, title=args.title,
    )
    pushed = run_cli(["kernels", "push", "-p", str(bundle)], environment)
    print(pushed.stdout.strip())
    print(f"Submitted CUDA-required run: {reference} ({run_id})")
    if args.submit_only:
        return 0

    wait_for_kernel(reference, environment, args.poll_seconds)
    download_dir = args.download_dir.resolve() / run_id
    if download_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing download directory: {download_dir}")
    download_dir.mkdir(parents=True)
    downloaded = run_cli(["kernels", "output", reference, "-p", str(download_dir), "--force"], environment)
    print(downloaded.stdout.strip())
    results = completed_run_root(download_dir, run_id)
    run = validate_results(results)
    write_dataset_metadata(results, username, args.dataset_slug)
    dataset_reference = publish_dataset(results, username, args.dataset_slug, environment)
    print(f"Verified GPU output: {run['summary']}")
    print(f"Private Kaggle dataset: https://www.kaggle.com/datasets/{dataset_reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
