"""Fetch the pinned official Multi-HMR source required by the 3-D runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "https://github.com/naver/multi-hmr.git"
REVISION = "651fb411e1cbcc626aaa5f38805ecab9cc891f7a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "third_party" / "multi-hmr")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkout: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPOSITORY, str(destination)], check=True)
    subprocess.run(["git", "checkout", "--detach", REVISION], cwd=destination, check=True)
    print(f"Pinned official Multi-HMR source: {destination} @ {REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
