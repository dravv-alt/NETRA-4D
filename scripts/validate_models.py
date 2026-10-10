"""Validate registered model files and class maps without running inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from drishti.release.models import ModelRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    parser.add_argument("--include-unapproved", action="store_true", help="Audit experimental weights; does not approve them.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    registry = ModelRegistry(ROOT / args.registry, ROOT)
    report = registry.validate(include_unapproved=args.include_unapproved)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        target = (ROOT / args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
