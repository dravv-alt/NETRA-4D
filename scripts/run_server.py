"""Serve the read-only verified-run API and dashboard on localhost."""

from pathlib import Path
import sys

import uvicorn


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    uvicorn.run("drishti.api.app:app", host="127.0.0.1", port=8000, log_level="info")
