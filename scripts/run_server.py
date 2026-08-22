"""
Drishti-4D Server Runner.
Starts the FastAPI server and hosts the Frontend Forensic Dashboard.
"""

import uvicorn
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def main():
    print("=" * 80)
    print("[*] STARTING DRISHTI-4D / NETRA-4D FORENSIC SURVEILLANCE SERVER")
    print("=" * 80)
    print("    Dashboard UI: http://localhost:8000")
    print("    Swagger Docs: http://localhost:8000/docs")
    print("=" * 80)

    uvicorn.run(
        "drishti.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
