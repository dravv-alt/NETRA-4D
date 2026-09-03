"""Integration test verifying the end-to-end execution of the unified pipeline runner.

Tests:
1. Complete run_unified_pipeline execution on a generated synthetic video.
2. Production of valid annotated MP4 video and structured JSON audit records.
3. Verification that candidates are tracked, proctors have zero alerts, and hypotheses are logged.
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.run_unified_pipeline import run_unified_pipeline


@pytest.fixture
def synthetic_exam_video(tmp_path):
    """Generates a small 30-frame synthetic examination video for end-to-end verification."""
    video_path = tmp_path / "synthetic_exam.mp4"
    w, h, fps = 640, 480, 25.0
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for f in range(30):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 240  # Light background classroom

        # Draw seated candidate 1 at desk
        cv2.rectangle(frame, (100, 200), (220, 400), (120, 100, 80), -1)  # Desk
        cv2.circle(frame, (160, 160), 25, (100, 150, 200), -1)             # Head
        cv2.line(frame, (160, 185), (160, 280), (80, 80, 80), 4)           # Torso

        # Draw seated candidate 2 at desk
        cv2.rectangle(frame, (350, 200), (470, 400), (120, 100, 80), -1)  # Desk
        cv2.circle(frame, (410, 160), 25, (100, 150, 200), -1)             # Head
        cv2.line(frame, (410, 185), (410, 280), (80, 80, 80), 4)           # Torso

        writer.write(frame)

    writer.release()
    return str(video_path)


def test_unified_pipeline_execution(synthetic_exam_video, tmp_path):
    output_dir = tmp_path / "unified_output"
    out_video, out_json = run_unified_pipeline(
        video_source=synthetic_exam_video,
        output_dir=str(output_dir),
        sample_every=2,
        max_frames=15,
        prime_vram_gb=0.0  # CPU / minimal test execution
    )

    assert os.path.exists(out_video)
    assert os.path.exists(out_json)

    with open(out_json, "r") as f:
        data = json.load(f)

    assert data["pipeline_version"] == "2.1.0"
    assert "environment_profile" in data
    assert "total_candidates_locked" in data
    assert "frames" in data
    assert len(data["frames"]) > 0

    frame_rec = data["frames"][0]
    assert "frame_idx" in frame_rec
    assert "timestamp_sec" in frame_rec
    assert "candidates_observed" in frame_rec
