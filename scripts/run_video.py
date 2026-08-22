"""
Drishti-4D CLI Pipeline Execution Script.
Runs YOLO26-Pose / RTMPose / Multi-HMR on any video or mock stream,
computes 4D temporal behavior & risk, and generates explainable review events.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from drishti.video.reader import FileVideoSource
from drishti.context.camera_profile import CameraContextManager
from drishti.tracking.seat_anchor import SeatAnchorManager
from drishti.providers.yolo26_provider import YOLO26PoseProvider
from drishti.providers.object_provider import ObjectDisambiguationProvider
from drishti.behaviour.engine import BehaviourEngine
from drishti.risk.state_machine import RiskStateMachine
from drishti.events.builder import EventBuilder


def main():
    parser = argparse.ArgumentParser(description="Drishti-4D Video Inference Runner")
    parser.add_argument("--source", type=str, required=True, help="Path to video file")
    parser.add_argument("--camera-config", type=str, default=None, help="Path to camera YAML config")
    parser.add_argument("--sample-fps", type=float, default=12.0, help="Sampling frame rate")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for reports")
    parser.add_argument("--display", action="store_true", help="Display OpenCV window")
    args = parser.parse_args()

    print("=" * 80)
    print(f"👁️ DRISHTI-4D PIPELINE: {Path(args.source).name}")
    print("=" * 80)

    # 1. Video Source
    source = FileVideoSource(args.source)
    meta = source.get_metadata()
    print(f"[*] Video: {meta.width}x{meta.height} @ {meta.fps} FPS ({meta.duration_sec:.1f}s)")

    # 2. Camera Profile
    config_path = args.camera_config
    if not config_path:
        # Auto-match with configs/cameras
        cam_dir = Path(__file__).resolve().parent.parent / "drishti4d" / "configs" / "cameras"
        if cam_dir.exists():
            candidates = list(cam_dir.glob("*.yaml"))
            if candidates:
                config_path = str(candidates[0])

    cm = CameraContextManager(config_path)
    if cm.profile:
        print(f"[✔] Loaded Camera Profile: {cm.profile.camera_id} with {len(cm.profile.seats)} configured seats.")

    # 3. Initialize Pipeline Modules
    seat_mgr = SeatAnchorManager(cm.profile) if cm.profile else None
    pose_provider = YOLO26PoseProvider(conf_thresh=0.25)
    obj_provider = ObjectDisambiguationProvider(camera_profile=cm.profile, conf_thresh=0.25)
    behaviour_engine = BehaviourEngine(camera_context=cm, history_seconds=12.0)
    risk_machine = RiskStateMachine()
    event_builder = EventBuilder()

    pose_provider.start()
    obj_provider.start()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Processing frames...")
    start_wall_t = time.time()
    processed_count = 0

    for f_idx, ts, frame in source.read_frames(sample_fps=args.sample_fps):
        # A. Run Pose & Object Perception
        p_obs_list, _ = pose_provider.process_frame(f_idx, ts, frame, camera_id=cm.profile.camera_id if cm.profile else "CAM01")
        _, o_obs_list = obj_provider.process_frame(f_idx, ts, frame, camera_id=cm.profile.camera_id if cm.profile else "CAM01", person_obs_list=p_obs_list)

        # B. Seat Anchoring
        if seat_mgr:
            for obs in p_obs_list:
                seat_mgr.update(obs)

        # C. 4D Spatio-Temporal Behaviour Reasoning
        evidence = behaviour_engine.update(p_obs_list, o_obs_list)

        # D. Risk State Machine
        snapshots = risk_machine.update(ts, evidence, p_obs_list)

        # E. Event Builder
        event_builder.update(ts, snapshots, evidence, video_id=meta.filename)

        processed_count += 1
        if processed_count % 30 == 0:
            print(f"  -> Processed {processed_count} frames ({ts:.1f}s / {meta.duration_sec:.1f}s) | Active Tracks: {len(p_obs_list)}")

    # Finalize remaining events
    final_events = event_builder.flush(meta.duration_sec)
    source.close()
    pose_provider.close()

    elapsed = time.time() - start_wall_t
    print("=" * 80)
    print(f"[✔] FINISHED: Processed {processed_count} frames in {elapsed:.2f}s ({processed_count/elapsed:.1f} FPS)")
    print(f"[✔] Total Flagged Review Events: {len(event_builder.completed_events)}")
    print("=" * 80)

    # Save Output Report
    report_file = out_dir / f"{Path(args.source).stem}_forensic_report.json"
    event_builder.save_events_json(str(report_file))
    print(f"[✔] Forensic Report saved to: {report_file}")


if __name__ == "__main__":
    main()
