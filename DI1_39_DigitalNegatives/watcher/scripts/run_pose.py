#!/usr/bin/env python3
"""
Phase 4 Validation Script: Pose Estimation and Keypoint Stability

This script validates the pose estimation module by:
1. Loading configs (pose, detector, seats, source)
2. Running detection + tracking + pose estimation on video frames
3. Drawing pose skeletons and keypoints
4. Applying smoothing and confidence gating
5. Saving annotated output video/images
6. Printing pose estimation summary

Usage:
    python scripts/run_pose.py
    python scripts/run_pose.py --frames 100 --output outputs/overlays/pose_sample.mp4
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from watcher.source import VideoSource, create_source
from watcher.tracker import create_tracker
from watcher.pose import create_pose_estimator
from watcher.utils import setup_logging, get_project_root, print_source_info


def main():
    parser = argparse.ArgumentParser(description="Validate Watcher pose estimation")
    parser.add_argument(
        "--pose-config",
        default="configs/pose.yaml",
        help="Path to pose config YAML"
    )
    parser.add_argument(
        "--detector-config",
        default="configs/detector.yaml",
        help="Path to detector config YAML"
    )
    parser.add_argument(
        "--seats-config",
        default="configs/seats.yaml",
        help="Path to seats config YAML"
    )
    parser.add_argument(
        "--source-config",
        default="configs/source.yaml",
        help="Path to source config YAML"
    )
    parser.add_argument(
        "--source",
        help="Override source string"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process (default: 100)"
    )
    parser.add_argument(
        "--output",
        default="outputs/overlays/pose_sample.mp4",
        help="Path to save annotated output video"
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=5,
        help="Number of annotated frames to save as images"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save output video"
    )
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Don't draw skeleton connections"
    )
    parser.add_argument(
        "--no-keypoints",
        action="store_true",
        help="Don't draw keypoints"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Resolve config paths relative to watcher/ project root
    watcher_root = get_project_root() / "watcher"
    
    pose_config_path = Path(args.pose_config)
    if not pose_config_path.is_absolute():
        pose_config_path = watcher_root / pose_config_path
    
    detector_config_path = Path(args.detector_config)
    if not detector_config_path.is_absolute():
        detector_config_path = watcher_root / detector_config_path
    
    seats_config_path = Path(args.seats_config)
    if not seats_config_path.is_absolute():
        seats_config_path = watcher_root / seats_config_path
    
    source_config_path = Path(args.source_config)
    if not source_config_path.is_absolute():
        source_config_path = watcher_root / source_config_path
    
    for name, path in [("Pose", pose_config_path), ("Detector", detector_config_path), 
                       ("Seats", seats_config_path), ("Source", source_config_path)]:
        if not path.exists():
            print(f"ERROR: {name} config not found: {path}")
            sys.exit(1)
    
    # Override source if provided
    if args.source:
        import yaml
        with open(source_config_path, 'r') as f:
            source_config = yaml.safe_load(f)
        source_config["source"]["active"] = args.source
        with open(source_config_path, 'w') as f:
            yaml.safe_dump(source_config, f)
        print(f"Overrode source to: {args.source}")
    
    # Create video source
    print("Initializing video source...")
    source = create_source(str(source_config_path))
    source_info = source.open()
    print_source_info(source_info)
    
    if not source_info.is_opened:
        print(f"ERROR: Failed to open source: {source_info.error_message}")
        sys.exit(1)
    
    # Create tracker (includes detector + seat map)
    print("Initializing tracker...")
    tracker = create_tracker(str(detector_config_path), str(seats_config_path))
    
    # Create pose estimator
    print("Initializing pose estimator...")
    pose_estimator = create_pose_estimator(str(pose_config_path))
    
    # Prepare output video writer
    video_writer = None
    if not args.no_video:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = watcher_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            source_info.fps if source_info.fps > 0 else 25.0,
            (source_info.width, source_info.height)
        )
        print(f"Output video: {output_path}")
    
    # Prepare frame image output directory
    frames_output_dir = watcher_root / "outputs" / "debug" / "pose_frames"
    frames_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process frames
    num_frames = args.frames
    print(f"Processing {num_frames} frames...")
    
    total_poses = 0
    total_inference_time = 0.0
    frames_saved = 0
    
    for i in range(num_frames):
        ret, frame = source.read_frame()
        if not ret:
            print(f"Stopped at frame {i + 1}: no more frames or read error")
            break
        
        # Run tracking
        track_result = tracker.process_frame(frame, frame_idx=i)
        
        # Extract track info for pose matching
        track_ids = [t.track_id for t in track_result.tracks]
        track_bboxes = [t.bbox for t in track_result.tracks]
        
        # Run pose estimation
        pose_result = pose_estimator.estimate(
            frame, frame_idx=i, 
            track_ids=track_ids, track_bboxes=track_bboxes
        )
        
        # Get summary
        summary = pose_estimator.get_summary(pose_result)
        total_poses += summary["num_poses"]
        total_inference_time += summary["inference_time_ms"]
        
        # Draw result: seats + tracks + poses
        annotated_frame = tracker.draw_result(frame, track_result, draw_seats=True)
        annotated_frame = pose_estimator.draw_poses(
            annotated_frame, pose_result,
            draw_skeleton=not args.no_skeleton,
            draw_keypoints=not args.no_keypoints
        )
        
        # Write to output video
        if video_writer is not None:
            video_writer.write(annotated_frame)
        
        # Save sample frames
        if frames_saved < args.save_frames:
            frame_path = frames_output_dir / f"pose_frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)
            frames_saved += 1
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1} frames... (poses: {summary['num_poses']}, "
                  f"avg_conf: {summary['avg_pose_confidence']:.2f}, "
                  f"avg_valid_kp: {summary['avg_valid_keypoints']:.1f}, "
                  f"FPS: {summary['fps']:.1f})")
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    source.close()
    
    # Print summary
    avg_fps = 1000.0 / (total_inference_time / num_frames) if num_frames > 0 and total_inference_time > 0 else 0
    
    print("\n" + "=" * 50)
    print("PHASE 4 VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Frames Processed:       {num_frames}")
    print(f"Total Poses:            {total_poses}")
    print(f"Avg Poses/Frame:        {total_poses / num_frames:.1f}")
    print(f"Avg Inference Time:     {total_inference_time / num_frames:.2f} ms")
    print(f"Avg FPS:                {avg_fps:.1f}")
    if not args.no_video:
        print(f"Output Video:           {args.output}")
    print(f"Sample Frames Saved:    {frames_saved} to {frames_output_dir}")
    print("=" * 50)
    
    # Human verification prompts
    print("\nHUMAN VERIFICATION REQUIRED:")
    print("1. Check outputs/overlays/pose_sample.mp4 - are pose skeletons visible?")
    print("2. Check outputs/debug/pose_frames/ - do keypoints and connections look correct?")
    print("3. Are major keypoints visible (shoulders, nose, wrists, elbows)?")
    print("4. Is skeleton less jittery after smoothing?")
    print("5. Does pose estimation work when person is partially visible?")


if __name__ == "__main__":
    main()