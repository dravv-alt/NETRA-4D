#!/usr/bin/env python3
"""
Phase 5 Validation Script: Head Orientation and Hand/Wrist Features

This script validates the feature extraction module by:
1. Loading configs (rules, pose, detector, seats, source)
2. Running detection + tracking + pose + feature extraction on video frames
3. Drawing feature visualization (head direction, hand states, torso angle)
4. Outputting feature JSON for inspection
5. Saving annotated output video/images

Usage:
    python scripts/run_features.py
    python scripts/run_features.py --frames 100 --output outputs/overlays/features_sample.mp4
"""

import argparse
import sys
import cv2
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from watcher.source import VideoSource, create_source
from watcher.tracker import create_tracker
from watcher.pose import create_pose_estimator
from watcher.features import create_feature_extractor
from watcher.utils import setup_logging, get_project_root, print_source_info


def main():
    parser = argparse.ArgumentParser(description="Validate Watcher head/hand feature extraction")
    parser.add_argument(
        "--rules-config",
        default="configs/rules.yaml",
        help="Path to rules config YAML"
    )
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
        default="outputs/overlays/features_sample.mp4",
        help="Path to save annotated output video"
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=5,
        help="Number of annotated frames to save as images"
    )
    parser.add_argument(
        "--save-json",
        default="outputs/debug/features_log.json",
        help="Path to save feature JSON log"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save output video"
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
    
    rules_config_path = Path(args.rules_config)
    if not rules_config_path.is_absolute():
        rules_config_path = watcher_root / rules_config_path
    
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
    
    for name, path in [("Rules", rules_config_path), ("Pose", pose_config_path),
                       ("Detector", detector_config_path), ("Seats", seats_config_path),
                       ("Source", source_config_path)]:
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
    
    # Create feature extractor
    print("Initializing feature extractor...")
    feature_extractor = create_feature_extractor(str(rules_config_path))
    
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
    frames_output_dir = watcher_root / "outputs" / "debug" / "feature_frames"
    frames_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare JSON log
    json_log_path = Path(args.save_json)
    if not json_log_path.is_absolute():
        json_log_path = watcher_root / json_log_path
    json_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process frames
    num_frames = args.frames
    print(f"Processing {num_frames} frames...")
    
    all_features_json = []
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
        
        # Extract features
        timestamp = i / source_info.fps if source_info.fps > 0 else i * 0.04
        features_list = feature_extractor.extract(
            pose_result, track_result, i, timestamp, tracker.seat_map
        )
        
        # Convert to JSON
        features_json = feature_extractor.features_to_json(features_list)
        all_features_json.extend(features_json)
        
        # Draw result: seats + tracks + poses + features
        annotated_frame = tracker.draw_result(frame, track_result, draw_seats=True)
        annotated_frame = pose_estimator.draw_poses(annotated_frame, pose_result)
        annotated_frame = feature_extractor.draw_features(annotated_frame, features_list)
        
        # Write to output video
        if video_writer is not None:
            video_writer.write(annotated_frame)
        
        # Save sample frames
        if frames_saved < args.save_frames:
            frame_path = frames_output_dir / f"feature_frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)
            frames_saved += 1
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1} frames... (features: {len(features_list)})")
            # Print sample feature
            if features_list:
                f = features_list[0]
                print(f"    Track {f.track_id}: Head={f.head.direction.value}, "
                      f"L-Hand={f.hands.left_hand_state.value}, "
                      f"R-Hand={f.hands.right_hand_state.value}, "
                      f"Torso={f.torso.angle:.1f}deg")
    
    # Save JSON log
    with open(json_log_path, 'w') as f:
        json.dump(all_features_json, f, indent=2)
    print(f"Feature JSON log saved to: {json_log_path}")
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    source.close()
    
    # Print summary
    print("\n" + "=" * 50)
    print("PHASE 5 VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Frames Processed:       {num_frames}")
    print(f"Total Feature Entries:  {len(all_features_json)}")
    if not args.no_video:
        print(f"Output Video:           {args.output}")
    print(f"Sample Frames Saved:    {frames_saved} to {frames_output_dir}")
    print(f"Feature JSON Log:       {json_log_path}")
    print("=" * 50)
    
    # Human verification prompts
    print("\nHUMAN VERIFICATION REQUIRED:")
    print("1. Check outputs/overlays/features_sample.mp4 - are head directions visible?")
    print("2. Check outputs/debug/feature_frames/ - do hand states (BELOW_DESK, NEAR_FACE) appear?")
    print("3. Check outputs/debug/features_log.json - is feature JSON structured correctly?")
    print("4. Do head directions change appropriately (FRONT/LEFT/RIGHT/DOWN/UP)?")
    print("5. Does normal posture NOT constantly trigger suspicious states?")
    print("6. Are torso angles reasonable?")


if __name__ == "__main__":
    main()