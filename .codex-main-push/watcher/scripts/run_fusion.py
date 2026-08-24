#!/usr/bin/env python3
"""
Phase 6 Validation Script: Object-Hand Fusion and Temporal Feature Window

This script validates the fusion and temporal window module by:
1. Loading configs (rules, pose, detector, seats, source)
2. Running detection + tracking + pose + features + fusion on video frames
3. Drawing fusion visualization (object-hand connections, fusion scores)
4. Maintaining temporal feature windows (5s, 30s, 90s)
5. Outputting fusion JSON and temporal stats
6. Saving annotated output video/images

Usage:
    python scripts/run_fusion.py
    python scripts/run_fusion.py --frames 200 --output outputs/overlays/fusion_sample.mp4
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
from watcher.fusion import create_fusion_manager
from watcher.utils import setup_logging, get_project_root, print_source_info


def main():
    parser = argparse.ArgumentParser(description="Validate Watcher object-hand fusion and temporal windows")
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
        default=200,
        help="Number of frames to process (default: 200)"
    )
    parser.add_argument(
        "--output",
        default="outputs/overlays/fusion_sample.mp4",
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
        default="outputs/debug/fusion_log.json",
        help="Path to save fusion JSON log"
    )
    parser.add_argument(
        "--save-temporal",
        default="outputs/debug/temporal_stats.json",
        help="Path to save temporal window stats"
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
    
    # Create fusion manager
    print("Initializing fusion manager...")
    fps = source_info.fps if source_info.fps > 0 else 25.0
    fusion_manager = create_fusion_manager(str(rules_config_path), fps)
    
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
            fps,
            (source_info.width, source_info.height)
        )
        print(f"Output video: {output_path}")
    
    # Prepare frame image output directory
    frames_output_dir = watcher_root / "outputs" / "debug" / "fusion_frames"
    frames_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare JSON log paths
    json_log_path = Path(args.save_json)
    if not json_log_path.is_absolute():
        json_log_path = watcher_root / json_log_path
    json_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    temporal_log_path = Path(args.save_temporal)
    if not temporal_log_path.is_absolute():
        temporal_log_path = watcher_root / temporal_log_path
    temporal_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process frames
    num_frames = args.frames
    print(f"Processing {num_frames} frames...")
    
    all_fusion_json = []
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
        timestamp = i / fps
        behavioral_features_list = feature_extractor.extract(
            pose_result, track_result, i, timestamp, tracker.seat_map
        )
        
        # Run detection for objects (using tracker's detector)
        detector_result = tracker.detector.detect(frame, frame_idx=i)
        
        # Process fusion and temporal windows
        fusion_results = fusion_manager.process_frame(
            frame_idx=i,
            timestamp=timestamp,
            track_result=track_result,
            pose_result=pose_result,
            behavioral_features_list=behavioral_features_list,
            detector_result=detector_result
        )
        
        # Convert fusion results to JSON
        fusion_json = fusion_manager.fusion_results_to_json(fusion_results)
        all_fusion_json.extend(fusion_json)
        
        # Draw result: seats + tracks + poses + features + fusion
        annotated_frame = tracker.draw_result(frame, track_result, draw_seats=True)
        annotated_frame = pose_estimator.draw_poses(annotated_frame, pose_result)
        annotated_frame = feature_extractor.draw_features(annotated_frame, behavioral_features_list)
        annotated_frame = fusion_manager.draw_fusion(annotated_frame, fusion_results)
        
        # Write to output video
        if video_writer is not None:
            video_writer.write(annotated_frame)
        
        # Save sample frames
        if frames_saved < args.save_frames:
            frame_path = frames_output_dir / f"fusion_frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)
            frames_saved += 1
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1} frames... (fusion events: {len(fusion_results)})")
            if fusion_results:
                fr = fusion_results[0]
                print(f"    Seat {fr.seat_id}: {fr.object_class} fusion={fr.fusion_score:.2f}")
    
    # Save fusion JSON log
    with open(json_log_path, 'w') as f:
        json.dump(all_fusion_json, f, indent=2)
    print(f"Fusion JSON log saved to: {json_log_path}")
    
    # Save temporal stats
    temporal_stats = fusion_manager.get_temporal_stats()
    with open(temporal_log_path, 'w') as f:
        json.dump(temporal_stats, f, indent=2)
    print(f"Temporal stats saved to: {temporal_log_path}")
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    source.close()
    
    # Print summary
    print("\n" + "=" * 50)
    print("PHASE 6 VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Frames Processed:       {num_frames}")
    print(f"Total Fusion Events:    {len(all_fusion_json)}")
    print(f"Seats with Temporal Data: {len(temporal_stats)}")
    if not args.no_video:
        print(f"Output Video:           {args.output}")
    print(f"Sample Frames Saved:    {frames_saved} to {frames_output_dir}")
    print(f"Fusion JSON Log:        {json_log_path}")
    print(f"Temporal Stats:         {temporal_log_path}")
    print("=" * 50)
    
    # Print sample temporal stats
    if temporal_stats:
        print("\nSAMPLE TEMPORAL STATS:")
        for seat_id, stats in list(temporal_stats.items())[:2]:
            print(f"  Seat {seat_id}:")
            for window_name in ["window_5s", "window_30s", "window_90s"]:
                if window_name in stats:
                    w = stats[window_name]
                    print(f"    {window_name}: frames={w.get('num_frames', 0)}, "
                          f"head={w.get('head_direction_counts', {})}, "
                          f"fusion={w.get('avg_fusion_score', 0):.3f}")
    
    # Human verification prompts
    print("\nHUMAN VERIFICATION REQUIRED:")
    print("1. Check outputs/overlays/fusion_sample.mp4 - are object-hand connections visible?")
    print("2. Check outputs/debug/fusion_frames/ - do fusion scores appear when objects near hands?")
    print("3. Check outputs/debug/fusion_log.json - is fusion JSON structured correctly?")
    print("4. Check outputs/debug/temporal_stats.json - do 5s/30s/90s windows update?")
    print("5. Do random objects away from hands NOT create high fusion scores?")
    print("6. Does temporal window correctly accumulate over time?")


if __name__ == "__main__":
    main()