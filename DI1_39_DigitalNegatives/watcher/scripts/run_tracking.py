#!/usr/bin/env python3
"""
Phase 3 Validation Script: Tracking and SeatTrack Binding

This script validates the tracking module by:
1. Loading configs (detector, seats, source)
2. Running detection + tracking on video frames
2. Binding tracks to seat regions
3. Drawing seat polygons, track IDs, and seat assignments
4. Saving annotated output video/images
5. Printing tracking summary

Usage:
    python scripts/run_tracking.py
    python scripts/run_tracking.py --frames 100 --output outputs/overlays/tracking_sample.mp4
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
from watcher.utils import setup_logging, get_project_root, print_source_info


def main():
    parser = argparse.ArgumentParser(description="Validate Watcher tracking and SeatTrack binding")
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
        default="outputs/overlays/tracking_sample.mp4",
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
        "--no-seats",
        action="store_true",
        help="Don't draw seat polygons"
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
    
    detector_config_path = Path(args.detector_config)
    if not detector_config_path.is_absolute():
        detector_config_path = watcher_root / detector_config_path
    
    seats_config_path = Path(args.seats_config)
    if not seats_config_path.is_absolute():
        seats_config_path = watcher_root / seats_config_path
    
    source_config_path = Path(args.source_config)
    if not source_config_path.is_absolute():
        source_config_path = watcher_root / source_config_path
    
    for name, path in [("Detector", detector_config_path), ("Seats", seats_config_path), ("Source", source_config_path)]:
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
    print("Initializing tracker with seat map...")
    tracker = create_tracker(str(detector_config_path), str(seats_config_path))
    
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
    frames_output_dir = watcher_root / "outputs" / "debug" / "tracking_frames"
    frames_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process frames
    num_frames = args.frames
    print(f"Processing {num_frames} frames...")
    
    total_tracks = 0
    total_seated = 0
    total_unassigned = 0
    total_inference_time = 0.0
    frames_saved = 0
    
    for i in range(num_frames):
        ret, frame = source.read_frame()
        if not ret:
            print(f"Stopped at frame {i + 1}: no more frames or read error")
            break
        
        # Run tracking
        result = tracker.process_frame(frame, frame_idx=i)
        
        # Get summary
        summary = tracker.get_summary(result)
        total_tracks += summary["total_tracks"]
        total_seated += summary["seated_tracks"]
        total_unassigned += summary["unassigned_tracks"]
        total_inference_time += summary["inference_time_ms"]
        
        # Draw result
        annotated_frame = tracker.draw_result(frame, result, draw_seats=not args.no_seats)
        
        # Write to output video
        if video_writer is not None:
            video_writer.write(annotated_frame)
        
        # Save sample frames
        if frames_saved < args.save_frames:
            frame_path = frames_output_dir / f"tracking_frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)
            frames_saved += 1
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            seat_info = ", ".join([f"{k}:{v}" for k,v in summary["seat_assignments"].items()])
            print(f"  Processed {i + 1} frames... (tracks: {summary['total_tracks']}, seated: {summary['seated_tracks']}, {seat_info})")
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    source.close()
    
    # Print summary
    avg_fps = 1000.0 / (total_inference_time / num_frames) if num_frames > 0 and total_inference_time > 0 else 0
    
    print("\n" + "=" * 50)
    print("PHASE 3 VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Frames Processed:       {num_frames}")
    print(f"Avg Tracks/Frame:       {total_tracks / num_frames:.1f}")
    print(f"  Seated:               {total_seated}")
    print(f"  Unassigned:           {total_unassigned}")
    print(f"Avg Inference Time:     {total_inference_time / num_frames:.2f} ms")
    print(f"Avg FPS:                {avg_fps:.1f}")
    if not args.no_video:
        print(f"Output Video:           {args.output}")
    print(f"Sample Frames Saved:    {frames_saved} to {frames_output_dir}")
    print("=" * 50)
    
    # Human verification prompts
    print("\nHUMAN VERIFICATION REQUIRED:")
    print("1. Check outputs/overlays/tracking_sample.mp4 - are seat polygons visible?")
    print("2. Check outputs/debug/tracking_frames/ - do track IDs and seat assignments look correct?")
    print("3. Are tracks stable (same ID maintained across frames)?")
    print("4. Are tracks correctly assigned to seat regions?")
    print("5. Does tracking handle occlusion reasonably?")


if __name__ == "__main__":
    main()