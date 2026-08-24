#!/usr/bin/env python3
"""
Phase 2 Validation Script: Person and Object Detection

This script validates the detection module by:
1. Loading config from configs/detector.yaml
2. Loading the YOLO model
3. Running detection on video frames from source
4. Drawing bounding boxes
5. Saving annotated output video/images
6. Printing detection summary

Usage:
    python scripts/run_detection.py
    python scripts/run_detection.py --config configs/detector.yaml
    python scripts/run_detection.py --source file://data/videos/sample.mkv
    python scripts/run_detection.py --frames 100 --output outputs/overlays/detection_sample.mp4
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
from watcher.detector import Detector, create_detector
from watcher.utils import setup_logging, get_project_root


def main():
    parser = argparse.ArgumentParser(description="Validate Watcher person/object detection")
    parser.add_argument(
        "--config",
        default="configs/detector.yaml",
        help="Path to detector config YAML (default: configs/detector.yaml)"
    )
    parser.add_argument(
        "--source-config",
        default="configs/source.yaml",
        help="Path to source config YAML (default: configs/source.yaml)"
    )
    parser.add_argument(
        "--source",
        help="Override source string (e.g., file://data/videos/sample.mkv, webcam://0)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process (default: 100)"
    )
    parser.add_argument(
        "--output",
        default="outputs/overlays/detection_sample.mp4",
        help="Path to save annotated output video (default: outputs/overlays/detection_sample.mp4)"
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=5,
        help="Number of annotated frames to save as images (default: 5)"
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
    
    detector_config_path = Path(args.config)
    if not detector_config_path.is_absolute():
        detector_config_path = watcher_root / detector_config_path
    
    source_config_path = Path(args.source_config)
    if not source_config_path.is_absolute():
        source_config_path = watcher_root / source_config_path
    
    if not detector_config_path.exists():
        print(f"ERROR: Detector config not found: {detector_config_path}")
        sys.exit(1)
    
    if not source_config_path.exists():
        print(f"ERROR: Source config not found: {source_config_path}")
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
    
    from watcher.utils import print_source_info
    print_source_info(source_info)
    
    if not source_info.is_opened:
        print(f"ERROR: Failed to open source: {source_info.error_message}")
        sys.exit(1)
    
    # Create detector
    print("Loading YOLO detector...")
    detector = create_detector(str(detector_config_path))
    
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
    frames_output_dir = watcher_root / "outputs" / "debug" / "detection_frames"
    frames_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process frames
    num_frames = args.frames
    print(f"Processing {num_frames} frames...")
    
    total_detections = 0
    total_persons = 0
    total_objects = 0
    total_inference_time = 0.0
    frames_saved = 0
    
    for i in range(num_frames):
        ret, frame = source.read_frame()
        if not ret:
            print(f"Stopped at frame {i + 1}: no more frames or read error")
            break
        
        # Run detection
        result = detector.detect(frame, frame_idx=i)
        
        # Get summary
        summary = detector.get_summary(result)
        total_detections += summary["total_detections"]
        total_persons += summary["person_count"]
        total_objects += summary["object_count"]
        total_inference_time += summary["inference_time_ms"]
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, result)
        
        # Write to output video
        if video_writer is not None:
            video_writer.write(annotated_frame)
        
        # Save sample frames
        if frames_saved < args.save_frames:
            frame_path = frames_output_dir / f"detection_frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)
            frames_saved += 1
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1} frames... (persons: {summary['person_count']}, objects: {summary['object_count']}, FPS: {summary['fps']:.1f})")
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    source.close()
    
    # Print summary
    avg_fps = 1000.0 / (total_inference_time / num_frames) if num_frames > 0 and total_inference_time > 0 else 0
    
    print("\n" + "=" * 50)
    print("PHASE 2 VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Frames Processed:     {num_frames}")
    print(f"Total Detections:     {total_detections}")
    print(f"  Persons:            {total_persons}")
    print(f"  Objects:            {total_objects}")
    print(f"Avg Inference Time:   {total_inference_time / num_frames:.2f} ms")
    print(f"Avg FPS:              {avg_fps:.1f}")
    if not args.no_video:
        print(f"Output Video:         {args.output}")
    print(f"Sample Frames Saved:  {frames_saved} to {frames_output_dir}")
    print("=" * 50)
    
    # Human verification prompts
    print("\nHUMAN VERIFICATION REQUIRED:")
    print("1. Check outputs/overlays/detection_sample.mp4 - are persons detected?")
    print("2. Check outputs/debug/detection_frames/ - do boxes look reasonable?")
    print("3. Are there too many false detections?")
    print("4. Is the speed acceptable for your use case?")
    print("5. If phones/objects visible in video, are they detected?")


if __name__ == "__main__":
    main()