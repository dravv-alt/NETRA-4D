#!/usr/bin/env python3
"""
Phase 1 Validation Script: Check Video Source

This script validates the video source abstraction by:
1. Loading config from configs/source.yaml
2. Opening the configured source
3. Printing source metadata (type, status, resolution, FPS, frame count)
4. Reading 100 frames
5. Saving one test frame to outputs/debug/frame_0001.jpg

Usage:
    python scripts/check_source.py
    python scripts/check_source.py --config configs/source.yaml
    python scripts/check_source.py --source file://data/videos/sample.mp4
    python scripts/check_source.py --source webcam://0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from watcher.source import VideoSource, create_source
from watcher.utils import setup_logging, print_source_info, get_project_root


def main():
    parser = argparse.ArgumentParser(description="Validate Watcher video source")
    parser.add_argument(
        "--config", 
        default="configs/source.yaml",
        help="Path to source config YAML (default: configs/source.yaml)"
    )
    parser.add_argument(
        "--source",
        help="Override source string (e.g., file://data/videos/sample.mp4, webcam://0)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to read (default: 100)"
    )
    parser.add_argument(
        "--output",
        default="outputs/debug/frame_0001.jpg",
        help="Path to save test frame (default: outputs/debug/frame_0001.jpg)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save test frame"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Config is relative to the watcher/ project folder (parent of scripts/)
        watcher_root = Path(__file__).parent.parent
        config_path = watcher_root / config_path
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    # Create video source
    source = create_source(str(config_path))
    
    # Override source if provided
    if args.source:
        # We need to modify the config temporarily
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config["source"]["active"] = args.source
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        print(f"Overrode source to: {args.source}")
        # Recreate source with new config
        source = create_source(str(config_path))
    
    try:
        # Open source and get info
        print("Opening video source...")
        info = source.open()
        print_source_info(info)
        
        if not info.is_opened:
            print(f"ERROR: Failed to open source: {info.error_message}")
            sys.exit(1)
        
        # Read frames
        num_frames = args.frames
        print(f"Reading {num_frames} frames...")
        
        frames_read = 0
        first_frame = None
        
        for i in range(num_frames):
            ret, frame = source.read_frame()
            if not ret:
                print(f"Stopped at frame {i + 1}: no more frames or read error")
                break
            frames_read += 1
            if first_frame is None:
                first_frame = frame
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Read {i + 1} frames...")
        
        print(f"Successfully read {frames_read} frames")
        
        # Save test frame
        if not args.no_save and first_frame is not None:
            output_path = Path(args.output)
            if not output_path.is_absolute():
                output_path = get_project_root() / output_path
            
            print(f"Saving test frame to {output_path}...")
            if source.save_frame(first_frame, str(output_path)):
                print(f"SUCCESS: Test frame saved to {output_path}")
            else:
                print("ERROR: Failed to save test frame")
                sys.exit(1)
        elif args.no_save:
            print("Skipping frame save (--no-save flag)")
        else:
            print("WARNING: No frames were read, nothing to save")
        
        # Summary
        print("\n" + "=" * 50)
        print("PHASE 1 VALIDATION COMPLETE")
        print("=" * 50)
        print(f"Source:           {info.source_string}")
        print(f"Type:             {info.source_type.value}")
        print(f"Resolution:       {info.width}x{info.height}")
        print(f"FPS:              {info.fps:.2f}")
        print(f"Frames Read:      {frames_read}/{num_frames}")
        if info.frame_count > 0:
            print(f"Total Frames:     {info.frame_count}")
        print(f"Test Frame Saved: {'YES' if not args.no_save and first_frame is not None else 'NO'}")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        source.close()


if __name__ == "__main__":
    import logging
    main()