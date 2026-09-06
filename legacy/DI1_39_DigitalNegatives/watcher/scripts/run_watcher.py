#!/usr/bin/env python3
"""
Full Pipeline Validation Script: Watcher Phases 1-10

This script runs the complete Watcher pipeline:
1. Video Ingestion with Quality Enhancement
2. Person/Object Detection
3. Tracking & SeatTrack Binding
4. Pose Estimation
5. Feature Extraction (Head/Hand/Torso)
6. Object-Hand Fusion & Temporal Windows
7. Rule Engine & Event Detection
8. Risk Scoring & Alert Generation
9. SQLite Logging
10. Output Video & Debug Logs

Usage:
    python scripts/run_watcher.py --source file://data/videos/sample.mkv --frames 0
    python scripts/run_watcher.py --source file://data/videos/video_01.mp4 --output outputs/overlays/video_01_pipeline.mp4
"""

import argparse
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from watcher.source import create_source
from watcher.tracker import create_tracker
from watcher.pose import create_pose_estimator
from watcher.features import create_feature_extractor
from watcher.fusion import create_fusion_manager
from watcher.rules import create_rule_engine
from watcher.risk import create_risk_engine, create_database_manager, AlertManager
from watcher.utils import setup_logging, get_project_root, print_source_info


def main():
    parser = argparse.ArgumentParser(description="Run full Watcher pipeline")
    parser.add_argument(
        "--source-config",
        default="configs/source.yaml",
        help="Path to source config YAML"
    )
    parser.add_argument(
        "--source",
        help="Override source string (e.g., file://data/videos/sample.mkv)"
    )
    parser.add_argument(
        "--detector-config",
        default="configs/detector.yaml",
        help="Path to detector config"
    )
    parser.add_argument(
        "--seats-config",
        default="configs/seats.yaml",
        help="Path to seats config"
    )
    parser.add_argument(
        "--pose-config",
        default="configs/pose.yaml",
        help="Path to pose config"
    )
    parser.add_argument(
        "--rules-config",
        default="configs/rules.yaml",
        help="Path to rules config"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of frames to process (0 = all frames)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride (process every Nth frame, 1=all frames)"
    )
    parser.add_argument(
        "--output",
        default="outputs/overlays/pipeline_full.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--output-json",
        default="outputs/logs/pipeline_features.json",
        help="Output JSON log path"
    )
    parser.add_argument(
        "--db-path",
        default="watcher.db",
        help="SQLite database path"
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=10,
        help="Number of annotated frames to save"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save output video"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=None,
        help="Show live preview window (default: auto-enabled for webcam sources)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable live preview window even for webcam sources"
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
    
    # Resolve config paths
    watcher_root = get_project_root() / "watcher"
    
    configs = {
        "source": args.source_config,
        "detector": args.detector_config,
        "seats": args.seats_config,
        "pose": args.pose_config,
        "rules": args.rules_config
    }
    
    resolved = {}
    for name, path in configs.items():
        p = Path(path)
        if not p.is_absolute():
            p = watcher_root / p
        if not p.exists():
            print(f"ERROR: {name} config not found: {p}")
            sys.exit(1)
        resolved[name] = str(p)
    
    # Override source if provided
    if args.source:
        import yaml
        with open(resolved["source"], 'r') as f:
            source_config = yaml.safe_load(f)
        source_config["source"]["active"] = args.source
        with open(resolved["source"], 'w') as f:
            yaml.safe_dump(source_config, f)
        print(f"Overrode source to: {args.source}")
    
    # Determine if live preview should be shown
    is_webcam = args.source and args.source.lower().startswith("webcam://")
    show_preview = not args.no_show and (args.show or is_webcam)
    if show_preview:
        print("Live preview: ENABLED (press 'q' to quit)")
    
    # Initialize database
    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        db_path = watcher_root / db_path
    print(f"Database: {db_path}")
    db = create_database_manager(str(db_path))
    
    # Create video source
    print("Initializing video source...")
    source = create_source(resolved["source"])
    source_info = source.open()
    print_source_info(source_info)
    
    if not source_info.is_opened:
        print(f"ERROR: Failed to open source: {source_info.error_message}")
        sys.exit(1)
    
    # Log source
    source_id = f"src_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.insert_source(source_id, source_info.source_type.value, 
                     {"source_string": source_info.source_string})
    
    # Initialize pipeline components
    print("Initializing pipeline components...")
    tracker = create_tracker(resolved["detector"], resolved["seats"])
    pose_estimator = create_pose_estimator(resolved["pose"])
    feature_extractor = create_feature_extractor(resolved["rules"])
    
    fps = source_info.fps if source_info.fps > 0 else 25.0
    effective_fps = fps / args.stride if args.stride > 0 else fps
    fusion_manager = create_fusion_manager(resolved["rules"], effective_fps)
    rule_engine = create_rule_engine(resolved["rules"], effective_fps)
    risk_engine = create_risk_engine(resolved["rules"], effective_fps)
    alert_manager = AlertManager(risk_engine, db)
    
    # Prepare output video
    video_writer = None
    if not args.no_video:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = watcher_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use XVID+AVI — natively playable on Windows without extra codecs
        # Auto-switch extension to .avi for compatibility
        if output_path.suffix.lower() == '.mp4':
            output_path = output_path.with_suffix('.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, effective_fps,
            (source_info.width, source_info.height)
        )
        if not video_writer.isOpened():
            print("XVID codec unavailable, falling back to MJPG")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_writer = cv2.VideoWriter(
                str(output_path), fourcc, effective_fps,
                (source_info.width, source_info.height)
            )
        print(f"Output video: {output_path} (codec=XVID .avi)")
    
    # Prepare JSON log
    json_log_path = Path(args.output_json)
    if not json_log_path.is_absolute():
        json_log_path = watcher_root / json_log_path
    json_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare frame output directory
    frames_output_dir = watcher_root / "outputs" / "debug" / "pipeline_frames"
    frames_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare snapshots directory
    snapshots_dir = watcher_root / "outputs" / "debug" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of frames
    total_frames = source_info.frame_count if source_info.frame_count > 0 else float('inf')
    if args.frames > 0:
        total_frames = min(total_frames, args.frames)
    
    print(f"\nProcessing {total_frames if total_frames != float('inf') else 'all'} frames (stride={args.stride})...")
    print(f"Video: {source_info.width}x{source_info.height} @ {fps:.2f} FPS (effective: {effective_fps:.2f} FPS)")
    print(f"Estimated duration: {total_frames / fps:.1f}s" if total_frames != float('inf') else "Live stream")
    
    # Processing loop
    all_features_json = []
    all_events_json = []
    all_alerts_json = []
    frames_saved = 0
    snapshot_count = 0
    
    frame_idx = 0
    processed_count = 0
    try:
        while True:
            if args.frames > 0 and frame_idx >= args.frames:
                break
            
            ret, frame = source.read_frame()
            if not ret:
                print(f"End of video at frame {frame_idx}")
                break
            
            if args.stride > 1 and (frame_idx % args.stride != 0):
                frame_idx += 1
                continue
            
            timestamp = frame_idx / fps
            
            # 1. Tracking
            track_result = tracker.process_frame(frame, frame_idx=frame_idx)
            
            # 2. Pose estimation
            track_ids = [t.track_id for t in track_result.tracks]
            track_bboxes = [t.bbox for t in track_result.tracks]
            pose_result = pose_estimator.estimate(
                frame, frame_idx=frame_idx,
                track_ids=track_ids, track_bboxes=track_bboxes
            )
            
            # 3. Feature extraction
            behavioral_features = feature_extractor.extract(
                pose_result, track_result, frame_idx, timestamp, tracker.seat_map
            )
            
            # 4. Detection for objects
            detector_result = tracker.detector.detect(frame, frame_idx=frame_idx)
            
            # 5. Fusion & temporal windows
            fusion_results = fusion_manager.process_frame(
                frame_idx=frame_idx,
                timestamp=timestamp,
                track_result=track_result,
                pose_result=pose_result,
                behavioral_features_list=behavioral_features,
                detector_result=detector_result
            )
            
            # 6. Rule engine & risk scoring per track
            frame_events = []
            frame_alerts = []
            
            # Build track_id -> fusion mapping
            fusion_by_track = {fr.track_id: fr for fr in fusion_results}
            
            for track in track_result.tracks:
                seat_id = track.seat_id
                if not seat_id:
                    continue
                
                track_id = track.track_id
                
                # Get fusion result for this track
                fusion_result = fusion_by_track.get(track_id)
                
                # Get temporal features for this seat
                temporal_stats = fusion_manager.get_temporal_stats().get(seat_id, {})
                
                # Find the behavioral feature for this track
                feat = next((bf for bf in behavioral_features if bf.track_id == track_id), None)
                if not feat:
                    continue
                
                # Evaluate rules
                events = rule_engine.evaluate(
                    seat_id=seat_id,
                    track_id=track_id,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    behavioral_features=feat,
                    fusion_result=fusion_result
                )
                
                if events:
                    frame_events.extend(events)
                
                # Calculate pose confidence for this track
                pose_conf = 0.5
                for pose in pose_result.poses:
                    if pose.track_id == track_id:
                        pose_conf = pose.pose_confidence
                        break
                
                # Process alerts
                alerts = alert_manager.process_frame(
                    seat_id=seat_id,
                    track_id=track_id,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    events=events,
                    fusion_result=fusion_result,
                    pose_confidence=pose_conf,
                    evidence_frame=frame if events else None
                )
                
                if alerts:
                    frame_alerts.extend(alerts)
            
            # 7. Convert to JSON for logging
            if behavioral_features:
                features_json = feature_extractor.features_to_json(behavioral_features)
                for fj in features_json:
                    fj["frame_idx"] = frame_idx
                    fj["timestamp"] = timestamp
                all_features_json.extend(features_json)
            
            if frame_events:
                events_json = [rule_engine.event_to_json(e) for e in frame_events]
                for ej in events_json:
                    ej["frame_idx"] = frame_idx
                    ej["timestamp"] = timestamp
                all_events_json.extend(events_json)
            
            if frame_alerts:
                for alert in frame_alerts:
                    alert_dict = {
                        "alert_id": alert.alert_id,
                        "seat_id": alert.seat_id,
                        "timestamp": alert.timestamp,
                        "risk_score": alert.risk_score,
                        "severity": alert.severity,
                        "behaviors": alert.behaviors,
                        "explanation": alert.explanation,
                        "confidence": alert.confidence
                    }
                    all_alerts_json.append(alert_dict)
            
            # 8. Draw annotations
            annotated = tracker.draw_result(frame, track_result, draw_seats=True)
            annotated = pose_estimator.draw_poses(annotated, pose_result)
            annotated = feature_extractor.draw_features(annotated, behavioral_features)
            annotated = fusion_manager.draw_fusion(annotated, fusion_results)
            
            # Draw risk scores and alerts
            for track in track_result.tracks:
                seat_id = track.seat_id
                if seat_id and seat_id in risk_engine.risk_scores:
                    risk = risk_engine.risk_scores[seat_id]
                    x1, y1, x2, y2 = track.bbox
                    color = (0, 255, 0) if risk < 25 else (0, 255, 255) if risk < 50 else (0, 165, 255) if risk < 75 else (0, 0, 255)
                    cv2.putText(annotated, f"Risk: {risk:.0f}", (x1, y1 - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw active alerts
            for alert in frame_alerts:
                cv2.putText(annotated, f"ALERT: {alert.severity} - {alert.explanation[:80]}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw frame info
            cv2.putText(annotated, f"Frame: {frame_idx} | Time: {timestamp:.1f}s | FPS: {fps:.1f}",
                       (10, source_info.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write output video
            if video_writer is not None:
                video_writer.write(annotated)
            
            # Show live preview window
            if show_preview:
                cv2.imshow("Watcher Live Preview", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC to quit
                    print("\nLive preview closed by user")
                    break
            
            # Save sample frames
            if frames_saved < args.save_frames:
                frame_path = frames_output_dir / f"pipeline_frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated)
                frames_saved += 1
            
            # Save snapshots for alerts
            if frame_alerts and snapshot_count < 20:
                for alert in frame_alerts:
                    snap_path = snapshots_dir / f"alert_{alert.seat_id}_{frame_idx:06d}_{alert.severity}.jpg"
                    cv2.imwrite(str(snap_path), annotated)
                    snapshot_count += 1
            
            # Progress
            frame_idx += 1
            if frame_idx % 50 == 0:
                elapsed = frame_idx / fps
                print(f"  Frame {frame_idx} ({elapsed:.1f}s) - "
                      f"Tracks: {len(track_result.tracks)} | "
                      f"Events: {len(frame_events)} | "
                      f"Alerts: {len(frame_alerts)} | "
                      f"Risk scores: {dict(risk_engine.risk_scores)}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        # Cleanup
        if show_preview:
            cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        source.close()
        
        # Update source end
        db.update_source_end(source_id, frame_idx)
        
        # Save JSON log
        log_data = {
            "source": source_info.source_string,
            "total_frames": frame_idx,
            "duration_sec": frame_idx / fps if fps > 0 else 0,
            "features": all_features_json,
            "events": all_events_json,
            "alerts": all_alerts_json,
            "risk_scores": risk_engine.risk_scores,
            "temporal_stats": fusion_manager.get_temporal_stats()
        }
        
        def _json_serializable(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.ndarray, list)):
                return list(obj)
            return str(obj)
            
        with open(json_log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=_json_serializable)
        
        # Export database to JSON/CSV
        db.export_to_json(str(watcher_root / "outputs" / "logs" / "watcher_full_export.json"))
        db.export_to_csv(str(watcher_root / "outputs" / "logs" / "csv_export"))
        
        db.close()
        
        # Sync outputs to root project outputs if applicable
        root_dir = project_root.parent
        if root_dir != watcher_root and (root_dir / "WATCHER_SPEC.md").exists():
            import shutil
            for folder in ["overlays", "logs", "debug"]:
                src_folder = watcher_root / "outputs" / folder
                dst_folder = root_dir / "outputs" / folder
                if src_folder.exists():
                    dst_folder.mkdir(parents=True, exist_ok=True)
                    for item in src_folder.glob("*.*"):
                        if item.is_file():
                            shutil.copy2(item, dst_folder / item.name)
        
        # Summary
        print("\n" + "=" * 60)
        print("FULL PIPELINE PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Frames Processed:       {frame_idx}")
        print(f"Duration:               {frame_idx / fps:.1f}s")
        print(f"Total Features:         {len(all_features_json)}")
        print(f"Total Events:           {len(all_events_json)}")
        print(f"Total Alerts:           {len(all_alerts_json)}")
        print(f"Final Risk Scores:      {risk_engine.risk_scores}")
        print(f"Output Video:           {args.output}" if not args.no_video else "Output Video: SKIPPED")
        print(f"JSON Log:               {args.output_json}")
        print(f"Database:               {db_path}")
        print(f"Sample Frames:          {frames_saved}")
        print(f"Alert Snapshots:        {snapshot_count}")
        print("=" * 60)
        
        # Print alerts summary
        if all_alerts_json:
            print("\nALERTS GENERATED:")
            for alert in all_alerts_json:
                print(f"  [{alert['severity']}] Seat {alert['seat_id']}: {alert['explanation'][:100]}")
        else:
            print("\nNO ALERTS GENERATED - Check thresholds or video content")


if __name__ == "__main__":
    main()