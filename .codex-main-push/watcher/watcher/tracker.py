"""
Tracking and SeatTrack Binding for Watcher.

Uses ByteTrack (via Ultralytics) for multi-object tracking and binds tracks to seat regions.
"""

import cv2
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from ultralytics import YOLO
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Single track state."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    seat_id: Optional[str] = None
    seat_confidence: float = 0.0


@dataclass
class Seat:
    """Seat definition with polygon and properties."""
    seat_id: str
    row: str
    column: int
    polygon: List[List[int]]  # [[x1,y1], [x2,y2], ...]
    desk_line: List[List[int]]  # [[x1,y1], [x2,y2]]
    neighbours: List[str]
    polygon_np: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.polygon_np = np.array(self.polygon, dtype=np.int32)


@dataclass
class TrackingResult:
    """All tracks for a single frame."""
    frame_idx: int
    tracks: List[Track]
    inference_time_ms: float


class SeatMap:
    """Manages seat polygons and track-to-seat assignment."""
    
    def __init__(self, config_path: str = "configs/seats.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.seats: Dict[str, Seat] = {}
        self._load_seats()
    
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_seats(self):
        """Load seat definitions from config."""
        seats_config = self.config.get("seats", [])
        for seat_data in seats_config:
            seat = Seat(
                seat_id=seat_data["id"],
                row=seat_data["row"],
                column=seat_data["column"],
                polygon=seat_data["polygon"],
                desk_line=seat_data["desk_line"],
                neighbours=seat_data.get("neighbours", [])
            )
            self.seats[seat.seat_id] = seat
        logger.info(f"Loaded {len(self.seats)} seats")
    
    def point_in_polygon(self, point: Tuple[int, int], seat_id: str) -> bool:
        """Check if a point is inside a seat polygon."""
        seat = self.seats.get(seat_id)
        if not seat:
            return False
        return cv2.pointPolygonTest(seat.polygon_np, point, False) >= 0
    
    def get_seat_for_point(self, point: Tuple[int, int]) -> Optional[str]:
        """Find which seat contains the given point."""
        for seat_id, seat in self.seats.items():
            if cv2.pointPolygonTest(seat.polygon_np, point, False) >= 0:
                return seat_id
        return None
    
    def get_seat_for_bbox(self, bbox: Tuple[int, int, int, int], method: str = "bottom_center") -> Tuple[Optional[str], float]:
        """
        Assign a bounding box to a seat.
        
        Args:
            bbox: (x1, y1, x2, y2)
            method: "centroid", "bottom_center", or "bbox_overlap"
            
        Returns:
            (seat_id, confidence)
        """
        x1, y1, x2, y2 = bbox
        
        if method == "centroid":
            point = ((x1 + x2) // 2, (y1 + y2) // 2)
            seat_id = self.get_seat_for_point(point)
            return seat_id, 1.0 if seat_id else 0.0
        
        elif method == "bottom_center":
            # Use bottom center of bbox (feet position for seated person)
            point = ((x1 + x2) // 2, y2)
            seat_id = self.get_seat_for_point(point)
            return seat_id, 1.0 if seat_id else 0.0
        
        elif method == "bbox_overlap":
            # Calculate overlap with each seat polygon
            bbox_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            best_seat = None
            best_overlap = 0.0
            
            for seat_id, seat in self.seats.items():
                # Create mask for seat
                seat_mask = np.zeros((720, 1280), dtype=np.uint8)  # Assume 720p
                cv2.fillPoly(seat_mask, [seat.polygon_np], 255)
                
                # Create mask for bbox
                bbox_mask = np.zeros((720, 1280), dtype=np.uint8)
                cv2.fillPoly(bbox_mask, [bbox_poly], 255)
                
                # Calculate overlap
                intersection = cv2.bitwise_and(seat_mask, bbox_mask)
                overlap_area = cv2.countNonZero(intersection)
                bbox_area = cv2.countNonZero(bbox_mask)
                
                if bbox_area > 0:
                    overlap_ratio = overlap_area / bbox_area
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_seat = seat_id
            
            min_overlap = self.config.get("mapping", {}).get("min_overlap_ratio", 0.3)
            if best_overlap >= min_overlap:
                return best_seat, best_overlap
            return None, best_overlap
        
        return None, 0.0
    
    def get_desk_y(self, seat_id: str) -> Optional[int]:
        """Get desk line Y coordinate for a seat."""
        seat = self.seats.get(seat_id)
        if seat and seat.desk_line:
            # Return average Y of desk line
            return int(np.mean([p[1] for p in seat.desk_line]))
        return None
    
    def draw_seats(self, frame: np.ndarray, show_ids: bool = True) -> np.ndarray:
        """Draw seat polygons on frame."""
        annotated = frame.copy()
        overlay = frame.copy()
        
        for seat_id, seat in self.seats.items():
            # Draw polygon
            cv2.polylines(overlay, [seat.polygon_np], True, (0, 255, 255), 2)
            cv2.fillPoly(overlay, [seat.polygon_np], (0, 255, 255))
            
            # Draw desk line
            if seat.desk_line:
                pt1 = tuple(seat.desk_line[0])
                pt2 = tuple(seat.desk_line[1])
                cv2.line(overlay, pt1, pt2, (255, 255, 0), 2)
            
            # Draw seat ID
            if show_ids:
                centroid = np.mean(seat.polygon, axis=0).astype(int)
                cv2.putText(overlay, seat_id, tuple(centroid), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
        return annotated


class ByteTracker:
    """
    ByteTrack wrapper using Ultralytics' built-in tracking.
    """
    
    def __init__(self, detector: 'Detector', seat_map: SeatMap, config_path: str = "configs/seats.yaml"):
        self.detector = detector
        self.seat_map = seat_map
        self.config = self._load_config(config_path)
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        # Tracking parameters
        track_config = self.config.get("mapping", {})
        self.method = track_config.get("method", "bottom_center")
        self.max_age = track_config.get("max_age", 30)  # Max frames to keep lost track
        self.min_hits = track_config.get("min_hits", 3)  # Min hits before track is confirmed
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(Path(config_path), 'r') as f:
            return yaml.safe_load(f)
    
    def update(self, frame: np.ndarray, frame_idx: int) -> TrackingResult:
        """
        Update tracks with new frame.
        
        Args:
            frame: Input frame.
            frame_idx: Current frame index.
            
        Returns:
            TrackingResult with updated tracks.
        """
        import time
        start_time = time.time()
        
        # Run detection
        det_result = self.detector.detect(frame, frame_idx)
        
        # Update existing tracks and create new ones
        # Using Ultralytics tracking would be better, but for MVP we'll do simple IoU matching
        detections = det_result.detections
        
        # Simple tracking: match detections to existing tracks by IoU
        matched_tracks = self._match_detections_to_tracks(detections)
        
        # Update seat assignments
        for track in self.tracks.values():
            if track.time_since_update == 0:  # Only update seat for active tracks
                seat_id, confidence = self.seat_map.get_seat_for_bbox(track.bbox, self.method)
                track.seat_id = seat_id
                track.seat_confidence = confidence
        
        # Increment ages
        for track in self.tracks.values():
            track.age += 1
            if track.time_since_update > 0:
                track.time_since_update += 1
        
        # Remove old lost tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() 
                      if t.time_since_update <= self.max_age}
        
        # Prepare active tracks for output
        active_tracks = [t for t in self.tracks.values() if t.time_since_update == 0 and t.hits >= self.min_hits]
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return TrackingResult(
            frame_idx=frame_idx,
            tracks=active_tracks,
            inference_time_ms=inference_time_ms
        )
    
    def _match_detections_to_tracks(self, detections: List) -> Dict[int, int]:
        """Match detections to existing tracks using IoU."""
        # Simple greedy IoU matching
        matches = {}
        used_detections = set()
        
        for track_id, track in self.tracks.items():
            best_iou = 0.3  # Minimum IoU threshold
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                iou = self._calculate_iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx >= 0:
                matches[track_id] = best_det_idx
                used_detections.add(best_det_idx)
                
                # Update track with new detection
                det = detections[best_det_idx]
                track.bbox = det.bbox
                track.class_id = det.class_id
                track.class_name = det.class_name
                track.confidence = det.confidence
                track.hits += 1
                track.time_since_update = 0
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                # Create tracks for person class (class_id 0) and chair (class_id 56) for testing
                if det.class_id in [0, 56]:  # person, chair
                    new_track = Track(
                        track_id=self.next_track_id,
                        bbox=det.bbox,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        hits=1,
                        time_since_update=0
                    )
                    self.tracks[self.next_track_id] = new_track
                    self.next_track_id += 1
        
        return matches
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def draw_tracks(self, frame: np.ndarray, result: TrackingResult) -> np.ndarray:
        """Draw tracks with IDs and seat assignments."""
        annotated = frame.copy()
        
        for track in result.tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Color based on seat assignment
            if track.seat_id:
                color = (0, 255, 0)  # Green for seated
                label = f"ID:{track.track_id} Seat:{track.seat_id}"
            else:
                color = (0, 165, 255)  # Orange for unassigned
                label = f"ID:{track.track_id} UNASSIGNED"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw bottom center point (used for seat assignment)
            bottom_center = ((x1 + x2) // 2, y2)
            cv2.circle(annotated, bottom_center, 4, color, -1)
        
        return annotated


class Tracker:
    """
    Main tracker interface combining detection, ByteTrack, and SeatMap.
    """
    
    def __init__(self, detector_config: str = "configs/detector.yaml", 
                 seats_config: str = "configs/seats.yaml"):
        from watcher.detector import create_detector
        self.detector = create_detector(detector_config)
        self.seat_map = SeatMap(seats_config)
        self.byte_tracker = ByteTracker(self.detector, self.seat_map, seats_config)
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> TrackingResult:
        """Process a single frame through the full tracking pipeline."""
        return self.byte_tracker.update(frame, frame_idx)
    
    def draw_result(self, frame: np.ndarray, result: TrackingResult, 
                    draw_seats: bool = True) -> np.ndarray:
        """Draw seats and tracks on frame."""
        annotated = frame.copy()
        
        if draw_seats:
            annotated = self.seat_map.draw_seats(annotated)
        
        annotated = self.byte_tracker.draw_tracks(annotated, result)
        return annotated
    
    def get_summary(self, result: TrackingResult) -> Dict[str, Any]:
        """Get tracking summary statistics."""
        seated = sum(1 for t in result.tracks if t.seat_id)
        unassigned = len(result.tracks) - seated
        
        seat_assignments = {}
        for t in result.tracks:
            if t.seat_id:
                seat_assignments[t.seat_id] = seat_assignments.get(t.seat_id, 0) + 1
        
        return {
            "frame_idx": result.frame_idx,
            "total_tracks": len(result.tracks),
            "seated_tracks": seated,
            "unassigned_tracks": unassigned,
            "seat_assignments": seat_assignments,
            "inference_time_ms": result.inference_time_ms
        }


def create_tracker(detector_config: str = "configs/detector.yaml",
                   seats_config: str = "configs/seats.yaml") -> Tracker:
    """Factory function to create Tracker from configs."""
    return Tracker(detector_config, seats_config)