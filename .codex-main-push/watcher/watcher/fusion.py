"""
Object-Hand Fusion and Temporal Feature Window for Watcher.

Combines object detections with hand context and maintains rolling behavioral features
per seat/track across multiple time windows.
"""

import cv2
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

logger = logging.getLogger(__name__)


class ObjectClass(Enum):
    """Suspicious object classes."""
    PHONE = "cell phone"
    LAPTOP = "laptop"
    BOOK = "book"  # Placeholder for chit/note
    CHAIR = "chair"  # For testing


@dataclass
class ObjectDetection:
    """Object detection with metadata."""
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]


@dataclass
class FusionResult:
    """Object-hand fusion result for a track."""
    track_id: int
    seat_id: Optional[str]
    frame_idx: int
    timestamp: float
    # Object info
    object_detected: bool
    object_class: Optional[str]
    object_confidence: float
    object_bbox: Optional[Tuple[int, int, int, int]]
    # Hand context
    left_hand_state: str
    right_hand_state: str
    left_wrist_pos: Tuple[float, float]
    right_wrist_pos: Tuple[float, float]
    left_below_desk: bool
    right_below_desk: bool
    # Fusion
    object_near_left_hand: bool
    object_near_right_hand: bool
    object_hand_distance: float  # Min distance to either hand
    fusion_score: float  # Combined suspicion score


@dataclass
class TemporalFeatureWindow:
    """Rolling temporal feature window for a seat/track."""
    seat_id: str
    track_id: int
    # Window sizes in frames (at 25 FPS: 5s=125, 30s=750, 90s=2250)
    window_5s: deque = field(default_factory=lambda: deque(maxlen=125))
    window_30s: deque = field(default_factory=lambda: deque(maxlen=750))
    window_90s: deque = field(default_factory=lambda: deque(maxlen=2250))
    
    def add_feature(self, feature: Dict):
        """Add feature to all windows."""
        self.window_5s.append(feature)
        self.window_30s.append(feature)
        self.window_90s.append(feature)
    
    def get_window_stats(self, window_name: str) -> Dict[str, Any]:
        """Get statistics for a specific window."""
        window = getattr(self, f"window_{window_name}")
        if not window:
            return {}
        
        features = list(window)
        
        # Head direction counts
        head_dirs = [f.get("head_direction", "UNKNOWN") for f in features]
        head_counts = {d: head_dirs.count(d) for d in set(head_dirs)}
        
        # Hand states
        left_states = [f.get("left_hand_state", "NORMAL") for f in features]
        right_states = [f.get("right_hand_state", "NORMAL") for f in features]
        
        # Object interaction
        obj_interactions = [f.get("fusion_score", 0) for f in features]
        
        # Torso angles
        torso_angles = [f.get("torso_angle", 0) for f in features if f.get("torso_angle") is not None]
        
        return {
            "window": window_name,
            "num_frames": len(features),
            "head_direction_counts": head_counts,
            "left_hand_state_counts": {s: left_states.count(s) for s in set(left_states)},
            "right_hand_state_counts": {s: right_states.count(s) for s in set(right_states)},
            "avg_fusion_score": np.mean(obj_interactions) if obj_interactions else 0,
            "max_fusion_score": np.max(obj_interactions) if obj_interactions else 0,
            "avg_torso_angle": np.mean(torso_angles) if torso_angles else 0,
            "max_torso_angle": np.max(torso_angles) if torso_angles else 0,
            "below_desk_frames": sum(1 for f in features if f.get("left_below_desk") or f.get("right_below_desk")),
            "near_face_frames": sum(1 for f in features if f.get("left_near_face") or f.get("right_near_face")),
        }
    
    def to_json(self) -> Dict[str, Any]:
        """Export all window stats as JSON."""
        return {
            "seat_id": self.seat_id,
            "track_id": self.track_id,
            "window_5s": self.get_window_stats("5s"),
            "window_30s": self.get_window_stats("30s"),
            "window_90s": self.get_window_stats("90s"),
        }


class ObjectHandFusion:
    """
    Fuses object detections with hand/wrist positions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.distance_threshold = self.config.get("distance_threshold_px", 120)
        self.suspicious_classes = self.config.get("suspicious_classes", 
            ["cell phone", "book", "laptop", "remote", "mouse"])  # book = chit/note proxy
        self.min_persistence_frames = self.config.get("min_persistence_frames", 1)
        
        # Track object persistence per track
        self.object_persistence: Dict[int, Dict[str, int]] = {}
    
    def fuse(self, 
             objects: List[ObjectDetection],
             hand_features,
             track_id: int,
             seat_id: Optional[str],
             frame_idx: int,
             timestamp: float,
             track_bbox: Optional[Tuple[int, int, int, int]] = None) -> FusionResult:
        """
        Fuse object detections with hand context and person location.
        
        Args:
            objects: List of ObjectDetection from detector
            hand_features: HandFeatures from feature extractor
            track_id: Track ID
            seat_id: Seat ID
            frame_idx: Current frame index
            timestamp: Current timestamp
            track_bbox: Optional person bounding box (x1, y1, x2, y2)
            
        Returns:
            FusionResult
        """
        # Default result
        result = FusionResult(
            track_id=track_id,
            seat_id=seat_id,
            frame_idx=frame_idx,
            timestamp=timestamp,
            object_detected=False,
            object_class=None,
            object_confidence=0.0,
            object_bbox=None,
            left_hand_state=hand_features.left_hand_state.value,
            right_hand_state=hand_features.right_hand_state.value,
            left_wrist_pos=hand_features.left_wrist_pos,
            right_wrist_pos=hand_features.right_wrist_pos,
            left_below_desk=hand_features.left_below_desk,
            right_below_desk=hand_features.right_below_desk,
            object_near_left_hand=False,
            object_near_right_hand=False,
            object_hand_distance=float('inf'),
            fusion_score=0.0
        )
        
        if not objects:
            return result
        
        # Filter to suspicious classes
        suspicious_objects = [obj for obj in objects if obj.class_name in self.suspicious_classes]
        
        if not suspicious_objects:
            return result
        
        min_dist = float('inf')
        best_obj = None
        near_left = False
        near_right = False
        inside_person_bbox = False
        
        left_wrist = np.array(hand_features.left_wrist_pos) if hand_features.left_wrist_pos != (0, 0) else None
        right_wrist = np.array(hand_features.right_wrist_pos) if hand_features.right_wrist_pos != (0, 0) else None
        
        for obj in suspicious_objects:
            obj_center = np.array(obj.center)
            
            # Check if object is inside or near person bounding box
            if track_bbox is not None:
                px1, py1, px2, py2 = track_bbox
                # Expand box slightly for lap/desk perimeter
                if (px1 - 40 <= obj_center[0] <= px2 + 40) and (py1 - 20 <= obj_center[1] <= py2 + 40):
                    inside_person_bbox = True
            
            # Calculate distances to wrists
            left_dist = np.linalg.norm(obj_center - left_wrist) if left_wrist is not None else float('inf')
            right_dist = np.linalg.norm(obj_center - right_wrist) if right_wrist is not None else float('inf')
            
            dist = min(left_dist, right_dist)
            
            if dist < min_dist or (inside_person_bbox and best_obj is None):
                min_dist = dist
                best_obj = obj
                near_left = left_dist <= self.distance_threshold
                near_right = right_dist <= self.distance_threshold
        
        # If object is inside person bbox or near either wrist, accept it
        if best_obj is None or (not inside_person_bbox and min_dist > self.distance_threshold * 1.5):
            return result
        
        # Update persistence
        if track_id not in self.object_persistence:
            self.object_persistence[track_id] = {}
        
        obj_key = best_obj.class_name
        if obj_key not in self.object_persistence[track_id]:
            self.object_persistence[track_id][obj_key] = 0
        
        self.object_persistence[track_id][obj_key] += 1
        
        # Calculate fusion score
        # Base score from object confidence
        fusion_score = best_obj.confidence * 0.6
        
        # Boost if near hand
        if near_left or near_right:
            fusion_score += 0.3
        
        # Boost if inside person bbox/lap
        if inside_person_bbox:
            fusion_score += 0.25
        
        # Boost if hand is in suspicious state
        suspicious_hand_states = {"BELOW_DESK", "BELOW_SHOULDER", "NEAR_FACE", "NEAR_EAR", "NEAR_LAP", "FAST_MOVEMENT"}
        if hand_features.left_hand_state.value in suspicious_hand_states or hand_features.right_hand_state.value in suspicious_hand_states:
            fusion_score += 0.15
        
        # Boost if hand below desk
        if hand_features.left_below_desk or hand_features.right_below_desk:
            fusion_score += 0.2
        
        fusion_score = min(fusion_score, 1.0)
        
        result.object_detected = True
        result.object_class = best_obj.class_name
        result.object_confidence = best_obj.confidence
        result.object_bbox = best_obj.bbox
        result.object_near_left_hand = near_left
        result.object_near_right_hand = near_right
        result.object_hand_distance = min_dist if min_dist != float('inf') else 0.0
        result.fusion_score = fusion_score
        
        return result
    
    def reset_track(self, track_id: int):
        """Reset persistence for a track."""
        if track_id in self.object_persistence:
            del self.object_persistence[track_id]


class TemporalFeatureManager:
    """
    Manages temporal feature windows per seat/track.
    """
    
    def __init__(self, config: Dict[str, Any] = None, fps: float = 25.0):
        self.config = config or {}
        self.fps = fps
        self.windows: Dict[str, TemporalFeatureWindow] = {}  # seat_id -> window
        self.fusion = ObjectHandFusion(self.config.get("fusion", {}))
        
        # Window sizes in frames
        self.window_5s_frames = int(self.config.get("window_5s_seconds", 5) * fps)
        self.window_30s_frames = int(self.config.get("window_30s_seconds", 30) * fps)
        self.window_90s_frames = int(self.config.get("window_90s_seconds", 90) * fps)
    
    def update(self, 
               seat_id: str,
               track_id: int,
               frame_idx: int,
               timestamp: float,
               behavioral_features,
               objects: List[ObjectDetection],
               track_bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[FusionResult]:
        """
        Update temporal windows with new features.
        
        Args:
            seat_id: Seat ID
            track_id: Track ID
            frame_idx: Current frame index
            timestamp: Current timestamp
            behavioral_features: BehavioralFeatures from feature extractor
            objects: List of ObjectDetection from detector
            track_bbox: Optional person bounding box
            
        Returns:
            FusionResult if fusion occurred, None otherwise
        """
        # Create window if not exists
        if seat_id not in self.windows:
            self.windows[seat_id] = TemporalFeatureWindow(seat_id=seat_id, track_id=track_id)
        
        window = self.windows[seat_id]
        
        # Run fusion
        fusion_result = self.fusion.fuse(
            objects, 
            behavioral_features.hands,
            track_id,
            seat_id,
            frame_idx,
            timestamp,
            track_bbox=track_bbox
        )
        
        # Build feature dict for temporal window
        feature_dict = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "track_id": track_id,
            "head_direction": behavioral_features.head.direction.value,
            "head_yaw": behavioral_features.head.yaw_angle,
            "head_pitch": behavioral_features.head.pitch_angle,
            "left_hand_state": behavioral_features.hands.left_hand_state.value,
            "right_hand_state": behavioral_features.hands.right_hand_state.value,
            "left_wrist_pos": behavioral_features.hands.left_wrist_pos,
            "right_wrist_pos": behavioral_features.hands.right_wrist_pos,
            "left_below_desk": behavioral_features.hands.left_below_desk,
            "right_below_desk": behavioral_features.hands.right_below_desk,
            "left_near_face": behavioral_features.hands.left_near_face,
            "right_near_face": behavioral_features.hands.right_near_face,
            "torso_angle": behavioral_features.torso.angle,
            "torso_toward_neighbor": behavioral_features.torso.angle_toward_neighbor,
            "fusion_score": fusion_result.fusion_score,
            "object_detected": fusion_result.object_detected,
            "object_class": fusion_result.object_class,
            "object_hand_distance": fusion_result.object_hand_distance,
        }
        
        # Add to window
        window.add_feature(feature_dict)
        
        return fusion_result
    
    def get_window_stats(self, seat_id: str) -> Dict[str, Any]:
        """Get stats for all windows for a seat."""
        if seat_id not in self.windows:
            return {}
        return self.windows[seat_id].to_json()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get stats for all seats."""
        return {seat_id: window.to_json() for seat_id, window in self.windows.items()}
    
    def reset_seat(self, seat_id: str):
        """Reset window for a seat."""
        if seat_id in self.windows:
            self.fusion.reset_track(self.windows[seat_id].track_id)
            del self.windows[seat_id]


class FusionManager:
    """
    Main fusion and temporal feature manager.
    """
    
    def __init__(self, config_path: str = "configs/rules.yaml", fps: float = 25.0):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.temporal_manager = TemporalFeatureManager(self.config, fps)
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return {}
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def process_frame(self,
                      frame_idx: int,
                      timestamp: float,
                      track_result,
                      pose_result,
                      behavioral_features_list,
                      detector_result) -> List[FusionResult]:
        """
        Process a single frame through fusion and temporal windows.
        
        Args:
            frame_idx: Current frame index
            timestamp: Current timestamp
            track_result: TrackingResult
            pose_result: PoseResult
            behavioral_features_list: List of BehavioralFeatures
            detector_result: DetectionResult with objects
            
        Returns:
            List of FusionResult
        """
        # Extract object detections
        objects = []
        suspicious_classes = {"cell phone", "book", "laptop", "remote", "mouse"}
        for det in detector_result.detections:
            if det.class_name in suspicious_classes:
                objects.append(ObjectDetection(
                    class_name=det.class_name,
                    class_id=det.class_id,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    center=det.center
                ))
        
        # Build track_id -> behavioral_features mapping
        feat_by_track = {f.track_id: f for f in behavioral_features_list}
        
        # Build track_id -> track mapping
        track_by_id = {track.track_id: track for track in track_result.tracks}
        
        fusion_results = []
        
        for track in track_result.tracks:
            track_id = track.track_id
            seat_id = track.seat_id
            
            if seat_id is None or track_id not in feat_by_track:
                continue
            
            behavioral_features = feat_by_track[track_id]
            
            # Update temporal window and get fusion result
            fusion_result = self.temporal_manager.update(
                seat_id=seat_id,
                track_id=track_id,
                frame_idx=frame_idx,
                timestamp=timestamp,
                behavioral_features=behavioral_features,
                objects=objects,
                track_bbox=track.bbox
            )
            
            if fusion_result:
                fusion_results.append(fusion_result)
        
        return fusion_results
    
    def draw_fusion(self, frame: np.ndarray, fusion_results: List[FusionResult]) -> np.ndarray:
        """Draw fusion results on frame."""
        annotated = frame.copy()
        
        for fr in fusion_results:
            if fr.object_detected:
                # Draw object bbox
                if fr.object_bbox:
                    x1, y1, x2, y2 = fr.object_bbox
                    color = (0, 0, 255) if fr.fusion_score > 0.5 else (0, 165, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{fr.object_class} {fr.object_confidence:.2f} Fusion:{fr.fusion_score:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw hand-object connection lines
                if fr.object_near_left_hand and fr.left_wrist_pos != (0, 0):
                    wrist = (int(fr.left_wrist_pos[0]), int(fr.left_wrist_pos[1]))
                    obj_center = ((fr.object_bbox[0] + fr.object_bbox[2]) // 2,
                                  (fr.object_bbox[1] + fr.object_bbox[3]) // 2)
                    cv2.line(annotated, wrist, obj_center, (255, 0, 255), 2)
                
                if fr.object_near_right_hand and fr.right_wrist_pos != (0, 0):
                    wrist = (int(fr.right_wrist_pos[0]), int(fr.right_wrist_pos[1]))
                    obj_center = ((fr.object_bbox[0] + fr.object_bbox[2]) // 2,
                                  (fr.object_bbox[1] + fr.object_bbox[3]) // 2)
                    cv2.line(annotated, wrist, obj_center, (255, 0, 255), 2)
        
        return annotated
    
    def get_temporal_stats(self) -> Dict[str, Any]:
        """Get temporal stats for all seats."""
        return self.temporal_manager.get_all_stats()
    
    def fusion_results_to_json(self, fusion_results: List[FusionResult]) -> List[Dict]:
        """Convert fusion results to JSON."""
        result = []
        for fr in fusion_results:
            result.append({
                "track_id": fr.track_id,
                "seat_id": fr.seat_id,
                "frame_idx": fr.frame_idx,
                "timestamp": fr.timestamp,
                "object_detected": fr.object_detected,
                "object_class": fr.object_class,
                "object_confidence": fr.object_confidence,
                "object_bbox": fr.object_bbox,
                "left_hand_state": fr.left_hand_state,
                "right_hand_state": fr.right_hand_state,
                "left_below_desk": fr.left_below_desk,
                "right_below_desk": fr.right_below_desk,
                "object_near_left_hand": fr.object_near_left_hand,
                "object_near_right_hand": fr.object_near_right_hand,
                "object_hand_distance": fr.object_hand_distance,
                "fusion_score": fr.fusion_score
            })
        return result


def create_fusion_manager(config_path: str = "configs/rules.yaml", fps: float = 25.0) -> FusionManager:
    """Factory function to create FusionManager from config."""
    return FusionManager(config_path, fps)