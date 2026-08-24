"""
Head Orientation and Hand/Wrist Feature Extraction for Watcher.

Extracts behavioral features from pose keypoints:
- Coarse head direction (LEFT, RIGHT, FRONT, DOWN, UP)
- Torso angle
- Hand/wrist position relative to shoulders, torso, desk
- Hand states (below desk, near face, near lap, fast movement)
"""

import cv2
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class HeadDirection(Enum):
    """Coarse head direction states."""
    FRONT = "FRONT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    UP = "UP"
    UNKNOWN = "UNKNOWN"


class HandState(Enum):
    """Hand position states."""
    NORMAL = "NORMAL"
    BELOW_DESK = "BELOW_DESK"
    BELOW_SHOULDER = "BELOW_SHOULDER"
    NEAR_FACE = "NEAR_FACE"
    NEAR_EAR = "NEAR_EAR"
    NEAR_LAP = "NEAR_LAP"
    FAST_MOVEMENT = "FAST_MOVEMENT"
    UNKNOWN = "UNKNOWN"


@dataclass
class HeadFeatures:
    """Head orientation features for a single track."""
    track_id: int
    direction: HeadDirection
    yaw_angle: float  # Estimated yaw in degrees
    pitch_angle: float  # Estimated pitch in degrees
    confidence: float
    nose_offset_ratio: float  # Nose offset from shoulder center / shoulder width
    ear_visibility_ratio: float  # Visible ears / total ears


@dataclass
class HandFeatures:
    """Hand/wrist features for a single track."""
    track_id: int
    left_hand_state: HandState
    right_hand_state: HandState
    left_wrist_pos: Tuple[float, float]  # (x, y)
    right_wrist_pos: Tuple[float, float]
    left_wrist_vel: Tuple[float, float]  # Velocity (dx, dy)
    right_wrist_vel: Tuple[float, float]
    left_below_desk: bool
    right_below_desk: bool
    left_near_face: bool
    right_near_face: bool
    confidence: float


@dataclass
class TorsoFeatures:
    """Torso orientation features."""
    track_id: int
    angle: float  # Torso angle relative to vertical (degrees)
    angle_toward_neighbor: Optional[float]  # Angle toward neighbor seat
    confidence: float


@dataclass
class BehavioralFeatures:
    """Combined behavioral features for a track."""
    track_id: int
    seat_id: Optional[str]
    frame_idx: int
    timestamp: float
    head: HeadFeatures
    hands: HandFeatures
    torso: TorsoFeatures


class HeadOrientationEstimator:
    """
    Estimates coarse head direction from pose keypoints.
    
    Uses geometric relationships:
    - Nose offset from shoulder center
    - Ear visibility ratio
    - Eye-nose displacement
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # Thresholds for head direction classification
        self.yaw_threshold = self.config.get("yaw_threshold_deg", 20.0)
        self.pitch_threshold = self.config.get("pitch_threshold_deg", 15.0)
        
        # Keypoint indices (COCO format)
        self.kp_indices = {
            "nose": 0,
            "left_eye": 1,
            "right_eye": 2,
            "left_ear": 3,
            "right_ear": 4,
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
            "left_hip": 11,
            "right_hip": 12,
        }
    
    def estimate(self, keypoints: List, track_id: int) -> HeadFeatures:
        """
        Estimate head direction from keypoints.
        
        Args:
            keypoints: List of 17 Keypoint objects with x, y, confidence
            track_id: Track ID for logging
            
        Returns:
            HeadFeatures with direction and angles
        """
        # Extract required keypoints
        nose = self._get_kp(keypoints, "nose")
        left_eye = self._get_kp(keypoints, "left_eye")
        right_eye = self._get_kp(keypoints, "right_eye")
        left_ear = self._get_kp(keypoints, "left_ear")
        right_ear = self._get_kp(keypoints, "right_ear")
        left_shoulder = self._get_kp(keypoints, "left_shoulder")
        right_shoulder = self._get_kp(keypoints, "right_shoulder")
        
        # Check if we have enough keypoints
        required = [nose, left_shoulder, right_shoulder]
        if any(kp is None or kp.confidence < 0.3 for kp in required):
            return HeadFeatures(
                track_id=track_id,
                direction=HeadDirection.UNKNOWN,
                yaw_angle=0.0,
                pitch_angle=0.0,
                confidence=0.0,
                nose_offset_ratio=0.0,
                ear_visibility_ratio=0.0
            )
        
        # Calculate shoulder center
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        if shoulder_width < 10:  # Too narrow, unreliable
            shoulder_width = 10
        
        # Nose offset from shoulder center (normalized by shoulder width)
        nose_offset_x = nose.x - shoulder_center_x
        nose_offset_ratio = nose_offset_x / shoulder_width
        
        # Ear visibility
        ear_vis = 0
        if left_ear and left_ear.confidence > 0.3:
            ear_vis += 1
        if right_ear and right_ear.confidence > 0.3:
            ear_vis += 1
        ear_visibility_ratio = ear_vis / 2.0
        
        # Estimate yaw from nose offset and ear visibility
        # Positive = right, Negative = left
        yaw_angle = nose_offset_ratio * 90  # Approximate degrees
        
        # Adjust based on ear visibility
        if ear_visibility_ratio > 0:
            if left_ear and left_ear.confidence > 0.3 and (not right_ear or right_ear.confidence < 0.3):
                # Only left ear visible -> head turned right
                yaw_angle = max(yaw_angle, 30)
            elif right_ear and right_ear.confidence > 0.3 and (not left_ear or left_ear.confidence < 0.3):
                # Only right ear visible -> head turned left
                yaw_angle = min(yaw_angle, -30)
        
        # Estimate pitch from nose vertical position relative to eyes
        pitch_angle = 0.0
        if left_eye and left_eye.confidence > 0.3 and right_eye and right_eye.confidence > 0.3:
            eye_center_y = (left_eye.y + right_eye.y) / 2
            nose_eye_diff = nose.y - eye_center_y
            # Normalize by eye-nose distance (rough)
            eye_nose_dist = abs(eye_center_y - nose.y) if abs(eye_center_y - nose.y) > 5 else 5
            pitch_angle = (nose_eye_diff / eye_nose_dist) * 45  # Approximate degrees
        
        # Classify direction
        direction = self._classify_direction(yaw_angle, pitch_angle)
        
        # Confidence based on keypoint quality
        kp_confs = [kp.confidence for kp in required if kp]
        confidence = np.mean(kp_confs) if kp_confs else 0.0
        
        return HeadFeatures(
            track_id=track_id,
            direction=direction,
            yaw_angle=yaw_angle,
            pitch_angle=pitch_angle,
            confidence=confidence,
            nose_offset_ratio=nose_offset_ratio,
            ear_visibility_ratio=ear_visibility_ratio
        )
    
    def _get_kp(self, keypoints: List, name: str):
        """Get keypoint by name."""
        idx = self.kp_indices.get(name)
        if idx is not None and idx < len(keypoints):
            return keypoints[idx]
        return None
    
    def _classify_direction(self, yaw: float, pitch: float) -> HeadDirection:
        """Classify head direction from yaw/pitch angles."""
        if abs(pitch) > self.pitch_threshold:
            if pitch > 0:
                return HeadDirection.DOWN
            else:
                return HeadDirection.UP
        
        if yaw > self.yaw_threshold:
            return HeadDirection.RIGHT
        elif yaw < -self.yaw_threshold:
            return HeadDirection.LEFT
        else:
            return HeadDirection.FRONT


class HandFeatureExtractor:
    """
    Extracts hand/wrist features from pose keypoints.
    
    Analyzes wrist position relative to:
    - Shoulders
    - Torso
    - Desk line (if available)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.desk_line_y = self.config.get("desk_line_y", None)
        self.face_distance_threshold = self.config.get("face_distance_threshold", 0.15)  # Relative to shoulder width
        self.lap_distance_threshold = self.config.get("lap_distance_threshold", 0.3)
        self.velocity_threshold = self.config.get("velocity_threshold", 50)  # pixels/frame
        
        self.kp_indices = {
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
            "nose": 0,
            "left_eye": 1,
            "right_eye": 2,
            "left_ear": 3,
            "right_ear": 4,
            "left_hip": 11,
            "right_hip": 12,
        }
        
        # History for velocity calculation
        self.wrist_history: Dict[int, deque] = {}
        self.max_history = 5
    
    def update_desk_line(self, desk_y: float):
        """Update desk line Y coordinate."""
        self.desk_line_y = desk_y
    
    def extract(self, keypoints: List, track_id: int, frame_idx: int) -> HandFeatures:
        """
        Extract hand features from keypoints.
        
        Args:
            keypoints: List of 17 Keypoint objects
            track_id: Track ID
            frame_idx: Current frame index
            
        Returns:
            HandFeatures
        """
        # Extract required keypoints
        left_shoulder = self._get_kp(keypoints, "left_shoulder")
        right_shoulder = self._get_kp(keypoints, "right_shoulder")
        left_wrist = self._get_kp(keypoints, "left_wrist")
        right_wrist = self._get_kp(keypoints, "right_wrist")
        nose = self._get_kp(keypoints, "nose")
        left_eye = self._get_kp(keypoints, "left_eye")
        right_eye = self._get_kp(keypoints, "right_eye")
        left_ear = self._get_kp(keypoints, "left_ear")
        right_ear = self._get_kp(keypoints, "right_ear")
        left_hip = self._get_kp(keypoints, "left_hip")
        right_hip = self._get_kp(keypoints, "right_hip")
        
        # Calculate shoulder width for normalization
        shoulder_width = 100  # Default
        if left_shoulder and right_shoulder and left_shoulder.confidence > 0.3 and right_shoulder.confidence > 0.3:
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            if shoulder_width < 20:
                shoulder_width = 100
        
        # Shoulder center
        shoulder_center_y = 0
        if left_shoulder and right_shoulder and left_shoulder.confidence > 0.3 and right_shoulder.confidence > 0.3:
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Process left wrist
        left_state, left_below_desk, left_near_face = self._analyze_wrist(
            left_wrist, left_shoulder, shoulder_center_y, shoulder_width,
            nose, left_eye, left_ear, left_hip, "left", track_id, frame_idx
        )
        
        # Process right wrist
        right_state, right_below_desk, right_near_face = self._analyze_wrist(
            right_wrist, right_shoulder, shoulder_center_y, shoulder_width,
            nose, right_eye, right_ear, right_hip, "right", track_id, frame_idx
        )
        
        # Calculate velocities
        left_vel = self._calculate_velocity(track_id, "left", left_wrist, frame_idx)
        right_vel = self._calculate_velocity(track_id, "right", right_wrist, frame_idx)
        
        # Fast movement check
        left_speed = np.sqrt(left_vel[0]**2 + left_vel[1]**2)
        right_speed = np.sqrt(right_vel[0]**2 + right_vel[1]**2)
        
        if left_speed > self.velocity_threshold:
            left_state = HandState.FAST_MOVEMENT
        if right_speed > self.velocity_threshold:
            right_state = HandState.FAST_MOVEMENT
        
        # Confidence
        wrist_confs = []
        if left_wrist and left_wrist.confidence > 0.3:
            wrist_confs.append(left_wrist.confidence)
        if right_wrist and right_wrist.confidence > 0.3:
            wrist_confs.append(right_wrist.confidence)
        confidence = np.mean(wrist_confs) if wrist_confs else 0.0
        
        return HandFeatures(
            track_id=track_id,
            left_hand_state=left_state,
            right_hand_state=right_state,
            left_wrist_pos=(left_wrist.x, left_wrist.y) if left_wrist else (0, 0),
            right_wrist_pos=(right_wrist.x, right_wrist.y) if right_wrist else (0, 0),
            left_wrist_vel=left_vel,
            right_wrist_vel=right_vel,
            left_below_desk=left_below_desk,
            right_below_desk=right_below_desk,
            left_near_face=left_near_face,
            right_near_face=right_near_face,
            confidence=confidence
        )
    
    def _analyze_wrist(self, wrist, shoulder, shoulder_center_y, shoulder_width,
                       nose, eye, ear, hip, side: str, track_id: int, frame_idx: int) -> Tuple[HandState, bool, bool]:
        """Analyze a single wrist position."""
        below_desk = False
        near_face = False
        state = HandState.NORMAL
        
        if not wrist or wrist.confidence < 0.25:
            return state, below_desk, near_face
        
        # Check if near ear first (talking / listening)
        if ear and ear.confidence > 0.25:
            dist_ear = np.sqrt((wrist.x - ear.x)**2 + (wrist.y - ear.y)**2)
            if dist_ear < 0.35 * shoulder_width:
                near_face = True
                state = HandState.NEAR_EAR
                return state, below_desk, near_face
        
        # Check if near face / mouth (whispering / covering mouth)
        face_kps = [kp for kp in [nose, eye] if kp and kp.confidence > 0.25]
        if face_kps:
            for face_kp in face_kps:
                dist_face = np.sqrt((wrist.x - face_kp.x)**2 + (wrist.y - face_kp.y)**2)
                if dist_face < 0.30 * shoulder_width:
                    near_face = True
                    state = HandState.NEAR_FACE
                    return state, below_desk, near_face
        
        # Check if below desk line
        if self.desk_line_y is not None:
            if wrist.y > self.desk_line_y:
                below_desk = True
                state = HandState.BELOW_DESK
        elif shoulder and shoulder.confidence > 0.25:
            # Fallback: below shoulder level
            if wrist.y > shoulder.y + 40:
                below_desk = True
                state = HandState.BELOW_SHOULDER
        
        # Check if near lap (near hip)
        if hip and hip.confidence > 0.25:
            dist_hip = np.sqrt((wrist.x - hip.x)**2 + (wrist.y - hip.y)**2)
            if dist_hip < self.lap_distance_threshold * shoulder_width:
                below_desk = True
                if state in [HandState.NORMAL, HandState.BELOW_SHOULDER]:
                    state = HandState.NEAR_LAP
        
        return state, below_desk, near_face
    
    def _calculate_velocity(self, track_id: int, side: str, wrist, frame_idx: int) -> Tuple[float, float]:
        """Calculate wrist velocity from history."""
        key = (track_id, side)
        if key not in self.wrist_history:
            self.wrist_history[key] = deque(maxlen=self.max_history)
        
        hist = self.wrist_history[key]
        
        if wrist and wrist.confidence > 0.3:
            hist.append((frame_idx, wrist.x, wrist.y))
        else:
            # No valid wrist, can't calculate velocity
            return (0.0, 0.0)
        
        if len(hist) < 2:
            return (0.0, 0.0)
        
        # Calculate velocity from last two positions
        prev_frame, prev_x, prev_y = hist[-2]
        curr_frame, curr_x, curr_y = hist[-1]
        
        dt = curr_frame - prev_frame
        if dt <= 0:
            return (0.0, 0.0)
        
        vel_x = (curr_x - prev_x) / dt
        vel_y = (curr_y - prev_y) / dt
        
        return (vel_x, vel_y)
    
    def _get_kp(self, keypoints: List, name: str):
        idx = self.kp_indices.get(name)
        if idx is not None and idx < len(keypoints):
            return keypoints[idx]
        return None
    
    def reset_track(self, track_id: int):
        """Clear history for a track."""
        keys_to_remove = [k for k in self.wrist_history if k[0] == track_id]
        for k in keys_to_remove:
            del self.wrist_history[k]


class TorsoOrientationEstimator:
    """Estimates torso angle and orientation toward neighbor seats."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.kp_indices = {
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_hip": 11,
            "right_hip": 12,
            "nose": 0,
        }
    
    def estimate(self, keypoints: List, track_id: int, 
                 neighbor_seat_center: Optional[Tuple[float, float]] = None) -> TorsoFeatures:
        """Estimate torso orientation."""
        left_shoulder = self._get_kp(keypoints, "left_shoulder")
        right_shoulder = self._get_kp(keypoints, "right_shoulder")
        left_hip = self._get_kp(keypoints, "left_hip")
        right_hip = self._get_kp(keypoints, "right_hip")
        nose = self._get_kp(keypoints, "nose")
        
        # Check required keypoints
        if not left_shoulder or not right_shoulder or left_shoulder.confidence < 0.25 or right_shoulder.confidence < 0.25:
            return TorsoFeatures(
                track_id=track_id,
                angle=0.0,
                angle_toward_neighbor=None,
                confidence=0.0
            )
        
        shoulder_center = np.array([(left_shoulder.x + right_shoulder.x) / 2,
                                     (left_shoulder.y + right_shoulder.y) / 2])
        
        # Shoulder axis tilt angle
        shoulder_dx = right_shoulder.x - left_shoulder.x
        shoulder_dy = right_shoulder.y - left_shoulder.y
        shoulder_tilt = np.degrees(np.arctan2(shoulder_dy, shoulder_dx))
        
        # If hips are visible, calculate torso lean
        if left_hip and right_hip and left_hip.confidence > 0.25 and right_hip.confidence > 0.25:
            hip_center = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])
            torso_vector = hip_center - shoulder_center
            torso_angle = np.degrees(np.arctan2(torso_vector[0], torso_vector[1]))
        else:
            torso_angle = shoulder_tilt
        
        # Angle toward neighbor seat
        angle_toward_neighbor = None
        if neighbor_seat_center is not None:
            n_dx = neighbor_seat_center[0] - shoulder_center[0]
            n_dy = neighbor_seat_center[1] - shoulder_center[1]
            norm_n = np.linalg.norm([n_dx, n_dy])
            
            if nose and nose.confidence > 0.25 and norm_n > 1:
                nose_vec = np.array([nose.x - shoulder_center[0], nose.y - shoulder_center[1]])
                norm_nose = np.linalg.norm(nose_vec)
                if norm_nose > 1:
                    cos_a = np.dot([n_dx, n_dy], nose_vec) / (norm_n * norm_nose)
                    angle_toward_neighbor = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
            else:
                angle_toward_neighbor = float(abs(torso_angle))
        
        conf = (left_shoulder.confidence + right_shoulder.confidence) / 2.0
        
        return TorsoFeatures(
            track_id=track_id,
            angle=float(torso_angle),
            angle_toward_neighbor=angle_toward_neighbor,
            confidence=float(conf)
        )
    
    def _get_kp(self, keypoints: List, name: str):
        idx = self.kp_indices.get(name)
        if idx is not None and idx < len(keypoints):
            return keypoints[idx]
        return None


class FeatureExtractor:
    """
    Main feature extraction pipeline combining head, hand, and torso features.
    """
    
    def __init__(self, config_path: str = "configs/rules.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        self.head_estimator = HeadOrientationEstimator(
            self.config.get("rules", {}).get("sideward_glance", {})
        )
        self.hand_extractor = HandFeatureExtractor(
            self.config.get("rules", {}).get("hand_below_desk", {})
        )
        self.torso_estimator = TorsoOrientationEstimator()
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return {}
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def extract(self, pose_result, track_result, frame_idx: int, 
                timestamp: float, seat_map) -> List[BehavioralFeatures]:
        """
        Extract behavioral features for all tracked persons.
        
        Args:
            pose_result: PoseResult with poses
            track_result: TrackingResult with tracks
            frame_idx: Current frame index
            timestamp: Current timestamp
            seat_map: SeatMap for desk line and neighbor info
            
        Returns:
            List of BehavioralFeatures
        """
        # Build track_id -> pose mapping
        pose_by_track = {}
        for pose in pose_result.poses:
            if pose.track_id >= 0:
                pose_by_track[pose.track_id] = pose
        
        # Build track_id -> seat_id mapping
        seat_by_track = {}
        for track in track_result.tracks:
            seat_by_track[track.track_id] = track.seat_id
        
        features_list = []
        
        for track in track_result.tracks:
            track_id = track.track_id
            seat_id = seat_by_track.get(track_id)
            
            # Get pose for this track
            pose = pose_by_track.get(track_id)
            
            # Get desk line Y for this seat
            desk_y = None
            if seat_id:
                desk_y = seat_map.get_desk_y(seat_id)
            self.hand_extractor.update_desk_line(desk_y)
            
            # Get neighbor seat center for torso angle
            neighbor_center = None
            if seat_id and seat_id in seat_map.seats:
                seat = seat_map.seats[seat_id]
                if seat.neighbours:
                    neighbor_id = seat.neighbours[0]
                    if neighbor_id in seat_map.seats:
                        neighbor = seat_map.seats[neighbor_id]
                        centroid = np.mean(neighbor.polygon, axis=0)
                        neighbor_center = (centroid[0], centroid[1])
            
            # Extract features from pose or fallback
            if pose is not None:
                head_features = self.head_estimator.estimate(pose.keypoints, track_id)
                hand_features = self.hand_extractor.extract(pose.keypoints, track_id, frame_idx)
                torso_features = self.torso_estimator.estimate(pose.keypoints, track_id, neighbor_center)
            else:
                head_features = HeadFeatures(
                    track_id=track_id,
                    direction=HeadDirection.FRONT,
                    yaw_angle=0.0,
                    pitch_angle=0.0,
                    confidence=0.5,
                    nose_offset_ratio=0.0,
                    ear_visibility_ratio=0.5
                )
                hand_features = HandFeatures(
                    track_id=track_id,
                    left_hand_state=HandState.NORMAL,
                    right_hand_state=HandState.NORMAL,
                    left_wrist_pos=(0, 0),
                    right_wrist_pos=(0, 0),
                    left_wrist_vel=(0.0, 0.0),
                    right_wrist_vel=(0.0, 0.0),
                    left_below_desk=False,
                    right_below_desk=False,
                    left_near_face=False,
                    right_near_face=False,
                    confidence=0.5
                )
                torso_features = TorsoFeatures(
                    track_id=track_id,
                    angle=0.0,
                    angle_toward_neighbor=None,
                    confidence=0.5
                )
            
            features = BehavioralFeatures(
                track_id=track_id,
                seat_id=seat_id,
                frame_idx=frame_idx,
                timestamp=timestamp,
                head=head_features,
                hands=hand_features,
                torso=torso_features
            )
            features_list.append(features)
        
        return features_list
    
    def draw_features(self, frame: np.ndarray, features_list: List[BehavioralFeatures]) -> np.ndarray:
        """Draw feature visualization on frame."""
        annotated = frame.copy()
        
        for feat in features_list:
            # Draw head direction
            head = feat.head
            if head.direction != HeadDirection.UNKNOWN:
                # Find nose position for text placement
                color = (0, 255, 0) if head.direction == HeadDirection.FRONT else (0, 165, 255)
                cv2.putText(annotated, f"Head: {head.direction.value}", 
                           (10, 30 + feat.track_id * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(annotated, f"Yaw: {head.yaw_angle:.1f} Pitch: {head.pitch_angle:.1f}",
                           (10, 60 + feat.track_id * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw hand states
            hands = feat.hands
            left_color = (0, 255, 0) if hands.left_hand_state == HandState.NORMAL else (0, 0, 255)
            right_color = (0, 255, 0) if hands.right_hand_state == HandState.NORMAL else (0, 0, 255)
            
            cv2.putText(annotated, f"L: {hands.left_hand_state.value}", 
                       (10, 90 + feat.track_id * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
            cv2.putText(annotated, f"R: {hands.right_hand_state.value}", 
                       (10, 110 + feat.track_id * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
            
            if hands.left_below_desk or hands.right_below_desk:
                cv2.putText(annotated, "BELOW DESK!", 
                           (10, 130 + feat.track_id * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw torso angle
            torso = feat.torso
            cv2.putText(annotated, f"Torso: {torso.angle:.1f} deg",
                       (10, 150 + feat.track_id * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if torso.angle_toward_neighbor is not None:
                cv2.putText(annotated, f"To Neighbor: {torso.angle_toward_neighbor:.1f} deg",
                           (10, 170 + feat.track_id * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return annotated
    
    def features_to_json(self, features_list: List[BehavioralFeatures]) -> List[Dict]:
        """Convert features to JSON-serializable format."""
        result = []
        for feat in features_list:
            result.append({
                "track_id": feat.track_id,
                "seat_id": feat.seat_id,
                "frame_idx": feat.frame_idx,
                "timestamp": feat.timestamp,
                "head": {
                    "direction": feat.head.direction.value,
                    "yaw_angle": feat.head.yaw_angle,
                    "pitch_angle": feat.head.pitch_angle,
                    "confidence": feat.head.confidence,
                    "nose_offset_ratio": feat.head.nose_offset_ratio,
                    "ear_visibility_ratio": feat.head.ear_visibility_ratio
                },
                "hands": {
                    "left_state": feat.hands.left_hand_state.value,
                    "right_state": feat.hands.right_hand_state.value,
                    "left_wrist_pos": feat.hands.left_wrist_pos,
                    "right_wrist_pos": feat.hands.right_wrist_pos,
                    "left_wrist_vel": feat.hands.left_wrist_vel,
                    "right_wrist_vel": feat.hands.right_wrist_vel,
                    "left_below_desk": feat.hands.left_below_desk,
                    "right_below_desk": feat.hands.right_below_desk,
                    "left_near_face": feat.hands.left_near_face,
                    "right_near_face": feat.hands.right_near_face,
                    "confidence": feat.hands.confidence
                },
                "torso": {
                    "angle": feat.torso.angle,
                    "angle_toward_neighbor": feat.torso.angle_toward_neighbor,
                    "confidence": feat.torso.confidence
                }
            })
        return result


def create_feature_extractor(config_path: str = "configs/rules.yaml") -> FeatureExtractor:
    """Factory function to create FeatureExtractor from config."""
    return FeatureExtractor(config_path)