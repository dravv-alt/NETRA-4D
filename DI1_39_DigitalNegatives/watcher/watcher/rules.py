"""
Rule Engine and Event Detection for Watcher.

Detects behavioral events from temporal features using configurable rules.
"""

import yaml
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of behavioral events."""
    REPEATED_SIDEWARD_GLANCE = "REPEATED_SIDEWARD_GLANCE"
    SUSTAINED_HEAD_TURN = "SUSTAINED_HEAD_TURN"
    TORSO_ROTATION_TOWARD_NEIGHBOR = "TORSO_ROTATION_TOWARD_NEIGHBOR"
    HAND_BELOW_DESK = "HAND_BELOW_DESK"
    HAND_NEAR_EAR = "HAND_NEAR_EAR"
    HAND_NEAR_FACE = "HAND_NEAR_FACE"
    REPEATED_GESTURE = "REPEATED_GESTURE"
    OBJECT_NEAR_HAND = "OBJECT_NEAR_HAND"
    PHONE_DETECTED = "PHONE_DETECTED"


@dataclass
class BehaviorEvent:
    """A detected behavioral event."""
    event_type: EventType
    seat_id: str
    track_id: int
    timestamp: float
    frame_idx: int
    duration_sec: float
    count: int = 1
    confidence: float = 0.0
    details: Dict[str, Any] = None
    weight: float = 0.0
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class RuleEngine:
    """
    Rule-based event detection engine.
    
    Evaluates temporal feature windows against configured rules
    to detect suspicious behavioral patterns.
    """
    
    def __init__(self, config_path: str = "configs/rules.yaml", fps: float = 25.0):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.fps = fps
        self.rules_config = self.config.get("rules", {})
        
        # Event history per seat for temporal rules
        self.event_history: Dict[str, deque] = {}
        self.max_history_seconds = 300  # 5 minutes
        self.max_history_frames = int(self.max_history_seconds * fps)
        
        # Track active events to avoid duplicate firing
        self.active_events: Dict[str, Dict[EventType, float]] = {}
        self.cooldown_sec = self.config.get("alerts", {}).get("cooldown_sec", 30)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load rules configuration."""
        if not self.config_path.exists():
            logger.warning(f"Rules config not found: {self.config_path}, using defaults")
            return {}
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_seat_history(self, seat_id: str) -> deque:
        """Get or create event history for a seat."""
        if seat_id not in self.event_history:
            self.event_history[seat_id] = deque(maxlen=self.max_history_frames)
        return self.event_history[seat_id]
    
    def _check_cooldown(self, seat_id: str, event_type: EventType, timestamp: float) -> bool:
        """Check if event is in cooldown period."""
        if seat_id not in self.active_events:
            return True
        if event_type not in self.active_events[seat_id]:
            return True
        last_time = self.active_events[seat_id][event_type]
        return (timestamp - last_time) >= self.cooldown_sec
    
    def _update_cooldown(self, seat_id: str, event_type: EventType, timestamp: float):
        """Update cooldown for an event."""
        if seat_id not in self.active_events:
            self.active_events[seat_id] = {}
        self.active_events[seat_id][event_type] = timestamp
    
    def evaluate(self, 
                 seat_id: str,
                 track_id: int,
                 timestamp: float,
                 frame_idx: int,
                 behavioral_features = None,  # BehavioralFeatures object
                 fusion_result = None) -> List[BehaviorEvent]:
        """
        Evaluate rules against behavioral features.
        
        Args:
            seat_id: Seat ID
            track_id: Track ID
            timestamp: Current timestamp
            frame_idx: Current frame index
            behavioral_features: BehavioralFeatures object from feature extractor
            fusion_result: Optional fusion result for object detection
            
        Returns:
            List of detected BehaviorEvent
        """
        events = []
        history = self._get_seat_history(seat_id)
        
        # Extract features for history
        if behavioral_features:
            # Head features
            head_dir = behavioral_features.head.direction.value
            head_yaw = behavioral_features.head.yaw_angle
            head_pitch = behavioral_features.head.pitch_angle
            head_conf = behavioral_features.head.confidence
            
            # Hand features
            left_hand_state = behavioral_features.hands.left_hand_state.value
            right_hand_state = behavioral_features.hands.right_hand_state.value
            left_below_desk = behavioral_features.hands.left_below_desk
            right_below_desk = behavioral_features.hands.right_below_desk
            left_near_face = behavioral_features.hands.left_near_face
            right_near_face = behavioral_features.hands.right_near_face
            hands_conf = behavioral_features.hands.confidence
            
            # Torso features
            torso_angle = behavioral_features.torso.angle
            torso_toward_neighbor = behavioral_features.torso.angle_toward_neighbor
            torso_conf = behavioral_features.torso.confidence
            
            frame_features = {
                "head_direction": head_dir,
                "head_yaw": head_yaw,
                "head_pitch": head_pitch,
                "head_confidence": head_conf,
                "left_hand_state": left_hand_state,
                "right_hand_state": right_hand_state,
                "left_below_desk": left_below_desk,
                "right_below_desk": right_below_desk,
                "left_near_face": left_near_face,
                "right_near_face": right_near_face,
                "hands_confidence": hands_conf,
                "torso_angle": torso_angle,
                "torso_toward_neighbor": torso_toward_neighbor,
                "torso_confidence": torso_conf,
                "fusion_score": fusion_result.fusion_score if fusion_result else 0,
                "object_detected": fusion_result.object_detected if fusion_result else False,
                "object_class": fusion_result.object_class if fusion_result else None,
                "object_confidence": fusion_result.object_confidence if fusion_result else 0,
            }
        else:
            frame_features = {}
        
        # Add current frame features to history
        history.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "features": frame_features
        })
        
        # Check each rule
        # 1. Repeated sideward glance
        if self.rules_config.get("sideward_glance", {}).get("enabled", True):
            event = self._check_sideward_glance(seat_id, track_id, timestamp, frame_idx, history)
            if event and self._check_cooldown(seat_id, event.event_type, timestamp):
                events.append(event)
                self._update_cooldown(seat_id, event.event_type, timestamp)
        
        # 2. Sustained head turn
        if self.rules_config.get("sustained_head_turn", {}).get("enabled", True):
            event = self._check_sustained_head_turn(seat_id, track_id, timestamp, frame_idx, history)
            if event and self._check_cooldown(seat_id, event.event_type, timestamp):
                events.append(event)
                self._update_cooldown(seat_id, event.event_type, timestamp)
        
        # 3. Torso rotation toward neighbor
        if self.rules_config.get("torso_rotation", {}).get("enabled", True):
            event = self._check_torso_rotation(seat_id, track_id, timestamp, frame_idx, history)
            if event and self._check_cooldown(seat_id, event.event_type, timestamp):
                events.append(event)
                self._update_cooldown(seat_id, event.event_type, timestamp)
        
        # 4. Hand below desk
        if self.rules_config.get("hand_below_desk", {}).get("enabled", True):
            event = self._check_hand_below_desk(seat_id, track_id, timestamp, frame_idx, history)
            if event and self._check_cooldown(seat_id, event.event_type, timestamp):
                events.append(event)
                self._update_cooldown(seat_id, event.event_type, timestamp)
        
        # 5. Hand near ear
        if self.rules_config.get("hand_near_ear", {}).get("enabled", True):
            event = self._check_hand_near_ear(seat_id, track_id, timestamp, frame_idx, history)
            if event and self._check_cooldown(seat_id, event.event_type, timestamp):
                events.append(event)
                self._update_cooldown(seat_id, event.event_type, timestamp)
        
        # 6. Hand near face
        if self.rules_config.get("hand_near_face", {}).get("enabled", True):
            event = self._check_hand_near_face(seat_id, track_id, timestamp, frame_idx, history)
            if event and self._check_cooldown(seat_id, event.event_type, timestamp):
                events.append(event)
                self._update_cooldown(seat_id, event.event_type, timestamp)
        
        # 7. Object near hand (from fusion)
        if self.rules_config.get("object_near_hand", {}).get("enabled", True):
            event = self._check_object_near_hand(seat_id, track_id, timestamp, frame_idx, history, fusion_result)
            if event and self._check_cooldown(seat_id, event.event_type, timestamp):
                events.append(event)
                self._update_cooldown(seat_id, event.event_type, timestamp)
        
        # 8. Phone detected
        if self.rules_config.get("phone_detected", {}).get("enabled", True):
            event = self._check_phone_detected(seat_id, track_id, timestamp, frame_idx, fusion_result)
            if event and self._check_cooldown(seat_id, event.event_type, timestamp):
                events.append(event)
                self._update_cooldown(seat_id, event.event_type, timestamp)
        
        return events
    
    def _check_sideward_glance(self, seat_id: str, track_id: int, 
                                timestamp: float, frame_idx: int, 
                                history: deque) -> Optional[BehaviorEvent]:
        """Check for repeated sideward glances."""
        rule = self.rules_config.get("sideward_glance", {})
        yaw_threshold = rule.get("yaw_threshold_deg", 20)
        min_duration = rule.get("minimum_duration_sec", 0.4)
        min_count = rule.get("minimum_count", 2)
        time_window = rule.get("time_window_sec", 60)
        conf_threshold = rule.get("confidence_threshold", 0.45)
        weight = rule.get("weight", 3)
        
        window_frames = int(time_window * self.fps)
        recent = list(history)[-window_frames:] if len(history) > window_frames else list(history)
        
        glance_count = 0
        current_glance_start = None
        current_direction = None
        
        for entry in recent:
            head_dir = entry["features"].get("head_direction", "UNKNOWN")
            head_yaw = entry["features"].get("head_yaw", 0)
            conf = entry["features"].get("head_confidence", 0)
            
            is_sideward = (head_dir in ["LEFT", "RIGHT"] and 
                          abs(head_yaw) >= yaw_threshold and 
                          conf >= conf_threshold)
            
            if is_sideward:
                if current_glance_start is None:
                    current_glance_start = entry["timestamp"]
                    current_direction = head_dir
                elif head_dir == current_direction:
                    pass
                else:
                    if current_glance_start and (entry["timestamp"] - current_glance_start) >= min_duration:
                        glance_count += 1
                    current_glance_start = entry["timestamp"]
                    current_direction = head_dir
            else:
                if current_glance_start and (entry["timestamp"] - current_glance_start) >= min_duration:
                    glance_count += 1
                current_glance_start = None
                current_direction = None
        
        if current_glance_start and recent:
            if (recent[-1]["timestamp"] - current_glance_start) >= min_duration:
                glance_count += 1
        
        if glance_count >= min_count:
            return BehaviorEvent(
                event_type=EventType.REPEATED_SIDEWARD_GLANCE,
                seat_id=seat_id,
                track_id=track_id,
                timestamp=timestamp,
                frame_idx=frame_idx,
                duration_sec=time_window,
                count=glance_count,
                confidence=min(0.95, glance_count / min_count * 0.7 + 0.25),
                details={"direction": current_direction, "time_window_sec": time_window},
                weight=weight
            )
        return None
    
    def _check_sustained_head_turn(self, seat_id: str, track_id: int,
                                    timestamp: float, frame_idx: int,
                                    history: deque) -> Optional[BehaviorEvent]:
        """Check for sustained head turn or looking down at lap."""
        rule = self.rules_config.get("sustained_head_turn", {})
        yaw_threshold = rule.get("yaw_threshold_deg", 25)
        min_duration = rule.get("minimum_duration_sec", 1.5)
        conf_threshold = rule.get("confidence_threshold", 0.45)
        weight = rule.get("weight", 4)
        
        min_frames = int(min_duration * self.fps)
        recent = list(history)[-min_frames:]
        
        if len(recent) < min_frames:
            return None
        
        sustained_frames = 0
        direction = None
        
        for entry in recent:
            head_dir = entry["features"].get("head_direction", "UNKNOWN")
            head_yaw = entry["features"].get("head_yaw", 0)
            head_pitch = entry["features"].get("head_pitch", 0)
            conf = entry["features"].get("head_confidence", 0)
            
            is_turned = ((head_dir in ["LEFT", "RIGHT"] and abs(head_yaw) >= yaw_threshold) or
                         (head_dir == "DOWN" and abs(head_pitch) >= 15)) and (conf >= conf_threshold)
            
            if is_turned:
                sustained_frames += 1
                direction = head_dir
            else:
                sustained_frames = 0
        
        if sustained_frames >= min_frames:
            return BehaviorEvent(
                event_type=EventType.SUSTAINED_HEAD_TURN,
                seat_id=seat_id,
                track_id=track_id,
                timestamp=timestamp,
                frame_idx=frame_idx,
                duration_sec=sustained_frames / self.fps,
                confidence=min(0.95, sustained_frames / min_frames * 0.7 + 0.25),
                details={"direction": direction, "yaw_threshold": yaw_threshold},
                weight=weight
            )
        return None
    
    def _check_torso_rotation(self, seat_id: str, track_id: int,
                               timestamp: float, frame_idx: int,
                               history: deque) -> Optional[BehaviorEvent]:
        """Check for torso rotation toward neighbor."""
        rule = self.rules_config.get("torso_rotation", {})
        angle_threshold = rule.get("torso_angle_threshold_deg", 12)
        min_duration = rule.get("minimum_duration_sec", 2.0)
        conf_threshold = rule.get("confidence_threshold", 0.45)
        weight = rule.get("weight", 4)
        
        min_frames = int(min_duration * self.fps)
        recent = list(history)[-min_frames:]
        
        if len(recent) < min_frames:
            return None
        
        sustained_frames = 0
        for entry in recent:
            torso_angle = abs(entry["features"].get("torso_angle", 0))
            torso_to_neighbor = entry["features"].get("torso_toward_neighbor")
            conf = entry["features"].get("torso_confidence", 0)
            
            rotated = (torso_angle >= angle_threshold or (torso_to_neighbor is not None and torso_to_neighbor <= 45)) and (conf >= conf_threshold)
            if rotated:
                sustained_frames += 1
            else:
                sustained_frames = 0
        
        if sustained_frames >= min_frames:
            return BehaviorEvent(
                event_type=EventType.TORSO_ROTATION_TOWARD_NEIGHBOR,
                seat_id=seat_id,
                track_id=track_id,
                timestamp=timestamp,
                frame_idx=frame_idx,
                duration_sec=sustained_frames / self.fps,
                confidence=min(0.95, sustained_frames / min_frames * 0.7 + 0.25),
                details={"angle_threshold": angle_threshold},
                weight=weight
            )
        return None
    
    def _check_hand_below_desk(self, seat_id: str, track_id: int,
                                timestamp: float, frame_idx: int,
                                history: deque) -> Optional[BehaviorEvent]:
        """Check for hand below desk / in lap (chit concealment proxy)."""
        rule = self.rules_config.get("hand_below_desk", {})
        min_duration = rule.get("minimum_duration_sec", 2.0)
        conf_threshold = rule.get("confidence_threshold", 0.40)
        weight = rule.get("weight", 4)
        
        min_frames = int(min_duration * self.fps)
        recent = list(history)[-min_frames:]
        
        if len(recent) < min_frames:
            return None
        
        sustained_frames = 0
        for entry in recent:
            left_below = entry["features"].get("left_below_desk", False)
            right_below = entry["features"].get("right_below_desk", False)
            left_state = entry["features"].get("left_hand_state", "")
            right_state = entry["features"].get("right_hand_state", "")
            conf = entry["features"].get("hands_confidence", 0)
            
            is_below = (left_below or right_below or 
                        left_state in ["BELOW_DESK", "NEAR_LAP", "BELOW_SHOULDER"] or 
                        right_state in ["BELOW_DESK", "NEAR_LAP", "BELOW_SHOULDER"])
            
            if is_below and conf >= conf_threshold:
                sustained_frames += 1
            else:
                sustained_frames = 0
        
        if sustained_frames >= min_frames:
            return BehaviorEvent(
                event_type=EventType.HAND_BELOW_DESK,
                seat_id=seat_id,
                track_id=track_id,
                timestamp=timestamp,
                frame_idx=frame_idx,
                duration_sec=sustained_frames / self.fps,
                confidence=min(0.95, sustained_frames / min_frames * 0.7 + 0.25),
                details={"min_duration_sec": min_duration},
                weight=weight
            )
        return None
    
    def _check_hand_near_ear(self, seat_id: str, track_id: int,
                              timestamp: float, frame_idx: int,
                              history: deque) -> Optional[BehaviorEvent]:
        """Check for hand near ear (listening/communication)."""
        rule = self.rules_config.get("hand_near_ear", {})
        min_duration = rule.get("minimum_duration_sec", 1.0)
        conf_threshold = rule.get("confidence_threshold", 0.40)
        weight = rule.get("weight", 4)
        
        min_frames = int(min_duration * self.fps)
        recent = list(history)[-min_frames:]
        
        if len(recent) < min_frames:
            return None
        
        sustained_frames = 0
        for entry in recent:
            left_state = entry["features"].get("left_hand_state", "")
            right_state = entry["features"].get("right_hand_state", "")
            conf = entry["features"].get("hands_confidence", 0)
            
            near_ear = (left_state == "NEAR_EAR" or right_state == "NEAR_EAR")
            
            if near_ear and conf >= conf_threshold:
                sustained_frames += 1
            else:
                sustained_frames = 0
        
        if sustained_frames >= min_frames:
            return BehaviorEvent(
                event_type=EventType.HAND_NEAR_EAR,
                seat_id=seat_id,
                track_id=track_id,
                timestamp=timestamp,
                frame_idx=frame_idx,
                duration_sec=sustained_frames / self.fps,
                confidence=min(0.95, sustained_frames / min_frames * 0.7 + 0.25),
                details={"min_duration_sec": min_duration},
                weight=weight
            )
        return None
    
    def _check_hand_near_face(self, seat_id: str, track_id: int,
                               timestamp: float, frame_idx: int,
                               history: deque) -> Optional[BehaviorEvent]:
        """Check for hand near face/mouth (candidate talking/whispering)."""
        rule = self.rules_config.get("hand_near_face", {})
        min_duration = rule.get("minimum_duration_sec", 1.0)
        conf_threshold = rule.get("confidence_threshold", 0.40)
        weight = rule.get("weight", 3)
        
        min_frames = int(min_duration * self.fps)
        recent = list(history)[-min_frames:]
        
        if len(recent) < min_frames:
            return None
        
        sustained_frames = 0
        for entry in recent:
            left_state = entry["features"].get("left_hand_state", "")
            right_state = entry["features"].get("right_hand_state", "")
            left_near = entry["features"].get("left_near_face", False)
            right_near = entry["features"].get("right_near_face", False)
            conf = entry["features"].get("hands_confidence", 0)
            
            near_face = (left_state in ["NEAR_FACE", "NEAR_EAR"] or 
                         right_state in ["NEAR_FACE", "NEAR_EAR"] or 
                         left_near or right_near)
            
            if near_face and conf >= conf_threshold:
                sustained_frames += 1
            else:
                sustained_frames = 0
        
        if sustained_frames >= min_frames:
            return BehaviorEvent(
                event_type=EventType.HAND_NEAR_FACE,
                seat_id=seat_id,
                track_id=track_id,
                timestamp=timestamp,
                frame_idx=frame_idx,
                duration_sec=sustained_frames / self.fps,
                confidence=min(0.95, sustained_frames / min_frames * 0.7 + 0.25),
                details={"min_duration_sec": min_duration},
                weight=weight
            )
        return None
    
    def _check_object_near_hand(self, seat_id: str, track_id: int,
                                 timestamp: float, frame_idx: int,
                                 history: deque, fusion_result) -> Optional[BehaviorEvent]:
        """Check for suspicious object near hand/lap."""
        rule = self.rules_config.get("object_near_hand", {})
        min_duration = rule.get("minimum_duration_sec", 0.5)
        weight = rule.get("weight", 6)
        
        if fusion_result is None or not fusion_result.object_detected:
            return None
        
        return BehaviorEvent(
            event_type=EventType.OBJECT_NEAR_HAND,
            seat_id=seat_id,
            track_id=track_id,
            timestamp=timestamp,
            frame_idx=frame_idx,
            duration_sec=min_duration,
            confidence=max(0.7, fusion_result.fusion_score),
            details={
                "object_class": fusion_result.object_class,
                "object_confidence": fusion_result.object_confidence,
                "fusion_score": fusion_result.fusion_score
            },
            weight=weight
        )
    
    def _check_phone_detected(self, seat_id: str, track_id: int,
                               timestamp: float, frame_idx: int,
                               fusion_result) -> Optional[BehaviorEvent]:
        """Check for phone detection."""
        rule = self.rules_config.get("phone_detected", {})
        min_conf = rule.get("minimum_confidence", 0.15)
        min_duration = rule.get("minimum_duration_sec", 0.3)
        weight = rule.get("weight", 8)
        
        if fusion_result is None or not fusion_result.object_detected:
            return None
        
        if fusion_result.object_class != "cell phone":
            return None
        
        if fusion_result.object_confidence < min_conf:
            return None
        
        return BehaviorEvent(
            event_type=EventType.PHONE_DETECTED,
            seat_id=seat_id,
            track_id=track_id,
            timestamp=timestamp,
            frame_idx=frame_idx,
            duration_sec=min_duration,
            confidence=max(0.75, fusion_result.object_confidence),
            details={
                "object_class": "cell phone",
                "object_confidence": fusion_result.object_confidence,
                "fusion_score": fusion_result.fusion_score
            },
            weight=weight
        )
    
    def get_recent_events(self, seat_id: str, time_window_sec: float = 60) -> List[BehaviorEvent]:
        """Get recent events for a seat."""
        # This would need event storage - simplified for now
        return []
    
    def event_to_json(self, event: BehaviorEvent) -> Dict[str, Any]:
        """Convert event to JSON-serializable dict."""
        return {
            "event_type": event.event_type.value,
            "seat_id": event.seat_id,
            "track_id": event.track_id,
            "timestamp": event.timestamp,
            "frame_idx": event.frame_idx,
            "duration_sec": event.duration_sec,
            "count": event.count,
            "confidence": event.confidence,
            "details": event.details,
            "weight": event.weight
        }


def create_rule_engine(config_path: str = "configs/rules.yaml", fps: float = 25.0) -> RuleEngine:
    """Factory function to create RuleEngine from config."""
    return RuleEngine(config_path, fps)