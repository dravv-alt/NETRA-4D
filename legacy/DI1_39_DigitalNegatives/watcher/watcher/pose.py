"""
Pose Estimation and Keypoint Stability for Watcher.

Uses YOLO-Pose (Ultralytics) for keypoint detection with smoothing and confidence gating.
"""

import cv2
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Keypoint:
    """Single keypoint with position and confidence."""
    x: float
    y: float
    confidence: float
    name: str = ""


@dataclass
class Pose:
    """Pose for a single person/track."""
    track_id: int
    keypoints: List[Keypoint]  # 17 keypoints in COCO format
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    pose_confidence: float  # Overall pose confidence
    valid_keypoints: int = 0  # Number of keypoints above threshold


@dataclass
class PoseResult:
    """All poses for a single frame."""
    frame_idx: int
    poses: List[Pose]
    inference_time_ms: float


class KeypointSmoother:
    """
    Applies temporal smoothing to keypoints.
    Supports EMA (Exponential Moving Average) and One Euro Filter.
    """
    
    def __init__(self, method: str = "ema", ema_alpha: float = 0.3, 
                 one_euro_params: Dict = None):
        self.method = method
        self.ema_alpha = ema_alpha
        self.one_euro_params = one_euro_params or {
            "min_cutoff": 1.0, "beta": 0.0, "d_cutoff": 1.0
        }
        
        # Per-track keypoint history: track_id -> keypoint_idx -> deque of (x, y, conf)
        self.history: Dict[int, Dict[int, deque]] = {}
        self.max_history = 10
        
        # For One Euro Filter
        self.one_euro_state: Dict[int, Dict[int, Dict]] = {}
    
    def update(self, track_id: int, keypoints: List[Keypoint]) -> List[Keypoint]:
        """Update smoother with new keypoints and return smoothed keypoints."""
        if track_id not in self.history:
            self.history[track_id] = {}
            self.one_euro_state[track_id] = {}
        
        smoothed = []
        for i, kp in enumerate(keypoints):
            if i not in self.history[track_id]:
                self.history[track_id][i] = deque(maxlen=self.max_history)
                self.one_euro_state[track_id][i] = {"x_prev": kp.x, "y_prev": kp.y, 
                                                     "dx_prev": 0.0, "dy_prev": 0.0}
            
            hist = self.history[track_id][i]
            hist.append((kp.x, kp.y, kp.confidence))
            
            if self.method == "ema":
                smooth_x, smooth_y = self._ema_smooth(hist, kp.x, kp.y)
            elif self.method == "one_euro":
                smooth_x, smooth_y = self._one_euro_smooth(track_id, i, kp.x, kp.y)
            else:
                smooth_x, smooth_y = kp.x, kp.y
            
            smoothed.append(Keypoint(
                x=smooth_x,
                y=smooth_y,
                confidence=kp.confidence,
                name=kp.name
            ))
        
        return smoothed
    
    def _ema_smooth(self, hist: deque, curr_x: float, curr_y: float) -> Tuple[float, float]:
        """Exponential Moving Average smoothing."""
        if len(hist) == 1:
            return curr_x, curr_y
        
        # Weighted average of recent positions
        alpha = self.ema_alpha
        smooth_x = curr_x
        smooth_y = curr_y
        
        for hx, hy, hconf in hist:
            if hconf > 0.3:  # Only use confident keypoints
                smooth_x = alpha * hx + (1 - alpha) * smooth_x
                smooth_y = alpha * hy + (1 - alpha) * smooth_y
        
        return smooth_x, smooth_y
    
    def _one_euro_smooth(self, track_id: int, kp_idx: int, x: float, y: float) -> Tuple[float, float]:
        """One Euro Filter for adaptive smoothing."""
        # Simplified One Euro Filter
        state = self.one_euro_state[track_id][kp_idx]
        min_cutoff = self.one_euro_params.get("min_cutoff", 1.0)
        beta = self.one_euro_params.get("beta", 0.0)
        d_cutoff = self.one_euro_params.get("d_cutoff", 1.0)
        
        # Estimate velocity
        dx = x - state["x_prev"]
        dy = y - state["y_prev"]
        
        # Smooth velocity
        dx_smooth = state["dx_prev"] + beta * (dx - state["dx_prev"])
        dy_smooth = state["dy_prev"] + beta * (dy - state["dy_prev"])
        
        # Update state
        state["x_prev"] = x
        state["y_prev"] = y
        state["dx_prev"] = dx_smooth
        state["dy_prev"] = dy_smooth
        
        # For simplicity, return EMA-smoothed position
        # Full One Euro would adapt cutoff based on velocity
        return self._ema_smooth(self.history[track_id][kp_idx], x, y)
    
    def reset_track(self, track_id: int):
        """Clear history for a track."""
        if track_id in self.history:
            del self.history[track_id]
        if track_id in self.one_euro_state:
            del self.one_euro_state[track_id]


class PoseEstimator:
    """
    YOLO-Pose estimator with confidence gating and keypoint smoothing.
    """
    
    def __init__(self, config_path: str = "configs/pose.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model: Optional[YOLO] = None
        self.keypoint_names = self._get_keypoint_names()
        self.smoother = KeypointSmoother(
            method=self.config.get("smoothing", {}).get("method", "ema"),
            ema_alpha=self.config.get("smoothing", {}).get("ema_alpha", 0.3),
            one_euro_params=self.config.get("smoothing", {}).get("one_euro", {})
        )
        self._load_model()
    
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_keypoint_names(self) -> Dict[int, str]:
        """COCO 17-keypoint names."""
        return {
            0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
            5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
            9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
            13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
        }
    
    def _load_model(self):
        """Load YOLO-Pose model."""
        pose_config = self.config.get("pose", {})
        model_name = pose_config.get("model", "yolov8n-pose.pt")
        device = pose_config.get("device", "auto")
        imgsz = pose_config.get("imgsz", 640)
        half = pose_config.get("half", False)
        
        logger.info(f"Loading YOLO-Pose model: {model_name} on {device}")
        self.model = YOLO(model_name)
        
        if device != "auto":
            self.model.to(device)
        
        if half:
            self.model.model.half()
    
    def estimate(self, frame: np.ndarray, frame_idx: int, 
                 track_ids: List[int] = None, track_bboxes: List[Tuple] = None) -> PoseResult:
        """
        Run pose estimation on frame.
        
        Args:
            frame: Input frame.
            frame_idx: Frame index.
            track_ids: Optional list of track IDs (for smoothing per track).
            track_bboxes: Optional list of track bboxes to match poses to tracks.
            
        Returns:
            PoseResult with poses for each detected person.
        """
        import time
        start_time = time.time()
        
        pose_config = self.config.get("pose", {})
        conf_thresh = pose_config.get("confidence_threshold", 0.5)
        iou_thresh = pose_config.get("iou_threshold", 0.45)
        imgsz = pose_config.get("imgsz", 640)
        
        kp_config = self.config.get("keypoints", {})
        kp_conf_thresh = kp_config.get("keypoint_conf_threshold", 0.3)
        min_pose_conf = self.config.get("confidence_gating", {}).get("min_pose_confidence", 0.5)
        min_valid_kp = self.config.get("confidence_gating", {}).get("min_valid_keypoints", 8)
        
        # Run pose estimation
        results = self.model(
            frame,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=imgsz,
            verbose=False
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        poses = []
        
        for result in results:
            if result.keypoints is None:
                continue
            
            keypoints_data = result.keypoints.data  # (N, 17, 3) - x, y, conf
            boxes = result.boxes
            
            for i in range(len(keypoints_data)):
                kps = keypoints_data[i].cpu().numpy()
                bbox = boxes[i].xyxy[0].cpu().numpy().astype(int) if boxes is not None else (0, 0, 0, 0)
                
                # Build keypoint list
                keypoints = []
                valid_count = 0
                total_conf = 0.0
                
                for kp_idx in range(17):
                    x, y, conf = kps[kp_idx]
                    name = self.keypoint_names.get(kp_idx, f"kp_{kp_idx}")
                    
                    keypoints.append(Keypoint(
                        x=float(x), y=float(y), confidence=float(conf), name=name
                    ))
                    
                    if conf >= kp_conf_thresh:
                        valid_count += 1
                        total_conf += conf
                
                pose_conf = total_conf / valid_count if valid_count > 0 else 0.0
                
                # Confidence gating
                if pose_conf < min_pose_conf or valid_count < min_valid_kp:
                    continue
                
                # Determine track_id if provided
                track_id = -1
                if track_ids and track_bboxes:
                    best_match_score = -1.0
                    for tid, tbbox in zip(track_ids, track_bboxes):
                        iou = self._calculate_iou(tuple(bbox), tbbox)
                        # Check center inclusion
                        p_cx = (bbox[0] + bbox[2]) / 2.0
                        p_cy = (bbox[1] + bbox[3]) / 2.0
                        inside = (tbbox[0] <= p_cx <= tbbox[2]) and (tbbox[1] <= p_cy <= tbbox[3])
                        
                        match_score = iou + (0.5 if inside else 0.0)
                        if match_score > best_match_score and (iou > 0.1 or inside):
                            best_match_score = match_score
                            track_id = tid
                
                pose = Pose(
                    track_id=track_id,
                    keypoints=keypoints,
                    bbox=tuple(bbox),
                    pose_confidence=pose_conf,
                    valid_keypoints=valid_count
                )
                poses.append(pose)
        
        # Apply smoothing per track
        for pose in poses:
            if pose.track_id >= 0:
                pose.keypoints = self.smoother.update(pose.track_id, pose.keypoints)
        
        return PoseResult(
            frame_idx=frame_idx,
            poses=poses,
            inference_time_ms=inference_time_ms
        )
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                       bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def draw_poses(self, frame: np.ndarray, result: PoseResult, 
                   draw_skeleton: bool = True, draw_keypoints: bool = True) -> np.ndarray:
        """Draw pose skeletons and keypoints on frame."""
        annotated = frame.copy()
        
        # COCO skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        colors = {
            "head": (255, 0, 0),      # Blue
            "arms": (0, 255, 0),      # Green
            "torso": (0, 0, 255),     # Red
            "legs": (255, 255, 0)     # Cyan
        }
        
        for pose in result.poses:
            kps = pose.keypoints
            
            if draw_skeleton:
                for (i, j) in skeleton:
                    if i < len(kps) and j < len(kps):
                        kp1, kp2 = kps[i], kps[j]
                        if kp1.confidence > 0.3 and kp2.confidence > 0.3:
                            pt1 = (int(kp1.x), int(kp1.y))
                            pt2 = (int(kp2.x), int(kp2.y))
                            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)
            
            if draw_keypoints:
                for kp in kps:
                    if kp.confidence > 0.3:
                        pt = (int(kp.x), int(kp.y))
                        # Color by confidence
                        conf_color = int(255 * kp.confidence)
                        cv2.circle(annotated, pt, 4, (0, conf_color, 255 - conf_color), -1)
        
        return annotated
    
    def get_summary(self, result: PoseResult) -> Dict[str, Any]:
        """Get pose estimation summary."""
        avg_conf = 0.0
        avg_valid = 0
        if result.poses:
            avg_conf = sum(p.pose_confidence for p in result.poses) / len(result.poses)
            avg_valid = sum(p.valid_keypoints for p in result.poses) / len(result.poses)
        
        return {
            "frame_idx": result.frame_idx,
            "num_poses": len(result.poses),
            "avg_pose_confidence": avg_conf,
            "avg_valid_keypoints": avg_valid,
            "inference_time_ms": result.inference_time_ms,
            "fps": 1000.0 / result.inference_time_ms if result.inference_time_ms > 0 else 0
        }
    
    def reset_track(self, track_id: int):
        """Reset smoother state for a track."""
        self.smoother.reset_track(track_id)


def create_pose_estimator(config_path: str = "configs/pose.yaml") -> PoseEstimator:
    """Factory function to create PoseEstimator from config."""
    return PoseEstimator(config_path)