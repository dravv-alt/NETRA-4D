"""
Person and Object Detection for Watcher.

Uses YOLO (Ultralytics) for detecting persons and objects in video frames.
"""

import cv2
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # center x, center y


@dataclass
class DetectionResult:
    """All detections for a single frame."""
    frame_idx: int
    detections: List[Detection]
    inference_time_ms: float


class Detector:
    """
    YOLO-based person and object detector.
    
    Loads configuration from detector.yaml and provides detection interface.
    """
    
    def __init__(self, config_path: str = "configs/detector.yaml"):
        """
        Initialize detector from configuration.
        
        Args:
            config_path: Path to detector configuration YAML.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}
        self.target_class_ids: List[int] = []
        self._load_model()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self):
        """Load YOLO model and configure target classes."""
        det_config = self.config.get("detector", {})
        model_name = det_config.get("model", "yolov8n.pt")
        device = det_config.get("device", "auto")
        imgsz = det_config.get("imgsz", 640)
        half = det_config.get("half", False)
        
        logger.info(f"Loading YOLO model: {model_name} on {device}")
        self.model = YOLO(model_name)
        
        # Move model to device if specified
        if device != "auto":
            self.model.to(device)
        
        # Enable half precision if requested
        if half:
            self.model.model.half()
        
        # Get class names from model
        self.class_names = self.model.names
        
        # Map target class names to class IDs
        target_classes = det_config.get("target_classes", ["person"])
        self.target_class_ids = []
        
        for class_name in target_classes:
            # Find class ID by name
            found = False
            for class_id, name in self.class_names.items():
                if name.lower() == class_name.lower():
                    self.target_class_ids.append(class_id)
                    found = True
                    break
            if not found:
                logger.warning(f"Class '{class_name}' not found in model. Available: {list(self.class_names.values())}")
        
        logger.info(f"Target class IDs: {self.target_class_ids} ({[self.class_names.get(i) for i in self.target_class_ids]})")
    
    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> DetectionResult:
        """
        Run detection on a single frame.
        
        Args:
            frame: Input frame (BGR format).
            frame_idx: Frame index for tracking.
            
        Returns:
            DetectionResult with all detections.
        """
        import time
        start_time = time.time()
        
        det_config = self.config.get("detector", {})
        conf_thresh = det_config.get("confidence_threshold", 0.5)
        iou_thresh = det_config.get("iou_threshold", 0.45)
        imgsz = det_config.get("imgsz", 640)
        max_det = det_config.get("max_det", 300)
        
        # Run inference
        results = self.model(
            frame,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Filter by target classes
                    if self.target_class_ids and class_id not in self.target_class_ids:
                        continue
                    
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y)
                    ))
        
        return DetectionResult(
            frame_idx=frame_idx,
            detections=detections,
            inference_time_ms=inference_time_ms
        )
    
    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Draw detection boxes and labels on frame.
        
        Args:
            frame: Input frame.
            result: DetectionResult to visualize.
            
        Returns:
            Annotated frame.
        """
        output_config = self.config.get("output", {})
        draw_boxes = output_config.get("draw_boxes", True)
        draw_labels = output_config.get("draw_labels", True)
        draw_confidence = output_config.get("draw_confidence", True)
        box_thickness = output_config.get("box_thickness", 2)
        font_scale = output_config.get("font_scale", 0.6)
        
        annotated = frame.copy()
        
        if not draw_boxes:
            return annotated
        
        # Color map for different classes
        colors = {
            "person": (0, 255, 0),      # Green
            "cell phone": (0, 0, 255),   # Red
            "laptop": (255, 0, 0),       # Blue
            "book": (255, 255, 0),       # Cyan
            "chair": (255, 0, 255),      # Magenta
        }
        
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
            
            if draw_labels:
                label = det.class_name
                if draw_confidence:
                    label += f" {det.confidence:.2f}"
                
                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_h - 4),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    1
                )
        
        return annotated
    
    def get_summary(self, result: DetectionResult) -> Dict[str, Any]:
        """Get detection summary statistics."""
        person_count = sum(1 for d in result.detections if d.class_name == "person")
        object_count = len(result.detections) - person_count
        
        avg_conf = 0.0
        if result.detections:
            avg_conf = sum(d.confidence for d in result.detections) / len(result.detections)
        
        return {
            "frame_idx": result.frame_idx,
            "total_detections": len(result.detections),
            "person_count": person_count,
            "object_count": object_count,
            "avg_confidence": avg_conf,
            "inference_time_ms": result.inference_time_ms,
            "fps": 1000.0 / result.inference_time_ms if result.inference_time_ms > 0 else 0
        }


def create_detector(config_path: str = "configs/detector.yaml") -> Detector:
    """Factory function to create Detector from config."""
    return Detector(config_path)