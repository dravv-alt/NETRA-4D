"""
YOLO26-Pose SOTA Perception & Multi-Object Tracking Provider (Anuj Branch).
Implements End-to-End NMS-Free 17-Keypoint Human Pose Estimation + ByteTrack / BoT-SORT.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from .base import PerceptionProvider
from ..schemas import PersonObservation, ObjectObservation, Keypoint, ComponentQuality, QualityEnum

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class YOLO26PoseProvider(PerceptionProvider):
    def __init__(
        self,
        model_name: str = "yolo26n-pose.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.50,
        tracker: str = "bytetrack.yaml",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.tracker = tracker
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None

    def start(self):
        try:
            from ultralytics import YOLO
            try:
                self.model = YOLO(self.model_name)
            except Exception:
                # Fallback to local or ultralytics available weights
                self.model = YOLO("yolov8n-pose.pt")
        except ImportError:
            raise ImportError("Please install ultralytics (`pip install ultralytics`)")

    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        frame_bgr: np.ndarray,
        camera_id: str = "CAM01",
        video_id: str = "VIDEO_01"
    ) -> Tuple[List[PersonObservation], List[ObjectObservation]]:
        if self.model is None:
            self.start()

        h, w = frame_bgr.shape[:2]
        
        # Run tracking inference
        results = self.model.track(
            frame_bgr,
            persist=True,
            tracker=self.tracker,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False
        )

        person_observations: List[PersonObservation] = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else list(range(len(boxes_xyxy)))

            kpts_data = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

            for i, (box, conf, tid) in enumerate(zip(boxes_xyxy, confs, track_ids)):
                x1, y1, x2, y2 = box
                norm_box = (x1 / w, y1 / h, x2 / w, y2 / h)

                keypoints_dict: Dict[str, Keypoint] = {}
                head_confs = []
                torso_confs = []
                hand_confs = []

                if kpts_data is not None and i < len(kpts_data):
                    kpts = kpts_data[i]
                    for k_idx, (kx, ky, k_conf) in enumerate(kpts):
                        k_name = COCO_KEYPOINT_NAMES[k_idx] if k_idx < len(COCO_KEYPOINT_NAMES) else f"kpt_{k_idx}"
                        keypoints_dict[k_name] = Keypoint(
                            name=k_name,
                            x=float(kx / w),
                            y=float(ky / h),
                            confidence=float(k_conf)
                        )

                        if k_idx <= 4:
                            head_confs.append(k_conf)
                        elif k_idx in [5, 6, 11, 12]:
                            torso_confs.append(k_conf)
                        elif k_idx in [9, 10]:
                            hand_confs.append(k_conf)

                # Quality estimation
                h_qual = QualityEnum.GOOD if np.mean(head_confs or [0]) > 0.40 else QualityEnum.POOR
                t_qual = QualityEnum.GOOD if np.mean(torso_confs or [0]) > 0.50 else QualityEnum.POOR
                w_qual = QualityEnum.GOOD if np.mean(hand_confs or [0]) > 0.35 else QualityEnum.POOR
                tr_qual = QualityEnum.GOOD if conf > 0.40 else QualityEnum.POOR

                quality = ComponentQuality(
                    head=h_qual,
                    torso=t_qual,
                    hands=w_qual,
                    track=tr_qual
                )

                obs = PersonObservation(
                    provider="yolo26",
                    camera_id=camera_id,
                    video_id=video_id,
                    frame_index=frame_idx,
                    timestamp_sec=timestamp_sec,
                    entity_local_id=int(tid),
                    bbox_xyxy_px=(float(x1), float(y1), float(x2), float(y2)),
                    bbox_xyxy_norm=norm_box,
                    detection_confidence=float(conf),
                    keypoints_2d=keypoints_dict,
                    quality=quality,
                    metadata={"tracker": self.tracker}
                )
                person_observations.append(obs)

        return person_observations, []

    def close(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
