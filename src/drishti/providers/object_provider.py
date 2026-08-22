"""
Object Detection & Disambiguation Engine for Drishti-4D (Om Branch).
Implements Section 10A of the specification:
Distinguishes genuine PHONE interactions from static MONITORS, computer MICE, and KEYBOARDS.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from .base import PerceptionProvider
from ..schemas import PersonObservation, ObjectObservation, ObjectClassEnum, CameraProfile
from ..context.camera_profile import point_in_polygon


class ObjectDisambiguationProvider(PerceptionProvider):
    def __init__(
        self,
        camera_profile: Optional[CameraProfile] = None,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.25
    ):
        self.camera_profile = camera_profile
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.model = None

        # Static workstation memory: candidate_key -> history
        self.candidate_history: Dict[str, Dict[str, Any]] = {}

    def start(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
        except ImportError:
            self.model = None

    def disambiguate_candidate(
        self,
        raw_class: str,
        bbox_norm: Tuple[float, float, float, float],
        conf: float,
        timestamp_sec: float,
        person_obs_list: List[PersonObservation]
    ) -> Tuple[ObjectClassEnum, float, Optional[int], Optional[str]]:
        """
        Disambiguates a candidate box using workstation geometry, lap regions, and wrist distances.
        Returns: (resolved_class, resolved_confidence, associated_entity_id, associated_seat_id)
        """
        x1, y1, x2, y2 = bbox_norm
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        c_pt = (cx, cy)

        associated_seat = None
        associated_entity = None
        min_wrist_dist = 999.0

        # 1. Match with Seat Profile
        if self.camera_profile:
            for s_id, s_prof in self.camera_profile.seats.items():
                if point_in_polygon(c_pt, s_prof.polygon):
                    associated_seat = s_id
                    break

        # 2. Match with Person Observation & Wrist Proximity
        for p in person_obs_list:
            px1, py1, px2, py2 = p.bbox_xyxy_norm
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                associated_entity = p.entity_local_id
                if not associated_seat:
                    associated_seat = p.seat_id

            # Calculate distance to left/right wrists
            if p.keypoints_2d:
                lw = p.keypoints_2d.get("left_wrist")
                rw = p.keypoints_2d.get("right_wrist")
                for w in [lw, rw]:
                    if w and w.confidence > 0.25:
                        d = np.sqrt((w.x - cx)**2 + (w.y - cy)**2)
                        if d < min_wrist_dist:
                            min_wrist_dist = d
                            associated_entity = p.entity_local_id

        # 3. Apply Section 10A Disambiguation Filters
        if self.camera_profile and associated_seat in self.camera_profile.seats:
            seat_prof = self.camera_profile.seats[associated_seat]

            # A. Monitor Suppression
            if seat_prof.monitor_polygon and point_in_polygon(c_pt, seat_prof.monitor_polygon):
                if min_wrist_dist > 0.15:
                    return ObjectClassEnum.MONITOR, conf * 0.95, associated_entity, associated_seat

            # B. Mouse Suppression
            if seat_prof.mouse_region and point_in_polygon(c_pt, seat_prof.mouse_region):
                if min_wrist_dist < 0.10:
                    return ObjectClassEnum.MOUSE, conf * 0.90, associated_entity, associated_seat

            # C. Lap Region Boost (Phone Prior)
            if seat_prof.lap_region and point_in_polygon(c_pt, seat_prof.lap_region):
                # Strong Phone Indicator
                phone_conf = min(0.98, conf * 1.25 + (0.2 if min_wrist_dist < 0.12 else 0.0))
                return ObjectClassEnum.MOBILE_PHONE, phone_conf, associated_entity, associated_seat

        # 4. General Class Matching
        if raw_class in ["cell phone", "phone", "mobile"]:
            if min_wrist_dist < 0.15:
                return ObjectClassEnum.MOBILE_PHONE, conf, associated_entity, associated_seat
            else:
                return ObjectClassEnum.UNCERTAIN_OBJECT, conf * 0.5, associated_entity, associated_seat
        elif raw_class in ["mouse"]:
            return ObjectClassEnum.MOUSE, conf, associated_entity, associated_seat
        elif raw_class in ["tv", "monitor", "screen", "laptop"]:
            return ObjectClassEnum.MONITOR, conf, associated_entity, associated_seat
        elif "paper" in raw_class or "book" in raw_class:
            return ObjectClassEnum.PAPER_CHIT, conf, associated_entity, associated_seat

        return ObjectClassEnum.OTHER_PROHIBITED_OBJECT, conf * 0.5, associated_entity, associated_seat

    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        frame_bgr: np.ndarray,
        camera_id: str = "CAM01",
        video_id: str = "VIDEO_01",
        person_obs_list: Optional[List[PersonObservation]] = None
    ) -> Tuple[List[PersonObservation], List[ObjectObservation]]:
        if self.model is None:
            self.start()

        h, w = frame_bgr.shape[:2]
        object_observations: List[ObjectObservation] = []
        persons = person_obs_list or []

        if self.model:
            results = self.model(frame_bgr, verbose=False, conf=self.conf_thresh)
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.int().cpu().numpy()
                names = self.model.names

                for box, conf, cls_id in zip(boxes, confs, classes):
                    raw_cls_name = names.get(cls_id, "unknown")
                    if raw_cls_name in ["person"]:
                        continue  # Handled by pose provider

                    x1, y1, x2, y2 = box
                    norm_box = (x1 / w, y1 / h, x2 / w, y2 / h)

                    res_class, res_conf, assoc_id, assoc_seat = self.disambiguate_candidate(
                        raw_cls_name, norm_box, float(conf), timestamp_sec, persons
                    )

                    object_observations.append(ObjectObservation(
                        camera_id=camera_id,
                        video_id=video_id,
                        frame_index=frame_idx,
                        timestamp_sec=timestamp_sec,
                        object_class=res_class,
                        confidence=float(res_conf),
                        bbox_xyxy_px=(float(x1), float(y1), float(x2), float(y2)),
                        bbox_xyxy_norm=norm_box,
                        associated_entity_id=assoc_id,
                        associated_seat_id=assoc_seat,
                        association_confidence=0.85 if assoc_seat else 0.0,
                        is_static_equipment=(res_class in [ObjectClassEnum.MONITOR, ObjectClassEnum.KEYBOARD])
                    ))

        return [], object_observations

    def close(self):
        self.model = None
