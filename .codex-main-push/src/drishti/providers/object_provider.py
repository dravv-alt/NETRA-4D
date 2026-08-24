"""Raw Ultralytics object detections with transparent geometric association.

This provider preserves the model's class and confidence. It never turns a
generic book into a paper chit, boosts a confidence, or inserts a fallback box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from .base import PerceptionProvider
from ..schemas import ObjectClassEnum, ObjectObservation, PersonObservation


_CLASS_MAP = {
    "cell phone": ObjectClassEnum.MOBILE_PHONE,
    "mouse": ObjectClassEnum.MOUSE,
    "keyboard": ObjectClassEnum.KEYBOARD,
    "tv": ObjectClassEnum.MONITOR,
    "laptop": ObjectClassEnum.MONITOR,
}


class ObjectDisambiguationProvider(PerceptionProvider):
    def __init__(
        self,
        camera_profile=None,
        model_name: str = "yolo11n.pt",
        conf_thresh: float = 0.25,
        device: Optional[str] = None,
        enable_candidate_crops: bool = False,
        phone_proposal_confidence: float = 0.10,
        crop_every: int = 2,
        enable_tiled_inference: bool = False,
        tile_every: int = 4,
        chit_model_name: Optional[str] = None,
        chit_confidence: float = 0.25,
    ):
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.enable_candidate_crops = enable_candidate_crops
        self.phone_proposal_confidence = phone_proposal_confidence
        self.crop_every = max(1, crop_every)
        self.enable_tiled_inference = enable_tiled_inference
        self.tile_every = max(1, tile_every)
        self.chit_model_name = chit_model_name
        self.chit_confidence = chit_confidence
        self.chit_model = None

    def start(self) -> None:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device!r} was requested, but CUDA is unavailable."
            )
        if self.chit_model_name:
            chit_path = Path(self.chit_model_name)
            if not chit_path.is_file():
                raise FileNotFoundError(f"Configured real chit checkpoint does not exist: {chit_path}")

        from ultralytics import YOLO

        self.model = YOLO(self.model_name)
        if self.chit_model_name:
            self.chit_model = YOLO(str(chit_path))

    @staticmethod
    def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - intersection
        return intersection / union if union else 0.0

    def _predict_raw(
        self, model: Any, image: np.ndarray, confidence: float, *,
        offset_x: int = 0, offset_y: int = 0, keep: set[str] | None = None,
        aliases: set[str] | None = None, inference_view: str = "full_frame",
    ) -> list[dict[str, Any]]:
        raw = []
        for result in model.predict(image, conf=confidence, device=self.device, verbose=False):
            if result.boxes is None:
                continue
            for box, score, class_id in zip(
                result.boxes.xyxy.detach().cpu().numpy(),
                result.boxes.conf.detach().cpu().numpy(),
                result.boxes.cls.int().detach().cpu().numpy(),
            ):
                raw_class = str(model.names[int(class_id)])
                normalized_name = raw_class.strip().lower().replace("_", " ")
                if keep is not None and raw_class not in keep:
                    continue
                if aliases is not None and normalized_name not in aliases:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box)
                raw.append({
                    "raw_class": raw_class,
                    "confidence": float(score),
                    "box": (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                    "inference_view": inference_view,
                    "is_chit_model": aliases is not None,
                })
        return raw

    @staticmethod
    def _tiles(width: int, height: int, overlap: float = 0.12) -> list[tuple[int, int, int, int]]:
        tile_width, tile_height = int(width * (0.5 + overlap / 2)), int(height * (0.5 + overlap / 2))
        return [
            (0, 0, tile_width, tile_height),
            (width - tile_width, 0, width, tile_height),
            (0, height - tile_height, tile_width, height),
            (width - tile_width, height - tile_height, width, height),
        ]

    @staticmethod
    def _associate(
        box_norm: Tuple[float, float, float, float],
        persons: List[PersonObservation],
    ) -> Tuple[Optional[int], Optional[str], float, str]:
        x1, y1, x2, y2 = box_norm
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        closest = None
        for person in persons:
            if not person.keypoints_2d:
                continue
            for name in ("left_wrist", "right_wrist"):
                point = person.keypoints_2d.get(name)
                if point is None or point.confidence < 0.25:
                    continue
                distance = float(np.linalg.norm(center - np.array([point.x, point.y])))
                if closest is None or distance < closest[0]:
                    closest = (distance, person)
        if closest is not None and closest[0] <= 0.15:
            score = max(0.0, 1.0 - closest[0] / 0.15)
            person = closest[1]
            return person.entity_local_id, person.seat_id, score, "nearest_visible_wrist"
        for person in persons:
            px1, py1, px2, py2 = person.bbox_xyxy_norm
            if px1 <= center[0] <= px2 and py1 <= center[1] <= py2:
                return person.entity_local_id, person.seat_id, 0.75, "object_center_inside_person_box"
        return None, None, 0.0, "unassociated"

    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        frame_bgr: np.ndarray,
        camera_id: str = "CAM01",
        video_id: str = "VIDEO_01",
        person_obs_list: Optional[List[PersonObservation]] = None,
    ) -> Tuple[List[PersonObservation], List[ObjectObservation]]:
        if self.model is None:
            self.start()

        height, width = frame_bgr.shape[:2]
        persons = person_obs_list or []
        raw_predictions = self._predict_raw(self.model, frame_bgr, self.conf_thresh)
        if self.enable_tiled_inference and frame_idx % self.tile_every == 0:
            for tile_index, (tx1, ty1, tx2, ty2) in enumerate(self._tiles(width, height)):
                raw_predictions.extend(self._predict_raw(
                    self.model, frame_bgr[ty1:ty2, tx1:tx2], self.phone_proposal_confidence,
                    offset_x=tx1, offset_y=ty1, keep={"cell phone"},
                    inference_view=f"tile:{tile_index}",
                ))
        if (self.enable_candidate_crops or self.chit_model is not None) and frame_idx % self.crop_every == 0:
            for person in persons:
                px1, py1, px2, py2 = (int(value) for value in person.bbox_xyxy_px)
                padding_x, padding_y = int((px2 - px1) * 0.10), int((py2 - py1) * 0.08)
                cx1, cy1 = max(0, px1 - padding_x), max(0, py1 - padding_y)
                cx2, cy2 = min(width, px2 + padding_x), min(height, py2 + padding_y)
                crop = frame_bgr[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                if self.enable_candidate_crops:
                    raw_predictions.extend(self._predict_raw(
                        self.model, crop, self.phone_proposal_confidence,
                        offset_x=cx1, offset_y=cy1, keep={"cell phone"},
                        inference_view=f"candidate_crop:{person.entity_local_id}",
                    ))
                if self.chit_model is not None:
                    raw_predictions.extend(self._predict_raw(
                        self.chit_model, crop, self.chit_confidence,
                        offset_x=cx1, offset_y=cy1,
                        aliases={"chit", "paper chit", "cheat sheet", "piece of paper", "note paper"},
                        inference_view=f"candidate_crop:{person.entity_local_id}",
                    ))

        raw_predictions.sort(key=lambda item: item["confidence"], reverse=True)
        deduplicated: list[dict[str, Any]] = []
        for prediction in raw_predictions:
            semantic = "paper_chit" if prediction["is_chit_model"] else prediction["raw_class"]
            if any(
                ("paper_chit" if other["is_chit_model"] else other["raw_class"]) == semantic
                and self._iou(prediction["box"], other["box"]) >= 0.50
                for other in deduplicated
            ):
                continue
            deduplicated.append(prediction)

        detections: List[ObjectObservation] = []
        for prediction in deduplicated:
            raw_class = prediction["raw_class"]
            if raw_class == "person":
                continue
            confidence = prediction["confidence"]
            x1, y1, x2, y2 = prediction["box"]
            normalized = (x1 / width, y1 / height, x2 / width, y2 / height)
            entity_id, seat_id, assoc_score, assoc_method = self._associate(normalized, persons)
            object_class = ObjectClassEnum.PAPER_CHIT if prediction["is_chit_model"] else _CLASS_MAP.get(raw_class, ObjectClassEnum.UNCERTAIN_OBJECT)
            detections.append(
                ObjectObservation(
                    camera_id=camera_id,
                    video_id=video_id,
                    frame_index=frame_idx,
                    timestamp_sec=timestamp_sec,
                    object_class=object_class,
                    confidence=float(confidence),
                    bbox_xyxy_px=(x1, y1, x2, y2),
                    bbox_xyxy_norm=normalized,
                    associated_entity_id=entity_id,
                    associated_seat_id=seat_id,
                    association_confidence=assoc_score,
                    is_static_equipment=raw_class in {"tv", "laptop", "keyboard", "mouse"} and not prediction["is_chit_model"],
                    metadata={
                        "raw_model_class": raw_class,
                        "model_weight": str(Path(self.chit_model_name if prediction["is_chit_model"] else self.model_name)),
                        "device": self.device,
                        "association_method": assoc_method,
                        "inference_view": prediction["inference_view"],
                        "custom_chit_checkpoint": str(self.chit_model_name) if prediction["is_chit_model"] else None,
                    },
                )
            )
        return [], detections

    def close(self) -> None:
        self.model = None
        self.chit_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
