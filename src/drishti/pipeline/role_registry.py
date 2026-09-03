"""
Immutable Role Registry and Invigilator Re-Identification.
Identifies standing/roaming invigilators during opening frames, freezes their role immutably,
and re-identifies them if they leave and re-enter the camera view so they are never
misclassified as candidates. Enforces the Invigilator Exclusion Rule.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class RoleEnum(str, Enum):
    CANDIDATE = "CANDIDATE"
    INVIGILATOR = "INVIGILATOR"
    UNKNOWN = "UNKNOWN"


@dataclass
class VisualSignature:
    """Compact color and aspect ratio signature for visual re-identification."""
    mean_color_top: Tuple[float, float, float]  # Normalized RGB mean for shirt/torso
    mean_color_bottom: Tuple[float, float, float]  # Normalized RGB mean for pants/lower
    aspect_ratio: float  # Height / Width

    def distance(self, other: VisualSignature) -> float:
        d_top = np.linalg.norm(np.array(self.mean_color_top) - np.array(other.mean_color_top))
        d_bot = np.linalg.norm(np.array(self.mean_color_bottom) - np.array(other.mean_color_bottom))
        d_ar = abs(self.aspect_ratio - other.aspect_ratio)
        return float(0.45 * d_top + 0.45 * d_bot + 0.10 * min(d_ar, 1.0))


@dataclass
class RoleEntity:
    entity_id: str
    role: RoleEnum
    first_seen_frame: int
    last_seen_frame: int
    is_immutable: bool = False
    standing_confidence: float = 0.0
    trajectory_positions: List[Tuple[float, float]] = field(default_factory=list)
    signature: Optional[VisualSignature] = None


class ImmutableRoleRegistry:
    """
    Manages exam hall role classification and enforces immutable role declarations.
    Standing and roaming individuals are identified as INVIGILATORs.
    Once finalized:
    1. Role cannot change.
    2. Invigilator leaving and returning is re-associated via visual signature and roaming pattern.
    3. Invigilator Exclusion Rule is strictly applied (tracked for context, zero cheating alerts).
    """

    def __init__(self, warmup_frames: int = 45, roaming_distance_threshold: float = 0.12):
        self.warmup_frames = warmup_frames
        self.roaming_distance_threshold = roaming_distance_threshold
        self.current_frame = 0
        self.finalized = False
        self.registry: Dict[str, RoleEntity] = {}
        self.invigilator_signatures: List[VisualSignature] = []
        self.active_tracker_to_entity: Dict[int, str] = {}

    def extract_visual_signature(self, frame_bgr: np.ndarray, bbox_xyxy: List[float]) -> VisualSignature:
        h_frame, w_frame = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)
        
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ar = h / float(w)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return VisualSignature((0, 0, 0), (0, 0, 0), ar)

        mid_y = h // 2
        top_crop = crop[:mid_y, :]
        bot_crop = crop[mid_y:, :]

        top_mean = tuple(float(v) / 255.0 for v in cv2.mean(top_crop)[:3]) if top_crop.size > 0 else (0, 0, 0)
        bot_mean = tuple(float(v) / 255.0 for v in cv2.mean(bot_crop)[:3]) if bot_crop.size > 0 else (0, 0, 0)

        return VisualSignature(top_mean, bot_mean, ar)

    def observe_frame(self, frame_idx: int, tracked_people: List[Dict[str, Any]], frame_bgr: Optional[np.ndarray] = None) -> None:
        self.current_frame = frame_idx
        w_frame, h_frame = (frame_bgr.shape[1], frame_bgr.shape[0]) if frame_bgr is not None else (1280, 720)

        for person in tracked_people:
            t_id = person["track_id"]
            bbox = person["bbox"]
            bw = max(0.01, (bbox[2] - bbox[0]) / float(w_frame))
            bh = max(0.01, (bbox[3] - bbox[1]) / float(h_frame))
            norm_center = ((bbox[0] + bbox[2]) / (2.0 * w_frame), (bbox[1] + bbox[3]) / (2.0 * h_frame))

            # Check if this person matches an existing immutable invigilator
            if t_id not in self.active_tracker_to_entity:
                matched_invigilator = None
                if self.invigilator_signatures and frame_bgr is not None:
                    sig = self.extract_visual_signature(frame_bgr, bbox)
                    for inv_sig in self.invigilator_signatures:
                        if sig.distance(inv_sig) < 0.28:
                            matched_invigilator = inv_sig
                            break

                if matched_invigilator is not None:
                    entity_id = "INVIGILATOR_01"
                    self.active_tracker_to_entity[t_id] = entity_id
                    entity = self.registry.get(entity_id)
                    if entity:
                        entity.last_seen_frame = frame_idx
                        entity.trajectory_positions.append(norm_center)
                        if not hasattr(entity, "trajectory_sizes"):
                            entity.trajectory_sizes = []
                        entity.trajectory_sizes.append((bw, bh))
                    continue

                entity_id = f"TRACK_{t_id}"
                self.active_tracker_to_entity[t_id] = entity_id

            entity_id = self.active_tracker_to_entity[t_id]
            if entity_id not in self.registry:
                sig = self.extract_visual_signature(frame_bgr, bbox) if frame_bgr is not None else None
                ent = RoleEntity(
                    entity_id=entity_id,
                    role=RoleEnum.CANDIDATE,  # Default to CANDIDATE
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                    trajectory_positions=[norm_center],
                    signature=sig
                )
                ent.trajectory_sizes = [(bw, bh)]
                self.registry[entity_id] = ent
            else:
                ent = self.registry[entity_id]
                ent.last_seen_frame = frame_idx
                ent.trajectory_positions.append(norm_center)
                if not hasattr(ent, "trajectory_sizes"):
                    ent.trajectory_sizes = []
                ent.trajectory_sizes.append((bw, bh))
                if ent.signature is None and frame_bgr is not None:
                    ent.signature = self.extract_visual_signature(frame_bgr, bbox)

        if not self.finalized and self.current_frame >= self.warmup_frames:
            self.finalize_roles()

    def finalize_roles(self) -> None:
        """
        Examine scale-invariant body locomotion and aspect ratio.
        Walking invigilators travel multiple body-lengths across aisle corridors.
        Stationary seated candidates remain physically bounded within their chair/desk zone.
        """
        for entity_id, entity in self.registry.items():
            if len(entity.trajectory_positions) >= 5:
                pts = np.array(entity.trajectory_positions)
                sizes = np.array(getattr(entity, "trajectory_sizes", [(0.1, 0.2)] * len(pts)))
                
                mean_bw = max(0.01, float(np.mean(sizes[:, 0])))
                mean_bh = max(0.01, float(np.mean(sizes[:, 1])))
                mean_ar = mean_bh / mean_bw

                # Net translation from start to end of observation
                net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
                
                # Total peak-to-peak dispersion normalized by the person's own body dimensions
                body_disp = float(np.hypot(np.ptp(pts[:, 0]) / mean_bw, np.ptp(pts[:, 1]) / mean_bh))

                # True roaming requires net translation exceeding their body height AND high normalized body dispersion AND upright standing aspect ratio
                is_standing_roamer = (net_disp > mean_bh * 1.5) and (body_disp > 2.0) and (mean_ar >= 1.85)

                if is_standing_roamer:
                    entity.role = RoleEnum.INVIGILATOR
                    entity.is_immutable = True
                    if entity.signature:
                        self.invigilator_signatures.append(entity.signature)
                else:
                    entity.role = RoleEnum.CANDIDATE
                    entity.is_immutable = True
            else:
                entity.role = RoleEnum.CANDIDATE

        self.finalized = True

    def get_role_for_track(self, track_id: int) -> RoleEnum:
        entity_id = self.active_tracker_to_entity.get(track_id)
        if not entity_id or entity_id not in self.registry:
            return RoleEnum.CANDIDATE

        ent = self.registry[entity_id]
        if ent.is_immutable and ent.role != RoleEnum.UNKNOWN:
            return ent.role

        # Dynamic scale-invariant check during warmup
        if len(ent.trajectory_positions) >= 8:
            pts = np.array(ent.trajectory_positions)
            sizes = np.array(getattr(ent, "trajectory_sizes", [(0.1, 0.2)] * len(pts)))
            mean_bw = max(0.01, float(np.mean(sizes[:, 0])))
            mean_bh = max(0.01, float(np.mean(sizes[:, 1])))
            mean_ar = mean_bh / mean_bw
            net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
            body_disp = float(np.hypot(np.ptp(pts[:, 0]) / mean_bw, np.ptp(pts[:, 1]) / mean_bh))
            if (net_disp > mean_bh * 1.5) and (body_disp > 2.0) and (mean_ar >= 1.85):
                return RoleEnum.INVIGILATOR

        return RoleEnum.CANDIDATE

    def is_invigilator(self, track_id: int) -> bool:
        return self.get_role_for_track(track_id) == RoleEnum.INVIGILATOR
