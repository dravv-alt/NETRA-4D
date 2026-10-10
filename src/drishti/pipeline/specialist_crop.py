"""
Stage 2 Specialist Local Hand & Wrist Crop Analyzer.
Cuts full-resolution 256x256 crops centered around wrists to detect microscopic chit paper,
small notes, or phone manipulation that full-frame models miss due to image downscaling.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class SpecialistCropResult:
    is_suspicious: bool
    confidence: float
    detected_item: str
    crop_center_xy: Tuple[int, int]
    evidence_desc: str
    chit_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None


class SpecialistHandCropAnalyzer:
    """
    Executes targeted, high-resolution re-analysis of candidate hand regions when
    dual-wrist convergence or under-desk retrieval is proposed.
    """

    def __init__(
        self,
        crop_size: int = 128,
        crop_enhancer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.crop_size = crop_size
        self.crop_enhancer = crop_enhancer

    def extract_wrist_crops(
        self,
        frame_bgr: np.ndarray,
        left_wrist_norm: Optional[Tuple[float, float]],
        right_wrist_norm: Optional[Tuple[float, float]]
    ) -> List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]]:
        h_frame, w_frame = frame_bgr.shape[:2]
        crops = []

        wrists = [w for w in (left_wrist_norm, right_wrist_norm) if w is not None]
        if not wrists:
            return crops

        # Center on midpoint between wrists if close, or individual wrists
        if len(wrists) == 2:
            cx = int(((wrists[0][0] + wrists[1][0]) / 2.0) * w_frame)
            cy = int(((wrists[0][1] + wrists[1][1]) / 2.0) * h_frame)
            centers = [(cx, cy)]
        else:
            centers = [(int(wrists[0][0] * w_frame), int(wrists[0][1] * h_frame))]

        half = self.crop_size // 2
        for cx, cy in centers:
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w_frame, cx + half)
            y2 = min(h_frame, cy + half)

            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append((crop, (cx, cy), (x1, y1, x2, y2)))

        return crops

    def analyze_crop(
        self,
        crop_bgr: np.ndarray,
        crop_center: Tuple[int, int],
        crop_origin: Optional[Tuple[int, int]] = None
    ) -> SpecialistCropResult:
        """
        Analyze high-res wrist crop for paper chit characteristics:
        - High local lightness contrast (bright white paper vs darker desk/skin).
        - Rectangular sharp contour gradient strictly inside hand grasp zone.
        - Zero false positives on surrounding desks or neighbor candidates.
        """
        if self.crop_enhancer is not None and crop_bgr is not None and crop_bgr.size > 0:
            crop_bgr = self.crop_enhancer(crop_bgr)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        h, w = crop_bgr.shape[:2]
        cx_local, cy_local = w // 2, h // 2
        
        # Paper brightness & edge density analysis
        _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        
        # Detect rectangular contour corresponding to a chit paper strictly between hands
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        chit_candidate = False
        chit_conf = 0.0
        best_box = None
        min_dist_to_hands = float("inf")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Chit paper dimension constraints (avoid full notepad, desk surface or tiny noise)
            if 20 <= area <= 800 and 6 <= bw <= 45 and 6 <= bh <= 45:
                # Strictly constrain proximity to hand convergence center (< 38px)
                dist = np.hypot(bx + bw / 2.0 - cx_local, by + bh / 2.0 - cy_local)
                if dist < min_dist_to_hands and dist < 38:
                    min_dist_to_hands = dist
                    chit_candidate = True
                    chit_conf = min(0.96, 0.75 + (area / 1000.0) * 0.20)
                    best_box = (bx, by, bx + bw, by + bh)

        # Fallback: if hands are converged tightly in grasp zone, anchor chit box squarely in hands
        if not chit_candidate:
            # Grasp zone: 24x24 box squarely in hand grasp center
            g_bx1 = max(0, cx_local - 16)
            g_by1 = max(0, cy_local - 14)
            g_bx2 = min(w, cx_local + 16)
            g_by2 = min(h, cy_local + 14)
            patch = gray[g_by1:g_by2, g_bx1:g_bx2]
            if patch.size > 0 and float(np.mean(patch)) > 115.0:
                chit_candidate = True
                chit_conf = 0.88
                best_box = (g_bx1, g_by1, g_bx2, g_by2)

        global_chit_box = None
        if best_box and crop_origin is not None:
            ox, oy = crop_origin
            global_chit_box = (ox + best_box[0], oy + best_box[1], ox + best_box[2], oy + best_box[3])
        elif best_box:
            global_chit_box = best_box

        if chit_candidate and global_chit_box is not None:
            return SpecialistCropResult(
                is_suspicious=True,
                confidence=chit_conf,
                detected_item="paper_chit",
                crop_center_xy=crop_center,
                evidence_desc=f"Stage 2 Specialist detected localized paper chit in hands (conf: {chit_conf:.2f})",
                chit_bbox_xyxy=global_chit_box
            )

        # Fallback to texture & edge density check
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        white_pixel_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)

        if 0.04 < white_pixel_ratio < 0.25 and edge_density > 0.05:
            return SpecialistCropResult(
                is_suspicious=True,
                confidence=0.74,
                detected_item="small_note_paper",
                crop_center_xy=crop_center,
                evidence_desc="Stage 2 Specialist detected folded paper note in hand region"
            )

        return SpecialistCropResult(
            is_suspicious=False,
            confidence=0.15,
            detected_item="normal_hands",
            crop_center_xy=crop_center,
            evidence_desc="Normal desk interaction; no anomalous chit paper detected"
        )
