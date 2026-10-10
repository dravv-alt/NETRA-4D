"""Calibrated seat-relative context for evidence interpretation.

This module intentionally separates what a monocular CCTV image can observe
(image-plane adjacency) from what requires scene calibration (student-relative
left/right and ahead/behind).  It emits relationships, never a behavioural
verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class SeatPhysicsCalibration:
    """Camera-to-seat convention, supplied only after human verification.

    ``student_left_is_image_left`` and ``student_forward_is_image_up`` are
    deliberately optional.  An omitted value means the corresponding physical
    relationship must stay UNKNOWN rather than being guessed from pixels.
    """

    calibration_id: str = "UNCALIBRATED"
    student_left_is_image_left: Optional[bool] = None
    student_forward_is_image_up: Optional[bool] = None

    @property
    def validated(self) -> bool:
        return (
            self.student_left_is_image_left is not None
            and self.student_forward_is_image_up is not None
        )


@dataclass(frozen=True)
class SeatRelation:
    source_candidate_id: str
    target_candidate_id: str
    image_plane_relation: str
    physical_relation: str
    relation_confidence: float
    provenance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_candidate_id": self.source_candidate_id,
            "target_candidate_id": self.target_candidate_id,
            "image_plane_relation": self.image_plane_relation,
            "physical_relation": self.physical_relation,
            "relation_confidence": self.relation_confidence,
            "provenance": self.provenance,
        }


class ContextualPhysicsEngine:
    """Freeze a candidate seat graph and expose calibrated relative context."""

    def __init__(self, calibration: SeatPhysicsCalibration | None = None) -> None:
        self.calibration = calibration or SeatPhysicsCalibration()
        self._anchors: Dict[str, tuple[float, float, float, float]] = {}
        self._relations: Dict[str, list[SeatRelation]] = {}
        self._frozen = False
        self._last_emitted_signature: tuple | None = None

    @property
    def frozen(self) -> bool:
        return self._frozen

    def update(self, candidates: Iterable[Any], *, freeze: bool = False) -> None:
        """Ingest immutable candidate anchors, excluding invigilators.

        Candidate state is intentionally duck-typed so this engine remains
        independent of the identity manager implementation.
        """
        if self._frozen:
            return
        for candidate_id, state in self._candidate_items(candidates):
            identity = state.identity
            if identity.is_invigilator:
                continue
            cx, cy = identity.anchor_center_norm
            bw, bh = identity.anchor_box_size_norm
            self._anchors[candidate_id] = (float(cx), float(cy), float(bw), float(bh))
        if freeze and self._anchors:
            self._relations = self._build_relations()
            self._frozen = True

    @staticmethod
    def _candidate_items(candidates: Iterable[Any]):
        return candidates.items() if hasattr(candidates, "items") else candidates

    def _build_relations(self) -> Dict[str, list[SeatRelation]]:
        result: Dict[str, list[SeatRelation]] = {candidate_id: [] for candidate_id in self._anchors}
        widths = [anchor[2] for anchor in self._anchors.values()]
        heights = [anchor[3] for anchor in self._anchors.values()]
        row_tolerance = max(0.04, float(np.median(heights)) * 0.65)
        column_tolerance = max(0.04, float(np.median(widths)) * 0.65)
        for source_id, (sx, sy, _, _) in self._anchors.items():
            for target_id, (tx, ty, _, _) in self._anchors.items():
                if source_id == target_id:
                    continue
                dx, dy = tx - sx, ty - sy
                image_relation = self._image_relation(dx, dy, row_tolerance, column_tolerance)
                physical = self._physical_relation(dx, dy, row_tolerance, column_tolerance)
                proximity = float(np.hypot(dx, dy))
                confidence = float(max(0.0, min(1.0, 1.0 - proximity / 0.50)))
                result[source_id].append(SeatRelation(
                    source_candidate_id=source_id,
                    target_candidate_id=target_id,
                    image_plane_relation=image_relation,
                    physical_relation=physical,
                    relation_confidence=confidence,
                    provenance=(
                        f"seat_anchor_graph:{self.calibration.calibration_id}"
                        if self.calibration.validated else "seat_anchor_graph:image_plane_only"
                    ),
                ))
        return result

    @staticmethod
    def _image_relation(dx: float, dy: float, row_tol: float, column_tol: float) -> str:
        if abs(dy) <= row_tol and abs(dx) > column_tol:
            return "IMAGE_RIGHT" if dx > 0 else "IMAGE_LEFT"
        if abs(dx) <= column_tol and abs(dy) > row_tol:
            return "IMAGE_BELOW" if dy > 0 else "IMAGE_ABOVE"
        return "IMAGE_DIAGONAL"

    def _physical_relation(self, dx: float, dy: float, row_tol: float, column_tol: float) -> str:
        if not self.calibration.validated:
            return "UNKNOWN_UNCALIBRATED"
        if abs(dy) <= row_tol and abs(dx) > column_tol:
            image_left = dx < 0
            student_left = image_left == self.calibration.student_left_is_image_left
            return "STUDENT_LEFT" if student_left else "STUDENT_RIGHT"
        if abs(dx) <= column_tol and abs(dy) > row_tol:
            image_forward = dy < 0
            forward = image_forward == self.calibration.student_forward_is_image_up
            return "AHEAD_OF_STUDENT" if forward else "BEHIND_STUDENT"
        return "NON_ADJACENT_OR_DIAGONAL"

    def relations_for(self, candidate_id: str) -> list[SeatRelation]:
        return list(self._relations.get(candidate_id, ()))

    def context_for(self, candidate_id: str) -> Dict[str, Any]:
        relations = self.relations_for(candidate_id)
        return {
            "seat_graph_status": "FROZEN" if self._frozen else "WARMUP",
            "calibration_status": "VALIDATED" if self.calibration.validated else "IMAGE_PLANE_ONLY",
            "calibration_id": self.calibration.calibration_id,
            "relations": [relation.to_dict() for relation in relations],
        }

    def drain_graph_snapshot(self) -> list[Dict[str, Any]]:
        """Return one immutable snapshot after freeze, avoiding frame spam."""
        if not self._frozen:
            return []
        rows = [self.context_for(candidate_id) | {"candidate_id": candidate_id}
                for candidate_id in sorted(self._anchors)]
        signature = tuple((row["candidate_id"], tuple(
            (item["target_candidate_id"], item["physical_relation"])
            for item in row["relations"]
        )) for row in rows)
        if signature == self._last_emitted_signature:
            return []
        self._last_emitted_signature = signature
        return rows
