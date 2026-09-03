"""Semantic Context and Zone Engine for Examination Environments.

Conforms to docs/ARCHITECTURE.md Layer 3: Semantic Context Layer.
Defines and evaluates spatial semantics in normalized [0.0, 1.0] coordinates:
- DESK: Candidate's own assigned workstation
- NEIGHBOUR: Adjacent candidate workstations (left, right, front)
- BOARD: Classroom board, podium, or wall clock zone
- INVIGILATOR: Dynamically updated zone around proctor's active trajectory
- ENTRANCE / AISLE: Corridors and pathways
- BLIND_SPOT: Under-desk, lap, and under-partition occlusion zones

Works uniformly across both:
1. CBT Labs (pre-calibrated polygonal workstations)
2. Large Classrooms (auto-inferred from seat anchor layout during warmup)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class SemanticZoneType(str, Enum):
    DESK = "DESK"
    NEIGHBOUR = "NEIGHBOUR"
    BOARD = "BOARD"
    INVIGILATOR = "INVIGILATOR"
    AISLE = "AISLE"
    BLIND_SPOT = "BLIND_SPOT"
    UNKNOWN = "UNKNOWN"


@dataclass
class NormalizedPolygonZone:
    """Arbitrary 2D polygonal zone in normalized [0.0, 1.0] coordinates."""
    zone_id: str
    zone_type: SemanticZoneType
    points: List[Tuple[float, float]]  # [(x1, y1), (x2, y2), ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        if not self.points:
            return (0.5, 0.5)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (float(np.mean(xs)), float(np.mean(ys)))

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Standard ray casting algorithm for point-in-polygon test."""
        x, y = point
        n = len(self.points)
        inside = False
        p1x, p1y = self.points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


def box_to_polygon(
    x1: float, y1: float, x2: float, y2: float,
    zone_id: str, zone_type: SemanticZoneType
) -> NormalizedPolygonZone:
    return NormalizedPolygonZone(
        zone_id=zone_id,
        zone_type=zone_type,
        points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    )


class SemanticZoneEngine:
    """Manages spatial examination context and evaluates spatial intersections."""

    def __init__(self):
        self.zones: Dict[str, NormalizedPolygonZone] = {}
        self.candidate_desk_zones: Dict[str, NormalizedPolygonZone] = {}
        self.neighbour_mappings: Dict[str, List[str]] = {}  # candidate_id -> list of neighbour candidate_ids
        self.invigilator_position: Optional[Tuple[float, float]] = None
        self.invigilator_zone_radius: float = 0.15

    def add_zone(self, zone: NormalizedPolygonZone) -> None:
        self.zones[zone.zone_id] = zone

    def register_candidate_desk(
        self,
        candidate_id: str,
        desk_box_norm: Tuple[float, float, float, float]
    ) -> None:
        """Register a candidate's primary desk area."""
        x1, y1, x2, y2 = desk_box_norm
        zone = box_to_polygon(x1, y1, x2, y2, f"desk_{candidate_id}", SemanticZoneType.DESK)
        zone.metadata["candidate_id"] = candidate_id
        self.candidate_desk_zones[candidate_id] = zone
        self.zones[zone.zone_id] = zone

    def auto_infer_neighbours(self, max_lateral_dist: float = 0.25) -> None:
        """Dynamically infer neighbour candidates by lateral seat proximity."""
        cand_centers = {
            cid: zone.center for cid, zone in self.candidate_desk_zones.items()
        }
        for cid, (cx, cy) in cand_centers.items():
            neighbours = []
            for other_id, (ox, oy) in cand_centers.items():
                if cid == other_id:
                    continue
                # Check horizontal alignment within same seating row (small dy, moderate dx)
                dy = abs(cy - oy)
                dx = abs(cx - ox)
                if dy < 0.12 and dx <= max_lateral_dist:
                    neighbours.append(other_id)
            self.neighbour_mappings[cid] = neighbours

    def update_invigilator_position(self, center_norm: Optional[Tuple[float, float]]) -> None:
        """Continuously update the moving invigilator's spatial context zone."""
        self.invigilator_position = center_norm
        if center_norm is not None:
            cx, cy = center_norm
            r = self.invigilator_zone_radius
            invig_zone = box_to_polygon(
                max(0.0, cx - r), max(0.0, cy - r),
                min(1.0, cx + r), min(1.0, cy + r),
                "dynamic_invigilator_zone",
                SemanticZoneType.INVIGILATOR
            )
            self.zones["dynamic_invigilator_zone"] = invig_zone
        elif "dynamic_invigilator_zone" in self.zones:
            del self.zones["dynamic_invigilator_zone"]

    def identify_point_zone(self, point: Tuple[float, float]) -> SemanticZoneType:
        """Find the semantic zone containing the given normalized point."""
        for zone in self.zones.values():
            if zone.contains_point(point):
                return zone.zone_type
        return SemanticZoneType.UNKNOWN

    def is_invigilator_near_candidate(
        self,
        candidate_id: str,
        proximity_threshold: float = 0.22
    ) -> bool:
        """Determine if proctor is physically near a candidate's desk."""
        if not self.invigilator_position:
            return False
        desk_zone = self.candidate_desk_zones.get(candidate_id)
        if not desk_zone:
            return False
        dcx, dcy = desk_zone.center
        icx, icy = self.invigilator_position
        dist = float(np.hypot(dcx - icx, dcy - icy))
        return dist <= proximity_threshold

    def is_gaze_aligned_with_neighbour(
        self,
        candidate_id: str,
        gaze_direction_str: str
    ) -> Tuple[bool, Optional[str]]:
        """Verify if candidate gaze matches an actual adjacent neighbour workstation."""
        desk = self.candidate_desk_zones.get(candidate_id)
        if not desk or gaze_direction_str not in ("LEFT", "RIGHT"):
            return False, None

        cx, _ = desk.center
        neighbours = self.neighbour_mappings.get(candidate_id, [])

        for nid in neighbours:
            n_desk = self.candidate_desk_zones.get(nid)
            if not n_desk:
                continue
            ncx, _ = n_desk.center
            if gaze_direction_str == "LEFT" and ncx < cx:
                return True, nid
            elif gaze_direction_str == "RIGHT" and ncx > cx:
                return True, nid

        return False, None

    def is_hand_crossing_boundary(
        self,
        candidate_id: str,
        hand_point_norm: Tuple[float, float]
    ) -> Tuple[bool, Optional[str]]:
        """Check if candidate's wrist or palm crosses into a neighbour's desk area."""
        own_desk = self.candidate_desk_zones.get(candidate_id)
        if not own_desk:
            return False, None

        # If inside own desk, no crossing
        if own_desk.contains_point(hand_point_norm):
            return False, None

        # Check if point falls inside any neighbour's desk
        for other_id, other_desk in self.candidate_desk_zones.items():
            if other_id != candidate_id and other_desk.contains_point(hand_point_norm):
                return True, other_id

        return False, None
