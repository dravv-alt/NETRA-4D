"""
Camera Profile & Semantic Context Manager for Drishti-4D.
Loads camera calibration profiles, parses seat polygons & zones,
and resolves directional targets (own monitor vs board vs neighbour seat).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import numpy as np
from ..schemas import CameraProfile, SeatProfile


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """Ray casting algorithm for checking if point [x, y] is inside a normalized polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def polygon_centroid(polygon: List[List[float]]) -> Tuple[float, float]:
    pts = np.array(polygon)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


class CameraContextManager:
    def __init__(self, config_path: Optional[str] = None):
        self.profile: Optional[CameraProfile] = None
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> CameraProfile:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Camera config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        camera_id = raw.get("camera_id", path.stem)
        res = raw.get("resolution", {})
        width = res.get("width", 1280)
        height = res.get("height", 720)

        seats: Dict[str, SeatProfile] = {}
        raw_seats = raw.get("seats", {})
        raw_neighbors = raw.get("neighbors", {})

        for seat_id, seat_data in raw_seats.items():
            poly = seat_data.get("polygon", [])
            mon_poly = seat_data.get("monitor_polygon", None)
            mouse_poly = seat_data.get("mouse_region", None)
            lap_poly = seat_data.get("lap_region", None)
            keyboard_poly = seat_data.get("keyboard_region", None)

            # Neighbor relations
            nbrs = raw_neighbors.get(seat_id, {})
            if isinstance(nbrs, list):
                nbr_dict = {f"n_{i}": n for i, n in enumerate(nbrs)}
            elif isinstance(nbrs, dict):
                nbr_dict = nbrs
            else:
                nbr_dict = {}

            seats[seat_id] = SeatProfile(
                seat_id=seat_id,
                polygon=poly,
                monitor_polygon=mon_poly,
                mouse_region=mouse_poly,
                lap_region=lap_poly,
                keyboard_region=keyboard_poly,
                neighbors=nbr_dict
            )

        zones: Dict[str, Optional[List[List[float]]]] = {}
        for z_name, z_data in raw.get("zones", {}).items():
            if isinstance(z_data, list):
                zones[z_name] = z_data
            elif isinstance(z_data, dict):
                zones[z_name] = z_data.get("polygon", None)

        blind_spots = raw.get("blind_spots", [])

        self.profile = CameraProfile(
            camera_id=camera_id,
            resolution=(width, height),
            seats=seats,
            zones=zones,
            blind_spots=blind_spots
        )
        return self.profile

    def resolve_target(self, seat_id: Optional[str], gaze_ray_normalized: Tuple[float, float]) -> Tuple[str, Optional[str]]:
        """
        Resolves where a student is looking:
        Returns (target_type, target_id)
        e.g. ("OWN_MONITOR", "MONITOR"), ("BOARD", "BOARD"), ("NEIGHBOUR_SEAT", "A02"), ("UNKNOWN", None)
        """
        if not self.profile:
            return "UNKNOWN", None

        # 1. Check if looking at board
        board_poly = self.profile.zones.get("board")
        if board_poly and point_in_polygon(gaze_ray_normalized, board_poly):
            return "BOARD", "BOARD"

        # 2. Check if looking at invigilator zone
        invig_poly = self.profile.zones.get("invigilator")
        if invig_poly and point_in_polygon(gaze_ray_normalized, invig_poly):
            return "INVIGILATOR", "INVIGILATOR"

        # 3. Check own seat / monitor
        if seat_id and seat_id in self.profile.seats:
            own_seat = self.profile.seats[seat_id]
            if own_seat.monitor_polygon and point_in_polygon(gaze_ray_normalized, own_seat.monitor_polygon):
                return "OWN_MONITOR", "MONITOR"
            if point_in_polygon(gaze_ray_normalized, own_seat.polygon):
                return "OWN_DESK", "OWN_DESK"

        # 4. Check neighbouring seats
        for s_id, s_prof in self.profile.seats.items():
            if s_id != seat_id and point_in_polygon(gaze_ray_normalized, s_prof.polygon):
                return "NEIGHBOUR_SEAT", s_id

        return "GENERAL_AREA", None
