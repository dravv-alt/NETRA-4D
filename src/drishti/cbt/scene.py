"""Validated semantic workstation maps and stable candidate binding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from datetime import datetime, timezone
from hashlib import sha256
import json

import numpy as np
from pydantic import BaseModel, Field, model_validator
import yaml


class Polygon(BaseModel):
    points: list[tuple[float, float]] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_normalized(self):
        if any(not 0.0 <= coordinate <= 1.0 for point in self.points for coordinate in point):
            raise ValueError("CBT scene polygons must use normalized 0-1 coordinates")
        if abs(self.signed_area()) < 1e-6:
            raise ValueError("CBT scene polygon has zero area")
        return self

    def signed_area(self) -> float:
        return 0.5 * sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(self.points, self.points[1:] + self.points[:1])
        )

    @property
    def center(self) -> tuple[float, float]:
        return tuple(float(value) for value in np.mean(np.asarray(self.points), axis=0))

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        inside = False
        previous = self.points[-1]
        for current in self.points:
            x1, y1 = previous
            x2, y2 = current
            if (y1 > y) != (y2 > y):
                crossing_x = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
                if x < crossing_x:
                    inside = not inside
            previous = current
        return inside

    def contains_box_center(self, box: tuple[float, float, float, float]) -> bool:
        x1, y1, x2, y2 = box
        return self.contains(((x1 + x2) / 2.0, (y1 + y2) / 2.0))


class Workstation(BaseModel):
    workstation_id: str = Field(min_length=1)
    seat: Polygon
    own_monitor: Polygon
    candidate_body: Polygon | None = None
    keyboard: Polygon | None = None
    mouse: Polygon | None = None
    desk: Polygon | None = None
    under_desk: Polygon | None = None
    left_neighbour_id: str | None = None
    right_neighbour_id: str | None = None

    @property
    def neighbour_ids(self) -> tuple[str, ...]:
        return tuple(value for value in (self.left_neighbour_id, self.right_neighbour_id) if value)


class CalibrationProvenance(BaseModel):
    source_video: str
    source_frame_index: int = Field(ge=0)
    source_frame_sha256: str
    calibrated_by: str
    calibrated_at_utc: str
    method: str = "human_roi_calibration"


class CBTSceneMap(BaseModel):
    schema_version: str = "1.0.0"
    calibration_status: Literal["VALIDATED", "AUTOMATIC"] = "VALIDATED"
    camera_id: str
    video_id: str | None = None
    reference_width: int = Field(gt=0)
    reference_height: int = Field(gt=0)
    workstations: list[Workstation] = Field(min_length=1)
    aisle_regions: list[Polygon] = Field(default_factory=list)
    blind_regions: list[Polygon] = Field(default_factory=list)
    provenance: CalibrationProvenance

    @model_validator(mode="after")
    def validate_graph(self):
        identifiers = [station.workstation_id for station in self.workstations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("workstation_id values must be unique")
        known = set(identifiers)
        unknown = {
            neighbour for station in self.workstations for neighbour in station.neighbour_ids
            if neighbour not in known
        }
        if unknown:
            raise ValueError(f"Unknown neighbouring workstations: {sorted(unknown)}")
        return self

    def by_id(self) -> dict[str, Workstation]:
        return {station.workstation_id: station for station in self.workstations}

    def validate_frame_geometry(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Runtime did not expose valid frame geometry: {width}x{height}")
        reference_ratio = self.reference_width / self.reference_height
        actual_ratio = width / height
        if abs(actual_ratio - reference_ratio) / reference_ratio > 0.01:
            raise ValueError(
                f"Frame aspect ratio {width}x{height} does not match calibrated "
                f"geometry {self.reference_width}x{self.reference_height}"
            )


def load_scene_map(path: str | Path) -> CBTSceneMap:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CBT workstation map does not exist: {source}")
    return CBTSceneMap.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CandidateBinding:
    candidate_id: int
    workstation_id: str | None
    confidence: float
    method: str


class CandidateWorkstationBinder:
    """Bind tracked people to calibrated seats using temporal hysteresis."""

    def __init__(self, scene: CBTSceneMap, *, maximum_center_distance: float = 0.20):
        self.scene = scene
        self.maximum_center_distance = maximum_center_distance
        self._previous: dict[int, CandidateBinding] = {}

    @staticmethod
    def _anchor(person: Any) -> tuple[float, float]:
        x1, _, x2, y2 = person.bbox_xyxy_norm
        ankles = []
        for name in ("left_ankle", "right_ankle"):
            point = (person.keypoints_2d or {}).get(name)
            if point is not None and point.confidence >= 0.30:
                ankles.append((float(point.x), float(point.y)))
        return tuple(float(value) for value in np.mean(ankles, axis=0)) if ankles else ((x1 + x2) / 2.0, y2)

    def bind(self, people: list[Any]) -> dict[int, CandidateBinding]:
        bindings: dict[int, CandidateBinding] = {}
        for person in people:
            candidate_id = int(person.entity_local_id)
            anchor = self._anchor(person)
            containing = [station for station in self.scene.workstations if station.seat.contains(anchor)]
            if containing:
                station = min(containing, key=lambda item: np.linalg.norm(np.asarray(anchor) - np.asarray(item.seat.center)))
                binding = CandidateBinding(candidate_id, station.workstation_id, 1.0, "foot_or_bottom_anchor_inside_seat")
            else:
                nearest = min(
                    self.scene.workstations,
                    key=lambda item: np.linalg.norm(np.asarray(anchor) - np.asarray(item.seat.center)),
                )
                distance = float(np.linalg.norm(np.asarray(anchor) - np.asarray(nearest.seat.center)))
                if distance <= self.maximum_center_distance:
                    binding = CandidateBinding(
                        candidate_id, nearest.workstation_id,
                        max(0.25, 1.0 - distance / self.maximum_center_distance),
                        "nearest_calibrated_seat",
                    )
                else:
                    previous = self._previous.get(candidate_id)
                    if previous and previous.workstation_id:
                        prior_station = self.scene.by_id()[previous.workstation_id]
                        prior_distance = float(np.linalg.norm(np.asarray(anchor) - np.asarray(prior_station.seat.center)))
                        if prior_distance <= self.maximum_center_distance * 1.25:
                            binding = CandidateBinding(candidate_id, previous.workstation_id, previous.confidence * 0.8, "temporal_binding_hysteresis")
                        else:
                            binding = CandidateBinding(candidate_id, None, 0.0, "unassigned")
                    else:
                        binding = CandidateBinding(candidate_id, None, 0.0, "unassigned")
            bindings[candidate_id] = binding
        self._previous = bindings
        return bindings


@dataclass
class _SpatialCluster:
    centers: list[tuple[float, float]]
    boxes: list[tuple[float, float, float, float]]

    @property
    def center(self) -> tuple[float, float]:
        return tuple(float(value) for value in np.median(np.asarray(self.centers), axis=0))

    @property
    def box(self) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in np.median(np.asarray(self.boxes), axis=0))


def _rectangle(x1: float, y1: float, x2: float, y2: float) -> Polygon:
    x1, y1, x2, y2 = (
        float(np.clip(x1, 0.0, 1.0)), float(np.clip(y1, 0.0, 1.0)),
        float(np.clip(x2, 0.0, 1.0)), float(np.clip(y2, 0.0, 1.0)),
    )
    if x2 - x1 < 0.01:
        x2 = min(1.0, x1 + 0.01)
    if y2 - y1 < 0.01:
        y2 = min(1.0, y1 + 0.01)
    return Polygon(points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


class AutomaticWorkstationCalibrator:
    """Build a model-assisted workstation map from stable real observations."""

    def __init__(
        self, camera_id: str, video_id: str, reference_width: int, reference_height: int,
        *, warmup_sec: float = 8.0, minimum_observations: int = 6,
    ):
        self.camera_id = camera_id
        self.video_id = video_id
        self.reference_width = reference_width
        self.reference_height = reference_height
        self.warmup_sec = warmup_sec
        self.minimum_observations = minimum_observations
        self.started_at_sec: float | None = None
        self.people: list[_SpatialCluster] = []
        self.monitors: list[_SpatialCluster] = []
        self.frames_observed = 0

    @staticmethod
    def _add(clusters: list[_SpatialCluster], center, box, maximum_distance: float) -> None:
        nearest = min(
            clusters,
            key=lambda cluster: np.linalg.norm(np.asarray(center) - np.asarray(cluster.center)),
            default=None,
        )
        if nearest is not None and np.linalg.norm(np.asarray(center) - np.asarray(nearest.center)) <= maximum_distance:
            nearest.centers.append(center)
            nearest.boxes.append(box)
        else:
            clusters.append(_SpatialCluster([center], [box]))

    def observe(self, timestamp_sec: float, people: list[Any], objects: list[Any]) -> None:
        self.started_at_sec = timestamp_sec if self.started_at_sec is None else self.started_at_sec
        self.frames_observed += 1
        for person in people:
            x1, y1, x2, y2 = (float(value) for value in person.bbox_xyxy_norm)
            center = ((x1 + x2) / 2.0, y1 + 0.78 * (y2 - y1))
            self._add(self.people, center, (x1, y1, x2, y2), 0.10)
        for detected in objects:
            raw_class = str(detected.metadata.get("raw_model_class", "")).lower()
            if raw_class not in {"laptop", "tv", "monitor"}:
                continue
            x1, y1, x2, y2 = (float(value) for value in detected.bbox_xyxy_norm)
            self._add(self.monitors, ((x1 + x2) / 2.0, (y1 + y2) / 2.0), (x1, y1, x2, y2), 0.07)

    def ready(self, timestamp_sec: float) -> bool:
        return self.started_at_sec is not None and timestamp_sec - self.started_at_sec >= self.warmup_sec

    def finalize(self) -> CBTSceneMap | None:
        people = [cluster for cluster in self.people if len(cluster.boxes) >= self.minimum_observations]
        monitors = [cluster for cluster in self.monitors if len(cluster.boxes) >= self.minimum_observations]
        if not people or not monitors:
            return None
        candidates = []
        available = set(range(len(monitors)))
        for person in sorted(people, key=lambda cluster: (cluster.center[1], cluster.center[0])):
            eligible = [index for index in available if monitors[index].center[1] <= person.center[1] + 0.08]
            if not eligible:
                continue
            monitor_index = min(
                eligible,
                key=lambda index: abs(monitors[index].center[0] - person.center[0]) + 0.35 * abs(monitors[index].center[1] - person.center[1]),
            )
            available.remove(monitor_index)
            monitor = monitors[monitor_index]
            px1, py1, px2, py2 = person.box
            mx1, my1, mx2, my2 = monitor.box
            identifier = f"AUTO_WS_{len(candidates) + 1:02d}"
            candidates.append(Workstation(
                workstation_id=identifier,
                seat=_rectangle(px1 - 0.04, py1 + 0.45 * (py2 - py1), px2 + 0.04, py2 + 0.08),
                candidate_body=_rectangle(px1, py1, px2, py2),
                own_monitor=_rectangle(mx1, my1, mx2, my2),
                desk=_rectangle(min(px1, mx1) - 0.02, min(my2, py1 + 0.30 * (py2 - py1)), max(px2, mx2) + 0.02, py1 + 0.62 * (py2 - py1)),
                under_desk=_rectangle(px1 - 0.04, py1 + 0.58 * (py2 - py1), px2 + 0.04, py2 + 0.10),
            ))
        if not candidates:
            return None

        # Only link spatially adjacent monitors in the same approximate row.
        rows: list[list[Workstation]] = []
        for station in sorted(candidates, key=lambda item: item.own_monitor.center[1]):
            row = next((item for item in rows if abs(item[0].own_monitor.center[1] - station.own_monitor.center[1]) <= 0.10), None)
            if row is None:
                rows.append([station])
            else:
                row.append(station)
        linked = []
        for row in rows:
            ordered = sorted(row, key=lambda item: item.own_monitor.center[0])
            for index, station in enumerate(ordered):
                payload = station.model_dump()
                payload["left_neighbour_id"] = ordered[index - 1].workstation_id if index else None
                payload["right_neighbour_id"] = ordered[index + 1].workstation_id if index + 1 < len(ordered) else None
                linked.append(Workstation.model_validate(payload))
        digest = sha256(json.dumps([
            {"person": cluster.box, "count": len(cluster.boxes)} for cluster in people
        ] + [
            {"monitor": cluster.box, "count": len(cluster.boxes)} for cluster in monitors
        ], sort_keys=True).encode("utf-8")).hexdigest()
        return CBTSceneMap(
            calibration_status="AUTOMATIC", camera_id=self.camera_id, video_id=self.video_id,
            reference_width=self.reference_width, reference_height=self.reference_height,
            workstations=linked,
            provenance=CalibrationProvenance(
                source_video=self.video_id, source_frame_index=0,
                source_frame_sha256=digest, calibrated_by="AutomaticWorkstationCalibrator",
                calibrated_at_utc=datetime.now(timezone.utc).isoformat(),
                method="model_assisted_warmup_from_real_people_and_monitor_observations",
            ),
        )
