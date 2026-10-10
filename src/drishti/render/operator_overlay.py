"""Low-clutter live-review overlay.

This renderer deliberately presents *review evidence*, not every detector
intermediate.  The verbose pose/gaze/workstation rendering remains available
through the explicit ``debug`` overlay mode in the product runner.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


CYAN = (235, 190, 25)
AMBER = (20, 180, 255)
RED = (65, 65, 245)
PANEL = (20, 24, 30)
WHITE = (245, 245, 245)


def _box(value: Any) -> tuple[int, int, int, int]:
    box = getattr(value, "bbox_xyxy_px", None)
    if box is None and isinstance(value, dict):
        box = value.get("bbox_xyxy_px", value.get("bbox", (0, 0, 0, 0)))
    return tuple(int(v) for v in (box or (0, 0, 0, 0)))


def _name(value: Any) -> str:
    object_class = getattr(value, "object_class", None)
    if object_class is not None:
        return str(getattr(object_class, "value", object_class)).lower()
    if isinstance(value, dict):
        return str(value.get("object_class", value.get("class", ""))).lower()
    return ""


def _person_id(person: Any) -> str:
    entity_id = getattr(person, "entity_local_id", None)
    if entity_id is None and isinstance(person, dict):
        entity_id = person.get("entity_local_id", person.get("track_id"))
    return str(entity_id) if entity_id is not None else "?"


def _event_label(label: str) -> str:
    """Use short operator wording without implying misconduct."""
    upper = label.upper()
    if "PHONE" in upper:
        return "PHONE REVIEW"
    if "CHIT" in upper or "LOW HAND" in upper:
        return "HAND / OBJECT REVIEW"
    if "NEIGHBOUR" in upper:
        return "NEIGHBOUR ATTENTION REVIEW"
    if "HEAD" in upper or "GAZE" in upper:
        return "ORIENTATION REVIEW"
    return "REVIEW EVIDENCE"


def _panel(frame: np.ndarray, text: str, origin: tuple[int, int], *, color: tuple[int, int, int], scale: float = 0.47) -> None:
    x, y = origin
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(frame, (x, max(0, y - text_h - 8)), (x + text_w + 12, y + 5), PANEL, -1)
    cv2.putText(frame, text, (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_operator_overlay(frame: np.ndarray, people: list[Any], objects: list[Any], events: list[Any]) -> np.ndarray:
    """Render a compact human-review HUD.

    Semantic colour contract: cyan = tracked person, amber = object candidate,
    red = sustained review evidence.  Gaze/head proxy labels and pose skeletons
    are intentionally omitted because they are not reliable enough for the
    operator surface.
    """
    output = frame.copy()
    height, width = output.shape[:2]
    _panel(output, "LIVE AI OVERLAY  |  REVIEW MODE", (14, 27), color=WHITE, scale=0.52)
    _panel(output, f"{width}x{height}", (width - 120, 27), color=WHITE, scale=0.42)

    event_candidate_ids = {
        int(candidate_id)
        for event in events
        for candidate_id in getattr(event, "candidate_ids", [])
        if str(candidate_id).isdigit()
    }
    for person in people:
        x1, y1, x2, y2 = _box(person)
        candidate = _person_id(person)
        confidence = float(getattr(person, "detection_confidence", 0.0))
        event_related = candidate.isdigit() and int(candidate) in event_candidate_ids
        color = RED if event_related else CYAN
        thickness = 2 if event_related else 1
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        _panel(output, f"TRACK {candidate}  |  PERSON {confidence:.0%}", (x1, max(18, y1 - 5)), color=color, scale=0.38)

    # Generic furniture detections have no operator value.  Show only object
    # classes that can form a bounded review proposal.
    reviewable = {"phone", "cell phone", "mobilephone", "chit-paper", "paper", "chit"}
    for detected in objects:
        raw_name = _name(detected)
        metadata = getattr(detected, "metadata", {}) or {}
        raw_name = str(metadata.get("raw_model_class", raw_name)).lower()
        if raw_name not in reviewable:
            continue
        x1, y1, x2, y2 = _box(detected)
        confidence = float(getattr(detected, "confidence", 0.0))
        label = "PHONE CANDIDATE" if "phone" in raw_name else "OBJECT CANDIDATE"
        cv2.rectangle(output, (x1, y1), (x2, y2), AMBER, 2, cv2.LINE_AA)
        _panel(output, f"{label}  {confidence:.0%}", (x1, max(18, y1 - 5)), color=AMBER, scale=0.38)

    # The alert rail deliberately limits concurrent text: active evidence is
    # actionable, while raw event internals remain in JSONL/debug mode.
    for row, event in enumerate(events[:2]):
        candidate_ids = ", ".join(str(value) for value in getattr(event, "candidate_ids", [])) or "unassigned"
        duration = float(getattr(event, "duration_sec", 0.0))
        text = f"REVIEW  |  {_event_label(str(getattr(event, 'label', '')))}  |  TRACK {candidate_ids}  |  {duration:.1f}s"
        _panel(output, text, (14, 58 + row * 29), color=RED, scale=0.43)
    return output
