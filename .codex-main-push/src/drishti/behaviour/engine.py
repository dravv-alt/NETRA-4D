"""
4D Spatio-Temporal Behavioural Reasoning Engine for Drishti-4D / NETRA-4D.
Maintains a 12-second rolling temporal buffer per track, computes geometric trajectories,
tracks reciprocal candidate interactions, and generates explainable behavioural evidence.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..schemas import (
    PersonObservation,
    ObjectObservation,
    BehaviourEvidence,
    BehaviourEnum,
    CameraProfile,
    ObjectClassEnum
)
from ..context.camera_profile import CameraContextManager, point_in_polygon


class TrackTemporalBuffer:
    """Maintains timestamped history of observations for an individual candidate."""
    def __init__(self, track_id: int, max_history_sec: float = 12.0):
        self.track_id = track_id
        self.max_history_sec = max_history_sec
        self.buffer: deque[PersonObservation] = deque()

    def add(self, obs: PersonObservation):
        self.buffer.append(obs)
        # Prune old observations outside rolling time window
        current_time = obs.timestamp_sec
        while self.buffer and (current_time - self.buffer[0].timestamp_sec) > self.max_history_sec:
            self.buffer.popleft()

    def get_duration(self) -> float:
        if len(self.buffer) < 2:
            return 0.0
        return self.buffer[-1].timestamp_sec - self.buffer[0].timestamp_sec

    def is_valid(self, min_duration_sec: float = 1.0) -> bool:
        return self.get_duration() >= min_duration_sec


class BehaviourEngine:
    def __init__(
        self,
        camera_context: CameraContextManager,
        history_seconds: float = 12.0,
        head_turn_thresh_deg: float = 30.0,
        torso_turn_thresh_deg: float = 35.0,
        wrist_speed_thresh: float = 0.40
    ):
        self.context = camera_context
        self.history_seconds = history_seconds
        self.head_turn_thresh_deg = head_turn_thresh_deg
        self.torso_turn_thresh_deg = torso_turn_thresh_deg
        self.wrist_speed_thresh = wrist_speed_thresh

        # track_id -> TrackTemporalBuffer
        self.track_buffers: Dict[int, TrackTemporalBuffer] = {}
        # Object observation history (recent 5 seconds)
        self.recent_objects: deque[ObjectObservation] = deque()

    def _get_or_create_buffer(self, track_id: int) -> TrackTemporalBuffer:
        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = TrackTemporalBuffer(track_id, self.history_seconds)
        return self.track_buffers[track_id]

    def _estimate_2d_head_gaze(self, obs: PersonObservation) -> Tuple[float, str, Tuple[float, float]]:
        """
        Calculates 2D head turn angle (degrees), direction ('LEFT', 'RIGHT', 'DOWN', 'CENTER'),
        and projected gaze ray point.
        """
        kpts = obs.keypoints_2d
        if not kpts:
            return 0.0, "CENTER", (obs.bbox_xyxy_norm[0] + obs.bbox_xyxy_norm[2]) / 2, obs.bbox_xyxy_norm[1]

        nose = kpts.get("nose")
        l_eye = kpts.get("left_eye")
        r_eye = kpts.get("right_eye")
        l_ear = kpts.get("left_ear")
        r_ear = kpts.get("right_ear")
        l_sh = kpts.get("left_shoulder")
        r_sh = kpts.get("right_shoulder")

        if not nose or nose.confidence < 0.25:
            return 0.0, "CENTER", (0.5, 0.5)

        # Gaze point default
        gaze_pt = (nose.x, nose.y)

        # Check 3D Multi-HMR geometry first if available
        if obs.metadata and "smplx_geometry" in obs.metadata:
            geom = obs.metadata["smplx_geometry"]
            yaw = geom.get("head_yaw_deg", 0.0)
            pitch = geom.get("head_pitch_deg", 0.0)
            direction = "RIGHT" if yaw > 15 else ("LEFT" if yaw < -15 else ("DOWN" if pitch < -20 else "CENTER"))
            # Project gaze ray
            ray_x = nose.x + np.sin(np.radians(yaw)) * 0.25
            ray_y = nose.y + np.sin(np.radians(pitch)) * 0.20
            return abs(yaw), direction, (float(ray_x), float(ray_y))

        # 2D Ear/Eye asymmetry for head turn
        if l_ear and r_ear and l_ear.confidence > 0.25 and r_ear.confidence > 0.25:
            ear_mid_x = (l_ear.x + r_ear.x) / 2.0
            ear_dist = abs(l_ear.x - r_ear.x) + 1e-5
            displacement = (nose.x - ear_mid_x) / ear_dist
            angle_deg = float(np.clip(displacement * 60.0, -90.0, 90.0))
            direction = "RIGHT" if angle_deg > 15 else ("LEFT" if angle_deg < -15 else "CENTER")
            ray_x = nose.x + np.sign(angle_deg) * 0.20
            return abs(angle_deg), direction, (float(ray_x), float(nose.y))

        # Downward glance
        if l_sh and r_sh and l_sh.confidence > 0.3 and r_sh.confidence > 0.3:
            sh_mid_y = (l_sh.y + r_sh.y) / 2.0
            if (sh_mid_y - nose.y) < 0.05:  # Head is low
                return 35.0, "DOWN", (nose.x, nose.y + 0.15)

        return 0.0, "CENTER", gaze_pt

    def _estimate_torso_angle(self, obs: PersonObservation) -> Tuple[float, str]:
        """Calculates torso orientation relative to camera/desk axis."""
        kpts = obs.keypoints_2d
        if not kpts:
            return 0.0, "CENTER"

        l_sh = kpts.get("left_shoulder")
        r_sh = kpts.get("right_shoulder")
        if not l_sh or not r_sh or l_sh.confidence < 0.25 or r_sh.confidence < 0.25:
            return 0.0, "CENTER"

        dx = r_sh.x - l_sh.x
        dy = r_sh.y - l_sh.y
        # Shoulder vector tilt
        angle = np.degrees(np.arctan2(dy, dx))
        # Normalization
        dev = abs(angle)
        direction = "RIGHT" if angle > 10 else ("LEFT" if angle < -10 else "CENTER")
        return float(dev), direction

    def update(
        self,
        current_obs_list: List[PersonObservation],
        object_obs_list: Optional[List[ObjectObservation]] = None
    ) -> List[BehaviourEvidence]:
        """
        Processes current observations, updates temporal buffers,
        and generates atomic behavioural evidence signals.
        """
        all_evidence: List[BehaviourEvidence] = []
        now = current_obs_list[0].timestamp_sec if current_obs_list else 0.0

        # Update object buffer
        if object_obs_list:
            for obj in object_obs_list:
                self.recent_objects.append(obj)
            while self.recent_objects and (now - self.recent_objects[0].timestamp_sec) > 5.0:
                self.recent_objects.popleft()

        # Update person track buffers
        for obs in current_obs_list:
            buf = self._get_or_create_buffer(obs.entity_local_id)
            buf.add(obs)

        # -------------------------------------------------------------
        # 1. Per-Candidate Temporal Evidence Extraction
        # -------------------------------------------------------------
        for obs in current_obs_list:
            track_id = obs.entity_local_id
            seat_id = obs.seat_id
            buf = self.track_buffers[track_id]
            cam_id = obs.camera_id

            if len(buf.buffer) < 2:
                continue

            angle, direction, gaze_ray = self._estimate_2d_head_gaze(obs)
            target_type, target_id = self.context.resolve_target(seat_id, gaze_ray)
            torso_angle, torso_dir = self._estimate_torso_angle(obs)

            # Check counter-evidence (e.g. looking at board, invigilator, own monitor)
            is_legitimate_target = target_type in ["BOARD", "INVIGILATOR", "OWN_MONITOR", "OWN_DESK"]

            # ---------------------------------------------------------
            # B1 & B2: Head Turn Analysis (Repeated Glance / Sustained Turn)
            # ---------------------------------------------------------
            if angle >= self.head_turn_thresh_deg and not is_legitimate_target:
                # Measure duration in history
                turn_dur = 0.0
                turn_occurrences = 0
                for past_obs in buf.buffer:
                    p_ang, p_dir, p_ray = self._estimate_2d_head_gaze(past_obs)
                    if p_ang >= (self.head_turn_thresh_deg * 0.8) and p_dir == direction:
                        turn_dur += 0.25
                        turn_occurrences += 1

                if turn_dur >= 1.5:
                    # B2: Sustained Excessive Head Turn
                    all_evidence.append(BehaviourEvidence(
                        evidence_id=f"EV_HEAD_SUSTAINED_{track_id}_{int(now*10)}",
                        camera_id=cam_id,
                        timestamp_sec=now,
                        start_sec=now - turn_dur,
                        end_sec=now,
                        entity_id=track_id,
                        seat_id=seat_id,
                        evidence_type="SUSTAINED_EXCESSIVE_HEAD_TURN",
                        target_type=target_type,
                        target_id=target_id,
                        strength=min(1.0, 0.65 + (turn_dur / 5.0) * 0.35),
                        quality=1.0 if obs.quality.head == "GOOD" else 0.5,
                        direction=direction,
                        reason_codes=[f"Head turned {direction} ({angle:.1f}°) toward {target_id or 'neighbour'} for {turn_dur:.1f}s"]
                    ))
                elif turn_occurrences >= 3:
                    # B1: Repeated Sideward Glance
                    all_evidence.append(BehaviourEvidence(
                        evidence_id=f"EV_HEAD_GLANCE_{track_id}_{int(now*10)}",
                        camera_id=cam_id,
                        timestamp_sec=now,
                        start_sec=now - turn_dur,
                        end_sec=now,
                        entity_id=track_id,
                        seat_id=seat_id,
                        evidence_type="REPEATED_SIDEWARD_GLANCE",
                        target_type=target_type,
                        target_id=target_id,
                        strength=0.75,
                        quality=1.0 if obs.quality.head == "GOOD" else 0.5,
                        direction=direction,
                        reason_codes=[f"Repeated glances {direction} ({turn_occurrences} occurrences) toward {target_id or 'neighbour'}"]
                    ))

            # ---------------------------------------------------------
            # B3: Torso Rotation Toward Neighbour
            # ---------------------------------------------------------
            if torso_angle >= self.torso_turn_thresh_deg and not is_legitimate_target:
                all_evidence.append(BehaviourEvidence(
                    evidence_id=f"EV_TORSO_{track_id}_{int(now*10)}",
                    camera_id=cam_id,
                    timestamp_sec=now,
                    start_sec=now - 1.0,
                    end_sec=now,
                    entity_id=track_id,
                    seat_id=seat_id,
                    evidence_type="TORSO_ROTATION_TOWARD_NEIGHBOUR",
                    target_type="NEIGHBOUR_SEAT",
                    target_id=target_id,
                    strength=min(1.0, 0.60 + (torso_angle / 60.0) * 0.40),
                    quality=1.0 if obs.quality.torso == "GOOD" else 0.6,
                    direction=torso_dir,
                    reason_codes=[f"Torso rotated {torso_dir} ({torso_angle:.1f}°) toward neighbour"]
                ))

            # ---------------------------------------------------------
            # B4 & B6: Hand Movement / Object / Lap Region Association
            # ---------------------------------------------------------
            kpts = obs.keypoints_2d or {}
            l_wrist = kpts.get("left_wrist")
            r_wrist = kpts.get("right_wrist")

            # Check matching object in recent detections
            for obj in self.recent_objects:
                if obj.associated_entity_id == track_id or (seat_id and obj.associated_seat_id == seat_id):
                    if obj.object_class == ObjectClassEnum.MOBILE_PHONE:
                        all_evidence.append(BehaviourEvidence(
                            evidence_id=f"EV_PHONE_{track_id}_{int(now*10)}",
                            camera_id=cam_id,
                            timestamp_sec=now,
                            start_sec=now - 2.0,
                            end_sec=now,
                            entity_id=track_id,
                            seat_id=seat_id,
                            evidence_type="POSSIBLE_OBJECT_INTERACTION",
                            target_type="LAP_REGION",
                            target_id="PHONE_IN_LAP",
                            strength=0.92,
                            quality=1.0,
                            reason_codes=["Mobile phone interaction candidate in lap/desk region", f"Confidence: {obj.confidence:.2f}"]
                        ))
                    elif obj.object_class == ObjectClassEnum.PAPER_CHIT:
                        all_evidence.append(BehaviourEvidence(
                            evidence_id=f"EV_CHIT_{track_id}_{int(now*10)}",
                            camera_id=cam_id,
                            timestamp_sec=now,
                            start_sec=now - 1.5,
                            end_sec=now,
                            entity_id=track_id,
                            seat_id=seat_id,
                            evidence_type="POSSIBLE_OBJECT_INTERACTION",
                            target_type="NEIGHBOUR_DESK",
                            target_id="PAPER_CHIT",
                            strength=0.88,
                            quality=1.0,
                            reason_codes=["Paper/chit passing candidate across workstation partition"]
                        ))

        # -------------------------------------------------------------
        # 2. B5: Reciprocal Interaction Detection (A -> B and B -> A)
        # -------------------------------------------------------------
        if len(current_obs_list) >= 2:
            for i in range(len(current_obs_list)):
                for j in range(i + 1, len(current_obs_list)):
                    p_a = current_obs_list[i]
                    p_b = current_obs_list[j]

                    ang_a, dir_a, ray_a = self._estimate_2d_head_gaze(p_a)
                    ang_b, dir_b, ray_b = self._estimate_2d_head_gaze(p_b)

                    # Check mutual head turns toward each other
                    if ang_a > 20 and ang_b > 20 and dir_a != dir_b:
                        all_evidence.append(BehaviourEvidence(
                            evidence_id=f"EV_RECIPROCAL_{p_a.entity_local_id}_{p_b.entity_local_id}_{int(now*10)}",
                            camera_id=p_a.camera_id,
                            timestamp_sec=now,
                            start_sec=now - 1.0,
                            end_sec=now,
                            entity_id=p_a.entity_local_id,
                            seat_id=p_a.seat_id,
                            evidence_type="POSSIBLE_NONVERBAL_INTERACTION",
                            target_type="NEIGHBOUR_SEAT",
                            target_id=p_b.seat_id or f"Track_{p_b.entity_local_id}",
                            strength=0.89,
                            quality=0.90,
                            reason_codes=[f"Reciprocal mutual orientation between Seat {p_a.seat_id or 'A'} and Seat {p_b.seat_id or 'B'}"]
                        ))

        return all_evidence
