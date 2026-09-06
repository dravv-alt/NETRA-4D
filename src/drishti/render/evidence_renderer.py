"""
Clean, Surveillance-Grade Evidence Video Renderer for NETRA-4D.

Adheres strictly to the principle that surveillance evidence must be clear,
unoccluded, and free of visual clutter.

Rules:
1. MultiHMR 3D Mesh: STRICTLY DISABLED from output video.
2. Background Furniture: Zero bounding boxes on chairs, empty desks, or static screens.
3. Invigilator Exclusion: Standing / roaming invigilators get ZERO candidate skeleton,
   ZERO bounding boxes, ZERO alerts.
4. Face Privacy: Applied FIRST directly on raw pixels so all overlays are 100% crisp.
   Compact natural ellipse hugging the face contour without spilling onto shoulders or chest.
5. Sleek 2D Skeletons: Muted slate for normal candidates, warm amber for review,
   high-contrast red for active violations.
6. Targeted Evidence: Highlights phones (bounding box) and hand/wrist anomalies (focal ring).
7. Minimalist Surveillance HUD: Clean top bar with timecode and active alert summary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np


COCO_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # Arms & Shoulders
    (5, 11), (6, 12), (11, 12),                   # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)        # Legs
]

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class EvidenceRenderer:
    """Surveillance evidence renderer providing clean, high-impact video feeds."""

    def __init__(
        self,
        enable_face_blur: bool = True,
        min_keypoint_conf: float = 0.20,
    ):
        self.enable_face_blur = enable_face_blur
        self.min_keypoint_conf = min_keypoint_conf

        # Palette definition (BGR)
        self.color_normal_skeleton = (180, 195, 160)   # Muted soft slate/teal
        self.color_normal_joint = (220, 235, 210)
        self.color_review_skeleton = (0, 165, 255)     # Warm amber
        self.color_review_joint = (255, 255, 255)
        self.color_alert_skeleton = (0, 40, 240)       # High-visibility red
        self.color_alert_joint = (255, 255, 255)

        self.color_phone_box = (0, 50, 255)            # Bright red-orange for phone
        self.color_hand_focus = (0, 70, 255)           # Red focal ring
        self.color_gaze_peeking = (0, 140, 255)        # Amber gaze ray

    def apply_face_privacy(
        self,
        frame_bgr: np.ndarray,
        tracked_people: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Applies a compact, form-fitting natural face blur directly onto raw pixels."""
        if not self.enable_face_blur or not tracked_people:
            return frame_bgr.copy()

        h_img, w_img = frame_bgr.shape[:2]
        face_mask = np.zeros((h_img, w_img), dtype=np.uint8)

        for person in tracked_people:
            kps = person.get("keypoints")
            bx = person.get("bbox", [0, 0, 0, 0])
            bw = max(1.0, float(bx[2] - bx[0]))
            bh = max(1.0, float(bx[3] - bx[1]))

            fc_x: Optional[int] = None
            fc_y: Optional[int] = None
            rx: int = 15
            ry: int = 20

            if kps is not None and len(kps) >= 5:
                # Keypoints: 0=nose, 1=leye, 2=reye, 3=lear, 4=rear
                nose = kps[0]
                leye = kps[1]
                reye = kps[2]

                has_nose = nose[2] >= self.min_keypoint_conf
                has_eyes = leye[2] >= self.min_keypoint_conf and reye[2] >= self.min_keypoint_conf

                if has_eyes and has_nose:
                    fc_x = int((leye[0] + reye[0] + nose[0]) / 3.0)
                    fc_y = int((leye[1] + reye[1] + nose[1] * 2.0) / 4.0)
                    eye_span = float(np.hypot(reye[0] - leye[0], reye[1] - leye[1]))
                    rx = int(max(10, eye_span * 0.95))
                    ry = int(max(14, eye_span * 1.30))
                elif has_nose:
                    fc_x = int(nose[0])
                    fc_y = int(nose[1])
                    rx = int(max(10, bw * 0.18))
                    ry = int(max(14, bw * 0.24))
                elif has_eyes:
                    fc_x = int((leye[0] + reye[0]) / 2.0)
                    fc_y = int((leye[1] + reye[1]) / 2.0 + bw * 0.08)
                    eye_span = float(np.hypot(reye[0] - leye[0], reye[1] - leye[1]))
                    rx = int(max(10, eye_span * 0.95))
                    ry = int(max(14, eye_span * 1.30))

            if fc_x is None and bw > 15 and bh > 20:
                # Fallback: top of head
                fc_x = int((bx[0] + bx[2]) / 2.0)
                fc_y = int(bx[1] + bh * 0.15)
                rx = int(max(8, bw * 0.20))
                ry = int(max(12, bh * 0.14))

            if fc_x is not None and fc_y is not None:
                # Constrain ellipse to person bbox to prevent bleeding onto chest
                fc_x = max(int(bx[0]), min(int(bx[2]), fc_x))
                fc_y = max(int(bx[1]), min(int(bx[1] + bh * 0.40), fc_y))
                cv2.ellipse(face_mask, (fc_x, fc_y), (rx, ry), 0, 0, 360, 255, -1)

        if np.any(face_mask > 0):
            blurred_base = cv2.GaussianBlur(frame_bgr, (35, 35), 0)
            feathered = cv2.GaussianBlur(face_mask, (11, 11), 0)
            alpha = (feathered.astype(np.float32) / 255.0)[:, :, np.newaxis]
            output = (alpha * blurred_base.astype(np.float32) + (1.0 - alpha) * frame_bgr.astype(np.float32)).astype(np.uint8)
            return output

        return frame_bgr.copy()

    def render_candidate_skeletons(
        self,
        canvas: np.ndarray,
        candidates: Dict[str, Any],
        invigilator_cand_ids: Set[str]
    ) -> None:
        """Renders clean 2D pose skeletons for candidates only."""
        h_img, w_img = canvas.shape[:2]

        for cand_id, state in candidates.items():
            if cand_id in invigilator_cand_ids:
                # STRICT INVIGILATOR EXCLUSION
                continue

            if not getattr(state, "is_visible", True) or not state.current_observation:
                continue

            obs = state.current_observation
            b_state = getattr(state, "behavioural_state", "NORMAL")

            if b_state in ("HIGH_PRIORITY_REVIEW", "ALERT"):
                bone_col = self.color_alert_skeleton
                joint_col = self.color_alert_joint
            elif b_state in ("REVIEW", "SUSPICIOUS", "OBSERVING"):
                bone_col = self.color_review_skeleton
                joint_col = self.color_review_joint
            else:
                bone_col = self.color_normal_skeleton
                joint_col = self.color_normal_joint

            # Draw bones
            for p1_idx, p2_idx in COCO_SKELETON_PAIRS:
                p1_name = COCO_KEYPOINT_NAMES[p1_idx]
                p2_name = COCO_KEYPOINT_NAMES[p2_idx]
                kp1 = obs.get_keypoint(p1_name)
                kp2 = obs.get_keypoint(p2_name)
                if kp1 and kp2 and kp1.confidence >= self.min_keypoint_conf and kp2.confidence >= self.min_keypoint_conf:
                    pt1 = (int(kp1.x * w_img), int(kp1.y * h_img))
                    pt2 = (int(kp2.x * w_img), int(kp2.y * h_img))
                    cv2.line(canvas, pt1, pt2, bone_col, 2, cv2.LINE_AA)

            # Draw joint keypoints
            for kp in obs.keypoints.values():
                if kp.confidence >= self.min_keypoint_conf:
                    pt = (int(kp.x * w_img), int(kp.y * h_img))
                    cv2.circle(canvas, pt, 3, joint_col, -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 2, bone_col, -1, cv2.LINE_AA)

    def render_evidence_objects(
        self,
        canvas: np.ndarray,
        detected_objects: List[Any],
    ) -> List[Tuple[int, int, int, int, str, float]]:
        """Renders high-visibility boxes ONLY for evidence items like cell phones."""
        h_img, w_img = canvas.shape[:2]
        evidence_found = []

        for obj in detected_objects:
            # Check class name
            cls_name = getattr(obj, "class_name", "").lower()
            conf = float(getattr(obj, "confidence", 0.0))

            # Highlight mobile phones or chit paper evidence objects
            is_phone = ("phone" in cls_name or "cell" in cls_name) and conf >= 0.25
            is_chit = ("chit" in cls_name or "paper" in cls_name) and conf >= 0.20

            if is_phone or is_chit:
                bx_norm = getattr(obj, "bbox_xyxy_norm", None)
                if bx_norm is None:
                    continue

                x1 = max(0, int(bx_norm[0] * w_img))
                y1 = max(0, int(bx_norm[1] * h_img))
                x2 = min(w_img, int(bx_norm[2] * w_img))
                y2 = min(h_img, int(bx_norm[3] * h_img))

                box_color = (0, 0, 230) if is_chit else self.color_phone_box
                label = f"CHIT PAPER [{int(conf * 100)}%]" if is_chit else f"PHONE [{int(conf * 100)}%]"

                # Draw high-contrast bounding box
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 3, cv2.LINE_AA)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

                # Clean badge
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                by1 = max(0, y1 - th - 6)
                cv2.rectangle(canvas, (x1, by1), (x1 + tw + 8, y1), box_color, -1)
                cv2.putText(canvas, label, (x1 + 4, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

                evidence_found.append((x1, y1, x2, y2, cls_name, conf))

        return evidence_found

    def render_hand_anomalies(
        self,
        canvas: np.ndarray,
        candidates: Dict[str, Any],
        invigilator_cand_ids: Set[str]
    ) -> None:
        """Draws clean focal rings around suspicious wrists / paper retrieval areas."""
        h_img, w_img = canvas.shape[:2]

        for cand_id, state in candidates.items():
            if cand_id in invigilator_cand_ids or not getattr(state, "is_visible", True):
                continue

            b_state = getattr(state, "behavioural_state", "NORMAL")
            obs = state.current_observation
            if not obs:
                continue

            # Check if this candidate is under review or alert
            if b_state in ("REVIEW", "HIGH_PRIORITY_REVIEW", "ALERT"):
                # Find active wrist
                rw = obs.get_keypoint("right_wrist")
                lw = obs.get_keypoint("left_wrist")
                wrists = [w for w in (rw, lw) if w and w.confidence >= self.min_keypoint_conf]

                if wrists:
                    target_w = wrists[0]
                    wx, wy = int(target_w.x * w_img), int(target_w.y * h_img)

                    # Double focal ring
                    cv2.circle(canvas, (wx, wy), 20, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (wx, wy), 24, self.color_hand_focus, 2, cv2.LINE_AA)

                    # Small callout badge
                    callout_text = "HAND ANOMALY"
                    (tw, th), _ = cv2.getTextSize(callout_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                    bx1 = wx + 28
                    by1 = wy - 10
                    bx2 = bx1 + tw + 10
                    by2 = wy + 8
                    if bx2 < w_img:
                        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (15, 18, 22), -1)
                        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), self.color_hand_focus, 1, cv2.LINE_AA)
                        cv2.putText(canvas, callout_text, (bx1 + 5, wy + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1, cv2.LINE_AA)
                        cv2.line(canvas, (wx + 24, wy), (bx1, wy), self.color_hand_focus, 1, cv2.LINE_AA)

    def render_gaze_arrows(
        self,
        canvas: np.ndarray,
        candidates: Dict[str, Any],
        invigilator_cand_ids: Set[str]
    ) -> None:
        """Renders subtle gaze direction arrows ONLY when candidate is peeking or looking off-desk."""
        h_img, w_img = canvas.shape[:2]

        for cand_id, state in candidates.items():
            if cand_id in invigilator_cand_ids or not getattr(state, "is_visible", True):
                continue

            obs = state.current_observation
            if not obs:
                continue

            gaze_dir = getattr(state, "gaze_direction", "UNKNOWN")
            is_peeking = getattr(state, "is_peeking_neighbour", False)
            neighbour_cand = getattr(state, "neighbour_candidate", None)

            # Only draw gaze arrow if peeking or flagged
            if is_peeking or gaze_dir in ("LEFT", "RIGHT", "DOWN"):
                nose = obs.get_keypoint("nose")
                leye = obs.get_keypoint("left_eye")
                reye = obs.get_keypoint("right_eye")

                origin = None
                if leye and reye and leye.confidence >= self.min_keypoint_conf and reye.confidence >= self.min_keypoint_conf:
                    origin = (int(((leye.x + reye.x) / 2.0) * w_img), int(((leye.y + reye.y) / 2.0) * h_img))
                elif nose and nose.confidence >= self.min_keypoint_conf:
                    origin = (int(nose.x * w_img), int(nose.y * h_img))

                if origin is not None:
                    ray_len = 32
                    if gaze_dir == "LEFT":
                        target = (origin[0] - ray_len, origin[1] + 6)
                    elif gaze_dir == "RIGHT":
                        target = (origin[0] + ray_len, origin[1] + 6)
                    elif gaze_dir == "DOWN":
                        target = (origin[0], origin[1] + ray_len)
                    else:
                        target = (origin[0] + 12, origin[1] + 18)

                    arrow_col = self.color_alert_skeleton if is_peeking else self.color_gaze_peeking
                    cv2.arrowedLine(canvas, origin, target, arrow_col, 2, cv2.LINE_AA, tipLength=0.35)
                    cv2.circle(canvas, origin, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def render_candidate_pills(
        self,
        canvas: np.ndarray,
        candidates: Dict[str, Any],
        invigilator_cand_ids: Set[str]
    ) -> None:
        """Renders compact, semi-transparent pill badges above candidate heads."""
        h_img, w_img = canvas.shape[:2]

        for cand_id, state in candidates.items():
            if cand_id in invigilator_cand_ids:
                continue

            # Display badge for visible candidates or candidates temporarily occluded by passing invigilator
            is_vis = getattr(state, "is_visible", True)
            is_occ = getattr(state, "is_occluded", False)
            if not is_vis and not is_occ:
                continue

            obs = state.current_observation
            if not obs:
                continue

            bx_norm = obs.bbox_xyxy_norm
            x1 = max(0, int(bx_norm[0] * w_img))
            y1 = max(0, int(bx_norm[1] * h_img))
            x2 = min(w_img, int(bx_norm[2] * w_img))
            y2 = min(h_img, int(bx_norm[3] * h_img))

            b_state = getattr(state, "behavioural_state", "NORMAL")
            desk_id = getattr(state.identity, "workstation_id", "") or cand_id.replace("Candidate_", "Seat ")

            # Build compact badge text
            if b_state in ("HIGH_PRIORITY_REVIEW", "ALERT"):
                tag = f"{desk_id} | ALERT: ANOMALY"
                badge_border = self.color_alert_skeleton
                badge_bg = (0, 0, 160)
            elif b_state in ("REVIEW", "SUSPICIOUS"):
                is_peeking = getattr(state, "is_peeking_neighbour", False)
                reason = "Peeking" if is_peeking else "Suspicious"
                tag = f"{desk_id} | REVIEW: {reason}"
                badge_border = self.color_review_skeleton
                badge_bg = (0, 90, 160)
            else:
                tag = f"{desk_id}"
                badge_border = (80, 90, 80)
                badge_bg = (20, 24, 25)

            font_scale = 0.38
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            # Anchor directly over candidate head/neck keypoint for accurate visual alignment
            nose = obs.get_keypoint("nose")
            leye = obs.get_keypoint("left_eye")
            reye = obs.get_keypoint("right_eye")
            lear = obs.get_keypoint("left_ear")
            rear = obs.get_keypoint("right_ear")
            head_pts = [p for p in (nose, leye, reye, lear, rear) if p and p.confidence >= 0.20]
            if head_pts:
                hx = int(np.mean([p.x for p in head_pts]) * w_img)
                hy = int(np.mean([p.y for p in head_pts]) * h_img)
                top_x = max(2, min(w_img - tw - 16, hx - tw // 2))
                top_y = max(34 + th + 4, hy - 16)
            else:
                top_x = max(2, min(w_img - tw - 16, (x1 + x2) // 2 - tw // 2))
                # Ensure pill never hides under the top surveillance HUD (y=0..28)
                if y1 <= 36:
                    top_y = min(h_img - 8, max(34 + th + 4, y1 + int((y2 - y1) * 0.40)))
                else:
                    top_y = max(34 + th + 4, y1 - 6)

            # Draw translucent pill
            sub = canvas[max(0, top_y - th - 6):top_y + 2, top_x:top_x + tw + 12]
            if sub.shape[0] > 0 and sub.shape[1] > 0:
                rect = sub.copy()
                cv2.rectangle(rect, (0, 0), (sub.shape[1], sub.shape[0]), badge_bg, -1)
                canvas[max(0, top_y - th - 6):top_y + 2, top_x:top_x + tw + 12] = cv2.addWeighted(rect, 0.85, sub, 0.15, 0)
                cv2.rectangle(canvas, (top_x, max(0, top_y - th - 6)), (top_x + tw + 12, top_y + 2), badge_border, 1, cv2.LINE_AA)
                cv2.putText(canvas, tag, (top_x + 6, top_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def render_surveillance_hud(
        self,
        canvas: np.ndarray,
        video_title: str,
        frame_idx: int,
        total_frames: int,
        fps: float,
        active_alerts: List[str]
    ) -> None:
        """Renders minimalist surveillance status header at the top of the video."""
        h_img, w_img = canvas.shape[:2]

        # Sleek dark header bar
        cv2.rectangle(canvas, (0, 0), (w_img, 28), (14, 18, 22), -1)
        cv2.line(canvas, (0, 28), (w_img, 28), (40, 50, 60), 1, cv2.LINE_AA)

        # Left: Clean branding & timecode
        cur_sec = int(frame_idx / max(1.0, fps))
        tot_sec = int(total_frames / max(1.0, fps))
        timecode = f"{cur_sec // 60:02d}:{cur_sec % 60:02d} / {tot_sec // 60:02d}:{tot_sec % 60:02d}"

        cv2.putText(canvas, "NETRA-4D AI PROCTOR", (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 225, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"|   {video_title}   |   TIME: {timecode}", (185, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 195, 205), 1, cv2.LINE_AA)

        # Right: Dynamic Alert Status Pill
        if active_alerts:
            alert_count = len(active_alerts)
            summary = active_alerts[0]
            pill_text = f"● {alert_count} ACTIVE ANOMALY: {summary}" if alert_count == 1 else f"● {alert_count} ACTIVE ANOMALIES"
            (tw, th), _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            ax1 = w_img - tw - 24
            cv2.rectangle(canvas, (ax1, 4), (w_img - 10, 24), (0, 0, 180), -1)
            cv2.rectangle(canvas, (ax1, 4), (w_img - 10, 24), (0, 60, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, pill_text, (ax1 + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            status_text = "● ALL NORMAL"
            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            ax1 = w_img - tw - 24
            cv2.rectangle(canvas, (ax1, 4), (w_img - 10, 24), (20, 60, 30), -1)
            cv2.rectangle(canvas, (ax1, 4), (w_img - 10, 24), (40, 140, 60), 1, cv2.LINE_AA)
            cv2.putText(canvas, status_text, (ax1 + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 200), 1, cv2.LINE_AA)

    def render_frame(
        self,
        raw_frame: np.ndarray,
        tracked_people: List[Dict[str, Any]],
        candidates: Dict[str, Any],
        detected_objects: List[Any],
        invigilator_cand_ids: Set[str],
        video_title: str,
        frame_idx: int,
        total_frames: int,
        fps: float,
        active_alerts: List[str]
    ) -> np.ndarray:
        """
        Master render pipeline executing operations in strict visual layers:
        1. Natural Face Privacy Blur (applied on raw pixels)
        2. Clean 2D Candidate Skeletons (candidates only)
        3. Targeted Evidence Objects (phones only)
        4. Suspicious Hand/Chit Anomalies (wrist focal rings)
        5. Peeking Gaze Arrows (head direction)
        6. Candidate Pill Badges (compact)
        7. Minimalist Surveillance HUD (top bar)
        """
        # Layer 1: Face Privacy Blur on raw frame
        canvas = self.apply_face_privacy(raw_frame, tracked_people)

        # Layer 2: 2D Candidate Skeletons (Zero invigilator skeletons)
        self.render_candidate_skeletons(canvas, candidates, invigilator_cand_ids)

        # Layer 3: Targeted Evidence Objects (phones only, zero furniture)
        self.render_evidence_objects(canvas, detected_objects)

        # Layer 4: Hand Anomalies (Chit / Paper retrieval focal rings)
        self.render_hand_anomalies(canvas, candidates, invigilator_cand_ids)

        # Layer 5: Gaze Vectors
        self.render_gaze_arrows(canvas, candidates, invigilator_cand_ids)

        # Layer 6: Compact Candidate Pills
        self.render_candidate_pills(canvas, candidates, invigilator_cand_ids)

        # Layer 7: Minimalist Surveillance HUD
        self.render_surveillance_hud(canvas, video_title, frame_idx, total_frames, fps, active_alerts)

        return canvas
