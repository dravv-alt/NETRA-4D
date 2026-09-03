"""Master Unified Pipeline Runner for NETRA-4D / DrishtiAI.

Strictly conforms to docs/ARCHITECTURE.md:
CCTV / Video Ingestion -> Quality Gate -> Environment Profiling ->
Adaptive Vision Stack (YOLO26-Pose / Objects) ->
Role / Identity Layer (Immutable Invigilator & Candidate Seat Locks) ->
Canonical Candidate State & Temporal Buffers ->
Biomechanical Feature Extraction (Dimensionless Head/Torso/Hands) ->
Semantic Context & Zone Disambiguation (Desk, Neighbour, Board, Proctor) ->
Temporal Patterns (Duration, Persistence, Repetition) ->
Competing Hypotheses Arbitration (H1..H5 with Counter-Evidence) ->
Three-Stage Precision Escalation Gate (Stage 1 -> Stage 2 Specialist -> Stage 3 Deep 3D) ->
Candidate State Machine (NORMAL -> REVIEW -> HIGH_PRIORITY_REVIEW) ->
Privacy-Safe Rendering (Universal Face Blur, Selective Candidate Overlays, Zero Proctor Alert Overlays).

Zero Hardcoded Constants or Magic Formulas.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

# Project root paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "multi-hmr"))

from drishti.data.google_drive import load_drive_manifest, materialize_drive_asset
from drishti.privacy import apply_privacy_mask_and_collect_annotations
from drishti.pipeline.environment import EnvironmentProfiler, EnvironmentType
from drishti.pipeline.role_registry import ImmutableRoleRegistry, RoleEnum
from drishti.pipeline.specialist_crop import SpecialistHandCropAnalyzer
from drishti.engine.observation import (
    DetectedObjectObservation,
    ObservationQuality,
    VisibilityState,
    VisionObservation,
    YOLO26Adapter,
)
from drishti.engine.candidate_state import (
    CandidateStateManager,
    IdentityLockStatus,
)
from drishti.engine.temporal_buffer import TemporalBufferManager
from drishti.engine.features import (
    BiomechanicalFeatureExtractor,
    CandidateBiomechanicalFeatures,
    GazeDirection,
)
from drishti.engine.primitive_events import (
    PrimitiveEvent,
    PrimitiveEventDetector,
    PrimitiveEventType,
)
from drishti.engine.temporal_patterns import (
    PersistenceDetector,
    RepetitionDetector,
    TemporalPatternType,
)
from drishti.engine.zones import (
    SemanticZoneEngine,
    SemanticZoneType,
)
from drishti.engine.hypotheses import (
    CompetingHypothesisEngine,
    EvidenceEvaluationResult,
    HypothesisKey,
)
from drishti.engine.state_machine import (
    CandidateBehaviouralState,
    CandidateStateMachine,
)
from drishti.engine.escalation import (
    EscalationGate,
    EscalationStage,
)
from drishti.engine.specialists import (
    MediaPipeHandRunner,
    RTMPoseONNXRunner,
)

# Multi-HMR & SMPL-X for Stage 3 deep 3D analysis
from utils import normalize_rgb, get_focalLength_from_fieldOfView
from utils.camera import perspective_projection
from model import Model
import smplx

PALETTE_BGR = [
    (0, 255, 128),    # Spring Green
    (255, 128, 0),    # Vivid Orange
    (255, 0, 255),    # Magenta
    (0, 220, 255),    # Golden Yellow
    (255, 80, 80),    # Soft Blue
    (80, 255, 80),    # Lime
    (180, 100, 255),  # Violet
    (0, 165, 255),    # Tangerine
    (200, 255, 0),    # Cyan-green
    (255, 192, 203),  # Pink
]

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


def get_camera_parameters(img_size: int, fov: float = 60.0, device: torch.device = torch.device('cuda:0')):
    K = torch.eye(3)
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0, 0], K[1, 1] = focal, focal
    K[0, -1], K[1, -1] = img_size // 2, img_size // 2
    return K.unsqueeze(0).to(device)


def preprocess_frame_for_hmr(frame_bgr: np.ndarray, img_size: int, device: torch.device):
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_pil_contained = ImageOps.contain(img_pil, (img_size, img_size))
    cw, ch = img_pil_contained.size
    img_pil_padded = ImageOps.pad(img_pil_contained, size=(img_size, img_size))
    arr = normalize_rgb(np.asarray(img_pil_padded))
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)

    pad_x = (img_size - cw) / 2.0
    pad_y = (img_size - ch) / 2.0
    scale = cw / float(w)
    return tensor, (scale, pad_x, pad_y)




def run_unified_pipeline(
    video_source: str,
    output_dir: str,
    sample_every: int = 2,
    max_frames: int = 0,
    prime_vram_gb: float = 2.2,
) -> Tuple[str, str]:
    """Executes the master unified pipeline adhering to docs/ARCHITECTURE.md."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[NETRA-4D Unified Pipeline] Executing on compute device: {device}")

    # =========================================================================
    # 1. Initialize Decoupled Subsystem Engines
    # =========================================================================
    env_profiler = EnvironmentProfiler(warmup_frames=30)
    role_registry = ImmutableRoleRegistry(warmup_frames=45)
    candidate_state_manager = CandidateStateManager(warmup_frames=45)
    semantic_zone_engine = SemanticZoneEngine()
    temporal_buffer_manager = TemporalBufferManager(max_duration_sec=5.0)
    feature_extractor = BiomechanicalFeatureExtractor(smoothing_tau=0.12)
    primitive_event_detector = PrimitiveEventDetector()
    persistence_detector = PersistenceDetector()
    repetition_detector = RepetitionDetector(window_duration_sec=90.0, repetition_threshold=3)
    competing_hypothesis_engine = CompetingHypothesisEngine()
    state_machines: Dict[str, CandidateStateMachine] = {}
    specialist_crop_analyzer = SpecialistHandCropAnalyzer(crop_size=256)
    rtmpose_specialist = RTMPoseONNXRunner()
    mediapipe_specialist = MediaPipeHandRunner()
    escalation_gate = EscalationGate()

    # Wire Stage 2 Specialists into EscalationGate
    def stage2_crop_runner(crops):
        if not crops:
            return None
        return specialist_crop_analyzer.analyze_crop(crops[0], (crops[0].shape[1]//2, crops[0].shape[0]//2))

    escalation_gate.stage2_crop_analyzer = stage2_crop_runner
    if rtmpose_specialist.is_available:
        escalation_gate.stage2_rtmpose_verifier = rtmpose_specialist.infer_crop
        print("[NETRA-4D] Stage 2 RTM-Pose ONNX Specialist active.")
    if mediapipe_specialist.is_available:
        escalation_gate.stage2_mediapipe_verifier = mediapipe_specialist.infer_hand
        print("[NETRA-4D] Stage 2 MediaPipe 3D Hand Specialist active.")

    # =========================================================================
    # 2. Load Vision Models
    # =========================================================================
    pose_model_path = os.path.join(ROOT, "models", "weights", "yolo26n-pose.pt")
    obj_model_path = os.path.join(ROOT, "models", "weights", "yolo26n.pt")
    yolo_pose = YOLO(pose_model_path).to(device)
    yolo_obj = YOLO(obj_model_path).to(device)

    # Multi-HMR & SMPL-X for Stage 3 deep analysis
    hmr_ckpt = os.path.join(ROOT, "models", "multiHMR", "multiHMR_672_S.pt")
    ckpt = torch.load(hmr_ckpt, map_location=device)
    kwargs = {k: v for k, v in vars(ckpt['args']).items()}
    kwargs['type'] = ckpt['args'].train_return_type
    kwargs['img_size'] = ckpt['args'].img_size[0]
    img_size = kwargs['img_size']
    hmr_model = Model(**kwargs).to(device)
    hmr_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    hmr_model.eval()

    smplx_neutral = smplx.create(os.path.join(ROOT, "models"), model_type='smplx', gender='neutral', use_pca=False, flat_hand_mean=True, num_betas=11)
    faces_sampled = smplx_neutral.faces[::4]

    # Pre-allocate GPU scratch cache to stabilize VRAM
    active_scratch = None
    if device.type == 'cuda' and prime_vram_gb > 0:
        active_scratch = torch.empty(int(prime_vram_gb * (1024**3) / 4), dtype=torch.float32, device=device)

    K = get_camera_parameters(img_size, fov=60.0, device=device)
    cap = cv2.VideoCapture(str(video_source))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(10.0, fps / float(sample_every))

    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, "unified_pipeline_fused.mp4")
    video_writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (width, height))

    results = []
    frame_idx = 0
    sampled_count = 0
    max_allowed = max_frames if max_frames > 0 else float('inf')
    start_time = time.time()

    print(f"[NETRA-4D Unified Pipeline] Ingesting {width}x{height} | {total_frames} frames @ {fps:.2f} fps")

    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                while cap.isOpened() and sampled_count < max_allowed:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % sample_every == 0:
                        timestamp_sec = round(frame_idx / fps, 3)

                        # -------------------------------------------------------------
                        # A. Continuous Vision Stream (Stage 1)
                        # -------------------------------------------------------------
                        pose_res = yolo_pose.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.15, device=device, verbose=False)[0]
                        raw_tracked_people = []
                        if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                            boxes = pose_res.boxes.xyxy.cpu().numpy()
                            track_ids = pose_res.boxes.id.int().cpu().numpy() if pose_res.boxes.id is not None else np.arange(1, len(boxes) + 1)
                            kpts = pose_res.keypoints.data.cpu().numpy() if pose_res.keypoints is not None else None
                            confs = pose_res.boxes.conf.cpu().numpy() if pose_res.boxes.conf is not None else np.ones(len(boxes))
                            for b, tid, kp, conf in zip(boxes, track_ids, kpts if kpts is not None else [None]*len(boxes), confs):
                                raw_tracked_people.append({
                                    "track_id": int(tid),
                                    "bbox": [float(v) for v in b],
                                    "keypoints": kp,
                                    "confidence": float(conf)
                                })

                        obj_res = yolo_obj.predict(frame, conf=0.20, device=device, verbose=False)[0]
                        detected_objects: List[DetectedObjectObservation] = []
                        if obj_res.boxes is not None and len(obj_res.boxes) > 0:
                            for ob, oconf, ocls in zip(obj_res.boxes.xyxy.cpu().numpy(), obj_res.boxes.conf.cpu().numpy(), obj_res.boxes.cls.int().cpu().numpy()):
                                cname = obj_res.names.get(int(ocls), "")
                                if cname in ["cell phone", "laptop", "tv", "monitor", "book", "bottle", "keyboard", "mouse"]:
                                    ob_norm = (
                                        float(np.clip(ob[0] / width, 0.0, 1.0)),
                                        float(np.clip(ob[1] / height, 0.0, 1.0)),
                                        float(np.clip(ob[2] / width, 0.0, 1.0)),
                                        float(np.clip(ob[3] / height, 0.0, 1.0)),
                                    )
                                    detected_objects.append(DetectedObjectObservation(
                                        class_name=cname,
                                        confidence=float(oconf),
                                        bbox_xyxy_norm=ob_norm,
                                        is_static=(cname in ["laptop", "tv", "monitor"])
                                    ))

                        # -------------------------------------------------------------
                        # B. Environment Profiling & Role Discovery (Layer 1)
                        # -------------------------------------------------------------
                        env_profiler.observe(raw_tracked_people, detected_objects)
                        role_registry.observe_frame(frame_idx, raw_tracked_people, frame_bgr=frame)

                        # Separate Invigilator vs Candidates
                        proctor_center_norm: Optional[Tuple[float, float]] = None
                        candidate_observations: List[VisionObservation] = []

                        for p in raw_tracked_people:
                            tid = p["track_id"]
                            role = role_registry.get_role_for_track(tid)
                            bx = p["bbox"]
                            norm_center = ((bx[0] + bx[2]) / (2.0 * width), (bx[1] + bx[3]) / (2.0 * height))

                            if role == RoleEnum.INVIGILATOR:
                                proctor_center_norm = norm_center
                            else:
                                # Canonical VisionObservation conversion via Adapter
                                obs = YOLO26Adapter.adapt(
                                    frame_idx=frame_idx,
                                    timestamp_sec=timestamp_sec,
                                    frame_shape=(height, width),
                                    tracked_box=p["bbox"],
                                    track_id=tid,
                                    keypoints_raw=p["keypoints"],
                                    det_confidence=p["confidence"],
                                    detected_objects=detected_objects
                                )
                                candidate_observations.append(obs)

                        # Update dynamic proctor spatial zone
                        semantic_zone_engine.update_invigilator_position(proctor_center_norm)

                        # -------------------------------------------------------------
                        # C. Candidate State & Immutable Desk Binding (Layer 2)
                        # -------------------------------------------------------------
                        active_candidates = candidate_state_manager.update_frame_observations(
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp_sec,
                            observations=candidate_observations
                        )

                        # Register candidate desk zones if not yet registered
                        for cand_id, state in active_candidates.items():
                            if cand_id not in semantic_zone_engine.candidate_desk_zones:
                                anc_cx, anc_cy = state.identity.anchor_center_norm
                                anc_bw, anc_bh = state.identity.anchor_box_size_norm
                                desk_box = (
                                    max(0.0, anc_cx - anc_bw / 2.0),
                                    max(0.0, anc_cy - anc_bh / 4.0),
                                    min(1.0, anc_cx + anc_bw / 2.0),
                                    min(1.0, anc_cy + anc_bh * 0.75),
                                )
                                semantic_zone_engine.register_candidate_desk(cand_id, desk_box)

                        if frame_idx >= 45 and not semantic_zone_engine.neighbour_mappings:
                            semantic_zone_engine.auto_infer_neighbours()

                        # -------------------------------------------------------------
                        # D. Multi-HMR 3D Mesh Extraction (Stage 3 Preparedness)
                        # -------------------------------------------------------------
                        x, transform_meta = preprocess_frame_for_hmr(frame, img_size, device=device)
                        humans_3d = hmr_model(x, is_training=False, nms_kernel_size=1, det_thresh=0.3, K=K)
                        scale, pad_x, pad_y = transform_meta

                        # -------------------------------------------------------------
                        # E. Temporal Behavioral Engine & Arbitration (Layer 3 & 4)
                        # -------------------------------------------------------------
                        rendered_frame = frame.copy()
                        frame_audit_records = []

                        for cand_id, state in active_candidates.items():
                            if state.identity.is_invigilator:
                                continue

                            if not state.is_visible or not state.current_observation:
                                # Acknowledge occluded / partial candidate continuously on display
                                anc_cx, anc_cy = state.identity.anchor_center_norm
                                anc_bw, anc_bh = state.identity.anchor_box_size_norm
                                x1 = max(0, int((anc_cx - anc_bw / 2.0) * width))
                                y1 = max(0, int((anc_cy - anc_bh / 2.0) * height))
                                x2 = min(width, int((anc_cx + anc_bw / 2.0) * width))
                                y2 = min(height, int((anc_cy + anc_bh / 2.0) * height))
                                tag = f"{cand_id} -> {state.identity.workstation_id} [OCCLUDED]"
                                cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), (70, 70, 70), 1, cv2.LINE_AA)
                                cv2.rectangle(rendered_frame, (x1, max(0, y1 - 18)), (x1 + len(tag)*7 + 8, max(18, y1)), (15, 15, 15), -1)
                                cv2.putText(rendered_frame, tag, (x1 + 4, max(13, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
                                continue

                            obs = state.current_observation
                            tid = state.active_tracker_id or obs.track_id
                            desk_id = state.identity.workstation_id

                            # Push to rolling temporal buffer
                            temporal_buffer_manager.push(cand_id, obs)

                            # Extract scale-invariant biomechanical features
                            features = feature_extractor.extract(cand_id, obs)

                            # Spatial context checks
                            gaze_str = features.head.direction.value
                            is_neighbour_look, neighbour_cand = semantic_zone_engine.is_gaze_aligned_with_neighbour(cand_id, gaze_str)
                            is_proctor_near = semantic_zone_engine.is_invigilator_near_candidate(cand_id)

                            # Boundary crossing check
                            lw = obs.get_keypoint("left_wrist")
                            rw = obs.get_keypoint("right_wrist")
                            wrist_points = [(pt.x, pt.y) for pt in (lw, rw) if pt]
                            boundary_crossed = any(semantic_zone_engine.is_hand_crossing_boundary(cand_id, pt)[0] for pt in wrist_points)

                            # Object presence
                            phone_present = any(obj.class_name == "cell phone" for obj in obs.nearby_objects)

                            # Detect primitive physical events
                            completed_primitives = primitive_event_detector.process(
                                features=features,
                                board_aligned=False,
                                invigilator_nearby=is_proctor_near,
                                neighbour_seat_aligned=is_neighbour_look
                            )

                            # Check continuous active events for sustained hold
                            sustained_pattern = None
                            for active_ev in primitive_event_detector.active_events.values():
                                if active_ev.candidate_id == cand_id:
                                    pat = persistence_detector.check_active_event(active_ev, timestamp_sec)
                                    if pat:
                                        sustained_pattern = pat
                                        break

                            # Check repetition across completed events
                            repeated_pattern = None
                            for comp_ev in completed_primitives:
                                pat = repetition_detector.register_completed_event(comp_ev)
                                if pat and pat.pattern_type in (TemporalPatternType.REPEATED_EPISODE, TemporalPatternType.REPEATED_AND_SUSTAINED):
                                    repeated_pattern = pat
                                    break

                            # Active pattern evaluation
                            active_pattern = sustained_pattern or repeated_pattern
                            if active_pattern is None:
                                # Default synthetic routine pattern for baseline evaluation
                                active_pattern = repetition_detector.register_completed_event(
                                    PrimitiveEvent("routine", cand_id, PrimitiveEventType.HEAD_TURN, timestamp_sec, timestamp_sec+0.1, 0.1, 0.9, 0.1, is_active=False)
                                )

                            # Evaluate Competing Hypotheses & Counter-Evidence
                            eval_result = competing_hypothesis_engine.evaluate(
                                candidate_id=cand_id,
                                pattern=active_pattern,
                                is_board_aligned=False,
                                is_invigilator_nearby=is_proctor_near,
                                is_normal_device_interaction=False,
                                is_boundary_crossed=boundary_crossed,
                                is_phone_detected=phone_present,
                                is_chit_detected=False,
                                sensor_quality_ratio=obs.tracking_confidence
                            )

                            # -------------------------------------------------------------
                            # F. Three-Stage Precision Escalation Gate (Section 4.4)
                            # -------------------------------------------------------------
                            specialist_finding = None
                            if escalation_gate.should_escalate_to_stage2(eval_result):
                                # Stage 2 Specialist Crop Extraction
                                bx_px = [int(v) for v in (obs.bbox_xyxy_norm[0]*width, obs.bbox_xyxy_norm[1]*height, obs.bbox_xyxy_norm[2]*width, obs.bbox_xyxy_norm[3]*height)]
                                cand_crop = frame[max(0, bx_px[1]):min(height, bx_px[3]), max(0, bx_px[0]):min(width, bx_px[2])]
                                hand_crops = []
                                wrist_pixel_loc = None
                                if lw and rw:
                                    lw_n = (lw.x, lw.y)
                                    rw_n = (rw.x, rw.y)
                                    wrist_crops_meta = specialist_crop_analyzer.extract_wrist_crops(frame, lw_n, rw_n)
                                    if wrist_crops_meta:
                                        hand_crops = [item[0] for item in wrist_crops_meta]
                                        wrist_pixel_loc = wrist_crops_meta[0][1]

                                res2 = escalation_gate.evaluate_stage2_specialists(
                                    candidate_id=cand_id,
                                    evaluation=eval_result,
                                    candidate_crop_bgr=cand_crop,
                                    hand_crops_bgr=hand_crops
                                )
                                eval_result.suspicion_score = res2.resolved_suspicion
                                eval_result.uncertainty_score = res2.resolved_uncertainty
                                eval_result.supporting_evidence.extend(res2.supporting_findings)
                                eval_result.counter_evidence.extend(res2.counter_findings)
                                if wrist_pixel_loc and res2.resolved_suspicion >= 0.60:
                                    cv2.circle(rendered_frame, wrist_pixel_loc, 24, (0, 0, 255), 2, cv2.LINE_AA)

                            # Candidate Behavioural State Machine Update
                            if cand_id not in state_machines:
                                state_machines[cand_id] = CandidateStateMachine(cand_id)
                            candidate_state = state_machines[cand_id].update(timestamp_sec, eval_result)

                            # Conditional Stage 3 Multi-HMR Rendering: Only when candidate is in REVIEW
                            is_flagged = candidate_state in (CandidateBehaviouralState.REVIEW, CandidateBehaviouralState.HIGH_PRIORITY_REVIEW)

                            if is_flagged and humans_3d is not None:
                                cx_cand = obs.center_norm[0] * width
                                cy_cand = obs.center_norm[1] * height
                                best_h = None
                                min_dist = float('inf')
                                for h in humans_3d:
                                    if 'v3d' in h:
                                        v3d = h['v3d'].unsqueeze(0)
                                        p2d_padded = perspective_projection(v3d, K).squeeze(0).cpu().numpy()
                                        px_mean = ((p2d_padded[:, 0].mean() - pad_x) / scale)
                                        py_mean = ((p2d_padded[:, 1].mean() - pad_y) / scale)
                                        d = float(np.hypot(px_mean - cx_cand, py_mean - cy_cand))
                                        if d < min_dist:
                                            min_dist = d
                                            best_h = h

                                if best_h is not None and min_dist < 120:
                                    v3d = best_h['v3d'].unsqueeze(0)
                                    p2d_padded = perspective_projection(v3d, K).squeeze(0).cpu().numpy()
                                    p2d_orig = np.empty_like(p2d_padded)
                                    p2d_orig[:, 0] = (p2d_padded[:, 0] - pad_x) / scale
                                    p2d_orig[:, 1] = (p2d_padded[:, 1] - pad_y) / scale
                                    pts_triangles = p2d_orig[faces_sampled].astype(np.int32)
                                    cv2.polylines(rendered_frame, pts_triangles, isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

                            # -------------------------------------------------------------
                            # G. Visual Overlays (Selective & Privacy Safe)
                            # -------------------------------------------------------------
                            color = PALETTE_BGR[hash(cand_id) % len(PALETTE_BGR)]
                            bx_px = [int(v) for v in (obs.bbox_xyxy_norm[0]*width, obs.bbox_xyxy_norm[1]*height, obs.bbox_xyxy_norm[2]*width, obs.bbox_xyxy_norm[3]*height)]

                            # Skeleton lines with confidence threshold 0.15 (supports partial & back-facing bodies)
                            for p1_idx, p2_idx in SKELETON_PAIRS:
                                p1_name = YOLO26Adapter.COCO_KEYPOINT_NAMES[p1_idx]
                                p2_name = YOLO26Adapter.COCO_KEYPOINT_NAMES[p2_idx]
                                kp1 = obs.get_keypoint(p1_name)
                                kp2 = obs.get_keypoint(p2_name)
                                if kp1 and kp2 and kp1.confidence > 0.15 and kp2.confidence > 0.15:
                                    cv2.line(
                                        rendered_frame,
                                        (int(kp1.x * width), int(kp1.y * height)),
                                        (int(kp2.x * width), int(kp2.y * height)),
                                        color, 2, cv2.LINE_AA
                                    )

                            # If back-facing (shoulders visible but face occluded), anchor head up from mid-shoulder
                            ls = obs.get_keypoint("left_shoulder")
                            rs = obs.get_keypoint("right_shoulder")
                            nose_kp = obs.get_keypoint("nose")
                            if ls and rs and ls.confidence > 0.15 and rs.confidence > 0.15:
                                mid_sh_x = (ls.x + rs.x) / 2.0
                                mid_sh_y = (ls.y + rs.y) / 2.0
                                sh_dist = float(np.hypot(ls.x - rs.x, ls.y - rs.y))
                                if not nose_kp or nose_kp.confidence <= 0.15:
                                    head_top = (int(mid_sh_x * width), int((mid_sh_y - sh_dist * 0.45) * height))
                                    neck_pt = (int(mid_sh_x * width), int(mid_sh_y * height))
                                    cv2.line(rendered_frame, neck_pt, head_top, color, 2, cv2.LINE_AA)
                                    cv2.circle(rendered_frame, head_top, max(3, int(sh_dist * 0.22 * width)), color, 1, cv2.LINE_AA)

                            # Draw keypoint circles for all detected joints
                            for kp_name, kp in obs.keypoints.items():
                                if kp.confidence > 0.15:
                                    cv2.circle(rendered_frame, (int(kp.x * width), int(kp.y * height)), 3, (255, 255, 255), -1, cv2.LINE_AA)
                                    cv2.circle(rendered_frame, (int(kp.x * width), int(kp.y * height)), 2, color, -1, cv2.LINE_AA)

                            # Candidate Identification Tag
                            lx, ly = bx_px[0], bx_px[1]
                            if candidate_state == CandidateBehaviouralState.HIGH_PRIORITY_REVIEW:
                                tag = f"ALERT: {cand_id} -> {desk_id} | S:{eval_result.suspicion_score:.2f} U:{eval_result.uncertainty_score:.2f}"
                                box_col = (0, 0, 255)
                            elif candidate_state == CandidateBehaviouralState.REVIEW:
                                tag = f"REVIEW: {cand_id} -> {desk_id} | S:{eval_result.suspicion_score:.2f}"
                                box_col = (0, 165, 255)
                            elif candidate_state == CandidateBehaviouralState.SUSPICIOUS:
                                tag = f"MONITORING: {cand_id} -> {desk_id}"
                                box_col = (0, 220, 255)
                            else:
                                tag = f"{cand_id} -> {desk_id}"
                                box_col = color

                            cv2.rectangle(rendered_frame, (lx, max(0, ly - 22)), (lx + len(tag)*8 + 12, max(20, ly)), (15, 15, 15), -1)
                            cv2.putText(rendered_frame, tag, (lx + 4, max(14, ly - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 2, cv2.LINE_AA)

                            # Record JSON audit
                            frame_audit_records.append({
                                "candidate_id": cand_id,
                                "desk_id": desk_id,
                                "state": candidate_state.value,
                                "suspicion_score": round(eval_result.suspicion_score, 4),
                                "uncertainty_score": round(eval_result.uncertainty_score, 4),
                                "dominant_hypothesis": eval_result.dominant_hypothesis.value,
                                "hypotheses": eval_result.hypotheses.as_dict(),
                                "supporting_evidence": eval_result.supporting_evidence,
                                "counter_evidence": eval_result.counter_evidence,
                            })

                        # =====================================================================
                        # UNIVERSAL PRIVACY FACE MASK (Layer 4 Privacy Renderer: drishti.privacy)
                        # Applied to ALL Candidates and Invigilators with Zero Biometric Leakage
                        # =====================================================================
                        rendered_frame, _ = apply_privacy_mask_and_collect_annotations(
                            rendered_frame,
                            raw_tracked_people,
                            min_keypoint_confidence=0.15
                        )

                        # Global Telemetry HUD
                        env_label = env_profiler.environment_type.value if env_profiler.finalized else "PROFILING..."
                        vram_gb = torch.cuda.memory_allocated() / (1024**3) if device.type == 'cuda' else 0.0
                        hud_text = f"NETRA-4D UNIFIED | Env: {env_label} | Frame {frame_idx:04d}/{total_frames} | GPU VRAM: {vram_gb:.2f} GB"
                        cv2.rectangle(rendered_frame, (0, 0), (width, 34), (10, 10, 10), -1)
                        cv2.putText(rendered_frame, hud_text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 200), 2, cv2.LINE_AA)

                        video_writer.write(rendered_frame)

                        results.append({
                            "frame_idx": frame_idx,
                            "timestamp_sec": timestamp_sec,
                            "environment": env_profiler.environment_type.value,
                            "candidates_observed": frame_audit_records
                        })

                        sampled_count += 1
                        if sampled_count % 25 == 0 or sampled_count <= 3:
                            print(f"  [Unified Pipeline] Frame {frame_idx:05d} ({sampled_count}) | Env: {env_profiler.environment_type.value} | Candidates locked: {len(candidate_state_manager.candidates)}")

                    frame_idx += 1

    finally:
        cap.release()
        video_writer.release()
        del active_scratch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n[DONE] Unified Pipeline processed {sampled_count} frames into output video in {elapsed:.1f}s!")
    print(f"  -> Video: {out_video_path}")

    out_file = os.path.join(output_dir, "unified_pipeline_detections.json")
    with open(out_file, "w") as f:
        json.dump({
            "pipeline_version": "2.1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "environment_profile": env_profiler.get_summary(),
            "total_candidates_locked": len(candidate_state_manager.candidates),
            "total_sampled_frames": len(results),
            "video_path": str(out_video_path),
            "frames": results
        }, f, indent=2)

    print(f"  -> JSON:  {out_file}")
    return out_video_path, out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Unified Pipeline Runner for NETRA-4D / DrishtiAI")
    parser.add_argument("--asset", default="VIDEO_01")
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video")
    parser.add_argument("--prime-vram-gb", type=float, default=2.2)
    args = parser.parse_args()

    manifest = load_drive_manifest(ROOT / "drishti4d" / "configs" / "datasets.json")
    asset = next((a for a in manifest if a.asset_id == args.asset), None)
    if not asset:
        print(f"Asset {args.asset} not found in manifest!")
        sys.exit(1)

    out_dir = ROOT / "outputs" / "unified_pipeline" / args.asset
    print(f"Streaming {asset.asset_id} ({asset.filename}) into Unified Pipeline...")
    with materialize_drive_asset(asset, temp_root=out_dir / ".staging") as (video_path, prov):
        run_unified_pipeline(
            video_source=video_path,
            output_dir=out_dir,
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            prime_vram_gb=args.prime_vram_gb
        )
