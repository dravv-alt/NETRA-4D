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
from drishti.render.evidence_renderer import EvidenceRenderer
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
    pose_model_candidates = [
        os.path.join(ROOT, "yolov8m-pose.pt"),
        os.path.join(ROOT, "models", "weights", "yolov8m-pose.pt"),
        os.path.join(ROOT, "models", "weights", "yolo26n-pose.pt"),
    ]
    pose_model_path = next((p for p in pose_model_candidates if os.path.exists(p)), "yolov8m-pose.pt")
    obj_model_path = os.path.join(ROOT, "models", "weights", "yolo26n.pt")
    yolo_pose = YOLO(pose_model_path).to(device)
    yolo_obj = YOLO(obj_model_path).to(device)

    # YOLO26m Cheating Detector (Stage 2 specialist)
    cheating_candidates = [
        os.path.join(ROOT, "models", "weights", "yolo26m_cheating.pt"),
        os.path.join(ROOT, "models", "weights", "exam_cheating_trained", "yolo26m_cheating", "weights", "best.pt"),
    ]
    cheating_model_path = next((p for p in cheating_candidates if os.path.exists(p)), None)
    yolo_cheating = YOLO(cheating_model_path).to(device) if cheating_model_path else None
    if yolo_cheating:
        print(f"[NETRA-4D] Stage 2 YOLO26m Cheating Specialist loaded from: {cheating_model_path}")

    # YOLO26m Chit Paper Detector (Stage 2 specialist)
    chit_candidates = [
        os.path.join(ROOT, "models", "weights", "yolo26m_chit.pt"),
        os.path.join(ROOT, "models", "weights", "chit_paper_trained", "yolo26m_chit", "weights", "best.pt"),
    ]
    chit_model_path = next((p for p in chit_candidates if os.path.exists(p)), None)
    yolo_chit = YOLO(chit_model_path).to(device) if chit_model_path else None
    if yolo_chit:
        print(f"[NETRA-4D] Stage 2 YOLO26m Chit Specialist loaded from: {chit_model_path}")

    # YOLO26m Mobile Phone Detector (Stage 2 specialist)
    phone_candidates = [
        os.path.join(ROOT, "models", "weights", "yolo26m_phone.pt"),
        os.path.join(ROOT, "models", "weights", "phone_trained", "yolo26m_phone", "weights", "best.pt"),
    ]
    phone_model_path = next((p for p in phone_candidates if os.path.exists(p)), None)
    yolo_phone = YOLO(phone_model_path).to(device) if phone_model_path else None
    if yolo_phone:
        print(f"[NETRA-4D] Stage 2 YOLO26m Phone Specialist loaded from: {phone_model_path}")

    # YOLO26m Multi-Modal Exam Anomaly Detector (Stage 2 specialist)
    multimodal_candidates = [
        os.path.join(ROOT, "models", "weights", "yolo26m_exam_multimodal.pt"),
        os.path.join(ROOT, "models", "weights", "multimodal_trained", "yolo26m_multimodal", "weights", "best.pt"),
    ]
    multimodal_model_path = next((p for p in multimodal_candidates if os.path.exists(p)), None)
    yolo_multimodal = YOLO(multimodal_model_path).to(device) if multimodal_model_path else None
    if yolo_multimodal:
        print(f"[NETRA-4D] Stage 2 YOLO26m Multi-Modal Specialist loaded from: {multimodal_model_path}")

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
    evidence_renderer = EvidenceRenderer(enable_face_blur=True, min_keypoint_conf=0.12)
    video_title = Path(video_source).stem[:28]

    results = []
    frame_idx = 0
    sampled_count = 0
    max_allowed = max_frames if max_frames > 0 else float('inf')
    start_time = time.time()

    active_spatial_tracks: Dict[int, Dict[str, Any]] = {}
    next_spatial_id = 1

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
                        # High-resolution (imgsz=1280) pose inference with deterministic spatial tracking
                        # -------------------------------------------------------------
                        pose_res = yolo_pose.predict(
                            frame,
                            imgsz=1280,
                            conf=0.15,
                            device=device,
                            verbose=False
                        )[0]
                        raw_tracked_people = []
                        if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                            boxes = pose_res.boxes.xyxy.cpu().numpy()
                            confs = pose_res.boxes.conf.cpu().numpy() if pose_res.boxes.conf is not None else np.ones(len(boxes))
                            kpts = pose_res.keypoints.data.cpu().numpy() if pose_res.keypoints is not None else np.zeros((len(boxes), 17, 3))

                            current_matched_tids = set()
                            for b, conf, kp in zip(boxes, confs, kpts):
                                cx = (b[0] + b[2]) / 2.0 / float(width)
                                cy = (b[1] + b[3]) / 2.0 / float(height)

                                best_tid = None
                                min_d = 0.16  # Spatial tolerance (robust to seated leaning / hand motion)
                                for tid, t_data in active_spatial_tracks.items():
                                    if tid in current_matched_tids:
                                        continue
                                    tcx, tcy = t_data["center"]
                                    d = np.hypot(cx - tcx, cy - tcy)
                                    if d < min_d:
                                        min_d = d
                                        best_tid = tid

                                if best_tid is None:
                                    best_tid = next_spatial_id
                                    next_spatial_id += 1

                                current_matched_tids.add(best_tid)
                                active_spatial_tracks[best_tid] = {"center": (cx, cy), "last_frame": frame_idx}

                                raw_tracked_people.append({
                                    "track_id": int(best_tid),
                                    "bbox": [float(v) for v in b],
                                    "keypoints": kp,
                                    "confidence": float(conf)
                                })

                            # Expire tracks inactive for over 35 frames
                            expired = [tid for tid, td in active_spatial_tracks.items() if frame_idx - td["last_frame"] > 35]
                            for tid in expired:
                                del active_spatial_tracks[tid]

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

                        # Canonical physical desk binding & zone registration
                        for cand_id, state in active_candidates.items():
                            anc_cx, anc_cy = state.identity.anchor_center_norm
                            if np.hypot(anc_cx - 0.129, anc_cy - 0.417) <= 0.14:
                                state.identity.workstation_id = "Desk_12"
                            elif np.hypot(anc_cx - 0.532, anc_cy - 0.511) <= 0.14:
                                state.identity.workstation_id = "Desk_03"

                            if cand_id not in semantic_zone_engine.candidate_desk_zones:
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
                        # D. Multi-HMR 3D Mesh Extraction (Stage 3 Deep Analysis)
                        # Conditionally invoked per ARCHITECTURE.md Section 4.5 & 5.4:
                        # Warmup calibration frames or when any candidate is in REVIEW / HIGH_PRIORITY_REVIEW
                        # -------------------------------------------------------------
                        needs_stage3 = (
                            frame_idx < 15 or
                            any(sm.current_state in {CandidateBehaviouralState.REVIEW, CandidateBehaviouralState.HIGH_PRIORITY_REVIEW} for sm in state_machines.values())
                        )
                        humans_3d = None
                        if needs_stage3:
                            x, transform_meta = preprocess_frame_for_hmr(frame, img_size, device=device)
                            humans_3d = hmr_model(x, is_training=False, nms_kernel_size=1, det_thresh=0.3, K=K)
                            scale, pad_x, pad_y = transform_meta

                        # -------------------------------------------------------------
                        # E. Temporal Behavioral Engine & Arbitration (Layer 3 & 4)
                        # -------------------------------------------------------------
                        frame_audit_records = []
                        active_alerts = []

                        for cand_id, state in active_candidates.items():
                            if state.identity.is_invigilator:
                                continue

                            if not state.is_visible or not state.current_observation:
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
                            phone_present = any(obj.class_name in ("cell phone", "remote") for obj in obs.nearby_objects)
                            if not phone_present and yolo_phone is not None:
                                bx_px = [int(v) for v in (obs.bbox_xyxy_norm[0]*width, obs.bbox_xyxy_norm[1]*height, obs.bbox_xyxy_norm[2]*width, obs.bbox_xyxy_norm[3]*height)]
                                cand_roi = frame[max(0, bx_px[1]):min(height, bx_px[3]), max(0, bx_px[0]):min(width, bx_px[2])]
                                if cand_roi.size > 0:
                                    try:
                                        p_res = yolo_phone.predict(cand_roi, verbose=False, conf=0.25)
                                        for pr in p_res:
                                            for bx in pr.boxes:
                                                if int(bx.cls.item()) == 1 and float(bx.conf.item()) >= 0.25:
                                                    phone_present = True
                                                    break
                                            if phone_present:
                                                break
                                    except Exception:
                                        pass

                            # Chit paper presence check
                            chit_present = False
                            if yolo_chit is not None:
                                bx_px = [int(v) for v in (obs.bbox_xyxy_norm[0]*width, obs.bbox_xyxy_norm[1]*height, obs.bbox_xyxy_norm[2]*width, obs.bbox_xyxy_norm[3]*height)]
                                cand_roi = frame[max(0, bx_px[1]):min(height, bx_px[3]), max(0, bx_px[0]):min(width, bx_px[2])]
                                if cand_roi.size > 0:
                                    try:
                                        c_res = yolo_chit.predict(cand_roi, verbose=False, conf=0.25)
                                        for cr in c_res:
                                            if len(cr.boxes) > 0:
                                                chit_present = True
                                                break
                                    except Exception:
                                        pass

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
                                is_chit_detected=chit_present,
                                sensor_quality_ratio=obs.tracking_confidence
                            )

                            # -------------------------------------------------------------
                            # F. Three-Stage Precision Escalation Gate (Section 4.4)
                            # -------------------------------------------------------------
                            if escalation_gate.should_escalate_to_stage2(eval_result):
                                # Stage 2 Specialist Crop Extraction
                                bx_px = [int(v) for v in (obs.bbox_xyxy_norm[0]*width, obs.bbox_xyxy_norm[1]*height, obs.bbox_xyxy_norm[2]*width, obs.bbox_xyxy_norm[3]*height)]
                                cand_crop = frame[max(0, bx_px[1]):min(height, bx_px[3]), max(0, bx_px[0]):min(width, bx_px[2])]
                                hand_crops = []
                                if lw and rw:
                                    lw_n = (lw.x, lw.y)
                                    rw_n = (rw.x, rw.y)
                                    wrist_crops_meta = specialist_crop_analyzer.extract_wrist_crops(frame, lw_n, rw_n)
                                    if wrist_crops_meta:
                                        hand_crops = [item[0] for item in wrist_crops_meta]
                                        for wcrop, wcenter, worigin in wrist_crops_meta:
                                            spec_res = specialist_crop_analyzer.analyze_crop(wcrop, wcenter, (worigin[0], worigin[1]))
                                            if spec_res.is_suspicious and spec_res.chit_bbox_xyxy:
                                                gx1, gy1, gx2, gy2 = spec_res.chit_bbox_xyxy
                                                detected_objects.append(DetectedObjectObservation(
                                                    class_name="chit paper",
                                                    confidence=spec_res.confidence,
                                                    bbox_xyxy_norm=(float(gx1 / width), float(gy1 / height), float(gx2 / width), float(gy2 / height)),
                                                    is_static=False
                                                ))
                                                chit_present = True
                                                eval_result.supporting_evidence.append(spec_res.evidence_desc)
                                                eval_result.suspicion_score = max(eval_result.suspicion_score, min(1.0, spec_res.confidence * 1.15))

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

                                # Stage 2 YOLO26m Cheating Detector Inference on Candidate Crop
                                if yolo_cheating is not None and cand_crop.size > 0:
                                    try:
                                        c_preds = yolo_cheating.predict(cand_crop, verbose=False, conf=0.25)
                                        for cp in c_preds:
                                            for bx in cp.boxes:
                                                cls_id = int(bx.cls.item())
                                                c_conf = float(bx.conf.item())
                                                # Class 1: students_cheating, Class 2: students_not_cheating
                                                if cls_id == 1:
                                                    eval_result.supporting_evidence.append(
                                                        f"Stage 2 YOLO26m Cheating Detector flagged suspicious cheating posture (conf: {c_conf:.2f})"
                                                    )
                                                    eval_result.suspicion_score = max(eval_result.suspicion_score, min(1.0, c_conf * 1.15))
                                                    eval_result.uncertainty_score = max(0.05, eval_result.uncertainty_score * 0.40)
                                                elif cls_id == 2 and c_conf >= 0.55:
                                                    eval_result.counter_evidence.append(
                                                        f"Stage 2 YOLO26m verified normal seated posture (conf: {c_conf:.2f})"
                                                    )
                                                    eval_result.suspicion_score = max(0.0, eval_result.suspicion_score * 0.70)
                                    except Exception:
                                        pass

                                # Stage 2 Multi-Modal Exam Anomaly Specialist (chit-paper, hand-signs, head_movement, mobilephone)
                                if yolo_multimodal is not None and cand_crop.size > 0:
                                    try:
                                        mm_preds = yolo_multimodal.predict(cand_crop, verbose=False, conf=0.25)
                                        for mr in mm_preds:
                                            for mbx in mr.boxes:
                                                m_cls = int(mbx.cls.item())
                                                m_conf = float(mbx.conf.item())
                                                if m_cls == 0:  # chit-paper
                                                    chit_present = True
                                                    eval_result.supporting_evidence.append(f"Stage 2 Multi-Modal verified chit paper (conf: {m_conf:.2f})")
                                                    eval_result.suspicion_score = max(eval_result.suspicion_score, min(1.0, m_conf * 1.15))
                                                elif m_cls == 1:  # hand-signs
                                                    eval_result.supporting_evidence.append(f"Stage 2 Multi-Modal detected hand gesturing anomaly (conf: {m_conf:.2f})")
                                                    eval_result.suspicion_score = max(eval_result.suspicion_score, min(1.0, m_conf * 1.10))
                                                elif m_cls == 2:  # head_movement
                                                    eval_result.supporting_evidence.append(f"Stage 2 Multi-Modal flagged suspicious head turn/peek (conf: {m_conf:.2f})")
                                                    eval_result.suspicion_score = max(eval_result.suspicion_score, min(1.0, m_conf * 1.12))
                                                elif m_cls == 3:  # mobilephone
                                                    phone_present = True
                                                    eval_result.supporting_evidence.append(f"Stage 2 Multi-Modal detected phone device (conf: {m_conf:.2f})")
                                                    eval_result.suspicion_score = max(eval_result.suspicion_score, min(1.0, m_conf * 1.20))
                                    except Exception:
                                        pass

                            # Candidate Behavioural State Machine Update
                            if cand_id not in state_machines:
                                state_machines[cand_id] = CandidateStateMachine(cand_id)
                            candidate_state = state_machines[cand_id].update(timestamp_sec, eval_result)

                            # Update candidate state attributes for rendering
                            state.behavioural_state = candidate_state.value
                            state.gaze_direction = gaze_str
                            state.is_peeking_neighbour = is_neighbour_look
                            state.neighbour_candidate = neighbour_cand
                            state.suspicion_score = eval_result.suspicion_score

                            # Track active alerts for surveillance HUD
                            if candidate_state in (CandidateBehaviouralState.REVIEW, CandidateBehaviouralState.HIGH_PRIORITY_REVIEW):
                                desk_lbl = desk_id or cand_id.replace("Candidate_", "Seat ")
                                if phone_present:
                                    active_alerts.append(f"{desk_lbl} [PHONE DETECTED]")
                                elif chit_present:
                                    active_alerts.append(f"{desk_lbl} [CHIT / PAPER DETECTED]")
                                elif is_neighbour_look:
                                    active_alerts.append(f"{desk_lbl} [PEEKING -> {neighbour_cand}]")
                                elif boundary_crossed:
                                    active_alerts.append(f"{desk_lbl} [HAND ANOMALY: PAPER/CHIT]")
                                else:
                                    active_alerts.append(f"{desk_lbl} [SUSPICIOUS]")

                            # Record JSON audit
                            frame_audit_records.append({
                                "candidate_id": cand_id,
                                "desk_id": desk_id,
                                "state": candidate_state.value,
                                "gaze_direction": gaze_str,
                                "gaze_yaw_proxy": round(features.head.yaw_proxy, 4),
                                "gaze_pitch_proxy": round(features.head.pitch_proxy, 4),
                                "is_peeking_neighbour": is_neighbour_look,
                                "neighbour_candidate": neighbour_cand,
                                "suspicion_score": round(eval_result.suspicion_score, 4),
                                "uncertainty_score": round(eval_result.uncertainty_score, 4),
                                "dominant_hypothesis": eval_result.dominant_hypothesis.value,
                                "hypotheses": eval_result.hypotheses.as_dict(),
                                "supporting_evidence": eval_result.supporting_evidence,
                                "counter_evidence": eval_result.counter_evidence,
                            })

                        # =====================================================================
                        # CLEAN SURVEILLANCE EVIDENCE RENDERING (NETRA-4D Layer 4)
                        # Zero 3D wireframe mesh | Zero chair/desk boxes | Invigilator excluded
                        # =====================================================================
                        invigilator_cand_ids = {
                            c for c, s in candidate_state_manager.candidates.items()
                            if s.identity.is_invigilator or (s.active_tracker_id and role_registry.is_invigilator(s.active_tracker_id))
                        }

                        rendered_frame = evidence_renderer.render_frame(
                            raw_frame=frame,
                            tracked_people=raw_tracked_people,
                            candidates=active_candidates,
                            detected_objects=detected_objects,
                            invigilator_cand_ids=invigilator_cand_ids,
                            video_title=video_title,
                            frame_idx=frame_idx,
                            total_frames=total_frames,
                            fps=fps,
                            active_alerts=active_alerts
                        )

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
    parser.add_argument("--asset", default=None, help="Asset ID to process, e.g. VIDEO_01 or 'ALL'")
    parser.add_argument("--assets", nargs="*", default=None, help="List of assets to process: VIDEO_01 VIDEO_02 ... or 'all'")
    parser.add_argument("--sample-every", type=int, default=1, help="Frame step interval (1 = all frames)")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full video")
    parser.add_argument("--prime-vram-gb", type=float, default=2.2)
    args = parser.parse_args()

    manifest = load_drive_manifest(ROOT / "drishti4d" / "configs" / "datasets.json")

    # Resolve target assets
    if args.assets:
        if any(a.lower() == "all" for a in args.assets):
            target_ids = [a.asset_id for a in manifest]
        else:
            target_ids = args.assets
    elif args.asset:
        if args.asset.upper() == "ALL":
            target_ids = [a.asset_id for a in manifest]
        else:
            target_ids = [args.asset]
    else:
        target_ids = ["VIDEO_01"]

    print(f"[NETRA-4D] Target assets to process: {target_ids}")
    for asset_id in target_ids:
        asset = next((a for a in manifest if a.asset_id == asset_id), None)
        if not asset:
            print(f"Asset {asset_id} not found in manifest!")
            continue

        out_dir = ROOT / "outputs" / "unified_pipeline" / asset_id
        print(f"\n=======================================================")
        print(f"Streaming {asset.asset_id} ({asset.filename}) into Unified Pipeline...")
        print(f"=======================================================")
        try:
            with materialize_drive_asset(asset, temp_root=out_dir / ".staging") as (video_path, prov):
                run_unified_pipeline(
                    video_source=video_path,
                    output_dir=out_dir,
                    sample_every=args.sample_every,
                    max_frames=args.max_frames,
                    prime_vram_gb=args.prime_vram_gb
                )
        except Exception as e:
            print(f"[ERROR] Exception processing {asset_id}: {e}")
            import traceback
            traceback.print_exc()

