"""
Complete Situational Pipeline Runner for NETRA-4D Examination Monitoring.
Implements the exact end-to-end flowchart:
CCTV Ingestion -> Quality Gate -> Environment Profiling (CBT vs. Classroom) ->
Initial Role Discovery -> Immutable Role Registry (Invigilator Re-ID & Exclusion) ->
Candidate Discovery -> Immutable Candidate ID & 1:1 Desk Binding ->
Pose + Object Fusion -> Temporal Reasoning -> 3-Stage Cascaded Arbitration ->
Privacy-Safe Renderer -> Annotated MP4 + JSON Records.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

# Ensure project and multi-hmr imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "multi-hmr"))

from drishti.data.google_drive import load_drive_manifest, materialize_drive_asset
from drishti.pipeline.environment import EnvironmentProfiler, EnvironmentType
from drishti.pipeline.role_registry import ImmutableRoleRegistry, RoleEnum
from drishti.pipeline.candidate_registry import ImmutableCandidateRegistry
from drishti.pipeline.specialist_crop import SpecialistHandCropAnalyzer
from drishti.pipeline.arbitrator import EvidenceArbitrator, FinalReviewState
from drishti.preprocessing import PreprocessingPipeline
from drishti.data.video_policy import bounded_video_source
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

def get_camera_parameters(img_size, fov=60, device=torch.device('cuda:0')):
    K = torch.eye(3)
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0, 0], K[1, 1] = focal, focal
    K[0, -1], K[1, -1] = img_size // 2, img_size // 2
    return K.unsqueeze(0).to(device)

def preprocess_frame(frame_bgr, img_size, device=torch.device('cuda:0')):
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

def apply_face_blur(frame_bgr, head_box):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in head_box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        head_roi = frame_bgr[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(head_roi, (27, 27), 30)
        frame_bgr[y1:y2, x1:x2] = blurred
    return frame_bgr

def run_situational_pipeline(
    video_source,
    output_dir,
    sample_every=2,
    max_frames=0,
    prime_vram_gb=2.2,
    preprocessing_config=None
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[NETRA-4D Situational Pipeline] Running on Dedicated GPU: {device}")

    # Initialize Managers
    env_profiler = EnvironmentProfiler(warmup_frames=30)
    role_registry = ImmutableRoleRegistry(warmup_frames=45)
    candidate_registry = ImmutableCandidateRegistry()
    preprocessing_pipeline = PreprocessingPipeline(config=preprocessing_config)
    specialist_crop = SpecialistHandCropAnalyzer(
        crop_size=256,
        crop_enhancer=preprocessing_pipeline.enhance_crop,
    )
    arbitrator = EvidenceArbitrator()

    # Load YOLO26 Detectors
    pose_model_path = os.path.join(ROOT, "models", "weights", "yolo26n-pose.pt")
    obj_model_path = os.path.join(ROOT, "models", "weights", "yolo26n.pt")
    yolo_pose = YOLO(pose_model_path).to(device)
    yolo_obj = YOLO(obj_model_path).to(device)

    # Multi-HMR & SMPL-X
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

    # Pre-allocate GPU cache
    active_scratch = None
    if device.type == 'cuda' and prime_vram_gb > 0:
        active_scratch = torch.empty(int(prime_vram_gb * (1024**3) / 4), dtype=torch.float32, device=device)

    K = get_camera_parameters(img_size, fov=60, device=device)
    video_source, _ = bounded_video_source(video_source)
    cap = cv2.VideoCapture(str(video_source))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = max(10.0, fps / float(sample_every))

    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, "situational_pipeline_fused.mp4")
    video_writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (width, height))

    results = []
    frame_idx = 0
    sampled_count = 0
    max_allowed = max_frames if max_frames > 0 else float('inf')
    start_time = time.time()

    print(f"[NETRA-4D Situational Pipeline] Ingesting {width}x{height} | {total_frames} frames @ {fps:.2f} fps")

    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                while cap.isOpened() and sampled_count < max_allowed:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % sample_every == 0:
                        # Deep Preprocessing: Assess quality and conditionally enhance low-quality frames
                        enhanced_frame, quality_report = preprocessing_pipeline.process(frame, frame_idx=frame_idx)

                        # 1. YOLO26 Pose + ByteTrack
                        pose_res = yolo_pose.track(enhanced_frame, persist=True, tracker="bytetrack.yaml", conf=0.25, device=device, verbose=False)[0]
                        tracked_people = []
                        if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                            boxes = pose_res.boxes.xyxy.cpu().numpy()
                            track_ids = pose_res.boxes.id.int().cpu().numpy() if pose_res.boxes.id is not None else np.arange(1, len(boxes) + 1)
                            kpts = pose_res.keypoints.data.cpu().numpy() if pose_res.keypoints is not None else None
                            for b, tid, kp in zip(boxes, track_ids, kpts if kpts is not None else [None]*len(boxes)):
                                tracked_people.append({"track_id": int(tid), "bbox": [float(v) for v in b], "keypoints": kp})

                        # 2. YOLO26 Object Detection
                        obj_res = yolo_obj.predict(enhanced_frame, conf=0.20, device=device, verbose=False)[0]
                        detected_objects = []
                        if obj_res.boxes is not None and len(obj_res.boxes) > 0:
                            for ob, oconf, ocls in zip(obj_res.boxes.xyxy.cpu().numpy(), obj_res.boxes.conf.cpu().numpy(), obj_res.boxes.cls.int().cpu().numpy()):
                                cname = obj_res.names.get(int(ocls), "")
                                if cname in ["cell phone", "laptop", "tv", "monitor", "book", "bottle", "keyboard", "mouse"]:
                                    detected_objects.append({"class_name": cname, "confidence": float(oconf), "bbox": [float(v) for v in ob]})

                        # 3. Environment Profiling Gate (Warmup)
                        env_profiler.observe(tracked_people, detected_objects)

                        # 4. Role Discovery & Invigilator Lock
                        role_registry.observe_frame(frame_idx, tracked_people, frame_bgr=frame)

                        # 5. Candidate Discovery & 1:1 Desk Binding
                        assigned_entities = []
                        for p in tracked_people:
                            tid = p["track_id"]
                            role = role_registry.get_role_for_track(tid)
                            bbox = p["bbox"]
                            center_norm = ((bbox[0] + bbox[2]) / (2.0 * width), (bbox[1] + bbox[3]) / (2.0 * height))

                            if role == RoleEnum.INVIGILATOR:
                                cand_id = "INVIGILATOR"
                                desk_id = "AISLE"
                            else:
                                cand_id = candidate_registry.get_candidate_for_track(tid, center_norm)
                                lock = candidate_registry.get_desk_binding(cand_id)
                                desk_id = lock.desk_id if lock else "Desk_Unassigned"

                            assigned_entities.append({
                                "person": p,
                                "role": role,
                                "candidate_id": cand_id,
                                "desk_id": desk_id
                            })

                        # Deduplicate entities in case YOLO produced two overlapping tracker IDs for the same seat
                        unique_entities = {}
                        for ent in assigned_entities:
                            cid = ent["candidate_id"]
                            if cid not in unique_entities or (ent["person"]["bbox"][2] - ent["person"]["bbox"][0]) > (unique_entities[cid]["person"]["bbox"][2] - unique_entities[cid]["person"]["bbox"][0]):
                                unique_entities[cid] = ent
                        assigned_entities = list(unique_entities.values())

                        # 6. Multi-HMR 3D Mesh Recovery
                        x, transform_meta = preprocess_frame(frame, img_size, device=device)
                        humans_3d = hmr_model(x, is_training=False, nms_kernel_size=1, det_thresh=0.3, K=K)
                        scale, pad_x, pad_y = transform_meta

                        # 7. Render Base Canvas & Overlays
                        rendered_frame = frame.copy()
                        overlay_mesh = frame.copy()

                        frame_arbitrations = []

                        # Process each person
                        for entity in assigned_entities:
                            p = entity["person"]
                            role = entity["role"]
                            cand_id = entity["candidate_id"]
                            desk_id = entity["desk_id"]
                            tid = p["track_id"]
                            bbox = p["bbox"]
                            kpts = p["keypoints"]
                            is_inv = (role == RoleEnum.INVIGILATOR)

                            # HARD RULE: Invigilator Exclusion from Visual Overlay
                            # Track: YES (in logs), Monitor: YES, Visual overlay: NO standing-person annotations
                            if is_inv:
                                arb = arbitrator.arbitrate("INVIGILATOR", "AISLE", [], None, is_invigilator=True)
                                frame_arbitrations.append(arb)
                                continue

                            # Color assignment for Candidates
                            color = PALETTE_BGR[hash(cand_id) % len(PALETTE_BGR)]

                            # Privacy Face Blur
                            if kpts is not None and len(kpts) >= 5 and kpts[0][2] > 0.2:
                                head_w = abs(bbox[2] - bbox[0]) * 0.35
                                hx1, hy1 = int(kpts[0][0] - head_w), int(kpts[0][1] - head_w)
                                hx2, hy2 = int(kpts[0][0] + head_w), int(kpts[0][1] + head_w)
                                rendered_frame = apply_face_blur(rendered_frame, (hx1, hy1, hx2, hy2))

                            # Draw YOLO Pose Skeleton
                            if kpts is not None:
                                for p1, p2 in SKELETON_PAIRS:
                                    if p1 < len(kpts) and p2 < len(kpts) and kpts[p1][2] > 0.3 and kpts[p2][2] > 0.3:
                                        cv2.line(rendered_frame, (int(kpts[p1][0]), int(kpts[p1][1])), (int(kpts[p2][0]), int(kpts[p2][1])), color, 2, cv2.LINE_AA)

                            # Check Dual-Wrist Convergence & Chit Proposal (Gate behind Hand-Below/Excursion or Head Deviation)
                            proposals = []
                            specialist_res = None
                            if not is_inv and kpts is not None and len(kpts) >= 11:
                                lw, rw = kpts[9], kpts[10]  # Left, Right Wrists
                                ls, rs = kpts[5], kpts[6]   # Left, Right Shoulders
                                nose = kpts[0]              # Nose for head gaze orientation
                                
                                if lw[2] > 0.3 and rw[2] > 0.3:
                                    wrist_dist = np.hypot(lw[0] - rw[0], lw[1] - rw[1])
                                    shoulders_y = (ls[1] + rs[1]) / 2.0 if (ls[2] > 0.2 and rs[2] > 0.2) else (bbox[1] + 0.3 * (bbox[3] - bbox[1]))
                                    hands_on_desk = (lw[1] > shoulders_y and rw[1] > shoulders_y)

                                    # Check head orientation deviation (looking down/sideways away from straight posture)
                                    head_deviated = False
                                    if nose[2] > 0.2 and ls[2] > 0.2 and rs[2] > 0.2:
                                        shoulder_mid_x = (ls[0] + rs[0]) / 2.0
                                        shoulder_span = max(abs(rs[0] - ls[0]), 20.0)
                                        head_deviated = abs(nose[0] - shoulder_mid_x) / shoulder_span > 0.32 or (nose[1] > shoulders_y - 15)

                                    # Chit proposal only triggers when hands are tightly manipulating slip AND head is averted/suspicious
                                    if hands_on_desk and wrist_dist < 50 and head_deviated:
                                        # Trigger Stage 2 Specialist Hand Crop!
                                        lw_norm = (lw[0]/width, lw[1]/height)
                                        rw_norm = (rw[0]/width, rw[1]/height)
                                        crops = specialist_crop.extract_wrist_crops(frame, lw_norm, rw_norm)
                                        for crop_img, crop_c in crops:
                                            res = specialist_crop.analyze_crop(crop_img, crop_c)
                                            if res.is_suspicious:
                                                specialist_res = res
                                                proposals.append({"event_type": "DESK_CHIT_MANIPULATION", "confidence": 0.78})
                                                # Draw specialist detection crop circle
                                                cv2.circle(rendered_frame, crop_c, 24, (0, 0, 255), 2, cv2.LINE_AA)
                                                cv2.putText(rendered_frame, f"CHIT ({res.confidence:.2f})", (crop_c[0] - 40, crop_c[1] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                                                break

                            # Arbitrate Event
                            arb = arbitrator.arbitrate(cand_id, desk_id, proposals, specialist_res, is_invigilator=is_inv)
                            frame_arbitrations.append(arb)

                            # Stage 3 Conditional Multi-HMR: Only render 3D mesh if candidate is flagged in REVIEW or ALERT
                            is_candidate_flagged = (arb.state in [FinalReviewState.REVIEW, FinalReviewState.HIGH_PRIORITY_REVIEW])

                            if is_candidate_flagged and humans_3d is not None:
                                # Find matching 3D human by spatial proximity to candidate bbox
                                cx_cand = (bbox[0] + bbox[2]) / 2.0
                                cy_cand = (bbox[1] + bbox[3]) / 2.0
                                best_h = None
                                best_dist = float('inf')
                                for h in humans_3d:
                                    if 'v3d' in h:
                                        v3d = h['v3d'].unsqueeze(0)
                                        p2d_padded = perspective_projection(v3d, K).squeeze(0).cpu().numpy()
                                        px_mean = ((p2d_padded[:, 0].mean() - pad_x) / scale)
                                        py_mean = ((p2d_padded[:, 1].mean() - pad_y) / scale)
                                        d = np.hypot(px_mean - cx_cand, py_mean - cy_cand)
                                        if d < best_dist:
                                            best_dist = d
                                            best_h = h

                                if best_h is not None and best_dist < 120:
                                    v3d = best_h['v3d'].unsqueeze(0)
                                    p2d_padded = perspective_projection(v3d, K).squeeze(0).cpu().numpy()
                                    p2d_orig = np.empty_like(p2d_padded)
                                    p2d_orig[:, 0] = (p2d_padded[:, 0] - pad_x) / scale
                                    p2d_orig[:, 1] = (p2d_padded[:, 1] - pad_y) / scale
                                    pts_triangles = p2d_orig[faces_sampled].astype(np.int32)
                                    cv2.polylines(overlay_mesh, pts_triangles, isClosed=True, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

                            # Draw ID & Status Tag
                            lx, ly = int(bbox[0]), int(bbox[1])
                            if is_inv:
                                tag = "INVIGILATOR (ROLE LOCKED)"
                                box_col = (180, 180, 180)
                            elif arb.state == FinalReviewState.HIGH_PRIORITY_REVIEW:
                                tag = f"ALERT: {cand_id} -> {desk_id} | {arb.event_type}"
                                box_col = (0, 0, 255)
                                if preprocessing_pipeline.config.save_flagged_frames:
                                    preprocessing_pipeline.save_flagged_event_frame(
                                        orig_bgr=frame,
                                        enh_bgr=enhanced_frame,
                                        frame_idx=frame_idx,
                                        reason=f"ALERT_{cand_id}_{arb.event_type}",
                                        metadata={
                                            "candidate_id": cand_id,
                                            "desk_id": desk_id,
                                            "state": arb.state.value,
                                            "event_type": arb.event_type,
                                            "confidence": round(arb.confidence, 3),
                                            "quality_score": quality_report.overall_score,
                                        }
                                    )
                            elif arb.state == FinalReviewState.REVIEW:
                                tag = f"REVIEW: {cand_id} -> {desk_id}"
                                box_col = (0, 165, 255)
                                if preprocessing_pipeline.config.save_flagged_frames:
                                    preprocessing_pipeline.save_flagged_event_frame(
                                        orig_bgr=frame,
                                        enh_bgr=enhanced_frame,
                                        frame_idx=frame_idx,
                                        reason=f"REVIEW_{cand_id}_{arb.event_type}",
                                        metadata={
                                            "candidate_id": cand_id,
                                            "desk_id": desk_id,
                                            "state": arb.state.value,
                                            "event_type": arb.event_type,
                                            "confidence": round(arb.confidence, 3),
                                            "quality_score": quality_report.overall_score,
                                        }
                                    )
                            else:
                                tag = f"{cand_id} -> {desk_id}"
                                box_col = color

                            cv2.rectangle(rendered_frame, (lx, max(0, ly - 22)), (lx + len(tag)*8 + 12, max(20, ly)), (15, 15, 15), -1)
                            cv2.putText(rendered_frame, tag, (lx + 4, max(14, ly - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 2, cv2.LINE_AA)

                        rendered_frame = cv2.addWeighted(overlay_mesh, 0.50, rendered_frame, 0.50, 0)

                        # 8. Draw Global HUD
                        env_type = env_profiler.environment_type.value if env_profiler.finalized else "PROFILING..."
                        vram_now = torch.cuda.memory_allocated() / (1024**3) if device.type == 'cuda' else 0
                        hud = f"NETRA-4D SITUATIONAL | Env: {env_type} | Frame {frame_idx:04d}/{total_frames} | GPU VRAM: {vram_now:.2f} GB"
                        cv2.rectangle(rendered_frame, (0, 0), (width, 34), (10, 10, 10), -1)
                        cv2.putText(rendered_frame, hud, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 200), 2, cv2.LINE_AA)

                        video_writer.write(rendered_frame)

                        frame_record = {
                            "frame_idx": frame_idx,
                            "timestamp_sec": round(frame_idx / fps, 3),
                            "environment": env_profiler.environment_type.value,
                            "quality_report": quality_report.to_dict() if hasattr(quality_report, "to_dict") else None,
                            "arbitrated_events": [
                                {
                                    "candidate_id": a.candidate_id,
                                    "desk_id": a.desk_id,
                                    "state": a.state.value,
                                    "event_type": a.event_type,
                                    "confidence": round(a.confidence, 3),
                                    "supporting_evidence": a.supporting_evidence
                                }
                                for a in frame_arbitrations
                            ]
                        }
                        results.append(frame_record)
                        sampled_count += 1
                        if sampled_count % 15 == 0 or sampled_count <= 3:
                            print(f"  [Situational Runner] Frame {frame_idx:05d} ({sampled_count}) | Env: {env_profiler.environment_type.value} | Candidates locked: {len(candidate_registry.candidate_locks)}")

                    frame_idx += 1
    finally:
        cap.release()
        video_writer.release()
        del active_scratch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n[DONE] Successfully processed {sampled_count} frames into situational video in {elapsed:.1f}s!")
    print(f"  -> Video: {out_video_path}")

    out_file = os.path.join(output_dir, "situational_pipeline_detections.json")
    with open(out_file, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "environment_profile": env_profiler.get_summary(),
            "total_candidates_locked": len(candidate_registry.candidate_locks),
            "total_sampled_frames": len(results),
            "preprocessing_stats": preprocessing_pipeline.get_stats_summary(),
            "video_path": str(out_video_path),
            "frames": results
        }, f, indent=2)

    print(f"  -> Preprocessing Stats: {preprocessing_pipeline.get_stats_summary()}")

    print(f"  -> JSON:  {out_file}")
    return out_video_path, out_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="VIDEO_01")
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means entire video")
    parser.add_argument("--prime-vram-gb", type=float, default=2.2)
    parser.add_argument("--preprocessing-config", default=None, help="Path to preprocessing YAML config")
    args = parser.parse_args()

    manifest = load_drive_manifest(ROOT / "drishti4d" / "configs" / "datasets.json")
    asset = next((a for a in manifest if a.asset_id == args.asset), None)
    if not asset:
        print(f"Asset {args.asset} not found in manifest!")
        sys.exit(1)

    out_dir = ROOT / "outputs" / "situational_pipeline" / args.asset
    print(f"Streaming {asset.asset_id} ({asset.filename}) into Situational Pipeline...")
    with materialize_drive_asset(asset, temp_root=out_dir / ".staging") as (video_path, prov):
        run_situational_pipeline(
            video_source=video_path,
            output_dir=out_dir,
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            prime_vram_gb=args.prime_vram_gb,
            preprocessing_config=args.preprocessing_config
        )
