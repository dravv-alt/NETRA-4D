"""
Standalone GPU-optimized Unified Perception Runner:
Fuses Multi-HMR (SMPL-X 3D Meshes) with YOLO26-Pose (ByteTrack Persistent ID Tracking)
and YOLO26-Objects (Mobile Phone & Suspicious Objects) into a single, high-fidelity
full-length annotated video and comprehensive JSON output.
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
from drishti.preprocessing import PreprocessingPipeline
from drishti.data.video_policy import bounded_video_source
from utils import normalize_rgb, get_focalLength_from_fieldOfView
from utils.camera import perspective_projection
from model import Model
import smplx

# Distinct colors for tracked candidate meshes (BGR format)
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
    (128, 255, 255),  # Soft yellow
    (255, 215, 0),    # Gold
]

# 2D Pose Skeleton Connections (COCO 17 Keypoints)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head / Facial Keypoints
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms & Wrists
    (5, 11), (6, 12), (11, 12),              # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)   # Legs
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

def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def associate_humans_with_tracks(humans_3d, tracked_people, K, transform_meta, w_orig, h_orig):
    """
    Associate Multi-HMR 3D human predictions with YOLO ByteTrack persistent person IDs
    using 2D projected bounding box overlap (IoU) and proximity.
    """
    scale, pad_x, pad_y = transform_meta
    associated = []
    
    # Pre-extract projected 2D boxes for all 3D humans
    boxes_3d = []
    for h in humans_3d:
        if 'v3d' in h:
            v3d = h['v3d'].unsqueeze(0)
            p2d_padded = perspective_projection(v3d, K).squeeze(0).cpu().numpy()
            p2d_orig = np.empty_like(p2d_padded)
            p2d_orig[:, 0] = np.clip((p2d_padded[:, 0] - pad_x) / scale, 0, w_orig)
            p2d_orig[:, 1] = np.clip((p2d_padded[:, 1] - pad_y) / scale, 0, h_orig)
            x1, y1 = float(p2d_orig[:, 0].min()), float(p2d_orig[:, 1].min())
            x2, y2 = float(p2d_orig[:, 0].max()), float(p2d_orig[:, 1].max())
            boxes_3d.append([x1, y1, x2, y2, p2d_orig])
        else:
            boxes_3d.append([0, 0, 0, 0, None])

    used_track_ids = set()
    for h_idx, (h_3d, b3d_info) in enumerate(zip(humans_3d, boxes_3d)):
        box3d = b3d_info[:4]
        p2d_orig = b3d_info[4]
        
        best_track_id = None
        best_iou = 0.15 # threshold
        best_dist = float('inf')
        best_person = None

        center_3d = ((box3d[0] + box3d[2]) / 2.0, (box3d[1] + box3d[3]) / 2.0)

        for person in tracked_people:
            t_id = person['track_id']
            t_box = person['bbox']
            iou = compute_iou(box3d, t_box)
            if iou > best_iou:
                best_iou = iou
                best_track_id = t_id
                best_person = person
            elif best_track_id is None:
                # Proximity fallback if center falls inside tracked bbox
                t_center = ((t_box[0] + t_box[2]) / 2.0, (t_box[1] + t_box[3]) / 2.0)
                dist = np.hypot(center_3d[0] - t_center[0], center_3d[1] - t_center[1])
                if dist < 120.0 and dist < best_dist and t_id not in used_track_ids:
                    best_dist = dist
                    best_track_id = t_id
                    best_person = person

        if best_track_id is not None:
            used_track_ids.add(best_track_id)
            track_id = best_track_id
        else:
            # Consistent pseudo ID if no match
            track_id = 100 + h_idx

        associated.append({
            "human_3d": h_3d,
            "track_id": track_id,
            "box_2d": box3d,
            "p2d_orig": p2d_orig,
            "yolo_person": best_person
        })

    return associated

def run_fused_pipeline(
    video_source,
    output_dir,
    model_name="multiHMR_672_S",
    sample_every=4, # Run at 6.25 fps for smooth motion
    max_frames=300, # Process up to 300 frames (~12-15 seconds of dense continuous video)
    prime_vram_gb=2.5,
    preprocessing_config=None
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[NETRA-4D Fused Engine] Initializing on dedicated GPU: {device}")
    if device.type != 'cuda':
        print("[WARNING] Strict GPU requested, but CUDA runtime is not available!")

    # 1. Load Multi-HMR Model
    ckpt_path = os.path.join(ROOT, "models", "multiHMR", f"{model_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    print(f"[NETRA-4D Fused Engine] Loading Multi-HMR checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = {}
    for k, v in vars(ckpt['args']).items():
        kwargs[k] = v
    kwargs['type'] = ckpt['args'].train_return_type
    kwargs['img_size'] = ckpt['args'].img_size[0]
    img_size = kwargs['img_size']

    hmr_model = Model(**kwargs).to(device)
    hmr_model.load_state_dict(ckpt['model_state_dict'], strict=False)
    hmr_model.eval()
    print("[NETRA-4D Fused Engine] Multi-HMR Model loaded on GPU!")

    # 2. Load YOLO26-Pose with ByteTrack
    pose_model_path = os.path.join(ROOT, "models", "weights", "yolo26n-pose.pt")
    print(f"[NETRA-4D Fused Engine] Loading YOLO26 Pose Tracker: {pose_model_path}")
    yolo_pose = YOLO(pose_model_path)
    yolo_pose.to(device)

    # 3. Load YOLO26 Object Detection (Phone / Suspicious Items)
    obj_model_path = os.path.join(ROOT, "models", "weights", "yolo26n.pt")
    print(f"[NETRA-4D Fused Engine] Loading YOLO26 Object Detector: {obj_model_path}")
    yolo_obj = YOLO(obj_model_path)
    yolo_obj.to(device)

    # 4. Load SMPL-X Neutral mesh faces
    smplx_dir = os.path.join(ROOT, "models")
    smplx_neutral = smplx.create(
        smplx_dir,
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        flat_hand_mean=True,
        num_betas=11
    )
    all_faces = smplx_neutral.faces
    faces_sampled = all_faces[::4] # Sample for real-time overlay speed

    # 5. Prime GPU Memory to maximize performance (~4.0 - 4.5 GB target)
    active_scratch = None
    if device.type == 'cuda' and prime_vram_gb > 0:
        target_elements = int(prime_vram_gb * (1024**3) / 4)
        print(f"[NETRA-4D Fused Engine] Priming {prime_vram_gb:.1f} GB CUDA execution workspace...")
        active_scratch = torch.empty(target_elements, dtype=torch.float32, device=device)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_vram = torch.cuda.memory_allocated() / (1024**3)
        reserved_vram = torch.cuda.memory_reserved() / (1024**3)
        print(f"[NETRA-4D Fused Engine] GPU Memory Active: {allocated_vram:.2f} GB / {total_vram:.2f} GB (Reserved: {reserved_vram:.2f} GB)")

    K = get_camera_parameters(img_size, fov=60, device=device)

    video_source, _ = bounded_video_source(video_source)
    cap = cv2.VideoCapture(str(video_source))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_fps = max(10.0, fps / float(sample_every))
    print(f"[NETRA-4D Fused Engine] Ingesting Video: {width}x{height} | {total_frames} frames @ {fps:.2f} fps -> Output Video: {output_fps:.2f} fps")
    os.makedirs(output_dir, exist_ok=True)

    out_video_path = os.path.join(output_dir, "multihmr_yolo26_fused_3d.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, output_fps, (width, height))

    results = []
    frame_idx = 0
    sampled_count = 0
    start_time = time.time()
    preprocessing_pipeline = PreprocessingPipeline(config=preprocessing_config)

    max_allowed = max_frames if (max_frames is not None and max_frames > 0) else float('inf')

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            while cap.isOpened() and sampled_count < max_allowed:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every == 0:
                    enhanced_frame, q_report = preprocessing_pipeline.process(frame, frame_idx=frame_idx)

                    # -------------------------------------------------------------
                    # A. YOLO26 Pose Inference + ByteTrack Tracking
                    # -------------------------------------------------------------
                    pose_results = yolo_pose.track(
                        enhanced_frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=0.25,
                        device=device,
                        verbose=False
                    )[0]

                    tracked_people = []
                    if pose_results.boxes is not None and len(pose_results.boxes) > 0:
                        boxes_xyxy = pose_results.boxes.xyxy.cpu().numpy()
                        track_ids = (
                            pose_results.boxes.id.int().cpu().numpy()
                            if pose_results.boxes.id is not None
                            else np.arange(1, len(boxes_xyxy) + 1)
                        )
                        kpts_data = (
                            pose_results.keypoints.data.cpu().numpy()
                            if pose_results.keypoints is not None
                            else None
                        )

                        for idx_p, (b, t_id) in enumerate(zip(boxes_xyxy, track_ids)):
                            tracked_people.append({
                                "track_id": int(t_id),
                                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                "keypoints": kpts_data[idx_p] if kpts_data is not None else None
                            })

                    # -------------------------------------------------------------
                    # B. YOLO26 Object Detection (Phones, etc.)
                    # -------------------------------------------------------------
                    obj_results = yolo_obj.predict(
                        enhanced_frame,
                        conf=0.20,
                        device=device,
                        verbose=False
                    )[0]

                    detected_objects = []
                    if obj_results.boxes is not None and len(obj_results.boxes) > 0:
                        o_boxes = obj_results.boxes.xyxy.cpu().numpy()
                        o_confs = obj_results.boxes.conf.cpu().numpy()
                        o_classes = obj_results.boxes.cls.int().cpu().numpy()
                        for ob, oconf, ocls in zip(o_boxes, o_confs, o_classes):
                            c_name = obj_results.names.get(int(ocls), f"cls_{ocls}")
                            # Track relevant objects: cell phone, laptop, book, etc.
                            if c_name in ["cell phone", "laptop", "book", "bottle"]:
                                detected_objects.append({
                                    "class_name": c_name,
                                    "confidence": float(oconf),
                                    "bbox": [float(ob[0]), float(ob[1]), float(ob[2]), float(ob[3])]
                                })

                    # -------------------------------------------------------------
                    # C. Multi-HMR 3D SMPL-X Forward Pass
                    # -------------------------------------------------------------
                    x, transform_meta = preprocess_frame(frame, img_size, device=device)
                    humans_3d = hmr_model(
                        x,
                        is_training=False,
                        nms_kernel_size=1,
                        det_thresh=0.3,
                        K=K
                    )

                    # -------------------------------------------------------------
                    # D. Unified Association (Multi-HMR 3D Mesh <-> YOLO Track ID)
                    # -------------------------------------------------------------
                    associated_humans = associate_humans_with_tracks(
                        humans_3d, tracked_people, K, transform_meta, width, height
                    )

                    # -------------------------------------------------------------
                    # E. Visual Rendering: 3D SMPL-X + 2D Pose Skeleton + Objects
                    # -------------------------------------------------------------
                    rendered_frame = frame.copy()
                    overlay = frame.copy()
                    scale, pad_x, pad_y = transform_meta

                    # 1. Render 3D SMPL-X Meshes with Persistent Track Colors
                    for item in associated_humans:
                        t_id = item['track_id']
                        h_3d = item['human_3d']
                        p2d_orig = item['p2d_orig']
                        color = PALETTE_BGR[t_id % len(PALETTE_BGR)]

                        if p2d_orig is not None:
                            # Render 3D Wireframe triangles
                            pts_triangles = p2d_orig[faces_sampled].astype(np.int32)
                            cv2.polylines(overlay, pts_triangles, isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

                            # Major 3D joint dots
                            if 'j3d' in h_3d:
                                j3d = h_3d['j3d'].unsqueeze(0)
                                j2d_padded = perspective_projection(j3d, K).squeeze(0).cpu().numpy()
                                j2d_orig = np.empty_like(j2d_padded)
                                j2d_orig[:, 0] = (j2d_padded[:, 0] - pad_x) / scale
                                j2d_orig[:, 1] = (j2d_padded[:, 1] - pad_y) / scale
                                for pt in j2d_orig[:25]:
                                    ix, iy = int(pt[0]), int(pt[1])
                                    if 0 <= ix < width and 0 <= iy < height:
                                        cv2.circle(overlay, (ix, iy), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

                        # Draw Persistent Track ID & 3D Depth Header
                        box = item['box_2d']
                        lx, ly = int(box[0]), int(box[1])
                        dist_z = h_3d['transl'][2].item() if 'transl' in h_3d else 0.0
                        label = f"ID #{t_id} | Z: {dist_z:.2f}m"
                        cv2.rectangle(rendered_frame, (lx, max(0, ly - 22)), (lx + 150, max(20, ly)), (20, 20, 20), -1)
                        cv2.putText(rendered_frame, label, (lx + 4, max(14, ly - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

                    # Blend 3D mesh overlay translucently
                    rendered_frame = cv2.addWeighted(overlay, 0.70, rendered_frame, 0.30, 0)

                    # 2. Render YOLO26 Keypoints & Pose Skeletons
                    for p in tracked_people:
                        kpts = p['keypoints']
                        t_id = p['track_id']
                        color = PALETTE_BGR[t_id % len(PALETTE_BGR)]
                        if kpts is not None:
                            # Draw skeleton lines
                            for p1_idx, p2_idx in SKELETON_PAIRS:
                                if p1_idx < len(kpts) and p2_idx < len(kpts):
                                    pt1 = kpts[p1_idx]
                                    pt2 = kpts[p2_idx]
                                    if pt1[2] > 0.35 and pt2[2] > 0.35:
                                        cv2.line(rendered_frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2, cv2.LINE_AA)
                            # Draw keypoint dots
                            for pt in kpts:
                                if pt[2] > 0.35:
                                    cv2.circle(rendered_frame, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1, cv2.LINE_AA)

                    # 3. Render YOLO26 Detected Objects (Phone Alerts)
                    for obj in detected_objects:
                        ob = obj['bbox']
                        x1, y1, x2, y2 = int(ob[0]), int(ob[1]), int(ob[2]), int(ob[3])
                        obj_cls = obj['class_name']
                        obj_conf = obj['confidence']
                        box_col = (0, 0, 255) if obj_cls == "cell phone" else (255, 165, 0)
                        cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), box_col, 2, cv2.LINE_AA)
                        tag = f"ALERT: {obj_cls.upper()} ({obj_conf:.2f})"
                        cv2.rectangle(rendered_frame, (x1, max(0, y1 - 20)), (x1 + 180, max(20, y1)), (0, 0, 0), -1)
                        cv2.putText(rendered_frame, tag, (x1 + 4, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1, cv2.LINE_AA)

                    # 4. Render Global HUD Header
                    vram_now = torch.cuda.memory_allocated() / (1024**3) if device.type == 'cuda' else 0
                    hud = f"NETRA-4D UNIFIED | Frame {frame_idx:04d} | Multi-HMR 3D SMPL-X + YOLO26 Tracked | GPU VRAM: {vram_now:.2f} GB"
                    cv2.rectangle(rendered_frame, (0, 0), (width, 36), (15, 15, 15), -1)
                    cv2.putText(rendered_frame, hud, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 200), 2, cv2.LINE_AA)

                    # Write frame to video
                    video_writer.write(rendered_frame)

                    # Record frame metadata
                    frame_record = {
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 3),
                        "num_humans_detected": len(associated_humans),
                        "tracked_candidates": [
                            {
                                "persistent_track_id": item['track_id'],
                                "bbox_2d": item['box_2d'],
                                "transl_cam_3d": item['human_3d']['transl'].cpu().numpy().tolist() if 'transl' in item['human_3d'] else [],
                                "mesh_vertices_shape": list(item['human_3d']['v3d'].shape) if 'v3d' in item['human_3d'] else [],
                                "joints_3d": item['human_3d']['j3d'].cpu().numpy().tolist() if 'j3d' in item['human_3d'] else []
                            }
                            for item in associated_humans
                        ],
                        "detected_objects": detected_objects,
                        "quality_report": q_report.to_dict() if hasattr(q_report, "to_dict") else None,
                    }
                    results.append(frame_record)
                    sampled_count += 1
                    total_desc = f"/{total_frames}" if max_allowed == float('inf') else f"/{int(max_allowed)}"
                    if sampled_count % 10 == 0 or sampled_count <= 5:
                        print(f"  [Unified GPU Runner] Frame {frame_idx:05d} ({sampled_count}{total_desc}) | {len(associated_humans)} 3D humans mapped to persistent IDs")

                frame_idx += 1

    cap.release()
    video_writer.release()

    del active_scratch
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n[DONE] Successfully fused Multi-HMR and YOLO26 across {sampled_count} frames in {elapsed:.1f}s!")
    print(f"  -> Fused Video: {out_video_path}")

    out_file = os.path.join(output_dir, "multihmr_yolo26_fused_detections.json")
    with open(out_file, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "model_multi_hmr": model_name,
            "model_yolo_pose": "yolo26n-pose.pt",
            "model_yolo_object": "yolo26n.pt",
            "tracker": "ByteTrack",
            "total_sampled_frames": len(results),
            "preprocessing_stats": preprocessing_pipeline.get_stats_summary(),
            "video_path": str(out_video_path),
            "frames": results
        }, f, indent=2)

    print(f"  -> Preprocessing Stats: {preprocessing_pipeline.get_stats_summary()}")

    print(f"  -> Unified JSON: {out_file}")
    return out_video_path, out_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="VIDEO_01")
    parser.add_argument("--model-name", default="multiHMR_672_S")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means entire video")
    parser.add_argument("--prime-vram-gb", type=float, default=2.2)
    parser.add_argument("--preprocessing-config", default=None, help="Path to preprocessing YAML config")
    args = parser.parse_args()

    manifest = load_drive_manifest(ROOT / "drishti4d" / "configs" / "datasets.json")
    asset = next((a for a in manifest if a.asset_id == args.asset), None)
    if not asset:
        print(f"Asset {args.asset} not found in manifest!")
        sys.exit(1)

    out_dir = ROOT / "outputs" / "fused_pipeline" / args.asset
    print(f"Streaming {asset.asset_id} ({asset.filename}) directly into Fused GPU runner...")
    with materialize_drive_asset(asset, temp_root=out_dir / ".staging") as (video_path, prov):
        run_fused_pipeline(
            video_source=video_path,
            output_dir=out_dir,
            model_name=args.model_name,
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            prime_vram_gb=args.prime_vram_gb,
            preprocessing_config=args.preprocessing_config
        )
