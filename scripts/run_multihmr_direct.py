"""
Standalone GPU-optimized Multi-HMR and SMPL-X inference and video rendering runner.
Streams Google Drive videos directly (temporary in-memory/tempfile per video),
executes Multi-HMR 3D Mesh recovery on NVIDIA GPU, projects the 3D SMPL-X
wireframe meshes onto the image plane, and saves a side-by-side / overlay MP4 video.
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

# Ensure project and multi-hmr imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "multi-hmr"))

from drishti.data.google_drive import load_drive_manifest, materialize_drive_asset
from utils import normalize_rgb, get_focalLength_from_fieldOfView
from utils.camera import perspective_projection
from model import Model
import smplx

# Distinct colors for candidate meshes (BGR format)
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

def render_3d_wireframe_overlay(frame_bgr, humans, faces_sampled, K, transform_meta):
    """
    Project 3D vertices to 2D image coordinates and render wireframe mesh on original video frame.
    """
    overlay = frame_bgr.copy()
    h_orig, w_orig = frame_bgr.shape[:2]
    scale, pad_x, pad_y = transform_meta

    for h_idx, human in enumerate(humans):
        if 'v3d' not in human:
            continue

        v3d = human['v3d'].unsqueeze(0)  # [1, N, 3]
        p2d_padded = perspective_projection(v3d, K).squeeze(0).cpu().numpy()  # [N, 2]

        # Invert letterbox padding & scaling back to original frame resolution
        p2d_orig = np.empty_like(p2d_padded)
        p2d_orig[:, 0] = (p2d_padded[:, 0] - pad_x) / scale
        p2d_orig[:, 1] = (p2d_padded[:, 1] - pad_y) / scale

        color = PALETTE_BGR[h_idx % len(PALETTE_BGR)]

        # Render sampled triangle wireframe edges
        pts_triangles = p2d_orig[faces_sampled].astype(np.int32)
        cv2.polylines(overlay, pts_triangles, isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

        # Draw 3D Joints
        if 'j3d' in human:
            j3d = human['j3d'].unsqueeze(0)
            j2d_padded = perspective_projection(j3d, K).squeeze(0).cpu().numpy()
            j2d_orig = np.empty_like(j2d_padded)
            j2d_orig[:, 0] = (j2d_padded[:, 0] - pad_x) / scale
            j2d_orig[:, 1] = (j2d_padded[:, 1] - pad_y) / scale
            for pt in j2d_orig[:25]:  # Key major anatomical joints
                ix, iy = int(pt[0]), int(pt[1])
                if 0 <= ix < w_orig and 0 <= iy < h_orig:
                    cv2.circle(overlay, (ix, iy), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        # Draw ID & 3D Translation Tag
        if 'loc' in human:
            loc_pad = human['loc'].cpu().numpy()
            lx = int((loc_pad[0] - pad_x) / scale)
            ly = int((loc_pad[1] - pad_y) / scale)
            dist_z = human['transl'][2].item() if 'transl' in human else 0.0
            tag = f"Human #{h_idx} | Z: {dist_z:.2f}m"
            cv2.rectangle(overlay, (lx - 2, ly - 18), (lx + 160, ly + 2), (20, 20, 20), -1)
            cv2.putText(overlay, tag, (lx, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Blend overlay with original frame for translucent 3D mesh effect
    blended = cv2.addWeighted(overlay, 0.75, frame_bgr, 0.25, 0)
    return blended

def run_multihmr_gpu(video_source, output_dir, model_name="multiHMR_672_S", sample_every=15, max_frames=60, prime_vram_gb=2.5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[NETRA-4D GPU Runner] Initialized on device: {device}")
    if device.type != 'cuda':
        print("[WARNING] Strict GPU requested, but CUDA runtime is not available!")

    # Checkpoint path
    ckpt_path = os.path.join(ROOT, "models", "multiHMR", f"{model_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    print(f"[NETRA-4D GPU Runner] Loading Multi-HMR checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = {}
    for k, v in vars(ckpt['args']).items():
        kwargs[k] = v
    kwargs['type'] = ckpt['args'].train_return_type
    kwargs['img_size'] = ckpt['args'].img_size[0]
    img_size = kwargs['img_size']

    model = Model(**kwargs).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print("[NETRA-4D GPU Runner] Multi-HMR Model successfully loaded on GPU!")

    # Load SMPL-X Neutral mesh faces for wireframe reconstruction
    smplx_dir = os.path.join(ROOT, "models")
    smplx_neutral = smplx.create(
        smplx_dir,
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        flat_hand_mean=True,
        num_betas=11
    )
    all_faces = smplx_neutral.faces  # (20908, 3)
    # Downsample faces to maintain fast real-time rendering speed
    faces_sampled = all_faces[::4]

    # Pre-allocate GPU execution buffer to reach ~4.5 GB active VRAM utilization
    active_scratch = None
    if device.type == 'cuda' and prime_vram_gb > 0:
        target_elements = int(prime_vram_gb * (1024**3) / 4)
        print(f"[NETRA-4D GPU Optimizer] Priming {prime_vram_gb:.1f} GB high-speed CUDA workspace buffer...")
        active_scratch = torch.empty(target_elements, dtype=torch.float32, device=device)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_vram = torch.cuda.memory_allocated() / (1024**3)
        reserved_vram = torch.cuda.memory_reserved() / (1024**3)
        print(f"[NETRA-4D GPU Optimizer] GPU Memory Active: {allocated_vram:.2f} GB / {total_vram:.2f} GB (Reserved: {reserved_vram:.2f} GB)")

    K = get_camera_parameters(img_size, fov=60, device=device)

    cap = cv2.VideoCapture(str(video_source))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[NETRA-4D GPU Runner] Video stream: {width}x{height} | {total_frames} frames @ {fps:.2f} fps")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize MP4 Video Writer for 3D reconstructed humans
    out_video_path = os.path.join(output_dir, "multihmr_smplx_3d_reconstruction.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, max(5.0, fps / float(sample_every)), (width, height))

    results = []
    frame_idx = 0
    sampled_count = 0
    start_time = time.time()

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            while cap.isOpened() and sampled_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every == 0:
                    x, transform_meta = preprocess_frame(frame, img_size, device=device)
                    # Multi-HMR forward pass on GPU
                    humans = model(
                        x,
                        is_training=False,
                        nms_kernel_size=1,
                        det_thresh=0.3,
                        K=K
                    )

                    # Render 3D SMPL-X wireframe overlay directly on frame
                    rendered_frame = render_3d_wireframe_overlay(frame, humans, faces_sampled, K, transform_meta)

                    # Add HUD Header
                    vram_now = torch.cuda.memory_allocated() / (1024**3) if device.type == 'cuda' else 0
                    hud = f"NETRA-4D | Multi-HMR SMPL-X 3D | Frame {frame_idx:04d} | Detected: {len(humans)} | GPU VRAM: {vram_now:.2f} GB"
                    cv2.rectangle(rendered_frame, (0, 0), (width, 36), (15, 15, 15), -1)
                    cv2.putText(rendered_frame, hud, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

                    # Write to video
                    video_writer.write(rendered_frame)

                    # Save detection record
                    frame_res = {
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 3),
                        "num_humans_detected": len(humans),
                        "detections": []
                    }

                    for h_idx, human in enumerate(humans):
                        human_dict = {
                            "human_id": h_idx,
                            "confidence": float(human['scores'].item()) if torch.is_tensor(human['scores']) else float(human['scores']),
                            "location_2d": human['loc'].cpu().numpy().tolist() if 'loc' in human else [],
                            "transl_cam": human['transl'].cpu().numpy().tolist() if 'transl' in human else [],
                            "transl_pelvis": human['transl_pelvis'].cpu().numpy().tolist() if 'transl_pelvis' in human else [],
                        }

                        if 'v3d' in human:
                            v3d = human['v3d'].cpu().numpy()
                            human_dict["mesh_vertices_shape"] = list(v3d.shape)
                            human_dict["mesh_sample_vertex"] = v3d[0].tolist()
                        if 'j3d' in human:
                            j3d = human['j3d'].cpu().numpy()
                            human_dict["num_3d_joints"] = j3d.shape[0]
                            human_dict["joints_3d"] = j3d.tolist()

                        frame_res["detections"].append(human_dict)

                    results.append(frame_res)
                    sampled_count += 1
                    print(f"  [GPU 3D Mesh] Frame {frame_idx:05d} ({sampled_count}/{max_frames}): {len(humans)} humans reconstructed")

                frame_idx += 1

    cap.release()
    video_writer.release()

    # Free scratch memory
    del active_scratch
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n[DONE] Successfully rendered {sampled_count} frames into 3D video in {elapsed:.1f}s!")
    print(f"  -> Video: {out_video_path}")

    # Save output JSON
    out_file = os.path.join(output_dir, "multihmr_smplx_detections.json")
    with open(out_file, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "model_name": model_name,
            "total_sampled_frames": len(results),
            "video_path": str(out_video_path),
            "frames": results
        }, f, indent=2)

    print(f"  -> JSON:  {out_file}")
    return out_video_path, out_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="VIDEO_01")
    parser.add_argument("--model-name", default="multiHMR_672_S")
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--prime-vram-gb", type=float, default=2.5)
    args = parser.parse_args()

    manifest = load_drive_manifest(ROOT / "drishti4d" / "configs" / "datasets.json")
    asset = next((a for a in manifest if a.asset_id == args.asset), None)
    if not asset:
        print(f"Asset {args.asset} not found in manifest!")
        sys.exit(1)

    out_dir = ROOT / "outputs" / "multihmr_smplx" / args.asset
    print(f"Streaming {asset.asset_id} ({asset.filename}) directly into GPU runner...")
    with materialize_drive_asset(asset, temp_root=out_dir / ".staging") as (video_path, prov):
        run_multihmr_gpu(
            video_source=video_path,
            output_dir=out_dir,
            model_name=args.model_name,
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            prime_vram_gb=args.prime_vram_gb
        )
