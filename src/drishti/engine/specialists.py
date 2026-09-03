"""Real Neural Specialists for Stage 2 Precision Escalation.

Conforms to docs/ARCHITECTURE.md Layer 3:
1. RTMPoseONNXRunner: Lightweight ONNX Runtime inference using SimCC coordinate classification
   (17 keypoints x 384/512 sub-pixel bins) on high-resolution candidate crops.
2. MediaPipeHandRunner: Official MediaPipe HandLandmarker using hand_landmarker.task
   extracting full 21 3D landmarks for pinch ratio and finger curl measurement.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Safeguard: MediaPipe internal drawing_utils imports matplotlib.pyplot.
# If matplotlib is missing or corrupted on the host, mock pyplot in memory to allow
# pure neural inference without plotting dependencies.
if "matplotlib.pyplot" not in sys.modules:
    sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")
if "matplotlib" not in sys.modules:
    sys.modules["matplotlib"] = types.ModuleType("matplotlib")

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None
    mp_python = None
    mp_vision = None

from .observation import HandLandmarksObservation, NormalizedKeypoint, RTMPoseAdapter


class RTMPoseONNXRunner:
    """Stage 2 Specialist: Executes RTM-Pose ONNX with SimCC coordinate classification."""

    def __init__(self, onnx_path: str = "models/weights/rtmpose_m/rtmpose_m.onnx"):
        self.onnx_path = onnx_path
        self.session = None
        self.input_name = None
        self.input_shape = (256, 192)  # height, width
        self.simcc_split_ratio = 2.0

        if ort is not None and os.path.exists(onnx_path):
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name

    @property
    def is_available(self) -> bool:
        return self.session is not None

    def infer_crop(self, crop_bgr: np.ndarray) -> Optional[List[NormalizedKeypoint]]:
        """Runs RTM-Pose on a candidate BGR crop and returns 17 normalized keypoints."""
        if not self.is_available or crop_bgr is None or crop_bgr.size == 0:
            return None

        ch, cw = crop_bgr.shape[:2]
        th, tw = self.input_shape

        # Resize with aspect ratio preserved (letterbox)
        scale = min(tw / cw, th / ch)
        nw, nh = int(cw * scale), int(ch * scale)
        resized = cv2.resize(crop_bgr, (nw, nh))

        padded = np.zeros((th, tw, 3), dtype=np.uint8)
        pad_x = (tw - nw) // 2
        pad_y = (th - nh) // 2
        padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        # BGR -> RGB and normalize mean/std
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        norm_img = (rgb - mean) / std
        input_tensor = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # Run ONNX inference: outputs are (1, 17, W_simcc) and (1, 17, H_simcc)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        simcc_x, simcc_y = outputs[0], outputs[1]

        keypoints = RTMPoseAdapter.decode_simcc(
            simcc_x=simcc_x,
            simcc_y=simcc_y,
            crop_shape=(ch, cw),
            simcc_split_ratio=self.simcc_split_ratio
        )
        return keypoints


class MediaPipeHandRunner:
    """Stage 2 Specialist: Executes MediaPipe HandLandmarker for 21 3D hand joints."""

    def __init__(self, model_asset_path: str = "models/weights/hand_landmarker.task"):
        self.model_path = model_asset_path
        self.landmarker = None

        if mp_vision is not None and os.path.exists(model_asset_path):
            base_opts = mp_python.BaseOptions(model_asset_path=model_asset_path)
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    @property
    def is_available(self) -> bool:
        return self.landmarker is not None

    def infer_hand(self, hand_crop_bgr: np.ndarray, handedness: str = "right") -> Optional[HandLandmarksObservation]:
        """Runs 3D hand landmark estimation on a wrist/hand ROI crop."""
        if not self.is_available or hand_crop_bgr is None or hand_crop_bgr.size == 0:
            return None

        rgb = cv2.cvtColor(hand_crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        raw_landmarks = result.hand_landmarks[0]
        score = 0.85
        if result.handedness and result.handedness[0]:
            score = float(result.handedness[0][0].score)

        from .observation import HAND_LANDMARK_NAMES
        lms_dict = {}
        for i, lm in enumerate(raw_landmarks):
            kname = HAND_LANDMARK_NAMES[i] if i < len(HAND_LANDMARK_NAMES) else f"joint_{i}"
            lms_dict[kname] = NormalizedKeypoint(
                name=kname,
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=1.0,
                confidence=score
            )

        wrist_kp = lms_dict.get("wrist")
        mid_mcp = lms_dict.get("middle_finger_mcp")
        palm_center = ((wrist_kp.x + mid_mcp.x) / 2.0, (wrist_kp.y + mid_mcp.y) / 2.0) if wrist_kp and mid_mcp else None

        return HandLandmarksObservation(
            side=handedness.upper(),
            landmarks=lms_dict,
            palm_center_norm=palm_center,
            confidence=score,
            is_detected=True
        )
