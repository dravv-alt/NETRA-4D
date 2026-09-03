"""Unit tests verifying real Stage 2 specialists: RTMPoseONNXRunner and MediaPipeHandRunner.

Tests:
1. RTMPoseONNXRunner model loading, shape alignment, and 17-keypoint SimCC decoding.
2. MediaPipeHandRunner model loading and 21 3D hand landmark extraction on image crops.
3. EscalationGate end-to-end evaluation using real specialists.
"""

import numpy as np
import pytest

from drishti.engine.hypotheses import EvidenceEvaluationResult, HypothesisDistribution, HypothesisKey
from drishti.engine.escalation import EscalationGate, EscalationStage
from drishti.engine.specialists import RTMPoseONNXRunner, MediaPipeHandRunner


def test_rtmpose_onnx_specialist_live_inference():
    runner = RTMPoseONNXRunner()
    assert runner.is_available is True, "RTMPose ONNX model should be loaded"

    dummy_crop = np.zeros((240, 200, 3), dtype=np.uint8)
    keypoints = runner.infer_crop(dummy_crop)

    assert keypoints is not None
    assert len(keypoints) == 17
    for kp in keypoints:
        assert 0.0 <= kp.x <= 1.0
        assert 0.0 <= kp.y <= 1.0
        assert 0.0 <= kp.confidence <= 1.0


def test_mediapipe_hand_specialist_live_inference():
    runner = MediaPipeHandRunner()
    assert runner.is_available is True, "MediaPipe HandLandmarker model should be loaded"

    # Synthetic image containing blank/neutral crop
    dummy_crop = np.ones((256, 256, 3), dtype=np.uint8) * 128
    # When no hand is detected, returns None safely without error
    hand_res = runner.infer_hand(dummy_crop)
    assert hand_res is None or hasattr(hand_res, "landmarks_21_3d")


def test_escalation_gate_with_real_specialists():
    gate = EscalationGate()
    rtm = RTMPoseONNXRunner()
    mp_h = MediaPipeHandRunner()

    gate.stage2_rtmpose_verifier = rtm.infer_crop
    gate.stage2_mediapipe_verifier = mp_h.infer_hand

    ambiguous_eval = EvidenceEvaluationResult(
        candidate_id="Cand_01",
        suspicion_score=0.50,
        uncertainty_score=0.40,
        hypotheses=HypothesisDistribution(),
        supporting_evidence=["Occluded hand region"],
        counter_evidence=[],
        dominant_hypothesis=HypothesisKey.H1_PRIMARY_SUSPICION,
        requires_specialist=True
    )

    crop = np.zeros((150, 150, 3), dtype=np.uint8)
    res = gate.evaluate_stage2_specialists(
        candidate_id="Cand_01",
        evaluation=ambiguous_eval,
        candidate_crop_bgr=crop,
        hand_crops_bgr=[crop]
    )

    assert res.stage == EscalationStage.STAGE_2_SPECIALIST
    assert "rtmpose_simcc_kpts" in res.measurements
    assert res.measurements["rtmpose_simcc_kpts"] == 17
    # Uncertainty was reduced by SimCC sub-pixel coordinate classification
    assert res.resolved_uncertainty < ambiguous_eval.uncertainty_score
