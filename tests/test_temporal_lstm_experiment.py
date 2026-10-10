import numpy as np

from drishti.experimental.temporal_lstm import (
    FEATURE_NAMES,
    TemporalEvidenceLSTM,
    build_next_observation_windows,
    evidence_sample_from_record,
    smoke_evidence_data_flow,
)


def _record(frame):
    return {
        "frame_index": frame, "timestamp_sec": frame / 25, "track_id": 1, "candidate_id": "Candidate_01",
        "evidence": {"pose_evidence": {"head_direction_proxy": "LEFT"},
                     "hand_evidence": {"observed": False}, "phone_evidence": {"associated_phone": frame % 2 == 0},
                     "neighbour_evidence": {"near_neighbour": False}, "object_evidence": {"relations": 1},
                     "occlusion": {"status": "UNKNOWN"}},
    }


def test_evidence_schema_builds_contiguous_observation_windows():
    rows = [evidence_sample_from_record(_record(frame)) for frame in range(35)]
    windows = build_next_observation_windows(rows, window_size=32)
    assert len(windows) == 3
    assert windows[0][1].shape == (32, len(FEATURE_NAMES))
    assert windows[0][3] == 32


def test_lstm_shape_is_observable_feature_vector_only():
    model = TemporalEvidenceLSTM()
    output = model(__import__("torch").zeros((2, 32, len(FEATURE_NAMES))))
    assert tuple(output.shape) == (2, len(FEATURE_NAMES))
    assert np.all(output.detach().numpy() >= 0) and np.all(output.detach().numpy() <= 1)
