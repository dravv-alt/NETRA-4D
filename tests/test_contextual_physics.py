from types import SimpleNamespace

from drishti.engine.contextual_physics import ContextualPhysicsEngine, SeatPhysicsCalibration


def _state(candidate_id, center, *, invigilator=False):
    identity = SimpleNamespace(
        candidate_id=candidate_id,
        anchor_center_norm=center,
        anchor_box_size_norm=(0.10, 0.20),
        is_invigilator=invigilator,
    )
    return SimpleNamespace(identity=identity)


def test_uncalibrated_graph_keeps_physical_direction_unknown():
    engine = ContextualPhysicsEngine()
    engine.update({
        "A": _state("A", (0.30, 0.50)),
        "B": _state("B", (0.60, 0.50)),
        "INVIGILATOR": _state("INVIGILATOR", (0.50, 0.20), invigilator=True),
    }, freeze=True)
    relation = engine.relations_for("A")[0]
    assert relation.image_plane_relation == "IMAGE_RIGHT"
    assert relation.physical_relation == "UNKNOWN_UNCALIBRATED"
    assert engine.context_for("A")["calibration_status"] == "IMAGE_PLANE_ONLY"


def test_calibrated_graph_emits_student_relative_left_right_and_depth():
    engine = ContextualPhysicsEngine(SeatPhysicsCalibration(
        calibration_id="TEST_CAMERA", student_left_is_image_left=True,
        student_forward_is_image_up=True,
    ))
    engine.update({
        "A": _state("A", (0.50, 0.50)),
        "LEFT": _state("LEFT", (0.30, 0.50)),
        "BEHIND": _state("BEHIND", (0.50, 0.75)),
    }, freeze=True)
    relations = {row.target_candidate_id: row.physical_relation for row in engine.relations_for("A")}
    assert relations["LEFT"] == "STUDENT_LEFT"
    assert relations["BEHIND"] == "BEHIND_STUDENT"


def test_graph_snapshot_is_emitted_once_after_freeze():
    engine = ContextualPhysicsEngine()
    engine.update({"A": _state("A", (0.30, 0.50)), "B": _state("B", (0.60, 0.50))}, freeze=True)
    assert len(engine.drain_graph_snapshot()) == 2
    assert engine.drain_graph_snapshot() == []
