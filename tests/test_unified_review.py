from types import SimpleNamespace

from drishti.engine.hypotheses import CompetingHypothesisEngine
from drishti.pipeline.environment import EnvironmentProfiler, EnvironmentType
from drishti.release.unified_review import UnifiedReviewEngine
import numpy as np
import json
import pytest


def test_no_pattern_has_no_fabricated_movement_evidence():
    evaluation = CompetingHypothesisEngine().evaluate("candidate", None)
    assert evaluation.supporting_evidence == []
    assert evaluation.counter_evidence == []


def test_environment_warmup_finalizes_once():
    profiler = EnvironmentProfiler(warmup_frames=3)
    for _ in range(5):
        profiler.observe([], [])
    assert profiler.frame_count == 3
    assert profiler.finalized
    assert profiler.environment_type == EnvironmentType.LARGE_CLASSROOM


def test_phone_only_associated_with_its_detector_person():
    from enum import Enum
    class ObjectType(str, Enum):
        PHONE = "MOBILE_PHONE"
    people = [SimpleNamespace(entity_local_id=tid, bbox_xyxy_px=box,
        detection_confidence=.9, keypoints_2d={}) for tid, box in
        [(1,(10,10,80,180)), (2,(200,10,270,180))]]
    phone = SimpleNamespace(object_class=ObjectType.PHONE,confidence=.8,
        bbox_xyxy_norm=(.1,.3,.15,.4),associated_entity_id=1,metadata={})
    engine = UnifiedReviewEngine()
    frame = np.zeros((200,300,3),dtype=np.uint8)
    rows = engine.process(0,0.,frame,SimpleNamespace(people=people,objects=[phone]))
    assert len(rows) == 2
    by_track = {r['tracker_id']:r for r in rows}
    assert by_track[1]['phone_candidate_nearby']
    assert not by_track[2]['phone_candidate_nearby']
    assert all(r['pattern_types']==[] for r in rows)


def test_unassociated_phone_cannot_flag_every_candidate():
    from enum import Enum
    class ObjectType(str,Enum):
        PHONE = "MOBILE_PHONE"
    person = SimpleNamespace(entity_local_id=1,bbox_xyxy_px=(10,10,80,180),
        detection_confidence=.9,keypoints_2d={})
    phone = SimpleNamespace(object_class=ObjectType.PHONE,confidence=.8,
        bbox_xyxy_norm=(.1,.3,.15,.4),associated_entity_id=None,metadata={})
    rows = UnifiedReviewEngine().process(0,0.,np.zeros((200,300,3),dtype=np.uint8),
        SimpleNamespace(people=[person],objects=[phone]))
    assert not rows[0]['phone_candidate_nearby']


def test_capture_failure_writes_failed_manifest_and_closes_resources(tmp_path, monkeypatch):
    import drishti.release.unified_review as module
    closed = []
    monkeypatch.setattr(module, 'bounded_video_source', lambda *args: ('missing.mp4', 2200))
    monkeypatch.setattr(module, 'file_sha256', lambda path: 'source-hash')
    monkeypatch.setattr(module, 'ModelRegistry', lambda *args: SimpleNamespace(get=lambda *a, **k: None))
    monkeypatch.setattr(module, 'CanonicalInferenceService', lambda *a, **k:
        SimpleNamespace(close=lambda: closed.append('service')))
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda *a:
        SimpleNamespace(isOpened=lambda: False, release=lambda: closed.append('capture')))
    run = tmp_path/'failed_run'
    with pytest.raises(RuntimeError,match='Cannot open video'):
        module.run_video03_review('missing.mp4',run,allow_experimental_models=True)
    manifest = json.loads((run/'INCOMPLETE.json').read_text())
    assert manifest['status'] == 'FAILED'
    assert manifest['frames'] == 0
    assert not (run/'COMPLETE.json').exists()
    assert closed == ['capture','service']


def test_existing_run_is_never_overwritten(tmp_path, monkeypatch):
    import drishti.release.unified_review as module
    monkeypatch.setattr(module,'bounded_video_source',lambda *a:('unused.mp4',2200))
    monkeypatch.setattr(module,'ModelRegistry',lambda *a:SimpleNamespace(get=lambda *a,**k:None))
    marker = tmp_path/'keep.txt'
    marker.write_text('existing user output')
    with pytest.raises(FileExistsError):
        module.run_video03_review('unused.mp4',tmp_path,allow_experimental_models=True)
    assert marker.read_text() == 'existing user output'
