from drishti.providers.phone_crop import person_context_crops
from drishti.providers.phone_crop import detect_phone_candidates
from drishti.providers.phone_crop import PersonCropMemory
import numpy as np
from types import SimpleNamespace


def test_context_includes_hand_region_beyond_the_person_box_without_frame_overflow():
    region = person_context_crops([[870,469,1067,715]],1280,720)[0]
    x,y,x2,y2 = region
    assert x <= 1044 and y <= 549 and x2 >= 1091 and y2 >= 594
    assert 0 <= x < x2 <= 1280 and 0 <= y < y2 <= 720


def test_invalid_boxes_and_duplicate_regions_do_not_create_bad_crops():
    boxes = [[10,10,50,80],[10,10,50,80],[30,30,20,20],[float('nan'),0,10,10]]
    assert person_context_crops(boxes,100,90) == [(0,0,90,90)]


def test_crop_predictions_keep_confidence_and_are_remapped_to_frame_coordinates():
    box = SimpleNamespace(conf=np.float32(.7),xyxy=np.array([[10,20,30,40]],dtype=float))
    class Model:
        def predict(self, source, **kwargs):
            if isinstance(source,list):
                return [SimpleNamespace(boxes=[box]) for _ in source]
            return [SimpleNamespace(boxes=[])]
    image = np.zeros((720,1280,3),dtype=np.uint8)
    raw = detect_phone_candidates(Model(),image,[[870,469,1067,715]],device='cpu')
    region = person_context_crops([[870,469,1067,715]],1280,720)[0]
    assert raw[0]['bbox_xyxy_px'] == [10+region[0],20+region[1],30+region[0],40+region[1]]
    assert abs(raw[0]['confidence']-.7) < 1e-6
    assert raw[0]['inference_view'] == 'person_context_crop'


def test_overlapping_views_produce_one_raw_detection_without_score_boosting():
    box = SimpleNamespace(conf=np.float32(.7),xyxy=np.array([[10,20,30,40]],dtype=float))
    class Model:
        def predict(self, source, **kwargs):
            return [SimpleNamespace(boxes=[box]) for _ in (source if isinstance(source,list) else [source])]
    raw = detect_phone_candidates(Model(),np.zeros((90,100,3),dtype=np.uint8),[[10,10,50,80]],device='cpu')
    assert len(raw) == 1
    assert abs(raw[0]['confidence']-.7) < 1e-6


def test_held_roi_expires_and_never_fabricates_a_phone():
    state = PersonCropMemory(1.)
    people = [{'entity_local_id':7,'bbox_xyxy_px':[870,469,1067,715]}]
    assert state.regions(people,1280,720,0)[0]['proposal_age_sec'] == 0
    held = state.regions([],1280,720,.5)
    assert held[0]['proposal_age_sec'] == .5
    class EmptyModel:
        def predict(self,source,**kwargs):
            return [SimpleNamespace(boxes=[]) for _ in (source if isinstance(source,list) else [source])]
    assert detect_phone_candidates(EmptyModel(),np.zeros((720,1280,3),dtype=np.uint8),[],regions_override=held) == []
    assert state.regions([],1280,720,1.01) == []


def test_roi_memory_clears_on_timestamp_reset_and_frame_geometry_change():
    state = PersonCropMemory(1.)
    people = [{'entity_local_id':7,'bbox_xyxy_px':[10,10,50,80]}]
    state.regions(people,100,90,10)
    assert state.regions([],100,90,0) == []
    state.regions(people,100,90,10)
    assert state.regions([],200,180,10.1) == []
