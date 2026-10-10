"""Regression checks for partial annotations and incomplete prediction streams."""
import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('evaluate_cvat_boxes', SCRIPTS / 'evaluate_cvat_boxes.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_missing_predictions_do_not_disappear_from_recall_denominator():
    labels = [{'frame_index': i, 'track_id': 3, 'class': 'Phone', 'outside': False, 'geometry_valid': True, 'bbox_xyxy_px': [0, 0, 10, 10]} for i in range(2)]
    predictions = [{'frame_index': 0, 'people': [], 'objects': [{'object_class': 'MOBILE_PHONE', 'bbox_xyxy_px': [0, 0, 10, 10], 'confidence': .9}]}]
    report = module.evaluate(labels, predictions, 2)
    assert report['per_class']['Phone']['recall_at_iou_0_5'] == .5
    assert report['status'] == 'PARTIAL_PREDICTIONS'
    assert report['precision'] is None


def test_one_prediction_cannot_match_two_tracks_and_unknown_frames_are_excluded():
    labels = [{'frame_index': 0, 'track_id': i, 'class': 'Phone', 'outside': False, 'geometry_valid': True, 'bbox_xyxy_px': [0, 0, 10, 10]} for i in (3, 4)]
    obj = {'object_class': 'MOBILE_PHONE', 'bbox_xyxy_px': [0, 0, 10, 10], 'confidence': .9}
    predictions = [{'frame_index': i, 'people': [], 'objects': [obj]} for i in range(2)]
    report = module.evaluate(labels, predictions, 2)
    assert report['per_class']['Phone']['matched_boxes'] == 1
    assert report['per_class']['Phone']['unmatched_predictions_unknown'] == 0
    assert report['status'] == 'COMPLETE'
