"""Partial-label box recall. Unmatched predictions are UNKNOWN, never false positives."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from audit_cvat_video import digest, iou


def evaluate(truth_rows, predictions, max_frames=2200):
    truth = defaultdict(list)
    for row in truth_rows:
        if 0 <= row['frame_index'] < max_frames and not row['outside'] and row['geometry_valid']:
            truth[row['frame_index']].append(row)
    frames = set()
    counts = {name: {'labelled_boxes': 0, 'matched_boxes': 0, 'unmatched_predictions_unknown': 0, 'per_track': {}, 'matched_ious': []} for name in ('Person', 'Phone')}
    for objects in truth.values():
        for row in objects:
            name = 'Person' if row['class'] in ('Student', 'Invigilator') else 'Phone' if row['class'] == 'Phone' else None
            if name:
                metric = counts[name]
                metric['labelled_boxes'] += 1
                track = metric['per_track'].setdefault(str(row['track_id']), {'class': row['class'], 'labelled_boxes': 0, 'matched_boxes': 0})
                track['labelled_boxes'] += 1
    for prediction in predictions:
        frame = prediction['frame_index']
        if frame < 0 or frame >= max_frames or frame in frames:
            raise ValueError(f'Out-of-bound or duplicate prediction frame {frame}')
        frames.add(frame)
        for name, metric in counts.items():
            labelled = [r for r in truth.get(frame, []) if (r['class'] in ('Student', 'Invigilator') if name == 'Person' else r['class'] == 'Phone')]
            if not labelled:
                continue
            candidates = prediction['people'] if name == 'Person' else [p for p in prediction['objects'] if p['object_class'] == 'MOBILE_PHONE']
            used = set()
            for candidate in sorted(candidates, key=lambda p: p.get('detection_confidence', p.get('confidence', 0)), reverse=True):
                best = max(((iou(candidate['bbox_xyxy_px'], r['bbox_xyxy_px']), i) for i, r in enumerate(labelled) if i not in used), default=(0, -1))
                if best[0] >= .5:
                    used.add(best[1])
                    metric['matched_boxes'] += 1
                    metric['matched_ious'].append(best[0])
                    metric['per_track'][str(labelled[best[1]]['track_id'])]['matched_boxes'] += 1
                else:
                    metric['unmatched_predictions_unknown'] += 1
    for metric in counts.values():
        metric['recall_at_iou_0_5'] = metric['matched_boxes'] / metric['labelled_boxes'] if metric['labelled_boxes'] else None
        overlaps = metric.pop('matched_ious')
        metric['mean_matched_iou'] = sum(overlaps) / len(overlaps) if overlaps else None
        for track in metric['per_track'].values():
            track['recall_at_iou_0_5'] = track['matched_boxes'] / track['labelled_boxes']
    return {'status': 'COMPLETE' if frames == set(range(max_frames)) else 'PARTIAL_PREDICTIONS', 'prediction_frames': len(frames), 'missing_prediction_frames': sorted(set(range(max_frames)) - frames), 'processed_range_inclusive': [0, max_frames-1], 'iou_threshold': .5, 'per_class': counts, 'precision': None, 'f1': None, 'mAP': None, 'false_positives_per_hour': None, 'unavailable_reason': 'Exhaustive annotation coverage is not confirmed; no annotations means UNKNOWN. Semantic attributes and event metrics require compatible predictions and event windows.', 'validation_scope': 'Single video diagnostic, not an independent camera/session test'}


def load_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=2200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = evaluate(load_rows(args.labels), load_rows(args.predictions), args.max_frames)
    report['labels_sha256'] = digest(args.labels)
    report['predictions_sha256'] = digest(args.predictions)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'missing_prediction_frames'}, indent=2))


if __name__ == '__main__':
    main()
