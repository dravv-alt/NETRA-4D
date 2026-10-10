"""Continue a completed phone bootstrap, preserving prior checkpoints and evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import torch
import ultralytics
from ultralytics import YOLO

from audit_cvat_video import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-run', type=Path, default=Path('outputs/training/video03_phone_bootstrap_20261002_v2'))
    parser.add_argument('--output', type=Path, default=Path('outputs/training/video03_phone_continued_100epochs_20261002'))
    parser.add_argument('--additional-epochs', type=int, default=70)
    args = parser.parse_args()
    if args.additional_epochs <= 0:
        parser.error('additional epochs must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    previous = json.loads((args.previous_run/'training_manifest.json').read_text(encoding='utf-8'))
    if previous['status'] != 'COMPLETE':
        raise ValueError('Previous training must be complete')
    checkpoint = Path(previous['candidate_checkpoint'])
    source_hash = digest(checkpoint)
    if source_hash != previous['candidate_sha256']:
        raise ValueError('Previous candidate hash does not match its manifest')
    data = Path(previous.get('dataset_yaml', args.previous_run/'dataset/data.yaml')).resolve()
    raw = ROOT/'MANUAL_ANNOTATIONS(CVAT)/VIDEO_3_EXAM_HALL_PHONE_USE/annotations.xml'
    if digest(raw) != previous['annotations_sha256']:
        raise ValueError('Raw annotations changed; refusing an untracked continuation')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cumulative = previous.get('cumulative_epochs', previous['epochs_requested']) + args.additional_epochs
    manifest = {
        'status':'TRAINING', 'started_at':datetime.now(timezone.utc).isoformat(),
        'previous_run':str(args.previous_run.resolve()), 'base_model':str(checkpoint),
        'base_model_sha256':source_hash, 'dataset_yaml':str(data),
        'annotations_sha256':previous['annotations_sha256'], 'video':previous['video'],
        'video_sha256':previous['video_sha256'], 'max_frames':2200,
        'training_samples':previous['training_samples'], 'physical_phone_entities':1,
        'classes':['phone'], 'epochs_requested':args.additional_epochs,
        'cumulative_epochs':cumulative, 'resume_mode':'WEIGHTS_ONLY_NEW_OPTIMIZER: completed checkpoint has stripped optimizer state',
        'validation_policy':previous['validation_policy'], 'release_approved':False,
        'environment':{'ultralytics':ultralytics.__version__, 'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0)},
        'training_config':{'imgsz':640,'batch':8,'freeze':10,'optimizer':'AdamW','lr0':.001,'seed':42,'amp':False,'mosaic':0},
    }
    def save():
        (args.output/'training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save()
    try:
        model = YOLO(str(checkpoint))
        if set(model.names.values()) != {'phone'}:
            raise ValueError('Expected a phone-only checkpoint')
        model.train(data=str(data),epochs=args.additional_epochs,imgsz=640,batch=8,device=0,workers=0,project=str(args.output),name='fit',exist_ok=False,seed=42,deterministic=True,amp=False,freeze=10,optimizer='AdamW',lr0=.001,mosaic=0,mixup=0,fliplr=0,flipud=0,scale=.2,translate=.15,cache=False,plots=False,val=False,patience=0,save=True)
        candidate = args.output/'fit/weights/last.pt'
        manifest.update({'status':'TRAINED_EVALUATING','candidate_checkpoint':str(candidate),'candidate_sha256':digest(candidate),'training_fit_metrics':model.metrics.results_dict,'training_fit_metrics_checkpoint':str(args.output/'fit/weights/best.pt'),'training_fit_metrics_warning':'Same training crops; no held-out accuracy claim'})
        save()
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_trained_phone.py'),'--model',str(candidate),'--output',str(args.output/'full_video_evaluation')],cwd=ROOT,check=True)
        current = json.loads((args.output/'full_video_evaluation/metrics.json').read_text(encoding='utf-8'))
        baseline = json.loads((args.previous_run/'full_video_evaluation/metrics.json').read_text(encoding='utf-8'))
        if current['video_sha256'] != baseline['video_sha256'] or current['labels_sha256'] != baseline['labels_sha256'] or current['image_size'] != baseline['image_size'] or current['confidence_threshold'] != baseline['confidence_threshold']:
            raise ValueError('Evaluation protocol changed; comparison invalid')
        before = baseline['per_class']['Phone']['recall_at_iou_0_5']
        after = current['per_class']['Phone']['recall_at_iou_0_5']
        manifest.update({'status':'COMPLETE','previous_full_video_recall':before,'current_full_video_recall':after,'recall_delta_percentage_points':100*(after-before),'evaluation_frames':current['prediction_frames'],'comparison_scope':'Same video used for training; matched evaluation settings, not independent validation'})
    except Exception as error:
        manifest['status'] = 'FAILED'
        manifest['error'] = str(error)
        raise
    finally:
        if digest(checkpoint) != source_hash or digest(raw) != previous['annotations_sha256']:
            manifest['status'] = 'FAILED'
            manifest['error'] = 'Source checkpoint or raw annotation changed'
        save()
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
