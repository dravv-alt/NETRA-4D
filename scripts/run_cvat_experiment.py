"""One reproducible command for immutable CVAT audit, bounded inference and evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from drishti.data.video_policy import bounded_video_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--max-frames', type=int, default=2200, help='Exclusive count: 2200 means frame indices 0 through 2199')
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('cvat_%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    args.video, args.max_frames = bounded_video_source(args.video, args.max_frames)
    if args.max_frames <= 0 or Path(args.run_id).name != args.run_id:
        parser.error('Use a positive frame count and a plain run ID')
    output = ROOT/'outputs'/'cvat_experiments'/args.run_id
    product = ROOT/'outputs'/'product_runs'/args.run_id
    if product.exists():
        raise FileExistsError(product)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status':'RUNNING','max_frames':args.max_frames,'processed_frame_range':[0,args.max_frames-1],'video':str(args.video),'annotations':str(args.annotations),'completed_steps':[]}
    def save():
        (output/'experiment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def run(name, script, *options):
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,options)],cwd=ROOT,check=True)
        manifest['completed_steps'].append(name)
        save()
    save()
    try:
        run('audit_and_export','audit_cvat_video.py','--video',args.video,'--annotations',args.annotations,'--output',output/'dataset','--max-frames',args.max_frames)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(args.video),'-frames:v',str(args.max_frames),'-an','-c:v','libx264','-preset','fast','-crf','18',str(output/'input_bounded.mp4')],cwd=ROOT,check=True)
        manifest['completed_steps'].append('bounded_source_video'); save()
        run('canonical_inference','run_product_pipeline.py','--source',args.video,'--max-frames',args.max_frames,'--run-id',args.run_id,'--camera-id','CVAT_CAM','--device','cuda:0','--require-cuda','--database',output/'events.sqlite')
        run('box_evaluation','evaluate_cvat_boxes.py','--labels',output/'dataset'/'trusted_boxes.jsonl','--predictions',product/'detections.jsonl','--max-frames',args.max_frames,'--output',output/'metrics.json')
        run('phone_comparison','benchmark_cvat_phone.py','--dataset',output/'dataset','--output',output/'phone_comparison')
        manifest['status'] = 'COMPLETE'
    except Exception as error:
        manifest['status'] = 'FAILED'
        manifest['error'] = str(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
