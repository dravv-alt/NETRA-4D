"""CLI for the bounded Video-03 evidence augmentation experiment."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from drishti.experimental.video03_full import VIDEO03, DEFAULT_BASELINE, run_video03_full_experimental
def main():
 p=argparse.ArgumentParser(); p.add_argument("--output-dir",required=True); p.add_argument("--baseline-run",default=str(DEFAULT_BASELINE)); p.add_argument("--max-frames",type=int,default=2200); p.add_argument("--specialist-stride",type=int,default=1); p.add_argument("--specialist-policy",choices=("baseline_only","specialists_every_frame","conditional_escalation"),default="conditional_escalation"); p.add_argument("--specialist-policy-config",default="configs/specialist_escalation.yaml"); p.add_argument("--enable-rtmpose",action="store_true"); p.add_argument("--enable-hands-experimental",action="store_true"); p.add_argument("--enable-interaction-graph",action="store_true"); p.add_argument("--enable-evidence-fusion",action="store_true"); p.add_argument("--enable-3d-experimental",action="store_true"); p.add_argument("--enable-all-experimental",action="store_true")
 a=p.parse_args(); all_=a.enable_all_experimental
 print(run_video03_full_experimental(str(VIDEO03),a.output_dir,baseline_run=a.baseline_run,max_frames=a.max_frames,specialist_stride=a.specialist_stride,specialist_policy=a.specialist_policy,specialist_policy_config=a.specialist_policy_config,enable_rtmpose=a.enable_rtmpose or all_,enable_hands=a.enable_hands_experimental or all_,enable_interaction=a.enable_interaction_graph or all_,enable_fusion=a.enable_evidence_fusion or all_,enable_3d=a.enable_3d_experimental or all_))
if __name__=="__main__": main()
