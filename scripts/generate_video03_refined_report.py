"""Write a caveated refined-annotation report; never changes models or labels."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'outputs/video03_refined_single_phone_audit_20261003'; out.mkdir(exist_ok=True)
 review=json.loads((ROOT/'datasets/video03_refined_single_phone_20261003/REFINEMENT_MANIFEST.json').read_text())
 metric=json.loads((out/'partial_metrics.json').read_text())
 experimental=json.loads((ROOT/'outputs/experimental/video03_refined_full_evidence_20261003/experiment_manifest.json').read_text())
 phone=metric['per_class']['Phone']
 report={'status':'REFINED_EVALUATION_CANDIDATE_FROZEN','dataset':{'video':'VIDEO_03','frames':2200,'fps':25,'duration_sec':88,'one_physical_phone':True},'human_review':{'confirmed_phone_proposals':review['human_confirmed_proposals'],'not_phone':review['not_phone_excluded'],'ambiguous_excluded':review['ambiguous_excluded'],'remaining_unknown':0},'refined_box_diagnostic':{'iou_threshold':.5,'labelled_phone_boxes':phone['labelled_boxes'],'matched_boxes':phone['matched_boxes'],'recall':phone['recall_at_iou_0_5'],'mean_matched_iou':phone['mean_matched_iou']},'full_evidence_run':{'path':'outputs/experimental/video03_refined_full_evidence_20261003','frames':experimental['processed_frames'],'runtime_sec':experimental['runtime_sec'],'enabled_components':experimental['configuration']['enabled_components'],'device_note':'perception was from complete CUDA:0 baseline; deterministic evidence replay is CPU bookkeeping'},'metrics':{'precision':'NOT_AVAILABLE','f1':'NOT_AVAILABLE','mAP':'NOT_AVAILABLE','fp_per_hour':'NOT_AVAILABLE'},'limitations':['Confirmed phone geometry originated from reviewed detector proposals; it is not independent held-out object geometry.','AMBIGUOUS proposals are correctly excluded, not negative labels.','NOT_PHONE proposal decisions alone do not establish exhaustive negative coverage over all possible phone-like regions.','Same-video development/evaluation only; no generalization claim.','Head direction is a separate low-performing proxy and is not used as gaze evidence.']}
 (out/'phone_evaluation_report.json').write_text(json.dumps(report,indent=2))
 lines=['# Video 3 refined phone evaluation report','',f"- Reviewed proposals: {review['human_confirmed_proposals']} confirmed, {review['not_phone_excluded']} not-phone, {review['ambiguous_excluded']} ambiguous, 0 unknown.",f"- Refined box recall at IoU >= 0.5: **{phone['recall_at_iou_0_5']:.2%}** ({phone['matched_boxes']}/{phone['labelled_boxes']}).",f"- Mean matched IoU: **{phone['mean_matched_iou']:.4f}**.",'','Precision, F1, mAP and FP/hour: **NOT AVAILABLE**. The reviewed refinement is valuable ground-truth correction, but its box geometry came from model proposals and the review did not establish exhaustive negative coverage.','', 'The full 2,200-frame evidence run is recorded separately and contains no cheating verdict.']
 (out/'phone_evaluation_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
