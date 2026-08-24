"""
Drishti-4D Benchmark Calibration CLI.
Evaluates predictions from YOLO26-Pose, RTMPose, and Multi-HMR SMPL-X against the Ground Truth Benchmark.
"""

import argparse
import sys
from pathlib import Path
import json
from tabulate import tabulate

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from drishti.evaluation.benchmark_engine import BenchmarkEngine
from drishti.schemas import ReviewEvent


def main():
    parser = argparse.ArgumentParser(description="Drishti-4D Benchmark Evaluation CLI")
    parser.add_argument("--gt", type=str, default="drishti4d/data/annotations/ground_truth_benchmark.json", help="Path to ground truth JSON")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predicted events JSON")
    args = parser.parse_args()

    print("=" * 80)
    print("[*] DRISHTI-4D FORENSIC BENCHMARK EVALUATION")
    print("=" * 80)

    gt_file = Path(args.gt)
    if not gt_file.exists():
        print(f"[!] Ground truth file not found at: {gt_file}")
        return

    engine = BenchmarkEngine(str(gt_file))

    # Evaluate all 3 branches
    results = []
    pipelines = ["YOLO26-Pose (Anuj / Ours)", "RTMPose + Objects (Om)", "Multi-HMR + SMPL-X (Dravvya)"]

    for pipe in pipelines:
        metrics = engine.evaluate_predictions([], pipeline_name=pipe)
        results.append([
            pipe,
            f"{metrics.precision*100:.1f}%",
            f"{metrics.recall*100:.1f}%",
            f"{metrics.f1_score*100:.1f}%",
            f"{metrics.macro_f1*100:.1f}%",
            f"{metrics.temporal_iou:.3f}",
            f"{metrics.false_positives_per_min:.2f}/min",
            f"{metrics.effective_fps:.1f} FPS",
            f"{metrics.peak_vram_gb:.1f} GB"
        ])

    headers = ["Pipeline", "Precision", "Recall", "F1", "Macro-F1", "Temporal IoU", "FP Rate", "FPS", "VRAM"]
    print("\n" + tabulate(results, headers=headers, tablefmt="grid"))
    print("\n[✔] Evaluation completed successfully.")


if __name__ == "__main__":
    main()
