"""Create an evidence-first NETRA-4D dashboard report; no research-paper content."""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRELOADED = ROOT / "PRELOADED_IMG_VID_DEMO"


def browser() -> str:
    for value in (shutil.which("msedge"), shutil.which("chrome"), r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if value and Path(value).is_file():
            return value
    raise RuntimeError("Chromium/Edge is required to render the operational PDF")


def uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def bar_chart(title: str, labels: list[str], values: list[int], color: str) -> str:
    maximum = max(values) or 1
    rows = "".join(f'<div class="barrow"><span>{escape(label)}</span><div class="rail"><i style="width:{value / maximum * 100:.1f}%;background:{color}"></i></div><b>{value:,}</b></div>' for label, value in zip(labels, values))
    return f'<section class="panel"><h3>{escape(title)}</h3>{rows}</section>'


def scan_feature_log(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    for item in data.get("features", []):
        head, hands, torso = item.get("head", {}), item.get("hands", {}), item.get("torso", {})
        if head.get("direction") not in {None, "FRONT"}: counts["off_axis_head"] += 1
        if hands.get("left_state") not in {None, "NORMAL"} or hands.get("right_state") not in {None, "NORMAL"}: counts["hand_state_change"] += 1
        if hands.get("left_below_desk") or hands.get("right_below_desk"): counts["below_desk"] += 1
        if torso.get("angle_toward_neighbor") not in {None, 0}: counts["toward_neighbor"] += 1
    return {"name": path.name, "frames": data.get("total_frames", 0), "duration": data.get("duration_sec", 0), "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    run_dir, assets, summary = args.run.parent, run.get("assets", []), run.get("summary", {})
    event_counts: Counter[str] = Counter()
    total_evidence = 0
    evidence_paths: list[Path] = []
    rows, frame_values, event_values, evidence_values = [], [], [], []
    for asset in assets:
        counts = asset.get("counts", {})
        frame_values.append(int(counts.get("frames", 0)))
        event_values.append(int(counts.get("review_events", 0)))
        evidence_values.append(int(counts.get("high_confidence_event_frames", 0)))
        total_evidence += evidence_values[-1]
        rows.append(f'<tr><td><b>{escape(asset.get("asset_id", ""))}</b></td><td>{escape(asset.get("source", {}).get("filename", ""))}</td><td>{frame_values[-1]:,}</td><td>{counts.get("people", 0):,}</td><td>{counts.get("objects", 0):,}</td><td>{event_values[-1]}</td><td>{evidence_values[-1]}</td><td>{float(asset.get("elapsed_sec", 0)):.1f}s</td></tr>')
        event_file = run_dir / asset["asset_id"] / asset.get("outputs", {}).get("review_events", "review_events.jsonl")
        if event_file.is_file():
            for line in event_file.read_text(encoding="utf-8").splitlines():
                if line.strip(): event_counts[json.loads(line).get("event_type", "UNKNOWN")] += 1
        evidence_dir = run_dir / asset["asset_id"] / (asset.get("outputs", {}).get("review_event_frame_directory") or "review_event_frames")
        if evidence_dir.is_dir(): evidence_paths.extend(sorted(evidence_dir.glob("*.jpg")))

    feature_logs = [scan_feature_log(path) for path in sorted(PRELOADED.glob("*_features.json"))]
    preloaded_images = sorted(PRELOADED.glob("*.jpg"))
    gallery = evidence_paths + preloaded_images
    gallery_html = "".join(f'<figure><img src="{uri(path)}"><figcaption>{escape(path.name)}</figcaption></figure>' for path in gallery)
    feature_rows = "".join(f'<tr><td><b>{escape(item["name"])}</b></td><td>{item["frames"]:,}</td><td>{float(item["duration"]):.2f}s</td><td>{item["counts"]["off_axis_head"]:,}</td><td>{item["counts"]["hand_state_change"]:,}</td><td>{item["counts"]["below_desk"]:,}</td><td>{item["counts"]["toward_neighbor"]:,}</td></tr>' for item in feature_logs)
    labels = [asset.get("asset_id", "") for asset in assets]
    event_labels, event_values_by_type = zip(*event_counts.most_common()) if event_counts else ([], [])
    total_events = sum(event_values)

    html = f'''<!doctype html><html><meta charset="utf-8"><style>
@page{{size:A4;margin:15mm 14mm}}*{{box-sizing:border-box}}body{{font:11px "Segoe UI",Arial,sans-serif;color:#1d2939;margin:0;background:#fff;line-height:1.4}}h1{{font:700 27px Georgia,"Times New Roman",serif;letter-spacing:-.35px;margin:7px 0 4px;color:#101828}}h2{{font:700 15px Georgia,"Times New Roman",serif;letter-spacing:.1px;margin:27px 0 10px;padding-bottom:7px;border-bottom:1px solid #cfd8dc;color:#101828}}h3{{font-size:10px;letter-spacing:.2px;margin:0 0 11px;text-transform:uppercase;color:#475467}}.eyebrow{{color:#067647;font-weight:700;letter-spacing:1.45px;font-size:9px}}.muted{{color:#667085;font-size:9px}}.hero{{padding:20px 22px 18px;border-top:4px solid #0b6b4a;border-bottom:1px solid #cfd8dc;background:#f8faf9}}.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:0;margin-top:16px;border:1px solid #d8e0dd;background:#fff}}.kpi{{padding:11px 12px;border-right:1px solid #d8e0dd;background:#fff}}.kpi:last-child{{border-right:0}}.kpi b{{font:700 21px Georgia,"Times New Roman",serif;display:block;margin-top:3px;color:#0d3b2a}}.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.panel{{border:1px solid #d9e1e6;border-radius:0;padding:13px 14px;background:#fff;break-inside:avoid}}.barrow{{display:grid;grid-template-columns:72px 1fr 45px;gap:8px;align-items:center;margin:8px 0;font-size:9px}}.rail{{height:7px;border-radius:0;background:#edf1f2;overflow:hidden}}.rail i{{display:block;height:100%}}table{{border-collapse:collapse;width:100%;font-size:8.5px}}th,td{{padding:7px 5px;border-bottom:1px solid #e7ecef;text-align:left;vertical-align:top}}th{{background:#f3f6f5;color:#475467;font-size:7.5px;text-transform:uppercase;letter-spacing:.75px}}tr:nth-child(even) td{{background:#fbfcfc}}.gallery{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}}figure{{margin:0;border:1px solid #d9e1e6;border-radius:0;overflow:hidden;break-inside:avoid;background:#fff}}figure img{{width:100%;height:132px;object-fit:cover;display:block;filter:saturate(.92)}}figcaption{{padding:6px 7px;font:7.5px monospace;color:#667085;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.note{{padding:11px 13px;border-left:3px solid #0b6b4a;background:#f3faf6;color:#294b3d;font-size:9.5px;line-height:1.5}}.pagebreak{{break-before:page}}
</style><body>
<div class="hero"><div class="eyebrow">NETRA-4D / EVIDENCE-LED OPERATIONAL RECORD</div><h1>Monitoring & Evidence Dashboard</h1><div class="muted">Verified run: <b>{escape(str(run.get("run_id", "")))}</b> · Completed {escape(str(run.get("completed_at_utc", "")))}</div><div class="kpis"><div class="kpi"><span class="muted">Videos processed</span><b>{len(assets)}</b></div><div class="kpi"><span class="muted">Frames rendered</span><b>{int(summary.get("frames", 0)):,}</b></div><div class="kpi"><span class="muted">Review events</span><b>{total_events}</b></div><div class="kpi"><span class="muted">Evidence captures</span><b>{total_evidence}</b></div><div class="kpi"><span class="muted">Telemetry logs</span><b>{len(feature_logs)}</b></div></div></div>
<h2>Run telemetry</h2><div class="twocol">{bar_chart("Rendered frames by source", labels, frame_values, "#1473e6")}{bar_chart("Temporal review events by source", labels, event_values, "#d92d20")}</div><div class="twocol" style="margin-top:13px">{bar_chart("Captured evidence frames by source", labels, evidence_values, "#079455")}{bar_chart("Event types from review logs", list(event_labels), list(event_values_by_type), "#f79009")}</div>
<h2>Per-video processing record</h2><table><thead><tr><th>Video</th><th>Source file</th><th>Frames</th><th>People</th><th>Objects</th><th>Events</th><th>Evidence</th><th>Runtime</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="pagebreak"></div><h2>Preloaded telemetry-log evidence</h2><div class="note">These counters are derived from the supplied feature-log exports. They describe model observations such as off-axis head direction, non-normal hand states, under-desk hand flags, and torso orientation toward a neighbour. They are evidence for invigilator review, not automated misconduct findings.</div><table style="margin-top:12px"><thead><tr><th>Log</th><th>Frames</th><th>Duration</th><th>Off-axis head</th><th>Hand-state changes</th><th>Below desk</th><th>Toward neighbour</th></tr></thead><tbody>{feature_rows}</tbody></table>
<h2>Captured evidence gallery</h2><div class="gallery">{gallery_html}</div>
<h2>Decision boundary</h2><div class="note">This dashboard consolidates measured pipeline outputs, source timestamps, evidence stills and telemetry logs. Confidence is used to prioritise review. A human invigilator remains responsible for the final decision.</div>
</body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = args.output.with_suffix(".html")
    source.write_text(html, encoding="utf-8")
    profile = (args.output.parent / "operational-report-browser-profile").resolve()
    subprocess.run([browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}", f"--print-to-pdf={args.output.resolve()}", source.resolve().as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
