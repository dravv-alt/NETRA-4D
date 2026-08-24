"""Generate a research-paper style PDF from a validated NETRA-4D run.

The renderer deliberately uses the locally installed Chromium/Edge print engine
instead of a Python plotting stack. This keeps the report reproducible on the
demo laptop even when optional scientific-plotting packages are unavailable.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import mimetypes
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def load_events(run_dir: Path, assets: list[dict]) -> tuple[Counter, list[dict]]:
    counts: Counter = Counter()
    events: list[dict] = []
    for asset in assets:
        path = run_dir / asset["asset_id"] / asset["outputs"]["review_events"]
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["asset_id"] = asset["asset_id"]
            counts[record.get("event_type", "UNKNOWN")] += 1
            events.append(record)
    return counts, events


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def locate_browser() -> Path:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError("No Chromium-compatible browser was found for PDF rendering")


def page_header(section_name: str, title: str) -> str:
    return f"""
      <header><span>NETRA-4D / TECHNICAL RESEARCH REPORT</span><span>{escape(section_name)}</span></header>
      <h1 class="page-title">{escape(title)}</h1>
      <div class="rule"></div>
    """


def page_footer(number: int) -> str:
    return f"""
      <footer><span>Privacy-preserving review evidence; not an automated misconduct verdict.</span><span>{number}</span></footer>
    """


def section(title: str, body: str) -> str:
    return f'<section class="copy"><h2>{escape(title)}</h2><p>{escape(body)}</p></section>'


def build_html(run_path: Path) -> str:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run_dir = run_path.parent
    assets = run["assets"]
    event_counts, events = load_events(run_dir, assets)
    summary = run["summary"]
    evidence_total = sum(a["counts"].get("high_confidence_event_frames", 0) for a in assets)

    events_by_asset: dict[str, list[dict]] = {
        asset["asset_id"]: [event for event in events if event["asset_id"] == asset["asset_id"]]
        for asset in assets
    }
    event_candidate_tracks = sum(
        len({candidate_id for event in asset_events for candidate_id in event.get("candidate_ids", [])})
        for asset_events in events_by_asset.values()
    )
    mapped_candidate_tracks = sum(
        len({node.get("candidate_id") for node in asset.get("cbt_graph", {}).get("nodes", []) if node.get("device_id")})
        for asset in assets
    )
    total_video_duration = sum(
        asset["video"]["source_frame_count"] / asset["video"]["fps"]
        for asset in assets
        if asset["video"].get("fps")
    )
    durations = [float(event.get("duration_sec", 0.0)) for event in events]
    total_temporal_duration = sum(durations)
    average_temporal_duration = total_temporal_duration / len(durations) if durations else 0.0
    maximum_temporal_duration = max(durations, default=0.0)
    timestamp_values = [float(event.get("timestamp_sec", 0.0)) for event in events]
    timestamp_min = min(timestamp_values, default=0.0)
    timestamp_max = max(timestamp_values, default=0.0)
    total_privacy_masks = sum(asset["counts"].get("privacy_masks", 0) for asset in assets)
    total_gaze_observations = sum(asset["counts"].get("gaze_proxy_known", 0) for asset in assets)

    result_rows = "".join(
        "<tr>"
        f"<td><strong>{escape(a['asset_id'])}</strong></td>"
        f"<td>{a['counts']['frames']:,}</td>"
        f"<td>{a['video']['source_frame_count'] / a['video']['fps']:.1f}</td>"
        f"<td>{len({candidate_id for event in events_by_asset[a['asset_id']] for candidate_id in event.get('candidate_ids', [])})}</td>"
        f"<td>{len({event.get('event_type') for event in events_by_asset[a['asset_id']]})}</td>"
        f"<td>{a['counts']['review_events']}</td>"
        f"<td>{a['counts'].get('high_confidence_event_frames', 0)}</td>"
        "</tr>"
        for a in assets
    )
    largest_event_count = max(event_counts.values(), default=1)
    event_bars = "".join(
        f'<div class="bar-row"><span>{escape(label.replace("_", " "))}</span>'
        f'<div class="bar-track"><i style="width:{max(3, count / largest_event_count * 100):.1f}%"></i></div>'
        f"<strong>{count}</strong></div>"
        for label, count in sorted(event_counts.items(), key=lambda item: item[1], reverse=True)
    )

    gallery: list[tuple[Path, str]] = []
    for asset in assets:
        directory_name = asset["outputs"].get("review_event_frame_directory")
        if directory_name:
            gallery.extend(
                (image, asset["asset_id"])
                for image in sorted((run_dir / asset["asset_id"] / directory_name).glob("*.jpg"))
            )
    gallery_cards = "".join(
        f'<figure><img src="{data_uri(image_path)}" alt="Evidence from {escape(asset_id)}">'
        f"<figcaption>{escape(asset_id)} · {escape(image_path.stem.replace('frame_', 'frame ').replace('_', ' '))}</figcaption></figure>"
        for image_path, asset_id in gallery[:4]
    )

    runtime = run["runtime"]
    generated_at = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    stats = [
        ("VIDEOS", summary["videos"]),
        ("FRAMES", f"{summary['frames']:,}"),
        ("EVENT CANDIDATE TRACKS", event_candidate_tracks),
        ("REVIEW EVENTS", summary["review_events"]),
        ("EVIDENCE STILLS", evidence_total),
        ("SOURCE DURATION", f"{total_video_duration:.1f}s"),
    ]
    stat_cards = "".join(
        f'<div class="stat"><b>{value}</b><span>{label}</span></div>' for label, value in stats
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>NETRA-4D Technical Research Report</title>
<style>
  @page {{ size: A4; margin: 0; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: #d8d8d8; color: #171717; font-family: "Segoe UI", Arial, sans-serif; }}
  body {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
  .page {{ position: relative; width: 210mm; height: 297mm; padding: 17mm 18mm 16mm; margin: 0 auto; overflow: hidden; background: #fbfaf7; page-break-after: always; }}
  .page:last-child {{ page-break-after: auto; }}
  header, footer {{ position: absolute; left: 18mm; right: 18mm; display: flex; justify-content: space-between; color: #687076; font-size: 7.5pt; font-weight: 700; letter-spacing: .08em; }}
  header {{ top: 9mm; }} footer {{ bottom: 7mm; font-weight: 500; letter-spacing: 0; }}
  .page-title {{ margin: 7mm 0 3mm; font: 700 20pt/1.1 Georgia, serif; }}
  .rule {{ height: 1px; background: #cfd2d1; margin-bottom: 9mm; }}
  h2 {{ margin: 0 0 2.3mm; font: 700 11pt/1.2 Georgia, serif; }}
  p {{ margin: 0; font-size: 9.1pt; line-height: 1.55; color: #303536; }}
  .copy {{ margin-bottom: 6mm; }}
  .eyebrow {{ display: inline-block; padding: 2.3mm 5mm; border-radius: 12mm; background: #171717; color: white; font-size: 8pt; font-weight: 800; letter-spacing: .12em; }}
  .cover h1 {{ margin: 17mm 0 4mm; font: 800 38pt/1 Georgia, serif; }}
  .cover h2 {{ max-width: 160mm; font: 700 25pt/1.18 Georgia, serif; }}
  .cover .subtitle {{ margin-top: 9mm; max-width: 158mm; font-size: 11pt; line-height: 1.55; color: #5e6669; }}
  .cover .meta {{ margin-top: 12mm; padding-top: 8mm; border-top: 1px solid #c9cdcc; font-size: 9.5pt; line-height: 1.75; color: #596164; }}
  .cover .meta strong {{ color: #171717; }}
  .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 4mm; margin-top: 14mm; }}
  .stat {{ min-height: 27mm; padding: 5mm; border: 1px solid #d8dbda; border-radius: 3mm; background: white; }}
  .stat b {{ display: block; color: #0f766e; font: 800 18pt/1 Georgia, serif; }}
  .stat span {{ display: block; margin-top: 3mm; color: #697174; font-size: 7pt; font-weight: 800; letter-spacing: .08em; }}
  .architecture {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8mm; align-items: center; margin: 9mm 0 14mm; }}
  .node {{ position: relative; min-height: 29mm; padding: 5mm 4mm; border: 1px solid #ccd2d0; border-radius: 3mm; background: white; }}
  .node:not(:last-child)::after {{ content: "→"; position: absolute; right: -7mm; top: 9mm; color: #0f766e; font-size: 18pt; font-weight: 700; }}
  .node b {{ display: block; font-size: 8pt; letter-spacing: .04em; }} .node span {{ display: block; margin-top: 3mm; color: #626a6c; font-size: 7.5pt; line-height: 1.4; }}
  .outputs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 7mm; margin: 3mm 18mm 14mm; }}
  .outputs .node {{ border-top: 3px solid #0f766e; }}
  .methods {{ display: grid; grid-template-columns: 1fr 1fr; gap: 5mm 8mm; }}
  .methods .copy {{ padding: 5mm; margin: 0; border: 1px solid #d9dcdb; border-radius: 3mm; background: white; }}
  table {{ width: 100%; border-collapse: collapse; margin: 5mm 0 9mm; font-size: 7.8pt; }}
  th {{ padding: 3.4mm 2mm; background: #171717; color: white; text-align: center; }} td {{ padding: 3.1mm 2mm; border: 1px solid #d8dcda; text-align: center; background: white; }}
  .bars {{ margin-top: 4mm; }} .bar-row {{ display: grid; grid-template-columns: 54mm 1fr 9mm; gap: 4mm; align-items: center; margin: 3.3mm 0; font-size: 8pt; text-transform: capitalize; }}
  .bar-track {{ height: 5mm; border-radius: 3mm; background: #e5e8e7; overflow: hidden; }} .bar-track i {{ display: block; height: 100%; border-radius: 3mm; background: #0f766e; }}
  .provenance {{ margin-top: 8mm; padding: 5mm; border-left: 3px solid #b45309; background: #fff8ed; color: #596164; font-size: 8pt; line-height: 1.5; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 3mm; margin: 5mm 0 8mm; }}
  .metric {{ min-height: 18mm; padding: 3.5mm; border: 1px solid #d8dcda; border-radius: 2mm; background: white; }}
  .metric b {{ display: block; color: #0f766e; font: 800 13pt/1 Georgia, serif; }} .metric span {{ display: block; margin-top: 2mm; color: #687073; font-size: 6.6pt; font-weight: 700; line-height: 1.25; }}
  .gallery {{ display: grid; grid-template-columns: 1fr 1fr; gap: 7mm; }} figure {{ margin: 0; }} figure img {{ width: 100%; height: 67mm; object-fit: contain; background: #171717; border: 1px solid #cbd0ce; }} figcaption {{ margin-top: 2mm; color: #5e6669; font-size: 7pt; line-height: 1.35; }}
  .note {{ margin-top: 7mm; padding: 5mm; border-radius: 3mm; background: #edf7f5; color: #3b5652; font-size: 8pt; line-height: 1.5; }}
  .references {{ margin: 4mm 0 0; padding-left: 6mm; color: #303536; font-size: 8pt; line-height: 1.5; }} .references li {{ margin-bottom: 2.5mm; }}
</style></head><body>
  <article class="page cover">
    <span class="eyebrow">TECHNICAL RESEARCH DRAFT</span>
    <h1>NETRA-4D</h1>
    <h2>Privacy-Aware Temporal Evidence for Examination Surveillance</h2>
    <p class="subtitle">A candidate–workstation contextual pipeline for per-frame inference, sustained behavioral evidence, real-time delivery, and human adjudication.</p>
    <div class="meta"><strong>Team DrishtiAI — DI1_39 Digital Negatives</strong><br>Validated run: {escape(run['run_id'])}<br>Generated: {generated_at}<br>Document status: Technical research draft</div>
    <div class="stats">{stat_cards}</div>
    {page_footer(1)}
  </article>

  <article class="page">{page_header('01', 'Abstract and Research Scope')}
    {section('Abstract', 'NETRA-4D is a privacy-aware computer-vision pipeline for transforming examination video into time-aligned review evidence. It combines person pose, object detection, ByteTrack identities, face masking, a 2-D head-direction gaze proxy, candidate-to-workstation association, and temporal persistence. The system emits annotated video, exact source timestamps, structured JSONL event records, high-confidence snapshots, clips, and real-time browser telemetry. It is designed to prioritize auditable human review rather than issue autonomous cheating verdicts.')}
    {section('Research objective', 'Evaluate whether a modular temporal and contextual reasoning layer can organize real per-frame detections into useful examination-review evidence while preserving identity continuity, source-time semantics, and explicit uncertainty.')}
    {section('Primary contributions', '(1) Lossless frame-by-frame processing for uploaded evidence, with low-latency stale-frame handling for live sources. (2) Candidate and assigned-PC mapping with workstation-relative attention and interaction proxies. (3) Confidence accumulation tied to sustained behavior rather than a single frame. (4) Privacy masking before display and persistence. (5) A complete backend-to-frontend path using REST, WebSocket telemetry, MJPEG delivery, RTSP/MediaMTX ingest, SQLite review state, and verified proof-of-work fallback media.')}
    {section('Evaluation boundary', 'The run reports measured counts and runtime provenance. It does not report precision, recall, F1, or cheating-detection accuracy because no adjudicated ground-truth corpus was available. Event confidence is an evidence score for prioritization, not the probability that misconduct occurred.')}
    {page_footer(2)}
  </article>

  <article class="page">{page_header('02', 'System Architecture and Data Flow')}
    <div class="architecture"><div class="node"><b>VIDEO SOURCE</b><span>Browser webcam / upload / RTSP</span></div><div class="node"><b>PER-FRAME MODELS</b><span>Pose · objects · ByteTrack</span></div><div class="node"><b>PRIVACY + CONTEXT</b><span>Blur · gaze proxy · PC map</span></div><div class="node"><b>TEMPORAL ENGINE</b><span>Persistence · confidence</span></div></div>
    <div class="outputs"><div class="node"><b>DURABLE EVIDENCE</b><span>JSONL · JPEG · clips · annotated MP4</span></div><div class="node"><b>REAL-TIME DELIVERY</b><span>WebSocket · MJPEG · REST · MediaMTX</span></div></div>
    {section('Input policy', 'Uploaded files use decoder back-pressure and source timestamps (frame index / FPS), ensuring every decoded frame is processed. Webcam and RTSP feeds discard stale queued frames when inference falls behind, preserving current-view latency. A live source is armed first; only an invigilator Start Monitoring action permits frames to enter inference.')}
    {section('Single-source data contract', 'Every live session exposes one coherent source: annotated frames, current detections, candidate and workstation assignments, temporal events, evidence references, counters, and health state all share the same session identifier. The frontend therefore cannot accidentally mix telemetry from one source with imagery from another.')}
    {section('Output roots', 'OUTPUT/live_sessions stores active source state; OUTPUT/live_runs stores completed live evidence; OUTPUT/netra4d_final_20260823 stores the validated offline proof-of-work run. Browser-compatible H.264 samples provide a deterministic fallback when live inference or device permissions are unavailable.')}
    {page_footer(3)}
  </article>

  <article class="page">{page_header('03', 'Methodology')}
    <div class="methods">
      {section('Person pose and identity', 'YOLO26 pose generates person boxes and 17 COCO keypoints. ByteTrack provides candidate identity continuity. Candidate IDs are evidence-track identifiers, not biometric identities.')}
      {section('Objects and small-item review', 'YOLO26 object detections retain their raw class and confidence. Cell-phone detections remain review evidence. Low-hand possible-chit events are behavioral proxies until a genuinely labeled custom chit model is trained and validated.')}
      {section('Privacy and gaze', 'Facial keypoints define a face/head mask applied before rendering or persistence. Gaze is a 2-D head-direction proxy from eye/nose geometry; it is explicitly not iris tracking.')}
      {section('Candidate–PC context', 'Tracked candidates are associated one-to-one with workstation screens where geometry supports the mapping. Assigned device labels use PC-<candidate ID>. Unassigned chairs are omitted; assigned-chair overlays are limited to the initial configured period.')}
      {section('Temporal confidence', 'Evidence begins with model or geometry signals and increases only while the behavior remains active. The sustained boost is bounded, and activation/crossing timestamps are written using source-video time.')}
      {section('Human review and monitoring control', 'Events are persisted with lifecycle, evidence kind, score, source signals, candidate/workstation IDs, exact timestamps, keyframes, clips, and specialist-resolution fields. An invigilator-controlled gate records armed, start, pause, resume, and stop actions with UTC time and operator ID; paused live frames are not inferred or logged. Final disposition belongs to a reviewer.')}
    </div>{page_footer(4)}
  </article>

  <article class="page">{page_header('04', 'Measured Results from the Validated GPU Run')}
    <div class="metric-grid">
      <div class="metric"><b>{event_candidate_tracks}</b><span>EVENT-INVOLVED CANDIDATE TRACK IDs</span></div>
      <div class="metric"><b>{mapped_candidate_tracks}</b><span>CANDIDATE TRACK IDs MAPPED TO PCs</span></div>
      <div class="metric"><b>{len(event_counts)}</b><span>TEMPORAL EVENT TYPES</span></div>
      <div class="metric"><b>{total_temporal_duration:.1f}s</b><span>SUMMED EVENT PERSISTENCE</span></div>
      <div class="metric"><b>{average_temporal_duration:.2f}s</b><span>MEAN EVENT DURATION</span></div>
      <div class="metric"><b>{maximum_temporal_duration:.2f}s</b><span>MAX EVENT DURATION</span></div>
      <div class="metric"><b>{total_privacy_masks:,}</b><span>PRIVACY MASK OPERATIONS</span></div>
      <div class="metric"><b>{total_gaze_observations:,}</b><span>KNOWN GAZE-PROXY OBSERVATIONS</span></div>
    </div>
    <table><thead><tr><th>Asset</th><th>Frames</th><th>Duration (s)</th><th>Event candidate IDs</th><th>Event types</th><th>Events</th><th>Evidence stills</th></tr></thead><tbody>{result_rows}</tbody></table>
    <h2>Temporal event distribution</h2><div class="bars">{event_bars}</div>
    <div class="provenance"><strong>Temporal coverage:</strong> source-local event timestamps range from {timestamp_min:.2f}s to {timestamp_max:.2f}s; exact start, crossing, duration, confidence, and candidate IDs remain in each JSONL record. <strong>Candidate counts are session-local track IDs, not biometric person counts</strong>, because tracker IDs can fragment after long occlusions.<br><strong>Runtime provenance:</strong> {escape(str(runtime['gpu_name']))} · CUDA device {escape(str(runtime['actual_device']))} · Torch {escape(str(runtime['torch']))} · Python {escape(str(runtime['python']))}. Frame accounting and hashes are preserved in <code>run.json</code> and <code>COMPLETE.json</code>.</div>
    {page_footer(5)}
  </article>

  <article class="page">{page_header('05', 'High-Confidence Evidence Gallery')}
    <div class="gallery">{gallery_cards}</div>
    <div class="note"><strong>Selection rule.</strong> A snapshot is written when a sustained review event crosses the configured evidence threshold. Filename fields retain frame index, exact source timestamp, evidence confidence, and event label. Images are privacy-masked and intended for reviewer inspection.</div>
    {page_footer(6)}
  </article>

  <article class="page">{page_header('06', 'Limitations, Ethics, and Conclusion')}
    {section('Known limitations', 'No reviewed ground-truth labels are available, so accuracy metrics are intentionally absent. The validated run produced no raw cell-phone still at the strict COCO confidence threshold of 0.70. A trained custom chit checkpoint is not present. Head direction is not eye gaze. Automatic workstation calibration is camera-dependent and may fail under severe occlusion or insufficient stable observations. CPU live inference can run below source FPS; GPU deployment is recommended for higher throughput.')}
    {section('Risk controls', 'Face/head masking precedes display; exact timestamps and provenance are retained; all behavioral labels include review semantics; uncertainty and counter-evidence remain visible; local file integrity gates exclude incomplete or size-mismatched runs; and no event is converted into a disciplinary conclusion without a human decision.')}
    {section('Conclusion', 'The validated evidence demonstrates a coherent source-to-interface pipeline: real videos are decoded, inferred, tracked, privacy-masked, contextually mapped, temporally accumulated, persisted, and streamed to a browser. Scientific claims about misconduct classification require a separately labeled dataset, blinded human adjudication, predefined metrics, and external validation.')}
    <h2>References</h2><ol class="references"><li>Y. Zhang et al., “ByteTrack: Multi-Object Tracking by Associating Every Detection Box,” ECCV, 2022.</li><li>Ultralytics, official YOLO model and tracking documentation; checkpoint hashes are recorded in the run manifest.</li><li>OpenCV, video decoding, encoding, and image-processing runtime used by the evidence pipeline.</li><li>NETRA-4D internal artifacts: <code>run.json</code>, <code>detections.jsonl</code>, <code>review_events.jsonl</code>, <code>COMPLETE.json</code>, and CBT workstation maps.</li></ol>
    {page_footer(7)}
  </article>
</body></html>"""


def generate(run_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    html_output = output.with_suffix(".html")
    html_output.write_text(build_html(run_path), encoding="utf-8")
    browser = locate_browser()
    with tempfile.TemporaryDirectory(prefix="netra4d-report-") as profile:
        result = subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile}",
                f"--print-to-pdf={output}",
                html_output.resolve().as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    if result.returncode != 0 or not output.is_file() or output.stat().st_size < 10_000:
        detail = (result.stderr or result.stdout or "unknown browser error").strip()
        raise RuntimeError(f"PDF rendering failed: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("OUTPUT/netra4d_final_20260823/run.json"))
    parser.add_argument("--output", type=Path, default=Path("OUTPUT/NETRA_4D_Research_Report.pdf"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    generate((ROOT / args.run).resolve(), output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
