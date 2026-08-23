"""Transcode completed NETRA-4D videos without modifying verified originals.

AV1 provides the smallest archival files but is CPU encoded on Tesla T4/P100.
A source is never deleted unless the new file passes ffprobe, is smaller, and
deletion is explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ENCODERS = {
    "av1": ("libsvtav1", ".av1.mp4"),
    "h265": ("libx265", ".hevc.mp4"),
    "h264": ("libx264", ".h264.mp4"),
}


def build_command(source: Path, destination: Path, codec: str, crf: int, preset: str) -> list[str]:
    encoder, _ = ENCODERS[codec]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
        "-c:v", encoder, "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
    ]
    if codec == "av1":
        command.extend(["-tag:v", "av01"])
    command.append(str(destination))
    return command


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        raise RuntimeError(f"No video stream in {path}")
    return payload


def validate(source: Path, destination: Path) -> tuple[dict, dict]:
    source_probe = probe(source)
    output_probe = probe(destination)
    src_stream = source_probe["streams"][0]
    dst_stream = output_probe["streams"][0]
    if (src_stream.get("width"), src_stream.get("height")) != (dst_stream.get("width"), dst_stream.get("height")):
        raise RuntimeError(f"Resolution changed while transcoding {source.name}")
    src_duration = float(source_probe["format"].get("duration", 0))
    dst_duration = float(output_probe["format"].get("duration", 0))
    if src_duration <= 0 or abs(src_duration - dst_duration) > max(0.25, src_duration * 0.002):
        raise RuntimeError(f"Duration mismatch for {source.name}: {src_duration} vs {dst_duration}")
    if destination.stat().st_size >= source.stat().st_size:
        raise RuntimeError(f"Compressed file is not smaller than {source.name}")
    return source_probe, output_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed run containing VIDEO_XX folders")
    parser.add_argument("--codec", choices=ENCODERS, default="av1")
    parser.add_argument("--crf", type=int, default=28, help="Lower is higher quality and larger")
    parser.add_argument("--preset", default="8", help="SVT-AV1 8 is a practical archival default")
    parser.add_argument("--delete-source-after-verify", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process N videos for a trial; 0 means all")
    args = parser.parse_args()

    if not 0 <= args.crf <= 63:
        parser.error("--crf must be between 0 and 63")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise SystemExit(f"{executable} is required but was not found")

    run_dir = args.run_dir.resolve()
    if not (run_dir / "run.json").is_file():
        raise SystemExit(f"Not a NETRA-4D run directory (missing run.json): {run_dir}")

    sources = sorted(run_dir.glob("VIDEO_*/VIDEO_*_annotated.mp4"))
    sources = [p for p in sources if not any(tag in p.stem for tag in (".av1", ".hevc", ".h264"))]
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit("No annotated source videos found")

    _, suffix = ENCODERS[args.codec]
    receipt = {"codec": args.codec, "crf": args.crf, "preset": args.preset, "files": []}
    for source in sources:
        destination = source.with_name(source.stem + suffix)
        print(f"Transcoding {source.name} -> {destination.name}", flush=True)
        subprocess.run(build_command(source, destination, args.codec, args.crf, args.preset), check=True)
        try:
            _, output_probe = validate(source, destination)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        source_bytes = source.stat().st_size
        output_bytes = destination.stat().st_size
        item = {
            "source": str(source.relative_to(run_dir)),
            "output": str(destination.relative_to(run_dir)),
            "source_bytes": source_bytes,
            "output_bytes": output_bytes,
            "reduction_percent": round((1 - output_bytes / source_bytes) * 100, 2),
            "output_probe": output_probe,
            "source_deleted": False,
        }
        if args.delete_source_after_verify:
            source.unlink()
            item["source_deleted"] = True
        receipt["files"].append(item)

    receipt_path = run_dir / f"compression_{args.codec}_crf{args.crf}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"Verified receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
