"""Resolve known Video 3 inputs to the canonical 2200-frame source."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VIDEO03 = ROOT / 'input/videos/VIDEO_03_mobile_phone_exam_hall_01_first_2200_frames.mp4'


def bounded_video_source(source, max_frames=0, video_id=None):
    name = Path(str(source)).name.lower()
    is_video03 = str(video_id).upper() == 'VIDEO_03' or name.startswith('video_03_mobile_phone_exam_hall') or name == '01.candidate was found using a mobile phone in the examination hall..mkv'
    if not is_video03:
        return source, max_frames
    if not VIDEO03.is_file():
        raise FileNotFoundError(f'Canonical Video 3 input missing: {VIDEO03}')
    return str(VIDEO03), min(max_frames, 2200) if max_frames > 0 else 2200
