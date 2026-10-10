from pathlib import Path

import pytest

from drishti.data import video_policy
from drishti.data.google_drive import DriveAsset, load_drive_manifest, materialize_drive_asset


def test_original_video03_is_redirected_and_cannot_exceed_2200(monkeypatch):
    monkeypatch.setattr(Path, 'is_file', lambda self: True)
    source, bound = video_policy.bounded_video_source('input/videos/VIDEO_03_mobile_phone_exam_hall_01.mkv', 3200)
    assert source == str(video_policy.VIDEO03)
    assert bound == 2200
    assert video_policy.bounded_video_source('original.mkv', 0, 'VIDEO_03')[1] == 2200
    assert video_policy.bounded_video_source('original.mkv', 10, 'VIDEO_03')[1] == 10
    assert video_policy.bounded_video_source('VIDEO_04.mkv', 0) == ('VIDEO_04.mkv', 0)


def test_missing_canonical_video_does_not_fall_back_to_long_original(monkeypatch):
    monkeypatch.setattr(Path, 'is_file', lambda self: False)
    with pytest.raises(FileNotFoundError):
        video_policy.bounded_video_source('anything.mkv', video_id='VIDEO_03')
    asset = DriveAsset('VIDEO_03', 'original_drive_id', 'source', 'original.mkv', local_path=video_policy.VIDEO03, max_frames=2200)
    with pytest.raises(FileNotFoundError):
        with materialize_drive_asset(asset):
            pass


def test_both_manifest_entrypoints_resolve_to_same_bounded_source():
    root = Path(__file__).resolve().parents[1]
    for filename in ('configs/datasets.json', 'drishti4d/configs/datasets.json'):
        asset = next(a for a in load_drive_manifest(root/filename) if a.asset_id == 'VIDEO_03')
        assert asset.local_path == video_policy.VIDEO03
        assert asset.max_frames == 2200
