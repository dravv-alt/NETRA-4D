from pathlib import Path

import pytest

from drishti.data.google_drive import extract_drive_file_id, load_drive_manifest


def test_extract_drive_file_id_from_shared_url():
    value = "https://drive.google.com/file/d/12AuvNrLQun7DLr3Oxq60WC6bLeZNWeQI/view?usp=sharing"
    assert extract_drive_file_id(value) == "12AuvNrLQun7DLr3Oxq60WC6bLeZNWeQI"


def test_manifest_contains_only_the_five_supplied_assets():
    manifest = Path(__file__).resolve().parents[1] / "drishti4d" / "configs" / "datasets.json"
    assets = load_drive_manifest(manifest)
    assert [asset.asset_id for asset in assets] == [
        "VIDEO_01", "VIDEO_02", "VIDEO_03", "VIDEO_04", "VIDEO_05"
    ]
    assert all(asset.expected_bytes and asset.expected_bytes > 0 for asset in assets)


def test_invalid_drive_reference_is_rejected():
    with pytest.raises(ValueError):
        extract_drive_file_id("bad")
