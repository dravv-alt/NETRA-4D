from unittest.mock import patch

from scripts.run_kaggle_gpu_pipeline import completed_run_root
from scripts.submit_kaggle_gpu_job import ENTRYPOINT_TEMPLATE


def test_completed_run_root_requires_the_expected_unique_run():
    from pathlib import Path

    run = Path("download") / "nested" / "privacy_v1"
    with patch.object(Path, "rglob", return_value=[run / "run.json"]):
        assert completed_run_root(Path("download"), "privacy_v1") == run


def test_submission_entrypoint_renders_unique_run_id_without_credentials():
    entrypoint = ENTRYPOINT_TEMPLATE.format(
        run_id="privacy_v1", asset_arguments="", source_archive_b64="payload"
    )
    assert '"--run-id", \'privacy_v1\'' in entrypoint
    assert "kaggle.json" not in entrypoint
