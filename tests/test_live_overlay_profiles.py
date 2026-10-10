from fastapi.params import Query

from drishti.api.app import _live_runner_for, start_webcam


def test_full_context_profile_reuses_established_cbt_live_runner():
    assert _live_runner_for({"source_kind": "RTSP", "overlay_profile": "full_context"}) == "run_live_webcam.py"


def test_product_operator_profile_retains_canonical_rtsp_runner():
    assert _live_runner_for({"source_kind": "RTSP", "overlay_profile": "product_operator"}) == "run_product_pipeline.py"


def test_old_session_without_profile_keeps_product_runner():
    assert _live_runner_for({"source_kind": "RTSP"}) == "run_product_pipeline.py"


def test_new_rtsp_sessions_default_to_full_context_profile():
    default = start_webcam.__defaults__[-1]
    assert isinstance(default, Query)
    assert default.default == "full_context"
