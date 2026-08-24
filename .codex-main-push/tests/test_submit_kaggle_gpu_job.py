from scripts.submit_kaggle_gpu_job import ENTRYPOINT


def test_kaggle_entrypoint_requires_cuda_and_full_video_renderer():
    assert "render_annotated_videos.py" in ENTRYPOINT
    assert '"--require-cuda"' in ENTRYPOINT
    assert '"--device", "cuda:0"' in ENTRYPOINT
    assert '"--phone-confidence", "0.70"' in ENTRYPOINT
