from pathlib import Path

from scripts.compress_output_videos import build_command


def test_av1_command_uses_crf_and_av1_tag() -> None:
    command = build_command(Path("input.mp4"), Path("output.av1.mp4"), "av1", 28, "8")
    assert command[command.index("-c:v") + 1] == "libsvtav1"
    assert command[command.index("-crf") + 1] == "28"
    assert command[command.index("-tag:v") + 1] == "av01"
    assert "-map" in command


def test_h265_command_does_not_set_av1_tag() -> None:
    command = build_command(Path("input.mp4"), Path("output.hevc.mp4"), "h265", 24, "medium")
    assert command[command.index("-c:v") + 1] == "libx265"
    assert "-tag:v" not in command
