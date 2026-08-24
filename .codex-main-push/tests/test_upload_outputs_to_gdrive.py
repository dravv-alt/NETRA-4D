from pathlib import Path

from scripts.upload_outputs_to_gdrive import (
    DEFAULT_DESTINATION_FOLDER_ID,
    escape_drive_query,
    iter_local_files,
)


def test_destination_is_the_user_supplied_folder():
    assert DEFAULT_DESTINATION_FOLDER_ID == "1vEfaikd1G9_dDr0-fbDm01w20Mk0TI18"


def test_drive_query_escaping():
    assert escape_drive_query("candidate's \\ video") == "candidate\\'s \\\\ video"


def test_iter_local_files_is_recursive_and_sorted():
    fixture = Path(__file__).parent / "fixtures" / "upload_tree"
    relative = [path.relative_to(fixture).as_posix() for path in iter_local_files(fixture)]
    assert relative == ["VIDEO_01/annotated.mp4", "VIDEO_02/summary.json"]
