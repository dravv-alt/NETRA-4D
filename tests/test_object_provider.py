import pytest

from drishti.providers.object_provider import ObjectDisambiguationProvider


def test_iou_deduplicates_overlapping_small_object_proposals():
    assert ObjectDisambiguationProvider._iou((0, 0, 10, 10), (1, 1, 9, 9)) == pytest.approx(0.64)
    assert ObjectDisambiguationProvider._iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_four_overlapping_tiles_cover_full_frame():
    tiles = ObjectDisambiguationProvider._tiles(100, 80)
    assert len(tiles) == 4
    assert tiles[0][:2] == (0, 0)
    assert tiles[-1][2:] == (100, 80)


def test_configured_chit_checkpoint_must_really_exist():
    provider = ObjectDisambiguationProvider(
        model_name="unused.pt",
        device="cpu",
        chit_model_name="outputs/definitely_missing_chit_checkpoint.pt",
    )
    with pytest.raises(FileNotFoundError, match="real chit checkpoint"):
        provider.start()
