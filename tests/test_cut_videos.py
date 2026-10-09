import pytest

from genesis.finetuning.cut_videos import clip_starts, cut_videos


def test_clip_starts_with_overlap():
    assert clip_starts(30, 8, 2) == [0, 6, 12, 18]


def test_clip_starts_exact_fit_and_too_short():
    assert clip_starts(8, 8, 0) == [0]
    assert clip_starts(7.9, 8, 0) == []


@pytest.mark.parametrize("overlap", [8, 9, -1])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError):
        clip_starts(30, 8, overlap)
    assert next(cut_videos(".", ".", "m.jsonl", clip_length=8, overlap=overlap)).status == "error"
