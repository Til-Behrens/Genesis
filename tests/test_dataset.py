import pytest

from genesis.finetuning.dataset import prepare_dataset, read_jsonl, to_diffsynth_row, write_jsonl


def test_row_uses_diffsynth_keys():
    row = to_diffsynth_row({"file_name": "a.mp4", "prompt": " Folie ", "start_time": 6, "extra": 1})
    assert row == {"video": "a.mp4", "prompt": "Folie", "start_time": 6}


def test_row_without_video_is_rejected():
    with pytest.raises(ValueError):
        to_diffsynth_row({"prompt": "x"})


def test_prepare_skips_missing_and_uncaptioned(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (clips / name).touch()
    metadata = tmp_path / "metadata.jsonl"
    write_jsonl(metadata, [
        {"file_name": "a.mp4", "prompt": "eine Folie"},
        {"file_name": "b.mp4", "prompt": ""},
        {"file_name": "c.mp4", "prompt": "fehlt"},
        {"prompt": "kaputt"},
    ])
    output = tmp_path / "out.jsonl"

    result = list(prepare_dataset(clips, metadata, output))

    assert result[-1].status == "complete"
    assert result[-1].data["skipped"] == {"invalid": 1, "missing video": 1, "no caption": 1}
    assert read_jsonl(output) == [{"video": "a.mp4", "prompt": "eine Folie"}]


def test_prepare_fails_without_usable_rows(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    write_jsonl(metadata, [{"file_name": "a.mp4", "prompt": ""}])
    assert list(prepare_dataset(tmp_path, metadata, tmp_path / "out.jsonl"))[-1].status == "error"
