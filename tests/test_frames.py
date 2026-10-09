import numpy as np

from genesis.finetuning.captioning.frames import select_keyframes


def _frames(values):
    return [np.full((4, 4, 3), v, dtype=np.uint8) for v in values]


def test_static_clip_yields_one_keyframe():
    assert select_keyframes(_frames([10] * 8), max_frames=6) == [0]


def test_changes_become_keyframes():
    assert select_keyframes(_frames([10, 10, 50, 50, 90]), max_frames=6) == [0, 2, 4]


def test_keyframes_are_thinned_to_max():
    keys = select_keyframes(_frames(range(0, 200, 10)), max_frames=4)
    assert len(keys) == 4 and keys[0] == 0
