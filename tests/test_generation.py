import pytest

pytest.importorskip("diffsynth")

from genesis import generation  # noqa: E402
from genesis.generation import frame_count, resolve_onload_device  # noqa: E402


@pytest.mark.parametrize(("duration", "fps", "expected"), [(5, 16, 77), (5, 24, 117), (0, 16, 13), (1, 24, 21)])
def test_frame_count_is_4n_plus_1(duration, fps, expected):
    assert frame_count(duration, fps) == expected
    assert (expected - 1) % 4 == 0


def test_onload_device(monkeypatch):
    monkeypatch.setattr(generation, "total_ram_gb", lambda: 14.0)
    assert resolve_onload_device("auto") == "disk"
    monkeypatch.setattr(generation, "total_ram_gb", lambda: 64.0)
    assert resolve_onload_device("auto") == "cpu"
    assert resolve_onload_device("disk") == "disk"
