import pytest

from genesis.config import WAN_MODELS
from genesis.finetuning import training
from genesis.finetuning.training import TrainingOptions, build_command, origin_paths, train_lora


def _flags(cmd):
    args = cmd[cmd.index(next(a for a in cmd if a.endswith("train.py"))) + 1:]
    return dict(zip(args[::2], args[1::2], strict=True))


def test_origin_paths_match_diffsynth_recipe():
    spec = WAN_MODELS["5B"]
    assert origin_paths(spec, spec.train_runs[0]) == (
        "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,"
        "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,"
        "Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
    )


def test_command_uses_video_key_and_skips_unset_options(tmp_path):
    spec = WAN_MODELS["1.3B"]
    cmd = build_command(tmp_path / "train.py", spec, spec.train_runs[0], tmp_path, tmp_path / "m.jsonl",
                        tmp_path / "out", TrainingOptions(epochs=2))
    flags = _flags(cmd)
    assert cmd[:2] == ["accelerate", "launch"]
    assert flags["--data_file_keys"] == "video"
    assert flags["--num_epochs"] == "2"
    assert "--height" not in flags and "--find_unused_parameters" not in cmd


@pytest.mark.parametrize("options", [TrainingOptions(height=500), TrainingOptions(num_frames=80)])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        options.validate()


def test_moe_model_trains_one_adapter_per_expert(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(training, "find_train_script", lambda root=None: tmp_path / "train.py")
    monkeypatch.setattr(training, "_stream", lambda cmd, env: calls.append((cmd, env)) or iter(["step 1"]))
    metadata = tmp_path / "m.jsonl"
    metadata.touch()

    result = list(train_lora("14B", tmp_path, metadata, tmp_path / "out"))

    assert result[-1].status == "complete"
    outputs = [_flags(cmd)["--output_path"] for cmd, _ in calls]
    assert outputs == [str(tmp_path / "out" / "high_noise"), str(tmp_path / "out" / "low_noise")]
    assert all("DIFFSYNTH_MODEL_BASE_PATH" in env for _, env in calls)


def test_unknown_model_is_an_error_event(tmp_path):
    assert list(train_lora("7B", tmp_path, tmp_path, tmp_path))[-1].status == "error"


def test_failed_run_stops_training(tmp_path, monkeypatch):
    def failing(cmd, env):
        yield "loading"
        raise RuntimeError("exit code 1")

    monkeypatch.setattr(training, "find_train_script", lambda root=None: tmp_path / "train.py")
    monkeypatch.setattr(training, "_stream", failing)
    metadata = tmp_path / "m.jsonl"
    metadata.touch()
    result = list(train_lora("14B", tmp_path, metadata, tmp_path / "out"))
    assert result[-1].status == "error" and "high_noise" in result[-1].message
    assert sum(event.status == "info" for event in result) == 1
