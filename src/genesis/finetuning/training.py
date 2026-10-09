"""LoRA fine-tuning through DiffSynth's wanvideo training script."""
import logging
import os
import shlex
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genesis import events
from genesis.config import CACHE_DIR, DIFFSYNTH_ROOT, WAN_MODELS, TrainRun, WanModelSpec
from genesis.events import Event

logger = logging.getLogger(__name__)

TRAIN_SCRIPT = Path("examples/wanvideo/model_training/train.py")
TEXT_ENCODER_FILE = "models_t5_umt5-xxl-enc-bf16.pth"


@dataclass
class TrainingOptions:
    """Hyperparameters passed through to DiffSynth's train.py; `None` keeps its default.

    Leaving height and width unset trains at the clip resolution (dynamic resolution,
    capped by DiffSynth's `--max_pixels`).
    """

    epochs: int = 5
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    lora_rank: int = 32
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    dataset_repeat: int = 100
    dataset_num_workers: int = 0
    save_steps: int | None = 100
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    find_unused_parameters: bool = False
    initialize_model_on_cpu: bool = False
    accelerate_config_file: str | None = None

    def validate(self) -> None:
        """Raise `ValueError` for values Wan cannot train with."""
        for name in ("height", "width"):
            value = getattr(self, name)
            if value is not None and value % 16:
                raise ValueError(f"{name} must be a multiple of 16, got {value}")
        if self.num_frames is not None and (self.num_frames - 1) % 4:
            raise ValueError(f"num_frames must be 4n+1, got {self.num_frames}")


def origin_paths(spec: WanModelSpec, run: TrainRun) -> str:
    """DiffSynth's `--model_id_with_origin_paths` value for one training run."""
    patterns = (run.dit_pattern, TEXT_ENCODER_FILE, spec.train_vae_file)
    return ",".join(f"{spec.model_id}:{pattern}" for pattern in patterns)


def find_train_script(diffsynth_root: Path | str | None = None) -> Path:
    """Locate train.py in `diffsynth_root`, `GENESIS_DIFFSYNTH_ROOT` or above the installed package.

    The script ships with the DiffSynth-Studio repository, not the pip package; an
    editable install from a clone is found through the package path.

    Raises:
        FileNotFoundError: No candidate directory contains the script.
    """
    candidates = [Path(root) for root in (diffsynth_root, DIFFSYNTH_ROOT) if root]
    try:
        import diffsynth
    except ImportError:
        pass
    else:
        candidates += Path(diffsynth.__file__).resolve().parents

    for candidate in dict.fromkeys(candidates):
        if (candidate / TRAIN_SCRIPT).exists():
            return candidate / TRAIN_SCRIPT
    raise FileNotFoundError(
        "DiffSynth train script not found. Set GENESIS_DIFFSYNTH_ROOT to a DiffSynth-Studio clone "
        f"(containing {TRAIN_SCRIPT})."
    )


def build_command(
    script: Path,
    spec: WanModelSpec,
    run: TrainRun,
    dataset_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    options: TrainingOptions,
) -> list[str]:
    """Assemble the `accelerate launch` command for one training run."""
    cmd = ["accelerate", "launch"]
    if options.accelerate_config_file:
        cmd += ["--config_file", str(options.accelerate_config_file)]
    cmd.append(str(script))

    flags: dict[str, Any] = {
        "dataset_base_path": dataset_dir,
        "dataset_metadata_path": metadata_path,
        "dataset_repeat": options.dataset_repeat,
        "dataset_num_workers": options.dataset_num_workers,
        "data_file_keys": "video",
        "model_id_with_origin_paths": origin_paths(spec, run),
        "learning_rate": options.learning_rate,
        "num_epochs": options.epochs,
        "remove_prefix_in_ckpt": "pipe.dit.",
        "output_path": output_dir,
        "lora_base_model": "dit",
        "lora_target_modules": options.lora_target_modules,
        "lora_rank": options.lora_rank,
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "min_timestep_boundary": run.min_timestep_boundary,
        "max_timestep_boundary": run.max_timestep_boundary,
        "height": options.height,
        "width": options.width,
        "num_frames": options.num_frames,
        "save_steps": options.save_steps,
    }
    for flag, value in flags.items():
        if value is not None:
            cmd += [f"--{flag}", str(value)]
    if options.find_unused_parameters:
        cmd.append("--find_unused_parameters")
    if options.initialize_model_on_cpu:
        cmd.append("--initialize_model_on_cpu")
    return cmd


def _stream(cmd: list[str], env: dict[str, str]) -> Iterator[str]:
    """Yield the process output line by line; stop the process if the consumer stops early."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if line.strip():
                yield line.rstrip()
        if process.wait() != 0:
            raise RuntimeError(f"exit code {process.returncode}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


def train_lora(
    model_size: str,
    dataset_dir: Path | str,
    metadata_path: Path | str,
    output_dir: Path | str,
    options: TrainingOptions | None = None,
    diffsynth_root: Path | str | None = None,
) -> Iterator[Event]:
    """Train LoRA adapters for one Wan model.

    Runs one `accelerate launch` per entry in the model's `train_runs`; Wan2.2 A14B gets
    one adapter per expert, in `output_dir/high_noise` and `output_dir/low_noise`.

    Yields:
        `info` per run start, `progress` per log line, then `complete` (or `error`).
    """
    options = options or TrainingOptions()
    dataset_dir, metadata_path, output_dir = Path(dataset_dir), Path(metadata_path), Path(output_dir)

    try:
        if model_size not in WAN_MODELS:
            raise ValueError(f"Unknown model size {model_size!r}, expected one of {list(WAN_MODELS)}")
        for path in (dataset_dir, metadata_path):
            if not path.exists():
                raise FileNotFoundError(f"Not found: {path}")
        options.validate()
        spec = WAN_MODELS[model_size]
        script = find_train_script(diffsynth_root)
    except (ValueError, FileNotFoundError) as e:
        yield events.error(str(e))
        return

    env = {**os.environ, "DIFFSYNTH_MODEL_BASE_PATH": str(CACHE_DIR)}
    runs = spec.train_runs
    for number, run in enumerate(runs, 1):
        run_dir = output_dir / run.name if len(runs) > 1 else output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(script, spec, run, dataset_dir, metadata_path, run_dir, options)
        yield events.info(
            f"Run {number}/{len(runs)}: {spec.model_id} ({run.name})\nCommand: {shlex.join(cmd)}",
            command=cmd,
        )
        try:
            for line in _stream(cmd, env):
                yield events.progress(number, len(runs), line)
        except RuntimeError as e:
            yield events.error(f"Training run {run.name} failed: {e}")
            return

    yield events.complete(f"LoRA saved to {output_dir}", output_path=str(output_dir))
