"""DiffSynth-based Wan LoRA fine-tuning backend."""

import os
import pathlib
import logging
import shlex
import subprocess
from typing import Generator, Dict, Any

from src.core.config import CACHE_DIR, WAN_MODELS, TrainRun, WanModelSpec

logger = logging.getLogger("LoRA")


def build_origin_paths(spec: WanModelSpec, run: TrainRun) -> str:
    """Build DiffSynth's `--model_id_with_origin_paths` value for one training run."""
    patterns = (run.dit_pattern, "models_t5_umt5-xxl-enc-bf16.pth", spec.train_vae_file)
    return ",".join(f"{spec.model_id}:{pattern}" for pattern in patterns)


def _resolve_train_script(diffsynth_root: pathlib.Path | str | None) -> pathlib.Path:
    """Find DiffSynth's wanvideo train.py in the given root, `GENESIS_DIFFSYNTH_ROOT` or the installed package."""
    candidates: list[pathlib.Path] = []

    if diffsynth_root:
        candidates.append(pathlib.Path(diffsynth_root))

    env_root = os.getenv("GENESIS_DIFFSYNTH_ROOT")
    if env_root:
        candidates.append(pathlib.Path(env_root))

    try:
        import diffsynth  # type: ignore

        package_path = pathlib.Path(diffsynth.__file__).resolve()
        for parent in [package_path.parent, *package_path.parents]:
            candidates.append(parent)
    except Exception:
        pass

    visited = set()
    for candidate in candidates:
        if str(candidate) in visited:
            continue
        visited.add(str(candidate))

        script = candidate / "examples" / "wanvideo" / "model_training" / "train.py"
        if script.exists():
            return script

    raise FileNotFoundError(
        "Could not find DiffSynth train script. Set GENESIS_DIFFSYNTH_ROOT to your "
        "DiffSynth-Studio root (containing examples/wanvideo/model_training/train.py)."
    )


def _sanitize_num_frames(num_frames: int | None) -> int | None:
    """Reject frame counts that do not follow wan's 4n+1 rule."""
    if num_frames is None:
        return None
    if (num_frames - 1) % 4 != 0:
        raise ValueError("num_frames must follow 4n+1 for Wan training.")
    return num_frames


def _build_command(
    train_script_path: pathlib.Path,
    accelerate_config_file: pathlib.Path | str | None,
    *flag_values: Any,
    find_unused_parameters: bool,
    initialize_model_on_cpu: bool,
) -> list[str]:
    """Assemble the accelerate command; `flag_values` alternates flag and value, `None` values are skipped."""
    cmd = ["accelerate", "launch"]
    if accelerate_config_file:
        cmd += ["--config_file", str(accelerate_config_file)]
    cmd.append(str(train_script_path))
    for flag, value in zip(flag_values[::2], flag_values[1::2]):
        if value is not None and value != "":
            cmd += [flag, str(value)]
    if find_unused_parameters:
        cmd.append("--find_unused_parameters")
    if initialize_model_on_cpu:
        cmd.append("--initialize_model_on_cpu")
    return cmd


def train_lora_pipeline(
    model_size: str = "1.3B",
    dataset_base_path: pathlib.Path | str = "src/finetuning/dataset/cut_videos",
    dataset_metadata_path: pathlib.Path | str = "src/finetuning/dataset/metadata.jsonl",
    output_dir: pathlib.Path | str = "models/lora_checkpoints",
    diffsynth_root: pathlib.Path | str | None = None,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    dataset_repeat: int = 100,
    dataset_num_workers: int = 0,
    data_file_keys: str = "file_name",
    lora_base_model: str = "dit",
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2",
    lora_rank: int = 32,
    remove_prefix_in_ckpt: str = "pipe.dit.",
    gradient_accumulation_steps: int = 1,
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
    save_steps: int | None = None,
    extra_inputs: str | None = None,
    find_unused_parameters: bool = False,
    accelerate_config_file: pathlib.Path | str | None = None,
    initialize_model_on_cpu: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """Train LoRA adapters with DiffSynth's wanvideo training script.

    Runs one `accelerate launch` per entry in the model's `train_runs`; mixture-of-experts
    models (Wan2.2 A14B) get one adapter per expert in a subfolder of `output_dir`.
    Remaining arguments map one-to-one onto DiffSynth's train.py flags.

    Yields:
        Progress events: `init` per run, `training` per log line, then `complete` or `error`.
    """

    try:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_base_path = pathlib.Path(dataset_base_path)
        dataset_metadata_path = pathlib.Path(dataset_metadata_path)

        if not dataset_base_path.exists():
            raise FileNotFoundError(f"Dataset base path not found: {dataset_base_path}")
        if not dataset_metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata path not found: {dataset_metadata_path}")

        if height is not None and height % 16 != 0:
            raise ValueError("height must be a multiple of 16 for Wan training.")
        if width is not None and width % 16 != 0:
            raise ValueError("width must be a multiple of 16 for Wan training.")
        _sanitize_num_frames(num_frames)

        if model_size not in WAN_MODELS:
            raise ValueError(f"Unknown model size {model_size!r}, expected one of {list(WAN_MODELS)}")
        spec = WAN_MODELS[model_size]
        train_script_path = _resolve_train_script(diffsynth_root)

        env = {**os.environ, "DIFFSYNTH_MODEL_BASE_PATH": str(CACHE_DIR)}
        multi_run = len(spec.train_runs) > 1

        for run in spec.train_runs:
            run_output_dir = output_dir / run.name if multi_run else output_dir
            cmd = _build_command(
                train_script_path, accelerate_config_file,
                "--dataset_base_path", dataset_base_path,
                "--dataset_metadata_path", dataset_metadata_path,
                "--dataset_repeat", dataset_repeat,
                "--dataset_num_workers", dataset_num_workers,
                "--data_file_keys", data_file_keys,
                "--model_id_with_origin_paths", build_origin_paths(spec, run),
                "--learning_rate", learning_rate,
                "--num_epochs", epochs,
                "--remove_prefix_in_ckpt", remove_prefix_in_ckpt,
                "--output_path", run_output_dir,
                "--lora_base_model", lora_base_model,
                "--lora_target_modules", lora_target_modules,
                "--lora_rank", lora_rank,
                "--gradient_accumulation_steps", gradient_accumulation_steps,
                "--min_timestep_boundary", run.min_timestep_boundary,
                "--max_timestep_boundary", run.max_timestep_boundary,
                "--height", height,
                "--width", width,
                "--num_frames", num_frames,
                "--save_steps", save_steps,
                "--extra_inputs", extra_inputs,
                find_unused_parameters=find_unused_parameters,
                initialize_model_on_cpu=initialize_model_on_cpu,
            )

            yield {
                "status": "init",
                "message": f"Starting DiffSynth Wan LoRA training ({spec.model_id}, {run.name})...",
                "backend": "diffsynth",
                "command": " ".join(shlex.quote(part) for part in cmd),
                "train_script": str(train_script_path),
            }

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if line:
                    yield {"status": "training", "message": line}

            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"DiffSynth training ({run.name}) failed with exit code {return_code}")

        yield {
            "status": "complete",
            "message": "DiffSynth training completed.",
            "output_path": str(output_dir),
        }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        yield {"status": "error", "message": str(e)}
