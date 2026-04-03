"""DiffSynth-based Wan LoRA fine-tuning backend."""

import os
import pathlib
import logging
import shlex
import subprocess
from typing import Generator, Dict, Any

logger = logging.getLogger("LoRA")

MODEL_ORIGIN_MAP = {
    "Wan-AI/Wan2.1-T2V-1.3B": "diffusion_pytorch_model*.safetensors,models_t5_umt5-xxl-enc-bf16.pth,Wan2.1_VAE.pth",
    "Wan-AI/Wan2.1-T2V-14B": "diffusion_pytorch_model*.safetensors,models_t5_umt5-xxl-enc-bf16.pth,Wan2.1_VAE.pth",
}


def _format_model_id_with_origin_paths(model_id: str, model_id_with_origin_paths: str | None) -> str:
    if model_id_with_origin_paths:
        return model_id_with_origin_paths

    origin_patterns = MODEL_ORIGIN_MAP.get(model_id)
    if not origin_patterns:
        raise ValueError(
            "No default origin-file mapping found for this model. "
            "Please provide `model_id_with_origin_paths` manually."
        )

    return ",".join(f"{model_id}:{pattern}" for pattern in origin_patterns.split(","))


def _resolve_train_script(diffsynth_root: pathlib.Path | str | None) -> pathlib.Path:
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
    if num_frames is None:
        return None
    if (num_frames - 1) % 4 != 0:
        raise ValueError("num_frames must follow 4n+1 for Wan training.")
    return num_frames


def train_lora_pipeline(
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    dataset_base_path: pathlib.Path | str = "src/finetuning/dataset/cut_videos",
    dataset_metadata_path: pathlib.Path | str = "src/finetuning/dataset/metadata.jsonl",
    output_dir: pathlib.Path | str = "models/lora_checkpoints",
    diffsynth_root: pathlib.Path | str | None = None,
    model_id_with_origin_paths: str | None = None,
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
    """Train LoRA using DiffSynth's official wanvideo training script."""

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

        train_script_path = _resolve_train_script(diffsynth_root)
        model_id_with_origin_paths = _format_model_id_with_origin_paths(model_id, model_id_with_origin_paths)

        cmd: list[str] = [
            "accelerate",
            "launch",
        ]

        if accelerate_config_file:
            cmd.extend(["--config_file", str(pathlib.Path(accelerate_config_file))])

        cmd.extend([
            str(train_script_path),
            "--dataset_base_path", str(dataset_base_path),
            "--dataset_metadata_path", str(dataset_metadata_path),
            "--dataset_repeat", str(dataset_repeat),
            "--dataset_num_workers", str(dataset_num_workers),
            "--data_file_keys", data_file_keys,
            "--model_id_with_origin_paths", model_id_with_origin_paths,
            "--learning_rate", str(learning_rate),
            "--num_epochs", str(epochs),
            "--remove_prefix_in_ckpt", remove_prefix_in_ckpt,
            "--output_path", str(output_dir),
            "--lora_base_model", lora_base_model,
            "--lora_target_modules", lora_target_modules,
            "--lora_rank", str(lora_rank),
            "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        ])

        if height is not None:
            cmd.extend(["--height", str(height)])
        if width is not None:
            cmd.extend(["--width", str(width)])
        if num_frames is not None:
            cmd.extend(["--num_frames", str(num_frames)])
        if save_steps is not None:
            cmd.extend(["--save_steps", str(save_steps)])
        if extra_inputs:
            cmd.extend(["--extra_inputs", extra_inputs])
        if find_unused_parameters:
            cmd.append("--find_unused_parameters")
        if initialize_model_on_cpu:
            cmd.append("--initialize_model_on_cpu")

        pretty_cmd = " ".join(shlex.quote(part) for part in cmd)
        yield {
            "status": "init",
            "message": "Starting DiffSynth Wan LoRA training...",
            "backend": "diffsynth",
            "command": pretty_cmd,
            "train_script": str(train_script_path),
        }

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            yield {"status": "training", "message": line}

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"DiffSynth training failed with exit code {return_code}")

        yield {
            "status": "complete",
            "message": "DiffSynth training completed.",
            "output_path": str(output_dir),
        }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        yield {"status": "error", "message": str(e)}



