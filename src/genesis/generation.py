"""Wan text-to-video generation via DiffSynth, one model in memory at a time."""
import gc
import logging
import os
import threading
from pathlib import Path

import torch
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video

from genesis.config import (
    CACHE_DIR,
    DISK_ONLOAD_BELOW_RAM_GB,
    NEGATIVE_PROMPT,
    ONLOAD_DEVICE,
    OUTPUTS_DIR,
    SHARED_COMPONENTS_ID,
    TOKENIZER_MODEL_ID,
    VRAM_HEADROOM_GB,
    WAN_MODELS,
)
from genesis.device import get_device_info, total_ram_gb

logger = logging.getLogger(__name__)


def _log_vram(stage: str) -> None:
    if os.getenv("GENESIS_DEBUG_VRAM") != "1" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    logger.info(
        "VRAM %s | allocated=%.2f GB | reserved=%.2f GB | max=%.2f GB",
        stage,
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.memory_reserved() / 1e9,
        torch.cuda.max_memory_allocated() / 1e9,
    )


def resolve_onload_device(setting: str = ONLOAD_DEVICE) -> str:
    """Map `auto` to `disk` on machines with less than `DISK_ONLOAD_BELOW_RAM_GB` of ram."""
    if setting in ("cpu", "disk"):
        return setting
    ram = total_ram_gb()
    return "disk" if 0 < ram < DISK_ONLOAD_BELOW_RAM_GB else "cpu"


def frame_count(duration_sec: float, fps: int) -> int:
    """Largest valid Wan frame count (4n+1) up to `duration_sec * fps`, with a floor of about one second."""
    requested = max(fps, int(duration_sec * fps))
    return (requested - 1) // 4 * 4 + 1


class GenesisPipeline:
    """DiffSynth Wan pipeline for one model size.

    Weights stay on disk until needed and are streamed to the GPU within a VRAM budget
    of total VRAM minus `VRAM_HEADROOM_GB`.
    """

    def __init__(self, model_size: str):
        if model_size not in WAN_MODELS:
            raise ValueError(f"Unknown model size {model_size!r}, expected one of {list(WAN_MODELS)}")
        self.model_size = model_size
        self.spec = WAN_MODELS[model_size]

        dtype = torch.bfloat16
        onload = resolve_onload_device()
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk" if onload == "disk" else dtype,
            "onload_device": onload,
            "preparing_dtype": dtype,
            "preparing_device": "cuda",
            "computation_dtype": dtype,
            "computation_device": "cuda",
        }

        def config(model_id: str, pattern: str, **kwargs) -> ModelConfig:
            return ModelConfig(
                model_id=model_id, origin_file_pattern=pattern, local_model_path=str(CACHE_DIR), **kwargs
            )

        model_configs = [config(self.spec.model_id, pattern, **vram_config) for pattern in self.spec.dit_patterns]
        model_configs += [
            config(SHARED_COMPONENTS_ID, "models_t5_umt5-xxl-enc-bf16*.safetensors", **vram_config),
            config(SHARED_COMPONENTS_ID, self.spec.vae_file, **vram_config),
        ]

        total_vram_gb = torch.cuda.mem_get_info()[1] / 1024**3
        vram_limit_gb = max(total_vram_gb - VRAM_HEADROOM_GB, 1.0)
        logger.info(
            "Loading %s (onload: %s, VRAM limit %.1f of %.1f GB)",
            self.spec.model_id, onload, vram_limit_gb, total_vram_gb,
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=dtype,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=config(TOKENIZER_MODEL_ID, "google/umt5-xxl/"),
            vram_limit=vram_limit_gb,
        )
        logger.info("Model loaded on %s", get_device_info().name)

    def generate(
        self,
        prompt: str,
        duration_sec: float = 5,
        output_path: str | Path | None = None,
        seed: int | None = None,
    ) -> Path:
        """Generate a video and save it as mp4.

        Args:
            prompt: Text prompt.
            duration_sec: Target length, rounded down to a valid frame count.
            output_path: Target file; derived from the prompt if omitted.
            seed: Random seed; drawn randomly if omitted.

        Returns:
            Path of the saved video.
        """
        fps = self.spec.fps
        frames = frame_count(duration_sec, fps)
        seed = int(seed) if seed is not None else torch.seed() % 2**32

        if output_path is None:
            safe_name = "".join(c if c.isalnum() else "_" for c in prompt[:30])
            output_path = OUTPUTS_DIR / f"genesis_{self.model_size}_{seed}_{safe_name}.mp4"
        output_path = Path(output_path)

        if os.getenv("GENESIS_DEBUG_VRAM") == "1" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info("Generating %d frames with seed %d", frames, seed)
        _log_vram("before_generate")

        try:
            with torch.inference_mode():
                video = self.pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=self.spec.height,
                    width=self.spec.width,
                    num_frames=frames,
                    num_inference_steps=self.spec.num_inference_steps,
                    seed=seed,
                    tiled=False,
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_video(video, str(output_path), fps=fps, quality=5)
            logger.info("Video saved to %s", output_path)
            return output_path
        finally:
            video = None
            torch.cuda.empty_cache()
            _log_vram("after_generate")

    def unload(self) -> None:
        """Release the pipeline and free GPU memory."""
        logger.info("Unloading %s", self.spec.model_id)
        if getattr(self, "pipe", None) is not None:
            # back to the offload state; pipe.to("cpu") would pull every weight into ram
            self.pipe.load_models_to_device([])
            self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_vram("after_unload")


_instance: GenesisPipeline | None = None
_instance_lock = threading.Lock()


def get_pipeline(model_size: str) -> GenesisPipeline:
    """Return the loaded pipeline, swapping models if a different size is requested."""
    global _instance
    with _instance_lock:
        if _instance is not None and _instance.model_size != model_size:
            _instance.unload()
            _instance = None
        if _instance is None:
            _instance = GenesisPipeline(model_size)
        return _instance


def unload_pipeline() -> None:
    """Free the loaded pipeline, if any; called before captioning and training."""
    global _instance
    with _instance_lock:
        if _instance is not None:
            _instance.unload()
            _instance = None
