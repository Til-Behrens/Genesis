"""DiffSynth-based Wan text-to-video generation with lazy, singleton model loading."""
import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from pathlib import Path
import logging
import os
from src.core.config import (
    CACHE_DIR, OUTPUTS_DIR, SHARED_COMPONENTS_ID, TOKENIZER_MODEL_ID, WAN_MODELS, get_device_info,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Genesis")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def _log_cuda_memory(stage: str) -> None:
    if os.getenv("GENESIS_DEBUG_VRAM", "0") != "1":
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_alloc = torch.cuda.max_memory_allocated() / 1e9
    logger.info("VRAM %s | allocated=%.2f GB | reserved=%.2f GB | max=%.2f GB", stage, allocated, reserved, max_alloc)

class GenesisPipeline:
    """DiffSynth Wan pipeline for one model size, with weights offloaded to disk and cpu."""

    def __init__(self, model_size: str = "5B"):
        if model_size not in WAN_MODELS:
            raise ValueError(f"Unknown model size {model_size!r}, expected one of {list(WAN_MODELS)}")
        self.model_size = model_size
        self.spec = WAN_MODELS[model_size]
        self.model_id = self.spec.model_id
        self.torch_dtype = torch.bfloat16
        logger.info(f"Loading Model -> {self.model_id}")
        logger.info("Shared encoder/VAE source: %s", SHARED_COMPONENTS_ID)

        _log_cuda_memory("before_vae_load")

        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": self.torch_dtype,
            "onload_device": "cpu",
            "preparing_dtype": self.torch_dtype,
            "preparing_device": "cuda",
            "computation_dtype": self.torch_dtype,
            "computation_device": "cuda",
        }

        def config(model_id: str, pattern: str, **kwargs) -> ModelConfig:
            return ModelConfig(model_id=model_id, origin_file_pattern=pattern, local_model_path=CACHE_DIR, **kwargs)

        model_configs = [config(self.model_id, pattern, **vram_config) for pattern in self.spec.dit_patterns]
        model_configs += [
            config(SHARED_COMPONENTS_ID, "models_t5_umt5-xxl-enc-bf16*.safetensors", **vram_config),
            config(SHARED_COMPONENTS_ID, self.spec.vae_file, **vram_config),
        ]

        # leave headroom for activations outside the offload budget
        total_vram_gb = torch.cuda.mem_get_info()[1] / (1024 ** 3)
        vram_limit_gb = max(total_vram_gb - 2, 1.0)
        logger.info("Total VRAM: %.1f GB | Setting pipeline VRAM limit to %.1f GB", total_vram_gb, vram_limit_gb)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self.torch_dtype,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=config(TOKENIZER_MODEL_ID, "google/umt5-xxl/"),
            vram_limit=vram_limit_gb,
        )

        _, device_name, vram_gb = get_device_info()
        logger.info("Model loaded. GPU: %s | VRAM: %.1f GB", device_name, vram_gb)

    def generate(self, prompt: str, duration_sec: int = 5, output_path: str | None = None, seed: int | None = None) -> str:
        """Generate a video and save it as mp4.

        Args:
            prompt: Text prompt.
            duration_sec: Target length; rounded down to a valid Wan frame count.
            output_path: Target file, derived from the prompt if omitted.
            seed: Random seed, drawn randomly if omitted.

        Returns:
            Path of the saved video.
        """
        fps = self.spec.fps
        requested_frames = max(fps, int(duration_sec * fps))
        # wan needs 4n+1 frames
        frames = ((requested_frames - 1) // 4) * 4 + 1

        negative_prompt = "low quality, blurry, distorted, noisy, artifacts, unreadable text, static, still image"

        seed = int(seed) if seed is not None else torch.seed() % 2**32

        if os.getenv("GENESIS_DEBUG_VRAM", "0") == "1" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info(f"Generating {frames} frames with seed {seed}...")
        _log_cuda_memory("before_generate")

        try:

            with torch.inference_mode():
                video = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=self.spec.height,
                    width=self.spec.width,
                    num_frames=frames,
                    num_inference_steps=self.spec.num_inference_steps,
                    seed=seed,
                    tiled=False,
                )
            _log_cuda_memory("after_generate")

            if not output_path:
                safe_name = "".join(c if c.isalnum() else "_" for c in prompt[:30])
                output_path = str(OUTPUTS_DIR / f"genesis_{duration_sec}s_{safe_name}.mp4")

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            save_video(video, output_path, fps=fps, quality=5)
            _log_cuda_memory("after_export")
            logger.info(f"Video saved → {output_path}")
            return output_path
        finally:
            # Clean up intermediate tensors
            if 'video' in locals():
                del video
            # Clear CUDA cache to release memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _log_cuda_memory("after_cleanup")

    def unload(self):
        """Unload the pipeline and free GPU memory."""
        logger.info(f"Unloading pipeline ({self.model_size})...")
        _log_cuda_memory("before_unload")

        pipe = getattr(self, "pipe", None)
        if pipe is not None:
            # Move pipeline to CPU to free GPU memory
            if hasattr(pipe, 'to'):
                pipe.to('cpu')

            self.pipe = None

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        _log_cuda_memory("after_unload")
        logger.info("Pipeline unloaded and memory freed")


# Singleton instance (lazy loaded)
_genesis_instance: GenesisPipeline | None = None
_genesis_lock = None

def get_genesis_pipeline(model_size: str = "5B") -> GenesisPipeline:
    """Get or create the Genesis pipeline instance (singleton)."""
    global _genesis_instance, _genesis_lock

    if _genesis_lock is None:
        import threading
        _genesis_lock = threading.Lock()

    with _genesis_lock:
        # If switching models, unload the old pipeline first
        if _genesis_instance is not None and _genesis_instance.model_size != model_size:
            logger.info(f"Switching from {_genesis_instance.model_size} to {model_size}")
            pipeline = _genesis_instance
            pipeline.unload()
            _genesis_instance = None

        # Create new pipeline if needed
        if _genesis_instance is None:
            logger.info(f"Initializing Genesis Pipeline ({model_size})...")
            _genesis_instance = GenesisPipeline(model_size)

        return _genesis_instance

def unload_genesis_pipeline():
    """Explicitly unload the current pipeline and free memory."""
    global _genesis_instance, _genesis_lock

    if _genesis_lock is None:
        import threading
        _genesis_lock = threading.Lock()

    with _genesis_lock:
        if _genesis_instance is not None:
            pipeline = _genesis_instance
            pipeline.unload()
            _genesis_instance = None
            logger.info("Global pipeline instance cleared")
