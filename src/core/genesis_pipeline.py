import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from pathlib import Path
import logging
import os
from src.core.config import WAN_MODEL_MAP, get_device_info

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
    def __init__(self, model_size: str = "5B"):
        self.model_size = model_size
        self.model_id = WAN_MODEL_MAP.get(model_size, WAN_MODEL_MAP["5B"])
        self.shared_components_id = os.getenv(
            "GENESIS_DIFFSYNTH_SHARED_MODEL_ID",
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        )
        self.torch_dtype = torch.bfloat16
        logger.info(f"Loading Model -> {self.model_id}")
        logger.info("Shared encoder/VAE source: %s", self.shared_components_id)
        logger.info("Using torch dtype: %s", self.torch_dtype)

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

        total_vram_gb = torch.cuda.mem_get_info()[1] / (1024 ** 3)
        vram_limit_gb = max(total_vram_gb - 2, 1.0)
        logger.info("Total VRAM: %.1f GB | Setting pipeline VRAM limit to %.1f GB", total_vram_gb, vram_limit_gb)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self.torch_dtype,
            device="cuda",
            model_configs=[
                ModelConfig(model_id=self.model_id,
                            origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
                ModelConfig(
                    model_id=self.shared_components_id,
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16*.safetensors",
                    **vram_config,
                ),
                ModelConfig(
                    model_id=self.shared_components_id,
                    origin_file_pattern="Wan2.1_VAE*.safetensors",
                    **vram_config,
                ),
            ],
            tokenizer_config=ModelConfig(model_id=self.model_id, origin_file_pattern="google/umt5-xxl/"),
            vram_limit=vram_limit_gb,
        )

        _, device_name, vram_gb = get_device_info()
        logger.info("Model loaded. GPU: %s | VRAM: %.1f GB", device_name, vram_gb)

    def _get_model_params(self) -> dict:
        """Get model-specific parameters for generation."""
        params = {
            "height": 720,
            "width": 1280,
            "num_inference_steps": 20,
        }

        match self.model_size:
            case "5B":
                params["height"] = 704
                params["num_inference_steps"] = 50
            case "1.3B":
                params["height"] = 480
                params["width"] = 832
                params["num_inference_steps"] = 40

        return params

    def generate(self, prompt: str, duration_sec: int = 8, output_path: str | None = None, seed: int | None = None):
        fps = 16
        requested_frames = max(fps, int(duration_sec * fps))
        # Wan works best with frame counts following 4n+1.
        frames = ((requested_frames - 1) // 4) * 4 + 1

        params = self._get_model_params()

        negative_prompt = "low quality, blurry, distorted, noisy, artifacts, unreadable text, static, still image"

        if seed is None:
            seed = torch.seed()

        if os.getenv("GENESIS_DEBUG_VRAM", "0") == "1" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info(f"Generating {frames} frames with seed {seed}...")
        _log_cuda_memory("before_generate")

        try:

            with torch.inference_mode():
                video = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=params["height"],
                    width=params["width"],
                    num_frames=frames,
                    num_inference_steps=params["num_inference_steps"],
                    seed=seed,
                    tiled=False,
                )
            _log_cuda_memory("after_generate")

            if not output_path:
                safe_name = "".join(c if c.isalnum() else "_" for c in prompt[:30])
                output_path = f"outputs/genesis_{duration_sec}s_{safe_name}.mp4"

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
