from diffusers import AutoencoderKLWan, WanPipeline
import torch
from pathlib import Path
import logging
import os
from src.core.config import CACHE_DIR, WAN_MODEL_MAP, get_optimal_device, get_device_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Genesis")

# Suppress verbose logging from dependencies
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
        logger.info(f"Loading Model -> {self.model_id}")

        device = get_optimal_device()
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        if device == "cpu":
            logger.warning("⚠️  GPU not available, using CPU (will be slow)")

        _log_cuda_memory("before_vae_load")
        # Load VAE separately with float32 for precision
        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=False,
            cache_dir=CACHE_DIR,
        )
        vae.eval()  # Set VAE to evaluation mode for better quality
        _log_cuda_memory("after_vae_load")

        # Load pipeline with the VAE
        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=dtype,
            local_files_only=False,
            cache_dir=CACHE_DIR,
        )
        _log_cuda_memory("after_pipe_load")
        self.pipe.to(device)
        self.device = device
        self.dtype = dtype
        _log_cuda_memory("after_pipe_to_device")

        device_type, device_name, vram_gb = get_device_info()
        if device_type == "cuda":
            logger.info("✓ Model loaded. GPU: %s | VRAM: %.1f GB | Precision: %s",
                        device_name,
                        vram_gb,
                        "bf16" if dtype == torch.bfloat16 else "fp16")
        else:
            logger.info("✓ Model loaded on CPU")

    def _get_model_params(self) -> dict:
        """Get model-specific parameters for generation."""
        params = {
            "height": 704,
            "width": 1280,
            "guidance_scale": 5.0,
            "num_inference_steps": 50,
        }

        match self.model_size:
            case "14B":
                params["guidance_scale"] = 5.0
                params["guidance_scale_2"] = 3.0
                params["num_inference_steps"] = 50
            case "14B-2.1":
                params["guidance_scale"] = 5.0
                params["guidance_scale_2"] = 3.0
                params["num_inference_steps"] = 50
            case "5B":
                params["guidance_scale"] = 5.0
                params["num_inference_steps"] = 50
            case "1.3B":
                params["height"] = 480
                params["width"] = 832
                params["guidance_scale"] = 5.0
                params["num_inference_steps"] = 40

        return params

    def generate(self, prompt: str, duration_sec: int = 8, output_path: str | None = None, seed: int | None = None):
        fps = 16
        frames = max(fps, int(duration_sec * fps))

        params = self._get_model_params()

        # Better negative prompt for quality
        negative_prompt = "low quality, blurry, distorted, noisy, artifacts, unreadable text, static, still image"

        # Create generator for reproducibility and quality
        if seed is None:
            seed = torch.seed()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Build the pipeline call arguments
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": params["height"],
            "width": params["width"],
            "num_frames": frames,
            "guidance_scale": params["guidance_scale"],
            "num_inference_steps": params["num_inference_steps"],
            "generator": generator,
            "output_type": "np",  # Use numpy output for better quality
        }

        # Only add guidance_scale_2 for 14B model
        if "guidance_scale_2" in params:
            pipe_kwargs["guidance_scale_2"] = params["guidance_scale_2"]

        if os.getenv("GENESIS_DEBUG_VRAM", "0") == "1" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info(f"Generating {frames} frames with seed {seed}...")
        _log_cuda_memory("before_generate")

        try:
            with torch.inference_mode():
                video = self.pipe(**pipe_kwargs).frames[0]
            _log_cuda_memory("after_generate")

            if not output_path:
                safe_name = "".join(c if c.isalnum() else "_" for c in prompt[:30])
                output_path = f"outputs/genesis_{duration_sec}s_{safe_name}.mp4"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            from diffusers.utils import export_to_video
            export_to_video(video, output_path, fps=fps)
            _log_cuda_memory("after_export")
            logger.info(f"Video saved → {output_path}")
            return output_path
        finally:
            # Clean up intermediate tensors
            del generator
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

        if hasattr(self, 'pipe') and self.pipe is not None:
            # Move pipeline to CPU to free GPU memory
            if hasattr(self.pipe, 'to'):
                self.pipe.to('cpu')

            # Delete pipeline components
            del self.pipe
            self.pipe = None

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        _log_cuda_memory("after_unload")
        logger.info("✓ Pipeline unloaded and memory freed")


# Singleton instance (lazy loaded)
_genesis_instance = None
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
            _genesis_instance.unload()
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
            _genesis_instance.unload()
            _genesis_instance = None
            logger.info("✓ Global pipeline instance cleared")
