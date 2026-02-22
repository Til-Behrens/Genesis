from diffusers import AutoencoderKLWan, WanPipeline
import torch
from pathlib import Path
import logging
from src.core.config import CACHE_DIR, WAN_MODEL_MAP, get_optimal_device, get_device_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Genesis")

# Suppress verbose logging from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class GenesisPipeline:
    def __init__(self, model_size: str = "5B"):
        self.model_size = model_size
        self.model_id = WAN_MODEL_MAP.get(model_size, WAN_MODEL_MAP["5B"])
        logger.info(f"Loading Model -> {self.model_id}")

        device = get_optimal_device()
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        if device == "cpu":
            logger.warning("⚠️  GPU not available, using CPU (will be slow)")

        # Load VAE separately with float32 for precision
        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=False,
            cache_dir=CACHE_DIR,
        )

        # Load pipeline with the VAE
        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=dtype,
            local_files_only=False,
            cache_dir=CACHE_DIR,
        )
        self.pipe.to(device)

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
            "guidance_scale": 4.0,
            "num_inference_steps": 40,
        }

        match self.model_size:
            case "14b":
                params["guidance_scale_2"] = 3.0
            case "5B":
                params["guidance_scale"] = 5.0
                params["num_inference_steps"] = 50
            case "1.3b":
                params["height"] = 480
                params["width"] = 832
                params["guidance_scale"] = 5.0

        return params

    def generate(self, prompt: str, duration_sec: int = 8, output_path: str | None = None):
        fps = 16
        frames = max(fps, int(duration_sec * fps))

        params = self._get_model_params()

        negative_prompt = "blurry, low quality, distorted text, unreadable text"

        # Build the pipeline call arguments
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": params["height"],
            "width": params["width"],
            "num_frames": frames,
            "guidance_scale": params["guidance_scale"],
            "num_inference_steps": params["num_inference_steps"],
        }

        # Only add guidance_scale_2 for 14B model
        if "guidance_scale_2" in params:
            pipe_kwargs["guidance_scale_2"] = params["guidance_scale_2"]

        video = self.pipe(**pipe_kwargs).frames[0]

        if not output_path:
            safe_name = "".join(c if c.isalnum() else "_" for c in prompt[:30])
            output_path = f"outputs/genesis_{duration_sec}s_{safe_name}.mp4"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from diffusers.utils import export_to_video
        export_to_video(video, output_path, fps=fps)
        logger.info(f"Video saved → {output_path}")
        return output_path


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
        if _genesis_instance is None or _genesis_instance.model_size != model_size:
            logger.info(f"Initializing Genesis Pipeline ({model_size})...")
            _genesis_instance = GenesisPipeline(model_size)
        return _genesis_instance

