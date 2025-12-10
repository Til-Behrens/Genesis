from diffusers import AutoencoderKLWan, WanPipeline
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Genesis")

class GenesisPipeline:
    def __init__(self, model_size: str = "5B"):
        model_map = {
            "14b":  "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "14B": "Wan-AI/Wan2.2-T2V-A14B",
            "5B": "Wan-AI/Wan2.2-TI2V-5B",
        }

        self.model_size = model_size
        self.model_id = model_map.get(model_size, model_map["5B"])
        logger.info(f"Lade Modell -> {self.model_id}")

        dtype = torch.bfloat16
        device = "cuda"
        vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=False,
            resume_download=True,
            cache_dir="/content/drive/MyDrive/Genesis/models",
        )

        self.pipe = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            vae=vae,
            torch_dtype=dtype,
            local_files_only=False,
            resume_download=True,
            cache_dir="/content/drive/MyDrive/Genesis/models",
        )
        self.pipe.to(device)

        logger.info("Modell geladen. GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                    torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0)

    def generate(self, prompt: str, duration_sec: int = 8, output_path: str | None = None):
        frames = max(24, int(duration_sec * 24))

        height = 720
        width = 1280
        guidance_scale = 4.0,
        guidance_scale_2 = 3.0,
        num_inference_steps = 40,

        match self.model_size:
            case "14b":
                guidance_scale = 5.0,
            case "5B":
                height = 704
                guidance_scale = 5.0,
                num_inference_steps = 50,
            case "1.3b":
                height = 480,
                width = 832,
                guidance_scale = 5.0

        negative_prompt = "blurry, low quality, distorted text, unreadable text"

        video = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=frames,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            num_inference_steps=num_inference_steps,
        ).frames[0]

        if not output_path:
            safe_name = "".join(c if c.isalnum() else "_" for c in prompt[:30])
            output_path = f"outputs/genesis_{duration_sec}s_{safe_name}.mp4"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        from diffusers.utils import export_to_video
        export_to_video(video, output_path, fps=24)
        print(f"Video gespeichert → {output_path}")
        return output_path


# global instance
# genesis = GenesisPipeline(model_size="1.3b")  # change model size here
genesis = GenesisPipeline(model_size="5B")