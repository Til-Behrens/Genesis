from diffusers import DiffusionPipeline, PipelineQuantizationConfig
from transformers import BitsAndBytesConfig
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

        self.model_id = model_map.get(model_size, model_map["5B"])
        print(f"Genesis: Lade Modell -> {self.model_id}")

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

        logger.info("Modell geladen. GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                    torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0)

    def generate(self, prompt: str, duration_sec: int = 8, output_path: str | None = None):
        frames = max(24, int(duration_sec * 24))

        steps = 50 if "14B" in self.model_id else 20

        video = self.pipe(
            prompt,
            num_inference_steps=steps,
            # model specific resolution
            height=512 if "1.7b" in self.model_id else 480 if "1.3b" in self.model_id else 720,
            width=512 if "1.7b" in self.model_id else 832 if "1.3b" in self.model_id else 1280,
            num_frames=frames,
            guidance_scale=6.0,
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