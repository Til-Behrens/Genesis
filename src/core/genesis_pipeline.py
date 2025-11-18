from diffusers import DiffusionPipeline
import torch
from pathlib import Path

class GenesisPipeline:
    def __init__(self, model_size: str = "1.7b"):   # 1.7b ist jetzt Default → läuft sofort
        model_map = {
            "14b":  "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "5b":   "THUDM/CogVideoX-5b",
            "2b":   "THUDM/CogVideoX-2b",
            "1.7b": "damo-vilab/text-to-video-ms-1.7b",   # ← das alte, stabile, superschnelle
            "1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",            # alternativ
        }

        self.model_id = model_map.get(model_size.lower(), model_map["1.7b"])
        print(f"Genesis: Lade Modell → {self.model_id}")

        variant = None if "Wan" in self.model_id else "fp16"

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            variant=variant,
            safety_checker=None,        # spart RAM + keine nervigen Warnings
            requires_safety_checker=False
        )

        # WICHTIG: für das alte 1.7b-Modell
        if "text-to-video-ms-1.7b" in self.model_id or "1.3B" in self.model_id:
            self.pipe.scheduler = self.pipe.scheduler.__class__.from_config(self.pipe.scheduler.config)

        self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        print("Genesis ist bereit – gib mir einen Prompt!")

    def generate(self, prompt: str, duration_sec: int = 8, output_path: str | None = None):
        frames = max(25, int(duration_sec * 24))

        # Unterschiedliche Schritte je nach Modell
        steps = 50 if "14B" in self.model_id else 28

        video = self.pipe(
            prompt,
            num_inference_steps=steps,
            height=512 if "1.7b" in self.model_id else 480 if "1.3b" in self.model_id else 720,   # 1.7b unterstützt nur 512
            width=512 if "1.7b" in self.model_id else 832 if "1.3b" in self.model_id else 1280,
            num_frames=frames,
            guidance_scale=9.0,
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
genesis = GenesisPipeline(model_size="1.3b")  # change model size here
# genesis = GenesisPipeline(model_size="14b")