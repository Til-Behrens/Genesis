"""Paths, model table and defaults for Genesis.

Every path derives from `GENESIS_HOME` (default: the current working directory), so an
editable install run from the repository root keeps all runtime data inside the checkout.
"""
import os
from dataclasses import dataclass
from pathlib import Path

HOME_DIR = Path(os.getenv("GENESIS_HOME", Path.cwd())).resolve()

CACHE_DIR = Path(os.getenv("GENESIS_CACHE_DIR", HOME_DIR / "cache" / "models"))
CAPTION_CACHE_DIR = CACHE_DIR / "captions"

DATA_DIR = HOME_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
CLIPS_DIR = DATA_DIR / "clips"
METADATA_FILE = DATA_DIR / "metadata.jsonl"
DIFFSYNTH_METADATA_FILE = DATA_DIR / "metadata_diffsynth.jsonl"

OUTPUTS_DIR = HOME_DIR / "outputs"
LORA_OUTPUT_DIR = HOME_DIR / "models" / "lora"

DIFFSYNTH_ROOT = os.getenv("GENESIS_DIFFSYNTH_ROOT", "")


@dataclass(frozen=True)
class TrainRun:
    """One DiffSynth training job; mixture-of-experts models need one per expert."""

    name: str
    dit_pattern: str
    min_timestep_boundary: float = 0.0
    max_timestep_boundary: float = 1.0


@dataclass(frozen=True)
class WanModelSpec:
    """Files and defaults for one Wan checkpoint, shared by generation and training.

    Generation loads text encoder and VAE as safetensors from `SHARED_COMPONENTS_ID`
    (required for disk offload); training uses the original files of the model repo,
    as in DiffSynth's training recipes.
    """

    model_id: str
    dit_patterns: tuple[str, ...]
    vae_file: str
    train_vae_file: str
    height: int
    width: int
    num_inference_steps: int
    fps: int
    train_runs: tuple[TrainRun, ...]


SHARED_COMPONENTS_ID = os.getenv(
    "GENESIS_DIFFSYNTH_SHARED_MODEL_ID",
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
)
TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"

_DIT = "diffusion_pytorch_model*.safetensors"

# keys are the labels shown in the ui
WAN_MODELS = {
    "5B": WanModelSpec(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        dit_patterns=(_DIT,),
        vae_file="Wan2.2_VAE.safetensors",
        train_vae_file="Wan2.2_VAE.pth",
        height=704, width=1280, num_inference_steps=50, fps=24,
        train_runs=(TrainRun("dit", _DIT),),
    ),
    "14B": WanModelSpec(
        model_id="Wan-AI/Wan2.2-T2V-A14B",
        dit_patterns=(f"high_noise_model/{_DIT}", f"low_noise_model/{_DIT}"),
        vae_file="Wan2.1_VAE.safetensors",
        train_vae_file="Wan2.1_VAE.pth",
        height=720, width=1280, num_inference_steps=20, fps=16,
        # boundaries from diffsynth's a14b lora recipe
        train_runs=(
            TrainRun("high_noise", f"high_noise_model/{_DIT}", 0.0, 0.417),
            TrainRun("low_noise", f"low_noise_model/{_DIT}", 0.417, 1.0),
        ),
    ),
    "14B-2.1": WanModelSpec(
        model_id="Wan-AI/Wan2.1-T2V-14B",
        dit_patterns=(_DIT,),
        vae_file="Wan2.1_VAE.safetensors",
        train_vae_file="Wan2.1_VAE.pth",
        height=720, width=1280, num_inference_steps=20, fps=16,
        train_runs=(TrainRun("dit", _DIT),),
    ),
    "1.3B": WanModelSpec(
        model_id="Wan-AI/Wan2.1-T2V-1.3B",
        dit_patterns=(_DIT,),
        vae_file="Wan2.1_VAE.safetensors",
        train_vae_file="Wan2.1_VAE.pth",
        height=480, width=832, num_inference_steps=40, fps=16,
        train_runs=(TrainRun("dit", _DIT),),
    ),
}
DEFAULT_GENERATION_MODEL = "5B"
DEFAULT_TRAINING_MODEL = "1.3B"

NEGATIVE_PROMPT = "low quality, blurry, distorted, noisy, artifacts, unreadable text, static, still image"
MAX_DURATION_SEC = 10

# "cpu" keeps onloaded weights in ram (faster), "disk" streams layers from safetensors
# files for machines with too little ram for the text encoder (about 11 GB)
ONLOAD_DEVICE = os.getenv("GENESIS_ONLOAD_DEVICE", "auto")
DISK_ONLOAD_BELOW_RAM_GB = 32
VRAM_HEADROOM_GB = 2.0

# clips
CLIP_LENGTH_SEC = 8
CLIP_OVERLAP_SEC = 2
CLIP_FPS = 24
CLIP_WIDTH = 1280
CLIP_HEIGHT = 704  # multiple of 16 for the vae

# captioning; qwen3-vl takes all keyframes plus the transcript, the others caption single frames
CAPTION_BACKEND = os.getenv("GENESIS_CAPTION_BACKEND", "qwen3-vl")
CAPTION_MODELS = {
    "qwen3-vl": "Qwen/Qwen3-VL-2B-Instruct",
    "vit-gpt2": "nlpconnect/vit-gpt2-image-captioning",
    "blip": "Salesforce/blip-image-captioning-base",
    "blip2": "Salesforce/blip2-opt-2.7b",
}
CAPTION_MAX_FRAMES = 6
CAPTION_MAX_NEW_TOKENS = 256
WHISPER_MODEL_ID = "openai/whisper-base"


def ensure_directories() -> None:
    """Create the data, output and cache directories."""
    for path in (RAW_VIDEOS_DIR, CLIPS_DIR, OUTPUTS_DIR, LORA_OUTPUT_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
