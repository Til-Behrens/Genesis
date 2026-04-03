"""
Centralized configuration management for Genesis.
"""
import os
from pathlib import Path
import yaml
import logging
from typing import Tuple

logger = logging.getLogger("Config")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Model cache directory (can be overridden by env var)
CACHE_DIR = os.getenv(
    "GENESIS_CACHE_DIR",
    str(PROJECT_ROOT / "cache" / "models")
)

# Dataset paths
DATASET_ROOT = PROJECT_ROOT / "src" / "finetuning" / "dataset"
RAW_VIDEOS_DIR = DATASET_ROOT / "raw_videos"
CUT_VIDEOS_DIR = DATASET_ROOT / "cut_videos"
METADATA_FILE = DATASET_ROOT / "metadata.jsonl"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
LORA_CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "lora_checkpoints"

# Model IDs (Wan models - Diffusers versions are the official supported format)
WAN_MODEL_MAP = {
    "14B": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "14B-2.1": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "5B": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "1.3B": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}

# Model IDs for DiffSynth training backend
WAN_TRAINING_MODEL_MAP = {
    "1.3B": "Wan-AI/Wan2.1-T2V-1.3B",
    "14B-2.1": "Wan-AI/Wan2.1-T2V-14B",
}

# Optional DiffSynth-Studio root. If empty, training auto-detects from installed package.
DIFFSYNTH_ROOT = os.getenv("GENESIS_DIFFSYNTH_ROOT", "")

# Captioning configuration
# Backends:
# - 'vit-gpt2': Lightweight, low VRAM, single-frame only
# - 'blip': Better accuracy, single-frame only
# - 'blip2': Stronger model, higher VRAM, single-frame only
# - 'qwen3-vl': Advanced multi-frame understanding, best quality for sequential video analysis
CAPTION_BACKEND = os.getenv("GENESIS_CAPTION_BACKEND", "qwen3-vl")

# Mapping of backend keys to Hugging Face model IDs
CAPTION_MODELS = {
    "vit-gpt2": "nlpconnect/vit-gpt2-image-captioning",
    "blip": "Salesforce/blip-image-captioning-base",
    "blip2": "Salesforce/blip2-opt-2.7b",
    "qwen3-vl": "Qwen/Qwen3-VL-2B-Instruct",
}

# Optional summarizer (disabled by default). If enabled, this will be run on the
# per-frame captions to produce a single concise German prompt.
CAPTION_USE_SUMMARIZER = False
CAPTION_SUMMARIZER_ID = "google/mt5-small"

# How many frames to sample per clip when generating captions
CAPTION_MAX_FRAMES = 6

# Quantization / offload hooks (reserved for advanced users; disabled by default)
CAPTION_QUANTIZE = False
CAPTION_OFFLOAD = False

# Training defaults
DEFAULT_TRAINING_CONFIG = {
    # Note: epochs and learning_rate are passed explicitly from UI.
    "dataset_repeat": 100,
    "dataset_num_workers": 0,
    "data_file_keys": "file_name",
    "lora_base_model": "dit",
    "lora_target_modules": "q,k,v,o,ffn.0,ffn.2",
    "lora_rank": 32,
    "remove_prefix_in_ckpt": "pipe.dit.",
    "gradient_accumulation_steps": 1,
    "save_steps": 100,
    "extra_inputs": None,
    "find_unused_parameters": False,
    "accelerate_config_file": None,
    "initialize_model_on_cpu": False,
}

# Video processing defaults
VIDEO_CONFIG = {
    "clip_length": 8,  # seconds
    "overlap": 2,  # seconds
    "target_fps": 24,
    "target_width": 1280,
    "target_height": 704,  # Must be divisible by 32 for VAE (patch_size=2, scale_factor=16)
}


def get_optimal_device() -> str:
    """Get the best available compute device.

    Returns:
        Device string: "cuda" for both NVIDIA and AMD GPUs, "cpu" otherwise.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def get_device_info() -> Tuple[str, str, float]:
    """Get detailed information about the compute device.

    Returns:
        Tuple of (device_type, device_name, vram_gb):
        - device_type: "cuda" or "cpu"
        - device_name: GPU name with platform (e.g., "AMD Radeon RX 9060 XT (ROCm)")
        - vram_gb: Total VRAM in GB (0.0 for CPU)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return ("cpu", "CPU", 0.0)

        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Detect if we're running on ROCm (AMD) or CUDA (NVIDIA)
        platform = "CUDA"
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            platform = "ROCm"

        # Format device name with platform info
        if platform == "ROCm":
            full_name = f"{device_name} ({platform})"
        else:
            full_name = f"{device_name} (CUDA)"

        return ("cuda", full_name, vram_gb)

    except ImportError:
        return ("cpu", "CPU", 0.0)


def ensure_directories():
    """Create all necessary directories."""
    dirs = [
        RAW_VIDEOS_DIR,
        CUT_VIDEOS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        LORA_CHECKPOINTS_DIR,
        Path(CACHE_DIR),  # Ensure cache directory exists
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directories initialized")


def load_config_yaml() -> dict:
    """Load config from models.yaml if it exists."""
    config_file = PROJECT_ROOT / "config" / "models.yaml"
    if config_file.exists() and config_file.stat().st_size > 0:
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    return {}


# Initialize on import
ensure_directories()
