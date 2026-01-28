"""
Centralized configuration management for Genesis.
"""
import os
from pathlib import Path
import yaml
import logging

logger = logging.getLogger("Config")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Model cache directory (can be overridden by env var)
CACHE_DIR = os.getenv(
    "GENESIS_CACHE_DIR",
    "/content/drive/MyDrive/Genesis/models"
)

# Dataset paths
DATASET_ROOT = PROJECT_ROOT / "src" / "finetuning" / "dataset"
RAW_VIDEOS_DIR = DATASET_ROOT / "raw_videos"
CUT_VIDEOS_DIR = DATASET_ROOT / "cut_videos"
METADATA_FILE = DATASET_ROOT / "metadata.jsonl"
PREPROCESSED_LATENTS_DIR = PROJECT_ROOT / "data" / "preprocessed_latents"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
LORA_CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "lora_checkpoints"

# Model IDs
WAN_MODEL_MAP = {
    "14B": "Wan-AI/Wan2.2-T2V-A14B",
    "5B": "Wan-AI/Wan2.2-TI2V-5B",
    "1.3B": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}

CAPTION_MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Training defaults
DEFAULT_TRAINING_CONFIG = {
    "epochs": 5,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "lora_r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "save_steps": 100,
    "logging_steps": 10,
}

# Video processing defaults
VIDEO_CONFIG = {
    "clip_length": 8,  # seconds
    "overlap": 2,  # seconds
    "target_fps": 24,
    "target_width": 1280,
    "target_height": 720,
}


def ensure_directories():
    """Create all necessary directories."""
    dirs = [
        RAW_VIDEOS_DIR,
        CUT_VIDEOS_DIR,
        PREPROCESSED_LATENTS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        LORA_CHECKPOINTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Directories initialized")


def load_config_yaml() -> dict:
    """Load config from models.yaml if it exists."""
    config_file = PROJECT_ROOT / "config" / "models.yaml"
    if config_file.exists() and config_file.stat().st_size > 0:
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    return {}


# Initialize on import
ensure_directories()
