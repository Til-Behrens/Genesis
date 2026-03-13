"""
Efficient video preprocessing for training.
Encodes videos to VAE latents once and caches them for fast training.
"""
import json
import torch
from pathlib import Path
from typing import Generator, Dict, Any
import logging
import cv2
import numpy as np

logger = logging.getLogger("Preprocessor")


class LatentDataset(torch.utils.data.Dataset):
    """Dataset that loads pre-encoded VAE latents from disk."""

    def __init__(self, latents_dir: Path, metadata_path: Path):
        self.latents_dir = Path(latents_dir)
        self.metadata_path = Path(metadata_path)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

        # Filter samples that have cached latents
        self.samples = [
            s for s in self.samples
            if (self.latents_dir / f"{Path(s['file_name']).stem}.pt").exists()
        ]

        logger.info(f"Loaded {len(self.samples)} samples with cached latents")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        latent_path = self.latents_dir / f"{Path(sample['file_name']).stem}.pt"

        # Load cached latent
        latent = torch.load(latent_path, map_location="cpu")

        # Validate shape: should be [B, 16, T, H, W]
        if latent.ndim != 5:
            raise ValueError(
                f"Invalid latent shape from {latent_path.name}: "
                f"expected 5D [B, 16, T, H, W], got {latent.ndim}D {tuple(latent.shape)}"
            )

        if latent.shape[1] != 16:
            raise ValueError(
                f"Invalid latent channels from {latent_path.name}: "
                f"expected 16 channels, got {latent.shape[1]}. Full shape: {tuple(latent.shape)}"
            )

        return {
            "latents": latent,
            "text": sample["text"],
            "file_name": sample["file_name"]
        }


def video_to_frames_tensor(
    video_path: Path,
    num_frames: int = 33,  # Adjusted for Wan: 4n+1 pattern, e.g., 33 for ~1.3s at 24fps
    target_height: int = 720,  # Divisible by 8 for VAE
    target_width: int = 1280,
) -> torch.Tensor:
    """
    Load video and convert to tensor [channels, num_frames, height, width].
    Output format matches VAE input expectation: [C, T, H, W]
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise ValueError(f"Video {video_path} has no frames")

    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, (target_width, target_height))
            frames.append(frame)

    cap.release()

    if len(frames) != num_frames:
        raise ValueError(f"Could not extract {num_frames} frames from {video_path}")

    # Convert to tensor: [T, H, W, C] -> [C, T, H, W] for VAE input
    # First: [T, H, W, C] -> [T, C, H, W]
    frames_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)
    # Then: [T, C, H, W] -> [C, T, H, W]
    frames_tensor = frames_tensor.permute(1, 0, 2, 3)
    # Normalize to [-1, 1]
    frames_tensor = frames_tensor.float() / 127.5 - 1.0

    return frames_tensor


def preprocess_videos_to_latents(
    videos_dir: Path | str,
    metadata_path: Path | str,
    output_cache_dir: Path | str,
    vae,
    device: str = "cuda",
    num_frames: int = 33,
    target_height: int = 720,
    target_width: int = 1280,
) -> Generator[Dict[str, Any], None, None]:
    """
    Preprocess all videos to VAE latents and cache them.

    Args:
        videos_dir: Directory containing video files
        metadata_path: Path to metadata.jsonl file
        output_cache_dir: Directory to cache encoded latents
        vae: VAE model for encoding
        device: Device to use for encoding ("cuda" or "cpu")
        num_frames: Number of frames to extract (4n+1 for Wan, e.g., 33)
        target_height: Target video height (divisible by 8)
        target_width: Target video width (divisible by 8)

    Yields:
        Progress updates with format:
        {"status": "processing", "video": filename, "progress": (current, total)}
        {"status": "complete", "total_processed": int, "cache_dir": str}
        {"status": "error", "message": str}
    """
    # Validate dimensions for VAE (spatial divide by 8, temporal by 4)
    if target_height % 8 != 0 or target_width % 8 != 0:
        yield {
            "status": "error",
            "message": f"Height/width must be divisible by 8 for VAE. Use 720x1280."
        }
        return

    if (num_frames - 1) % 4 != 0:
        yield {
            "status": "error",
            "message": f"Num frames must be 4n+1 for Wan VAE temporal compression."
        }
        return

    videos_dir = Path(videos_dir)
    metadata_path = Path(metadata_path)
    output_cache_dir = Path(output_cache_dir)
    output_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    if not samples:
        yield {"status": "error", "message": "No samples in metadata"}
        return

    vae = vae.to(device)
    vae.eval()

    processed_count = 0

    for idx, sample in enumerate(samples, 1):
        video_path = videos_dir / sample["file_name"]
        latent_path = output_cache_dir / f"{Path(sample['file_name']).stem}.pt"

        # Skip if already cached
        if latent_path.exists():
            yield {
                "status": "processing",
                "video": sample["file_name"],
                "progress": (idx, len(samples)),
                "message": "Already cached, skipping"
            }
            processed_count += 1
            continue

        if not video_path.exists():
            yield {
                "status": "processing",
                "video": sample["file_name"],
                "progress": (idx, len(samples)),
                "message": "Video file not found, skipping"
            }
            continue

        try:
            # Load video
            frames_tensor = video_to_frames_tensor(
                video_path,
                num_frames=num_frames,
                target_height=target_height,
                target_width=target_width
            )

            # Encode to latent
            with torch.no_grad():
                # VAE expects [B, C, T, H, W] with B=1, C=3
                frames_tensor = frames_tensor.unsqueeze(0)  # [1, 3, T, H, W]

                # Move to device and convert to VAE's dtype
                vae_dtype = next(vae.parameters()).dtype
                frames_tensor = frames_tensor.to(device=device, dtype=vae_dtype)

                # DEBUG: Log VAE input shape
                logger.info(f"VAE Input shape: {frames_tensor.shape}")

                # Get VAE encoder output
                vae_output = vae.encode(frames_tensor)
                logger.info(f"VAE encode() output type: {type(vae_output)}")
                logger.info(f"VAE encode() output: {vae_output}")

                # Extract latent distribution
                latent_dist = vae_output.latent_dist
                logger.info(f"latent_dist type: {type(latent_dist)}")
                logger.info(f"latent_dist.mean shape: {latent_dist.mean.shape}")
                logger.info(f"latent_dist.logvar shape: {latent_dist.logvar.shape}")

                # Use .mean() instead of .sample() for deterministic, reproducible training
                # (Official Wan2.2 code uses mean for stable LoRA training)
                latent = latent_dist.mean
                logger.info(f"Latent shape (from .mean): {latent.shape}")

                latent = latent.cpu()

            # Validate latent shape: should be [B, 16, T_latent, H_latent, W_latent]
            if latent.ndim != 5:
                error_msg = f"Invalid latent dimensionality: expected 5D, got {latent.ndim}D with shape {latent.shape}"
                logger.error(error_msg)
                yield {
                    "status": "processing",
                    "video": sample["file_name"],
                    "progress": (idx, len(samples)),
                    "message": f"Error: {error_msg}"
                }
                continue

            if latent.shape[1] != 16:
                error_msg = f"Invalid latent channels: expected 16, got {latent.shape[1]} with shape {latent.shape}"
                logger.error(error_msg)
                yield {
                    "status": "processing",
                    "video": sample["file_name"],
                    "progress": (idx, len(samples)),
                    "message": f"Error: {error_msg}"
                }
                continue

            torch.save(latent, latent_path)
            processed_count += 1

            yield {
                "status": "processing",
                "video": sample["file_name"],
                "progress": (idx, len(samples)),
                "message": f"Encoded and cached | Shape: {list(latent.shape)}"
            }

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}", exc_info=True)
            yield {
                "status": "processing",
                "video": sample["file_name"],
                "progress": (idx, len(samples)),
                "message": f"Error: {str(e)}"
            }

    yield {
        "status": "complete",
        "total_processed": processed_count,
        "cache_dir": str(output_cache_dir)
    }


def clear_cache(cache_dir: Path | str) -> int:
    """Clear all cached latents. Returns number of files deleted."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0

    count = 0
    for file in cache_dir.glob("*.pt"):
        file.unlink()
        count += 1

    return count