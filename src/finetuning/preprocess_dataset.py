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

        return {
            "latents": latent,
            "text": sample["text"],
            "file_name": sample["file_name"]
        }


def video_to_frames_tensor(
    video_path: Path,
    num_frames: int = 192,  # 8 seconds @ 24fps
    target_height: int = 720,
    target_width: int = 1280,
) -> torch.Tensor:
    """
    Load video and convert to tensor [num_frames, channels, height, width].
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

    # Convert to tensor: [num_frames, height, width, channels] -> [num_frames, channels, height, width]
    frames_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)
    # Normalize to [-1, 1]
    frames_tensor = frames_tensor.float() / 127.5 - 1.0

    return frames_tensor


def preprocess_videos_to_latents(
    videos_dir: Path | str,
    metadata_path: Path | str,
    output_cache_dir: Path | str,
    vae,
    device: str = "cuda",
    num_frames: int = 192,
    target_height: int = 720,
    target_width: int = 1280,
) -> Generator[Dict[str, Any], None, None]:
    """
    Preprocess all videos to VAE latents and cache them.

    Yields:
        Progress updates with format:
        {"status": "processing", "video": filename, "progress": (current, total)}
        {"status": "complete", "total_processed": int, "cache_dir": str}
        {"status": "error", "message": str}
    """
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
                frames_tensor = frames_tensor.unsqueeze(0).to(device)  # Add batch dimension
                latent = vae.encode(frames_tensor).latent_dist.sample()
                latent = latent.cpu()

            # Save latent
            torch.save(latent, latent_path)
            processed_count += 1

            yield {
                "status": "processing",
                "video": sample["file_name"],
                "progress": (idx, len(samples)),
                "message": f"Encoded and cached (shape: {latent.shape})"
            }

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
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
