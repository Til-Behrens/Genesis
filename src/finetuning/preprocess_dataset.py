"""Prepare DiffSynth-compatible dataset metadata using diffsynth.core.data."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator

logger = logging.getLogger("Preprocessor")


def _load_records_with_diffsynth(metadata_path: Path) -> list[dict[str, Any]]:
    """Load metadata through DiffSynth's UnifiedDataset parser (csv/json/jsonl)."""
    try:
        from diffsynth.core.data import UnifiedDataset
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "DiffSynth data module is unavailable. Install DiffSynth training deps "
            "(including torchaudio) before preprocessing dataset metadata."
        ) from e

    dataset = UnifiedDataset(base_path="", metadata_path=str(metadata_path), repeat=1)
    return [dict(row) for row in dataset.data]


def _normalize_for_diffsynth(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one record to canonical fields used by Genesis + DiffSynth."""
    file_name = record.get("file_name") or record.get("video")
    if not file_name:
        raise ValueError("Metadata record is missing 'file_name' (or fallback 'video').")

    prompt = record.get("prompt")
    if prompt is None:
        prompt = record.get("text", "")

    normalized: dict[str, Any] = {
        "file_name": str(file_name),
        "prompt": str(prompt or ""),
        # Keep text for backward compatibility with existing flows.
        "text": str(record.get("text", prompt or "")),
    }

    for key in ("audio_text", "original_video", "start_time"):
        if key in record:
            normalized[key] = record[key]

    return normalized


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_diffsynth_dataset(
    videos_dir: Path | str,
    metadata_path: Path | str,
    output_metadata_path: Path | str | None = None,
) -> Generator[Dict[str, Any], None, None]:
    """Validate videos and export a normalized jsonl metadata file for DiffSynth training."""
    videos_dir = Path(videos_dir)
    metadata_path = Path(metadata_path)
    output_metadata_path = Path(output_metadata_path) if output_metadata_path else metadata_path.with_name(
        f"{metadata_path.stem}_diffsynth.jsonl"
    )

    if not metadata_path.exists():
        yield {"status": "error", "message": f"Metadata file not found: {metadata_path}"}
        return

    try:
        raw_records = _load_records_with_diffsynth(metadata_path)
    except ModuleNotFoundError as e:
        yield {"status": "error", "message": str(e)}
        return
    if not raw_records:
        yield {"status": "error", "message": "No samples in metadata."}
        return

    normalized_records: list[dict[str, Any]] = []
    missing_videos = 0

    for idx, record in enumerate(raw_records, 1):
        try:
            normalized = _normalize_for_diffsynth(record)
            video_path = videos_dir / normalized["file_name"]

            if not video_path.exists():
                missing_videos += 1
                yield {
                    "status": "processing",
                    "video": normalized["file_name"],
                    "progress": (idx, len(raw_records)),
                    "message": "Video file not found, skipping",
                }
                continue

            normalized_records.append(normalized)
            yield {
                "status": "processing",
                "video": normalized["file_name"],
                "progress": (idx, len(raw_records)),
                "message": "Validated",
            }
        except Exception as e:
            logger.error("Error normalizing metadata row %s: %s", idx, e, exc_info=True)
            yield {
                "status": "processing",
                "video": str(record.get("file_name", "unknown")),
                "progress": (idx, len(raw_records)),
                "message": f"Error: {e}",
            }

    if not normalized_records:
        yield {"status": "error", "message": "No valid records after normalization."}
        return

    _write_jsonl(normalized_records, output_metadata_path)
    yield {
        "status": "complete",
        "total_processed": len(normalized_records),
        "total_input_records": len(raw_records),
        "missing_videos": missing_videos,
        "diffsynth_metadata_path": str(output_metadata_path),
    }


def preprocess_videos_to_latents(
    videos_dir: Path | str,
    metadata_path: Path | str,
    output_cache_dir: Path | str,
    vae=None,
    device: str = "cuda",
    num_frames: int = 33,
    target_height: int = 720,
    target_width: int = 1280,
    export_diffsynth_metadata: bool = True,
    diffsynth_metadata_path: Path | str | None = None,
) -> Generator[Dict[str, Any], None, None]:
    """Compatibility wrapper: old name now delegates to metadata preparation only."""
    del output_cache_dir, vae, device, num_frames, target_height, target_width, export_diffsynth_metadata
    yield from prepare_diffsynth_dataset(videos_dir, metadata_path, diffsynth_metadata_path)
