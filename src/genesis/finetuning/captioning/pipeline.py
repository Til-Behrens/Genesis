"""Caption every clip in the metadata file."""
import logging
from collections.abc import Iterator
from pathlib import Path

from genesis import events
from genesis.config import CAPTION_BACKEND, CAPTION_CACHE_DIR, CAPTION_MAX_FRAMES
from genesis.events import Event
from genesis.finetuning.captioning.backends import create_backend
from genesis.finetuning.captioning.frames import extract_keyframes
from genesis.finetuning.captioning.transcription import Transcriber
from genesis.finetuning.dataset import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def generate_captions(
    metadata_path: Path | str,
    clips_dir: Path | str,
    backend: str = CAPTION_BACKEND,
    cache_dir: Path | str = CAPTION_CACHE_DIR,
    max_frames: int = CAPTION_MAX_FRAMES,
) -> Iterator[Event]:
    """Fill the `prompt` of every clip row and rewrite the metadata file after each clip.

    Clips that are missing or fail keep an empty prompt, which the dataset step skips.
    Backends that use transcripts also store the clip's speech as `audio_text`.

    Yields:
        `info` while loading, one `progress` event per clip, then `complete` (or `error`).
    """
    metadata_path, clips_dir = Path(metadata_path), Path(clips_dir)
    if not metadata_path.exists():
        yield events.error(f"Metadata file not found: {metadata_path}")
        return
    clips = read_jsonl(metadata_path)
    if not clips:
        yield events.error("No clips in metadata.")
        return

    try:
        captioner = create_backend(backend, cache_dir)
        yield events.info(f"Loading {backend} ({captioner.model_id})...")
        captioner.load()
    except Exception as e:
        logger.exception("Loading caption backend failed")
        yield events.error(f"Failed to load {backend}: {e}")
        return
    transcriber = Transcriber(cache_dir) if captioner.uses_transcript else None

    captioned = 0
    try:
        for index, clip in enumerate(clips, 1):
            name = clip["file_name"]
            clip["prompt"] = clip["audio_text"] = ""
            video = clips_dir / name
            try:
                if not video.exists():
                    yield events.progress(index, len(clips), f"{name}: not found, skipped")
                    continue
                transcript = transcriber.transcribe(video) if transcriber else ""
                caption = captioner.caption(extract_keyframes(video, max_frames), transcript)
                clip["audio_text"] = transcript
                if not caption:
                    yield events.progress(index, len(clips), f"{name}: empty caption, skipped")
                    continue
                clip["prompt"] = caption
                captioned += 1
                yield events.progress(index, len(clips), f"{name}\n   -> {caption[:100]}", caption=caption)
            except Exception as e:
                logger.exception("Captioning %s failed", name)
                yield events.progress(index, len(clips), f"{name}: error ({e}), skipped")
            finally:
                # saved per clip so an interrupted run keeps finished captions
                write_jsonl(metadata_path, clips)
    finally:
        captioner.unload()
        if transcriber:
            transcriber.unload()

    yield events.complete(f"{captioned}/{len(clips)} clips captioned", captioned=captioned, total=len(clips))
