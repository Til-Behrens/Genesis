"""Cut long tutorial recordings into overlapping, uniformly encoded training clips."""
import json
import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path

from genesis import events
from genesis.config import CLIP_FPS, CLIP_HEIGHT, CLIP_LENGTH_SEC, CLIP_OVERLAP_SEC, CLIP_WIDTH
from genesis.events import Event

logger = logging.getLogger(__name__)


def ffmpeg_available(ffmpeg_cmd: str = "ffmpeg") -> bool:
    """Return whether `ffmpeg_cmd` can be executed."""
    try:
        subprocess.run([ffmpeg_cmd, "-version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def clip_starts(duration: float, clip_length: int, overlap: int) -> list[int]:
    """Start times in whole seconds of all full-length clips that fit into `duration`."""
    if not 0 <= overlap < clip_length:
        raise ValueError(f"Overlap ({overlap}s) must be smaller than clip length ({clip_length}s).")
    return list(range(0, int(duration) - clip_length + 1, clip_length - overlap))


def probe_duration(video: Path, ffprobe_cmd: str = "ffprobe") -> float:
    """Container duration of `video` in seconds."""
    output = subprocess.check_output(
        [ffprobe_cmd, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        text=True,
        timeout=30,
    ).strip()
    return float(output)


def _cut_command(
    ffmpeg_cmd: str, video: Path, start: int, clip_length: int, output: Path,
    width: int, height: int, fps: int,
) -> list[str]:
    # letterbox to the target size so every clip shares one resolution
    scale = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    )
    return [
        ffmpeg_cmd, "-y", "-ss", str(start), "-i", str(video), "-t", str(clip_length),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-vf", scale, "-r", str(fps),
        "-c:a", "aac", "-b:a", "128k", str(output),
    ]


def cut_videos(
    input_dir: Path | str,
    output_dir: Path | str,
    metadata_path: Path | str,
    clip_length: int = CLIP_LENGTH_SEC,
    overlap: int = CLIP_OVERLAP_SEC,
    width: int = CLIP_WIDTH,
    height: int = CLIP_HEIGHT,
    fps: int = CLIP_FPS,
    ffmpeg_cmd: str = "ffmpeg",
    ffprobe_cmd: str = "ffprobe",
) -> Iterator[Event]:
    """Cut every `.mp4` in `input_dir` into clips and write a fresh metadata file.

    Each metadata row has `file_name`, `original_video`, `start_time` and an empty
    `prompt` that the captioning step fills in. Videos shorter than one clip are skipped.

    Yields:
        One `progress` event per source video, then `complete` (or `error`).
    """
    input_dir, output_dir, metadata_path = Path(input_dir), Path(output_dir), Path(metadata_path)

    if not 0 <= overlap < clip_length:
        yield events.error(f"Overlap ({overlap}s) must be smaller than clip length ({clip_length}s).")
        return
    if not ffmpeg_available(ffmpeg_cmd):
        yield events.error("ffmpeg not found, install it first.")
        return
    videos = sorted(p for p in input_dir.glob("*") if p.suffix.lower() == ".mp4")
    if not videos:
        yield events.error(f"No .mp4 files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []

    for index, video in enumerate(videos, 1):
        try:
            duration = probe_duration(video, ffprobe_cmd)
        except (subprocess.SubprocessError, ValueError) as e:
            yield events.progress(index, len(videos), f"{video.name}: cannot read duration ({e}), skipped")
            continue

        starts = clip_starts(duration, clip_length, overlap)
        if not starts:
            yield events.progress(index, len(videos), f"{video.name}: shorter than {clip_length}s, skipped")
            continue

        created = 0
        for start in starts:
            output = output_dir / f"clip_{len(clips):04d}_{video.stem}.mp4"
            cmd = _cut_command(ffmpeg_cmd, video, start, clip_length, output, width, height, fps)
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logger.warning("Cutting %s at %ss failed, clip skipped", video.name, start)
                continue
            clips.append({
                "file_name": output.name,
                "original_video": video.name,
                "start_time": start,
                "prompt": "",
            })
            created += 1

        yield events.progress(index, len(videos), f"{video.name}: {created} clips (total {len(clips)})")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        for clip in clips:
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")

    yield events.complete(
        f"{len(clips)} clips written to {output_dir}, metadata: {metadata_path}", total_clips=len(clips)
    )
