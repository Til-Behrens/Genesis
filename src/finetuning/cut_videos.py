"""
Cuts tutorial videos down to 8s clips.
"""

import subprocess
import json
from pathlib import Path
import logging
from typing import Generator, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Cutter")


def check_ffmpeg_available(ffmpeg_cmd: str = "ffmpeg") -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            [ffmpeg_cmd, "-version"],
            capture_output=True,
            check=True,
            timeout=5
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def cut_videos_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    metadata_path: Path | str,
    clip_length: int = 8,
    overlap: int = 2,
    ffmpeg_cmd: str = "ffmpeg",
    ffprobe_cmd: str = "ffprobe",
    target_width: int = 1280,
    target_height: int = 720,
    target_fps: int = 24,
) -> Generator[Dict[str, Any], None, None]:
    """
    Cut videos into clips with progress updates.

    Yields:
        Progress updates with format:
        {"status": "processing", "video": filename, "clips_created": int, "total_clips": int}
        {"status": "complete", "total_clips": int, "metadata_path": str}
        {"status": "error", "message": str}
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    metadata_path = Path(metadata_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check ffmpeg availability
    if not check_ffmpeg_available(ffmpeg_cmd):
        yield {"status": "error", "message": f"ffmpeg not found. Please install ffmpeg first."}
        return

    clips = []
    clip_id = 0

    video_files = list(input_dir.glob("*.mp4"))
    if not video_files:
        yield {"status": "error", "message": f"No .mp4 files found in {input_dir}"}
        return

    for video_file in video_files:
        try:
            # Get video duration
            duration_str = subprocess.check_output(
                [ffprobe_cmd,
                 "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1",
                 str(video_file)],
                text=True,
                timeout=30
            ).strip()

            if not duration_str:
                logger.warning(f"Error: {video_file} has no duration – skip.")
                continue

            duration = float(duration_str)
            if duration < clip_length:
                logger.warning(f"Too short: {video_file} ({duration}s) – skip.")
                continue

            # Calculate clips
            step = clip_length - overlap
            num_clips = max(1, int((duration - clip_length) / step) + 1)

            clips_from_this_video = 0
            for start in range(0, int(duration) - clip_length + 1, step):
                output_file = output_dir / f"clip_{clip_id:04d}_{video_file.stem}.mp4"

                cmd = [
                    ffmpeg_cmd,
                    "-y",
                    "-i", str(video_file),
                    "-ss", str(start),
                    "-t", str(clip_length),
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "fast",
                    "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black",
                    "-r", str(target_fps),
                    "-c:a", "aac",
                    "-b:a", "128k",
                    str(output_file)
                ]

                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=60
                    )
                    clips.append({
                        "file_name": output_file.name,
                        "original_video": video_file.name,
                        "start_time": start,
                        "text": ""
                    })
                    clip_id += 1
                    clips_from_this_video += 1

                except subprocess.CalledProcessError:
                    logger.warning(f"Error cutting {video_file} at {start}s – skip clip.")

            yield {
                "status": "processing",
                "video": video_file.name,
                "clips_created": clips_from_this_video,
                "total_clips": clip_id
            }

        except Exception as e:
            logger.error(f"Error processing {video_file}: {e}")
            yield {"status": "error", "message": f"Error processing {video_file.name}: {str(e)}"}

    # Save metadata
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        for clip in clips:
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")

    logger.info(f"\nFinished! {len(clips)} clips created in {output_dir}")
    yield {
        "status": "complete",
        "total_clips": len(clips),
        "metadata_path": str(metadata_path)
    }


# Legacy main for backward compatibility
def main():
    script_dir = Path(__file__).parent
    input_dir = script_dir / "dataset" / "raw_videos"
    output_dir = script_dir / "dataset" / "cut_videos"
    metadata_file = script_dir / "dataset" / "metadata.jsonl"

    for update in cut_videos_pipeline(input_dir, output_dir, metadata_file):
        if update["status"] == "error":
            logger.error(update["message"])
        elif update["status"] == "processing":
            logger.info(f"{update['video']}: {update['clips_created']} clips created")


if __name__ == "__main__":
    main()

