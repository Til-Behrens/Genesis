"""
Cuts tutoial videos down to 8s clips.
"""

import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Cutter")

script_dir = Path(__file__).parent
input_dir = script_dir / "dataset" / "raw_videos"
output_dir = script_dir / "dataset" / "cut_videos"
output_dir.mkdir(parents=True, exist_ok=True)

clips = []
clip_id = 0

# use this if you have ffmpeg installed
FFMPEG  = ["ffmpeg"]
FFPROBE = ["ffprobe"]

# FFMPEG  = ["distrobox", "enter", "NeoVim", "--", "ffmpeg"]
# FFPROBE = ["distrobox", "enter", "NeoVim", "--", "ffprobe"]

for video_file in tqdm(list(input_dir.glob("*.mp4")), desc="Processing videos\n"):
    duration_str = subprocess.check_output(
        FFPROBE + [
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_file)
        ],
        text=True
    ).strip()

    if not duration_str:
        logger.warning(f"Error: {video_file} has no duration – skip.")
        continue

    duration = float(duration_str)
    if duration < 8:
        logger.warning(f"Too short: {video_file} ({duration}s) – skip.")
        continue

    # 8s clips with 2s overlap (for better coherence)
    overlap = 2
    step = 8 - overlap
    for start in range(0, int(duration) - 8, step):
        output_file = output_dir / f"clip_{clip_id:04d}_{video_file.stem}.mp4"

        cmd = FFMPEG + [
            "-y",
            "-i", str(video_file),
            "-ss", str(start),
            "-t", "8",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black",
            "-r", "24",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_file)
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clips.append({
                "file_name": output_file.name,
                "original_video": video_file.name,
                "start_time": start,
                "text": ""
            })
            clip_id += 1
        except subprocess.CalledProcessError:
            logger.warning(f"Error cutting {video_file} at {start}s – skip clip.")

    logger.info(f"{video_file}: {int((duration - 8) / step) + 1} clips created")

# Store metadata (JSONL for LoRA)
with open("dataset/metadata.jsonl", "w", encoding="utf-8") as f:
    for clip in clips:
        f.write(json.dumps(clip, ensure_ascii=False) + "\n")

logger.info(f"\nFinished! {len(clips)} Clips created in {output_dir}")
