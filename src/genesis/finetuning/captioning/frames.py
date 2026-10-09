"""Keyframe selection: one frame for static clips, more where the picture changes."""
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# mean absolute pixel difference (0-255) that counts as a change; tutorial clips change subtly
CHANGE_THRESHOLD = 0.5
# candidate frames sampled per requested keyframe
OVERSAMPLING = 4


def select_keyframes(frames: list[np.ndarray], max_frames: int, threshold: float = CHANGE_THRESHOLD) -> list[int]:
    """Return indices of frames that differ from their predecessor, thinned out to `max_frames`.

    The first frame is always kept, so a static clip yields exactly one keyframe.
    """
    if not frames:
        return []
    keys = [0] + [
        i for i in range(1, len(frames))
        if cv2.absdiff(frames[i - 1], frames[i]).mean() > threshold
    ]
    if len(keys) > max_frames:
        step = len(keys) / max_frames
        keys = [keys[int(j * step)] for j in range(max_frames)]
    return keys


def extract_keyframes(video_path: Path, max_frames: int) -> list[Image.Image]:
    """Sample a clip densely and return its keyframes as RGB images."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total // (max_frames * OVERSAMPLING)) if total else 1
        frames = []
        for index in range(0, total, interval):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    keys = select_keyframes(frames, max_frames)
    logger.info("%s: %d keyframes from %d frames", video_path.name, len(keys), total)
    return [Image.fromarray(frames[i]) for i in keys]
