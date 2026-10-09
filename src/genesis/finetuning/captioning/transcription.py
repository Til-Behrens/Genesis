"""Speech transcription of clips with Whisper, used as caption context."""
import logging
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import pipeline

from genesis.config import WHISPER_MODEL_ID

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 128


class Transcriber:
    """Whisper wrapper; loads lazily and calls the model directly.

    The transformers ASR pipeline call fails on short clips (`num_frames` KeyError), so
    only its feature extractor, model and tokenizer are used.
    """

    def __init__(self, cache_dir: Path | str, model_id: str = WHISPER_MODEL_ID):
        self.cache_dir = str(cache_dir)
        self.model_id = model_id
        self._asr = None

    def load(self) -> None:
        """Load whisper if needed."""
        if self._asr is None:
            self._asr = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=0 if torch.cuda.is_available() else -1,
                model_kwargs={"cache_dir": self.cache_dir},
            )

    def unload(self) -> None:
        """Release the model."""
        self._asr = None

    def transcribe(self, video_path: Path) -> str:
        """Speech in the clip as text; empty if it has no audio or transcription fails."""
        try:
            self.load()
            with tempfile.TemporaryDirectory() as tmp_dir:
                wav_path = Path(tmp_dir) / "audio.wav"
                if not extract_audio(video_path, wav_path):
                    return ""
                return self._transcribe_wav(wav_path)
        except Exception as e:
            logger.warning("Transcription of %s failed: %s", video_path.name, e)
            return ""

    def _transcribe_wav(self, wav_path: Path) -> str:
        with wave.open(str(wav_path), "rb") as wav:
            channels, rate = wav.getnchannels(), wav.getframerate()
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        model = self._asr.model
        features = self._asr.feature_extractor(audio, sampling_rate=rate, return_tensors="pt")
        features = features["input_features"].to(next(model.parameters()).device, model.dtype)
        with torch.no_grad():
            ids = model.generate(features, max_new_tokens=MAX_NEW_TOKENS)
        text = self._asr.tokenizer.batch_decode(ids, skip_special_tokens=True)
        return text[0].strip() if text else ""


def extract_audio(video_path: Path, wav_path: Path) -> bool:
    """Write the clip's audio as 16 kHz mono pcm wav; False if it has none or ffmpeg fails."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav_path)],
            capture_output=True, timeout=300, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Audio extraction from %s failed: %s", video_path.name, e)
        return False
    return wav_path.exists() and wav_path.stat().st_size > 0
