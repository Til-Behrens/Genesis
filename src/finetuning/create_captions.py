"""Per-clip caption generation for the fine-tuning dataset (vision-language backends plus whisper transcripts)."""

import json
from pathlib import Path
import cv2
from PIL import Image
import torch
import subprocess
import tempfile
import atexit
import gc
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    ViTImageProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    pipeline,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
import logging
from typing import Generator, Dict, Any, Optional

from src.core.config import (
    CAPTION_MODELS,
    CAPTION_BACKEND,
    CAPTION_MAX_FRAMES,
    CAPTION_USE_SUMMARIZER,
    CAPTION_SUMMARIZER_ID,
    get_optimal_device, CACHE_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CaptionGen")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

WHISPER_MODEL_ID = "openai/whisper-base"
TRANSCRIPT_BACKENDS = {"qwen3-vl"}


SYSTEM_PROMPT = """**Situation**
Du bist ein Experte für die Erstellung von Text-to-Video-Prompts für deutsche Universitäts-Tutoriumsvideos im Bereich Informatik. Du analysierst Bildmaterial aus akademischen Lehrvideos, um präzise Beschreibungen für die Video-Generierung zu erstellen.

**Aufgabe**
Die Assistenz soll die bereitgestellten Bilder analysieren und daraus einen exakten Text-to-Video-Prompt formulieren. Der Prompt muss alle visuellen Elemente, Bewegungen, Übergänge, Textelemente, Diagramme, Code-Snippets und didaktischen Darstellungen präzise erfassen. Die Beschreibung soll so detailliert sein, dass ein Video-Generierungsmodell das ursprüngliche Tutoriumsvideo originalgetreu rekonstruieren kann.

**Ziel**
Das Ziel ist die Erstellung eines produktionsreifen Text-to-Video-Prompts, der die Essenz und alle relevanten Details eines deutschen Informatik-Tutoriumsvideos vollständig einfängt, sodass das generierte Video für Lehrzwecke an deutschen Universitäten verwendet werden kann.

**Wissen**
Bei der Analyse müssen folgende Aspekte berücksichtigt werden:
- Visuelle Komposition: Kamerawinkel, Bildaufbau, Farbschema, Beleuchtung
- Personen: Anzahl, Position, Kleidung, Gestik, Mimik, Bewegungen
- Technische Elemente: Whiteboards, Präsentationsfolien, Bildschirme, Projektionen
- Textinhalte: Alle sichtbaren Texte, Formeln, Code-Zeile für Zeile, Diagrammbeschriftungen
- Didaktische Elemente: Zeigegesten, Markierungen, Hervorhebungen, Animationen
- Zeitliche Abfolge: Reihenfolge der Ereignisse, Übergänge zwischen Szenen
- Audio-visuelle Hinweise: Hinweise auf gesprochene Inhalte durch Lippenbewegungen oder Präsentationskontext
- Akademischer Kontext: Raumgestaltung, universitäre Atmosphäre, Formalitätsgrad

Der Prompt muss in deutscher Sprache verfasst werden und die spezifischen Konventionen deutscher Universitätslehre widerspiegeln.

**Output-Format**
Die Assistenz soll ausschließlich den fertigen Text-to-Video-Prompt ausgeben ohne jegliche Einleitung, Erklärung, Metakommentare oder abschließende Bemerkungen. Der Prompt beginnt direkt mit der Beschreibung des Videos."""


def extract_audio_from_video(video_path: Path | str, output_wav: Path | str) -> bool:
    """Extract audio from video file to WAV format using ffmpeg.

    Returns:
        True if extraction successful, False otherwise.
    """
    try:
        # Extract audio with specific format for Whisper compatibility
        # - vn: no video
        # - acodec pcm_s16le: 16-bit PCM audio codec
        # - ar 16000: 16kHz sample rate (Whisper's native rate)
        # - ac 1: mono audio
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output file
                str(output_wav)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
            check=True,
        )
        output_file = Path(output_wav)
        if output_file.exists() and output_file.stat().st_size > 0:
            return True
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"FFmpeg failed to extract audio from {video_path}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to extract audio from {video_path}: {e}")
        return False


def transcribe_audio_whisper(audio_path: Path | str, whisper_model: Any) -> str:
    """Transcribe audio to text using Whisper model.

    Note: whisper_model here is actually an ASRPipeline, but we access its underlying
    model and processor directly to avoid the "num_frames" KeyError in the pipeline.

    Handles short audio clips (typically 8 seconds from video clips).

    Returns:
        Transcribed text or empty string if transcription failed.
    """
    try:
        import wave
        import numpy as np
        import torch

        # Read the WAV file manually to avoid pipeline issues with file paths
        with wave.open(str(audio_path), 'rb') as wav_file:
            # Get audio parameters
            n_channels = wav_file.getnchannels()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            # Read all frames
            audio_data = wav_file.readframes(n_frames)

            # Convert byte data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # If stereo, convert to mono by taking mean
            if n_channels > 1:
                audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)

        # Get the underlying model and feature extractor from the pipeline
        model = whisper_model.model
        feature_extractor = whisper_model.feature_extractor
        tokenizer = whisper_model.tokenizer

        # Process audio using feature extractor directly
        # This bypasses the pipeline's problematic preprocessing
        inputs = feature_extractor(
            audio_array,
            sampling_rate=framerate,
            return_tensors="pt"
        )

        # Ensure inputs are on the same device as the model
        device = next(model.parameters()).device
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].to(device)

        # Generate token IDs
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"],
                max_new_tokens=128
            )

        # Decode tokens to text
        text = tokenizer.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        if text and text[0]:
            return text[0].strip()
        return ""

    except Exception as e:
        logger.warning(f"Whisper transcription failed for {audio_path}: {e}")
        return ""



class BaseCaptionGenerator:
    """Interface for caption generator implementations."""

    def __init__(self, model_id: str, cache_dir: str):
        self.model_id = model_id
        self.cache_dir = cache_dir


    def video_to_frames(self, video_path: Path, max_frames: int = 8) -> list[Image.Image]:
        """
        Intelligently extracts frames from a video based on scene changes.

        Uses pixel difference detection to find keyframes where content changes.
        For static content (e.g., code editor unchanged), returns 1 frame.
        For dynamic content, returns up to max_frames frames at change points.

        This ensures:
        - Static tutorial videos get one caption per clip (not e.g. 8 identical ones)
        - Dynamic videos get proper frame sampling showing progression
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []

        # Read all frames (or sample if too many)
        sample_interval = max(1, total_frames // (max_frames * 4))  # Sample densely for change detection
        all_frames = []
        frame_indices = []

        for idx in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame_rgb)
                frame_indices.append(idx)

        cap.release()

        if not all_frames:
            return []

        # Detect keyframes based on pixel differences
        keyframe_indices = [0]  # Always include first frame

        # Lower threshold to detect more subtle changes (tutorial videos often have small changes)
        threshold = 0.5  # Pixel difference threshold (0-255 scale, per channel)

        for i in range(1, len(all_frames)):
            prev_frame = all_frames[i - 1]
            curr_frame = all_frames[i]

            # Compute mean absolute difference
            diff = cv2.absdiff(prev_frame.astype('float32'), curr_frame.astype('float32'))
            mean_diff = diff.mean()

            # If difference is significant, mark as keyframe
            if mean_diff > threshold:
                keyframe_indices.append(i)

        # Don't force include last frame since clips overlap
        # and the last frame of one clip is similar to the first frame of the next

        # Limit to max_frames by selecting evenly spaced keyframes
        if len(keyframe_indices) > max_frames:
            step = len(keyframe_indices) / max_frames
            keyframe_indices = [keyframe_indices[int(j * step)] for j in range(max_frames)]

        # Extract keyframes as PIL Images
        result_frames = []
        for idx in keyframe_indices:
            frame_array = all_frames[idx]
            result_frames.append(Image.fromarray(frame_array))

        logger.info(f"Selected {len(result_frames)} keyframes from {total_frames} total frames based on pixel changes")
        return result_frames


class VitGpt2Generator(BaseCaptionGenerator):
    """ViT-GPT2 image captioning, smallest backend; captions each frame separately in english."""

    def __init__(self, model_id: str, cache_dir: str, device: Optional[str] = None):
        super().__init__(model_id, cache_dir)
        if device is None:
            self.device = get_optimal_device()
        else:
            self.device = device
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        # Register cleanup on exit
        atexit.register(self.cleanup)

    def cleanup(self):
        """Explicitly free VRAM by deleting model and clearing cache."""
        if self.model is not None:
            logger.info("Cleaning up VitGpt2 model from VRAM...")
            del self.model
            self.model = None
        if self.feature_extractor is not None:
            del self.feature_extractor
            self.feature_extractor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading image captioning model: {self.model_id} (device={self.device})...")
        try:
            self.feature_extractor = ViTImageProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            )

            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            )

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise

        logger.info("Caption model loaded")

    def generate_caption(self, frames: list[Image.Image], transcript: str = "") -> str:
        self.load_model()

        if not frames:
            return ""

        # Process each frame and generate descriptions
        try:
            frame_descriptions = []
            for i, frame in enumerate(frames):
                # Prepare inputs using feature extractor (not processor)
                pixel_values = self.feature_extractor(
                    images=frame,
                    return_tensors="pt"
                ).pixel_values.to(self.device)

                # Generate caption
                with torch.no_grad():
                    output_ids = self.model.generate(
                        pixel_values,
                        max_length=50,
                        num_beams=3,
                        early_stopping=True
                    )

                # Decode the generated text
                caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                frame_descriptions.append(f"[Frame {i+1}] {caption.strip()}")

        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise

        # Create a single merged caption describing the sequence
        if len(frame_descriptions) == 1:
            # For static content with single frame, just return the description
            merged = frame_descriptions[0].replace("[Frame 1] ", "")
        else:
            # For multiple frames, create a narrative describing the progression
            merged = "Sequence showing: " + " → ".join([desc.replace(f"[Frame {i+1}] ", "") for i, desc in enumerate(frame_descriptions)])

        # Minimal postprocessing: ensure no surrounding quotes
        return merged.strip().strip('"').strip("'")


class BlipGenerator(BaseCaptionGenerator):
    """BLIP image captioning generator.

    Uses Salesforce BLIP model for better caption quality.
    """

    def __init__(self, model_id: str, cache_dir: str, device: Optional[str] = None):
        super().__init__(model_id, cache_dir)
        if device is None:
            self.device = get_optimal_device()
        else:
            self.device = device
        self.model = None
        self.processor = None
        # Register cleanup on exit
        atexit.register(self.cleanup)

    def cleanup(self):
        """Explicitly free VRAM by deleting model and clearing cache."""
        if self.model is not None:
            logger.info("Cleaning up BLIP model from VRAM...")
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading BLIP model: {self.model_id} (device={self.device})...")
        try:
            self.processor = BlipProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            )

            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            ).to(self.device)

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load BLIP model {self.model_id}: {e}")
            raise

        logger.info("BLIP model loaded")

    def generate_caption(self, frames: list[Image.Image], transcript: str = "") -> str:
        self.load_model()

        if not frames:
            return ""

        try:
            frame_descriptions = []
            for i, frame in enumerate(frames):
                inputs = self.processor(frame, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_length=50)

                caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
                frame_descriptions.append(f"[Frame {i+1}] {caption.strip()}")

        except Exception as e:
            logger.error(f"BLIP inference error: {e}")
            raise

        # Create a single merged caption describing the sequence
        if len(frame_descriptions) == 1:
            merged = frame_descriptions[0].replace("[Frame 1] ", "")
        else:
            merged = "Sequence showing: " + " → ".join([desc.replace(f"[Frame {i+1}] ", "") for i, desc in enumerate(frame_descriptions)])

        return merged.strip().strip('"').strip("'")


class Blip2Generator(BaseCaptionGenerator):
    """BLIP-2 image captioning generator."""

    def __init__(self, model_id: str, cache_dir: str, device: Optional[str] = None):
        super().__init__(model_id, cache_dir)
        if device is None:
            self.device = get_optimal_device()
        else:
            self.device = device
        self.model = None
        self.processor = None
        # Register cleanup on exit
        atexit.register(self.cleanup)

    def cleanup(self):
        """Explicitly free VRAM by deleting model and clearing cache."""
        if self.model is not None:
            logger.info("Cleaning up BLIP-2 model from VRAM...")
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading BLIP-2 model: {self.model_id} (device={self.device})...")
        try:
            self.processor = Blip2Processor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            )

            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model {self.model_id}: {e}")
            raise

        logger.info("BLIP-2 model loaded")

    def generate_caption(self, frames: list[Image.Image], transcript: str = "") -> str:
        self.load_model()

        if not frames:
            return ""

        try:
            frame_descriptions = []
            for i, frame in enumerate(frames):
                inputs = self.processor(frame, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_length=50)

                caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
                frame_descriptions.append(f"[Frame {i+1}] {caption.strip()}")

        except Exception as e:
            logger.error(f"BLIP-2 inference error: {e}")
            raise

        # Create a single merged caption describing the sequence
        if len(frame_descriptions) == 1:
            merged = frame_descriptions[0].replace("[Frame 1] ", "")
        else:
            merged = "Sequence showing: " + " → ".join([desc.replace(f"[Frame {i+1}] ", "") for i, desc in enumerate(frame_descriptions)])

        return merged.strip().strip('"').strip("'")


class QwenVLGenerator(BaseCaptionGenerator):
    """Qwen3-VL vision-language model caption generator.

    Uses Qwen/Qwen3-VL-2B-Instruct for advanced vision-language understanding.
    """

    def __init__(self, model_id: str, cache_dir: str, device: Optional[str] = None):
        super().__init__(model_id, cache_dir)
        if device is None:
            self.device = get_optimal_device()
        else:
            self.device = device
        self.model = None
        self.processor = None
        # Register cleanup on exit
        atexit.register(self.cleanup)

    def cleanup(self):
        """Explicitly free VRAM by deleting model and clearing cache."""
        if self.model is not None:
            logger.info("Cleaning up Qwen VL model from VRAM...")
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading Qwen VL model: {self.model_id} (device={self.device})...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                dtype=torch.bfloat16,
                device_map="auto",
            )

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load Qwen VL model {self.model_id}: {e}")
            raise

        logger.info("Qwen VL model loaded")

    def generate_caption(self, frames: list[Image.Image], transcript: str = "") -> str:
        """Generate a single comprehensive caption from multiple keyframes.

        Leverages Qwen3-VL's multi-image understanding to synthesize one cohesive
        caption from all keyframes, focusing on visual changes and temporal progression.
        Uses the SYSTEM_PROMPT to guide generation toward training-appropriate descriptions.

        Args:
            frames: Keyframes in temporal order.
            transcript: Speech in the clip, passed as context; may be empty.

        Returns:
            The caption, or an empty string if there are no frames.
        """
        self.load_model()

        if not frames:
            return ""

        inputs = None
        generated_ids = None
        generated_ids_trimmed = None
        message_content = None
        messages = None

        try:
            # Build multi-frame message content
            # Qwen3-VL can process multiple images in a single conversation
            message_content = []

            # Add all frames to the content
            for frame in frames:
                message_content.append({
                    "type": "image",
                    "image": frame,
                })

            # Add comprehensive instruction with SYSTEM_PROMPT
            # This guides the model to create a single cohesive caption focusing on
            # visual changes rather than repetitively describing static elements
            instruction_text = f"{SYSTEM_PROMPT}\n\nAnalysiere die oben gezeigten Bilder. Sie zeigen eine zeitliche Abfolge aus einem Video. Erstelle einen einzelnen, zusammenhängenden Text-to-Video-Prompt, der die visuelle Entwicklung und Veränderungen über die Zeit beschreibt."
            if transcript:
                instruction_text += (
                    "\n\nTranskript des Gesprochenen in diesem Abschnitt (nur als Kontext, nicht wörtlich übernehmen):\n"
                    f"{transcript}"
                )

            message_content.append({
                "type": "text",
                "text": instruction_text,
            })

            messages = [
                {
                    "role": "user",
                    "content": message_content,
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            try:
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,  # Allow longer output for comprehensive caption
                        do_sample=False,
                    )
            except RuntimeError as e:
                if "hardware exception" in str(e).lower() or "hsa_status" in str(e).lower():
                    logger.error(f"GPU hardware exception during generation: {e}")
                    logger.error("This may be due to ROCm/dtype incompatibility. Try restarting.")
                    raise
                raise

            # Trim the input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            caption = output_text[0].strip() if output_text else ""
            return caption.strip().strip('"').strip("'")

        except Exception as e:
            logger.error(f"Qwen VL inference error: {e}")
            raise
        finally:
            # Ensure per-caption tensors are released promptly
            if inputs is not None:
                del inputs
            if generated_ids is not None:
                del generated_ids
            if generated_ids_trimmed is not None:
                del generated_ids_trimmed
            if message_content is not None:
                del message_content
            if messages is not None:
                del messages
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def cleanup_step(self):
        """Extra cleanup after each caption to avoid VRAM accumulation."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


def get_caption_generator(backend_key: Optional[str], model_id: str, cache_dir: str):
    """Factory returning an appropriate caption generator instance.

    backend_key can be one of the keys in CAPTION_MODELS ("vit-gpt2", "blip", "blip2", "qwen3-vl").
    If backend_key is None or not recognized, default to VitGpt2Generator.
    """
    key = (backend_key or "").lower()

    # Resolve common alias where model_id was supplied as a backend key
    if model_id in CAPTION_MODELS and not key:
        key = model_id

    if key == "blip":
        return BlipGenerator(model_id=model_id, cache_dir=cache_dir)
    elif key == "blip2":
        return Blip2Generator(model_id=model_id, cache_dir=cache_dir)
    elif key == "qwen3-vl":
        return QwenVLGenerator(model_id=model_id, cache_dir=cache_dir)
    else:
        return VitGpt2Generator(model_id=model_id or CAPTION_MODELS.get("vit-gpt2"), cache_dir=cache_dir)


def generate_captions_pipeline(
    metadata_path: Path | str,
    clips_dir: Path | str,
    model_id: Optional[str] = None,
    cache_dir: str = CACHE_DIR,
    max_frames: int = CAPTION_MAX_FRAMES,
    backend: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generate captions for all clips with progress updates.

    model_id may be either a backend key (e.g. 'vit-gpt2') or a full Hugging Face model id.
    If neither model_id nor backend are provided, the default configured backend is used.
    """
    metadata_path = Path(metadata_path)
    clips_dir = Path(clips_dir)

    if not metadata_path.exists():
        yield {"status": "error", "message": f"Metadata file not found: {metadata_path}"}
        return

    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        clips = [json.loads(line) for line in f]

    if not clips:
        yield {"status": "error", "message": "No clips found in metadata"}
        return

    # Resolve backend and model id
    resolved_backend = backend or CAPTION_BACKEND
    resolved_model_id = None

    # If model_id provided and matches a backend key, treat it as backend
    if model_id and model_id in CAPTION_MODELS:
        resolved_backend = model_id
        resolved_model_id = CAPTION_MODELS[model_id]
    elif model_id and ("/" in model_id or model_id.startswith("hf/")):
        # model_id looks like a HF model id
        resolved_model_id = model_id
    else:
        # Use mapping from backend key
        resolved_model_id = CAPTION_MODELS.get(resolved_backend, CAPTION_MODELS.get("vit-gpt2"))

    yield {"status": "loading", "message": f"Loading caption backend ({resolved_backend}) -> {resolved_model_id}..."}

    # Initialize generator
    generator = get_caption_generator(resolved_backend, resolved_model_id, cache_dir)

    try:
        generator.load_model()
    except Exception as e:
        yield {"status": "error", "message": f"Failed to load model: {str(e)}"}
        return

    # Optional summarizer pipeline (disabled by default)
    summarizer = None
    if CAPTION_USE_SUMMARIZER:
        try:
            summarizer = pipeline("summarization", model=CAPTION_SUMMARIZER_ID, device=(0 if torch.cuda.is_available() else -1), cache_dir=cache_dir)
            logger.info(f" Summarizer loaded: {CAPTION_SUMMARIZER_ID}")
        except Exception as e:
            logger.error(f"Failed to load summarizer {CAPTION_SUMMARIZER_ID}: {e}")
            summarizer = None

    # transcripts only help backends that take text context
    whisper_model = None
    if resolved_backend in TRANSCRIPT_BACKENDS:
        try:
            whisper_model = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_ID,
                device=(0 if torch.cuda.is_available() else -1),
                model_kwargs={"cache_dir": cache_dir},
            )
            logger.info("Whisper model loaded")
        except Exception as e:
            logger.warning(f"Failed to load Whisper model: {e}")

    captioned = 0
    try:
        for idx, clip in enumerate(clips, 1):
            video_path = clips_dir / clip["file_name"]
            # cleared until a caption succeeds, so failed clips are skipped later
            clip["prompt"] = clip["text"] = clip["audio_text"] = ""

            if not video_path.exists():
                yield {
                    "status": "processing",
                    "clip": clip["file_name"],
                    "caption": "",
                    "progress": (idx, len(clips)),
                    "message": "Video not found, skipped",
                }
                continue

            try:
                audio_text = _transcribe_clip(video_path, whisper_model) if whisper_model else ""
                frames = generator.video_to_frames(video_path, max_frames)
                caption = generator.generate_caption(frames, transcript=audio_text)

                if summarizer is not None and caption:
                    try:
                        summary = summarizer(caption, max_length=160, min_length=40, do_sample=False)
                        if isinstance(summary, list) and summary:
                            caption = summary[0].get("summary_text", caption)
                    except Exception as e:
                        logger.error(f"Summarizer error for {clip['file_name']}: {e}")

                clip["audio_text"] = audio_text
                if not caption:
                    yield {
                        "status": "processing",
                        "clip": clip["file_name"],
                        "caption": "",
                        "progress": (idx, len(clips)),
                        "message": "Empty caption, skipped",
                    }
                    continue

                clip["text"] = clip["prompt"] = caption
                captioned += 1
                yield {
                    "status": "processing",
                    "clip": clip["file_name"],
                    "caption": caption,
                    "audio_text": audio_text,
                    "progress": (idx, len(clips)),
                }

            except Exception as e:
                logger.error(f"Error generating caption for {clip['file_name']}: {e}")
                yield {
                    "status": "processing",
                    "clip": clip["file_name"],
                    "caption": "",
                    "progress": (idx, len(clips)),
                    "message": f"Error: {e}",
                }
            finally:
                if hasattr(generator, "cleanup_step"):
                    generator.cleanup_step()
                # saved per clip so an interrupted run keeps finished captions
                _write_metadata(metadata_path, clips)

        logger.info(f"Finished! {captioned}/{len(clips)} clips captioned.")
        yield {"status": "complete", "total_clips": len(clips), "captioned": captioned}

    finally:
        generator.cleanup()


def _transcribe_clip(video_path: Path, whisper_model: Any) -> str:
    """Extract a clip's audio to a temporary wav and transcribe it; empty string on failure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "audio.wav"
        if not extract_audio_from_video(video_path, wav_path):
            logger.warning(f"Failed to extract audio from {video_path.name}")
            return ""
        return transcribe_audio_whisper(wav_path, whisper_model)


def _write_metadata(metadata_path: Path, clips: list[dict[str, Any]]) -> None:
    """Rewrite the metadata jsonl atomically."""
    tmp_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for clip in clips:
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")
    tmp_path.replace(metadata_path)
