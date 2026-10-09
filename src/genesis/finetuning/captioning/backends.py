"""Caption backends: Qwen3-VL captions a whole clip, the others caption single frames."""
import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
    Qwen3VLForConditionalGeneration,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

from genesis.config import CAPTION_MAX_NEW_TOKENS, CAPTION_MODELS
from genesis.device import optimal_device
from genesis.finetuning.captioning.prompts import build_instruction

logger = logging.getLogger(__name__)


class CaptionBackend(ABC):
    """Lazily loaded captioning model.

    Subclasses set `components` to the attribute names that hold loaded objects, so
    `unload` can free them uniformly.
    """

    uses_transcript = False
    components: tuple[str, ...] = ("model", "processor")

    def __init__(self, model_id: str, cache_dir: Path | str, device: str | None = None):
        self.model_id = model_id
        self.cache_dir = str(cache_dir)
        self.device = device or optimal_device()
        self.model = None

    def load(self) -> None:
        """Load the model if needed."""
        if self.model is None:
            logger.info("Loading caption model %s on %s", self.model_id, self.device)
            self._load()
            self.model.eval()

    def unload(self) -> None:
        """Drop all loaded components and free GPU memory."""
        for name in self.components:
            setattr(self, name, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def caption(self, frames: list[Image.Image], transcript: str = "") -> str:
        """Caption the keyframes of one clip; empty string if there are none.

        Args:
            frames: Keyframes in temporal order.
            transcript: Speech in the clip; only used if `uses_transcript` is set.
        """
        if not frames:
            return ""
        self.load()
        with torch.inference_mode():
            text = self._caption(frames, transcript)
        return text.strip().strip("\"'")

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def _caption(self, frames: list[Image.Image], transcript: str) -> str: ...


class SingleFrameBackend(CaptionBackend):
    """Captions each keyframe separately and joins them into one sequence description."""

    max_length = 50

    def _caption(self, frames: list[Image.Image], transcript: str) -> str:
        captions = [self._caption_frame(frame).strip() for frame in frames]
        if len(captions) == 1:
            return captions[0]
        return "Sequence showing: " + " -> ".join(captions)

    @abstractmethod
    def _caption_frame(self, frame: Image.Image) -> str: ...


class VitGpt2Backend(SingleFrameBackend):
    """ViT-GPT2, the smallest backend; english captions."""

    components = ("model", "processor", "tokenizer")

    def _load(self) -> None:
        self.processor = ViTImageProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device)

    def _caption_frame(self, frame: Image.Image) -> str:
        pixels = self.processor(images=frame, return_tensors="pt").pixel_values.to(self.device)
        ids = self.model.generate(pixels, max_length=self.max_length, num_beams=3, early_stopping=True)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


class BlipBackend(SingleFrameBackend):
    """BLIP base; english captions, better than ViT-GPT2 at similar cost."""

    processor_class = BlipProcessor
    model_class = BlipForConditionalGeneration
    dtype = torch.float32

    def _load(self) -> None:
        self.processor = self.processor_class.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = self.model_class.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, torch_dtype=self.dtype
        ).to(self.device)

    def _caption_frame(self, frame: Image.Image) -> str:
        inputs = self.processor(frame, return_tensors="pt").to(self.device, self.dtype)
        ids = self.model.generate(**inputs, max_length=self.max_length)
        return self.processor.decode(ids[0], skip_special_tokens=True)


class Blip2Backend(BlipBackend):
    """BLIP-2 (OPT 2.7B); strongest single-frame backend, needs about 8 GB VRAM in fp16."""

    processor_class = Blip2Processor
    model_class = Blip2ForConditionalGeneration
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32


class QwenVLBackend(CaptionBackend):
    """Qwen3-VL; one German prompt from all keyframes plus the transcript."""

    uses_transcript = True

    def _load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, dtype=torch.bfloat16, device_map="auto"
        )

    def _caption(self, frames: list[Image.Image], transcript: str) -> str:
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": build_instruction(transcript)})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        ids = self.model.generate(**inputs, max_new_tokens=CAPTION_MAX_NEW_TOKENS, do_sample=False)
        # drop the prompt tokens from the output
        new_ids = ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]


BACKENDS: dict[str, type[CaptionBackend]] = {
    "qwen3-vl": QwenVLBackend,
    "vit-gpt2": VitGpt2Backend,
    "blip": BlipBackend,
    "blip2": Blip2Backend,
}


def create_backend(name: str, cache_dir: Path | str, model_id: str | None = None) -> CaptionBackend:
    """Instantiate a backend by name, with its default model unless `model_id` is given.

    Raises:
        ValueError: Unknown backend name.
    """
    if name not in BACKENDS:
        raise ValueError(f"Unknown caption backend {name!r}, expected one of {list(BACKENDS)}")
    return BACKENDS[name](model_id or CAPTION_MODELS[name], cache_dir)
