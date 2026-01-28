import json
import yaml
from pathlib import Path
import cv2
from PIL import Image
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import logging
from typing import Generator, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CaptionGen")


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


class CaptionGenerator:
    """Lazy-loading caption generator using Llama Vision."""

    def __init__(self, model_id: str, cache_dir: str):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None

    def load_model(self):
        """Load model and processor (lazy initialization)."""
        if self.model is not None:
            return

        logger.info(f"Loading {self.model_id}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir
        )
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=False,
        )
        logger.info("✓ Caption model loaded")

    def video_to_frames(self, video_path: Path, max_frames: int = 8) -> list[Image.Image]:
        """Extracts evenly distributed frames from a video."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []

        frames = []
        indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))

        cap.release()
        return frames

    def generate_caption(self, frames: list[Image.Image]) -> str:
        """Generate caption from video frames."""
        self.load_model()  # Ensure model is loaded

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"} for _ in frames
                ] + [{"type": "text", "text": SYSTEM_PROMPT}]
            }
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(frames, input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                temperature=0.1,
            )

        result = self.processor.decode(output[0], skip_special_tokens=True)
        return result.strip().strip('"').strip("'")


def generate_captions_pipeline(
    metadata_path: Path | str,
    clips_dir: Path | str,
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    cache_dir: str = "/content/drive/MyDrive/Genesis/models/llama",
    max_frames: int = 8,
) -> Generator[Dict[str, Any], None, None]:
    """
    Generate captions for all clips with progress updates.

    Yields:
        Progress updates with format:
        {"status": "loading", "message": "Loading model..."}
        {"status": "processing", "clip": filename, "caption": str, "progress": (current, total)}
        {"status": "complete", "total_clips": int}
        {"status": "error", "message": str}
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

    yield {"status": "loading", "message": f"Loading caption model ({model_id})..."}

    # Initialize generator
    generator = CaptionGenerator(model_id, cache_dir)

    try:
        generator.load_model()
    except Exception as e:
        yield {"status": "error", "message": f"Failed to load model: {str(e)}"}
        return

    # Process clips
    for idx, clip in enumerate(clips, 1):
        video_path = clips_dir / clip["file_name"]

        if not video_path.exists():
            clip["text"] = "white screen with text"
            yield {
                "status": "processing",
                "clip": clip["file_name"],
                "caption": clip["text"],
                "progress": (idx, len(clips)),
                "message": f"Video not found, using fallback caption"
            }
            continue

        try:
            frames = generator.video_to_frames(video_path, max_frames)
            caption = generator.generate_caption(frames)
            clip["text"] = caption

            yield {
                "status": "processing",
                "clip": clip["file_name"],
                "caption": caption,
                "progress": (idx, len(clips))
            }

        except Exception as e:
            logger.error(f"Error generating caption for {clip['file_name']}: {e}")
            clip["text"] = "error generating caption"
            yield {
                "status": "processing",
                "clip": clip["file_name"],
                "caption": clip["text"],
                "progress": (idx, len(clips)),
                "message": f"Error: {str(e)}"
            }

    # Save updated metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        for clip in clips:
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")

    logger.info(f"Finished! {len(clips)} clips now have captions.")
    yield {"status": "complete", "total_clips": len(clips)}


# Legacy main for backward compatibility
def main():
    from src.core.config import METADATA_FILE, CUT_VIDEOS_DIR, CAPTION_MODEL_ID, CACHE_DIR

    for update in generate_captions_pipeline(
        METADATA_FILE,
        CUT_VIDEOS_DIR,
        model_id=CAPTION_MODEL_ID,
        cache_dir=CACHE_DIR + "/llama"
    ):
        if update["status"] == "error":
            logger.error(update["message"])
        elif update["status"] == "processing":
            logger.info(f"[{update['progress'][0]}/{update['progress'][1]}] {update['clip']}: {update['caption']}")


if __name__ == "__main__":
    main()

