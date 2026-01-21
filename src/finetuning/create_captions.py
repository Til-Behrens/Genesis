import json
import yaml
from pathlib import Path
import cv2
from PIL import Image
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CaptionGen")

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
CACHE_DIR = "/content/drive/MyDrive/Genesis/models/llama"
CLIPS_DIR = Path("src/finetuning/dataset/cut_videos")
METADATA_FILE = Path("src/finetuning/dataset/metadata.jsonl")

logger.info("Loading Llama 3.2 11B Vision")
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR,
    local_files_only=False,
)

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


def video_to_frames(video_path: Path, max_frames: int = 8) -> list[Image.Image]:
    """Extracts evenly distributed frames from an 8s video."""
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


def generate_caption(frames: list[Image.Image]) -> str:

    messages = [
        {
            "role": "user",
            "content": [
                           {"type": "image"} for _ in frames
                       ] + [{"type": "text", "text": SYSTEM_PROMPT}]
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(frames, input_text, return_tensors="pt").to(model.device)

    # Generate (deterministically for consistent captions)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.1,
        )

    result = processor.decode(output[0], skip_special_tokens=True)
    return result.strip().strip('"').strip("'")


def main():
    logger.info("Start caption generation with Llama 3.2 Vision...")

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        clips = [json.loads(line) for line in f]

    for clip in tqdm(clips, desc="generate captions"):
        video_path = CLIPS_DIR / clip["file_name"]
        if not video_path.exists():
            clip["text"] = "white screen with text"
            continue

        frames = video_to_frames(video_path)
        caption = generate_caption(frames)
        clip["text"] = caption

        logger.info(f"\n{clip['file_name']}")
        logger.info(f"→ {caption}")

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        for clip in clips:
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")

    logger.info(f"Finished! {len(clips)} Clips now have perfect llama captions.")


if __name__ == "__main__":
    main()