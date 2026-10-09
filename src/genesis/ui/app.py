"""Gradio interface: video generation and the four-step fine-tuning workflow."""
import logging
import random
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import gradio as gr

from genesis.config import (
    CAPTION_BACKEND,
    CAPTION_MODELS,
    CLIP_LENGTH_SEC,
    CLIP_OVERLAP_SEC,
    CLIPS_DIR,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_TRAINING_MODEL,
    DIFFSYNTH_METADATA_FILE,
    DIFFSYNTH_ROOT,
    LORA_OUTPUT_DIR,
    MAX_DURATION_SEC,
    METADATA_FILE,
    RAW_VIDEOS_DIR,
    WAN_MODELS,
)
from genesis.device import get_device_info
from genesis.events import Event
from genesis.finetuning.captioning import generate_captions
from genesis.finetuning.cut_videos import cut_videos, ffmpeg_available
from genesis.finetuning.dataset import prepare_dataset
from genesis.finetuning.training import TrainingOptions, train_lora
from genesis.jobs import JobBusyError, job_lock

logger = logging.getLogger(__name__)

LOG_LINES = 30
DEFAULT_PROMPT = (
    "A powerpoint-like presentation title slide about machine learning with white background and black text."
)


def run_job(job: str, start: Callable[[], Iterator[Event]], free_gpu: bool = False) -> Iterator[str]:
    """Run a pipeline under the job lock and stream its accumulated log to the ui.

    Args:
        job: Name shown while the job holds the lock.
        start: Creates the pipeline's event iterator.
        free_gpu: Unload the generation model first; captioning and training need the VRAM.
    """
    lines: list[str] = []
    try:
        with job_lock.acquire(job):
            if free_gpu:
                from genesis.generation import unload_pipeline
                unload_pipeline()
            for event in start():
                lines = (lines + [str(event)])[-LOG_LINES:]
                yield "\n".join(lines)
    except JobBusyError as e:
        yield str(e)
    except Exception as e:
        logger.exception("%s failed", job)
        yield "\n".join(lines + [f"Error: {e}"])


def _path(value: str, default: Path) -> Path:
    return Path(value) if value and value.strip() else default


def generate_video(prompt: str, duration: float, model_size: str, seed: float) -> tuple[str | None, str]:
    """Generate one video; returns the file path (or None) and a status message."""
    try:
        with job_lock.acquire("video generation"):
            from genesis.generation import get_pipeline
            pipeline = get_pipeline(model_size)
            path = pipeline.generate(prompt, duration, seed=None if seed < 0 else int(seed))
            return str(path), f"Saved to {path}"
    except JobBusyError as e:
        return None, str(e)
    except Exception as e:
        logger.exception("Generation failed")
        return None, f"Error: {e}"


def gpu_status() -> str:
    """One line with device, VRAM and the running job."""
    device = get_device_info()
    if not device.is_gpu:
        return "GPU not available"
    import torch
    used = torch.cuda.memory_allocated(0) / 1e9
    job = job_lock.active_job
    state = f"Running: {job} ({job_lock.elapsed:.0f}s)" if job else "Idle"
    return f"{device.name} | VRAM {used:.1f}/{device.vram_gb:.1f} GB | {state}"


def _generation_tab() -> None:
    gr.Markdown("### Generate tutorial videos from text prompts")
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                value=DEFAULT_PROMPT,
            )
        with gr.Column(scale=1):
            model = gr.Dropdown(label="Model", choices=list(WAN_MODELS), value=DEFAULT_GENERATION_MODEL)
            duration = gr.Slider(label="Duration (seconds)", minimum=1, maximum=MAX_DURATION_SEC, value=5, step=1)
            with gr.Row():
                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                reroll = gr.Button("Random seed", size="sm")
            button = gr.Button("Generate Video", variant="primary", size="lg")
    status = gr.Textbox(label="Status", interactive=False)
    video = gr.Video(label="Generated Video")

    reroll.click(fn=lambda: random.randint(0, 2**31 - 1), outputs=seed)
    button.click(fn=generate_video, inputs=[prompt, duration, model, seed], outputs=[video, status])


def _cut_step() -> None:
    gr.Markdown("Split long recordings into overlapping training clips")
    with gr.Row():
        input_dir = gr.Textbox(label="Input Directory (.mp4)", value=str(RAW_VIDEOS_DIR))
        output_dir = gr.Textbox(label="Clips Directory", value=str(CLIPS_DIR))
    with gr.Row():
        length = gr.Slider(label="Clip Length (seconds)", minimum=4, maximum=16, value=CLIP_LENGTH_SEC, step=1)
        overlap = gr.Slider(label="Overlap (seconds)", minimum=0, maximum=4, value=CLIP_OVERLAP_SEC, step=1)
    gr.Markdown(f"**ffmpeg:** {'available' if ffmpeg_available() else 'not found, install ffmpeg first'}")
    button = gr.Button("Cut Videos", variant="primary")
    log = gr.Textbox(label="Log", lines=10, interactive=False)

    def run(source: str, target: str, clip_length: float, clip_overlap: float) -> Iterator[str]:
        yield from run_job("video cutting", lambda: cut_videos(
            _path(source, RAW_VIDEOS_DIR), _path(target, CLIPS_DIR), METADATA_FILE,
            clip_length=int(clip_length), overlap=int(clip_overlap),
        ))

    button.click(fn=run, inputs=[input_dir, output_dir, length, overlap], outputs=log)


def _caption_step() -> None:
    gr.Markdown("Write a text prompt for each clip; failed clips stay empty and are skipped in Step 3")
    with gr.Row():
        clips_dir = gr.Textbox(label="Clips Directory", value=str(CLIPS_DIR))
        metadata = gr.Textbox(label="Metadata File", value=str(METADATA_FILE))
    backend = gr.Dropdown(
        label="Caption Backend",
        choices=[
            (f"qwen3-vl ({CAPTION_MODELS['qwen3-vl']}) - all keyframes and speech, German", "qwen3-vl"),
            (f"vit-gpt2 ({CAPTION_MODELS['vit-gpt2']}) - smallest, per frame", "vit-gpt2"),
            (f"blip ({CAPTION_MODELS['blip']}) - per frame", "blip"),
            (f"blip2 ({CAPTION_MODELS['blip2']}) - strongest per frame", "blip2"),
        ],
        value=CAPTION_BACKEND,
    )
    button = gr.Button("Generate Captions", variant="primary")
    log = gr.Textbox(label="Log", lines=15, interactive=False)

    def run(clips: str, meta: str, name: str) -> Iterator[str]:
        yield from run_job("captioning", lambda: generate_captions(
            _path(meta, METADATA_FILE), _path(clips, CLIPS_DIR), backend=name,
        ), free_gpu=True)

    button.click(fn=run, inputs=[clips_dir, metadata, backend], outputs=log)


def _dataset_step() -> None:
    gr.Markdown("Drop missing and uncaptioned clips and write the training metadata")
    with gr.Row():
        clips_dir = gr.Textbox(label="Clips Directory", value=str(CLIPS_DIR))
        metadata = gr.Textbox(label="Metadata File", value=str(METADATA_FILE))
        output = gr.Textbox(label="Training Metadata (.jsonl)", value=str(DIFFSYNTH_METADATA_FILE))
    button = gr.Button("Prepare Dataset", variant="primary")
    log = gr.Textbox(label="Log", lines=12, interactive=False)

    def run(clips: str, meta: str, target: str) -> Iterator[str]:
        yield from run_job("dataset preparation", lambda: prepare_dataset(
            _path(clips, CLIPS_DIR), _path(meta, METADATA_FILE), _path(target, DIFFSYNTH_METADATA_FILE),
        ))

    button.click(fn=run, inputs=[clips_dir, metadata, output], outputs=log)


def _training_step() -> None:
    gr.Markdown("Fine-tune a LoRA adapter with DiffSynth-Studio")
    defaults = TrainingOptions()
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(
                label="Base Model",
                choices=[(f"{key} ({spec.model_id})", key) for key, spec in WAN_MODELS.items()],
                value=DEFAULT_TRAINING_MODEL,
            )
            clips_dir = gr.Textbox(label="Clips Directory", value=str(CLIPS_DIR))
            metadata = gr.Textbox(label="Training Metadata (from Step 3)", value=str(DIFFSYNTH_METADATA_FILE))
            diffsynth_root = gr.Textbox(
                label="DiffSynth-Studio Clone (optional)",
                value=DIFFSYNTH_ROOT,
                placeholder="found automatically for editable installs",
            )
        with gr.Column():
            epochs = gr.Slider(label="Epochs", minimum=1, maximum=20, value=defaults.epochs, step=1)
            grad_accum = gr.Slider(
                label="Gradient Accumulation", minimum=1, maximum=16, value=defaults.gradient_accumulation_steps, step=1
            )
            learning_rate = gr.Number(label="Learning Rate", value=defaults.learning_rate, precision=6)
            rank = gr.Slider(label="LoRA Rank", minimum=4, maximum=128, value=defaults.lora_rank, step=4)
    output_dir = gr.Textbox(label="Output Directory (empty = new run folder)", value="")
    button = gr.Button("Start Training", variant="primary", size="lg")
    log = gr.Textbox(label="Log", lines=15, interactive=False)

    def run(size: str, clips: str, meta: str, root: str, n_epochs: float, accum: float, lr: float,
            lora_rank: float, target: str) -> Iterator[str]:
        options = TrainingOptions(
            epochs=int(n_epochs), gradient_accumulation_steps=int(accum),
            learning_rate=float(lr), lora_rank=int(lora_rank),
        )
        run_dir = LORA_OUTPUT_DIR / f"{size}_{datetime.now():%Y%m%d_%H%M%S}"
        yield from run_job("lora training", lambda: train_lora(
            size, _path(clips, CLIPS_DIR), _path(meta, DIFFSYNTH_METADATA_FILE), _path(target, run_dir),
            options=options, diffsynth_root=root.strip() or None,
        ), free_gpu=True)

    button.click(
        fn=run,
        inputs=[model, clips_dir, metadata, diffsynth_root, epochs, grad_accum, learning_rate, rank, output_dir],
        outputs=log,
    )


def build_ui() -> gr.Blocks:
    """Assemble the interface."""
    with gr.Blocks(title="Genesis") as app:
        gr.Markdown("# Genesis\nText-to-video generation and LoRA fine-tuning for tutorial videos")
        with gr.Row():
            status = gr.Textbox(label="GPU", value=gpu_status, interactive=False, scale=4)
            refresh = gr.Button("Refresh", size="sm", scale=1)
        refresh.click(fn=gpu_status, outputs=status)

        with gr.Tab("Video Generation"):
            _generation_tab()
        with gr.Tab("Fine-tuning"):
            gr.Markdown("Run the steps in order. Only one job runs at a time.")
            for title, step in (
                ("Step 1: Cut Videos", _cut_step),
                ("Step 2: Generate Captions", _caption_step),
                ("Step 3: Prepare Dataset", _dataset_step),
                ("Step 4: Train LoRA", _training_step),
            ):
                with gr.Accordion(title, open=step is _cut_step):
                    step()
    return app


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    """Build and serve the interface; blocks until stopped."""
    app = build_ui()
    app.queue()
    app.launch(server_name=host, server_port=port, share=share, show_error=True, theme=gr.themes.Soft())
