"""
Unified Genesis GUI: Video Generation + Finetuning Pipeline
"""
import gradio as gr
import torch
from pathlib import Path
import logging
from datetime import datetime
import traceback

from src.core.genesis_pipeline import get_genesis_pipeline
from src.core.job_manager import job_manager
from src.core.config import (
    RAW_VIDEOS_DIR, CUT_VIDEOS_DIR, METADATA_FILE,
    PREPROCESSED_LATENTS_DIR, LORA_CHECKPOINTS_DIR,
    CACHE_DIR, CAPTION_MODELS, CAPTION_BACKEND,
    DEFAULT_TRAINING_CONFIG, VIDEO_CONFIG, WAN_MODEL_MAP
)
from src.finetuning.cut_videos import cut_videos_pipeline, check_ffmpeg_available
from src.finetuning.create_captions import generate_captions_pipeline
from src.finetuning.preprocess_dataset import preprocess_videos_to_latents, clear_cache
from src.finetuning.train_lora import train_lora_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GenesisUI")

# Suppress verbose logging from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ============================================================================
# GENERATION TAB
# ============================================================================

def generate_video(prompt: str, duration_seconds: float, model_size: str, seed: int = -1):
    """Generate video with job locking."""
    try:
        with job_manager.acquire_gpu("video_generation", timeout=2.0):
            logger.info(f"Generating video: {prompt[:50]}...")
            pipeline = get_genesis_pipeline(model_size)
            # Use None for seed if -1 (random), otherwise use the specified seed
            seed_value = None if seed == -1 else seed
            output_path = pipeline.generate(prompt, int(duration_seconds), seed=seed_value)
            return output_path, f"✓ Video generated successfully!"
    except RuntimeError as e:
        return None, f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return None, f"❌ Error: {str(e)}"


# ============================================================================
# FINETUNING TAB - CUT VIDEOS
# ============================================================================

def cut_videos_ui(input_dir: str, output_dir: str, clip_length: int, overlap: int):
    """Cut videos with progress updates."""
    try:
        with job_manager.acquire_gpu("video_cutting", timeout=2.0):
            input_path = Path(input_dir) if input_dir else RAW_VIDEOS_DIR
            output_path = Path(output_dir) if output_dir else CUT_VIDEOS_DIR

            if not input_path.exists():
                yield f"❌ Input directory not found: {input_path}", ""
                return

            log_messages = []

            for update in cut_videos_pipeline(
                input_path,
                output_path,
                METADATA_FILE,
                clip_length=clip_length,
                overlap=overlap,
                target_width=VIDEO_CONFIG["target_width"],
                target_height=VIDEO_CONFIG["target_height"],
                target_fps=VIDEO_CONFIG["target_fps"],
            ):
                if update["status"] == "error":
                    log_messages.append(f"❌ {update['message']}")
                    yield "\n".join(log_messages), ""
                    return

                elif update["status"] == "processing":
                    msg = f"📹 {update['video']}: {update['clips_created']} clips created (Total: {update['total_clips']})"
                    log_messages.append(msg)
                    yield "\n".join(log_messages), ""

                elif update["status"] == "complete":
                    msg = f"✓ Complete! {update['total_clips']} clips created\n📁 Metadata: {update['metadata_path']}"
                    log_messages.append(msg)
                    yield "\n".join(log_messages), str(output_path)

    except RuntimeError as e:
        yield f"❌ {str(e)}", ""
    except Exception as e:
        logger.error(f"Cut videos error: {e}", exc_info=True)
        yield f"❌ Error: {str(e)}", ""


# ============================================================================
# FINETUNING TAB - GENERATE CAPTIONS
# ============================================================================

def generate_captions_ui(clips_dir: str, metadata_path: str, backend: str):
    """Generate captions with progress updates."""
    try:
        with job_manager.acquire_gpu("caption_generation", timeout=2.0):
            clips_path = Path(clips_dir) if clips_dir else CUT_VIDEOS_DIR
            meta_path = Path(metadata_path) if metadata_path else METADATA_FILE

            if not meta_path.exists():
                yield f"❌ Metadata file not found: {meta_path}"
                return

            log_messages = []

            # Resolve model_id from backend key
            model_id = CAPTION_MODELS.get(backend, CAPTION_MODELS.get("vit-gpt2"))

            for update in generate_captions_pipeline(
                meta_path,
                clips_path,
                model_id=model_id,
                cache_dir=str(CACHE_DIR) + "/captions",
                backend=backend,
            ):
                if update["status"] == "error":
                    log_messages.append(f"❌ {update['message']}")
                    yield "\n".join(log_messages)
                    return

                elif update["status"] == "loading":
                    msg = f"⏳ {update['message']}"
                    log_messages.append(msg)
                    yield "\n".join(log_messages)

                elif update["status"] == "processing":
                    progress = update["progress"]
                    msg = f"[{progress[0]}/{progress[1]}] {update['clip'][:50]}"
                    if "message" in update:
                        msg += f" - {update['message']}"
                    else:
                        msg += f"\n   → {update['caption'][:100]}..."
                    log_messages.append(msg)

                    # Keep only last 20 messages for readability
                    if len(log_messages) > 20:
                        log_messages = log_messages[-20:]

                    yield "\n".join(log_messages)

                elif update["status"] == "complete":
                    msg = f"✓ Complete! {update['total_clips']} clips captioned"
                    log_messages.append(msg)
                    yield "\n".join(log_messages)

    except RuntimeError as e:
        yield f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"Caption generation error: {e}", exc_info=True)
        yield f"❌ Error: {str(e)}"


# ============================================================================
# FINETUNING TAB - PREPROCESS DATASET
# ============================================================================

def preprocess_dataset_ui(videos_dir: str, metadata_path: str, cache_dir: str):
    """Preprocess videos to VAE latents."""
    try:
        with job_manager.acquire_gpu("preprocessing", timeout=2.0):
            videos_path = Path(videos_dir) if videos_dir else CUT_VIDEOS_DIR
            meta_path = Path(metadata_path) if metadata_path else METADATA_FILE
            cache_path = Path(cache_dir) if cache_dir else PREPROCESSED_LATENTS_DIR

            # Load VAE
            yield "⏳ Loading VAE for encoding..."
            from diffusers import AutoencoderKLWan
            from src.core.config import WAN_MODEL_MAP

            # Use the 5B model's VAE
            model_id = WAN_MODEL_MAP["5B"]

            # Determine device and dtype for optimal performance
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Use bfloat16 on GPU for faster encoding, float32 on CPU
            dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

            # Load VAE from vae subfolder and move to GPU
            vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=dtype,
                cache_dir=CACHE_DIR
            ).to(device)

            log_messages = [f"✓ VAE loaded on {device.upper()} ({dtype})"]

            for update in preprocess_videos_to_latents(
                videos_path,
                meta_path,
                cache_path,
                vae,
                device=device,
                num_frames=VIDEO_CONFIG["clip_length"] * VIDEO_CONFIG["target_fps"],
                target_height=VIDEO_CONFIG["target_height"],
                target_width=VIDEO_CONFIG["target_width"],
            ):
                if update["status"] == "error":
                    log_messages.append(f"❌ {update['message']}")
                    yield "\n".join(log_messages)
                    return

                elif update["status"] == "processing":
                    progress = update["progress"]
                    msg = f"[{progress[0]}/{progress[1]}] {update['video']}: {update['message']}"
                    log_messages.append(msg)

                    # Keep only last 15 messages
                    if len(log_messages) > 15:
                        log_messages = log_messages[-15:]

                    yield "\n".join(log_messages)

                elif update["status"] == "complete":
                    msg = f"✓ Complete! {update['total_processed']} videos preprocessed\n📁 Cache: {update['cache_dir']}"
                    log_messages.append(msg)
                    yield "\n".join(log_messages)

    except RuntimeError as e:
        yield f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"Preprocessing error: {e}", exc_info=True)
        yield f"❌ Error: {str(e)}"


def clear_cache_ui():
    """Clear preprocessed latents cache."""
    try:
        count = clear_cache(PREPROCESSED_LATENTS_DIR)
        return f"✓ Cleared {count} cached files from {PREPROCESSED_LATENTS_DIR}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# FINETUNING TAB - TRAIN LORA
# ============================================================================

def train_lora_ui(
    model_id: str,
    dataset_dir: str,
    use_cached_latents: bool,
    latents_dir: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
):
    """Train LoRA with progress updates."""
    try:
        with job_manager.acquire_gpu("lora_training", timeout=2.0):
            dataset_path = Path(dataset_dir) if dataset_dir else Path("src/finetuning/dataset")
            latents_path = Path(latents_dir) if latents_dir else PREPROCESSED_LATENTS_DIR
            output_path = Path(output_dir) if output_dir else LORA_CHECKPOINTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            log_messages = []
            current_loss = None

            def progress_callback(logs):
                nonlocal current_loss
                if "loss" in logs:
                    current_loss = logs["loss"]

            for update in train_lora_pipeline(
                model_id=model_id,
                dataset_path=dataset_path,
                use_cached_latents=use_cached_latents,
                latents_cache_dir=latents_path if use_cached_latents else None,
                output_dir=output_path,
                cache_dir=CACHE_DIR,
                epochs=epochs,
                batch_size=batch_size,
                grad_accum_steps=grad_accum,
                learning_rate=learning_rate,
                progress_callback=progress_callback,
                **DEFAULT_TRAINING_CONFIG,
            ):
                if update["status"] == "error":
                    log_messages.append(f"❌ {update['message']}")
                    yield "\n".join(log_messages), None
                    return

                elif update["status"] == "init":
                    msg = f"🚀 {update['message']}\n   Precision: {update['precision']} | Device: {update['device']}"
                    log_messages.append(msg)
                    yield "\n".join(log_messages), None

                elif update["status"] == "loading":
                    log_messages.append(f"⏳ {update['message']}")
                    yield "\n".join(log_messages), None

                elif update["status"] == "training":
                    if "loss" in update:
                        msg = f"Step {update['step']}: loss = {update['loss']:.4f}"
                        log_messages.append(msg)

                        # Keep only last 10 messages
                        if len(log_messages) > 10:
                            log_messages = log_messages[-10:]
                    elif "message" in update:
                        log_messages.append(update['message'])

                    yield "\n".join(log_messages), current_loss

                elif update["status"] == "saving":
                    log_messages.append(f"💾 {update['message']}")
                    yield "\n".join(log_messages), current_loss

                elif update["status"] == "complete":
                    msg = f"✓ {update['message']}\n📁 LoRA weights: {update['output_dir']}"
                    log_messages.append(msg)
                    yield "\n".join(log_messages), current_loss

    except RuntimeError as e:
        yield f"❌ {str(e)}", None
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        yield f"❌ Error: {str(e)}\n{traceback.format_exc()}", None


# ============================================================================
# GPU STATUS
# ============================================================================

def get_gpu_status():
    """Get current GPU status."""
    from src.core.config import get_device_info

    status = job_manager.get_status()
    device_type, device_name, vram_gb = get_device_info()

    if device_type == "cpu":
        return "❌ GPU not available"

    try:
        vram_used = torch.cuda.memory_allocated(0) / 1e9
    except:
        vram_used = 0.0

    status_text = f"🖥️ {device_name} | VRAM: {vram_used:.1f}/{vram_gb:.1f} GB"

    if status["active"]:
        duration = int(status["duration"])
        status_text += f"\n🔒 Locked: {status['job_type']} ({duration}s)"
    else:
        status_text += "\n✓ Available"

    return status_text


# ============================================================================
# BUILD GRADIO UI
# ============================================================================

def build_ui():
    """Build the unified Gradio interface."""

    with gr.Blocks(title="Genesis - Video Generation & Finetuning") as app:
        gr.Markdown("# 🎬 Genesis - Video Generation & Finetuning Pipeline")
        gr.Markdown("Generate educational videos or fine-tune Wan2.2 on your own tutorial videos")

        # GPU Status
        with gr.Row():
            gpu_status = gr.Textbox(
                label="GPU Status",
                value=get_gpu_status(),
                interactive=False,
                lines=2
            )
            refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")

        refresh_status_btn.click(fn=get_gpu_status, outputs=gpu_status)

        # Main tabs
        with gr.Tabs() as tabs:

            # ================================================================
            # GENERATION TAB
            # ================================================================
            with gr.Tab("🎥 Video Generation"):
                gr.Markdown("### Generate tutorial videos from text prompts")

                with gr.Row():
                    with gr.Column(scale=2):
                        gen_prompt = gr.Textbox(
                            label="Prompt",
                            lines=4,
                            value="A powerpoint-like presentation title slide about machine learning with white background and black text.",
                            placeholder="Describe the video you want to generate..."
                        )
                    with gr.Column(scale=1):
                        gen_model = gr.Dropdown(
                            label="Model Size",
                            choices=["5B", "14B", "14B-2.1", "1.3B"],
                            value="5B"
                        )
                        gen_duration = gr.Slider(
                            label="Duration (seconds)",
                            minimum=1,
                            maximum=30,
                            value=8,
                            step=1
                        )
                        with gr.Row():
                            gen_seed = gr.Number(
                                label="Seed (-1 = random)",
                                value=-1,
                                precision=0
                            )
                            randomize_seed = gr.Button("🎲", size="sm")
                        gen_button = gr.Button("🎬 Generate Video", variant="primary", size="lg")

                gen_status = gr.Textbox(label="Status", interactive=False)
                gen_output = gr.Video(label="Generated Video")

                # Randomize seed button
                def randomize():
                    import random
                    return random.randint(0, 2**31 - 1)

                randomize_seed.click(fn=randomize, outputs=gen_seed)

                gen_button.click(
                    fn=generate_video,
                    inputs=[gen_prompt, gen_duration, gen_model, gen_seed],
                    outputs=[gen_output, gen_status]
                )

            # ================================================================
            # FINETUNING TAB
            # ================================================================
            with gr.Tab("🔧 Finetuning Pipeline"):
                gr.Markdown("### Fine-tune Wan2.2 on your tutorial videos")
                gr.Markdown("Follow the steps in order: Cut Videos → Generate Captions → Preprocess → Train LoRA")

                with gr.Accordion("Step 1: Cut Videos", open=True):
                    gr.Markdown("Split long tutorial videos into 8-second training clips")

                    with gr.Row():
                        cut_input_dir = gr.Textbox(
                            label="Input Directory",
                            value=str(RAW_VIDEOS_DIR),
                            placeholder="Directory containing raw .mp4 videos"
                        )
                        cut_output_dir = gr.Textbox(
                            label="Output Directory",
                            value=str(CUT_VIDEOS_DIR),
                            placeholder="Where to save cut clips"
                        )

                    with gr.Row():
                        cut_length = gr.Slider(label="Clip Length (seconds)", minimum=4, maximum=16, value=8, step=1)
                        cut_overlap = gr.Slider(label="Overlap (seconds)", minimum=0, maximum=4, value=2, step=1)

                    # Check ffmpeg on load
                    ffmpeg_status = "✓ ffmpeg available" if check_ffmpeg_available() else "❌ ffmpeg not found - please install ffmpeg"
                    gr.Markdown(f"**Status:** {ffmpeg_status}")

                    cut_button = gr.Button("✂️ Cut Videos", variant="primary")
                    cut_log = gr.Textbox(label="Progress Log", lines=10, interactive=False)
                    cut_result_dir = gr.Textbox(label="Output Directory", interactive=False)

                    cut_button.click(
                        fn=cut_videos_ui,
                        inputs=[cut_input_dir, cut_output_dir, cut_length, cut_overlap],
                        outputs=[cut_log, cut_result_dir]
                    )

                with gr.Accordion("Step 2: Generate Captions", open=False):
                    gr.Markdown("Generate descriptive captions for each clip using lightweight image captioning models")

                    with gr.Row():
                        caption_clips_dir = gr.Textbox(
                            label="Clips Directory",
                            value=str(CUT_VIDEOS_DIR)
                        )
                        caption_metadata = gr.Textbox(
                            label="Metadata File",
                            value=str(METADATA_FILE)
                        )

                    caption_model = gr.Dropdown(
                        label="Caption Model Backend",
                        choices=[
                            ("qwen3-vl (Qwen/Qwen3-VL-2B-Instruct) - Multi-frame, Best Quality ⭐", "qwen3-vl"),
                            ("vit-gpt2 (nlpconnect/vit-gpt2-image-captioning) - Low VRAM, Fast", "vit-gpt2"),
                            ("blip (Salesforce/blip-image-captioning-base) - Better Accuracy", "blip"),
                            ("blip2 (Salesforce/blip2-flan-t5-small) - Strongest Single-frame", "blip2"),
                        ],
                        value=CAPTION_BACKEND,
                        info="qwen3-vl (default) analyzes multiple frames for coherent captions. Others process single frames."
                    )

                    caption_button = gr.Button("📝 Generate Captions", variant="primary")
                    caption_log = gr.Textbox(label="Progress Log", lines=15, interactive=False)

                    caption_button.click(
                        fn=generate_captions_ui,
                        inputs=[caption_clips_dir, caption_metadata, caption_model],
                        outputs=caption_log
                    )

                with gr.Accordion("Step 3: Preprocess Dataset (Optional - Recommended)", open=False):
                    gr.Markdown("Pre-encode videos to VAE latents for **much faster training** (recommended for H200)")

                    with gr.Row():
                        prep_videos_dir = gr.Textbox(
                            label="Videos Directory",
                            value=str(CUT_VIDEOS_DIR)
                        )
                        prep_metadata = gr.Textbox(
                            label="Metadata File",
                            value=str(METADATA_FILE)
                        )
                        prep_cache_dir = gr.Textbox(
                            label="Cache Directory",
                            value=str(PREPROCESSED_LATENTS_DIR)
                        )

                    with gr.Row():
                        prep_button = gr.Button("⚡ Preprocess Videos", variant="primary")
                        clear_cache_button = gr.Button("🗑️ Clear Cache", variant="secondary")

                    prep_log = gr.Textbox(label="Progress Log", lines=12, interactive=False)

                    prep_button.click(
                        fn=preprocess_dataset_ui,
                        inputs=[prep_videos_dir, prep_metadata, prep_cache_dir],
                        outputs=prep_log
                    )

                    clear_cache_button.click(
                        fn=clear_cache_ui,
                        outputs=prep_log
                    )

                with gr.Accordion("Step 4: Train LoRA", open=False):
                    gr.Markdown("Fine-tune the model on your tutorial videos using LoRA")

                    with gr.Row():
                        with gr.Column():
                            train_model_id = gr.Dropdown(
                                label="Base Model",
                                choices=list(WAN_MODEL_MAP.values()),
                                value=WAN_MODEL_MAP["5B"]
                            )
                            train_dataset_dir = gr.Textbox(
                                label="Dataset Directory",
                                value="src/finetuning/dataset"
                            )
                            train_use_cache = gr.Checkbox(
                                label="Use Cached Latents (faster)",
                                value=True,
                                info="Enable if you ran Step 3"
                            )
                            train_latents_dir = gr.Textbox(
                                label="Latents Cache Directory",
                                value=str(PREPROCESSED_LATENTS_DIR)
                            )

                        with gr.Column():
                            train_epochs = gr.Slider(label="Epochs", minimum=1, maximum=20, value=5, step=1)
                            train_batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=4, value=1, step=1)
                            train_grad_accum = gr.Slider(label="Gradient Accumulation", minimum=1, maximum=16, value=4, step=1)
                            train_lr = gr.Number(label="Learning Rate", value=1e-4, precision=6)

                    train_output_dir = gr.Textbox(
                        label="Output Directory",
                        value=str(LORA_CHECKPOINTS_DIR / "tutorial_style")
                    )

                    train_button = gr.Button("🚀 Start Training", variant="primary", size="lg")

                    with gr.Row():
                        train_log = gr.Textbox(label="Training Log", lines=12, interactive=False)
                        train_loss = gr.Number(label="Current Loss", interactive=False)

                    train_button.click(
                        fn=train_lora_ui,
                        inputs=[
                            train_model_id, train_dataset_dir, train_use_cache,
                            train_latents_dir, train_output_dir, train_epochs,
                            train_batch_size, train_grad_accum, train_lr
                        ],
                        outputs=[train_log, train_loss]
                    )

        gr.Markdown("---")
        gr.Markdown("💡 **Tip:** All operations are GPU-locked - only one job can run at a time to prevent resource conflicts on your H200.")

    return app


# ============================================================================
# LAUNCH
# ============================================================================

def main():
    """Launch the Gradio app."""
    app = build_ui()
    app.queue()  # Enable queuing for long-running tasks
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()

