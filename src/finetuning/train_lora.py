"""
LoRA fine-tuning for Wan2.2 using efficient cached latents.
"""
from diffusers import WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, TrainerCallback
import torch
import pathlib
import logging
from typing import Generator, Dict, Any
from src.finetuning.preprocess_dataset import LatentDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LoRA")

# Suppress verbose logging from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class ProgressCallback(TrainerCallback):
    """Callback to track training progress."""

    def __init__(self, callback_fn=None):
        self.callback_fn = callback_fn

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.callback_fn and logs:
            self.callback_fn(logs)


def train_lora_pipeline(
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    dataset_path: pathlib.Path | str = None,
    use_cached_latents: bool = True,
    latents_cache_dir: pathlib.Path | str = None,
    output_dir: pathlib.Path | str = "models/lora_checkpoints/tutorial_style",
    cache_dir: str = "/content/drive/MyDrive/Genesis/models",
    epochs: int = 5,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    learning_rate: float = 1e-4,
    lora_r: int = 64,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    save_steps: int = 100,
    logging_steps: int = 10,
    progress_callback = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Train LoRA on Wan2.2 with efficient preprocessing.

    Args:
        model_id: Model ID to fine-tune
        dataset_path: Path to dataset folder (with metadata.jsonl)
        use_cached_latents: Whether to use pre-encoded latents (much faster)
        latents_cache_dir: Directory with cached latents
        output_dir: Where to save LoRA weights
        cache_dir: Model cache directory
        epochs: Number of training epochs
        batch_size: Batch size per device
        grad_accum_steps: Gradient accumulation steps
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        progress_callback: Optional callback for progress updates

    Yields:
        Progress updates with format:
        {"status": "init", "message": str}
        {"status": "loading", "message": str}
        {"status": "training", "step": int, "loss": float, "epoch": int}
        {"status": "complete", "output_dir": str}
        {"status": "error", "message": str}
    """

    # GPU availability check
    from src.core.config import get_device_info, get_optimal_device

    device = get_optimal_device()
    if device == "cpu":
        yield {"status": "error", "message": "GPU is not available. A GPU is required for training."}
        return

    # Detect precision support
    use_bf16 = torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    device_type, device_name, vram_gb = get_device_info()
    yield {
        "status": "init",
        "message": f"Initializing training on {device_name}",
        "precision": "bf16" if use_bf16 else "fp16",
        "device": device_name
    }

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load VAE
        yield {"status": "loading", "message": "Loading VAE..."}
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=cache_dir
        )

        # Load pipeline
        yield {"status": "loading", "message": f"Loading {model_id}..."}
        pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )

        # Configure scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            flow_shift=5.0
        )

        # Wrap transformer with LoRA
        yield {"status": "loading", "message": "Applying LoRA configuration..."}
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=lora_dropout,
            bias="none",
        )

        pipe.transformer = get_peft_model(pipe.transformer, lora_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in pipe.transformer.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        # Move to CUDA (important: don't use cpu_offload for training!)
        pipe.transformer.to("cuda")

        # Load dataset
        yield {"status": "loading", "message": "Loading dataset..."}

        if use_cached_latents and latents_cache_dir:
            metadata_path = pathlib.Path(dataset_path) / "metadata.jsonl" if dataset_path else pathlib.Path("src/finetuning/dataset/metadata.jsonl")
            dataset = LatentDataset(latents_cache_dir, metadata_path)

            if len(dataset) == 0:
                yield {
                    "status": "error",
                    "message": f"No cached latents found in {latents_cache_dir}. Please run preprocessing first."
                }
                return
        else:
            # Fallback to loading videos directly (slower)
            from datasets import load_dataset
            dataset = load_dataset("video_folder", data_dir=str(dataset_path), split="train")

        logger.info(f"Dataset loaded: {len(dataset)} samples")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=learning_rate,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,  # Keep only last 3 checkpoints
            report_to="none",
            logging_dir=str(output_dir / "logs"),
            remove_unused_columns=False,
        )

        # Create trainer with callback
        callback = ProgressCallback(callback_fn=progress_callback)

        trainer = Trainer(
            model=pipe.transformer,
            args=training_args,
            train_dataset=dataset,
            callbacks=[callback],
        )

        # Start training
        yield {"status": "training", "message": "Starting LoRA training...", "total_steps": trainer.state.max_steps if hasattr(trainer.state, 'max_steps') else None}

        logger.info("Starting LoRA training...")
        trainer.train()

        # Save LoRA weights using PEFT API (not pipe.save_pretrained!)
        yield {"status": "saving", "message": "Saving LoRA weights..."}
        pipe.transformer.save_pretrained(output_dir)

        logger.info(f"✓ LoRA training complete! Saved to {output_dir}")
        logger.info(f"Load with: pipe.load_lora_weights('{output_dir}')")

        yield {
            "status": "complete",
            "output_dir": str(output_dir),
            "message": f"Training complete! LoRA weights saved to {output_dir}"
        }

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        yield {"status": "error", "message": f"Training failed: {str(e)}"}


# Legacy main for backward compatibility
def main():
    """Run training with default settings."""
    for update in train_lora_pipeline(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        dataset_path="src/finetuning/dataset",
        use_cached_latents=False,  # Set to True after running preprocessing
        output_dir="models/lora_checkpoints/tutorial_style",
    ):
        if update["status"] == "error":
            logger.error(update["message"])
            break
        elif update["status"] == "training":
            if "loss" in update:
                logger.info(f"Step {update.get('step', '?')}: loss = {update['loss']:.4f}")
        elif update["status"] == "complete":
            logger.info(update["message"])


if __name__ == "__main__":
    main()
