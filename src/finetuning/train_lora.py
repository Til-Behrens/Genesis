"""
LoRA fine-tuning for Wan2.2 using cached latents as training targets.

APPROACH:
- Load pre-encoded latents (from preprocessing) as ground truth
- Add noise via diffusion scheduler
- Train transformer to denoise them
- Minimize MSE loss between predicted and actual noise
- Fine-tune only LoRA parameters for efficiency
"""

from diffusers import WanPipeline, AutoencoderKLWan
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pathlib
import logging
from typing import Generator, Dict, Any
from src.finetuning.preprocess_dataset import LatentDataset
import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LoRA")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def collate_fn(batch):
    """Collate latents and texts."""
    latents = torch.stack([item['latents'] for item in batch], dim=0)
    texts = [item['text'] for item in batch]

    if latents.ndim == 6 and latents.shape[1] == 1:
        latents = latents.squeeze(1)

    return {'latents': latents, 'text': texts}


def train_lora_pipeline(
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    latents_cache_dir: pathlib.Path | str = None,
    metadata_path: pathlib.Path | str = None,
    output_dir: pathlib.Path | str = "models/lora_checkpoints",
    cache_dir: str = None,
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    lora_r: int = 64,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    save_steps: int = 100,
    logging_steps: int = 10,
) -> Generator[Dict[str, Any], None, None]:
    """Train LoRA on Wan2.2 using cached latents."""

    from src.core.config import get_optimal_device
    device = get_optimal_device()

    if device == "cpu":
        yield {"status": "error", "message": "GPU required"}
        return

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    yield {"status": "init", "device": str(device), "precision": "bf16" if use_bf16 else "fp16"}

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latents_cache_dir = pathlib.Path(latents_cache_dir or "data/preprocessed_latents")
    metadata_path = pathlib.Path(metadata_path or "data/cut_videos/metadata.jsonl")

    try:
        # Load model
        yield {"status": "loading", "message": "Loading VAE..."}
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=cache_dir
        )

        yield {"status": "loading", "message": "Loading pipeline..."}
        pipe = WanPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        pipe = pipe.to(device)

        # Apply LoRA
        yield {"status": "loading", "message": "Applying LoRA..."}
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        pipe.transformer = get_peft_model(pipe.transformer, lora_config)

        trainable = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in pipe.transformer.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        # Load dataset
        yield {"status": "loading", "message": "Loading dataset..."}
        dataset = LatentDataset(latents_cache_dir, metadata_path)

        if len(dataset) == 0:
            yield {"status": "error", "message": f"No latents in {latents_cache_dir}"}
            return

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        logger.info(f"Dataset: {len(dataset)} samples")

        optimizer = torch.optim.AdamW(
            [p for p in pipe.transformer.parameters() if p.requires_grad],
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        from torch.optim.lr_scheduler import CosineAnnealingLR
        total_steps = len(dataloader) * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate * 0.1)

        # Training
        yield {"status": "training", "message": "Starting...", "total_steps": total_steps}

        pipe.transformer.train()
        global_step = 0
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            first_batch = True

            progress = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in progress:
                target_latents = batch['latents'].to(device, dtype=dtype)
                texts = batch['text']

                if first_batch:
                    logger.info(f"Batch shape: {target_latents.shape}, dtype: {target_latents.dtype}")
                    first_batch = False

                if target_latents.ndim != 5 or target_latents.shape[1] != 16:
                    raise ValueError(f"Invalid shape: {target_latents.shape}")

                # Diffusion: add noise
                num_train_timesteps = pipe.scheduler.config.num_train_timesteps
                timesteps = torch.randint(0, num_train_timesteps, (target_latents.shape[0],), device=device)
                noise = torch.randn_like(target_latents)
                noisy_latents = pipe.scheduler.add_noise(target_latents, noise, timesteps)

                # Encode text
                text_embeddings = []
                for text in texts:
                    with torch.no_grad():
                        tokens = pipe.tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
                        emb = pipe.text_encoder(tokens.input_ids)[0]
                    text_embeddings.append(emb)
                text_embeddings = torch.cat(text_embeddings, dim=0)

                # Forward pass
                with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu", dtype=dtype):
                    model_output = pipe.transformer(noisy_latents, timesteps, encoder_hidden_states=text_embeddings)

                    if hasattr(model_output, 'sample'):
                        predicted_noise = model_output.sample
                    else:
                        predicted_noise = model_output[0] if isinstance(model_output, tuple) else model_output

                    loss = F.mse_loss(predicted_noise, noise, reduction='mean')

                # Backward
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                losses.append(loss.item())

                if global_step % logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(f"Step {global_step}: loss={avg_loss:.4f}")
                    progress.set_postfix({"loss": avg_loss})

                if global_step % save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    pipe.transformer.save_pretrained(ckpt_dir)
                    logger.info(f"Checkpoint: {ckpt_dir}")
                    yield {"status": "checkpoint", "step": global_step, "loss": epoch_loss / num_batches}

        # Final save
        yield {"status": "saving", "message": "Saving..."}
        pipe.transformer.save_pretrained(output_dir / "final")

        final_loss = sum(losses[-100:]) / min(100, len(losses)) if losses else 0
        logger.info(f"Complete! Loss: {final_loss:.4f}, saved to {output_dir}")

        yield {"status": "complete", "output_dir": str(output_dir), "final_loss": final_loss}

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        yield {"status": "error", "message": str(e)}


if __name__ == "__main__":
    for update in train_lora_pipeline():
        print(f"{update['status']}: {update.get('message', '')}")

