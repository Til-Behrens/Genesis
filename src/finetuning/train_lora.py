from diffusers import WanPipeline, AutoencoderKLWan, UniPCMultistepScheduler
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import Trainer, TrainingArguments
import torch
import json
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LoRA")

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
CACHE_DIR = "/content/drive/MyDrive/Genesis/models"
DATASET_PATH = "src/finetuning/dataset"       # folder with videos/ + captions.jsonl
OUTPUT_DIR = "models/lora_checkpoints/tutorial_style"

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.05,
    bias="none",
)

vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    torch_dtype=torch.bfloat16,
    cache_dir=CACHE_DIR,
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
pipe.enable_model_cpu_offload()

pipe.transformer = get_peft_model(pipe.transformer, lora_config)
pipe.transformer.print_trainable_parameters()

# --- loading datasets (JSONL-Format) ---
# example captions.jsonl:
# {"file_name": "video_001.mp4", "text": "white slide, title 'pointers', bulletpoints appear one after the other"}
from datasets import load_dataset
dataset = load_dataset("video_folder", data_dir=DATASET_PATH, split="train")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=pipe.transformer,
    args=training_args,
    train_dataset=dataset,
)
logger.info("Starting LoRA-training on example clips...")
trainer.train()

pipe.save_pretrained(OUTPUT_DIR)

logger.info(f"LoRA finished! Saved in {OUTPUT_DIR}")
logger.info("Use it later with: pipe.load_lora_weights('{OUTPUT_DIR}')")