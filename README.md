# Genesis

Text-to-video generation and fine-tuning pipeline for university tutorial videos, built on Wan2.x. Developed at TU Berlin.

Genesis provides a Gradio interface for two workflows:

- **Generation:** create short videos from a text prompt with Wan2.2 5B, Wan2.2 14B or Wan2.1 1.3B (Hugging Face diffusers).
- **Fine-tuning:** train a LoRA adapter on your own tutorial recordings.

## Fine-tuning pipeline

1. **Cut:** split raw `.mp4` recordings into overlapping 8-second clips (ffmpeg).
2. **Caption:** generate a text prompt per clip with a selectable vision-language backend (Qwen3-VL by default; BLIP, BLIP-2 and ViT-GPT2 as lighter alternatives). The clip's speech is transcribed with Whisper and added as context.
3. **Preprocess (optional):** encode clips to VAE latents once and cache them to speed up training.
4. **Train:** fine-tune a LoRA adapter on the Wan transformer with PEFT and the Hugging Face Trainer.

Only one GPU job runs at a time; models are loaded on first use.

## Requirements

- Python 3.11–3.13 (3.12 recommended)
- NVIDIA GPU (CUDA) or AMD GPU (ROCm, Linux), 24 GB+ VRAM recommended
- ffmpeg

## Getting started

    python setup_genesis.py   # detects the GPU and installs the matching PyTorch build
    python run_genesis.py

The interface runs at http://127.0.0.1:7860. Model weights are downloaded to `cache/models` (override with `GENESIS_CACHE_DIR`). To re-run setup later: `python run_genesis.py --force-setup`.

## Project structure

    src/core/         configuration, generation pipeline, GPU job lock
    src/finetuning/   cutting, captioning, preprocessing, LoRA training
    src/ui/app.py     Gradio interface
    setup_genesis.py  first-time setup wizard
    run_genesis.py    launcher with environment checks
