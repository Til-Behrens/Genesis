# Genesis

Text-to-video generation and fine-tuning pipeline for university tutorial videos, built on Wan2.x and DiffSynth-Studio. Developed at TU Berlin.

Genesis provides a Gradio interface for two workflows:

- **Generation:** create videos from a text prompt with Wan2.2 5B, Wan2.2 14B, Wan2.1 14B or Wan2.1 1.3B. Weights are offloaded to disk and CPU and streamed to the GPU, which keeps VRAM usage below the card's capacity.
- **Fine-tuning:** train a LoRA adapter on your own tutorial recordings.

## Fine-tuning pipeline

1. **Cut:** split raw `.mp4` recordings into overlapping clips at 1280x704, 24 fps (ffmpeg).
2. **Caption:** generate a German text-to-video prompt per clip with a vision-language model (Qwen3-VL by default; BLIP, BLIP-2 and ViT-GPT2 as lighter alternatives), with speech transcribed by Whisper as additional context.
3. **Prepare metadata:** validate clips and export a DiffSynth-compatible `metadata.jsonl`.
4. **Train:** run DiffSynth-Studio's Wan LoRA training script through `accelerate`, with the log streamed to the interface.

Only one GPU job runs at a time; generation models are loaded on first use and unloaded when switching.

## Requirements

- Python 3.11–3.13 (3.12 recommended)
- NVIDIA GPU (CUDA) or AMD GPU (ROCm, Linux)
- ffmpeg
- For training: a local clone of DiffSynth-Studio, which contains the training script:

      git clone https://github.com/modelscope/DiffSynth-Studio
      pip install -e DiffSynth-Studio

## Getting started

    python setup_genesis.py   # detects the GPU and installs the matching PyTorch build
    python run_genesis.py

The interface runs at http://127.0.0.1:7860. To re-run setup later: `python run_genesis.py --force-setup`.

## Configuration

| Variable | Purpose |
|---|---|
| `GENESIS_CACHE_DIR` | Model download directory (default `cache/models`) |
| `GENESIS_CAPTION_BACKEND` | `qwen3-vl`, `blip2`, `blip` or `vit-gpt2` |
| `GENESIS_DIFFSYNTH_ROOT` | Path to the DiffSynth-Studio clone used for training |
| `GENESIS_DEBUG_VRAM` | Set to `1` to log VRAM usage during generation |

Further defaults (clip length, LoRA rank, target modules) live in `src/core/config.py`.

## Project structure

    src/core/         configuration, generation pipeline, GPU job lock
    src/finetuning/   cutting, captioning, metadata preparation, LoRA training
    src/ui/app.py     Gradio interface
    setup_genesis.py  first-time setup wizard
    run_genesis.py    launcher with environment checks

## License

Genesis is free software released under the [GNU General Public License v3](LICENSE).
