# Genesis

Text-to-video generation and fine-tuning pipeline for university tutorial videos, built on Wan2.x and DiffSynth-Studio. Developed at TU Berlin.

Genesis provides a Gradio interface for two workflows:

- **Generation:** create videos from a text prompt with Wan2.2 5B, Wan2.2 14B, Wan2.1 14B or Wan2.1 1.3B. Weights are offloaded to disk and CPU and streamed to the GPU, which keeps VRAM usage below the card's capacity.
- **Fine-tuning:** train a LoRA adapter on your own tutorial recordings.

## Fine-tuning pipeline

1. **Cut:** split raw `.mp4` recordings into overlapping clips at 1280x704, 24 fps (ffmpeg).
2. **Caption:** generate a German text-to-video prompt per clip with a vision-language model (Qwen3-VL by default; BLIP, BLIP-2 and ViT-GPT2 as lighter alternatives), with speech transcribed by Whisper as additional context.
3. **Prepare dataset:** drop missing and uncaptioned clips and export the training metadata for DiffSynth.
4. **Train:** run DiffSynth-Studio's Wan LoRA training script through `accelerate`, with the log streamed to the interface.

Only one job runs at a time; the generation model is loaded on first use and unloaded before captioning or training.

## Requirements

- Python 3.11–3.13 (3.12 recommended)
- NVIDIA GPU (CUDA) or AMD GPU (ROCm, Linux)
- ffmpeg
- For training: a local clone of DiffSynth-Studio, which contains the training script:

      git clone https://github.com/modelscope/DiffSynth-Studio
      pip install -e DiffSynth-Studio

## Getting started

    python setup_genesis.py   # detects the GPU, installs the matching PyTorch build, then Genesis
    genesis                   # checks the environment and starts the interface

The interface runs at http://127.0.0.1:7860 (`genesis --help` for host, port and sharing). Setup can be re-run at any time.

Runtime data lives below the working directory: source recordings in `data/raw_videos`, clips and metadata in `data/`, generated videos in `outputs/`, LoRA adapters in `models/lora/`.

## Configuration

| Variable | Purpose |
|---|---|
| `GENESIS_HOME` | Base directory for data, outputs and models (default: working directory) |
| `GENESIS_CACHE_DIR` | Model download directory (default `cache/models`) |
| `GENESIS_ONLOAD_DEVICE` | `cpu`, `disk` or `auto` (default); `disk` streams weights from disk on machines with less than 32 GB RAM |
| `GENESIS_CAPTION_BACKEND` | `qwen3-vl`, `blip2`, `blip` or `vit-gpt2` |
| `GENESIS_DIFFSYNTH_ROOT` | Path to the DiffSynth-Studio clone used for training |
| `GENESIS_DEBUG_VRAM` | Set to `1` to log VRAM usage during generation |

The model table and further defaults (clip length, caption settings) live in `src/genesis/config.py`.

## Development

    pip install -e ".[dev]"
    pytest
    ruff check

## Project structure

    src/genesis/
      config.py         paths, model table, defaults
      generation.py     text-to-video pipeline
      finetuning/       cutting, captioning, dataset preparation, LoRA training
      ui/app.py         Gradio interface
      cli.py            `genesis` command
    setup_genesis.py    setup wizard (standard library only)

## License

Genesis is free software released under the [GNU General Public License v3](LICENSE).
