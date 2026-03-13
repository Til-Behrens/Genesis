#!/usr/bin/env python3
"""
Quick start script for Genesis.
Checks dependencies and launches the GUI.
"""
import sys
import subprocess
import logging
from pathlib import Path

# Suppress verbose logging from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SETUP_MARKER = Path.home() / ".genesis_setup_complete"

def check_first_time_setup():
    """Check if first-time setup has been completed."""
    # Allow --force-setup flag to re-run setup
    if "--force-setup" in sys.argv:
        print("🔄 Forcing setup re-run...\n")
        return False

    return SETUP_MARKER.exists()

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 11 or version.minor > 13:
        print(f"Warning: Python 3.11-3.13 recommended, you have {version.major}.{version.minor}")
        if version.minor > 13:
            print("   PyTorch does not support Python 3.14+ yet")
            print("   Please use Python 3.12 for this project")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

def check_gpu():
    """Check GPU availability (NVIDIA CUDA or AMD ROCm)."""
    try:
        import torch
        torch_version = torch.__version__
        print(f"PyTorch: {torch_version}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Detect platform
            platform = "CUDA"
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                platform = "ROCm"

            print(f"GPU available ({platform}): {gpu_name} ({vram_gb:.1f} GB)")
            return True
        else:
            print("GPU not available - GPU acceleration disabled")
            print("   Training will not be possible without GPU")
            return False
    except ImportError:
        print("PyTorch not installed")
        return False

def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("ffmpeg available")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("   ffmpeg not found - video cutting will not work")
    print("   Install: sudo apt install ffmpeg (Ubuntu) or brew install ffmpeg (macOS)")
    return False

def check_dependencies():
    """Check if key dependencies are installed."""
    required = [
        "gradio",
        "torch",
        "diffusers",
        "transformers",
        "peft",
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"   Missing packages: {', '.join(missing)}")
        print("   Run: pip install -e .")
        return False

    print("  All required packages installed")
    return True

def main():
    """Run startup checks and launch GUI."""
    # Check if first-time setup has been completed
    if not check_first_time_setup():
        print("=" * 60)
        print("🎬 Genesis - First Time Setup Required")
        print("=" * 60)
        print("\nGenesis needs to be set up before first use.")
        print("This will detect your GPU and install the correct PyTorch version.")

        # If --force-setup flag, run setup directly
        if "--force-setup" in sys.argv:
            print("\nRunning setup wizard...\n")
            result = subprocess.run([sys.executable, 'setup_genesis.py'])
            sys.exit(result.returncode)
        else:
            print("\nRun: python setup_genesis.py")
            print("\nOr pass --force-setup to re-run setup:")
            print("  python run_genesis.py --force-setup")
            sys.exit(1)

    print("=" * 60)
    print("Genesis - Video Generation & Finetuning Pipeline")
    print("=" * 60)
    print()

    print("Running startup checks...")
    print()

    check_python_version()
    has_gpu = check_gpu()
    has_ffmpeg = check_ffmpeg()
    has_deps = check_dependencies()

    print()

    if not has_deps:
        print("Cannot start - missing dependencies")
        sys.exit(1)

    if not has_gpu:
        print("Running without GPU - generation may be slow")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    print()
    print("=" * 60)
    print("Launching Genesis GUI...")
    print("=" * 60)
    print()
    print("GUI will be available at: http://127.0.0.1:7860")
    print("Press Ctrl+C to stop")
    print()

    # Launch the GUI
    try:
        from src.ui.app import main as launch_app
        launch_app()
    except KeyboardInterrupt:
        print("\n\nGenesis stopped")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()




