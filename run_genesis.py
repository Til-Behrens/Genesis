#!/usr/bin/env python3
"""
Quick start script for Genesis.
Checks dependencies and launches the GUI.
"""
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.13."""
    version = sys.version_info
    if version.major != 3 or version.minor != 13:
        print(f"⚠️  Warning: Python 3.13 recommended, you have {version.major}.{version.minor}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ CUDA available: {gpu_name} ({vram_gb:.1f} GB)")
            return True
        else:
            print("⚠️  CUDA not available - GPU acceleration disabled")
            print("   Training will not be possible without CUDA")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed")
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
            print("✓ ffmpeg available")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("⚠️  ffmpeg not found - video cutting will not work")
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
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -e .")
        return False

    print("✓ All required packages installed")
    return True

def main():
    """Run startup checks and launch GUI."""
    print("=" * 60)
    print("🎬 Genesis - Video Generation & Finetuning Pipeline")
    print("=" * 60)
    print()

    print("Running startup checks...")
    print()

    check_python_version()
    has_cuda = check_cuda()
    has_ffmpeg = check_ffmpeg()
    has_deps = check_dependencies()

    print()

    if not has_deps:
        print("❌ Cannot start - missing dependencies")
        sys.exit(1)

    if not has_cuda:
        print("⚠️  Running without CUDA - generation may be slow")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    print()
    print("=" * 60)
    print("🚀 Launching Genesis GUI...")
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
        print("\n\n✓ Genesis stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
