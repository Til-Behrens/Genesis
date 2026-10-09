"""`genesis` command: check the environment, then serve the web interface."""
import argparse
import importlib.util
import logging
import sys

from genesis import __version__
from genesis.config import HOME_DIR, ensure_directories
from genesis.device import get_device_info, total_ram_gb
from genesis.finetuning.cut_videos import ffmpeg_available

REQUIRED_PACKAGES = ("torch", "gradio", "diffsynth", "transformers", "cv2")
QUIET_LOGGERS = ("httpx", "httpcore", "urllib3")


def configure_logging(verbose: bool = False) -> None:
    """Configure the root logger once for the whole process."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def check_environment() -> list[str]:
    """Print the environment summary and return blocking problems (empty if none)."""
    problems = []
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        problems.append(f"Python {sys.version_info.major}.{sys.version_info.minor} is unsupported, use 3.11 to 3.13.")

    missing = [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        problems.append(f"Missing packages: {', '.join(missing)}. Run: python setup_genesis.py")

    device = get_device_info() if "torch" not in missing else None
    if device and device.is_gpu:
        print(f"GPU: {device.name}, {device.vram_gb:.1f} GB VRAM")
    else:
        print("GPU: none found, generation and training will not work")
    print(f"RAM: {total_ram_gb():.1f} GB")
    if not ffmpeg_available():
        print("ffmpeg: not found, cutting videos and transcription will not work")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `genesis` console script."""
    parser = argparse.ArgumentParser(prog="genesis", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="interface address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=7860, help="interface port (default: %(default)s)")
    parser.add_argument("--share", action="store_true", help="create a public gradio link")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version=f"genesis {__version__}")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    print(f"Genesis {__version__}, data in {HOME_DIR}")
    problems = check_environment()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1

    ensure_directories()
    from genesis.ui.app import launch

    print(f"Interface: http://{args.host}:{args.port}")
    try:
        launch(args.host, args.port, args.share)
    except KeyboardInterrupt:
        pass
    return 0
