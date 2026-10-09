"""Compute device detection for NVIDIA (CUDA) and AMD (ROCm) GPUs."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """Primary compute device; ROCm builds of PyTorch also report as `cuda`."""

    type: str
    name: str
    vram_gb: float

    @property
    def is_gpu(self) -> bool:
        """Whether a CUDA or ROCm device is available."""
        return self.type == "cuda"


def get_device_info() -> DeviceInfo:
    """Describe the first GPU, or the CPU if PyTorch sees none."""
    try:
        import torch
    except ImportError:
        return DeviceInfo("cpu", "CPU", 0.0)

    if not torch.cuda.is_available():
        return DeviceInfo("cpu", "CPU", 0.0)

    platform = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
    name = f"{torch.cuda.get_device_name(0)} ({platform})"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return DeviceInfo("cuda", name, vram_gb)


def optimal_device() -> str:
    """Return `cuda` if a GPU is usable, else `cpu`."""
    return get_device_info().type


def total_ram_gb() -> float:
    """Total system memory in GB, 0.0 if the platform does not report it."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, ValueError, OSError):
        return 0.0
