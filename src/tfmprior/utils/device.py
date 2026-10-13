"""Device choice: cuda > mps > cpu, as in the original nanoTabPFN."""

import torch


def get_default_device() -> str:
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return device


def release_cached_memory(device: str) -> None:
    """MPS keeps a memory cache per tensor shape and runs out of memory when batch shapes vary; no-op elsewhere."""
    if device == "mps":
        torch.mps.empty_cache()
