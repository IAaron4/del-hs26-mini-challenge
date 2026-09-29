"""Device selection and reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def available_backends() -> list[str]:
    """Accelerator backends this build can reach, in preference order."""
    found = []
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        found.append("xpu")
    if torch.cuda.is_available():
        found.append("cuda")
    if torch.backends.mps.is_available():
        found.append("mps")
    return found


def select_device(prefer: str | None = None) -> torch.device:
    """Pick an accelerator, falling back to CPU.

    Args:
        prefer: force a specific backend; raises if that backend is unavailable,
            so a silent fallback can never be mistaken for a working accelerator.
    """
    if prefer is not None:
        if prefer != "cpu" and prefer not in available_backends():
            raise RuntimeError(f"backend {prefer!r} requested but not available")
        return torch.device(prefer)
    backends = available_backends()
    return torch.device(backends[0] if backends else "cpu")


def device_name(device: torch.device) -> str:
    """Human-readable device name for logging and run metadata."""
    index = device.index or 0
    if device.type == "xpu":
        return torch.xpu.get_device_name(index)
    if device.type == "cuda":
        return torch.cuda.get_device_name(index)
    return device.type


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    Args:
        deterministic: trade throughput for bit-exact reproducibility. Off by
            default because the hypothesis experiments report seed variance
            explicitly rather than pretending single runs are exact.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
