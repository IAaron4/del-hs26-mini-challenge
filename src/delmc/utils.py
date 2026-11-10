"""Device selection and seeding."""

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
    """First available accelerator, otherwise the CPU. `prefer` raises if that backend is missing."""
    if prefer is not None:
        if prefer != "cpu" and prefer not in available_backends():
            raise RuntimeError(f"backend {prefer!r} requested but not available")
        return torch.device(prefer)
    backends = available_backends()
    return torch.device(backends[0] if backends else "cpu")


def seed_everything(seed: int) -> None:
    """Seeds python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
