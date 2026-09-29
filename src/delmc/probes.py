"""Per-layer gradient and activation instrumentation.

The vanishing-gradient hypotheses stand or fall on these numbers, so they are
measured rather than asserted.
"""

from __future__ import annotations

import torch
from torch import nn

WEIGHTED = (nn.Conv2d, nn.Linear)


def weighted_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Conv and linear layers in forward order, which is the order they are declared."""
    return [(name, module) for name, module in model.named_modules() if isinstance(module, WEIGHTED)]


class GradientFlowProbe:
    """Gradient norm of every weight tensor, layer by layer.

    Reads `.grad` directly instead of registering backward hooks: after
    `loss.backward()` the gradients are already there, and avoiding hooks keeps
    the training loop free of side effects.
    """

    def __init__(self, model: nn.Module) -> None:
        self.layers = weighted_layers(model)

    def snapshot(self) -> dict[str, float]:
        """Current gradient norms. Call between backward() and step()."""
        return {
            name: float(module.weight.grad.detach().norm())
            for name, module in self.layers
            if module.weight.grad is not None
        }

    def snapshot_rms(self) -> dict[str, float]:
        """Root-mean-square gradient per weight element.

        The raw norm grows with the number of elements in W, so comparing norms
        across layers of different width conflates gradient magnitude with
        parameter count. RMS divides that out and is the honest depth diagnostic.
        """
        return {
            name: float(module.weight.grad.detach().pow(2).mean().sqrt())
            for name, module in self.layers
            if module.weight.grad is not None
        }

    @staticmethod
    def ratio(snapshot: dict[str, float]) -> float:
        """First-layer over last-layer gradient norm.

        The headline diagnostic: it decays by orders of magnitude exactly when
        the Jacobian product collapses. Feed it RMS values, not raw norms.
        """
        if len(snapshot) < 2:
            return float("nan")
        values = list(snapshot.values())
        last = values[-1]
        return values[0] / last if last > 0 else float("inf")


class ActivationStatsProbe:
    """Mean and std of each layer's output, the forward twin of gradient decay."""

    def __init__(self, model: nn.Module) -> None:
        self._stats: dict[str, tuple[float, float]] = {}
        self._handles = [
            module.register_forward_hook(self._record(name)) for name, module in weighted_layers(model)
        ]

    def _record(self, name: str):
        def hook(_module, _inputs, output):
            detached = output.detach()
            self._stats[name] = (float(detached.mean()), float(detached.std()))

        return hook

    def snapshot(self) -> dict[str, tuple[float, float]]:
        return dict(self._stats)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ActivationStatsProbe":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


@torch.no_grad()
def update_to_weight_ratio(model: nn.Module, previous: dict[str, torch.Tensor]) -> dict[str, float]:
    """Relative parameter change since `previous`, per layer.

    Karpathy's rule of thumb puts a healthy value near 1e-3; far below means the
    layer is barely learning.
    """
    ratios = {}
    for name, module in weighted_layers(model):
        if name not in previous:
            continue
        weight = module.weight.detach()
        norm = weight.norm()
        if norm > 0:
            ratios[name] = float((weight - previous[name]).norm() / norm)
    return ratios


@torch.no_grad()
def weight_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy of current weights, for a later update_to_weight_ratio call."""
    return {name: module.weight.detach().clone() for name, module in weighted_layers(model)}
