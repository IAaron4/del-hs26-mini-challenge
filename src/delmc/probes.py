"""Per-layer gradients and activations, for the depth and initialisation hypotheses."""

from __future__ import annotations

from torch import nn

WEIGHTED = (nn.Conv2d, nn.Linear)


def weighted_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Conv and linear layers in the order they are declared, which is the forward order here."""
    return [(name, module) for name, module in model.named_modules() if isinstance(module, WEIGHTED)]


class GradientFlowProbe:
    """Reads the weight gradients layer by layer. Call between backward() and step()."""

    def __init__(self, model: nn.Module) -> None:
        self.layers = weighted_layers(model)

    def snapshot(self) -> dict[str, float]:
        """Gradient norm per layer."""
        return {
            name: float(module.weight.grad.detach().norm())
            for name, module in self.layers
            if module.weight.grad is not None
        }

    def snapshot_rms(self) -> dict[str, float]:
        """Root mean square of the gradient per layer.

        The norm grows with the number of weights, the RMS does not, so layers of
        different size can be compared.
        """
        return {
            name: float(module.weight.grad.detach().pow(2).mean().sqrt())
            for name, module in self.layers
            if module.weight.grad is not None
        }

    @staticmethod
    def ratio(snapshot: dict[str, float]) -> float:
        """First layer divided by last layer."""
        if len(snapshot) < 2:
            return float("nan")
        values = list(snapshot.values())
        last = values[-1]
        return values[0] / last if last > 0 else float("inf")


class ActivationStatsProbe:
    """Mean and std of the output of every conv and linear layer, via forward hooks."""

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
