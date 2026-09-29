"""Evaluation metrics for the classification task."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MetricResult:
    """Metrics for one evaluation pass."""

    accuracy: float
    macro_f1: float
    per_class_f1: tuple[float, ...]


class MetricSet:
    """Accumulates predictions over an epoch and reduces them to metrics.

    Stores a confusion matrix rather than raw predictions, so memory stays at
    O(C^2) regardless of dataset size and the matrix is available for free.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch."""
        preds = logits.argmax(dim=1).to(torch.long).cpu()
        truth = targets.to(torch.long).cpu()
        flat = truth * self.num_classes + preds
        counts = torch.bincount(flat, minlength=self.num_classes**2)
        self._matrix += counts.reshape(self.num_classes, self.num_classes)

    @property
    def confusion(self) -> torch.Tensor:
        """Rows are true classes, columns are predictions."""
        return self._matrix.clone()

    def compute(self) -> MetricResult:
        matrix = self._matrix.to(torch.float64)
        total = matrix.sum()
        if total == 0:
            raise RuntimeError("no samples accumulated")

        true_positive = matrix.diag()
        predicted = matrix.sum(dim=0)
        actual = matrix.sum(dim=1)

        denominator = 2 * true_positive + (predicted - true_positive) + (actual - true_positive)
        # Classes absent from both truth and predictions score 0 instead of NaN.
        f1 = torch.where(denominator > 0, 2 * true_positive / denominator, torch.zeros_like(denominator))

        return MetricResult(
            accuracy=float(true_positive.sum() / total),
            macro_f1=float(f1.mean()),
            per_class_f1=tuple(float(value) for value in f1),
        )
