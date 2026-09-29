"""Training loop and the history it produces."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import MetricSet
from .probes import ActivationStatsProbe, GradientFlowProbe
from .tracking import NullTracker


@dataclass
class EpochRecord:
    """One epoch of training and validation."""

    epoch: int
    train_loss: float
    train_accuracy: float
    train_macro_f1: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    seconds: float
    gradient_norms: dict[str, float] | None = None
    gradient_rms: dict[str, float] | None = None
    gradient_ratio: float | None = None
    activation_std: dict[str, float] | None = None


@dataclass
class OverfitResult:
    """Outcome of the one-batch sanity test."""

    losses: list[float]
    accuracy: float

    @property
    def passed(self) -> bool:
        return self.accuracy >= 0.99


@dataclass
class History:
    """Everything a run produced, serialisable so plots never need a retrain."""

    config: dict[str, Any] = field(default_factory=dict)
    records: list[EpochRecord] = field(default_factory=list)
    confusion: list[list[int]] | None = None

    def append(self, record: EpochRecord) -> None:
        self.records.append(record)

    def series(self, field_name: str) -> list[float]:
        return [getattr(record, field_name) for record in self.records]

    def best(self, field_name: str = "val_accuracy") -> EpochRecord:
        return max(self.records, key=lambda record: getattr(record, field_name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "records": [asdict(record) for record in self.records],
            "confusion": self.confusion,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "History":
        payload = json.loads(Path(path).read_text())
        return cls(
            config=payload["config"],
            records=[EpochRecord(**record) for record in payload["records"]],
            confusion=payload.get("confusion"),
        )


class Trainer:
    """Trains one model and reports per-epoch metrics.

    Deliberately takes the epoch count as an argument and keeps the loss
    function on the instance, so neither can be shadowed by a running total.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_classes: int = 10,
        loss_fn: nn.Module | None = None,
        tracker=None,
        probe: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.loss_fn = (loss_fn or nn.CrossEntropyLoss()).to(device)
        self.tracker = tracker or NullTracker()
        self.probe = probe
        self._gradient_probe = GradientFlowProbe(self.model) if probe else None
        self.last_confusion = None

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, config: dict | None = None) -> History:
        history = History(config=dict(config or {}))

        for epoch in range(epochs):
            started = time.perf_counter()
            train_loss, train_metrics, gradients, gradient_rms = self._train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_metrics.accuracy,
                train_macro_f1=train_metrics.macro_f1,
                val_loss=val_loss,
                val_accuracy=val_metrics.accuracy,
                val_macro_f1=val_metrics.macro_f1,
                seconds=time.perf_counter() - started,
                gradient_norms=gradients,
                gradient_rms=gradient_rms,
                gradient_ratio=GradientFlowProbe.ratio(gradient_rms) if gradient_rms else None,
            )
            history.append(record)
            self.tracker.log({"Epoch": epoch, **{k: v for k, v in asdict(record).items() if isinstance(v, (int, float))}})

        if self.last_confusion is not None:
            history.confusion = self.last_confusion.tolist()
        return history

    def _train_epoch(self, loader: DataLoader):
        self.model.train()
        metrics = MetricSet(self.num_classes)
        running = 0.0
        gradients: dict[str, float] | None = None
        gradient_rms: dict[str, float] | None = None

        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            loss.backward()

            # Sample gradients once per epoch; every step would dominate runtime.
            if self._gradient_probe is not None and step == 0:
                gradients = self._gradient_probe.snapshot()
                gradient_rms = self._gradient_probe.snapshot_rms()

            self.optimizer.step()

            running += loss.item()
            metrics.update(logits.detach(), targets)

        return running / len(loader), metrics.compute(), gradients, gradient_rms

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        metrics = MetricSet(self.num_classes)
        running = 0.0

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(inputs)
            running += self.loss_fn(logits, targets).item()
            metrics.update(logits, targets)

        self.last_confusion = metrics.confusion
        return running / len(loader), metrics.compute()

    def overfit_batch(self, loader: DataLoader, steps: int = 400) -> OverfitResult:
        """Drive a single batch to near-zero loss.

        The correctness gate for the whole engine: if this fails, the loop, the
        loss or the label mapping is wrong and no later result means anything.
        Accuracy is measured on the very batch that was trained -- taking a
        fresh batch from a shuffled loader would silently measure something else.
        """
        inputs, targets = next(iter(loader))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.model.train()
        losses = []
        for _ in range(steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model(inputs), targets)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        self.model.eval()
        with torch.no_grad():
            accuracy = float((self.model(inputs).argmax(dim=1) == targets).float().mean())
        return OverfitResult(losses=losses, accuracy=accuracy)

    def activation_stats(self, loader: DataLoader) -> dict[str, float]:
        """Per-layer activation std on one batch."""
        inputs, _ = next(iter(loader))
        self.model.eval()
        with ActivationStatsProbe(self.model) as probe, torch.no_grad():
            self.model(inputs.to(self.device))
            return {name: std for name, (_mean, std) in probe.snapshot().items()}
