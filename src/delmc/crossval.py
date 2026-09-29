"""Stratified K-fold cross-validation for the statistical error estimate."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .engine import History, Trainer
from .utils import seed_everything


@dataclass
class CVResult:
    """Per-fold histories plus the aggregate the spec asks for."""

    folds: list[History] = field(default_factory=list)

    def final(self, metric: str = "val_accuracy") -> list[float]:
        return [history.records[-1].__getattribute__(metric) for history in self.folds]

    def best(self, metric: str = "val_accuracy") -> list[float]:
        return [getattr(history.best(metric), metric) for history in self.folds]

    def mean_std(self, metric: str = "val_accuracy", use_best: bool = False) -> tuple[float, float]:
        """Mean and sample standard deviation across folds."""
        values = self.best(metric) if use_best else self.final(metric)
        if len(values) < 2:
            raise ValueError("need at least two folds for a standard deviation")
        return statistics.fmean(values), statistics.stdev(values)

    def summary(self, metrics=("val_accuracy", "val_macro_f1"), use_best: bool = False) -> dict[str, dict[str, float]]:
        return {
            metric: dict(zip(("mean", "std"), self.mean_std(metric, use_best)))
            for metric in metrics
        }


class CrossValidator:
    """Retrains the model once per fold and aggregates to mean ± std.

    Folds are drawn from the training split only, so the test split stays
    untouched until the very end.
    """

    def __init__(
        self,
        data_module,
        model_factory: Callable[[], torch.nn.Module],
        optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
        device: torch.device,
        folds: int = 5,
        epochs: int = 20,
        seed: int = 42,
        batch_size: int | None = None,
    ) -> None:
        self.data_module = data_module
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.device = device
        self.folds = folds
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size

    def run(self, tracker_factory: Callable[[int], object] | None = None, verbose: bool = True) -> CVResult:
        train_indices = self.data_module.split.train
        labels = self.data_module.targets[train_indices]
        splitter = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)

        result = CVResult()
        for fold, (inner_train, inner_val) in enumerate(splitter.split(train_indices, labels)):
            # Same starting seed per fold, so variation comes from the data split
            # rather than from initialisation.
            seed_everything(self.seed)

            model = self.model_factory()
            trainer = Trainer(
                model=model,
                optimizer=self.optimizer_factory(model),
                device=self.device,
                num_classes=self.data_module.num_classes,
                tracker=tracker_factory(fold) if tracker_factory else None,
            )

            train_loader = self.data_module.subset_loader(
                train_indices[inner_train],
                transform=self.data_module.train_transform(),
                shuffle=True,
                batch_size=self.batch_size,
            )
            val_loader = self.data_module.subset_loader(
                train_indices[inner_val], shuffle=False, batch_size=self.batch_size
            )

            history = trainer.fit(train_loader, val_loader, self.epochs, config={"fold": fold})
            result.folds.append(history)
            if verbose:
                last = history.records[-1]
                print(f"fold {fold}: val_acc={last.val_accuracy:.4f} val_f1={last.val_macro_f1:.4f}")

        return result
