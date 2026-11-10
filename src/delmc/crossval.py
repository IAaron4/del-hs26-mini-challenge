"""Stratified k-fold cross-validation on the training split."""

from __future__ import annotations

from typing import Callable

import torch
from sklearn.model_selection import StratifiedKFold

from .engine import History, Trainer
from .utils import seed_everything


class CrossValidator:
    """Trains a fresh model per fold. The folds come from the training split only."""

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

    def run(self, verbose: bool = True) -> list[History]:
        train_indices = self.data_module.split.train
        labels = self.data_module.targets[train_indices]
        splitter = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)

        histories = []
        for fold, (inner_train, inner_val) in enumerate(splitter.split(train_indices, labels)):
            # Same seed in every fold, so only the data split changes.
            seed_everything(self.seed)

            model = self.model_factory()
            trainer = Trainer(
                model=model,
                optimizer=self.optimizer_factory(model),
                device=self.device,
                num_classes=self.data_module.num_classes,
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
            histories.append(history)
            if verbose:
                last = history.records[-1]
                print(f"fold {fold}: val_acc={last.val_accuracy:.4f} val_f1={last.val_macro_f1:.4f}")

        return histories
