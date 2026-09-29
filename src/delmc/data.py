"""EuroSAT: download, stratified split, normalization and loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

from .utils import available_backends

# Anchored to the repo root so the path holds wherever the notebook is run from.
REPO_ROOT = Path(__file__).resolve().parents[2]
NORM_CACHE = REPO_ROOT / "results" / "normalization.json"


class TransformedSubset(Dataset):
    """A view on the shared dataset that applies its own transform.

    The base EuroSAT instance is scanned once and yields PIL images; keeping it
    shared avoids re-listing 27k files for every loader that gets built.
    """

    def __init__(self, base, indices, transform) -> None:
        self.base = base
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position):
        image, target = self.base[int(self.indices[position])]
        return self.transform(image), target


@dataclass(frozen=True)
class Split:
    """Index sets into the full dataset."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


class EuroSATDataModule:
    """Owns the dataset, its stratified split and the loaders built from it.

    Normalization statistics are computed from the training split alone, so no
    information from validation or test leaks into the inputs.
    """

    def __init__(
        self,
        root: str | Path = "data",
        seed: int = 42,
        fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
        batch_size: int = 64,
        num_workers: int = 4,
        train_subset: int | None = None,
    ) -> None:
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError(f"fractions must sum to 1, got {fractions}")
        self.root = Path(root)
        self.seed = seed
        self.fractions = fractions
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_subset = train_subset

        self._raw: datasets.EuroSAT | None = None
        self._split: Split | None = None
        self._stats: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    # -- setup ---------------------------------------------------------------

    def prepare(self) -> "EuroSATDataModule":
        """Download if needed, build the split, and resolve normalization stats."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._raw = datasets.EuroSAT(root=str(self.root), download=True)
        self._split = self._build_split()
        self._stats = self._resolve_stats()
        return self

    def _build_split(self) -> Split:
        targets = np.asarray(self._raw.targets)
        indices = np.arange(len(targets))
        train_frac, val_frac, test_frac = self.fractions

        train_idx, holdout_idx = train_test_split(
            indices,
            train_size=train_frac,
            stratify=targets,
            random_state=self.seed,
        )
        # Re-normalise so val/test keep their intended ratio within the holdout.
        val_share = val_frac / (val_frac + test_frac)
        val_idx, test_idx = train_test_split(
            holdout_idx,
            train_size=val_share,
            stratify=targets[holdout_idx],
            random_state=self.seed,
        )

        if self.train_subset is not None and self.train_subset < len(train_idx):
            train_idx, _ = train_test_split(
                train_idx,
                train_size=self.train_subset,
                stratify=targets[train_idx],
                random_state=self.seed,
            )

        return Split(train=np.sort(train_idx), val=np.sort(val_idx), test=np.sort(test_idx))

    def _resolve_stats(self):
        key = self._stats_key()
        if NORM_CACHE.exists():
            cached = json.loads(NORM_CACHE.read_text())
            if cached.get("key") == key:
                return tuple(cached["mean"]), tuple(cached["std"])

        mean, std = self._compute_stats()
        NORM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        NORM_CACHE.write_text(json.dumps({"key": key, "mean": list(mean), "std": list(std)}, indent=2))
        return mean, std

    def _stats_key(self) -> str:
        """Identifies the exact training split the statistics belong to."""
        return f"seed={self.seed};fractions={self.fractions};subset={self.train_subset}"

    def _compute_stats(self):
        to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        dataset = TransformedSubset(self._raw, self._split.train, to_tensor)
        loader = DataLoader(dataset, batch_size=256, num_workers=self.num_workers)

        total = 0
        channel_sum = torch.zeros(3, dtype=torch.float64)
        channel_sq = torch.zeros(3, dtype=torch.float64)
        for images, _ in loader:
            pixels = images.to(torch.float64).permute(1, 0, 2, 3).reshape(3, -1)
            total += pixels.shape[1]
            channel_sum += pixels.sum(dim=1)
            channel_sq += (pixels**2).sum(dim=1)

        mean = channel_sum / total
        std = (channel_sq / total - mean**2).clamp_min(0).sqrt()
        return tuple(mean.tolist()), tuple(std.tolist())

    # -- transforms ----------------------------------------------------------

    @property
    def normalization(self):
        self._require_prepared()
        return self._stats

    def eval_transform(self):
        mean, std = self.normalization
        return v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=list(mean), std=list(std))]
        )

    def train_transform(self, augment: list | None = None):
        """Augmentation runs on the raw image, before tensor conversion."""
        mean, std = self.normalization
        steps = list(augment or [])
        steps += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=list(mean), std=list(std))]
        return v2.Compose(steps)

    # -- loaders -------------------------------------------------------------

    def train_loader(self, augment: list | None = None, batch_size: int | None = None) -> DataLoader:
        return self._loader(self._split.train, self.train_transform(augment), shuffle=True, batch_size=batch_size)

    def val_loader(self, batch_size: int | None = None) -> DataLoader:
        return self._loader(self._split.val, self.eval_transform(), shuffle=False, batch_size=batch_size)

    def test_loader(self, batch_size: int | None = None) -> DataLoader:
        return self._loader(self._split.test, self.eval_transform(), shuffle=False, batch_size=batch_size)

    def subset_loader(self, indices, transform=None, shuffle: bool = True, batch_size: int | None = None) -> DataLoader:
        """Loader over arbitrary indices; cross-validation builds its folds with this."""
        return self._loader(np.asarray(indices), transform or self.eval_transform(), shuffle, batch_size)

    def _loader(self, indices, transform, shuffle: bool, batch_size: int | None) -> DataLoader:
        self._require_prepared()
        dataset = TransformedSubset(self._raw, np.asarray(indices), transform)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=bool(available_backends()),
            persistent_workers=self.num_workers > 0,
        )

    # -- description ---------------------------------------------------------

    @property
    def split(self) -> Split:
        self._require_prepared()
        return self._split

    @property
    def targets(self) -> np.ndarray:
        """Label of every image in the full dataset, indexed like the splits."""
        self._require_prepared()
        return np.asarray(self._raw.targets)

    @property
    def class_names(self) -> list[str]:
        self._require_prepared()
        return list(self._raw.classes)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_counts(self, which: str = "all") -> dict[str, int]:
        """Images per class, for the full dataset or one split."""
        self._require_prepared()
        targets = np.asarray(self._raw.targets)
        if which != "all":
            targets = targets[getattr(self._split, which)]
        counts = np.bincount(targets, minlength=self.num_classes)
        return dict(zip(self.class_names, (int(c) for c in counts)))

    def _require_prepared(self) -> None:
        if self._split is None:
            raise RuntimeError("call prepare() first")
