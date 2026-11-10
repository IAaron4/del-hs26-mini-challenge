"""Experiment specification, result caching, and the runner that ties them together."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

import torch

from .data import AUGMENTATIONS
from .engine import History, Trainer
from .models import CNNConfig, ConfigurableCNN
from .utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


def build_optimizer(name: str, parameters, lr: float, momentum: float = 0.0, weight_decay: float = 0.0):
    parameters = [p for p in parameters if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


class CacheKey:
    """Hash of a run config, for frozen dataclasses with `name` and `group` fields."""

    def identity(self) -> dict:
        """All fields except `name` and `group`, which are only labels."""
        payload = asdict(self)
        payload.pop("name")
        payload.pop("group")
        return payload

    def key(self) -> str:
        blob = json.dumps(self.identity(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ExperimentConfig(CacheKey):
    """Everything that defines one training run of the own CNN."""

    name: str
    group: str
    cnn: CNNConfig = field(default_factory=CNNConfig)
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 20
    seed: int = 42
    augment: tuple[str, ...] = ()
    train_subset: int | None = 6000
    probe: bool = False

    def with_(self, **changes) -> "ExperimentConfig":
        return replace(self, **changes)

    def build_model(self) -> torch.nn.Module:
        return ConfigurableCNN(self.cnn)

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return build_optimizer(self.optimizer, model.parameters(), self.lr, self.momentum, self.weight_decay)


class ResultCache:
    """Run histories as JSON under results/<group>/<name>-<key>.json."""

    def __init__(self, root: Path | str = RESULTS_ROOT) -> None:
        self.root = Path(root)

    def path(self, config: ExperimentConfig) -> Path:
        return self.root / config.group / f"{config.name}-{config.key()}.json"

    def get(self, config: ExperimentConfig) -> History | None:
        path = self.path(config)
        return History.load(path) if path.exists() else None

    def put(self, config: ExperimentConfig, history: History) -> Path:
        return history.save(self.path(config))

    def group(self, group: str) -> dict[str, History]:
        """Every cached run in one hypothesis group, keyed by run name."""
        directory = self.root / group
        if not directory.exists():
            return {}
        return {path.stem.rsplit("-", 1)[0]: History.load(path) for path in sorted(directory.glob("*.json"))}


class ExperimentRunner:
    """Trains an ExperimentConfig, or loads the result if it is already in the cache."""

    def __init__(
        self,
        data_module,
        device: torch.device,
        cache: ResultCache | None = None,
        tracker_factory: Callable[[ExperimentConfig], object] | None = None,
        augmentations: dict[str, object] | None = None,
    ) -> None:
        self.data_module = data_module
        self.device = device
        self.cache = cache or ResultCache()
        self.tracker_factory = tracker_factory
        self.augmentations = augmentations or AUGMENTATIONS

    def run(self, config: ExperimentConfig, force: bool = False, verbose: bool = True) -> History:
        if not force:
            cached = self.cache.get(config)
            if cached is not None:
                if verbose:
                    print(f"cached  {config.group}/{config.name}")
                return cached

        # The subset is set on the data module but is part of the cache key, so both have to agree.
        actual = getattr(self.data_module, "train_subset", None)
        if actual != config.train_subset:
            raise ValueError(
                f"data module was built with train_subset={actual}, "
                f"but {config.group}/{config.name} declares {config.train_subset}"
            )

        seed_everything(config.seed)
        model = config.build_model()
        trainer = Trainer(
            model=model,
            optimizer=config.build_optimizer(model),
            device=self.device,
            num_classes=self.data_module.num_classes,
            tracker=self.tracker_factory(config) if self.tracker_factory else None,
            probe=config.probe,
        )

        transforms = [self.augmentations[name] for name in config.augment]
        train_loader = self.data_module.train_loader(augment=transforms, batch_size=config.batch_size)
        val_loader = self.data_module.val_loader(batch_size=max(config.batch_size, 256))

        history = trainer.fit(
            train_loader,
            val_loader,
            config.epochs,
            config={**config.identity(), "name": config.name, "group": config.group,
                    "params": sum(p.numel() for p in model.parameters())},
        )
        history.final_train = trainer.final_train_metrics(
            self.data_module.train_eval_loader(batch_size=max(config.batch_size, 256))
        )
        self.cache.put(config, history)
        # The weights are kept so the test evaluation does not need a retrain.
        checkpoint = self.cache.path(config).with_suffix(".pt")
        torch.save(model.state_dict(), checkpoint)
        if verbose:
            last = history.records[-1]
            print(f"ran     {config.group}/{config.name}  val_acc={last.val_accuracy:.4f}")
        return history

    def run_many(self, configs: Iterable[ExperimentConfig], force: bool = False) -> dict[str, History]:
        return {config.name: self.run(config, force=force) for config in configs}
