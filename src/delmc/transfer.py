"""Transfer learning from an ImageNet ResNet18, with the frozen part computed once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.transforms import v2

from .engine import History, Trainer
from .experiments import CacheKey, ResultCache, build_optimizer
from .utils import seed_everything

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Mode = Literal["frozen", "finetune_last"]


def imagenet_transform(size: int = 224):
    """Upscaled to the resolution the pretrained filters were trained on."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size, antialias=True),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])


def split_resnet(mode: Mode, num_classes: int = 10) -> tuple[nn.Module, nn.Module]:
    """Frozen prefix and trainable suffix of an ImageNet ResNet18 with a new head."""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    stem = [net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3]
    head = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(net.fc.in_features, num_classes)]
    if mode == "frozen":
        return nn.Sequential(*stem, net.layer4, *head[:2]), head[2]
    return nn.Sequential(*stem), nn.Sequential(net.layer4, *head)


@torch.no_grad()
def extract(prefix: nn.Module, loader: DataLoader, device: torch.device) -> TensorDataset:
    prefix.eval().to(device)
    features, labels = [], []
    for inputs, targets in loader:
        features.append(prefix(inputs.to(device)).cpu())
        labels.append(targets)
    return TensorDataset(torch.cat(features), torch.cat(labels))


@dataclass(frozen=True)
class TransferConfig(CacheKey):
    """One transfer-learning run; cached like an ExperimentConfig."""

    name: str
    group: str
    mode: Mode = "frozen"
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.0
    batch_size: int = 64
    epochs: int = 30
    seed: int = 42
    image_size: int = 224
    train_subset: int | None = 6000


class TransferRunner:
    """Trains the trainable part on features of the frozen part, computed once per run.

    The frozen part does not change, so its output is the same in every epoch.
    Augmentation is not possible this way.
    """

    def __init__(self, data_module, device: torch.device, cache: ResultCache | None = None,
                 tracker_factory=None) -> None:
        self.data_module = data_module
        self.device = device
        self.cache = cache or ResultCache()
        self.tracker_factory = tracker_factory
        self._features: dict[tuple[str, int], dict[str, TensorDataset]] = {}

    def features(self, mode: Mode, image_size: int) -> dict[str, TensorDataset]:
        if (mode, image_size) not in self._features:
            prefix, _ = split_resnet(mode, self.data_module.num_classes)
            transform = imagenet_transform(image_size)
            split = self.data_module.split
            self._features[(mode, image_size)] = {
                name: extract(
                    prefix,
                    self.data_module.subset_loader(getattr(split, name), transform, shuffle=False, batch_size=128),
                    self.device,
                )
                for name in ("train", "val")
            }
        return self._features[(mode, image_size)]

    def run(self, config: TransferConfig, force: bool = False, verbose: bool = True) -> History:
        if not force:
            cached = self.cache.get(config)
            if cached is not None:
                if verbose:
                    print(f"cached  {config.group}/{config.name}")
                return cached

        actual = getattr(self.data_module, "train_subset", None)
        if actual != config.train_subset:
            raise ValueError(f"data module was built with train_subset={actual}, config declares {config.train_subset}")

        features = self.features(config.mode, config.image_size)
        seed_everything(config.seed)
        _, model = split_resnet(config.mode, self.data_module.num_classes)
        trainer = Trainer(
            model=model,
            optimizer=build_optimizer(config.optimizer, model.parameters(), config.lr, config.momentum),
            device=self.device,
            num_classes=self.data_module.num_classes,
            tracker=self.tracker_factory(config) if self.tracker_factory else None,
        )
        history = trainer.fit(
            DataLoader(features["train"], batch_size=config.batch_size, shuffle=True),
            DataLoader(features["val"], batch_size=256),
            config.epochs,
            config={**config.identity(), "name": config.name, "group": config.group,
                    "params": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        )
        history.final_train = trainer.final_train_metrics(DataLoader(features["train"], batch_size=256))
        self.cache.put(config, history)
        # Only the trained part, the frozen part is the unchanged ImageNet network.
        torch.save(model.state_dict(), self.cache.path(config).with_suffix(".pt"))
        if verbose:
            print(f"ran     {config.group}/{config.name}  val_acc={history.records[-1].val_accuracy:.4f}")
        return history
