"""One configurable CNN, plus the pretrained variants for transfer learning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import nn

Activation = Literal["relu", "tanh", "sigmoid"]
Pooling = Literal["max", "avg", "stride"]
InitScheme = Literal["default", "kaiming", "xavier", "small"]

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


@dataclass(frozen=True)
class CNNConfig:
    """Every axis the architecture hypotheses vary.

    The number of downsampling stages is fixed by `stage_widths`, so changing
    `convs_per_stage` changes depth without changing the spatial reduction --
    otherwise "deeper" would silently also mean "differently pooled".
    """

    stage_widths: tuple[int, ...] = (32, 64, 128)
    convs_per_stage: int = 1
    kernel_size: int = 3
    pooling: Pooling = "max"
    fc_widths: tuple[int, ...] = (256,)
    dropout: float = 0.0
    conv_dropout: float = 0.0
    batch_norm: bool = False
    activation: Activation = "relu"
    init: InitScheme = "default"
    residual: bool = False
    num_classes: int = 10
    image_size: int = 64
    in_channels: int = 3

    def __post_init__(self) -> None:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so padding preserves the spatial size")
        if self.image_size % 2 ** len(self.stage_widths) != 0:
            raise ValueError(f"image_size {self.image_size} is not divisible by {2 ** len(self.stage_widths)}")

    @property
    def conv_depth(self) -> int:
        return len(self.stage_widths) * self.convs_per_stage

    @property
    def spatial_out(self) -> int:
        return self.image_size // 2 ** len(self.stage_widths)

    def with_(self, **changes) -> "CNNConfig":
        """A copy with fields replaced; how hypotheses derive variants from the baseline."""
        return replace(self, **changes)


class ConvStage(nn.Module):
    """Convolutions at one resolution, followed by a halving downsample."""

    def __init__(self, in_channels: int, out_channels: int, config: CNNConfig) -> None:
        super().__init__()
        self.residual = config.residual
        padding = config.kernel_size // 2

        layers = []
        channels = in_channels
        for _ in range(config.convs_per_stage):
            layers.append(nn.Conv2d(channels, out_channels, config.kernel_size, padding=padding, bias=not config.batch_norm))
            if config.batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(_ACTIVATIONS[config.activation]())
            if config.conv_dropout > 0:
                layers.append(nn.Dropout2d(config.conv_dropout))
            channels = out_channels
        self.body = nn.Sequential(*layers)

        # 1x1 projection only when the shortcut has to change channel count.
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if config.residual and in_channels != out_channels
            else nn.Identity()
        )
        self.downsample = self._downsample(out_channels, config)

    @staticmethod
    def _downsample(channels: int, config: CNNConfig) -> nn.Module:
        if config.pooling == "max":
            return nn.MaxPool2d(2)
        if config.pooling == "avg":
            return nn.AvgPool2d(2)
        return nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        if self.residual:
            out = out + self.shortcut(x)
        return self.downsample(out)


class ConfigurableCNN(nn.Module):
    """Conv feature extractor plus a fully connected classifier head."""

    def __init__(self, config: CNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or CNNConfig()

        stages = []
        channels = self.config.in_channels
        for width in self.config.stage_widths:
            stages.append(ConvStage(channels, width, self.config))
            channels = width
        self.features = nn.Sequential(*stages)

        head: list[nn.Module] = [nn.Flatten()]
        size = self.config.stage_widths[-1] * self.config.spatial_out**2
        for width in self.config.fc_widths:
            head.append(nn.Linear(size, width))
            head.append(_ACTIVATIONS[self.config.activation]())
            if self.config.dropout > 0:
                head.append(nn.Dropout(self.config.dropout))
            size = width
        head.append(nn.Linear(size, self.config.num_classes))
        self.classifier = nn.Sequential(*head)

        self.apply(self._initialise)

    def _initialise(self, module: nn.Module) -> None:
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            return
        scheme = self.config.init
        # "default" deliberately leaves PyTorch's kaiming_uniform_(a=sqrt(5)) in
        # place; the init hypothesis measures it against the alternatives.
        if scheme == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity=self.config.activation)
        elif scheme == "xavier":
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain(self.config.activation))
        elif scheme == "small":
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif scheme != "default":
            raise ValueError(f"unknown init scheme {scheme!r}")

        if scheme != "default" and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PretrainedCNN(nn.Module):
    """ImageNet ResNet18 with a new head, frozen or partially fine-tuned."""

    def __init__(self, num_classes: int = 10, mode: Literal["frozen", "finetune_last"] = "frozen") -> None:
        super().__init__()
        from torchvision import models

        self.mode = mode
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        for name, parameter in self.backbone.named_parameters():
            trainable = name.startswith("fc") or (mode == "finetune_last" and name.startswith("layer4"))
            parameter.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
