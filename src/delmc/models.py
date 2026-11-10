"""The configurable CNN behind the baseline and all architecture variants."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import nn

Activation = Literal["relu", "tanh", "sigmoid"]
Pooling = Literal["max", "avg"]
InitScheme = Literal["default", "kaiming"]

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


@dataclass(frozen=True)
class CNNConfig:
    """Architecture settings; every field is part of the cache key of a run.

    `stage_widths` fixes the number of pooling stages. `convs_per_stage` makes the
    network deeper without changing how often the image is downsampled.
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
    def spatial_out(self) -> int:
        return self.image_size // 2 ** len(self.stage_widths)

    def with_(self, **changes) -> "CNNConfig":
        """Copy with some fields changed."""
        return replace(self, **changes)


class ConvStage(nn.Module):
    """Convolutions at one resolution, then pooling that halves the image."""

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

        # 1x1 convolution only when the shortcut has to change the number of channels.
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if config.residual and in_channels != out_channels
            else nn.Identity()
        )
        self.downsample = nn.MaxPool2d(2) if config.pooling == "max" else nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        if self.residual:
            out = out + self.shortcut(x)
        return self.downsample(out)


class ConfigurableCNN(nn.Module):
    """Conv stages for the features, fully connected layers for the classification."""

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
        # "default" keeps PyTorch's own initialisation, kaiming_uniform_ with a = sqrt(5).
        if self.config.init == "default" or not isinstance(module, (nn.Conv2d, nn.Linear)):
            return
        if self.config.init != "kaiming":
            raise ValueError(f"unknown init scheme {self.config.init!r}")
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity=self.config.activation)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
