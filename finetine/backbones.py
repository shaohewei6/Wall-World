"""Backbone registry and a shared spatial-feature interface.

`VisualBackbone` is intentionally defined in terms of output features rather
than ViT-specific concepts.  A ViT returns one feature per patch; a CNN returns
one feature per cell in its final feature map.  Both are represented as
``[batch, locations, channels]`` plus an optional two-dimensional grid shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


class VisualBackbone(nn.Module, ABC):
    """Common contract required by the Finetine inference pipeline."""

    backbone_id: str

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Number of channels in each spatial feature."""

    @abstractmethod
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Convert raw RGB image tensors to this backbone's input space."""

    @abstractmethod
    def forward_features(self, normalized_images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None, torch.Tensor | None]:
        """Return spatial features, optional grid shape, and optional global feature.

        The spatial tensor must be shaped ``[B, N, D]``.  ``grid_shape`` is
        ``(height, width)`` when the N locations form a regular feature grid.
        """

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None, torch.Tensor | None]:
        return self.forward_features(self.preprocess(images))


class CNNFeatureMapBackbone(VisualBackbone):
    """Adapter for a CNN or any module exposing a spatial feature map.

    The wrapped module can return either ``[B,C,H,W]`` (the normal CNN case),
    ``[B,N,D]`` (already flattened features), or a mapping whose requested key
    contains one of these tensors.  This lets a future ResNet/ConvNeXt model
    participate in exactly the same Finetine inference and LoRA checkpoint
    flow as DINOv2.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        *,
        feature_dim: int,
        backbone_id: str = "cnn_feature_map",
        output_key: str | None = None,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self._feature_dim = int(feature_dim)
        self.backbone_id = backbone_id
        self.output_key = output_key
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1))

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected raw RGB images [B,3,H,W], got {tuple(images.shape)}")
        images = images.to(dtype=torch.float32)
        if images.numel() and float(images.detach().amax()) > 1.0:
            images = images / 255.0
        return (images - self.image_mean) / self.image_std

    def forward_features(self, normalized_images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None, torch.Tensor | None]:
        output = self.feature_extractor(normalized_images)
        if isinstance(output, dict):
            if self.output_key is None:
                raise ValueError("CNN feature extractor returned a mapping; set output_key when constructing the adapter.")
            output = output[self.output_key]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"CNN feature extractor must return a tensor or mapping, got {type(output)!r}")
        if output.ndim == 4:
            batch, channels, height, width = output.shape
            if channels != self.feature_dim:
                raise ValueError(f"CNN output has {channels} channels, expected feature_dim={self.feature_dim}")
            return output.flatten(2).transpose(1, 2), (height, width), output.mean(dim=(2, 3))
        if output.ndim == 3:
            if output.shape[-1] != self.feature_dim:
                raise ValueError(f"CNN output has feature dimension {output.shape[-1]}, expected {self.feature_dim}")
            return output, None, output.mean(dim=1)
        raise ValueError(f"CNN output must have shape [B,C,H,W] or [B,N,D], got {tuple(output.shape)}")


BackboneFactory = Callable[..., VisualBackbone]
_BACKBONE_REGISTRY: dict[str, BackboneFactory] = {}


def register_backbone(name: str) -> Callable[[BackboneFactory], BackboneFactory]:
    """Register a factory without modifying Finetine's generic pipeline."""

    def decorator(factory: BackboneFactory) -> BackboneFactory:
        if name in _BACKBONE_REGISTRY:
            raise KeyError(f"Backbone factory already registered: {name}")
        _BACKBONE_REGISTRY[name] = factory
        return factory

    return decorator


def create_backbone(name: str, **kwargs: Any) -> VisualBackbone:
    try:
        return _BACKBONE_REGISTRY[name](**kwargs)
    except KeyError as error:
        available = ", ".join(sorted(_BACKBONE_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown Finetine backbone {name!r}; registered backbones: {available}") from error

