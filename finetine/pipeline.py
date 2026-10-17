"""Inference-only assembly of a backbone and an optional Finetine adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .backbones import VisualBackbone, create_backbone
from .checkpoint import apply_adapter_checkpoint, read_adapter_checkpoint


@dataclass(frozen=True)
class EncodedFeatures:
    """Backbone-neutral visual representation returned by inference."""

    spatial_features: torch.Tensor  # [B, N, D]
    grid_shape: tuple[int, int] | None
    global_feature: torch.Tensor | None  # [B, D] when provided by backbone
    backbone_id: str
    adapter_loaded: bool

    def __post_init__(self) -> None:
        if self.spatial_features.ndim != 3:
            raise ValueError(f"spatial_features must be [B,N,D], got {tuple(self.spatial_features.shape)}")
        if self.grid_shape is not None and self.grid_shape[0] * self.grid_shape[1] != self.spatial_features.shape[1]:
            raise ValueError("grid_shape does not match the spatial feature count")


class FinetineInferencePipeline:
    """Loads an arbitrary registered backbone plus an optional LoRA adapter."""

    def __init__(self, backbone: VisualBackbone, *, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.backbone = backbone.to(self.device).eval()
        self.adapter_loaded = False
        self.adapter_metadata: dict[str, Any] = {}

    @classmethod
    def from_registry(cls, backbone_name: str, *, device: torch.device | str = "cpu", **backbone_kwargs: Any) -> "FinetineInferencePipeline":
        return cls(create_backbone(backbone_name, **backbone_kwargs), device=device)

    def load_adapter(self, checkpoint_path: str | Path) -> list[str]:
        if self.adapter_loaded:
            raise RuntimeError("This pipeline already has an adapter; construct a new pipeline to load another one.")
        payload = read_adapter_checkpoint(checkpoint_path)
        wrapped = apply_adapter_checkpoint(self.backbone, backbone_id=self.backbone.backbone_id, payload=payload)
        self.backbone.to(self.device).eval()
        self.adapter_loaded = True
        self.adapter_metadata = dict(payload.get("metadata", {}))
        return wrapped

    @torch.inference_mode()
    def encode(self, raw_images: torch.Tensor, *, return_cpu: bool = True) -> EncodedFeatures:
        raw_images = raw_images.to(self.device, non_blocking=True)
        spatial, grid_shape, global_feature = self.backbone(raw_images)
        if return_cpu:
            spatial = spatial.float().cpu()
            global_feature = global_feature.float().cpu() if global_feature is not None else None
        return EncodedFeatures(
            spatial_features=spatial,
            grid_shape=grid_shape,
            global_feature=global_feature,
            backbone_id=self.backbone.backbone_id,
            adapter_loaded=self.adapter_loaded,
        )
