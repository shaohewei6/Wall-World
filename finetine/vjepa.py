"""LoRA-efficient V-JEPA training components for temporal visual adaptation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .dinov2 import DinoV2Backbone
from .lora import LoRAConfig, inject_lora_modules, lora_parameters


@dataclass(frozen=True)
class SpatiotemporalMaskConfig:
    """Target-mask geometry expressed in frame and DINO patch coordinates."""

    future_target_frames: int = 2
    spatial_blocks_per_clip: int = 2
    spatial_block_min: int = 2
    spatial_block_max: int = 4

    def __post_init__(self) -> None:
        if self.future_target_frames < 0 or self.spatial_blocks_per_clip < 0:
            raise ValueError("Mask counts must be non-negative")
        if self.spatial_block_min < 1 or self.spatial_block_max < self.spatial_block_min:
            raise ValueError("Invalid spatial block size range")


class SpatiotemporalBlockMasker:
    """Build future-frame and local spatial target blocks for each video clip."""

    def __init__(self, config: SpatiotemporalMaskConfig) -> None:
        self.config = config

    def __call__(self, batch: int, time: int, grid_shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        height, width = grid_shape
        if self.config.future_target_frames >= time:
            raise ValueError("future_target_frames must be smaller than clip length")
        mask = torch.zeros((batch, time, height, width), device=device, dtype=torch.bool)
        if self.config.future_target_frames:
            mask[:, time - self.config.future_target_frames :] = True
        visible_times = max(time - self.config.future_target_frames, 1)
        for batch_index in range(batch):
            for _ in range(self.config.spatial_blocks_per_clip):
                block_h = int(torch.randint(self.config.spatial_block_min, self.config.spatial_block_max + 1, (1,), device=device))
                block_w = int(torch.randint(self.config.spatial_block_min, self.config.spatial_block_max + 1, (1,), device=device))
                block_h, block_w = min(block_h, height), min(block_w, width)
                frame = int(torch.randint(visible_times, (1,), device=device))
                top = int(torch.randint(height - block_h + 1, (1,), device=device))
                left = int(torch.randint(width - block_w + 1, (1,), device=device))
                mask[batch_index, frame, top : top + block_h, left : left + block_w] = True
        if not bool(mask.any()):
            raise RuntimeError("V-JEPA mask contains no target tokens")
        return mask.flatten(2)


class TemporalLatentPredictor(nn.Module):
    """Predict masked target tokens from a masked sequence of spatial tokens."""

    def __init__(
        self,
        *,
        feature_dim: int,
        max_clip_length: int,
        max_spatial_locations: int,
        predictor_dim: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if predictor_dim % heads:
            raise ValueError("predictor_dim must be divisible by heads")
        self.max_clip_length = int(max_clip_length)
        self.input_projection = nn.Identity() if feature_dim == predictor_dim else nn.Linear(feature_dim, predictor_dim)
        self.temporal_position = nn.Parameter(torch.zeros(1, max_clip_length, 1, predictor_dim))
        self.spatial_position = nn.Parameter(torch.zeros(1, 1, max_spatial_locations, predictor_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=heads,
            dim_feedforward=4 * predictor_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(predictor_dim))
        self.output_projection = nn.Sequential(nn.LayerNorm(predictor_dim), nn.Linear(predictor_dim, feature_dim))
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        nn.init.trunc_normal_(self.spatial_position, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, time, locations, _ = tokens.shape
        if time > self.max_clip_length:
            raise ValueError(f"Clip length {time} exceeds predictor maximum {self.max_clip_length}")
        if locations > self.spatial_position.shape[2]:
            raise ValueError(f"Feature locations {locations} exceed predictor maximum {self.spatial_position.shape[2]}")
        value = self.input_projection(tokens)
        value = value + self.temporal_position[:, :time] + self.spatial_position[:, :, :locations]
        value = self.transformer(value.reshape(batch, time * locations, value.shape[-1]))
        return self.output_projection(value).reshape(batch, time, locations, -1)


class VJEPAAdapterModel(nn.Module):
    """DINO-initialized context/target encoders with a trainable latent predictor.

    The DINO base weights remain frozen.  The context encoder receives gradient
    only through LoRA modules; the target encoder is an EMA copy and never
    receives gradients.  This makes the model a V-JEPA-style *domain-adapter*
    rather than a full from-scratch V-JEPA pretraining run.
    """

    def __init__(
        self,
        context_backbone: DinoV2Backbone,
        *,
        lora_config: LoRAConfig,
        clip_length: int,
        predictor_dim: int = 384,
        predictor_depth: int = 4,
        predictor_heads: int = 8,
        predictor_dropout: float = 0.0,
        max_spatial_locations: int = 256,
        mask_config: SpatiotemporalMaskConfig = SpatiotemporalMaskConfig(),
    ) -> None:
        super().__init__()
        context_backbone.requires_grad_(False)
        self.context_encoder = context_backbone
        self.wrapped_modules = inject_lora_modules(self.context_encoder, lora_config)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.lora_config = lora_config
        self.masker = SpatiotemporalBlockMasker(mask_config)
        self.predictor = TemporalLatentPredictor(
            feature_dim=context_backbone.feature_dim,
            max_clip_length=clip_length,
            max_spatial_locations=max_spatial_locations,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            dropout=predictor_dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Teacher must remain deterministic even while the context branch trains.
        self.target_encoder.eval()
        return self

    def trainable_parameters(self):
        yield from lora_parameters(self.context_encoder)
        yield from self.predictor.parameters()

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if frames.ndim != 5 or frames.shape[2:] != (3, 224, 224):
            raise ValueError(f"Expected raw clips [B,T,3,224,224], got {tuple(frames.shape)}")
        batch, time = frames.shape[:2]
        flattened = frames.flatten(0, 1)
        normalized_context = self.context_encoder.preprocess(flattened)
        grid_shape = (
            normalized_context.shape[-2] // self.context_encoder.patch_size,
            normalized_context.shape[-1] // self.context_encoder.patch_size,
        )
        target_mask = self.masker(batch, time, grid_shape, frames.device)
        context_tokens, _, _ = self.context_encoder.forward_masked_features(
            normalized_context,
            target_mask.flatten(0, 1),
        )
        context_tokens = context_tokens.reshape(batch, time, -1, context_tokens.shape[-1])
        with torch.no_grad():
            normalized_target = self.target_encoder.preprocess(flattened)
            target_tokens, _, _ = self.target_encoder.forward_features(normalized_target)
            target_tokens = target_tokens.reshape(batch, time, -1, target_tokens.shape[-1])
        predicted_tokens = self.predictor(context_tokens)
        masked_prediction = predicted_tokens[target_mask]
        masked_target = target_tokens[target_mask].detach()
        latent_loss = functional.smooth_l1_loss(masked_prediction.float(), masked_target.float())
        cosine = functional.cosine_similarity(masked_prediction.float(), masked_target.float(), dim=-1).mean()
        metrics = {
            "latent_loss": latent_loss.detach(),
            "masked_token_count": target_mask.sum().detach(),
            "target_cosine": cosine.detach(),
        }
        return latent_loss, metrics

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("EMA momentum must lie in [0, 1)")
        source = dict(self.context_encoder.named_parameters())
        for name, target_parameter in self.target_encoder.named_parameters():
            # Only LoRA can change, but this form is future-proof if an adapter
            # configuration later enables additional trainable parameters.
            target_parameter.mul_(momentum).add_(source[name], alpha=1.0 - momentum)
