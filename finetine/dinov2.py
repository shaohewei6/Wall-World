"""DINOv2-specific backbone and self-supervised-learning building blocks.

This module owns every concern that is specific to the DINOv2 ViT family:
offline checkpoint loading, ImageNet normalization, patch-token extraction, and
native masked-patch encoding.  DINOv2-style SSL training will live here next,
parallel to the V-JEPA implementation in :mod:`finetine.vjepa`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .backbones import VisualBackbone, register_backbone
from .lora import LoRAConfig, inject_lora_modules, lora_parameters

if '.' in __package__:
    from ..model.vendor.dinov2 import vit_small
else:
    from model.vendor.dinov2 import vit_small


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location="cpu")


class DinoV2Backbone(VisualBackbone):
    """Offline DINOv2 ViT-S/14 wrapper with standard and masked token paths."""

    backbone_id = "dinov2_vits14"

    def __init__(self, *, checkpoint_path: str | Path | None = None,
                 repo_path: str | Path | None = None) -> None:
        """Use bundled backbone code; repo_path is retained as a legacy argument.

        No external repository is read, even if an old caller supplies repo_path.
        An explicit checkpoint_path may select other compatible weight assets.
        """
        super().__init__()
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).resolve().parents[1]/'checkpoints/encoder/dinov2_vits14_pretrain.pth'
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DINOv2 checkpoint does not exist: {checkpoint_path}")
        self.base_model = vit_small(img_size=518,patch_size=14,init_values=1.0,
            ffn_layer='mlp',block_chunks=0,num_register_tokens=0,
            interpolate_antialias=False,interpolate_offset=0.1)
        checkpoint = _safe_torch_load(checkpoint_path)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        self.base_model.load_state_dict(checkpoint, strict=True)
        self.patch_size = int(self.base_model.patch_size)
        self._feature_dim = int(self.base_model.num_features)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected raw RGB images [B,3,H,W], got {tuple(images.shape)}")
        images = images.to(dtype=torch.float32)
        if images.numel() and float(images.detach().amax()) > 1.0:
            images = images / 255.0
        if images.numel() and float(images.detach().amin()) < 0.0:
            raise ValueError("Finetine expects raw uint8 RGB or float RGB in [0, 1], not pre-normalized images.")
        return (images - self.image_mean) / self.image_std

    def forward_features(self, normalized_images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor]:
        height, width = normalized_images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"DINOv2 input {(height, width)} must be divisible by patch size {self.patch_size}")
        output = self.base_model.forward_features(normalized_images)
        return output["x_norm_patchtokens"], (height // self.patch_size, width // self.patch_size), output["x_norm_clstoken"]

    def forward_masked_features(
        self,
        normalized_images: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor]:
        """Encode images after replacing ``True`` positions with DINO's mask token."""

        height, width = normalized_images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"DINOv2 input {(height, width)} must be divisible by patch size {self.patch_size}")
        grid_shape = (height // self.patch_size, width // self.patch_size)
        expected = (normalized_images.shape[0], grid_shape[0] * grid_shape[1])
        if tuple(patch_mask.shape) != expected or patch_mask.dtype != torch.bool:
            raise ValueError(f"patch_mask must be bool {expected}, got {tuple(patch_mask.shape)} / {patch_mask.dtype}")
        output = self.base_model.forward_features(normalized_images, masks=patch_mask)
        return output["x_norm_patchtokens"], grid_shape, output["x_norm_clstoken"]


@register_backbone("dinov2_vits14")
def create_dinov2_vits14(**kwargs: Any) -> DinoV2Backbone:
    return DinoV2Backbone(**kwargs)


class DINOProjectionHead(nn.Module):
    """Small prototype head for DINO global and iBOT patch objectives."""

    def __init__(self, feature_dim: int, *, hidden_dim: int = 1024, bottleneck_dim: int = 256, prototypes: int = 2048) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.prototypes = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, prototypes, bias=False))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.prototypes(functional.normalize(self.mlp(features), dim=-1))


class DINOGlobalAugmenter(nn.Module):
    """Two geometry-preserving global views for Wall World image SSL.

    Horizontal/vertical flips and rotations are intentionally absent: they
    would destroy the absolute action-direction semantics needed later by the
    action-conditioned world model.  The two views differ through mild camera
    translation, brightness and contrast only.
    """

    def __init__(self, *, translation_fraction: float = 0.025, brightness: float = 0.08, contrast: float = 0.08) -> None:
        super().__init__()
        self.translation_fraction = float(translation_fraction)
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if images.numel() and float(images.detach().amax()) > 1.0:
            images = images / 255.0
        batch = images.shape[0]
        theta = torch.zeros((batch, 2, 3), dtype=images.dtype, device=images.device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, :, 2] = torch.empty((batch, 2), device=images.device, dtype=images.dtype).uniform_(
            -2.0 * self.translation_fraction, 2.0 * self.translation_fraction
        )
        grid = functional.affine_grid(theta, images.shape, align_corners=False)
        view = functional.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        brightness = torch.empty((batch, 1, 1, 1), device=images.device, dtype=images.dtype).uniform_(1.0 - self.brightness, 1.0 + self.brightness)
        contrast = torch.empty((batch, 1, 1, 1), device=images.device, dtype=images.dtype).uniform_(1.0 - self.contrast, 1.0 + self.contrast)
        return (((view - 0.5) * contrast + 0.5) * brightness).clamp(0.0, 1.0)


class RandomPatchMasker:
    def __init__(self, mask_ratio: float = 0.30) -> None:
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie in (0, 1)")
        self.mask_ratio = float(mask_ratio)

    def __call__(self, batch: int, locations: int, device: torch.device) -> torch.Tensor:
        count = max(1, int(round(locations * self.mask_ratio)))
        random_order = torch.rand((batch, locations), device=device).argsort(dim=-1)
        mask = torch.zeros((batch, locations), device=device, dtype=torch.bool)
        mask.scatter_(1, random_order[:, :count], True)
        return mask


class DINOv2SSLAdapterModel(nn.Module):
    """DINOv2-style image SSL continuation using a frozen DINO base plus LoRA.

    The objective combines cross-view DINO self-distillation, masked patch
    (iBOT-style) distillation, and optional KoLeo feature spreading.  It is a
    domain-adaptation recipe, not a claim of reproducing the original
    internet-scale DINOv2 pretraining run.
    """

    def __init__(
        self,
        student_backbone: DinoV2Backbone,
        *,
        lora_config: LoRAConfig,
        prototypes: int = 2048,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
        center_momentum: float = 0.9,
        ibot_mask_ratio: float = 0.30,
        dino_weight: float = 1.0,
        ibot_weight: float = 1.0,
        koleo_weight: float = 0.02,
    ) -> None:
        super().__init__()
        student_backbone.requires_grad_(False)
        self.student_encoder = student_backbone
        self.wrapped_modules = inject_lora_modules(self.student_encoder, lora_config)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.teacher_encoder.requires_grad_(False)
        self.teacher_encoder.eval()
        self.student_head = DINOProjectionHead(student_backbone.feature_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, prototypes=prototypes)
        self.teacher_head = copy.deepcopy(self.student_head)
        self.teacher_head.requires_grad_(False)
        self.teacher_head.eval()
        self.augmenter = DINOGlobalAugmenter()
        self.masker = RandomPatchMasker(ibot_mask_ratio)
        self.lora_config = lora_config
        self.student_temperature = float(student_temperature)
        self.teacher_temperature = float(teacher_temperature)
        self.center_momentum = float(center_momentum)
        self.dino_weight = float(dino_weight)
        self.ibot_weight = float(ibot_weight)
        self.koleo_weight = float(koleo_weight)
        self.register_buffer("teacher_center", torch.zeros(1, prototypes))

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_head.eval()
        return self

    def trainable_parameters(self):
        yield from lora_parameters(self.student_encoder)
        yield from self.student_head.parameters()

    def _teacher_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return functional.softmax((logits - self.teacher_center) / self.teacher_temperature, dim=-1).detach()

    @staticmethod
    def _cross_entropy(student_logits: torch.Tensor, teacher_probabilities: torch.Tensor, temperature: float) -> torch.Tensor:
        return -(teacher_probabilities * functional.log_softmax(student_logits / temperature, dim=-1)).sum(dim=-1).mean()

    @staticmethod
    def _koleo_loss(features: torch.Tensor) -> torch.Tensor:
        if features.shape[0] < 2:
            return features.new_zeros(())
        normalized = functional.normalize(features.float(), dim=-1)
        distances = torch.cdist(normalized, normalized)
        distances = distances.masked_fill(torch.eye(features.shape[0], device=features.device, dtype=torch.bool), float("inf"))
        return -torch.log(distances.min(dim=-1).values.clamp_min(1e-6)).mean()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected raw frames [B,3,224,224], got {tuple(images.shape)}")
        view_a, view_b = self.augmenter(images), self.augmenter(images)
        normalized_a = self.student_encoder.preprocess(view_a)
        normalized_b = self.student_encoder.preprocess(view_b)
        locations = (normalized_a.shape[-2] // self.student_encoder.patch_size) * (normalized_a.shape[-1] // self.student_encoder.patch_size)
        mask_a = self.masker(images.shape[0], locations, images.device)
        mask_b = self.masker(images.shape[0], locations, images.device)
        student_patch_a, _, student_cls_a = self.student_encoder.forward_masked_features(normalized_a, mask_a)
        student_patch_b, _, student_cls_b = self.student_encoder.forward_masked_features(normalized_b, mask_b)
        with torch.no_grad():
            teacher_patch_a, _, teacher_cls_a = self.teacher_encoder.forward_features(self.teacher_encoder.preprocess(view_a))
            teacher_patch_b, _, teacher_cls_b = self.teacher_encoder.forward_features(self.teacher_encoder.preprocess(view_b))
            teacher_cls_logits_a = self.teacher_head(teacher_cls_a)
            teacher_cls_logits_b = self.teacher_head(teacher_cls_b)
            teacher_patch_logits_a = self.teacher_head(teacher_patch_a[mask_a])
            teacher_patch_logits_b = self.teacher_head(teacher_patch_b[mask_b])
        student_cls_logits_a = self.student_head(student_cls_a)
        student_cls_logits_b = self.student_head(student_cls_b)
        student_patch_logits_a = self.student_head(student_patch_a[mask_a])
        student_patch_logits_b = self.student_head(student_patch_b[mask_b])
        dino_loss = 0.5 * (
            self._cross_entropy(student_cls_logits_a, self._teacher_probabilities(teacher_cls_logits_b), self.student_temperature)
            + self._cross_entropy(student_cls_logits_b, self._teacher_probabilities(teacher_cls_logits_a), self.student_temperature)
        )
        ibot_loss = 0.5 * (
            self._cross_entropy(student_patch_logits_a, self._teacher_probabilities(teacher_patch_logits_a), self.student_temperature)
            + self._cross_entropy(student_patch_logits_b, self._teacher_probabilities(teacher_patch_logits_b), self.student_temperature)
        )
        koleo_loss = 0.5 * (self._koleo_loss(student_cls_a) + self._koleo_loss(student_cls_b))
        loss = self.dino_weight * dino_loss + self.ibot_weight * ibot_loss + self.koleo_weight * koleo_loss
        if self.training:
            with torch.no_grad():
                teacher_logits = torch.cat((teacher_cls_logits_a, teacher_cls_logits_b, teacher_patch_logits_a, teacher_patch_logits_b), dim=0)
                self.teacher_center.mul_(self.center_momentum).add_(teacher_logits.mean(dim=0, keepdim=True), alpha=1.0 - self.center_momentum)
        return loss, {
            "loss": loss.detach(),
            "dino_loss": dino_loss.detach(),
            "ibot_loss": ibot_loss.detach(),
            "koleo_loss": koleo_loss.detach(),
            "masked_token_count": (mask_a.sum() + mask_b.sum()).detach(),
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("EMA momentum must lie in [0, 1)")
        student_encoder = dict(self.student_encoder.named_parameters())
        for name, parameter in self.teacher_encoder.named_parameters():
            parameter.mul_(momentum).add_(student_encoder[name], alpha=1.0 - momentum)
        student_head = dict(self.student_head.named_parameters())
        for name, parameter in self.teacher_head.named_parameters():
            parameter.mul_(momentum).add_(student_head[name], alpha=1.0 - momentum)
