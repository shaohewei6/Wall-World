"""Frozen DINOv2 ViT-S/14 + EXP174 LoRA, constructed entirely from local code."""
import numpy as np
import torch
from torch import nn

from .vendor.dinov2 import vit_small
from .lora import LoRAConfig, inject_lora_modules


class AgentDINOv2Encoder(nn.Module):
    backbone_id = "dinov2_vits14"

    def __init__(self, lora_config, *, image_size=224, patch_size=14, pretrain_image_size=518,
                 layer_scale=1.0, interpolate_antialias=False, interpolate_offset=0.1):
        super().__init__()
        self.base_model = vit_small(img_size=pretrain_image_size, patch_size=patch_size, init_values=layer_scale,
                                   ffn_layer="mlp", block_chunks=0, num_register_tokens=0,
                                   interpolate_antialias=interpolate_antialias, interpolate_offset=interpolate_offset)
        self.adapter_modules = inject_lora_modules(self, LoRAConfig.from_state_dict(lora_config))
        self.patch_size, self.feature_dim = patch_size, 384
        self.image_size, self.visual_token_count, self.visual_state_dim = image_size, (image_size//patch_size)**2, 384
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def preprocess(self, observation):
        if isinstance(observation, np.ndarray):
            images = torch.from_numpy(np.asarray(observation).copy())
        elif isinstance(observation, torch.Tensor):
            images = observation
        else:
            raise TypeError("observation must be a numpy array or torch tensor")
        if images.ndim == 3:
            if images.shape[-1] == 3:
                images = images.permute(2, 0, 1)
            elif images.shape[0] != 3:
                raise ValueError("observation must have three RGB channels")
            images = images.unsqueeze(0)
        elif images.ndim == 4:
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
            elif images.shape[1] != 3:
                raise ValueError("observations must have three RGB channels")
        else:
            raise ValueError("Expected [H,W,3], [3,H,W], [B,H,W,3] or [B,3,H,W]")
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size} RGB observations")
        images = images.to(device=self.image_mean.device, dtype=torch.float32, non_blocking=True)
        if images.numel() and float(images.detach().amax()) > 1.0:
            images = images / 255.0
        if images.numel() and (float(images.detach().amin()) < 0.0 or float(images.detach().amax()) > 1.0):
            raise ValueError("observation values must be uint8 RGB or float RGB in [0,1]")
        return (images - self.image_mean) / self.image_std

    @torch.inference_mode()
    def forward(self, observation):
        tokens = self.base_model.forward_features(self.preprocess(observation))["x_norm_patchtokens"]
        if tuple(tokens.shape[1:]) != (self.visual_token_count, self.visual_state_dim):
            raise ValueError("DINOv2 patch tokens differ from configured visual dimensions")
        return tokens
