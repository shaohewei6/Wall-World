"""Backbone-agnostic inference primitives for Finetine adapters.

The package deliberately exposes a common spatial-feature contract instead of
assuming that every visual encoder is a Vision Transformer.  DINOv2 is the
first registered implementation; CNN feature-map adapters can use the same
pipeline and checkpoint machinery.
"""

from .backbones import CNNFeatureMapBackbone, VisualBackbone, create_backbone, register_backbone
from .checkpoint import apply_adapter_checkpoint, read_adapter_checkpoint, save_adapter_checkpoint
from .dinov2 import DinoV2Backbone, DINOv2SSLAdapterModel
from .lora import LoRAConfig, inject_lora_modules
from .pipeline import EncodedFeatures, FinetineInferencePipeline
from .trajectory_dataset import LargePatchClipDataset, LargePatchFrameDataset
from .vjepa import SpatiotemporalMaskConfig, VJEPAAdapterModel

__all__ = [
    "CNNFeatureMapBackbone",
    "DinoV2Backbone",
    "DINOv2SSLAdapterModel",
    "VisualBackbone",
    "create_backbone",
    "register_backbone",
    "LoRAConfig",
    "inject_lora_modules",
    "read_adapter_checkpoint",
    "apply_adapter_checkpoint",
    "save_adapter_checkpoint",
    "EncodedFeatures",
    "FinetineInferencePipeline",
    "LargePatchClipDataset",
    "LargePatchFrameDataset",
    "SpatiotemporalMaskConfig",
    "VJEPAAdapterModel",
]
