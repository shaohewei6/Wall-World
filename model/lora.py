"""Dependency-free LoRA modules and model surgery helpers.

Injection is name/configuration driven, rather than tied to ViT internals.  A
future CNN can opt into LoRA on its 1x1 convolutions using the same API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_patterns: tuple[str, ...] = ("*.attn.qkv", "*.attn.proj")
    target_blocks: tuple[int, ...] | None = (6, 7, 8, 9, 10, 11)
    adapt_conv1x1: bool = False

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must lie in [0, 1)")

    def state_dict(self) -> dict:
        value = asdict(self)
        value["target_patterns"] = list(self.target_patterns)
        value["target_blocks"] = list(self.target_blocks) if self.target_blocks is not None else None
        return value

    @classmethod
    def from_state_dict(cls, value: dict) -> "LoRAConfig":
        value = dict(value)
        value["target_patterns"] = tuple(value["target_patterns"])
        if value.get("target_blocks") is not None:
            value["target_blocks"] = tuple(value["target_blocks"])
        return cls(**value)


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(config.rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, config.rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = (self.dropout(value) @ self.lora_a.transpose(0, 1)) @ self.lora_b.transpose(0, 1)
        return self.base(value) + residual * self.scaling


class LoRAConv1x1(nn.Module):
    """Frozen 1x1 convolution plus a trainable low-rank 1x1 residual."""

    def __init__(self, base: nn.Conv2d, config: LoRAConfig) -> None:
        if base.kernel_size != (1, 1) or base.groups != 1:
            raise ValueError("LoRAConv1x1 only supports ungrouped 1x1 convolutions")
        super().__init__()
        self.base = base
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout2d(config.dropout) if config.dropout else nn.Identity()
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_down = nn.Conv2d(base.in_channels, config.rank, kernel_size=1, stride=base.stride, bias=False, **factory)
        self.lora_up = nn.Conv2d(config.rank, base.out_channels, kernel_size=1, bias=False, **factory)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.lora_up(self.lora_down(self.dropout(value))) * self.scaling


def _block_index(module_name: str) -> int | None:
    segments = module_name.split(".")
    for index, segment in enumerate(segments[:-1]):
        if segment == "blocks" and segments[index + 1].isdigit():
            return int(segments[index + 1])
    return None


def _matches(module_name: str, module: nn.Module, config: LoRAConfig) -> bool:
    if not any(fnmatch(module_name, pattern) for pattern in config.target_patterns):
        return False
    index = _block_index(module_name)
    if config.target_blocks is not None and index is not None and index not in config.target_blocks:
        return False
    return isinstance(module, nn.Linear) or (config.adapt_conv1x1 and isinstance(module, nn.Conv2d))


def _set_submodule(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parent_name, attribute = qualified_name.rsplit(".", maxsplit=1)
    parent = root.get_submodule(parent_name)
    setattr(parent, attribute, replacement)


def inject_lora_modules(backbone: nn.Module, config: LoRAConfig) -> list[str]:
    """Wrap matching frozen modules and return their original qualified names.

    Calling this twice on a backbone is an error: it would otherwise create
    nested adapters with ambiguous checkpoint names.
    """

    candidates = [(name, module) for name, module in backbone.named_modules() if name and _matches(name, module, config)]
    wrapped: list[str] = []
    for name, module in candidates:
        if isinstance(module, (LoRALinear, LoRAConv1x1)):
            raise ValueError(f"Backbone already contains a Finetine LoRA wrapper at {name}")
        if isinstance(module, nn.Linear):
            replacement: nn.Module = LoRALinear(module, config)
        elif isinstance(module, nn.Conv2d):
            replacement = LoRAConv1x1(module, config)
        else:  # pragma: no cover - _matches guards the types
            continue
        _set_submodule(backbone, name, replacement)
        wrapped.append(name)
    if not wrapped:
        patterns = ", ".join(config.target_patterns)
        raise ValueError(f"No modules matched LoRA target patterns: {patterns}")
    return wrapped


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return only trainable adapter tensors, never a copy of base weights."""

    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items() if ".lora_" in name}


def load_lora_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = lora_state_dict(module)
    missing = sorted(set(expected).difference(state))
    unexpected = sorted(set(state).difference(expected))
    if missing or unexpected:
        raise ValueError(f"Adapter state is incompatible; missing={missing}, unexpected={unexpected}")
    result = module.load_state_dict(state, strict=False)
    non_lora_missing = [name for name in result.missing_keys if ".lora_" in name]
    non_lora_unexpected = [name for name in result.unexpected_keys if ".lora_" in name]
    if non_lora_missing or non_lora_unexpected:
        raise RuntimeError(f"Failed to load LoRA tensors: missing={non_lora_missing}, unexpected={non_lora_unexpected}")


def lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for name, parameter in module.named_parameters():
        if ".lora_" in name:
            yield parameter
