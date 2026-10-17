"""Versioned, adapter-only Finetine checkpoint I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .lora import LoRAConfig, inject_lora_modules, load_lora_state_dict, lora_state_dict


FORMAT_VERSION = "finetine.adapter.v1"


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def read_adapter_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Finetine adapter checkpoint does not exist: {path}")
    payload = _safe_torch_load(path)
    required = {"format_version", "backbone_id", "lora_config", "lora_state"}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Finetine checkpoint {path}; expected a mapping payload")
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Invalid Finetine checkpoint {path}; missing fields: {sorted(missing)}")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported Finetine checkpoint format {payload['format_version']!r}")
    return payload


def apply_adapter_checkpoint(backbone: nn.Module, *, backbone_id: str, payload: dict[str, Any]) -> list[str]:
    """Inject exactly the recorded LoRA structure and load its adapter tensors."""

    if payload["backbone_id"] != backbone_id:
        raise ValueError(f"Adapter targets {payload['backbone_id']!r}, but loaded backbone is {backbone_id!r}")
    config = LoRAConfig.from_state_dict(payload["lora_config"])
    injected = inject_lora_modules(backbone, config)
    expected_names = payload.get("wrapped_modules")
    if expected_names is not None and list(expected_names) != injected:
        raise ValueError(f"Adapter module layout differs from checkpoint; expected={expected_names}, got={injected}")
    load_lora_state_dict(backbone, payload["lora_state"])
    return injected


def save_adapter_checkpoint(
    path: str | Path,
    *,
    backbone: nn.Module,
    backbone_id: str,
    lora_config: LoRAConfig,
    wrapped_modules: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save only LoRA tensors and descriptive metadata, never base weights."""

    path = Path(path).expanduser()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing adapter checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "backbone_id": backbone_id,
            "lora_config": lora_config.state_dict(),
            "wrapped_modules": wrapped_modules,
            "lora_state": lora_state_dict(backbone),
            "metadata": metadata or {},
        },
        path,
    )
