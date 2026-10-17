"""Strict EXP174 assets and atomic, complete training-state I/O."""
from pathlib import Path
import hashlib
import os
import random
import uuid

import numpy as np
import torch


E6_SHA256 = "21f25dbd1357b882a22f3b117f0e41af516bc592bb4158f18c24dfd354b1fdfc"
ENCODER_SHA256 = "f7b4f847db7717a0c8e261e21156c2dfe11aa675a212072c465f4695c9978889"
CONFIG_SHA256 = "9d816432834101212c643ee9a4be14a29ba47715e04fa40b89b3f62d72b58869"
SOURCE_SHA256 = "2134436da7dcae9bdcc3624ba033944dcd4d1d31b27eb7a2d117a8935dcd300b"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_initialization(encoder_reference):
    """Canonical untrained world + pretrained frozen encoder, not a training run."""
    if sha256(encoder_reference) != ENCODER_SHA256:
        raise ValueError("Canonical initialization SHA256 mismatch")
    saved = torch.load(encoder_reference, map_location="cpu", weights_only=False)
    if saved.get("global_step") != 0 or saved["optimizer"]["state"]:
        raise ValueError("Expected the canonical step-zero initialization")
    return saved["model"]


def load_model_checkpoint(checkpoint, encoder_reference=None, *, expected_sha256=None, encoder_sha256=None):
    """Load an explicitly selected trusted checkpoint, with optional configured hashes."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    actual_sha = sha256(checkpoint)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(f"Checkpoint SHA256 mismatch: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint must be a state_dict or a mapping containing model")
    state = saved.get("model", saved)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint model must contain named tensors")
    state = dict(state)
    encoder_path = encoder_actual = None
    if not any(name.startswith("encoder.") for name in state):
        if encoder_reference is None:
            raise ValueError("Compact checkpoint requires an explicit encoder_reference path")
        encoder_path = Path(encoder_reference).expanduser().resolve()
        encoder_actual = sha256(encoder_path)
        recorded = saved.get("encoder_reference", {}).get("sha256")
        for expected in (encoder_sha256, recorded):
            if expected is not None and encoder_actual != expected:
                raise ValueError(f"Encoder reference SHA256 mismatch: {encoder_path}")
        initial = torch.load(encoder_path, map_location="cpu", weights_only=False)
        initial_state = initial.get("model", initial)
        encoder = {name:v for name,v in initial_state.items() if name.startswith("encoder.")}
        if not encoder:
            raise ValueError("encoder_reference contains no encoder tensors")
        encoder.update(state)
        state = encoder
    reference = {"checkpoint": str(checkpoint), "checkpoint_sha256": actual_sha,
                 "encoder_reference": None if encoder_path is None else str(encoder_path),
                 "encoder_sha256": encoder_actual, "global_step": saved.get("global_step")}
    recorded_config = saved.get("config")
    if isinstance(recorded_config, dict):
        reference["model_config"] = recorded_config.get("model", recorded_config)
    return state, reference


def load_exp174(checkpoint, encoder_reference):
    """Compatibility entry for explicitly requesting the historical E6 assets."""
    return load_model_checkpoint(checkpoint, encoder_reference, expected_sha256=E6_SHA256,
                                 encoder_sha256=ENCODER_SHA256)


def atomic_save(path, payload):
    """Write/fsync first, then atomically replace; the old file survives write failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _optimizer_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    try:
        return [[names[id(parameter)] for parameter in group["params"]]
                for group in optimizer.param_groups]
    except KeyError as error:
        raise ValueError("Optimizer contains parameters outside the supplied model") from error


def save_training_state(path, model, optimizer, *, scheduler=None, scaler=None,
                        epoch=0, global_step=0, stream=None, config=None, encoder_reference=None):
    """Self-contained checkpoint; callers control save cadence and historical retention."""
    state = model.state_dict()
    if encoder_reference is not None:
        if not all(k in encoder_reference for k in ('path','sha256')):
            raise ValueError('Compact training checkpoint requires encoder path and SHA256')
        state = {k:v for k,v in state.items() if not k.startswith('encoder.')}
    payload = {"format_version": "wall_world.training.v1", "model": state,
               "optimizer": optimizer.state_dict(),
               "optimizer_parameter_names": _optimizer_names(model, optimizer),
               "scheduler": None if scheduler is None else scheduler.state_dict(),
               "scaler": None if scaler is None else scaler.state_dict(),
               "epoch": int(epoch), "global_step": int(global_step), "stream": stream,
               "config": model.config if config is None else config,
               "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                       "torch": torch.get_rng_state(),
                       "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    if encoder_reference is not None:
        payload['encoder_reference'] = dict(encoder_reference)
    atomic_save(path, payload)


def load_training_state(path, model, optimizer, *, scheduler=None, scaler=None, encoder_reference=None):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format_version") != "wall_world.training.v1":
        raise ValueError("Use the historical trainer to resume original EXP174 stream checkpoints")
    names = _optimizer_names(model, optimizer)
    if names != saved["optimizer_parameter_names"]:
        raise ValueError("Optimizer parameter order differs from the saved training model")
    for name, target in (("scheduler", scheduler), ("scaler", scaler)):
        if (saved[name] is None) != (target is None):
            raise ValueError(f"Restore requires the same {name} setup")
    if 'encoder_reference' in saved:
        reference = encoder_reference or saved['encoder_reference']['path']
        state, _ = load_model_checkpoint(path, reference, encoder_sha256=saved['encoder_reference']['sha256'])
    else:
        state = saved['model']
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(saved["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(saved["scaler"])
    rng = saved["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None:
        if not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA RNG state requires the same CUDA device count")
        torch.cuda.set_rng_state_all(rng["cuda"])
    return {key: saved[key] for key in ("epoch", "global_step", "stream", "config")}
