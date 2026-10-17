"""Frozen EXP141 inference with the causal inputs used by EXP189."""
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .model import AUX_DIM, RecurrentActor

if __package__.count(".") >= 2:
    from ...model.encoder import AgentDINOv2Encoder
    from ...model.lora import load_lora_state_dict
    from ...util.checkpoint import sha256
else:
    from model.encoder import AgentDINOv2Encoder
    from model.lora import load_lora_state_dict
    from util.checkpoint import sha256


MOVES = np.asarray([(0, -1), (0, 1), (-1, 0), (1, 0),
                    (-1, -1), (1, -1), (-1, 1), (1, 1)], dtype=np.int64)


@dataclass(frozen=True)
class InferenceConfig:
    """Checkpoint fields that affect actor shape or its recurrent inputs."""
    action_embedding_size: int
    reward_embedding_size: int
    state_embedding_size: int
    rnn_hidden_size: int
    episode_limit: int
    collision_count_scale: float
    collision_cap: int
    collision_increment: float
    success_reward: float
    step_cost: float
    shaping_weight: float
    first_visit_bonus: float
    repeat_edge_cost: float
    repeat_edge_cap: int
    seed: int

    @classmethod
    def from_checkpoint(cls, config):
        expected = {"recurrent_architecture": "separate", "navigation_v2": True,
                    "collision_memory": False, "reward_shaping": "her_navigation"}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Expected EXP141-style separate GRU actor, HER rewards, and no collision mask")
        result = cls(**{f.name: config[f.name] for f in fields(cls)})
        for name in ("action_embedding_size", "reward_embedding_size", "state_embedding_size",
                     "rnn_hidden_size", "episode_limit", "collision_cap", "repeat_edge_cap", "collision_count_scale"):
            if not np.isfinite(getattr(result, name)) or getattr(result, name) <= 0:
                raise ValueError(f"Invalid checkpoint inference field: {name}")
        if not all(np.isfinite(getattr(result, f.name)) for f in fields(cls)):
            raise ValueError("Checkpoint inference fields must be finite")
        return result


def _asset(path, expected):
    path = Path(path).expanduser().resolve()
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path, actual


def _load_encoder(base_path, adapter_path):
    payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != "finetine.adapter.v1" or payload.get("backbone_id") != "dinov2_vits14":
        raise ValueError("Expected DINOv2 ViT-S/14 Finetine adapter")
    encoder = AgentDINOv2Encoder(payload["lora_config"])
    if payload.get("wrapped_modules", encoder.adapter_modules) != encoder.adapter_modules:
        raise ValueError("Adapter module layout differs from checkpoint")
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    base = base.get("model", base)
    state, used = encoder.state_dict(), set()
    for name in state:
        if name.startswith("base_model.") and not (name.endswith("lora_a") or name.endswith("lora_b")):
            raw_name = name.removeprefix("base_model.").replace(".base.", ".")
            state[name] = base[raw_name]
            used.add(raw_name)
    if used != set(base):
        raise ValueError("Pretrained encoder tensor names differ from ViT-S/14")
    encoder.load_state_dict(state, strict=True)
    load_lora_state_dict(encoder, payload["lora_state"])
    return encoder


class Agent:
    """A single recurrent episode; receives local RGB, relative goal and feedback.

    No map, world model, critic, replay, HER relabeling or optimizer is constructed.
    Reward bookkeeping is necessary because received reward is an actor input.
    """
    def __init__(self, actor, encoder, config, *, device, precision, assets):
        self.device = torch.device(device)
        self.precision, self.config, self.assets = precision, config, assets
        if precision not in ("float32", "bfloat16") or (precision == "bfloat16" and self.device.type != "cuda"):
            raise ValueError("Use float32 on CPU, or float32/bfloat16 on CUDA")
        self.actor = actor.to(self.device).eval().requires_grad_(False)
        self.encoder = encoder.to(self.device).eval().requires_grad_(False)
        self._ready = False

    @classmethod
    def from_checkpoint(cls, checkpoint, *, encoder_checkpoint, encoder_adapter,
                        device="cpu", precision="float32", checkpoint_sha256=None,
                        encoder_sha256=None, adapter_sha256=None):
        """Load trusted assets explicitly; only actor.* tensors are retained."""
        paths = [_asset(p, h) for p, h in ((checkpoint, checkpoint_sha256),
                 (encoder_checkpoint, encoder_sha256), (encoder_adapter, adapter_sha256))]
        saved = torch.load(paths[0][0], map_location="cpu", weights_only=False)
        config = InferenceConfig.from_checkpoint(saved["config"])
        actor = RecurrentActor(config)
        weights = {k.removeprefix("actor."): v for k, v in saved["model"].items() if k.startswith("actor.")}
        actor.load_state_dict(weights, strict=True)
        del saved, weights
        encoder = _load_encoder(paths[1][0], paths[2][0])
        assets = {key: {"path": str(p), "sha256": h} for key, (p, h) in
                  zip(("checkpoint", "encoder_checkpoint", "encoder_adapter"), paths)}
        return cls(actor, encoder, config, device=device, precision=precision, assets=assets)

    @torch.inference_mode()
    def encode(self, image):
        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.precision == "bfloat16" else nullcontext()
        with context:
            visual = self.encoder(image)
        # Preserve original Features/EXP189 quantization before the FP32 actor.
        return visual.to(torch.float16)[:, None]

    def reset(self, image, *, goal, mode="sample", seed=None):
        goal = np.asarray(goal)
        if goal.shape != (2,) or not np.isfinite(goal).all() or not np.equal(goal, np.round(goal)).all() or np.abs(goal).max() > 63:
            raise ValueError("goal must be a relative integer patch offset within [-63,63]")
        if mode not in ("sample", "argmax"):
            raise ValueError("mode must be sample or argmax")
        visual = self.encode(image)
        if visual.shape[0] != 1:
            raise ValueError("One Agent handles one episode; reset needs one RGB observation")
        self.goal, self.position = goal.astype(np.int64), np.zeros(2, dtype=np.int64)
        self.mode = mode
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.hidden, self.pending = None, None
        self.visual = visual
        self.previous_action, self.received_reward = -1, 0.
        self.steps, self.collision_count = 0, 0
        self.displacement, self.collision = np.zeros(2, dtype=np.int64), False
        self.visited, self.edges = {(0, 0)}, {}
        self.done = bool(np.all(self.goal == 0))
        self.end_reason = "success" if self.done else None
        self._ready = True
        return self.auxiliary()

    def auxiliary(self):
        if not self._ready:
            raise RuntimeError("Call reset before reading episode inputs")
        aux = np.zeros(AUX_DIM, dtype=np.float32)
        aux[:2] = self.position/63
        aux[2:4] = self.goal/63
        aux[4:6] = (self.goal-self.position)/63
        aux[6:8], aux[8] = self.displacement, self.collision
        aux[9] = self.collision_count/self.config.collision_count_scale
        aux[10] = max(0, self.config.episode_limit-self.steps)/self.config.episode_limit
        aux[-8:] = 1  # EXP141 has no collision mask.
        return aux

    @torch.inference_mode()
    def decide(self):
        if not self._ready:
            raise RuntimeError("Call reset before decide")
        if self.done:
            return {"action": None, "done": True, "end_reason": self.end_reason}
        if self.pending is None:
            logits, hidden, _ = self.actor(self.visual,
                torch.tensor([[self.previous_action]], device=self.device),
                torch.tensor([[self.received_reward]], device=self.device),
                torch.from_numpy(self.auxiliary()).to(self.device)[None, None], self.hidden)
            logits = logits[:, -1]
            probabilities = logits.log_softmax(-1).exp()[0].double().cpu().numpy()
            probabilities /= probabilities.sum()
            action = int(probabilities.argmax()) if self.mode == "argmax" else int(self.rng.choice(8, p=probabilities))
            self.pending = (action, hidden, logits.detach().clone(), probabilities.copy())
        action, _, logits, probabilities = self.pending
        return {"action": action, "done": False, "logits": logits.clone(),
                "probabilities": probabilities.copy()}

    def step(self, action, image, actual_displacement, *, collision):
        if not self._ready or self.done or self.pending is None:
            raise RuntimeError("step requires a pending decision in an active episode")
        if not isinstance(action, (int, np.integer)) or action != self.pending[0]:
            raise ValueError("Feedback action must match the pending decision")
        delta = np.asarray(actual_displacement)
        if not isinstance(collision, (bool, np.bool_, int, np.integer)) or collision not in (0, 1):
            raise ValueError("collision must be boolean")
        expected = np.zeros(2, dtype=np.int64) if collision else MOVES[action]
        if delta.shape != (2,) or not np.array_equal(delta, expected):
            raise ValueError("Expected an atomic one-patch move, or zero displacement on collision")
        visual = self.encode(image)
        if visual.shape[0] != 1:
            raise ValueError("step needs one RGB observation")
        c, previous = self.config, self.position
        position = previous+delta.astype(np.int64)
        point, edge = tuple(position), (tuple(previous), int(action))
        first_visit = point not in self.visited
        repeat = self.edges.get(edge, 0) if not collision else 0
        count = self.collision_count+int(collision)
        success = bool(np.all(self.goal-position == 0))
        distance = float(np.abs(self.goal-previous).max())
        next_distance = float(np.abs(self.goal-position).max())
        reward = (c.success_reward*success-c.step_cost
                  -c.collision_increment*(int(collision)*min(count, c.collision_cap))
                  +c.shaping_weight*(distance-next_distance))
        reward += c.first_visit_bonus*int(first_visit)
        reward -= c.repeat_edge_cost*min(repeat, c.repeat_edge_cap)
        if not collision:
            self.edges[edge] = repeat+1
        self.visited.add(point)
        self.hidden, self.pending = self.pending[1], None
        self.visual, self.position = visual, position
        self.previous_action, self.received_reward = int(action), reward
        self.displacement, self.collision = position-previous, bool(collision)
        self.collision_count, self.steps = count, self.steps+1
        self.done = success or self.steps >= c.episode_limit
        self.end_reason = "success" if success else ("timeout" if self.done else None)
        return {"reward": reward, "done": self.done, "end_reason": self.end_reason,
                "position": position.copy(), "collision_count": count}
