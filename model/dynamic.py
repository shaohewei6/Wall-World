"""AdaLN-Zero with a concatenated action/identity condition per state token."""
import torch
from torch import nn


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads or mlp_ratio <= 0 or int(dim * mlp_ratio) < 1:
            raise ValueError("Invalid AdaLN dimensions or attention heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(2 * dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, h: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3 or condition.shape != (*h.shape[:2], 2 * h.shape[-1]):
            raise ValueError("Expected state [B,H,D] and concatenated condition [B,H,2D]")
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(6, -1)
        x = self.norm1(h) * (1 + scale_a) + shift_a
        h = h + gate_a * self.attention(x, x, x, need_weights=False)[0]
        return h + gate_m * self.mlp(self.norm2(h) * (1 + scale_m) + shift_m)

"""Action-conditioned recurrent update with raw-state/identity retention gates."""
import torch
from torch import nn

from .normalization import RMSLimit


class Dynamic(nn.Module):
    def __init__(self, inner_state_dim: int = 256, action_count: int = 8,
                 depth: int = 4, heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, inner_token_count: int = 8,
                 identity: nn.Embedding | None = None, state_rms_limit: float = 1.0) -> None:
        super().__init__()
        if min(inner_state_dim, inner_token_count, action_count, depth) <= 0:
            raise ValueError("Dynamic dimensions and depth must be positive")
        self.inner_state_dim, self.inner_token_count = int(inner_state_dim), int(inner_token_count)
        self.action_count = int(action_count)
        if identity is None:
            self._owned_identity = nn.Embedding(inner_token_count, inner_state_dim)
            nn.init.normal_(self._owned_identity.weight, std=0.02)
            identity = self._owned_identity
        if identity.weight.shape != (inner_token_count, inner_state_dim):
            raise ValueError("Identity embedding must match [H,D]")
        # Agent owns the shared parameter once. Standalone modules own their fallback.
        object.__setattr__(self, "_shared_identity", identity)
        self.action_embedding = nn.Embedding(action_count, inner_state_dim)
        nn.init.normal_(self.action_embedding.weight, std=0.02)
        self.blocks = nn.ModuleList([
            AdaLNBlock(inner_state_dim, heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.state_norm = RMSLimit(state_rms_limit)
        self.gate = nn.Sequential(
            nn.Linear(2 * inner_state_dim, inner_state_dim), nn.SiLU(),
            nn.Linear(inner_state_dim, inner_state_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.last_gate = None

    @property
    def identity(self) -> nn.Embedding:
        return self._shared_identity

    def _validate_inputs(self, h: torch.Tensor, actions: torch.Tensor) -> None:
        if h.ndim != 3 or h.shape[1:] != (self.inner_token_count, self.inner_state_dim):
            raise ValueError("State must have shape [B,inner_token_count,inner_state_dim]")
        if not h.is_floating_point():
            raise TypeError("Recurrent state must be floating point")
        if actions.shape != (len(h),):
            raise ValueError("action_ids must have shape [B]")
        if actions.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("action_ids must use an integer dtype")
        if bool(((actions < 0) | (actions >= self.action_count)).any()):
            raise ValueError(f"action_ids must lie in [0,{self.action_count - 1}]")

    def forward(self, h: torch.Tensor, a: torch.Tensor, *, return_gate: bool = False):
        self._validate_inputs(h, a)
        identity = self.identity.weight[None].expand(len(h), -1, -1)
        action = self.action_embedding(a.to(device=h.device, dtype=torch.long))[:, None].expand_as(identity)
        condition = torch.cat((action, identity), dim=-1)
        processed = h
        for block in self.blocks:
            processed = block(processed, condition)
        processed = self.state_norm(processed)
        gate = self.gate(torch.cat((h, identity), dim=-1)).float().sigmoid()
        result = self.state_norm((1 - gate) * h.float() + gate * processed)
        self.last_gate = gate.detach()
        return (result, gate) if return_gate else result
