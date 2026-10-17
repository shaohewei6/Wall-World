"""Visual-token self-attention used by Observation."""
import torch
from torch import nn


class ObservationSelfAttentionBlock(nn.Module):
    def __init__(self,inner_state_dim: int,heads: int,mlp_ratio: float = 4.0,dropout: float = 0.0,) -> None:
        super().__init__()
        if inner_state_dim <= 0:
            raise ValueError("dim must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")
        if inner_state_dim % heads != 0:
            raise ValueError(
                f"dim={inner_state_dim} must be divisible by heads={heads}"
            )
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.dim = int(inner_state_dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, self.dim),
            nn.Dropout(dropout),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [B, V, D]")
        if visual_tokens.shape[-1] != self.dim:
            raise ValueError(
                f"Expected visual token dimension {self.dim}, "
                f"got {visual_tokens.shape[-1]}"
            )
        normalized = self.norm1(visual_tokens)
        attended_tokens, _ = self.self_attention(
            query=normalized,
            key=normalized,
            value=normalized,
            need_weights=False,
        )
        visual_tokens = visual_tokens + attended_tokens
        visual_tokens = (visual_tokens + self.mlp(self.norm2(visual_tokens)))
        return visual_tokens

"""Independent visual/coordinate evidence reads and gated residual state correction."""
import math

import torch
from torch import nn

from .coordinates import CoordinateEmbedding
from .normalization import RMSLimit


def present_mask(mask, value, batch, device, name):
    if mask is None:
        return torch.full((batch,), value is not None, dtype=torch.bool, device=device)
    mask = torch.as_tensor(mask, device=device)
    if mask.dtype != torch.bool or mask.shape != (batch,):
        raise ValueError(f"{name} must be boolean [B]")
    if value is None and bool(mask.any()):
        raise ValueError(f"{name} marks a missing input as available")
    return mask


class ModalCorrection(nn.Module):
    """One modality owns its Q, attention, write network and feature-wise gate."""
    def __init__(self, dim, heads, state_rms_limit=1.0, initial_gate=0.1, dropout=0.0):
        super().__init__()
        if not 0 < initial_gate < 1:
            raise ValueError("initial_gate must lie in (0,1)")
        self.modality = nn.Parameter(torch.randn(dim) * 0.02)
        self.source_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.query = nn.Linear(3 * dim, dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.update = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.gate = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.correction_norm = RMSLimit(state_rms_limit)
        nn.init.normal_(self.update[-1].weight, std=1e-3)
        nn.init.zeros_(self.update[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(initial_gate / (1 - initial_gate)))

    def forward(self, h, identity, source):
        identity = identity[None].expand(len(h), -1, -1)
        modality = self.modality[None, None].expand_as(h)
        # Raw h enters both Q and gates; no centering or mean extraction.
        q = self.query(torch.cat((h, identity, modality), dim=-1))
        kv = self.source_norm(source + self.modality)
        read = self.attention(q, kv, kv, need_weights=False)[0]
        context = torch.cat((h, identity, read), dim=-1)
        update = self.correction_norm(self.update(context))
        gate = self.gate(context).float().sigmoid()
        return gate * update, gate


class Observation(nn.Module):
    def __init__(self, visual_state_dim=384, inner_state_dim=256, visual_token_count=256,
                 inner_token_count=8, visual_depth=1, heads=8, mlp_ratio=4.0, dropout=0.0,
                 identity=None, state_rms_limit=1.0, initial_gate=0.1,
                 fusion_mode="modal_gated_residual"):
        super().__init__()
        if min(visual_state_dim, inner_state_dim, visual_token_count, inner_token_count, heads) <= 0:
            raise ValueError("Observation dimensions must be positive")
        if inner_state_dim % heads or visual_depth < 0 or mlp_ratio <= 0:
            raise ValueError("Invalid Observation attention dimensions or depth")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if fusion_mode not in ("modal_gated_residual", "learned_gate"):
            raise ValueError("Unsupported Observation fusion_mode")
        self.visual_state_dim, self.inner_state_dim = int(visual_state_dim), int(inner_state_dim)
        self.visual_token_count, self.inner_token_count = int(visual_token_count), int(inner_token_count)
        self.visual_depth, self.heads = int(visual_depth), int(heads)
        self.fusion_mode = "modal_gated_residual"
        if identity is None:
            self._owned_identity = nn.Embedding(inner_token_count, inner_state_dim)
            nn.init.normal_(self._owned_identity.weight, std=0.02)
            identity = self._owned_identity
        if identity.weight.shape != (inner_token_count, inner_state_dim):
            raise ValueError("Identity embedding must match [H,D]")
        object.__setattr__(self, "_shared_identity", identity)
        self.visual_projection = nn.Linear(visual_state_dim, inner_state_dim)
        nn.init.trunc_normal_(self.visual_projection.weight, std=0.02)
        nn.init.zeros_(self.visual_projection.bias)
        self.visual_blocks = nn.ModuleList([
            ObservationSelfAttentionBlock(inner_state_dim, heads, mlp_ratio, dropout)
            for _ in range(visual_depth)
        ])
        self.coordinate_encoder = CoordinateEmbedding(inner_state_dim)
        # Axis roles are represented by separate projections, not extra token identities.
        self.x_projection = nn.Linear(inner_state_dim, inner_state_dim, bias=False)
        self.y_projection = nn.Linear(inner_state_dim, inner_state_dim, bias=False)
        self.visual_correction = ModalCorrection(inner_state_dim, heads, state_rms_limit, initial_gate, dropout)
        self.coordinate_correction = ModalCorrection(inner_state_dim, heads, state_rms_limit, initial_gate, dropout)
        self.state_norm = RMSLimit(state_rms_limit)
        self.last_diagnostics = {}

    @property
    def identity(self):
        return self._shared_identity

    def _validate_prior(self, h, batch_size):
        if h.shape != (batch_size, self.inner_token_count, self.inner_state_dim):
            raise ValueError("Expected prior [B,inner_token_count,inner_state_dim]")
        if not h.is_floating_point():
            raise TypeError("Recurrent state must be floating point")

    def encode_visual(self, visual_tokens):
        if visual_tokens.ndim != 3 or visual_tokens.shape[1:] != (self.visual_token_count, self.visual_state_dim):
            raise ValueError("Expected visual features [B,visual_token_count,visual_state_dim]")
        encoded = self.visual_projection(visual_tokens)
        for block in self.visual_blocks:
            encoded = block(encoded)
        return encoded

    def encode_coordinates(self, coordinates):
        if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
            raise ValueError("Expected coordinates [B,2] in origin-relative patch units")
        encoded = self.coordinate_encoder(coordinates)
        return torch.stack((self.x_projection(encoded[:, 0]), self.y_projection(encoded[:, 1])), dim=1)

    def correct(self, h_prior, visual_tokens=None, *, coordinates=None, visual_present=None,
                coordinate_present=None, return_diagnostics=False, return_update_gate=False):
        self._validate_prior(h_prior, len(h_prior))
        batch, device = len(h_prior), h_prior.device
        vm = present_mask(visual_present, visual_tokens, batch, device, "visual_present")
        pm = present_mask(coordinate_present, coordinates, batch, device, "coordinate_present")
        for value in (visual_tokens, coordinates):
            if value is not None and (len(value) != batch or value.device != device):
                raise ValueError("Observation inputs must match the prior batch and device")
        writes = torch.zeros_like(h_prior, dtype=torch.float32)
        modal_gates = h_prior.new_zeros((*h_prior.shape, 2), dtype=torch.float32)
        diagnostics = {}
        for name, mask, value, encode, branch, column in (
            ("visual", vm, visual_tokens, self.encode_visual, self.visual_correction, 0),
            ("coordinate", pm, coordinates, self.encode_coordinates, self.coordinate_correction, 1),
        ):
            rows = mask.nonzero(as_tuple=True)[0]
            if len(rows):
                # Slice before encoding: absent/invalid placeholders never reach a lookup.
                write, gate = branch(h_prior[rows], self.identity.weight, encode(value[rows]))
                writes = writes.index_add(0, rows, write)
                modal_gates[..., column] = modal_gates[..., column].index_copy(0, rows, gate)
                diagnostics[name + "/gate"] = gate.detach().mean()
                diagnostics[name + "/write_rms"] = write.detach().square().mean().sqrt()
        active = (vm | pm).nonzero(as_tuple=True)[0]
        result = h_prior if not len(active) else h_prior.float().index_copy(
            0, active, self.state_norm(h_prior[active] + writes[active]))
        diagnostics["active_fraction"] = (vm | pm).float().mean().detach()
        diagnostics["state_change_rms"] = (result - h_prior).detach().square().mean().sqrt()
        self.last_diagnostics = diagnostics
        if return_diagnostics:
            return result, {**diagnostics, "modal_gates": modal_gates.detach()}
        if return_update_gate:
            # Retain the legacy scalar visual-gate diagnostic without losing feature gates internally.
            return result, modal_gates[..., 0].mean(dim=-1, keepdim=True)
        return result

    def initialize(self, visual_tokens=None, *, coordinates=None, **kwargs):
        evidence = visual_tokens if visual_tokens is not None else coordinates
        if evidence is None:
            raise ValueError("Initialization requires at least one observed modality")
        h = torch.zeros(len(evidence), self.inner_token_count, self.inner_state_dim, device=evidence.device)
        return self.correct(h, visual_tokens, coordinates=coordinates, **kwargs)

    def forward(self, visual_tokens=None, h_prior=None, *, coordinates=None, **kwargs):
        if h_prior is None:
            return self.initialize(visual_tokens, coordinates=coordinates, **kwargs)
        return self.correct(h_prior, visual_tokens, coordinates=coordinates, **kwargs)


class ConcatObservation(Observation):
    """One coordinate token: shared lookup, X then Y, with no encoder transform."""
    def __init__(self, coordinate_limit=63, **kwargs):
        super().__init__(**kwargs)
        if self.inner_state_dim % 2:
            raise ValueError('Concatenated coordinate encoding needs an even width')
        self.coordinate_encoder = CoordinateEmbedding(self.inner_state_dim // 2, coordinate_limit)
        del self.x_projection, self.y_projection
        self.diagnostic_inputs = None

    def encode_coordinates(self, coordinates):
        if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
            raise ValueError('Expected coordinates [B,2]')
        token = self.coordinate_encoder(coordinates).reshape(len(coordinates), 1, self.inner_state_dim)
        if self.diagnostic_inputs is not None:
            self.diagnostic_inputs.append(token)
        return token
