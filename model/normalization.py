"""EXP174 FP32 fixed RMS and the base RMS bound."""
import math
import torch
from torch import nn

class RMSLimit(nn.Module):
    def __init__(self, limit: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("RMS limit must be finite and positive")
        self.limit = float(limit)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim < 1 or tokens.shape[-1] == 0 or not tokens.is_floating_point():
            raise ValueError("Expected floating point tokens with a nonempty feature dimension")
        # Keep the iterative state and the bound calculation in FP32 under AMP.
        values = tokens.float()
        rms = torch.linalg.vector_norm(values, dim=-1, keepdim=True) / math.sqrt(values.shape[-1])
        return values / (rms / self.limit).clamp_min(1.0)

class FixedRMSNorm(nn.Module):
    """FP32 fixed RMS above epsilon; exact zero remains zero, finite derivative."""
    def __init__(self, target_rms=1.0, eps=1e-8):
        super().__init__()
        self.target_rms = float(target_rms)
        self.eps = float(eps)

    def forward(self, tokens):
        if not tokens.is_floating_point() or tokens.ndim < 1 or tokens.shape[-1] == 0:
            raise ValueError("Expected nonempty floating point token features")
        value = tokens.float()
        rms = torch.linalg.vector_norm(value, dim=-1, keepdim=True) / math.sqrt(value.shape[-1])
        return value * (self.target_rms / rms.clamp_min(self.eps))
