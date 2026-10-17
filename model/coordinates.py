"""Origin-relative patch coordinates: two axes, each with 128 vocabulary entries."""
import torch
from torch import nn


COORDINATE_CLASSES = 128
COORDINATE_OFFSET = 64


def coordinate_targets(coordinates: torch.Tensor, limit: int = 63) -> torch.Tensor:
    if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
        raise ValueError("Coordinates must have shape [...,2] in relative patch units")
    if coordinates.dtype == torch.bool or coordinates.is_complex():
        raise TypeError("Coordinates must be real integer patch positions")
    # Validate before any narrowing cast: float64 fractions must not round into
    # valid integers in FP32 and then be truncated differently by long().
    valid = (torch.isfinite(coordinates) & (coordinates == coordinates.round())
             & (coordinates >= -limit) & (coordinates <= limit))
    if not bool(valid.all()):
        raise ValueError(f"Coordinates must be integer patch positions in [-{limit},{limit}]")
    return coordinates.long() + limit + 1


def decode_coordinates(
    logits: torch.Tensor, previous_coordinates: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode each axis; optionally restrict to previous prediction +/- one patch.

    Call once per physical timestep and carry the returned coordinates forward.
    Reset previous_coordinates to None at episode start (global first-frame
    argmax). This is a readout constraint, not feedback into the latent state.
    Ties select the smaller coordinate, as in the global argmax.
    """
    classes = logits.shape[-1]
    if logits.shape[-2] != 2 or classes < 4 or classes % 2:
        raise ValueError("Coordinate logits must have shape [...,2,2*limit+2]")
    offset, limit = classes//2, classes//2-1
    if previous_coordinates is None:
        return logits[..., 1:].argmax(-1) + 1 - offset
    if previous_coordinates.shape != logits.shape[:-1]:
        raise ValueError("Previous coordinates must match logits batch dimensions [...,2]")
    if previous_coordinates.device != logits.device:
        raise ValueError("Previous coordinates and logits must be on the same device")
    previous_classes = coordinate_targets(previous_coordinates, limit)
    candidates = previous_classes.unsqueeze(-1) + torch.arange(-1, 2, device=logits.device)
    valid = (candidates >= 1) & (candidates < classes)
    safe = candidates.clamp(1, classes - 1)
    scores = logits.gather(-1, safe).masked_fill(~valid, -torch.inf)
    if not bool(torch.isfinite(scores).any(-1).all()) or bool(torch.isnan(scores).any()):
        raise ValueError("Each axis must have a finite candidate score and no NaN")
    chosen = scores.argmax(-1, keepdim=True)
    return safe.gather(-1, chosen).squeeze(-1) - offset


class CoordinateEmbedding(nn.Module):
    """One lookup shared across maps and axes; axis projections live in Observation."""
    def __init__(self, dim: int, limit: int = 63) -> None:
        super().__init__()
        self.limit = limit
        self.embedding = nn.Embedding(2*limit+2, dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.embedding(coordinate_targets(coordinates, self.limit))
