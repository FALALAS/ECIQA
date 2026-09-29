"""Mask-aware components used by the released ECIQA baseline."""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _token_mask(tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
    if tokens.ndim != 3 or tokens.shape[1] == 0:
        raise ValueError("tokens must have shape [batch, nonempty sequence, dim]")
    if not tokens.is_floating_point():
        raise TypeError("tokens must be floating point")
    if mask is None:
        return torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    if mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
        raise ValueError("mask must be bool [batch, sequence], True means valid")
    if mask.device != tokens.device:
        raise ValueError("mask and tokens must be on the same device")
    return mask


class CandidateCrossAttention(nn.Module):
    """Candidate queries attend to all input tokens, retaining their residual."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0,
                 residual_scale: float = 0.1):
        super().__init__()
        self.dim = dim
        self.candidate_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)
        self.coord_projection = nn.Linear(2, dim, bias=False)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.tensor(float(residual_scale)))

    def _position(self, tokens, coords, mask):
        if coords is None:
            return tokens
        if coords.shape != (*tokens.shape[:2], 2) or coords.device != tokens.device:
            raise ValueError("coordinates must have shape [batch, sequence, 2] on the token device")
        coords = coords.to(tokens.dtype).masked_fill(~mask.unsqueeze(-1), 0)
        return tokens + self.coord_projection(coords)

    def forward(self, candidate, input, candidate_mask=None, input_mask=None,
                candidate_coords=None, input_coords=None):
        cmask, imask = _token_mask(candidate, candidate_mask), _token_mask(input, input_mask)
        if candidate.shape[0] != input.shape[0] or candidate.shape[-1] != self.dim or input.shape[-1] != self.dim:
            raise ValueError("candidate/input must share configured batch and embedding sizes")
        candidate = candidate.masked_fill(~cmask.unsqueeze(-1), 0)
        input = input.masked_fill(~imask.unsqueeze(-1), 0)
        query = self._position(self.candidate_norm(candidate), candidate_coords, cmask)
        value = self.input_norm(input)
        key = self._position(value, input_coords, imask)
        query = query.masked_fill(~cmask.unsqueeze(-1), 0)
        key = key.masked_fill(~imask.unsqueeze(-1), 0)
        value = value.masked_fill(~imask.unsqueeze(-1), 0)
        has_input = imask.any(dim=1)
        safe_mask = imask.clone()
        safe_mask[:, 0] |= ~has_input
        update, _ = self.attention(query, key, value, key_padding_mask=~safe_mask, need_weights=False)
        update = update.masked_fill(~has_input[:, None, None], 0)
        result = candidate + self.gamma * self.dropout(update)
        return result.masked_fill(~cmask.unsqueeze(-1), 0)


class QMapTokenFusion(nn.Module):
    """Projected concatenation updates the original visual tokens residually."""

    def __init__(self, dim):
        super().__init__()
        self.dim, self.mode = dim, "concat"
        self.visual_norm = nn.LayerNorm(dim)
        self.qmap_norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(0.0)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, visual, qmap, mask=None):
        valid = _token_mask(visual, mask)
        if visual.shape != qmap.shape or visual.shape[-1] != self.dim:
            raise ValueError("visual and projected qmap tokens must share [B,N,dim]")
        visual = visual.masked_fill(~valid.unsqueeze(-1), 0)
        qmap = qmap.masked_fill(~valid.unsqueeze(-1), 0)
        pair = torch.cat((self.visual_norm(visual), self.qmap_norm(qmap)), dim=-1)
        update = self.projection(pair)
        return (visual + self.gamma * self.dropout(update)).masked_fill(~valid.unsqueeze(-1), 0)


class PatchQualityPool(nn.Module):
    """Valid-area mean of patch features, without a separate patch score head."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tokens, mask=None, token_area=None):
        valid = _token_mask(tokens, mask)
        if tokens.shape[-1] != self.dim:
            raise ValueError("unexpected token embedding dimension")
        if token_area is not None:
            if token_area.shape != valid.shape or token_area.device != tokens.device:
                raise ValueError("token_area must share token shape and device")
            if not torch.isfinite(token_area).all() or (token_area < 0).any():
                raise ValueError("token_area must be finite and nonnegative")
            valid = valid & (token_area > 0)
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
        raw_weights = torch.ones_like(tokens[..., 0]).masked_fill(~valid, 0)
        if token_area is not None:
            raw_weights = raw_weights * token_area.to(raw_weights.dtype)
        weights = raw_weights / raw_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return None, (tokens * weights.unsqueeze(-1)).sum(dim=1), weights


class CheckpointLayerAggregator(nn.Module):
    """Retained inactive state for strict compatibility with the trained model.

    The published baseline uses only the final visual block. Its original
    constructor still registered this frozen scalar, which is never read.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers), requires_grad=False)
