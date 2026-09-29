"""CLIP text encoder with externally supplied learnable context embeddings."""

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokens: torch.Tensor, from_tokens: bool = False):
        x = self.token_embedding(prompts).type(self.dtype) if from_tokens else prompts
        x = x + self.positional_embedding.type(self.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)] @ self.text_projection
