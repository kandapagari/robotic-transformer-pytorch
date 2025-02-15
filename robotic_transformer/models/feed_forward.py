# -*- coding: utf-8 -*-
from torch import nn

from robotic_transformer.models.layer_norm import LayerNorm
from robotic_transformer.utils import exists


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond_fn=None):
        x = self.norm(x)
        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)
        return self.net(x)
