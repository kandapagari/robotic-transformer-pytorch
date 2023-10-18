# -*- coding: utf-8 -*-
from typing import Callable

from beartype import beartype
from torch import nn

from robotic_transformer.models.attention import TransformerAttention
from robotic_transformer.models.feed_forward import FeedForward
from robotic_transformer.utils import default


@beartype
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        depth=6,
        attn_dropout=0.,
        ff_dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerAttention(dim=dim, heads=heads,
                                     dropout=attn_dropout),
                FeedForward(dim=dim, dropout=ff_dropout)
            ]))

    def forward(
        self,
        x,
        cond_fns: tuple[Callable, ...] | None = None,
        attn_mask=None
    ):
        cond_fns = iter(default(cond_fns, []))
        for attn, ff in self.layers:
            x = attn(x, attn_mask=attn_mask, cond_fn=next(cond_fns, None)) + x
            x = ff(x, cond_fn=next(cond_fns, None)) + x
        return x
