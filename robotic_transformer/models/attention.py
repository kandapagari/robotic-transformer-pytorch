# -*- coding: utf-8 -*-
from typing import Callable

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from robotic_transformer.models.layer_norm import LayerNorm
from robotic_transformer.utils import default, exists


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.,
        window_size=7,
        num_mem_kv=4
    ):
        super().__init__()
        assert (
            dim % dim_head) == 0, 'dimension should be divisible by dimension per head'
        self.norm = LayerNorm(dim)
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.mem_kv = nn.Parameter(torch.randn(
            2, self.heads, num_mem_kv, dim_head))
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        self.rel_pos_bias = nn.Embedding(
            (2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - \
            rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (
            rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer(
            'rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = * \
            x.shape, x.device, self.heads  # NOQA

        x = self.norm(x)
        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        # scale
        q = q * self.scale
        # null / memory / register kv
        mk, mv = map(lambda t: repeat(
            t, 'h n d -> b h n d', b=q.shape[0]), self.mem_kv)
        num_mem = mk.shape[-2]
        k = torch.cat((mk, k), dim=-2)
        v = torch.cat((mv, v), dim=-2)
        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        # add positional bias
        bias = self.rel_pos_bias(self.rel_pos_indices)
        bias = F.pad(bias, (0, 0, num_mem, 0), value=0.)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)',
                        w1=window_height, w2=window_width)
        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)


# attention

class TransformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        dim_head=64,
        dim_context=None,
        heads=8,
        norm_context=False,
        dropout=0.1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(
            dim_context) if norm_context else nn.Identity()
        self.attn_dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        attn_bias=None,
        attn_mask=None,
        cond_fn: Callable | None = None
    ):
        b = x.shape[0]  # NOQA
        if exists(context):
            context = self.context_norm(context)
        kv_input = default(context, x)
        x = self.norm(x)
        if exists(cond_fn):
            # adaptive layer-norm
            x = cond_fn(x)
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale
        sim = einsum('b h i d, b j d -> b h i j', q, k)
        if exists(attn_bias):
            sim = sim + attn_bias
        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(
                (i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
