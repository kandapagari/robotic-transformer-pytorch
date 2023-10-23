# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Robotic-transformer-torch
# Copyright (c) 2023 andyoung. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from vit-pytorch (https://github.com/lucidrains/vit-pytorch)
# ------------------------------------------------------------------------
"""RT1 decoder transformer implemented with pytorch."""
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# PositionalEncoding before transformer decoder


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.positional_encoding = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.positional_encoding[:, 0::2] = torch.sin(pos * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(pos * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        self.positional_encoding = self.positional_encoding.to(x.device)
        x = x + self.positional_encoding[:, :seq_len, :]
        return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         super().__init__()
#         pe = torch.zeros(max_seq_len, d_model)
#         position = torch.arange(0, max_seq_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         seq_len = x.size(1)
#         x = x + self.pe[:seq_len, :]
#         return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = heads != 1 or dim_head != dim

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class Transformers_Decoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., d_model=512,
                 max_seq_len=48, num_actions=11, vocab_size=256):
        super().__init__()
        self.positionalencoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.to_logits = nn.Sequential(
            LayerNorm(d_model),
            nn.Linear(d_model, num_actions * vocab_size),
            Rearrange('... (a b) -> ... a b', b=vocab_size)
        )
        # self.output_tokens = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, N, C = x.shape
        x += self.positionalencoding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_logits(x)
        return x


if __name__ == '__main__':
    # 目前输出的向量形式是正确的，不确定地方有：
    # 1.Transformer(dim, depth, heads, dim_head, mlp_dim)中dim = 512 or 4096, 代码和论文描述不一致
    # 2.关于action tokenizer步骤，是否需要根据tensorflow中的代码进行动作空间的初始化？
    x = torch.randn(48, 512).unsqueeze(0)
    # model = Transformers_Decoder(dim=4096, depth=8, heads=8, dim_head=512, mlp_dim=512,
    # dropout = 0., d_model = 512, max_seq_len = 48, num_actions = 11, vocab_size = 256)
    model = Transformers_Decoder(dim=512, depth=8, heads=8, dim_head=64, mlp_dim=512,
                                 dropout=0., d_model=512, max_seq_len=48, num_actions=11,
                                 vocab_size=256)
    logits_outputs = model(x)
    print(logits_outputs.shape)
