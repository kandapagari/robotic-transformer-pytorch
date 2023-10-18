# -*- coding: utf-8 -*-
from typing import Callable

from beartype import beartype
from einops.layers.torch import Rearrange, Reduce
from torch import nn

from robotic_transformer.models.attention import Attention
from robotic_transformer.models.feed_forward import FeedForward
from robotic_transformer.models.layer_norm import LayerNorm
from robotic_transformer.models.mb_conv import MBConv
from robotic_transformer.models.residual import Residual
from robotic_transformer.utils import default, exists


class MaxViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head=32,
        dim_conv_stem=None,
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
        channels=3
    ):
        super().__init__()
        assert isinstance(
            depth, tuple), 'depth needs to be tuple if integers indicating number of transformer\
                blocks at that stage'

        # convolutional stem
        dim_conv_stem = default(dim_conv_stem, dim)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride=2, padding=1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding=1)
        )
        # variables
        num_stages = len(depth)
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        self.layers = nn.ModuleList([])
        # shorthand for window size for efficient block - grid like attention
        w = window_size
        # iterate through stages
        cond_hidden_dims = []
        for (layer_dim_in, layer_dim), layer_depth in zip(dim_pairs, depth):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim
                cond_hidden_dims.append(stage_dim_in)
                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample=is_first,
                        expansion_rate=mbconv_expansion_rate,
                        shrinkage_rate=mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d',
                              w1=w, w2=w),  # block-like attention
                    Residual(Attention(dim=layer_dim, dim_head=dim_head,
                             dropout=dropout, window_size=w)),
                    Residual(FeedForward(dim=layer_dim, dropout=dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d',
                              w1=w, w2=w),  # grid-like attention
                    Residual(Attention(dim=layer_dim, dim_head=dim_head,
                             dropout=dropout, window_size=w)),
                    Residual(FeedForward(dim=layer_dim, dropout=dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )
                self.layers.append(block)
        embed_dim = dims[-1]
        self.embed_dim = dims[-1]
        self.cond_hidden_dims = cond_hidden_dims
        # mlp head out
        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    @beartype
    def forward(
        self,
        x,
        texts: list[str] | None = None,
        cond_fns: tuple[Callable, ...] | None = None,
        cond_drop_prob=0.,
        return_embeddings=False
    ):
        x = self.conv_stem(x)
        cond_fns = iter(default(cond_fns, []))
        for stage in self.layers:
            cond_fn = next(cond_fns, None)
            if exists(cond_fn):
                x = cond_fn(x)
            x = stage(x)
        return x if return_embeddings else self.mlp_head(x)
