from torch import nn
from utils import unpack_one, pack_one
from einops import rearrange, repeat, reduce


class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
        self,
        *,
        dim,
        ff_mult=2,
        num_output_tokens=8,
        num_layers=2
    ):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim,
                      1, groups=num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens,
                      1, groups=num_output_tokens),
        )

    def forward(self, x):
        x, ps = pack_one(x, '* c h w')
        x = repeat(x, 'b c h w -> b (g c) h w', g=self.num_output_tokens)
        attn = self.net(x)
        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g=self.num_output_tokens)
        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        x = unpack_one(x, ps, '* c n')
        return x
