import torch
from torch import nn

from typing import List, Optional
from beartype import beartype

from einops import repeat, reduce, rearrange
from einops.layers.torch import Rearrange

from classifier_free_guidance_pytorch import (TextConditioner, AttentionTextConditioner,
                                              classifier_free_guidance)

from utils import default, pack_one, unpack_one, posemb_sincos_1d
from models import LayerNorm, TokenLearner, Transformer, MaxViT


@beartype
class RT1(nn.Module):
    def __init__(
        self,
        *,
        vit: MaxViT,
        num_actions=11,
        action_bins=256,
        depth=6,
        heads=8,
        dim_head=64,
        token_learner_ff_mult=2,
        token_learner_num_layers=2,
        token_learner_num_output_tokens=8,
        cond_drop_prob=0.2,
        use_attn_conditioner=False,
        conditioner_kwargs: dict = dict()
    ):
        super().__init__()
        self.vit = vit
        self.num_vit_stages = len(vit.cond_hidden_dims)
        conditioner_klass = AttentionTextConditioner if use_attn_conditioner else TextConditioner
        self.conditioner = conditioner_klass(
            hidden_dims=(*tuple(vit.cond_hidden_dims), *
                         ((vit.embed_dim,) * depth * 2)),
            hiddens_channel_first=(
                *((True,) * self.num_vit_stages), *((False,) * depth * 2)),
            cond_drop_prob=cond_drop_prob,
            **conditioner_kwargs
        )
        self.token_learner = TokenLearner(
            dim=vit.embed_dim,
            ff_mult=token_learner_ff_mult,
            num_output_tokens=token_learner_num_output_tokens,
            num_layers=token_learner_num_layers
        )
        self.num_learned_tokens = token_learner_num_output_tokens
        self.transformer_depth = depth
        self.transformer = Transformer(
            dim=vit.embed_dim,
            dim_head=dim_head,
            heads=heads,
            depth=depth
        )
        self.cond_drop_prob = cond_drop_prob
        self.to_logits = nn.Sequential(
            LayerNorm(vit.embed_dim),
            nn.Linear(vit.embed_dim, num_actions * action_bins),
            Rearrange('... (a b) -> ... a b', b=action_bins)
        )

    @classifier_free_guidance
    def forward(
        self,
        video,
        texts: Optional[List[str]] = None,
        cond_drop_prob=0.
    ):
        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        frames, device = video.shape[2], video.device
        cond_fns = self.conditioner(
            texts,
            cond_drop_prob=cond_drop_prob,
            repeat_batch=(*((frames,) * self.num_vit_stages),
                          *((1,) * self.transformer_depth * 2))
        )
        vit_cond_fns, transformer_cond_fns = cond_fns[:-(
            depth * 2)], cond_fns[-(depth * 2):]
        video = rearrange(video, 'b c f h w -> b f c h w')
        images, packed_shape = pack_one(video, '* c h w')
        tokens = self.vit(
            images,
            texts=texts,
            cond_fns=vit_cond_fns,
            cond_drop_prob=cond_drop_prob,
            return_embeddings=True
        )
        tokens = unpack_one(tokens, packed_shape, '* c h w')
        learned_tokens = self.token_learner(tokens)
        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')
        # causal attention mask
        attn_mask = torch.ones(
            (frames, frames), dtype=torch.bool, device=device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)',
                           r1=self.num_learned_tokens, r2=self.num_learned_tokens)
        # sinusoidal positional embedding
        pos_emb = posemb_sincos_1d(
            frames, learned_tokens.shape[-1],
            dtype=learned_tokens.dtype,
            device=learned_tokens.device)
        learned_tokens = learned_tokens + \
            repeat(pos_emb, 'n d -> (n r) d', r=self.num_learned_tokens)
        # attention
        attended_tokens = self.transformer(
            learned_tokens, cond_fns=transformer_cond_fns, attn_mask=~attn_mask)
        pooled = reduce(attended_tokens, 'b (f n) d -> b f d',
                        'mean', f=frames)
        return self.to_logits(pooled)
