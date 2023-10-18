# -*- coding: utf-8 -*-
import torch

from robotic_transformer import RT1, MaxViT


def main():
    vit = MaxViT(
        num_classes=1000,
        dim_conv_stem=64,
        dim=96,
        dim_head=32,
        depth=(2, 2, 5, 2),
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1
    )
    # vit_1 = timm.create_model(
    #     'maxvit_base_tf_384.in1k', pretrained=True, features_only=True)
    model = RT1(
        vit=vit,
        num_actions=15,
        depth=6,
        heads=8,
        dim_head=64,
        cond_drop_prob=0.2
    )
    video = torch.randn(2, 3, 6, 224, 224)
    instructions = [
        'bring me that apple sitting on the table',
        'please pass the butter'
    ]
    # (2, 6, 11, 256) # (batch, frames, actions, bins)
    train_logits = model(video, instructions)
    # after much training
    model.eval()
    # classifier free guidance with conditional scale of 3
    eval_logits = model(video, instructions, cond_scale=3.)
    print(train_logits.shape, eval_logits.shape)


if __name__ == "__main__":
    main()
