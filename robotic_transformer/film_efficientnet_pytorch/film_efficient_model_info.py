# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet, get_model_params
from film_efficient_model import FiLM


class FiLMEfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Load EfficientNet backbone with pre-trained weights
        # blocks_args, global_params = get_model_params('efficientnet-b3', None)
        # self.backbone = EfficientNet(blocks_args=blocks_args, global_params=global_params)
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')

        # Replace MBConvBlock with FiLMBlock
        for idx, block in enumerate(self.backbone._blocks):
            if isinstance(block, EfficientNet.MBConvBlock):
                self.backbone._blocks[idx] = FiLMBlock(block)

        # Replace the last linear layer with a new one
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x, context):
        x = self.backbone.extract_features(x)  # Get features from the backbone
        x = self.backbone._avg_pooling(x)  # Global average pooling
        x = x.flatten(start_dim=1)  # Flatten
        x = self.classifier(x)  # Classification layer
        return x


class FiLMBlock(nn.Module):
    def __init__(self, base_block):
        super().__init__()
        self.base_block = base_block
        self.base_block._depthwise_conv.conv2d.weight.data.fill_(
            0)  # Set weights to 0
        self.film = FiLM(base_block._depthwise_conv.conv2d.in_channels,
                         base_block._depthwise_conv.conv2d.out_channels)

    def forward(self, x, context):
        x = self.base_block(x)
        x = self.film(x, context)
        return x


if __name__ == "__main__":
    # model = FiLMEfficientNet()
    blocks_args, global_params = get_model_params('efficientnet-b3', None)
    print(blocks_args)
    print('test')
    print(global_params)

    # pdb.set_trace()  # 设置断点

    model = EfficientNet(blocks_args=blocks_args, global_params=global_params)
    input_tensor = torch.randn(6, 3, 300, 300)
    # context_embed = torch.randn(2,3)
    # output_tensor = model(input_tensor,context_embed)
    output_tensor = model(input_tensor)
    # for idx, block in enumerate(model._blocks):

    for idx, block in enumerate(model._blocks):
        print(f"Block {idx+1} output shape: {block._project_conv.weight.shape}")
    print((output_tensor.shape))
    print(model.extract_features(input_tensor).shape)
    # print(model.extract_endpoints(input_tensor))
    print(type(model.extract_features(input_tensor)))
    # print(model.extract_endpoints(input_tensor))
    # print(model.extract_features(input_tensor))
