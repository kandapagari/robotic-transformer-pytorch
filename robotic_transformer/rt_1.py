# -*- coding: utf-8 -*-
"""RT-1 model architecture class."""
import torch
import torch.nn as nn

from robotic_transformer.film_efficientnet_pytorch import (
    EfficientNet, FiLM, MBConvBlock, TokenLearner, Transformers_Decoder,
    USEncoder, get_model_params)


class RT1model(nn.Module):
    def __init__(
        self,
        backbone_name: str = 'efficientnet-b3',
        d_model: int = 512,
        depth: int = 8,
        dim: int = 512,
        dim_head: int = 64,
        dropout: float = 0.,
        heads: int = 8,
        max_seq_len: int = 48,
        mlp_dim: int = 512,
        num_actions: int = 11,
        num_classes: int = 1000,
        token_learner_layers: int = 8,
        vocab_size: int = 256
    ):
        super().__init__()
        # define the USEncoder
        self.USEncoder = USEncoder()
        # Load EfficientNet backbone with pre-trained weights,
        # copy the weight to the backbone model
        self.pretrained_backbone = EfficientNet.from_pretrained(
            backbone_name)
        self.backbone = self.pretrained_backbone
        self.backbone._blocks_args, self.backbone._global_params = get_model_params(
            backbone_name, None)
        self.backbone = EfficientNet(
            blocks_args=self.backbone._blocks_args, global_params=self.backbone._global_params)
        self.backbone.load_state_dict(self.pretrained_backbone.state_dict())
        # self._swish = MemoryEfficientSwish()
        # self.film = FiLM(self.USEncoder._hidden_size, out_channels)
        self.backbone_with_film = []
        # Replace or append MBConvBlock with FiLMBlock, 添加到一个新的modulelist中去
        for block in self.backbone._blocks:
            # self.backbone._blocks[idx] = FiLMBlock(block)
            self.backbone_with_film.append(block)
            self.backbone_with_film.append(
                FiLM(self.USEncoder._hidden_size, block._bn2.num_features))
        self.backbone_with_film = nn.ModuleList(self.backbone_with_film)
        self.Linear_1b1_conv = nn.Conv2d(1536, 512, 1)
        self.tokenlearner = TokenLearner(S=token_learner_layers)
        self.transformers_decoder = Transformers_Decoder(
            d_model=d_model,
            depth=depth,
            dim=dim,
            dim_head=dim_head,
            dropout=dropout,
            heads=heads,
            max_seq_len=max_seq_len,
            mlp_dim=mlp_dim,
            num_actions=num_actions,
            vocab_size=vocab_size
        )
        # Replace the last linear layer with a new one
        self.classifier = nn.Linear(1280, num_classes)

    def to(self, device):
        self.USEncoder.to(device)
        self.backbone.to(device)
        self.backbone_with_film.to(device)
        self.Linear_1b1_conv.to(device)
        self.tokenlearner.to(device)
        self.transformers_decoder.to(device)
        return self

    def forward(self, x, context_sentences):
        # context换成sentences，然后把相关的处理放到前向函数中来！
        context = self.USEncoder(context_sentences)
        # Stem
        # x = inputs
        inputs = x
        x = self.backbone._swish(self.backbone._bn0(
            self.backbone._conv_stem(inputs)))
        for block in self.backbone_with_film:
            if isinstance(block, MBConvBlock):
                # if 'MBConv' in block.name:
                x = block(x)
            elif isinstance(block, FiLM):
                # elif 'FiLM' in block.name:
                x = block(x, context)
        x = self.backbone._swish(
            self.backbone._bn1(self.backbone._conv_head(x)))
        # 添加通道转换的卷积模块，从efficientnet最后的1536通道转为512通道
        x = self.Linear_1b1_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.tokenlearner(x)
        x = self.transformers_decoder(x)
        # 原来的模型尾部处理注释掉，主要是为了提取融合instruction的特征
        # x = self.backbone.extract_features(x)  # Get features from the backbone
        # x = self.backbone._avg_pooling(x)  # Global average pooling
        # x = x.flatten(start_dim=1)  # Flatten
        # x = self.classifier(x)  # Classification layer
        return x


if __name__ == '__main__':
    input_tensor = torch.randn(6, 3, 224, 224)
    sentences = ["Pick apple from top drawer and place on counter."]
    model = RT1model(num_actions=15, backbone_name='efficientnet-b0')
    output_tensor = model(input_tensor, sentences)
    print(output_tensor.shape)
