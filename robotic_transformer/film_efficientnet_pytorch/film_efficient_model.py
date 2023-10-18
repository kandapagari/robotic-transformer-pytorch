# -*- coding: utf-8 -*-
# import faulthandler
# faulthandler.enable()
import torch
import torch.nn as nn
import torch.nn.init as init

from robotic_transformer.film_efficientnet_pytorch.model import (EfficientNet,
                                                                 MBConvBlock)
from robotic_transformer.film_efficientnet_pytorch.tokenlearner_pytorch import \
    TokenLearner
from robotic_transformer.film_efficientnet_pytorch.USE import USEncoder
from robotic_transformer.film_efficientnet_pytorch.utils import \
    get_model_params

# import pdb
# print(dir(EfficientNet))

# FiLM模块应该是没有问题了


class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gamma = nn.Linear(in_channels, out_channels)
        self.beta = nn.Linear(in_channels, out_channels)
        # 模块权重及偏置初始化为零
        init.zeros_(self.gamma.weight)
        init.zeros_(self.gamma.bias)
        init.zeros_(self.beta.weight)
        init.zeros_(self.beta.bias)

    def forward(self, x, context):
        gamma = self.gamma(context).unsqueeze(2).unsqueeze(3)
        beta = self.beta(context).unsqueeze(2).unsqueeze(3)
        return x * (1 + gamma) + beta


class FiLMEfficientNet(nn.Module):
    def __init__(
        self,
        USEncoder,
        backbone,
        # film, # film可以不作为传递参数
        tokenlearner,
        # transformers_decoder,
        num_classes=1000
    ):
        super().__init__()

        # define the USEncoder
        self.USEncoder = USEncoder
        # Load EfficientNet backbone with pre-trained weights,
        # copy the weight to the backbone model
        # pretrained_backbone = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone = backbone
        self.backbone._blocks_args, self.backbone._global_params = get_model_params(
            'efficientnet-b3', None)
        self.backbone = EfficientNet(
            blocks_args=self.backbone._blocks_args, global_params=self.backbone._global_params)
        self.backbone.load_state_dict(pretrained_backbone.state_dict())
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
        self.tokenlearner = TokenLearner(S=8)
        # Replace the last linear layer with a new one
        self.classifier = nn.Linear(1280, num_classes)

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
        # 原来的模型尾部处理注释掉，主要是为了提取融合instruction的特征
        # x = self.backbone.extract_features(x)  # Get features from the backbone
        # x = self.backbone._avg_pooling(x)  # Global average pooling
        # x = x.flatten(start_dim=1)  # Flatten
        # x = self.classifier(x)  # Classification layer
        return x

# 暂时没有用这个类


class FiLMBlock(nn.Module):
    def __init__(self, base_block):
        super().__init__()
        self.base_block = base_block
        # 这里需要注意FilM模块的输入输出通道数量，需要和原来MBConv模块相匹配
        # self.film = FiLM(base_block._depthwise_conv.conv2d.in_channels,\
        # base_block._depthwise_conv.conv2d.out_channels)
        # self.film = FiLM(base_block._bn2.num_features, base_block._bn2.num_features)
        # 当指令被编码为embedding为384的向量时
        self.film = FiLM(384, base_block._bn2.num_features)

    def forward(self, x, context):
        x = self.base_block(x)
        x = self.film(x, context)
        return x


if __name__ == '__main__':

    USEncoder_model = USEncoder()
    pretrained_backbone = EfficientNet.from_pretrained('efficientnet-b3')
    tklr = TokenLearner
    model = FiLMEfficientNet(USEncoder_model, pretrained_backbone, tklr)
    # blocks_args, global_params = get_model_params('efficientnet-b3', None)
    # print(blocks_args)
    # print('test')
    # print(global_params)
    # pdb.set_trace()  # 设置断点
    # model = EfficientNet(blocks_args=blocks_args, global_params=global_params)
    # with open('output.txt', 'w') as f:
    #     # 重定向标准输出到文件对象
    #     # print('Hello, world!', file=f)
    #     # print('This is a test.', file=f)
    #     for name, module in model.named_modules():
    #         print(name, module, file=f)
    input_tensor = torch.randn(6, 3, 300, 300)
    sentences = ["Pick apple from top drawer and place on counter."]
    # USE_model = USEncoder()
    # embeddings = USE_model(sentences)
    # context_embed = torch.randn(2,3)
    # output_tensor = model(input_tensor,context_embed)
    output_tensor = model(input_tensor, sentences)
    # # for idx, block in enumerate(model._blocks):
    # #     print(f"Block {idx+1} output shape: {block._project_conv.weight.shape}")
    # print((output_tensor.shape))
    # print(model.extract_features(input_tensor).shape)
    # # print(model.extract_endpoints(input_tensor))
    # print(type(model.extract_features(input_tensor)))
    # # print(model.extract_endpoints(input_tensor))
    # # print(model.extract_features(input_tensor))
    print(output_tensor.shape)
