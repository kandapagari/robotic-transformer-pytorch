# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# 定义一个包含FiLM模块的EfficientNet模型


class FiLMEfficientNet(nn.Module):
    def __init__(self):
        super(FiLMEfficientNet, self).__init__()
        # 加载EfficientNet模型，并冻结所有层
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        # 获取EfficientNet模型的MBConvBlock模块
        self.mbconv_block = self.efficientnet._blocks[-1]
        # 定义FiLM模块
        self.gamma_fc = nn.Linear(2048, 2048)
        self.beta_fc = nn.Linear(2048, 2048)

    def forward(self, x):
        x = self.efficientnet._swish(
            self.efficientnet._bn0(self.efficientnet._conv_stem(x)))
        # 在MBConvBlock模块后添加FiLM模块
        for idx, block in enumerate(self.efficientnet._blocks):
            drop_connect_rate = self.efficientnet._global_params.drop_connect_rate
            if isinstance(block, EfficientNet.MBConvBlock):
                if idx == len(self.efficientnet._blocks) - 1:
                    # 在MBConvBlock模块的最后一层添加FiLM模块
                    x = block(x)
                    x = self.gamma_fc(x) * x + self.beta_fc(x)
                else:
                    # 在MBConvBlock模块的其他层中正常运行
                    x = block(x, drop_connect_rate)
            else:
                x = block(x)
        # 加入EfficientNet模型的其余部分
        x = self.efficientnet._conv_head(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.efficientnet._dropout(x)
        x = self.efficientnet._fc(x)

        return x


if __name__ == '__main__':
    # 创建一个FiLMEfficientNet模型实例
    model = FiLMEfficientNet()
    input_tensor = torch.randn(2, 3, 300, 300)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
    # 打印模型的结构
    print(model)
