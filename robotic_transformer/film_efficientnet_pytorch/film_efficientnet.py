# -*- coding: utf-8 -*-
# ---------------------------------------------
# 仿照rt-1 pytorch版本film_efficientnet架构改写
# By andyoung007
# 该文件输出了efficientnet模型的一些架构参数信息
# ---------------------------------------------

# from transformers import AutoImageProcessor, EfficientNetForImageClassification
# import torch
# from datasets import load_dataset

# # dataset = load_dataset("huggingface/cats-image")
# # image = dataset["test"]["image"][0]
# image = torch.randn(1,3,224,224)

# image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b3")
# model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b3")

# inputs = image_processor(image, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])

# 另外一个版本
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet, get_model_params


class FiLMBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gamma = nn.Linear(in_features, out_features)
        self.beta = nn.Linear(in_features, out_features)

    def forward(self, x, context):
        gamma = self.gamma(context)
        beta = self.beta(context)
        return gamma.view(*gamma.shape, 1, 1) * x + beta.view(*beta.shape, 1, 1)


class FiLMEfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        # Add FiLM blocks after every block in the backbone
        for i in range(len(self.backbone._blocks)):
            block = self.backbone._blocks[i]
            in_features = block._project_conv.out_channels
            out_features = block._project_conv.out_channels
            film_block = FiLMBlock(in_features, out_features)
            setattr(self, f'film{i}', film_block)
        self.classifier = nn.Linear(1280, num_classes)
        # Initialize the FiLM blocks to zero
        for m in self.modules():
            if isinstance(m, FiLMBlock):
                nn.init.zeros_(m.gamma.weight)
                nn.init.zeros_(m.beta.weight)

    def forward(self, x, context):
        # Backbone
        for i, block in enumerate(self.backbone._blocks):
            x = block(x)
            film_block = getattr(self, f'film{i}')
            x = film_block(x, context)
        # Head
        x = self.backbone._swish(
            self.backbone._bn1(self.backbone._conv_head(x)))
        x = self.backbone._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # model = FiLMEfficientNet()
    blocks_args, global_params = get_model_params('efficientnet-b3', None)
    model = EfficientNet(blocks_args=blocks_args, global_params=global_params)
    input_tensor = torch.randn(6, 3, 300, 300)
    # summary(model, (input_tensor))
    # context_embed = torch.randn(2,3)
    # output_tensor = model(input_tensor,context_embed)
    output_tensor = model(input_tensor)
    for idx, block in enumerate(model._blocks):
        print(f"Block {idx+1} output shape: {block._project_conv.weight.shape}")
        # model._blocks下就是MBConv模块，需要在这里加入Film模块并将其权重初始化为零，另存为新模型
    print((output_tensor.shape))
    print(model.extract_features(input_tensor).shape)
    # 下采样特征维度信息打印
    for i in range(len(model.extract_endpoints(input_tensor))):
        # print(i)
        print(
            f"the shape of the reduction_{i} is",
            model.extract_endpoints(input_tensor)[f'reduction_{i + 1}'].shape,
        )
        # print(model.extract_endpoints(input_tensor).shape)
    print(type(model.extract_features(input_tensor)))
    # print(model)
    # print(model.extract_endpoints(input_tensor))
    # print(model.extract_features(input_tensor))
