# -*- coding: utf-8 -*-
import torch

from robotic_transformer.rt_1 import RT1model

if __name__ == "__main__":
    input_tensor = torch.randn(6, 3, 224, 224)
    sentences = ["Pick apple from top drawer and place on counter."]
    model = RT1model(num_actions=15, backbone_name='efficientnet-b7')
    output_tensor = model(input_tensor, sentences)
    print(output_tensor.shape)
