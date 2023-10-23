# -*- coding: utf-8 -*-
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from robotic_transformer import RT1model
from robotic_transformer.data.dataloader import TFRecordDataset


def resize_images(images, size=(224, 224), device='cpu'):
    resized_images = []
    for image in images:
        _image = Image.fromarray(image)
        _image = _image.resize(size, Image.Resampling.NEAREST)
        resized_images.append(ToTensor()(_image))
    resized_images = torch.as_tensor(np.array(resized_images))
    return resized_images.to(device)


if __name__ == "__main__":
    input_tensor = torch.randn(6, 3, 224, 224)
    sentences = ["Pick apple from top drawer and place on counter."]
    data_dir = "/data/net/ml_data/google_dataset/rt-1-data-release"
    split = 'train'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    tf_record_dataset = TFRecordDataset(data_dir, split)
    model = RT1model(num_actions=7, backbone_name='efficientnet-b3').to(device)
    for item in iter(tf_record_dataset):
        images, actions, info, instruction = item
        input_tensor = resize_images(images[:6], device=device)
        sentences = [instruction]
        output_tensor = model(input_tensor, sentences)
        print(output_tensor.shape)
