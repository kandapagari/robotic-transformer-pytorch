# -*- coding: utf-8 -*-
from typing import Any

import tensorflow_datasets as tfds
from torch.utils.data import Dataset


class TFRecordDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train') -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.builder = tfds.builder_from_directory(builder_dir=data_dir)
        self.builder.download_and_prepare()
        self.loaded_dataset = self.builder.as_data_source()

    def __len__(self):
        return self.loaded_dataset[self.split].length

    def __getitem__(self, idx) -> Any:
        images = []
        actions = []
        info = []
        steps = self.loaded_dataset[self.split][idx]['steps']
        instruction = list(steps)[0]['observation']['natural_language_instruction'].decode('utf-8'
                                                                                           )
        for step in steps:
            images.append(step['observation']['image'])
            actions.append(step['action'])
            info.append(step['info'])
        assert len(images) == len(actions) == len(
            info), "Error loading dataset"
        return images, actions, info, instruction


if __name__ == '__main__':
    dataset_name = "fractal_fractal_20220817_data_traj_transform_rt_1_without_filters_disable_episode_padding_seq_length_6_no_preprocessor"  # NOQA
    data_dir = "/data/net/ml_data/google_dataset/rt-1-data-release"
    split = 'train'
    tf_record_dataset = TFRecordDataset(data_dir, split)
    for item in iter(tf_record_dataset):
        images, actions, info, instruction = item
        print(len(images))
    # images, actions, info, instruction = next(iter(tf_record_dataset))
    # images_1, actions_1, info_1, instruction_1 = next(iter(tf_record_dataset))
    print()
