# -*- coding: utf-8 -*-
# import os
# 设置最大递归深度，保证循环打印输出不报错，实际上没有起到作用，报错
import sys

# import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

# import torch

sys.setrecursionlimit(100000)
# os.environ.pop('TFDS_DATA_DIR', None)

# dataset_name = "RT_1_paper_release:1.0.0"
# 此处为了加载文件时默认生成的array_record文件名称和实际文件相匹配，修改了dataset_name，同时数据集文件夹中dataset_info.json第四行的名称也修改
# "RT_1_paper_release"为
# "fractal_fractal_20220817_data_traj_transform_rt_1_without_filters_disable_episode_padding_seq_length_6_no_preprocessor"
dataset_name = "fractal_fractal_20220817_data_traj_transform_rt_1_without_filters_disable_episode_padding_seq_length_6_no_preprocessor"  # NOQA
data_dir = "/data/net/ml_data/google_dataset/rt-1-data-release"
split = 'train'
builder = tfds.builder_from_directory(builder_dir=data_dir)
builder.download_and_prepare()
loaded_dataset = builder.as_data_source()
# loaded_dataset = builder.as_dataset()
images = []
for example in loaded_dataset['train']:
    steps = list(example['steps'])
    for step in steps:
        images.append(Image.fromarray(step['observation']['image']))
    print()

# if dataset_name in tfds.list_builders():
#     print(f"{dataset_name} is available in TensorFlow Datasets.")
# else:
#     print(f"{dataset_name} is not available in TensorFlow Datasets.")

load_dataset_path = '/data/net/ml_data/google_dataset/rt-1-data-release'  # 数据集所在目录
# load_dataset_path = 'RT_1_paper_release:1.0.0'
loaded_dataset = tfds.builder_from_directory(
    load_dataset_path).as_data_source(split='all')
# loaded_dataset = tfds.as_data_source(load_dataset_path)
# loaded_dataset = loaded_dataset.as_dataset()

example_keys = range(1)
examples = loaded_dataset.__getitems__(example_keys)


# for e in examples:
print(examples)

# builder = tfds.builder('RT1datasets', data_dir=data_dir)
# builder.download_and_prepare()
# dataset = builder.as_dataset()

# for episode in dataset:
#     # 将 episode 中的每个 step 转换为 PyTorch 张量
#     for step in episode['steps']:
#         obs = step['observation']
#         obs_tensor = {k: torch.tensor(v) for k, v in obs.items()}
#         action = step['action']
#         action_tensor = {k: torch.tensor(v) for k, v in action.items()}
#         reward = torch.tensor(step['reward'])
#         # 在此处使用观测、动作和奖励执行一些操作
#         ...

# # 创建 TFRecordReader 对象
# reader = tf.io.TFRecordReader()

# 打开 TFRecord 文件
filename = f'{data_dir}/fractal_fractal_20220817_data_traj_transform_rt_1_without_filters_disable_episode_padding_seq_length_6_no_preprocessor-train.array_record-00000-of-01024'  # NOQA
# filename_queue = tf.train.string_input_producer([filename], num_epochs=1)

# # 读取数据
# _, serialized_example = reader.read(filename_queue)

# # 解析数据
# features = tf.parse_single_example(
#     serialized_example,
#     features={
#         'feature_1': tf.FixedLenFeature([], tf.float32),
#         'feature_2': tf.FixedLenFeature([], tf.int64),
#         'feature_3': tf.FixedLenFeature([], tf.string)
#     })

# # 获取解析后的数据
# feature_1 = features['feature_1']
# feature_2 = features['feature_2']
# feature_3 = features['feature_3']

# # 启动会话并读取数据
# with tf.Session() as sess:
#     # 初始化变量
#     tf.global_variables_initializer().run()
#     tf.local_variables_initializer().run()

#     # 创建 Coordinator 对象，并启动队列
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#     # 读取数据
#     try:
#         while not coord.should_stop():
#             f1, f2, f3 = sess.run([feature_1, feature_2, feature_3])
#             # 在这里处理解析后的数据
#             print(f1, f2, f3)
#     except tf.errors.OutOfRangeError:
#         print('Done!')
#     finally:
#         # 停止队列
#         coord.request_stop()

#     coord.join(threads)

# filenames = [filename]
# raw_dataset = tf.data.TFRecordDataset(filenames)
# print(raw_dataset)

# # tf.python_io.tf_record_iterator
# for raw_record in tf.data.TFRecordDataset(filename):
#   print(repr(raw_record))
# for raw_record in raw_dataset.take(1):
#   print(repr(raw_record))
