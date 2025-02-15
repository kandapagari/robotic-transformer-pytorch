# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name='robotic-transformer',
    packages=find_packages(exclude=[]),
    version='0.2',
    license='MIT',
    description='Robotic Transformer - Pytorch',
    author='Phil Wang',
    author_email='lucidrains@gmail.com',
    long_description_content_type='text/markdown',
    url='https://github.com/lucidrains/robotic-transformer-pytorch',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformers',
        'attention mechanism',
        'robotics'
    ],
    install_requires=[
        'classifier-free-guidance-pytorch>=0.1.4',
        'einops>=0.6',
        'torch>=1.6',
        'efficientnet-pytorch>=0.7.1',
        'tfrecord>=1.14.4'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)
