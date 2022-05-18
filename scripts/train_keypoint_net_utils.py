# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import ConcatDataset, DataLoader

from kp2d.datasets.augmentations import (ha_augment_sample, resize_sample,
                                         spatial_augment_sample,
                                         to_tensor_sample, add_noise)
from kp2d.datasets.coco import COCOLoader
from kp2d.datasets.sonarsim import SonarSimLoader




def sample_to_cuda(data):
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        data_cuda = {}
        for key in data.keys():
            data_cuda[key] = sample_to_cuda(data[key])
        return data_cuda
    elif isinstance(data, list):
        data_cuda = []
        for key in data:
            data_cuda.append(sample_to_cuda(key))
        return data_cuda
    else:
        return data.to('cuda')


def image_transforms(noise_util, config):
    mode = config.augmentation.mode
    if mode=='sonar_sim':
        def train_transforms(sample):
            sample = resize_sample(sample, image_shape=config.augmentation.image_shape)
            sample = noise_util.pol_2_cart_sample(sample)
            sample = noise_util.augment_sample(sample)

            sample = noise_util.filter_sample(sample)
            sample = noise_util.cart_2_pol_sample(sample)
            sample = to_tensor_sample(sample)


            return sample
    elif mode=='sonar_real': #TODO
        def train_transforms(sample):

            sample = noise_util.pol_2_cart_sample(sample)
            sample = noise_util.augment_sample(sample)

            sample = noise_util.filter_sample(sample)
            sample = noise_util.cart_2_pol_sample(sample)
            sample = to_tensor_sample(sample)

            return sample
    elif mode=='default':
        def train_transforms(sample):
            sample = resize_sample(sample, image_shape=config.augmentation.image_shape)
            sample = spatial_augment_sample(sample)
            sample = to_tensor_sample(sample)
            sample = ha_augment_sample(sample, jitter_paramters=config.augmentation.jittering)
            return sample

    return {'train': train_transforms}




def _set_seeds(seed=42):
    """Set Python random seeding and PyTorch seeds.
    Parameters
    ----------
    seed: int, default: 42
        Random number generator seeds for PyTorch and python
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_datasets_and_dataloaders(config,noise_util):
    """Prepare datasets for training, validation and test."""
    def _worker_init_fn(worker_id):
        """Worker init fn to fix the seed of the workers"""
        _set_seeds(42 + worker_id)

    data_transforms = image_transforms(noise_util,config)
    train_dataset = SonarSimLoader(config.train.path, data_transform=data_transforms['train'])
    # Concatenate dataset to produce a larger one
    if config.train.repeat > 1:
        train_dataset = ConcatDataset([train_dataset for _ in range(config.train.repeat)])

    # Create loaders

    sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=config.train.batch_size,
                              pin_memory=False, # pin memory on seems to create an error
                              shuffle=True,
                              num_workers=config.train.num_workers,
                              worker_init_fn=_worker_init_fn,
                              sampler=sampler,
                              drop_last=True)
    return train_dataset, train_loader
