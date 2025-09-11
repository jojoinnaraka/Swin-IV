# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Modified by Shijie Zhou
# Modified date: 2025-08-14
# Modifications: for study on internal variability
# --------------------------------------------------------

import torch, os, gc, psutil
import numpy as np
import torch.distributed as dist
import xarray as xr
from natsort import natsorted
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, Sampler


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = natsorted([f for f in os.listdir(data_dir) if f.startswith('samples_') and f.endswith('.nc')])
        self.targets = natsorted([f for f in os.listdir(data_dir) if f.startswith('targets_') and f.endswith('.nc')])
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.samples[idx])
        target_path = os.path.join(self.data_dir, self.targets[idx])

        sample_in = xr.open_dataset(sample_path)
        sample_numpy = sample_in['data'].astype('float32').values
        sample_tensor = torch.from_numpy(sample_numpy).float()
        sample_tensor[torch.isnan(sample_tensor)] = 0
        if sample_tensor.dim() == 2:
            sample_tensor = sample_tensor.unsqueeze(0)

        target_in = xr.open_dataset(target_path)
        target_numpy = target_in['data'].astype('float32').values
        target_tensor = torch.from_numpy(target_numpy).float()
        target_tensor[torch.isnan(target_tensor)] = 0
        if target_tensor.dim() == 2:
            target_tensor = target_tensor.unsqueeze(0)

        return sample_tensor, target_tensor


class ProofDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = natsorted([f for f in os.listdir(data_dir) if f.startswith('sample') and f.endswith('.nc')])
        self.samples = self.samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.samples[idx])
        sample_in = xr.open_dataset(sample_path)
        sample_numpy = sample_in['data'].astype('float32').values
        sample_tensor = torch.from_numpy(sample_numpy).float()
        sample_tensor[torch.isnan(sample_tensor)] = 0
        if sample_tensor.dim() == 2:
            sample_tensor = sample_tensor.unsqueeze(0)

        return sample_tensor


def build_loader(config):
    dataset_train = CustomDataset(os.path.join(config.DATA.DATA_PATH, 'train'))
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully built train dataset")
    dataset_val = CustomDataset(os.path.join(config.DATA.DATA_PATH, 'val'))
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully built val dataset")
    dataset_test = ProofDataset(os.path.join(config.DATA.DATA_PATH, config.TEST_MODE_PATH))
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully built test dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    if config.TEST.SEQUENTIAL:
        sampler_val = SequentialSampler(dataset_val)
        sampler_test = SequentialSampler(dataset_test)
    else:
        sampler_val = DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=config.TEST.SHUFFLE)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test


def load_mean_std(config):

    mean_path = os.path.join(config.DATA.DATA_PATH, 'all_mean.nc')
    mean_in = xr.open_dataset(mean_path)
    mean_numpy = mean_in[config.DATA.SAMPLE_VAR_NAME].astype('float32').values
    mean = torch.from_numpy(mean_numpy).float().cuda()

    std_path = os.path.join(config.DATA.DATA_PATH, 'all_std.nc')
    std_in = xr.open_dataset(std_path)
    std_numpy = std_in[config.DATA.SAMPLE_VAR_NAME].astype('float32').values
    std = torch.from_numpy(std_numpy).float().cuda()

    return mean, std


def load_daily_mean_std(config):

    mean_path = os.path.join(config.DATA.DATA_PATH, 'daily_mean.nc')
    mean_in = xr.open_dataset(mean_path)
    mean_numpy = mean_in[config.DATA.SAMPLE_VAR_NAME].astype('float32').values
    mean = torch.from_numpy(mean_numpy).float().cuda()

    return mean


def load_constantmask(config):

    # load the land-sea mask data
    lsmask_path = os.path.join(config.DATA.DATA_PATH, 'land_sea_mask.nc')
    lsmask_in = xr.open_dataset(lsmask_path)
    lsmask_numpy = lsmask_in['data'].astype('float32').values
    lsmask = torch.from_numpy(lsmask_numpy).float()
    lsmask = lsmask.unsqueeze(0)

    # load the topography (topo) data
    # topo_path = os.path.join(config.DATA.DATA_PATH, 'topography.nc')
    # topo_in = xr.open_dataset(topo_path)
    # topo_numpy = topo_in['XXXXXXXXXXXXXX'].astype('float32').values
    # topo = torch.from_numpy(topo_numpy).float()

    #constantmasks = torch.stack(lsmask, topo)

    return lsmask