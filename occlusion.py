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

import xarray as xr
import netCDF4 as nc
import os
import json
import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

from config import get_config
from models import build_model
from logger import create_logger
from captum.attr import Occlusion


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--leadday', type=int, help="the number of lead days")
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    #parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--test', action='store_true', help='Perform test only')
    parser.add_argument('--test_path', type=str, help='Set the test path')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get('LOCAL_RANK', '0')), required=False, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def load_checkpoint(config, model, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    model_state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    load_result = model.load_state_dict(model_state, strict=False)
    
    if logger:
        logger.info(f"Missing keys: {load_result.missing_keys}")
        logger.info(f"Unexpected keys: {load_result.unexpected_keys}")
        logger.info(f"Model loaded from {config.MODEL.RESUME}")

    del checkpoint
    torch.cuda.empty_cache()

def main(config):

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, logger)

    model.eval()
    torch.cuda.empty_cache()

    test(model)


def occlusion_analysis(model, input_img):

    device = next(model.parameters()).device
    input_img = input_img.to(device)

    def create_latlon_mask(lat_min=-5, lat_max=5, lon_min=150, lon_max=240):
        lat = torch.linspace(89.375, -89.375, 144)
        lon = torch.linspace(0.625, 359.375, 288)

        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)

        mask = lat_mask.unsqueeze(1) & lon_mask.unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0).cuda().float()

        return mask

    def loss_fn(inputs):
        region_mask = create_latlon_mask(lat_min=-5, lat_max=5, lon_min=160, lon_max=240)
        #region_mask = create_latlon_mask(lat_min=-30, lat_max=30, lon_min=0, lon_max=360)
        outputs = model(inputs)

        global_mean = (outputs * region_mask).sum(dim=[1,2,3]) / region_mask.sum()
        return global_mean

    occlusion = Occlusion(loss_fn)
    
    with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        attributions = occlusion.attribute(inputs=input_img,strides = (1, 4, 4),sliding_window_shapes=(1, 16, 16),baselines=0,perturbations_per_eval=1,show_progress=True)

    print(f"Attributions shape: {attributions.shape}")
    return attributions.detach().squeeze().cpu().numpy()


def test(model):

    model.eval()
    torch.cuda.empty_cache()
    
    output_dir = "/home/shijie/Projects/swintrans/data_process/MMLEA/RFmodel_test_ts2pr/test_4r"

    input_path = "/home/shijie/Projects/swintrans/data_process/MMLEA/RFmodel_test_ts2pr/test_4r/samples_181.nc"
    input_in = xr.open_dataset(input_path)
    input_numpy = input_in['data'].astype('float32').values

    input_tensor = torch.from_numpy(input_numpy).float()
    input_tensor[torch.isnan(input_tensor)] = 0
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    attr = occlusion_analysis(model, input_tensor)

    output_ds = xr.Dataset(
        data_vars={
            'data': (['lat', 'lon'],  attr.astype('float32'))
        },
    coords={
            'lat': np.linspace(89.375, -89.375, 144),
            'lon': np.linspace(0.625, 359.375, 288)
        }
        )

    output_path = os.path.join(output_dir, 'oc_pacificmean.nc') #'oc_tropicalmean.nc'
    if os.path.exists(output_path):
        os.remove(output_path)
    output_ds.to_netcdf(output_path)

    return

if __name__ == '__main__':
    args, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name="TEST_LOGGER")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config_test.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)