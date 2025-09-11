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
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from data import load_mean_std, load_constantmask
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor

from kornia.losses import SSIMLoss


class SSIM_MSE_Loss(nn.Module):
    def __init__(self, ssim_weight: float = 0.5, mse_weight: float = 0.5, window_size: int = 13, ssim_max_val: float = 1.0, eps: float = 1e-8):

        super().__init__()
        self._validate_parameters(ssim_weight, mse_weight, window_size)
        
        self.ssim = SSIMLoss(window_size=window_size, reduction='mean', max_val=ssim_max_val)
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.weights = {'ssim': ssim_weight, 'mse': mse_weight}
        self.eps = eps

    def _validate_parameters(self, ssim_w: float,  mse_w: float, win_size: int):
        assert 0 <= ssim_w <= 1, f"SSIM weight must be in the range (0,1), current value: {ssim_w}"
        assert 0 <= mse_w <= 1, f"MSE weight must be in the range (0,1), current value: {mse_w}"
        assert abs(ssim_w + mse_w - 1.0) < 1e-3, f"The sum of weights must be 1.0, current sum: {ssim_w + mse_w}"
        assert win_size % 2 == 1, f"Window size must be an odd number, current size: {win_size}"
        assert win_size >= 3, f"Window size is too small (may cause unstable calculations), current size: {win_size}"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if target.ndim == 3:
            target = target.unsqueeze(1)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
            
        target = signed_normalize(target)
        pred = signed_normalize(pred)

        loss_ssim = self.ssim(pred, target) * 100.0
        loss_mse = self.mse(pred, target) * 10000.0
        
        total_loss = (
            self.weights['ssim'] * loss_ssim.clamp_min(self.eps) + 
            self.weights['mse'] * loss_mse.clamp_min(self.eps)
        )
        return total_loss

class SSIM_Loss(nn.Module):
    def __init__(self, window_size: int = 13, ssim_max_val: float = 1.0):

        super().__init__()
        self._validate_parameters(window_size)
        
        self.ssim = SSIMLoss(window_size=window_size, reduction='mean', max_val=ssim_max_val)

    def _validate_parameters(self, win_size: int):
        assert win_size % 2 == 1, f"Window size must be an odd number, current size: {win_size}"
        assert win_size >= 3, f"Window size is too small (may cause unstable calculations), current size: {win_size}"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if target.ndim == 3:
            target = target.unsqueeze(1)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
            
        target = signed_normalize(target)
        pred = signed_normalize(pred)

        loss_ssim = self.ssim(pred, target) * 100.0
        
        return loss_ssim

def signed_normalize(x):

    pos_mask = (x > 0)
    neg_mask = (x < 0)
    
    abs_max = 50.0
    
    x_norm = torch.zeros_like(x) + 0.5
    x_norm[pos_mask] = 0.5 + 0.5 * (x[pos_mask] / abs_max)  # [0.5,1]
    x_norm[neg_mask] = 0.5 + 0.5 * (x[neg_mask] / abs_max)       # [0,0.5]
    
    return x_norm

def create_lat_weight():
    lat = torch.linspace(89.375, -89.375, 144)
    lat_rad = torch.deg2rad(lat)
    lat_cos = torch.cos(lat_rad)

    lat_weight = torch.sqrt(144 * lat_cos / lat_cos.sum())
    lat_weight = lat_weight.unsqueeze(0).unsqueeze(1).unsqueeze(3)
    return lat_weight.cuda()

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


def main(config):
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(config)

    #constant_masks = load_constantmask(config)
    constant_masks = None

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

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    #criterion = torch.nn.L1Loss()
    #criterion = torch.nn.MSELoss(reduction='mean')
    #criterion = SSIM_Loss()
    criterion = SSIM_MSE_Loss()

    min_rmse = 10000.0 

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        min_rmse = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        if config.TEST_MODE:
            test(config, data_loader_test, constant_masks, model)
            return
        rmse, loss = validate(config, data_loader_val, constant_masks, model, criterion)
        logger.info(f"RMSE of the network on the {len(dataset_val)} test data: {rmse:.1f}")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, constant_masks, optimizer, epoch, lr_scheduler,
                        loss_scaler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, min_rmse, optimizer, lr_scheduler, loss_scaler,
                            logger)

        rmse, loss = validate(config, data_loader_val, constant_masks, model, criterion)
        logger.info(f"RMSE of the network on the {len(dataset_val)} test data: {rmse:.1f}")
        min_rmse = min(min_rmse, rmse)
        logger.info(f'Min RMSE: {min_rmse:.2f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, criterion, data_loader, constant_masks, optimizer, epoch, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()

    lat_weight = create_lat_weight()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples, constant_masks) # for Swin T
            #outputs = model(samples) # for Deep CNN

        mask = (outputs != 0) & (targets != 0)
        mask_float = mask.float()
        outputs_w = outputs * lat_weight * mask_float
        targets_w = targets * lat_weight * mask_float

        outputs_final = outputs_w
        targets_final = targets_w

        loss = criterion(outputs_final, targets_final)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(config, data_loader, constant_masks, model, criterion):
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    rmse_meter = AverageMeter()

    lat_weight = create_lat_weight()

    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples, constant_masks)

        mask = (outputs != 0) & (targets != 0)
        mask_float = mask.float()
        outputs_w = outputs * lat_weight * mask_float
        targets_w = targets * lat_weight * mask_float

        outputs_final = outputs_w
        targets_final = targets_w

        loss = criterion(outputs_final, targets_final)
        rmse = torch.sqrt(torch.nn.functional.mse_loss(outputs_w, targets_w, reduction='mean'))

        rmse = reduce_tensor(rmse)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), targets.size(0))
        rmse_meter.update(rmse.item(), targets.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'RMSE {rmse_meter.val:.3f} ({rmse_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * RMSE {rmse_meter.avg:.3f}')

    return rmse_meter.avg, loss_meter.avg

@torch.no_grad()
def test(config, data_loader, constant_masks, model):

    model.eval()
    torch.cuda.empty_cache()
    
    batch_time = AverageMeter()
    
    output_dir = os.path.join(config.DATA.DATA_PATH, config.TEST_MODE_PATH, config.TAG)
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        end = time.time()
        
        for idx, samples in enumerate(data_loader):
            samples = samples.cuda(non_blocking=True)
            
            outputs = model(samples, constant_masks)
            outputs_real = outputs.float()
            
            output_data = outputs_real.squeeze().cpu().numpy()

            output_ds = xr.Dataset(
                data_vars={
                    'data': (['lat', 'lon'], output_data.astype('float32'))
                },
            coords={
                    'lat': np.linspace(89.375, -89.375, output_data.shape[0]),
                    'lon': np.linspace(0.625, 359.375, output_data.shape[1])
                }
                )
        
            output_path = os.path.join(output_dir, f'output_{(idx+1):04d}.nc')
            output_ds.to_netcdf(output_path)
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx+1}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

    return

@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

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

    # linear scale the learning rate according to total batch size (original: 512), may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)