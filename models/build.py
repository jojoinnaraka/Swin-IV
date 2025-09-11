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

from .swin_transformer_v2 import SwinTransformerV2
from .deepCNN import DeepCNN


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  var_size=config.DATA.VAR_SIZE,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
    elif model_type == 'deepcnn':
        model = DeepCNN(img_size=[144, 288],
                        in_chans=1,
                        base_chans=32,
                        out_chans=1,
                        drop_rate=0.,
                        use_checkpoint=False)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
