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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
#from visualizer import get_local


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, num_windows, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_windows = num_windows
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.earth_pos_bias = nn.Parameter(torch.zeros((2 * window_size[1] - 1) * window_size[0] * window_size[0], self.num_windows, num_heads))
        trunc_normal_(self.earth_pos_bias, std=.02)

        # Index in the latitude of query matrix
        coords_hi = torch.arange(self.window_size[0])
        # Index in the latitude of key matrix
        coords_hj = -torch.arange(self.window_size[0])*self.window_size[0]

        # Index in the longitude of the key-value pair
        coords_w = torch.arange(self.window_size[1])

        # Change the order of the index to calculate the index in total
        coords_1 = torch.stack(torch.meshgrid(coords_hi, coords_w, indexing="ij"))
        coords_2 = torch.stack(torch.meshgrid(coords_hj, coords_w, indexing="ij"))
        coords_flatten_1 = torch.flatten(coords_1, 1) 
        coords_flatten_2 = torch.flatten(coords_2, 1)
        coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
        coords = coords.permute(1, 2, 0).contiguous()

        # Shift the index for each dimension to start from 0
        coords[:, :, 1] += self.window_size[1] - 1
        coords[:, :, 0] *= 2 * self.window_size[1] - 1

        # Sum up the indexes in three dimensions
        position_index = coords.sum(-1)

        # Flatten the position index to facilitate further indexing
        earth_position_index = torch.flatten(position_index)
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self._capture_attn = False
        self.attentions = []

    #@get_local('qkv_bias')
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        max_value = torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))
        logit_scale = torch.clamp(self.logit_scale, max=max_value).exp()
        # logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        earth_pos_bias = self.earth_pos_bias[self.earth_position_index].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], self.num_windows, self.num_heads)
        earth_pos_bias = earth_pos_bias.permute(2, 3, 0, 1).contiguous()
        earth_pos_bias = earth_pos_bias.unsqueeze(1)
        earth_pos_bias = earth_pos_bias.repeat(1, B_ // self.num_windows, 1, 1, 1)
        earth_pos_bias = earth_pos_bias.reshape(B_, self.num_heads, self.window_size[0] * self.window_size[1], -1)
        attn = attn + earth_pos_bias

        if self._capture_attn:
            self.attentions.append(attn.detach().cpu())

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=[6,12], shift_size=[0,0],
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_windows = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, num_windows=self.num_windows, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size[0] > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size) # nW, window_size[0], window_size[1], 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size[0] > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size[0] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))


        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size[0] / self.window_size[1]
        flops += nW * self.attn.flops(self.window_size[0] * self.window_size[1])
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        #assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        
        if H % 2 == 1:
            x = x.permute(0, 3, 1, 2)
            x = torch.nn.functional.pad(x, (0, 0, 0, 1), "replicate")
            x = x.permute(0, 2, 3, 1)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution

        if H % 2 == 1:
            H += 1

        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class PatchExpanding(nn.Module):
    r""" Patch Expanding Layer with PixelShuffle + Learnable Upsampling.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        
        self.expand = nn.Linear(dim, 2 * dim)
        self.ps = nn.PixelShuffle(2)
        
        self.conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, padding_mode='reflect', groups=dim // 2)
        self.norm = norm_layer(dim // 2)
        self.mixup = nn.Linear(dim // 2, dim // 2, bias=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = self.expand(x)
        x = x.view(B, H, W, 2 * C).permute(0, 3, 1, 2)
        
        x = self.ps(x)
        
        if H % 2 == 1:
            x = F.pad(x, (0, 1, 0, 1), mode='reflect')
        
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C // 2)
        
        x = self.norm(x)
        x = self.mixup(x)
        
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = 0
        
        flops += H * W * self.dim * 2 * self.dim

        if H % 2 == 1:
            eff_H = 2 * H + 1
        else:
            eff_H = 2 * H
        eff_W = 2 * W if W % 2 == 0 else 2 * W + 1
        
        flops += (self.dim // 2) * 9 * eff_H * eff_W
        flops += eff_H * eff_W * (self.dim // 2) * (self.dim // 2)
        
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size = [0, 0] if (i % 2 == 0) else [size // 2 for size in window_size],
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

         # patch recovery layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        blocks_output = x

        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)

        return x, blocks_output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size. Default: 144(145) * 288 . Resolution: 1.25 degree * 1.25 degree
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input variables. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=[144, 288], patch_size=4, in_chans=1, mask_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        if img_size[0] % patch_size[0] == 1:
            patches_resolution = [(img_size[0] // patch_size[0] + 1), img_size[1] // patch_size[1]]
        else:
            patches_resolution = [(img_size[0] // patch_size[0]), img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.mask_chans = mask_chans
        self.total_chans = in_chans + mask_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, y=None):
        B, _, _, _ = x.shape
        # y is the constant masks of land-sea mask and topography
        if y is not None:
            y = y.unsqueeze(0).repeat(B, 1, 1, 1)
            x = torch.cat((x, y), dim=1)

        #print(f"x.shape: {x.shape}")
        B, V, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # Zero-padding for the patch embedding: (145 + 3) // 4 = 37, if needed
        if H % self.patch_size[0] == 1:
            x = torch.nn.functional.pad(x, (0, 0, 0, 3), "constant", 0)

        x = x.view(B * V, 1, H, W)
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B Ph*Pw C
        patch_num, C = x.size(1), x.size(2)
        x = x.view(B, V, patch_num, C)

        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchRecovery(nn.Module):
    r""" Image to Patch Recovery

    Args:
        img_size (int): Image size. Default: 144(145) * 288 . Resolution: 1.25 degree * 1.25 degree
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96 * 2.
    """

    def __init__(self, img_size=[144, 288], patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        if img_size[0] % patch_size[0] == 1:
            patches_resolution = [(img_size[0] // patch_size[0] + 1), img_size[1] // patch_size[1]]
        else:
            patches_resolution = [(img_size[0] // patch_size[0]), img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, 3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim * 4, 3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(2),
            nn.Conv2d(embed_dim, in_chans, 3, padding=1, padding_mode='reflect')
        )
        
        nn.init.kaiming_normal_(self.proj[0].weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.zeros_(self.proj[-1].weight)


    def forward(self, x):
        B, V, P, C = x.shape
        Ph, Pw = self.patches_resolution
        x = x.transpose(2,3).contiguous().view(B * V, C, Ph, Pw)
        x = self.proj(x)
        x = x.view(B, V, self.img_size[0], self.img_size[1]).contiguous()

        # Crop x back to the original size, if needed
        if self.img_size[0] % self.patch_size[0] == 1:
            x = x[:,:,:-3,:]

        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        embed_dim = self.embed_dim
        in_chans = self.in_chans
        total_flops = 0
    
        for layer in self.proj:
            if isinstance(layer, nn.Conv2d):
                h_out = Ho if layer == self.proj[0] else Ho * 2
                w_out = Wo if layer == self.proj[0] else Wo * 2
            
                if layer == self.proj[0]:
                    total_flops += h_out * w_out * (embed_dim * 3 * 3 * embed_dim * 4)
                elif layer == self.proj[3]:
                    total_flops += h_out * w_out * (embed_dim * 3 * 3 * embed_dim * 4)
                else:
                    total_flops += h_out * w_out * (embed_dim * 3 * 3 * in_chans)
                
            elif isinstance(layer, nn.PixelShuffle):
                Ho *= 2
                Wo *= 2
    
        return total_flops


class VarAggregate(nn.Module):
    r""" Variable aggregation

    Args:
        img_size (int): Image size. Default: 144(145) * 288 . Resolution: 1.25 degree * 1.25 degree
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input variables. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=[144, 288], patch_size=4, embed_dim=96, variable=26, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        if img_size[0] % patch_size[0] == 1:
            patches_resolution = [(img_size[0] // patch_size[0] + 1), img_size[1] // patch_size[1]]
        else:
            patches_resolution = [(img_size[0] // patch_size[0]), img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.embed_dim = embed_dim
        self.variable = variable

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.atten_query = nn.Parameter(torch.zeros(embed_dim))
        trunc_normal_(self.atten_query, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # B, V, P, C = x.shape
        x = x.transpose(1, 2).contiguous()
        q = self.atten_query

        attn = torch.matmul(x, q)
        attn = self.softmax(attn)

        x = torch.sum(attn.unsqueeze(3) * x, dim=2)

        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = self.num_patches * self.variable * self.embed_dim * 4

        if self.norm is not None:
            flops += self.num_patches * self.embed_dim
        return flops


class VarPredict(nn.Module):
    r""" Variable prediction

    Args:
        img_size (int): Image size. Default: 144(145) * 288 . Resolution: 1.25 degree * 1.25 degree
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input variables. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=[144, 288], patch_size=4, embed_dim=96, variable=1, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        if img_size[0] % patch_size[0] == 1:
            patches_resolution = [(img_size[0] // patch_size[0] + 1), img_size[1] // patch_size[1]]
        else:
            patches_resolution = [(img_size[0] // patch_size[0]), img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.embed_dim = embed_dim
        self.variable = variable

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * variable)
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, y=None):
        B, P, C = x.shape
        x = x.view(B * P, C)
        x = self.mlp(x)
        x = x.view(B, P, self.variable, C).permute(0, 2, 1, 3).contiguous()

        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = self.num_patches * self.embed_dim * self.embed_dim
        flops += self.num_patches * self.embed_dim
        flops += self.num_patches * self.embed_dim * (self.embed_dim * self.variable)

        if self.norm is not None:
            flops += self.num_patches * self.variable * self.embed_dim
        return flops


class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 144(145) * 288 . Resolution: 1.25 degree * 1.25 degree
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input variables. Default: 1
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 6 * 12
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=[144, 288], patch_size=4, in_chans=1, in_vars=1, out_chans=1, out_vars=1,
                 embed_dim=[96, 192, 192, 96], depths=[2, 6, 6, 2], num_heads=[6, 12, 12, 6],
                 window_size=[6,12], mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, mask_chans=1, embed_dim=embed_dim[0],
            norm_layer=norm_layer if self.patch_norm else None)
        
        # recover the non-overlapping patches back to image
        self.patch_recovery = PatchRecovery(
            img_size=img_size, patch_size=patch_size, in_chans=out_chans, embed_dim=embed_dim[0])

        # aggregate the multiple variables into single variable
        self.var_aggregate = VarAggregate(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim[0], variable=in_vars)
        
        # predict the multiple variables from single variable
        self.var_predict = VarPredict(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim[0], variable=out_vars)

        # produce the patch resolutions for each layer
        divisors = [patch_size, patch_size*2, patch_size*2, patch_size]
        if img_size[0] % 2 == 1:
            patches_resolution = [[(size // divisor) + 1 if index == 0 else size // divisor for index, size in enumerate(img_size)] for divisor in divisors]
        else:
            patches_resolution = [[size // divisor for size in img_size] for divisor in divisors]

        self.patches_resolution = patches_resolution

        self.skip_conv2d = nn.Conv2d(
                in_channels=embed_dim[0] * 2,
                out_channels=embed_dim[0],
                kernel_size=3,
                padding=1
            )

        # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim[i_layer],
                               input_resolution=patches_resolution[i_layer],
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer == 0 ) else None,
                               upsample=PatchExpanding if (i_layer == 2 ) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x, y=None):
        if y is not None:
            x = self.patch_embed(x, y)
        else:
            x = self.patch_embed(x)

        x = self.var_aggregate(x)
        x = self.pos_drop(x)

        use_skip = False
        if use_skip:

            x_layer0, skip0 = self.layers[0](x)
            x = x_layer0

            x_layer1, _ = self.layers[1](x)
            x = x_layer1

            x_layer2, _ = self.layers[2](x)
            x = x_layer2

            x_concat = torch.cat([x, skip0], dim=-1)
            H, W = self.patches_resolution[0]
            B, N, C_concat = x_concat.shape
            x_spatial = x_concat.view(B, H, W, C_concat)
            x_spatial_reshaped = x_spatial.permute(0, 3, 1, 2)

            x_compressed = self.skip_conv2d(x_spatial_reshaped)  # 输出: (B, C, H, W)
            x = x_compressed.permute(0, 2, 3, 1).view(B, H*W, -1)

            x_layer3, _ = self.layers[3](x)
            x = x_layer3
        
        else:
            for layer in self.layers:
                x, _ = layer(x)

        x = self.var_predict(x)
        x = self.patch_recovery(x)
        return x

    def forward_embedding(self, x): # for special usage

        x = self.var_aggregate(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.var_predict(x)
        x = self.patch_recovery(x)

        return x


    def forward(self, x, y=None):
        x = self.forward_features(x, y) # x = self.forward_features(x, y=None)

        return x


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.patch_recovery.flops()

        return flops
