import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import kornia
from timm.layers import trunc_normal_
import json
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import transforms, functional


class PolarizationLoss(nn.Module):
    def __init__(self, lambda_s0=1.0, lambda_dolp=1.0, lambda_aop=1.0, eps=1e-8):
        super().__init__()
        self.lambda_s0 = lambda_s0
        self.lambda_dolp = lambda_dolp
        self.lambda_aop = lambda_aop
        self.eps = eps

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target_polarization):

        target_s0, target_dolp, target_aop = torch.chunk(target_polarization, 3, dim=1)


        pred_s0, pred_dolp, pred_aop = torch.chunk(pred, 3, dim=1)


        s0_loss = self.mse(pred_s0, target_s0) * self.lambda_s0
        dolp_loss = self.l1(pred_dolp, target_dolp) * self.lambda_dolp

        aop_diff = torch.abs(pred_aop - target_aop)
        aop_diff = torch.min(aop_diff, math.pi - aop_diff)  # 处理周期性
        aop_loss = aop_diff.mean() * self.lambda_aop

        total_loss = s0_loss + dolp_loss + aop_loss

        return total_loss


class DRLN(nn.Module):
    """Dynamic Rescale Layer Norm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False, use_meta=True,
                 meta_hidden_ratio=0.5, activation='leaky_relu'):
        super(DRLN, self).__init__()
        self.dim = dim
        self.eps = eps
        self.detach_grad = detach_grad
        self.use_meta = use_meta

        # Main transformation parameters - using more reasonable initialization
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

        if self.use_meta:
            # Dynamically compute hidden dimension
            hidden_dim = max(4, int(dim * meta_hidden_ratio))

            # More efficient meta network design
            self.meta_network = nn.Sequential(
                nn.Conv2d(2, hidden_dim, 1),  # Process both mean and std simultaneously
                self._get_activation(activation),
                nn.Conv2d(hidden_dim, dim * 2, 1)  # Output both rescale and rebias
            )

            # More refined initialization
            self._init_meta_weights()

        # Optional skip connection
        self.skip_connection = nn.Identity()

    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif name == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)

    def _init_meta_weights(self):
        # More reasonable initialization strategy
        for m in self.meta_network.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 2:  # Input layer
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                else:  # Output layer
                    nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Set specific initialization for output layer
        with torch.no_grad():
            # Initialize rescale channels to 1, rebias channels to 0
            self.meta_network[-1].weight[::2] = 1.0  # rescale channels
            self.meta_network[-1].weight[1::2] = 0.0  # rebias channels

    def forward(self, x):
        # Compute statistics - using more numerically stable method
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)

        # Normalization
        normalized_x = (x - mean) / std

        if self.use_meta:
            # Prepare input for meta network
            if self.detach_grad:
                stats_input = torch.cat([std.detach(), mean.detach()], dim=1)
            else:
                stats_input = torch.cat([std, mean], dim=1)

            # Pass through meta network
            meta_output = self.meta_network(stats_input)
            rescale, rebias = torch.chunk(meta_output, 2, dim=1)
        else:
            rescale, rebias = 1.0, 0.0

        # Apply transformation
        out = normalized_x * self.weight + self.bias
        out = out * (1 + rescale) + rebias  # Residual-style modification

        return out, rescale, rebias

class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
            fan_out = m.weight.shape[0] * m.weight.shape[2] * m.weight.shape[3]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())
    return relative_positions_log


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x


class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        # 卷积分支
        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )
        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        # 注意力分支
        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)
        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.cross_window_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect')
        self.cross_norm = nn.BatchNorm2d(dim)
        self.cross_act = nn.ReLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
            fan_out = m.weight.shape[0] * m.weight.shape[2] * m.weight.shape[3]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(
                x,
                (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                 self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                mode='reflect'
            )
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(X)

        if self.use_attn:
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)
            attn_windows = self.attn(qkv)
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            cross_out = self.cross_window_conv(attn_out)
            cross_out = self.cross_norm(cross_out)
            cross_out = self.cross_act(cross_out)
            attn_out = attn_out + cross_out  # 残差融合

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                out = self.proj(attn_out)
        else:
            if self.conv_type == 'Conv':
                out = self.conv(X)
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))
            else:
                out = X
        return out

class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        identity = x
        if self.use_attn: x, rescale, rebias = self.norm1(x)
        x = self.attn(x)
        if self.use_attn: x = x * rescale + rebias
        x = identity + x

        identity = x
        if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm: x = x * rescale + rebias
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=4, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=4, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class  PolarFusion(nn.Module):
    """
    PolarFusion: Multi-branch feature fusion with channel attention and spatial gating
    """

    def __init__(
            self,
            dim,
            height=4,
            reduction=8,
            ms_dilations=(1, 2, 3),
            use_spatial=True,
            residual=True,
            init_tau=0.5
    ):
        super().__init__()
        assert height >= 2
        self.height = height
        self.dim = dim
        self.use_spatial = use_spatial
        self.residual = residual

        d = max(dim // reduction, 4)

        # --- Branch-wise Channel Attention (shared weights) ---
        # For each branch: GAP -> MLP to get channel descriptor e_i ∈ R^{C}
        self.branch_mlp = nn.Sequential(
            nn.Conv2d(dim, d, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, dim, kernel_size=1, bias=False)
        )
        # Global context (sum of all branches) processed by MLP to get e_ctx ∈ R^{C}
        self.ctx_mlp = nn.Sequential(
            nn.Conv2d(dim, d, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, dim, kernel_size=1, bias=False)
        )
        # Concatenate [e_i, e_ctx] (2C), use 1x1 conv to get channel logits for each branch ∈ R^{C}
        self.logit_head = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False)

        # Learnable temperature parameter tau (controls softmax sharpness across branches)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau, dtype=torch.float32)))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)  # Softmax over branch dimension

        # --- Lightweight Multi-scale Spatial Gating ---
        if use_spatial:
            self.dw_blocks = nn.ModuleList([
                nn.Conv2d(dim, dim, kernel_size=3, padding=dil, dilation=dil, groups=dim, bias=False)
                for dil in ms_dilations
            ])
            # Fuse multi-scale features + compress to 1-channel spatial gate
            self.spatial_fuse = nn.Sequential(
                nn.Conv2d(dim * len(ms_dilations), dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

    def forward(self, in_feats):
        """
        Args:
            in_feats: list[Tensor], length M, each Tensor shape [B, C, H, W], C=dim
        Returns:
            Tensor [B, C, H, W]
        """
        assert isinstance(in_feats, (list, tuple)) and len(in_feats) == self.height
        B, C, H, W = in_feats[0].shape
        # [B, M, C, H, W]
        feats = torch.stack(in_feats, dim=1)

        # --- Global Context (sum over branches) ---
        sum_feats = feats.sum(dim=1)  # [B, C, H, W]
        ctx = self.avg_pool(sum_feats)  # [B, C, 1, 1]
        e_ctx = self.ctx_mlp(ctx)  # [B, C, 1, 1]

        # --- Per-branch Channel Descriptors + Context Fusion for Logits ---
        # Compute e_i for each branch
        e_list = []
        for i in range(self.height):
            gi = self.avg_pool(in_feats[i])  # [B, C, 1, 1]
            ei = self.branch_mlp(gi)  # [B, C, 1, 1]
            # Concatenate [e_i, e_ctx] -> 2C
            logits_i = self.logit_head(torch.cat([ei, e_ctx], dim=1))  # [B, C, 1, 1]
            e_list.append(logits_i)

        # [B, M, C, 1, 1]
        logits = torch.stack(e_list, dim=1)

        # Temperature scaling (smaller tau => more confident selection; learnable)
        tau = torch.clamp(self.log_tau.exp(), min=1e-3, max=10.0)
        attn_branch_channel = self.softmax(logits / tau)  # Softmax over branch dimension

        # --- Spatial Gating ---
        if self.use_spatial:
            # Use sum_feats for lightweight multi-scale feature extraction
            ms = [dw(sum_feats) for dw in self.dw_blocks]  # List, each [B, C, H, W]
            ms = torch.cat(ms, dim=1)  # [B, C*len(dils), H, W]
            spatial_gate = self.spatial_fuse(ms)  # [B, 1, H, W], in [0,1]
        else:
            spatial_gate = None

        # --- Fusion ---
        fused = (feats * attn_branch_channel).sum(dim=1)  # [B, C, H, W]

        if spatial_gate is not None:
            fused = fused * spatial_gate  # Spatial re-gating

        if self.residual:
            # Light residual connection to stabilize training
            # (equivalent to creating a bypass with mean or sum)
            fused = fused + 0.0 * sum_feats  # This doesn't change values, just maintains shape alignment for residual path
            # Can be modified to + alpha*sum_feats

        return fused


class AdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion = PolarFusion(dim, height=2)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, main_branch, skip_branch):
        fused = self.fusion([main_branch, skip_branch])
        output = self.alpha * fused + (1 - self.alpha) * main_branch
        return output

class PDFormer(nn.Module):
    def __init__(self, in_chans=4, out_chans=3, window_size=8,
                 embed_dims=[16, 32, 64, 32, 16],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[4, 4, 4, 2, 2],
                 num_heads=[2, 4, 4, 1, 1],
                 attn_ratio=[0, 0.5, 0.75, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[DRLN, DRLN, DRLN, DRLN, DRLN]):
        super(PDFormer, self).__init__()

        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios

        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = AdaptiveFusion(embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = AdaptiveFusion(embed_dims[4])

        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                 norm_layer=norm_layer[4], window_size=window_size,
                                 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])


        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)


        self.s0_activation = nn.Sigmoid()
        self.dolp_activation = nn.Sigmoid()
        self.aop_activation = nn.Tanh()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)

        x = self.fusion1(x, self.skip2(skip2))
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2(x, self.skip1(skip1))
        x = self.layer5(x)
        x = self.patch_unembed(x)

        s0 = self.s0_activation(x[:, 0:1])
        dolp = self.dolp_activation(x[:, 1:2])
        aop = self.aop_activation(x[:, 2:3]) * 0.5 * math.pi

        aop = aop % math.pi

        return torch.cat([s0, dolp, aop], dim=1)


class PolarizationDenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, mode='train', image_size=1024, crop_size=256):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.mode = mode
        self.image_size = image_size
        self.crop_size = crop_size
        self.stride = 0.5*crop_size  # 新增步长参数
        self.samples = []
        self.eps = 1e-8

        # 获取所有文件夹（001-108）
        folders = sorted(os.listdir(clean_dir))
        folders = [f for f in folders if os.path.isdir(os.path.join(clean_dir, f))]

        # 过滤出001-108文件夹
        folders = [f for f in folders if f.isdigit() and 1 <= int(f) <= 150]

        # 收集所有样本
        for folder in folders:
            clean_folder = os.path.join(clean_dir, folder)
            noisy_folder = os.path.join(noisy_dir, folder)

            # 检查四个角度图像是否存在
            clean_files = [
                os.path.join(clean_folder, f"0.bmp"),
                os.path.join(clean_folder, f"45.bmp"),
                os.path.join(clean_folder, f"90.bmp"),
                os.path.join(clean_folder, f"135.bmp")
            ]

            noisy_files = [
                os.path.join(noisy_folder, f"0.bmp"),
                os.path.join(noisy_folder, f"45.bmp"),
                os.path.join(noisy_folder, f"90.bmp"),
                os.path.join(noisy_folder, f"135.bmp")
            ]

            # 检查所有文件是否存在
            if all(os.path.exists(f) for f in clean_files + noisy_files):
                self.samples.append((clean_files, noisy_files))

        print(f"找到 {len(self.samples)} 个文件夹的有效样本")

        # 数据增强参数
        self.do_crop = (mode in ['train', 'val', 'test'])

        # 计算每个图像可以产生的裁剪块数量（修改为基于步长）
        if self.do_crop:
            self.num_crops_per_image = self._calculate_num_crops()
        else:
            self.num_crops_per_image = 1

        self.to_tensor = transforms.ToTensor()

        # 中心裁剪目标尺寸
        self.center_crop_size = 1024

    def _calculate_num_crops(self):
        """计算每张图像基于步长可以产生的裁剪块数量"""
        if not self.do_crop:
            return 1

        # 计算在x和y方向上的裁剪块数量
        num_x = math.floor((self.image_size - self.crop_size) / self.stride) + 1
        num_y = math.floor((self.image_size - self.crop_size) / self.stride) + 1
        return num_x * num_y

    def __len__(self):
        return len(self.samples) * self.num_crops_per_image if self.do_crop else len(self.samples)

    def _get_crop_coordinates(self, crop_idx):
        """根据裁剪索引计算裁剪坐标"""
        num_x = math.floor((self.image_size - self.crop_size) / self.stride) + 1

        i = crop_idx // num_x
        j = crop_idx % num_x

        x = j * self.stride
        y = i * self.stride

        # 确保裁剪不会超出图像边界
        x = min(x, self.image_size - self.crop_size)
        y = min(y, self.image_size - self.crop_size)

        return x, y

    def compute_polarization_parameters(self, angles_img):
        """从四个角度图像计算S0, DoLP, AOP"""
        if isinstance(angles_img, list):
            # 如果是PIL图像列表，转换为张量
            angles_tensor = torch.cat([self.to_tensor(img) for img in angles_img], dim=0)
        else:
            angles_tensor = angles_img

        I0, I45, I90, I135 = torch.chunk(angles_tensor, 4, dim=0)

        # 计算斯托克斯参数
        S0 = (I0 + I90) * 0.5
        S1 = (I0 - I90) * 0.5
        S2 = (I45 - I135) * 0.5

        # 计算DoLP
        dolp = torch.sqrt(S1 ** 2 + S2 ** 2 + self.eps) / (S0 + self.eps)
        dolp = torch.clamp(dolp, 0.0, 1.0)

        # 计算AOP
        aop = 0.5 * torch.atan2(S2, S1 + self.eps)
        aop = aop % math.pi  # 归一化到[0, π]

        return torch.cat([S0, dolp, aop], dim=0)

    def __getitem__(self, idx):
        # 索引解析
        if self.do_crop:
            sample_idx = idx // self.num_crops_per_image
            crop_idx = idx % self.num_crops_per_image
        else:
            sample_idx = idx

        clean_files, noisy_files = self.samples[sample_idx]

        # 加载原始清晰图像
        clean_imgs = [Image.open(p).convert('L') for p in clean_files]
        # 加载噪声图像
        noisy_imgs = [Image.open(p).convert('L') for p in noisy_files]

        # 步骤1: 中心裁剪到1024×1024
        clean_imgs = [
            transforms.CenterCrop(self.center_crop_size)(img)
            for img in clean_imgs
        ]
        noisy_imgs = [
            transforms.CenterCrop(self.center_crop_size)(img)
            for img in noisy_imgs
        ]

        # 步骤2: 统一调整尺寸到目标大小
        clean_imgs = [
            img.resize((self.image_size, self.image_size), Image.BICUBIC)
            for img in clean_imgs
        ]
        noisy_imgs = [
            img.resize((self.image_size, self.image_size), Image.BICUBIC)
            for img in noisy_imgs
        ]

        # 裁剪处理（基于步长的重叠裁剪）
        if self.do_crop:
            x, y = self._get_crop_coordinates(crop_idx)

            clean_imgs = [
                img.crop((x, y, x + self.crop_size, y + self.crop_size))
                for img in clean_imgs
            ]
            noisy_imgs = [
                img.crop((x, y, x + self.crop_size, y + self.crop_size))
                for img in noisy_imgs
            ]

        # 数据增强（仅训练模式）
        if self.mode == 'train':
            # 随机水平翻转
            if random.random() > 0.5:
                clean_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in clean_imgs]
                noisy_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in noisy_imgs]
            # 随机垂直翻转
            if random.random() > 0.5:
                clean_imgs = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in clean_imgs]
                noisy_imgs = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in noisy_imgs]
            # 随机旋转 (0°, 90°, 180°, 270°)
            if random.random() > 0.75:
                rotation = random.choice([0, 90, 180, 270])
                clean_imgs = [img.rotate(rotation, expand=False) for img in clean_imgs]
                noisy_imgs = [img.rotate(rotation, expand=False) for img in noisy_imgs]
            # 随机亮度改变
            if random.random() > 0.8:
                factor = random.uniform(0.9, 1.1)
                clean_imgs = [functional.adjust_contrast(img, factor) for img in clean_imgs]
                noisy_imgs = [functional.adjust_contrast(img, factor) for img in noisy_imgs]

        # 转换为张量 (4, H, W)
        noisy_tensor = torch.cat([self.to_tensor(img) for img in noisy_imgs], dim=0)

        # 对清晰图像计算偏振参数 (3, H, W)
        clean_polarization = self.compute_polarization_parameters(clean_imgs)

        return noisy_tensor, clean_polarization

class DenoiseModel:
    def __init__(self, clean_root, noisy_root):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PDFormer().to(self.device)
        self.pretrained_path = "best_denoise_model.pth"

        if self.pretrained_path and os.path.exists(self.pretrained_path):
            self.load_pretrained(self.pretrained_path)
            print(f"已加载预训练模型: {self.pretrained_path}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = PolarizationLoss(
            lambda_s0=1.0,
            lambda_dolp=0.1,
            lambda_aop=0.3
        ).to(self.device)

        self.train_dataset = PolarizationDenoisingDataset(
            clean_dir=os.path.join(clean_root, "train"),
            noisy_dir=os.path.join(noisy_root, "train"),
            mode='train',
            image_size=1024,
            crop_size=256
        )

        self.val_dataset = PolarizationDenoisingDataset(
            clean_dir=os.path.join(clean_root, "val"),
            noisy_dir=os.path.join(noisy_root, "val"),
            mode='val',
            image_size=1024,
            crop_size=256
        )

        print(f"训练集大小: {len(self.train_dataset)}, 验证集大小: {len(self.val_dataset)}")

        self.scheduler = None
        self.lr_decay_interval = 10  # 每10轮验证loss不降，学习率减半
        self.patience_counter = 0  # 记录验证loss未下降的轮数
        self.best_val_loss = float('inf')  # 记录最佳验证loss

        self.train_loss_history = []
        self.val_loss_history = []
        self.epochs_history = []
        self.learning_rates = []  # 记录学习率变化

    def load_pretrained(self, model_path):
        """加载预训练模型权重"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            if all(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)
            print("预训练模型加载成功")
        except Exception as e:
            print(f"加载预训练模型时出错: {e}")
            print("将从头开始训练")

    def train(self, epochs, resume_epoch=0):
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=2, eta_min=1e-6)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=1,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=1,
        )

        if resume_epoch == 0:
            with open("training_loss_log.txt", "w") as f:
                f.write("Epoch\tTrain_Loss\tVal_Loss\tLearning_Rate\n")
                f.write("===========================================\n")

        best_loss = float('inf')
        early_stop_counter = 0
        print("训练开始！")
        print(f"使用自适应学习率调度器，验证loss连续{self.lr_decay_interval}轮不降则学习率减半")

        for epoch in range(resume_epoch, resume_epoch + epochs):
            self.model.train()
            train_loss = 0.0
            for noisy_inputs, clean_polarization in tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{resume_epoch + epochs} - Training",
                    ncols=120):
                noisy_inputs = noisy_inputs.to(self.device)
                clean_polarization = clean_polarization.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(noisy_inputs)
                loss = self.criterion(outputs, clean_polarization)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for noisy_inputs, clean_polarization in tqdm(
                        val_loader,
                        desc=f"Epoch {epoch + 1}/{resume_epoch + epochs} - Validation",
                        ncols=120):
                    noisy_inputs = noisy_inputs.to(self.device)  # 修正：使用 noisy_inputs
                    clean_polarization = clean_polarization.to(self.device)
                    outputs = self.model(noisy_inputs)
                    val_loss += self.criterion(outputs, clean_polarization).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_val_loss)
            self.epochs_history.append(epoch + 1)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            with open("training_loss_log.txt", "a") as f:
                f.write(f"{epoch + 1}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\t{current_lr:.2e}\n")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stop_counter = 0
                torch.save(self.model.state_dict(), "best_denoise_model.pth")
                print(f"保存新的最佳模型 (验证损失: {avg_val_loss:.6f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= 50:
                    print(f"早停机制触发于 epoch {epoch}")
                    break

            print(f"Epoch {epoch + 1}/{resume_epoch + epochs} | "
                  f"训练损失: {avg_train_loss:.6f} | "
                  f"验证损失: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

        self.plot_training_curves()

    def plot_training_curves(self):
        """Plot training and validation loss curves along with learning rate curve"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot loss curves
        ax1.plot(self.epochs_history, self.train_loss_history, label='Training Loss', marker='o', markersize=3)
        ax1.plot(self.epochs_history, self.val_loss_history, label='Validation Loss', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss Curves')
        ax1.legend()
        ax1.grid(True)
        ax1.set_yscale('log')  # Use log scale to better show loss changes

        # Plot learning rate curve
        ax2.plot(self.epochs_history, self.learning_rates, label='Learning Rate', color='red', marker='^', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')  # Use log scale to better show learning rate changes

        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_checkpoint(self, path, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'epochs_history': self.epochs_history,
            'learning_rates': self.learning_rates
        }
        torch.save(checkpoint, path)

        with open(f"checkpoint_epoch_{epoch}_loss_log.txt", "w") as f:
            f.write("Epoch\tTrain_Loss\tVal_Loss\tLearning_Rate\n")
            f.write("===========================================\n")
            for i, (train_loss, val_loss, lr) in enumerate(zip(
                    self.train_loss_history,
                    self.val_loss_history,
                    self.learning_rates)):
                f.write(f"{i + 1}\t{train_loss:.6f}\t{val_loss:.6f}\t{lr:.2e}\n")

        print(f"检查点已保存: {path}")

    def load_checkpoint(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if checkpoint['scheduler_state_dict'] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.train_loss_history = checkpoint['train_loss_history']
            self.val_loss_history = checkpoint['val_loss_history']
            self.epochs_history = checkpoint['epochs_history']
            self.learning_rates = checkpoint.get('learning_rates', [])

            print(f"检查点加载成功，将从 epoch {checkpoint['epoch'] + 1} 继续训练")
            return checkpoint['epoch']
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            return 0

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def test(self, clean_root, noisy_root):
        self.model.load_state_dict(torch.load("best_denoise_model.pth"))
        self.model.eval()

        test_dataset = PolarizationDenoisingDataset(
            clean_dir=os.path.join(clean_root, "test"),
            noisy_dir=os.path.join(noisy_root, "test"),
            mode='test',
            image_size=1024,
            crop_size=512
        )

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        os.makedirs("polarization_results", exist_ok=True)

        all_metrics = []

        print("开始偏振参数输出测试...")
        with torch.no_grad():
            for idx, (noisy_input, clean_polarization) in enumerate(test_loader):
                noisy_input = noisy_input.to(self.device)
                clean_polarization = clean_polarization.to(self.device)

                polarization_output = self.model(noisy_input)

                clean_s0, clean_dolp, clean_aop = torch.chunk(clean_polarization, 3, dim=1)

                pred_s0, pred_dolp, pred_aop = torch.chunk(polarization_output, 3, dim=1)

                sample_dir = f"polarization_results/sample_{idx:04d}"
                os.makedirs(sample_dir, exist_ok=True)

                self.save_polarization_images(pred_s0, pred_dolp, pred_aop,
                                              clean_s0, clean_dolp, clean_aop,
                                              sample_dir, idx)

                metrics = self.calculate_polarization_metrics(pred_s0, pred_dolp, pred_aop,
                                                              clean_s0, clean_dolp, clean_aop)
                metrics['sample_index'] = idx

                all_metrics.append(metrics)

                print(f"样本 {idx:04d} 处理完成 - "
                      f"S0 PSNR: {metrics['s0_psnr']:.2f}dB, "
                      f"DoLP PSNR: {metrics['dolp_psnr']:.2f}dB, "
                      f"AOP MAE: {metrics['aop_mae']:.4f}rad")

        self.save_final_polarization_metrics(all_metrics)
        return all_metrics

    def save_polarization_images(self, pred_s0, pred_dolp, pred_aop,
                                 clean_s0, clean_dolp, clean_aop, sample_dir, idx):

        self.save_tensor_image(pred_s0, f"pred_S0", sample_dir)
        self.save_tensor_image(pred_dolp, f"pred_DoLP", sample_dir, is_dolp=True)
        self.save_tensor_image(pred_aop, f"pred_AoP", sample_dir, is_aop=True)

        self.save_tensor_image(clean_s0, f"clean_S0", sample_dir)
        self.save_tensor_image(clean_dolp, f"clean_DoLP", sample_dir, is_dolp=True)
        self.save_tensor_image(clean_aop, f"clean_AoP", sample_dir, is_aop=True)

        self.create_comparison_figures(pred_s0, pred_dolp, pred_aop,
                                       clean_s0, clean_dolp, clean_aop,
                                       sample_dir, idx)

        metrics = self.calculate_polarization_metrics(pred_s0, pred_dolp, pred_aop,
                                                      clean_s0, clean_dolp, clean_aop)

        with open(f"{sample_dir}/polarization_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

    def save_tensor_image(self, tensor, prefix, path, is_aop=False, is_dolp=False):
        if torch.is_tensor(tensor):
            tensor_np = tensor.squeeze().cpu().numpy()
        else:
            tensor_np = np.array(tensor).squeeze()

        if is_aop:
            aop_colored = self.apply_color_to_aop(tensor_np, saturation=1.0, value=1.0)
            Image.fromarray(aop_colored).save(f"{path}/{prefix}.png")
        elif is_dolp:
            tensor_np = np.clip(tensor_np, 0, 1)
            dolp_colored = plt.cm.hot(tensor_np)
            dolp_img = (dolp_colored[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(dolp_img).save(f"{path}/{prefix}.png")
        else:
            tensor_np = np.clip(tensor_np, 0, 1)
            img_array = (tensor_np * 255).astype(np.uint8)
            Image.fromarray(img_array).save(f"{path}/{prefix}.png")

    def apply_color_to_aop(self, aop: np.ndarray, saturation: float = 1.0, value: float = 1.0) -> np.ndarray:
        aop = np.clip(aop, 0, math.pi)


        h = (aop / math.pi * 179).astype(np.uint8)
        s = np.full_like(aop, saturation * 255, dtype=np.uint8)
        v = np.full_like(aop, value * 255, dtype=np.uint8)

        hsv = np.stack([h, s, v], axis=-1)

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def calculate_polarization_metrics(self, pred_s0, pred_dolp, pred_aop, clean_s0, clean_dolp, clean_aop):
        metrics = {}

        metrics['s0_psnr'] = kornia.metrics.psnr(pred_s0, clean_s0, max_val=1.0).item()
        metrics['s0_ssim'] = kornia.metrics.ssim(pred_s0, clean_s0, window_size=11, max_val=1.0).mean().item()

        metrics['dolp_psnr'] = kornia.metrics.psnr(pred_dolp, clean_dolp, max_val=1.0).item()
        metrics['dolp_ssim'] = kornia.metrics.ssim(pred_dolp, clean_dolp, window_size=11, max_val=1.0).mean().item()

        aop_diff = torch.abs(pred_aop - clean_aop)
        aop_diff = torch.minimum(aop_diff, torch.tensor(math.pi).to(aop_diff.device) - aop_diff)  # 处理周期性
        metrics['aop_mae'] = torch.mean(aop_diff).item()

        return metrics

    def create_comparison_figures(self, pred_s0, pred_dolp, pred_aop,
                                  clean_s0, clean_dolp, clean_aop,
                                  sample_dir, idx):
        def to_numpy(tensor):
            return tensor.squeeze().cpu().numpy()

        pred_s0_np = to_numpy(pred_s0)
        pred_dolp_np = to_numpy(pred_dolp)
        pred_aop_np = to_numpy(pred_aop)

        clean_s0_np = to_numpy(clean_s0)
        clean_dolp_np = to_numpy(clean_dolp)
        clean_aop_np = to_numpy(clean_aop)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(pred_s0_np, cmap='gray')
        axes[0].set_title('Denoised S0')
        axes[0].axis('off')

        axes[1].imshow(clean_s0_np, cmap='gray')
        axes[1].set_title('Clean S0')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{sample_dir}/S0_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        im0 = axes[0].imshow(pred_dolp_np, cmap='hot', vmin=0, vmax=1)
        axes[0].set_title('Denoised DoLP')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(clean_dolp_np, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Clean DoLP')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        plt.savefig(f"{sample_dir}/DoLP_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        pred_aop_color = self.apply_color_to_aop(pred_aop_np)
        clean_aop_color = self.apply_color_to_aop(clean_aop_np)

        axes[0].imshow(pred_aop_color)
        axes[0].set_title('Denoised AoP')
        axes[0].axis('off')

        axes[1].imshow(clean_aop_color)
        axes[1].set_title('Clean AoP')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{sample_dir}/AoP_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_final_polarization_metrics(self, all_metrics):
        avg_metrics = {}
        metric_categories = ['s0_psnr', 's0_ssim', 'dolp_psnr', 'dolp_ssim', 'aop_mae']

        for metric in metric_categories:
            values = [m[metric] for m in all_metrics]
            avg_metrics[metric] = np.mean(values)

        results_summary = {
            'test_configuration': {
                'total_samples': len(all_metrics),
                'model_output': 'S0, DoLP, AOP',
                'output_channels': 3
            },
            'per_sample_metrics': all_metrics,
            'average_metrics': avg_metrics
        }

        with open("polarization_results/polarization_test_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=4)

        print(f"\n=== 偏振参数输出测试结果 ===")
        print(f"总样本数: {len(all_metrics)}")
        print(f"S0 - 平均PSNR: {avg_metrics['s0_psnr']:.2f} dB, 平均SSIM: {avg_metrics['s0_ssim']:.4f}")
        print(f"DoLP - 平均PSNR: {avg_metrics['dolp_psnr']:.2f} dB, 平均SSIM: {avg_metrics['dolp_ssim']:.4f}")
        print(f"AOP - 平均MAE: {avg_metrics['aop_mae']:.4f} rad")

if __name__ == "__main__":
    clean_root = "your_data\\gt"
    noisy_root = "your_data\\noisy"

    trainer = DenoiseModel(clean_root, noisy_root)

    trainer.train(epochs=300)

    trainer.test(clean_root, noisy_root)