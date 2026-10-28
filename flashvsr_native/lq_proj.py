# -*- coding: utf-8 -*-
"""LQ_proj_in modules — verbatim port of
FlashVSR/examples/WanVSR/utils/utils.py (Buffer_LQ4x_Proj / Causal_LQ4x_Proj).

The two variants differ ONLY in cache semantics of the causal convs:
- v1.0 "Buffer": assigns the CURRENT clip's tail as conv cache and consumes it
  in the same call (self-referential wrap; training-time behaviour).
- v1.1 "Causal": proper causal cache (previous clip's tail).

Both quirks must be preserved — "fixing" them breaks the output.
Pure torch; checkpoint keys match 1:1 (LQ_proj_in.ckpt / LQ_proj_in_v11.ckpt).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

CACHE_T = 2


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        bias = self.bias.to(x) if isinstance(self.bias, torch.Tensor) else self.bias
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x) + bias


class CausalConv3d(nn.Conv3d):
    """Causal 3d convolution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding, mode='replicate')
        return super().forward(x)


class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x):
        # x: (B, C, F, H, W)
        return rearrange(x,
                         'b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w',
                         ff=self.ff, hh=self.hh, ww=self.ww)


class _LQ4x_ProjBase(nn.Module):
    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)
        self.conv1 = CausalConv3d(in_dim * self.ff * self.hh * self.ww,
                                  self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1),
                                  padding=(1, 1, 1))
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()
        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3),
                                  stride=(2, 1, 1), padding=(1, 1, 1))
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])
        self.clip_idx = 0

    def clear_cache(self):
        self.cache = {}
        self.cache['conv1'] = None
        self.cache['conv2'] = None
        self.clip_idx = 0


class Buffer_LQ4x_Proj(_LQ4x_ProjBase):
    """v1.0 variant — cache is written BEFORE the conv consumes it (same call)."""

    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            x = self.conv2(x, self.cache['conv2'])
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, 'b c f h w -> b (f h w) c')
            outputs = []
            for i in range(self.layer_num):
                outputs.append(self.linear_layers[i](out_x))
            self.clip_idx += 1
            return outputs


class Causal_LQ4x_Proj(_LQ4x_ProjBase):
    """v1.1 variant — proper causal cache (written AFTER the conv consumes
    the previous clip's tail)."""

    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv2(x, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, 'b c f h w -> b (f h w) c')
            outputs = []
            for i in range(self.layer_num):
                outputs.append(self.linear_layers[i](out_x))
            self.clip_idx += 1
            return outputs


def build_lq_proj(version: str, in_dim=3, out_dim=1536, layer_num=1, device="cuda"):
    cls = Buffer_LQ4x_Proj if str(version) == "1.0" else Causal_LQ4x_Proj
    return cls(in_dim=in_dim, out_dim=out_dim, layer_num=layer_num).to(
        device=device, dtype=torch.bfloat16)
