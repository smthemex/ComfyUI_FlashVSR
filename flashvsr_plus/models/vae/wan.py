# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flashvsr_plus.model_io import load_checkpoint_file, normalize_key
except ModuleNotFoundError:
    from ...model_io import load_checkpoint_file, normalize_key


CACHE_T = 2
WAN_LATENT_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
WAN_LATENT_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


class WanCausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pt, ph, pw = self.padding
        self._causal_padding = (pw, pw, ph, ph, 2 * pt, 0)
        self.padding = (0, 0, 0)

    def forward(
        self, x: torch.Tensor, cache: torch.Tensor | None = None
    ) -> torch.Tensor:
        padding = list(self._causal_padding)
        if cache is not None and padding[4] > 0:
            x = torch.cat([cache.to(device=x.device, dtype=x.dtype), x], dim=2)
            padding[4] -= cache.shape[2]
        return super().forward(F.pad(x, padding))


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, *, images: bool):
        super().__init__()
        broadcast = (1, 1) if images else (1, 1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, *broadcast))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma


class WanResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.residual = nn.Sequential(
            WanRMSNorm(in_dim, images=False),
            nn.SiLU(),
            WanCausalConv3d(in_dim, out_dim, 3, padding=1),
            WanRMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            WanCausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            WanCausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feature_index: list[int] | None = None,
    ) -> torch.Tensor:
        residual = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, WanCausalConv3d) and feature_cache is not None:
                if feature_index is None:
                    raise ValueError("feature_index is required with feature_cache")
                index = feature_index[0]
                cache = x[:, :, -CACHE_T:].clone()
                if cache.shape[2] < CACHE_T and feature_cache[index] is not None:
                    cache = torch.cat(
                        [feature_cache[index][:, :, -1:].to(cache), cache], dim=2
                    )
                x = layer(x, feature_cache[index])
                feature_cache[index] = cache
                feature_index[0] += 1
            else:
                x = layer(x)
        return x + residual


class WanAttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = WanRMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch, channels, frames, height, width = x.shape
        value = x.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        value = self.norm(value)
        qkv = self.to_qkv(value).reshape(
            batch * frames, 1, channels * 3, height * width
        )
        q, k, v = qkv.permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        value = F.scaled_dot_product_attention(q, k, v)
        value = value.squeeze(1).permute(0, 2, 1).reshape(
            batch * frames, channels, height, width
        )
        value = self.proj(value)
        value = value.reshape(batch, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        )
        return value + identity


class WanDownsample(nn.Module):
    def __init__(self, dim: int, *, temporal: bool):
        super().__init__()
        self.mode = "downsample3d" if temporal else "downsample2d"
        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(dim, dim, 3, stride=(2, 2)),
        )
        self.time_conv = (
            WanCausalConv3d(
                dim,
                dim,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
            )
            if temporal
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feature_index: list[int] | None = None,
    ) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        value = x.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        value = self.resample(value)
        height, width = value.shape[-2:]
        value = value.reshape(batch, frames, channels, height, width).permute(
            0, 2, 1, 3, 4
        )
        if self.time_conv is not None and feature_cache is not None:
            if feature_index is None:
                raise ValueError("feature_index is required with feature_cache")
            index = feature_index[0]
            if feature_cache[index] is None:
                feature_cache[index] = value.clone()
                feature_index[0] += 1
            else:
                cache = value[:, :, -1:].clone()
                value = self.time_conv(
                    torch.cat([feature_cache[index][:, :, -1:].to(value), value], 2)
                )
                feature_cache[index] = cache
                feature_index[0] += 1
        return value


class WanUpsample(nn.Module):
    def __init__(self, dim: int, *, temporal: bool):
        super().__init__()
        self.temporal = bool(temporal)
        self.resample = nn.Sequential(
            nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
            nn.Conv2d(dim, dim // 2, 3, padding=1),
        )
        self.time_conv = (
            WanCausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
            )
            if temporal
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | str | None],
        feature_index: list[int],
    ) -> torch.Tensor:
        batch, channels, frames = x.shape[:3]
        if self.time_conv is not None:
            index = feature_index[0]
            cached = feature_cache[index]
            if cached is None:
                feature_cache[index] = "first"
                feature_index[0] += 1
            else:
                next_cache = x[:, :, -CACHE_T:].clone()
                if next_cache.shape[2] < CACHE_T:
                    if isinstance(cached, torch.Tensor):
                        next_cache = torch.cat(
                            [cached[:, :, -1:].to(next_cache), next_cache], dim=2
                        )
                    else:
                        next_cache = torch.cat(
                            [torch.zeros_like(next_cache), next_cache], dim=2
                        )
                x = self.time_conv(
                    x, cached if isinstance(cached, torch.Tensor) else None
                )
                feature_cache[index] = next_cache
                feature_index[0] += 1
                x = x.reshape(batch, 2, channels, frames, *x.shape[-2:])
                x = torch.stack((x[:, 0], x[:, 1]), dim=3).reshape(
                    batch, channels, frames * 2, *x.shape[-2:]
                )
        frames = x.shape[2]
        value = x.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, x.shape[1], x.shape[3], x.shape[4]
        )
        dtype = value.dtype
        value = F.interpolate(
            value.float(), scale_factor=2.0, mode="nearest-exact"
        ).to(dtype=dtype)
        value = self.resample[1](value)
        return value.reshape(
            batch, frames, value.shape[1], value.shape[2], value.shape[3]
        ).permute(0, 2, 1, 3, 4)


class WanEncoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 32,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_scales: Sequence[float] = (),
        temporal_downsample: Sequence[bool] = (False, True, True),
        dropout: float = 0.0,
    ):
        super().__init__()
        dimensions = [dim * value for value in (1, *dim_mult)]
        scale = 1.0
        self.conv1 = WanCausalConv3d(3, dimensions[0], 3, padding=1)
        downsample_layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            for _ in range(num_res_blocks):
                downsample_layers.append(
                    WanResidualBlock(in_dim, out_dim, dropout)
                )
                if scale in attention_scales:
                    downsample_layers.append(WanAttentionBlock(out_dim))
                in_dim = out_dim
            if index != len(dim_mult) - 1:
                downsample_layers.append(
                    WanDownsample(
                        out_dim,
                        temporal=bool(temporal_downsample[index]),
                    )
                )
                scale /= 2
        self.downsamples = nn.Sequential(*downsample_layers)
        self.middle = nn.Sequential(
            WanResidualBlock(out_dim, out_dim, dropout),
            WanAttentionBlock(out_dim),
            WanResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            WanRMSNorm(out_dim, images=False),
            nn.SiLU(),
            WanCausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None],
        feature_index: list[int],
    ) -> torch.Tensor:
        index = feature_index[0]
        cache = x[:, :, -CACHE_T:].clone()
        if cache.shape[2] < CACHE_T and feature_cache[index] is not None:
            cache = torch.cat(
                [feature_cache[index][:, :, -1:].to(cache), cache], dim=2
            )
        x = self.conv1(x, feature_cache[index])
        feature_cache[index] = cache
        feature_index[0] += 1

        for layer in self.downsamples:
            if isinstance(layer, (WanResidualBlock, WanDownsample)):
                x = layer(x, feature_cache, feature_index)
            else:
                x = layer(x)
        for layer in self.middle:
            if isinstance(layer, WanResidualBlock):
                x = layer(x, feature_cache, feature_index)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, WanCausalConv3d):
                index = feature_index[0]
                cache = x[:, :, -CACHE_T:].clone()
                if cache.shape[2] < CACHE_T and feature_cache[index] is not None:
                    cache = torch.cat(
                        [feature_cache[index][:, :, -1:].to(cache), cache], dim=2
                    )
                x = layer(x, feature_cache[index])
                feature_cache[index] = cache
                feature_index[0] += 1
            else:
                x = layer(x)
        return x


class WanDecoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_scales: Sequence[float] = (),
        temporal_upsample: Sequence[bool] = (True, True, False),
        dropout: float = 0.0,
    ):
        super().__init__()
        dimensions = [dim * value for value in (dim_mult[-1], *dim_mult[::-1])]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        self.conv1 = WanCausalConv3d(z_dim, dimensions[0], 3, padding=1)
        self.middle = nn.Sequential(
            WanResidualBlock(dimensions[0], dimensions[0], dropout),
            WanAttentionBlock(dimensions[0]),
            WanResidualBlock(dimensions[0], dimensions[0], dropout),
        )
        upsample_layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            if index in {1, 2, 3}:
                in_dim //= 2
            for _ in range(num_res_blocks + 1):
                upsample_layers.append(
                    WanResidualBlock(in_dim, out_dim, dropout)
                )
                if scale in attention_scales:
                    upsample_layers.append(WanAttentionBlock(out_dim))
                in_dim = out_dim
            if index != len(dim_mult) - 1:
                upsample_layers.append(
                    WanUpsample(
                        out_dim,
                        temporal=bool(temporal_upsample[index]),
                    )
                )
                scale *= 2
        self.upsamples = nn.Sequential(*upsample_layers)
        self.head = nn.Sequential(
            WanRMSNorm(out_dim, images=False),
            nn.SiLU(),
            WanCausalConv3d(out_dim, 3, 3, padding=1),
        )

    @staticmethod
    def _run_causal(
        layer: WanCausalConv3d,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | str | None],
        feature_index: list[int],
    ) -> torch.Tensor:
        index = feature_index[0]
        cached = feature_cache[index]
        next_cache = x[:, :, -CACHE_T:].clone()
        if next_cache.shape[2] < CACHE_T and isinstance(cached, torch.Tensor):
            next_cache = torch.cat(
                [cached[:, :, -1:].to(next_cache), next_cache], dim=2
            )
        x = layer(x, cached if isinstance(cached, torch.Tensor) else None)
        feature_cache[index] = next_cache
        feature_index[0] += 1
        return x

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | str | None],
        feature_index: list[int],
    ) -> torch.Tensor:
        x = self._run_causal(self.conv1, x, feature_cache, feature_index)
        for layer in self.middle:
            if isinstance(layer, WanResidualBlock):
                x = layer(x, feature_cache, feature_index)
            else:
                x = layer(x)
        for layer in self.upsamples:
            if isinstance(layer, WanResidualBlock):
                x = layer(x, feature_cache, feature_index)
            elif isinstance(layer, WanUpsample):
                x = layer(x, feature_cache, feature_index)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, WanCausalConv3d):
                x = self._run_causal(
                    layer, x, feature_cache, feature_index
                )
            else:
                x = layer(x)
        return x


def _count_causal_convs(module: nn.Module) -> int:
    return sum(isinstance(layer, WanCausalConv3d) for layer in module.modules())


class WanVAEEncoder(nn.Module):
    """Wan 2.1 VAE encoder with the official temporal cache and scaling."""

    def __init__(self, dim: int = 96, z_dim: int = 16):
        super().__init__()
        self.z_dim = int(z_dim)
        self.encoder = WanEncoder3d(
            dim=dim,
            z_dim=z_dim * 2,
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            attention_scales=(),
            temporal_downsample=(False, True, True),
            dropout=0.0,
        )
        self.conv1 = WanCausalConv3d(z_dim * 2, z_dim * 2, 1)
        if z_dim != len(WAN_LATENT_MEAN):
            raise ValueError("Built-in Wan latent normalization requires z_dim=16")
        self.register_buffer(
            "latent_mean",
            torch.tensor(WAN_LATENT_MEAN),
            persistent=False,
        )
        self.register_buffer(
            "latent_inv_std",
            torch.tensor(WAN_LATENT_STD).reciprocal(),
            persistent=False,
        )

    def encode_video(self, video: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Wan VAE input must be [B,3,F,H,W], got {video.shape}")
        frames = video.shape[2]
        chunks = 1 + (frames - 1) // 4
        feature_cache: list[torch.Tensor | None] = [
            None
        ] * _count_causal_convs(self.encoder)
        outputs = []
        for index in range(chunks):
            feature_index = [0]
            if index == 0:
                current = video[:, :, :1]
            else:
                current = video[:, :, 1 + 4 * (index - 1) : 1 + 4 * index]
            if current.shape[2] == 0:
                raise ValueError("Wan VAE received an empty temporal chunk")
            outputs.append(
                self.encoder(current, feature_cache, feature_index)
            )
        value = torch.cat(outputs, dim=2)
        mean, _log_variance = self.conv1(value).chunk(2, dim=1)
        latent_mean = self.latent_mean.to(device=mean.device, dtype=mean.dtype)
        latent_inv_std = self.latent_inv_std.to(
            device=mean.device, dtype=mean.dtype
        )
        return (mean - latent_mean.view(1, -1, 1, 1, 1)) * latent_inv_std.view(
            1, -1, 1, 1, 1
        )

    def encode(self, video: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encode_video(video, **kwargs)


class WanVAE(WanVAEEncoder):
    """Wan encoder/decoder codec with differentiable streaming decode."""

    def __init__(self, dim: int = 96, z_dim: int = 16):
        super().__init__(dim=dim, z_dim=z_dim)
        self.conv2 = WanCausalConv3d(z_dim, z_dim, 1)
        self.decoder = WanDecoder3d(
            dim=dim,
            z_dim=z_dim,
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            attention_scales=(),
            temporal_upsample=(True, True, False),
            dropout=0.0,
        )
        self.clear_cache()

    def clear_cache(self) -> None:
        self._decoder_cache: list[torch.Tensor | str | None] = [
            None
        ] * _count_causal_convs(self.decoder)

    def _stream_decode_untiled(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[1] != self.z_dim:
            raise ValueError(
                f"Wan VAE latents must be [B,{self.z_dim},F,H,W], "
                f"got {latents.shape}"
            )
        mean = self.latent_mean.to(device=latents.device, dtype=latents.dtype)
        inv_std = self.latent_inv_std.to(
            device=latents.device, dtype=latents.dtype
        )
        value = latents / inv_std.view(1, -1, 1, 1, 1)
        value = value + mean.view(1, -1, 1, 1, 1)
        value = self.conv2(value)
        outputs = []
        for index in range(value.shape[2]):
            feature_index = [0]
            outputs.append(
                self.decoder(
                    value[:, :, index : index + 1],
                    self._decoder_cache,
                    feature_index,
                )
            )
        return torch.cat(outputs, dim=2)

    @staticmethod
    def _tile_starts(length: int, size: int, stride: int) -> list[int]:
        size = min(size, length)
        starts = list(range(0, max(length - size, 0) + 1, stride))
        last = length - size
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    @staticmethod
    def _tile_blend_mask(
        height: int,
        width: int,
        *,
        top_bound: bool,
        bottom_bound: bool,
        left_bound: bool,
        right_bound: bool,
        border_height: int,
        border_width: int,
    ) -> torch.Tensor:
        vertical = torch.ones(height, dtype=torch.float32)
        horizontal = torch.ones(width, dtype=torch.float32)
        if not top_bound and border_height > 0:
            count = min(border_height, height)
            vertical[:count] = torch.linspace(1 / count, 1, count)
        if not bottom_bound and border_height > 0:
            count = min(border_height, height)
            vertical[-count:] = torch.linspace(1, 1 / count, count)
        if not left_bound and border_width > 0:
            count = min(border_width, width)
            horizontal[:count] = torch.linspace(1 / count, 1, count)
        if not right_bound and border_width > 0:
            count = min(border_width, width)
            horizontal[-count:] = torch.linspace(1, 1 / count, count)
        return torch.minimum(vertical[:, None], horizontal[None, :]).view(
            1, 1, 1, height, width
        )

    def _tiled_stream_decode(
        self,
        latents: torch.Tensor,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
    ) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[1] != self.z_dim:
            raise ValueError(
                f"Wan VAE latents must be [B,{self.z_dim},F,H,W], got {latents.shape}"
            )
        tile_height, tile_width = (int(value) for value in tile_size)
        stride_height, stride_width = (int(value) for value in tile_stride)
        if min(tile_height, tile_width, stride_height, stride_width) <= 0:
            raise ValueError("VAE tile sizes and strides must be positive")
        if stride_height > tile_height or stride_width > tile_width:
            raise ValueError("VAE tile stride cannot exceed tile size")

        batch, _channels, latent_frames, latent_height, latent_width = latents.shape
        tile_height = min(tile_height, latent_height)
        tile_width = min(tile_width, latent_width)
        height_starts = self._tile_starts(latent_height, tile_height, stride_height)
        width_starts = self._tile_starts(latent_width, tile_width, stride_width)
        parameter = next(self.parameters())
        computation_device = parameter.device
        computation_dtype = parameter.dtype
        source = latents.detach().to("cpu")
        values = weight = None

        for top in height_starts:
            for left in width_starts:
                bottom = top + tile_height
                right = left + tile_width
                current = source[:, :, :, top:bottom, left:right].to(
                    device=computation_device, dtype=computation_dtype
                )
                self.clear_cache()
                decoded_gpu = self._stream_decode_untiled(current).detach()
                self.clear_cache()
                decoded = decoded_gpu.float().cpu()
                del decoded_gpu
                del current
                if values is None:
                    scale_height = decoded.shape[-2] // tile_height
                    scale_width = decoded.shape[-1] // tile_width
                    if scale_height <= 0 or scale_width <= 0:
                        raise RuntimeError("VAE tiled decode produced an invalid spatial scale")
                    output_frames = decoded.shape[2]
                    values = torch.zeros(
                        batch,
                        decoded.shape[1],
                        output_frames,
                        latent_height * scale_height,
                        latent_width * scale_width,
                        dtype=torch.float32,
                    )
                    weight = torch.zeros(
                        1, 1, 1, values.shape[-2], values.shape[-1], dtype=torch.float32
                    )
                    blend_height = max(
                        0, (tile_height - stride_height) * scale_height
                    )
                    blend_width = max(
                        0, (tile_width - stride_width) * scale_width
                    )
                target_top = top * scale_height
                target_left = left * scale_width
                mask = self._tile_blend_mask(
                    decoded.shape[-2],
                    decoded.shape[-1],
                    top_bound=top == 0,
                    bottom_bound=bottom >= latent_height,
                    left_bound=left == 0,
                    right_bound=right >= latent_width,
                    border_height=blend_height,
                    border_width=blend_width,
                )
                target_bottom = target_top + decoded.shape[-2]
                target_right = target_left + decoded.shape[-1]
                values[..., target_top:target_bottom, target_left:target_right] += (
                    decoded * mask
                )
                weight[..., target_top:target_bottom, target_left:target_right] += mask
                del decoded, mask

        self.clear_cache()
        if values is None or weight is None:
            raise RuntimeError("VAE tiled decode produced no tiles")
        return (values / weight.clamp_min(1e-8)).to(dtype=latents.dtype)

    def stream_decode(
        self,
        latents: torch.Tensor,
        *,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if tiled:
            return self._tiled_stream_decode(latents, tile_size, tile_stride)
        return self._stream_decode_untiled(latents)

    def decode_video(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        self.clear_cache()
        return self.stream_decode(latents, **kwargs)

    def decode(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.decode_video(latents, **kwargs)


def load_wan_vae_encoder_checkpoint(
    model: WanVAEEncoder,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
) -> WanVAEEncoder:
    source = load_checkpoint_file(checkpoint)
    own = model.state_dict()
    accepted = {}
    for source_key, value in source.items():
        key = normalize_key(source_key)
        for prefix in ("vae.", "first_stage_model.", "autoencoder."):
            if key.startswith(prefix):
                key = normalize_key(key[len(prefix) :])
                break
        if key in own and own[key].shape == value.shape:
            accepted[key] = value
    result = model.load_state_dict(accepted, strict=False)
    missing_parameters = [
        key
        for key in result.missing_keys
        if key not in {"latent_mean", "latent_inv_std"}
    ]
    if missing_parameters:
        raise RuntimeError(
            "Wan VAE checkpoint is missing encoder parameters: "
            f"{missing_parameters[:8]}"
        )
    return model


def build_wan_vae_encoder(checkpoint_path: str | Path) -> WanVAEEncoder:
    return load_wan_vae_encoder_checkpoint(WanVAEEncoder(), checkpoint_path)


def load_wan_vae_checkpoint(
    model: WanVAE,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
) -> WanVAE:
    source = load_checkpoint_file(checkpoint)
    own = model.state_dict()
    accepted = {}
    for source_key, value in source.items():
        key = normalize_key(source_key)
        for prefix in ("vae.", "model.", "first_stage_model.", "autoencoder."):
            if key.startswith(prefix):
                key = normalize_key(key[len(prefix) :])
                break
        if key in own and own[key].shape == value.shape:
            accepted[key] = value
    result = model.load_state_dict(accepted, strict=False)
    missing = [
        key
        for key in result.missing_keys
        if key not in {"latent_mean", "latent_inv_std"}
    ]
    if missing:
        raise RuntimeError(
            "Wan VAE checkpoint is missing codec parameters: "
            f"{missing[:8]}"
        )
    return model


def build_wan_vae(checkpoint_path: str | Path) -> WanVAE:
    return load_wan_vae_checkpoint(WanVAE(), checkpoint_path)
