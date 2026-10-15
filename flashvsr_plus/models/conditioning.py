from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LQProjectionConfig


class CausalConv3d(nn.Conv3d):
    """Causal convolution matching the cache behavior used by FlashVSR 1.1."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pt, ph, pw = self.padding
        self._causal_padding = (pw, pw, ph, ph, 2 * pt, 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None) -> torch.Tensor:
        padding = list(self._causal_padding)
        if cache is not None and padding[4] > 0:
            x = torch.cat([cache.to(device=x.device, dtype=x.dtype), x], dim=2)
            padding[4] = max(0, padding[4] - cache.shape[2])
        x = F.pad(x, padding, mode="replicate")
        return super().forward(x)


class ChannelRMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        # Keep the FlashVSR 1.1 checkpoint shape: [C,1,1,1].
        self.weight = nn.Parameter(torch.ones(channels, 1, 1, 1))
        self.scale = math.sqrt(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.normalize(x, dim=1) * self.scale
        return value * self.weight.unsqueeze(0)


class PixelUnshuffle3D(nn.Module):
    def __init__(self, spatial_factor: int):
        super().__init__()
        self.spatial_factor = int(spatial_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = self.spatial_factor
        batch, channels, frames, height, width = x.shape
        if height % factor or width % factor:
            raise ValueError(f"LQ video spatial size {(height, width)} must be divisible by {factor}")
        x = x.view(batch, channels, frames, height // factor, factor, width // factor, factor)
        return x.permute(0, 1, 4, 6, 2, 3, 5).reshape(
            batch, channels * factor * factor, frames, height // factor, width // factor
        )


class LQProjection(nn.Module):
    """Independent replacement for the original Buffer_LQ4x_Proj."""

    def __init__(self, config: LQProjectionConfig, output_dim: int):
        super().__init__()
        self.config = config
        factor = config.spatial_factor
        self.pixel_shuffle = PixelUnshuffle3D(factor)
        self.conv1 = CausalConv3d(
            config.input_channels * factor * factor,
            config.hidden_dim,
            kernel_size=(4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm1 = ChannelRMSNorm(config.hidden_dim)
        self.conv2 = CausalConv3d(
            config.hidden_dim,
            config.hidden_dim_2,
            kernel_size=(4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm2 = ChannelRMSNorm(config.hidden_dim_2)
        self.act = nn.SiLU()
        self.linear_layers = nn.ModuleList(
            nn.Linear(config.hidden_dim_2, output_dim) for _ in range(config.num_injection_layers)
        )
        if config.zero_init:
            for layer in self.linear_layers:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.clear_cache()

    def clear_cache(self) -> None:
        self._stream_cache1: torch.Tensor | None = None
        self._stream_cache2: torch.Tensor | None = None
        self._stream_clip_index = 0

    def stream_forward(self, video_clip: torch.Tensor) -> list[torch.Tensor] | None:
        """Process one FlashVSR 1.1 LQ chunk while preserving causal conv state.

        The first call is the conv2 warm-up and intentionally returns ``None``.
        Every subsequent complete 4-frame call returns one latent-token frame.
        """
        if video_clip.ndim != 5:
            raise ValueError(f"LQProjection expects [B,C,F,H,W], got {video_clip.shape}")
        if self._stream_clip_index == 0:
            if video_clip.shape[2] != 1:
                raise ValueError(
                    "the first LQ stream chunk must contain exactly one frame"
                )
            first = video_clip[:, :, :1].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first, video_clip], dim=2)
        elif video_clip.shape[2] != 4:
            raise ValueError(
                f"later LQ stream chunks must contain 4 frames, got {video_clip.shape[2]}"
            )

        x = self.pixel_shuffle(video_clip)
        next_cache1 = x[:, :, -2:].clone()
        x = self.act(self.norm1(self.conv1(x, self._stream_cache1)))
        self._stream_cache1 = next_cache1
        next_cache2 = x[:, :, -2:].clone()
        if self._stream_clip_index == 0:
            self._stream_cache2 = next_cache2
            self._stream_clip_index += 1
            return None
        x = self.act(self.norm2(self.conv2(x, self._stream_cache2)))
        self._stream_cache2 = next_cache2
        self._stream_clip_index += 1
        tokens = x.permute(0, 2, 3, 4, 1).flatten(1, 3)
        return [layer(tokens) for layer in self.linear_layers]

    def forward(self, video: torch.Tensor) -> list[torch.Tensor]:
        if video.ndim != 5:
            raise ValueError(f"LQProjection expects [B,C,F,H,W], got {video.shape}")
        if self.config.pad_last_frames:
            video = torch.cat(
                [video, video[:, :, -1:].repeat(1, 1, self.config.pad_last_frames, 1, 1)], dim=2
            )
        iterations = 1 + (video.shape[2] - 1) // 4
        video = torch.cat([video[:, :, :1].repeat(1, 1, 3, 1, 1), video], dim=2)
        cache1 = None
        cache2 = None
        output_chunks = []
        for chunk_index in range(iterations):
            chunk = video[:, :, chunk_index * 4 : (chunk_index + 1) * 4]
            if chunk.shape[2] != 4:
                raise ValueError(
                    f"FlashVSR 1.1 LQ projection requires complete 4-frame chunks, got {chunk.shape[2]}"
                )
            x = self.pixel_shuffle(chunk)
            next_cache1 = x[:, :, -2:].clone()
            x = self.act(self.norm1(self.conv1(x, cache1)))
            cache1 = next_cache1
            next_cache2 = x[:, :, -2:].clone()
            if chunk_index == 0:
                cache2 = next_cache2
                continue
            x = self.act(self.norm2(self.conv2(x, cache2)))
            cache2 = next_cache2
            output_chunks.append(x)
        if not output_chunks:
            raise ValueError("LQ video is too short to produce projection tokens")
        x = torch.cat(output_chunks, dim=2)
        tokens = x.permute(0, 2, 3, 4, 1).flatten(1, 3)
        return [layer(tokens) for layer in self.linear_layers]
