from __future__ import annotations

from typing import Any

import torch

from .base import DecodeResult


class VAEDecoder:
    """Adapter for a Wan VAE-like module exposing ``decode``."""

    def __init__(self, module, *, tiled: bool = True, tile_size=(30, 52), tile_stride=(15, 26)):
        if not hasattr(module, "decode"):
            raise TypeError("VAE module must expose decode(latents, ...)")
        self.module = module
        self.tiled = tiled
        self.tile_size = tuple(tile_size)
        self.tile_stride = tuple(tile_stride)

    def reset(self) -> None:
        if hasattr(self.module, "clear_cache"):
            self.module.clear_cache()

    def decode(
        self,
        latents: torch.Tensor,
        *,
        lq_video: torch.Tensor | None = None,
        state: Any = None,
        stream: bool = False,
    ) -> DecodeResult:
        del lq_video
        tiled_kwargs = {}
        if self.tiled:
            tiled_kwargs.update(
                tiled=True, tile_size=self.tile_size, tile_stride=self.tile_stride
            )
        if stream and hasattr(self.module, "stream_decode"):
            value = self.module.stream_decode(latents, **tiled_kwargs)
        else:
            attempts = (
                {"device": latents.device, **tiled_kwargs},
                tiled_kwargs,
                {"device": latents.device},
                {},
            )
            last_error = None
            for kwargs in attempts:
                try:
                    value = self.module.decode(latents, **kwargs)
                    break
                except TypeError as error:
                    last_error = error
            else:
                raise TypeError("VAE decode signature is not supported") from last_error
        if isinstance(value, (tuple, list)):
            value = value[0]
        elif hasattr(value, "sample"):
            value = value.sample
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            value = value.unsqueeze(0)
        if not isinstance(value, torch.Tensor) or value.ndim != 5:
            raise TypeError("VAE decode must return [B,C,F,H,W] Tensor")
        next_state = state
        if stream and hasattr(self.module, "stream_state"):
            next_state = self.module.stream_state
        return DecodeResult(value.clamp(-1, 1), next_state)
