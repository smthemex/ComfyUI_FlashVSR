from __future__ import annotations

from ..config import DecoderConfig

from .tiny import TinyDecoder
from .vae import VAEDecoder


def build_decoder(config: DecoderConfig, module):
    config.validate()
    if config.kind == "vae":
        return VAEDecoder(
            module,
            tiled=config.tiled,
            tile_size=config.tile_size,
            tile_stride=config.tile_stride,
        )
    return TinyDecoder(
        module,
        parallel=False,
        parallel_mode=config.tiny_decoder_parallel_mode,
        channels_last=config.tiny_decoder_channels_last,
    )
