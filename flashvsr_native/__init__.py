# -*- coding: utf-8 -*-
"""FlashVSR native (comfy wan patches) implementation package.

Dependencies: torch + comfy only.  No diffusers / transformers / diffsynth.
"""
from .nodes_native import (
    FlashVSR_SM_Model,
    FlashVSR_SM_Sampler,
    FlashVSR_SM_Vae,
    FlashVSR_SM_Decoder,
)

__all__ = [
    "FlashVSR_SM_Model",
    "FlashVSR_SM_Sampler",
    "FlashVSR_SM_Vae",
    "FlashVSR_SM_Decoder",
]
