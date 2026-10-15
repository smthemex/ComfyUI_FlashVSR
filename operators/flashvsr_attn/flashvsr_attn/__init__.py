"""Inference-only block-sparse attention for FlashVSR stream inference."""

from .api import stream_block_sparse_attention
from .layout import BLOCK_TOKENS, block_mask_to_csr, block_mask_to_packed

__all__ = [
    "BLOCK_TOKENS",
    "block_mask_to_csr",
    "block_mask_to_packed",
    "stream_block_sparse_attention",
]

__version__ = "0.1.0"
