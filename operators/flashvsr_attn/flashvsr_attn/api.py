"""Public API for the narrow FlashVSR stream attention backend."""

from .triton_kernel import stream_block_sparse_attention

__all__ = ["stream_block_sparse_attention"]
