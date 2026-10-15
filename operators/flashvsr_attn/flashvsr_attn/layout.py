"""Utilities for the compact, block-level mask used by FlashVSR streaming."""

from __future__ import annotations

from typing import Tuple

import torch


BLOCK_TOKENS = 128


def _with_nonempty_rows(block_mask: torch.Tensor) -> torch.Tensor:
    """Return a mask in which every query row has at least one key.

    The stream scheduler normally produces non-empty rows.  Keeping this guard
    here makes the low-level kernel safe for top-k=0 and for unusual local
    ranges, where an empty softmax row would otherwise produce NaNs.
    """
    batch, heads, query_blocks, key_blocks = block_mask.shape
    flat = block_mask.reshape(batch * heads, query_blocks, key_blocks)
    empty = ~flat.any(dim=-1)
    # Avoid ``bool(empty.any())`` here: converting a CUDA tensor to a Python
    # bool would synchronize the stream on every inference step.  ``where``
    # and the indexed assignment remain asynchronous; with no empty rows the
    # index tensors simply have zero elements.
    flat = flat.clone()
    rows = torch.arange(query_blocks, device=flat.device)
    cols = rows.clamp_max(max(key_blocks - 1, 0))
    empty_bh, empty_rows = torch.where(empty)
    flat[empty_bh, empty_rows, cols[empty_rows]] = True
    return flat.reshape_as(block_mask)


def block_mask_to_csr(
    block_mask: torch.Tensor,
    *,
    ensure_nonempty: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert ``[B,H,Q_blocks,K_blocks]`` mask to flattened CSR.

    ``row_ptr`` has shape ``[B*H, Q_blocks+1]`` and points into one flattened
    ``col_idx`` array.  The batch/head offset is included in ``row_ptr``, so a
    kernel can use one simple lookup without a second offset array.
    """
    if block_mask.ndim != 4:
        raise ValueError(
            "block_mask must have shape [batch, heads, query_blocks, key_blocks]"
        )
    if block_mask.dtype != torch.bool:
        block_mask = block_mask.bool()
    batch, heads, query_blocks, key_blocks = block_mask.shape
    if key_blocks <= 0:
        raise ValueError("block_mask must contain at least one key block")
    if ensure_nonempty:
        block_mask = _with_nonempty_rows(block_mask)

    flat = block_mask.reshape(batch * heads, query_blocks, key_blocks)
    counts = flat.sum(dim=-1, dtype=torch.int32)
    local_ptr = torch.zeros(
        (batch * heads, query_blocks + 1),
        dtype=torch.int32,
        device=block_mask.device,
    )
    local_ptr[:, 1:] = torch.cumsum(counts, dim=-1, dtype=torch.int32)
    bh_nnz = counts.sum(dim=-1, dtype=torch.int32)
    bh_offsets = torch.zeros_like(bh_nnz)
    if batch * heads > 1:
        bh_offsets[1:] = torch.cumsum(bh_nnz[:-1], dim=0, dtype=torch.int32)
    row_ptr = local_ptr + bh_offsets[:, None]

    # nonzero() is intentionally kept out of the Triton kernel.  It is only a
    # compacting operation on the already-small block mask; the expensive QK/V
    # work remains fully fused in the kernel.
    col_idx = torch.nonzero(flat, as_tuple=False)[:, -1].to(torch.int32)
    return row_ptr.contiguous(), col_idx.contiguous()


def block_mask_to_packed(
    block_mask: torch.Tensor,
    *,
    ensure_nonempty: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack selected key indices to a fixed-width tensor without ``nonzero``.

    Returns ``row_counts`` with shape ``[B*H, Q_blocks]`` and ``col_idx`` with
    shape ``[B*H, Q_blocks, K_blocks]``.  Only the first ``row_counts[..., q]``
    entries of each row are valid.  The fixed-width representation avoids a
    CUDA host synchronization and global CSR offset construction on every
    dynamic stream step; its extra int32 storage is small for FlashVSR's
    bounded cache.
    """
    if block_mask.ndim != 4:
        raise ValueError(
            "block_mask must have shape [batch, heads, query_blocks, key_blocks]"
        )
    if block_mask.dtype != torch.bool:
        block_mask = block_mask.bool()
    batch, heads, query_blocks, key_blocks = block_mask.shape
    if key_blocks <= 0:
        raise ValueError("block_mask must contain at least one key block")
    if ensure_nonempty:
        block_mask = _with_nonempty_rows(block_mask)

    flat = block_mask.reshape(batch * heads, query_blocks, key_blocks)
    row_counts = flat.sum(dim=-1, dtype=torch.int32)
    rank = flat.to(torch.int32).cumsum(dim=-1) - 1
    positions = rank.clamp_min(0)
    key_numbers = torch.arange(
        key_blocks, device=flat.device, dtype=torch.int32
    ).view(1, 1, key_blocks)
    key_numbers = key_numbers.expand_as(flat)
    sentinel = torch.full_like(key_numbers, key_blocks)
    values = torch.where(flat, key_numbers, sentinel)
    packed = torch.full_like(key_numbers, key_blocks)
    packed.scatter_reduce_(
        2, positions, values, reduce="amin", include_self=True
    )
    return row_counts.contiguous(), packed.contiguous()


def validate_stream_tensors(
    q_windows: torch.Tensor,
    k_windows: torch.Tensor,
    v_windows: torch.Tensor,
    block_mask: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    """Validate the narrow stream-kernel contract and return its dimensions."""
    if q_windows.ndim != 5 or k_windows.ndim != 5 or v_windows.ndim != 5:
        raise ValueError("q/k/v must have shape [B,H,blocks,128,head_dim]")
    if q_windows.shape[0:2] != k_windows.shape[0:2] or k_windows.shape != v_windows.shape:
        raise ValueError("k and v must have identical [B,H,blocks,128,D] shapes")
    batch, heads, query_blocks, block_tokens, head_dim = q_windows.shape
    if block_tokens != BLOCK_TOKENS:
        raise ValueError(
            f"stream block sparse attention requires 128 tokens per block, got {block_tokens}"
        )
    if head_dim != 128:
        raise ValueError(
            f"stream block sparse attention currently requires head_dim=128, got {head_dim}"
        )
    if block_mask.shape != (batch, heads, query_blocks, k_windows.shape[2]):
        raise ValueError(
            "block_mask must have shape "
            f"[{batch}, {heads}, {query_blocks}, {k_windows.shape[2]}], "
            f"got {tuple(block_mask.shape)}"
        )
    if q_windows.device != k_windows.device or q_windows.device != v_windows.device:
        raise ValueError("q, k and v must be on the same device")
    if q_windows.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("stream block sparse attention requires FP16 or BF16 tensors")
    if q_windows.dtype != k_windows.dtype or q_windows.dtype != v_windows.dtype:
        raise TypeError("q, k and v must have the same dtype")
    return batch, heads, query_blocks, k_windows.shape[2], head_dim
