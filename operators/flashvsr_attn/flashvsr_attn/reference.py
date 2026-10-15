"""Small eager reference used by correctness tests (not a production backend)."""

from __future__ import annotations

import math

import torch

from .layout import _with_nonempty_rows, validate_stream_tensors


@torch.no_grad()
def reference_block_sparse_attention(
    q_windows: torch.Tensor,
    k_windows: torch.Tensor,
    v_windows: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    ensure_nonempty: bool = True,
) -> torch.Tensor:
    """Compute exact block sparse attention using eager PyTorch operations."""
    batch, heads, query_blocks, key_blocks, head_dim = validate_stream_tensors(
        q_windows, k_windows, v_windows, block_mask
    )
    if ensure_nonempty:
        block_mask = _with_nonempty_rows(block_mask.bool())
    output = torch.empty_like(q_windows)
    scale = 1.0 / math.sqrt(head_dim)
    for b in range(batch):
        for h in range(heads):
            for qb in range(query_blocks):
                selected = torch.where(block_mask[b, h, qb])[0]
                q = q_windows[b, h, qb].float()
                k = k_windows[b, h, selected].float().reshape(-1, head_dim)
                v = v_windows[b, h, selected].float().reshape(-1, head_dim)
                scores = torch.matmul(q, k.transpose(0, 1)) * scale
                probs = torch.softmax(scores, dim=-1)
                result = torch.matmul(probs, v)
                output[b, h, qb] = result.to(output.dtype)
    return output
