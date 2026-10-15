"""Inference-only Triton implementation of FlashVSR's stream attention.

This is deliberately narrow: 128 tokens per block and 128 channels per head.
The kernel uses the same online-softmax scheme as FlashAttention-2, but walks a
packed row list of selected key blocks instead of a dense key sequence.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .layout import BLOCK_TOKENS, block_mask_to_packed, validate_stream_tensors


@triton.jit
def _stream_block_sparse_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    row_counts,
    col_idx,
    out_ptr,
    stride_q_bh,
    stride_q_block,
    stride_q_token,
    stride_q_dim,
    stride_k_bh,
    stride_k_block,
    stride_k_token,
    stride_k_dim,
    stride_v_bh,
    stride_v_block,
    stride_v_token,
    stride_v_dim,
    stride_o_bh,
    stride_o_block,
    stride_o_token,
    stride_o_dim,
    q_blocks,
    key_blocks,
    row_stride,
    col_stride,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program computes one BLOCK_M-row tile of one query block/head.

    When BLOCK_M=64, two programs cover a 128-token query block; production
    uses BLOCK_M=128 so the selected K/V blocks are loaded only once. Online
    state is kept per query row while selected key blocks are streamed through
    shared/register tiles.
    """
    pid = tl.program_id(0)
    # Keep the stream contract literal inside the JIT function.  Triton does
    # not allow ordinary Python globals to be referenced from a kernel.
    tiles_per_query = 128 // BLOCK_M
    q_block = pid % q_blocks
    tile = (pid // q_blocks) % tiles_per_query
    bh = pid // (q_blocks * tiles_per_query)

    # These dimensions are fixed by the stream contract.  Supplying the
    # alignment facts lets Triton select vectorized loads and tensor-core
    # layouts without changing any values or operation ordering.
    offs_m = tl.max_contiguous(tl.arange(0, BLOCK_M), BLOCK_M)
    offs_m = tile * BLOCK_M + offs_m
    offs_d = tl.max_contiguous(tl.arange(0, HEAD_DIM), HEAD_DIM)
    tl.multiple_of(offs_d, 16)

    q_base = q_ptr + bh * stride_q_bh + q_block * stride_q_block
    q_ptrs = q_base + offs_m[:, None] * stride_q_token + offs_d[None, :] * stride_q_dim
    q = tl.load(q_ptrs)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    count = tl.load(row_counts + bh * row_stride + q_block)
    col_base = col_idx + bh * q_blocks * col_stride + q_block * col_stride

    # The dynamic packed-row length is small for the stream model.  Triton emits
    # a predicated loop here; no dense Q/K score matrix is materialized.
    for selected in tl.range(0, count):
        k_block = tl.load(col_base + selected)
        k_base = k_ptr + bh * stride_k_bh + k_block * stride_k_block
        v_base = v_ptr + bh * stride_v_bh + k_block * stride_v_block

        # Process one 64-token tile at a time to keep register pressure bounded
        # while still loading a complete 128-token sparse block per iteration.
        for n_start in tl.static_range(0, 128, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            k_ptrs = (
                k_base
                + offs_n[:, None] * stride_k_token
                + offs_d[None, :] * stride_k_dim
            )
            v_ptrs = (
                v_base
                + offs_n[:, None] * stride_v_token
                + offs_d[None, :] * stride_v_dim
            )
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)

            qk = tl.dot(q, tl.trans(k)) * softmax_scale
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            # Keep the tensor-core operands in the input dtype (BF16/FP16),
            # while retaining the online-softmax accumulator in FP32.
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
            m_i = m_new

    out = acc / l_i[:, None]
    o_base = out_ptr + bh * stride_o_bh + q_block * stride_o_block
    o_ptrs = o_base + offs_m[:, None] * stride_o_token + offs_d[None, :] * stride_o_dim
    tl.store(o_ptrs, out)


def stream_block_sparse_attention(
    q_windows: torch.Tensor,
    k_windows: torch.Tensor,
    v_windows: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    ensure_nonempty: bool = True,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int | None = None,
    num_stages: int = 2,
) -> torch.Tensor:
    """Run FlashVSR stream block-sparse attention.

    This function is inference-only.  It intentionally does not implement an
    autograd path; callers should invoke it under ``torch.inference_mode()``.
    """
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (q_windows, k_windows, v_windows)
    ):
        raise RuntimeError("flashvsr_attn is inference-only and does not support autograd")
    batch, heads, query_blocks, key_blocks, head_dim = validate_stream_tensors(
        q_windows, k_windows, v_windows, block_mask
    )
    if not q_windows.is_cuda:
        raise RuntimeError("flashvsr_attn Triton backend requires CUDA")
    if block_m not in (64, 128) or 128 % block_m:
        raise ValueError("block_m must be 64 or 128 for the 128-token stream block")
    if block_n not in (64, 128) or 128 % block_n:
        raise ValueError("block_n must be 64 or 128 for the 128-token stream block")
    if num_warps is None:
        num_warps = 8 if block_m == 128 else 4
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")

    q = q_windows.contiguous().reshape(batch * heads, query_blocks, BLOCK_TOKENS, head_dim)
    k = k_windows.contiguous().reshape(batch * heads, key_blocks, BLOCK_TOKENS, head_dim)
    v = v_windows.contiguous().reshape(batch * heads, key_blocks, BLOCK_TOKENS, head_dim)
    row_counts, col_idx = block_mask_to_packed(
        block_mask.to(device=q.device), ensure_nonempty=ensure_nonempty
    )
    out = torch.empty_like(q)

    grid = (batch * heads * query_blocks * (BLOCK_TOKENS // block_m),)
    _stream_block_sparse_fwd[grid](
        q,
        k,
        v,
        row_counts,
        col_idx,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        query_blocks,
        key_blocks,
        query_blocks,
        key_blocks,
        1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape(batch, heads, query_blocks, BLOCK_TOKENS, head_dim)
