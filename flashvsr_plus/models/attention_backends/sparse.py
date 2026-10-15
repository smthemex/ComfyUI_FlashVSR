from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import AttentionConfig


@dataclass(frozen=True)
class SparseAttentionLayout:
    block_mask: torch.Tensor
    padded_shape: tuple[int, int, int]
    grid_shape: tuple[int, int, int]
    block_token_size: int


@dataclass(frozen=True)
class StreamSparseAttentionLayout:
    block_mask: torch.Tensor
    cache_indices: torch.Tensor
    block_token_size: int


def _ceil_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def padded_token_shape(
    token_shape: tuple[int, int, int], window: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(_ceil_multiple(value, size) for value, size in zip(token_shape, window))


def partition_heads(
    value: torch.Tensor,
    token_shape: tuple[int, int, int],
    window: tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
    """Partition [B,H,S,D] into [B,H,num_blocks,block_tokens,D]."""
    batch, heads, tokens, dim = value.shape
    frames, height, width = token_shape
    if tokens != frames * height * width:
        raise ValueError(f"Token count {tokens} does not match token_shape={token_shape}")
    wf, wh, ww = window
    fp, hp, wp = padded_token_shape(token_shape, window)
    value = value.view(batch, heads, frames, height, width, dim)
    if fp != frames:
        value = torch.cat(
            [value, value[:, :, -1:].expand(batch, heads, fp - frames, height, width, dim)],
            dim=2,
        )
    if hp != height or wp != width:
        # F.pad starts with the last dimension; do not pad head_dim.
        value = F.pad(value, (0, 0, 0, wp - width, 0, hp - height))
    nf, nh, nw = fp // wf, hp // wh, wp // ww
    value = value.view(batch, heads, nf, wf, nh, wh, nw, ww, dim)
    value = value.permute(0, 1, 2, 4, 6, 3, 5, 7, 8).contiguous()
    return value.view(batch, heads, nf * nh * nw, wf * wh * ww, dim), (fp, hp, wp), (nf, nh, nw)


def reverse_heads(
    windows: torch.Tensor,
    token_shape: tuple[int, int, int],
    window: tuple[int, int, int],
    padded_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Reverse [B,H,num_blocks,block_tokens,D] to cropped [B,H,S,D]."""
    batch, heads, _blocks, _tokens, dim = windows.shape
    frames, height, width = token_shape
    fp, hp, wp = padded_shape
    wf, wh, ww = window
    nf, nh, nw = fp // wf, hp // wh, wp // ww
    value = windows.view(batch, heads, nf, nh, nw, wf, wh, ww, dim)
    value = value.permute(0, 1, 2, 5, 3, 6, 4, 7, 8).contiguous()
    value = value.view(batch, heads, fp, hp, wp, dim)
    return value[:, :, :frames, :height, :width].reshape(
        batch, heads, frames * height * width, dim
    )


def _spatial_local_mask(
    grid_shape: tuple[int, int, int], local_range: int, device: torch.device
) -> torch.Tensor:
    temporal, rows, cols = grid_shape
    block_ids = torch.arange(temporal * rows * cols, device=device)
    spatial = block_ids.remainder(rows * cols)
    row = torch.div(spatial, cols, rounding_mode="floor")
    col = spatial.remainder(cols)
    half = local_range // 2
    row_start = row - half
    col_start = col - half
    row_ok = (row[None, :] >= row_start[:, None]) & (
        row[None, :] <= row_start[:, None] + local_range - 1
    )
    col_ok = (col[None, :] >= col_start[:, None]) & (
        col[None, :] <= col_start[:, None] + local_range - 1
    )
    return row_ok & col_ok


@lru_cache(maxsize=64)
def _temporal_local_candidate_mask(
    blocks: int,
    temporal: int,
    rows: int,
    cols: int,
    group_size: int,
    temporal_mode: str,
    cache_temporal_windows: int,
    local_range: int | None,
    device_type: str,
    device_index: int,
) -> torch.Tensor:
    """Cache the Q/K-independent candidate relation for sliding-local layers."""
    device = torch.device(device_type, device_index)
    spatial = rows * cols
    block_ids = torch.arange(blocks, device=device)
    time = torch.div(block_ids, spatial, rounding_mode="floor")
    time_group = torch.div(time, group_size, rounding_mode="floor")
    if temporal_mode == "short_causal":
        query_group_start = time_group * group_size
        history_start = query_group_start - cache_temporal_windows
        candidates = (time[None, :] >= history_start[:, None]) & (
            time[None, :] < query_group_start[:, None] + group_size
        )
    else:
        candidates = time_group[:, None] == time_group[None, :]
    if local_range is not None:
        candidates &= _spatial_local_mask(
            (temporal, rows, cols), local_range, device
        )
    return candidates


@lru_cache(maxsize=64)
def _stream_candidate_mask(
    query_blocks: int,
    key_blocks: int,
    rows: int,
    cols: int,
    local_range: int | None,
    device_type: str,
    device_index: int,
) -> torch.Tensor:
    """Cache stream's static spatial candidate relation on each CUDA device."""
    device = torch.device(device_type, device_index)
    if local_range is None:
        return torch.ones((query_blocks, key_blocks), dtype=torch.bool, device=device)
    q_spatial = torch.arange(query_blocks, device=device) % (rows * cols)
    k_spatial = torch.arange(key_blocks, device=device) % (rows * cols)
    q_row, q_col = q_spatial // cols, q_spatial % cols
    k_row, k_col = k_spatial // cols, k_spatial % cols
    half = local_range // 2
    row_start = q_row - half
    col_start = q_col - half
    return (
        (k_row[None] >= row_start[:, None])
        & (k_row[None] <= row_start[:, None] + local_range - 1)
        & (k_col[None] >= col_start[:, None])
        & (k_col[None] <= col_start[:, None] + local_range - 1)
    )


def _candidate_mask(
    grid_shape: tuple[int, int, int],
    config: AttentionConfig,
    window_frames: int,
    device: torch.device,
) -> torch.Tensor:
    temporal, rows, cols = grid_shape
    spatial = rows * cols
    block_ids = torch.arange(temporal * spatial, device=device)
    time = torch.div(block_ids, spatial, rounding_mode="floor")
    allowed = torch.ones(
        temporal * spatial, temporal * spatial, dtype=torch.bool, device=device
    )
    if config.is_causal:
        # Keep the original Wan/FlashVSR convention: for sparse attention,
        # block_size is measured after 3D window partition. With the standard
        # (2, 8, 8) window, block_size=2 means two temporal windows, i.e. four
        # latent frames. Despite its historical field name, this value is not
        # a raw latent-frame count on the sparse path.
        causal_windows = config.causal.block_size_frames
        query_group = torch.div(time[:, None], causal_windows, rounding_mode="floor")
        key_group = torch.div(time[None, :], causal_windows, rounding_mode="floor")
        allowed = key_group <= query_group
        if not config.causal.include_current_block:
            allowed = key_group < query_group
        # FlashVSR/Wan level-1 causal attention operates in consecutive pairs:
        # T0 sees T1, T2 sees T3, and so on. Odd queries already see their even
        # partner through the ordinary causal mask; add the reverse edge for
        # every even query window.
        if (
            config.causal.include_current_block
            and causal_windows == 1
            and temporal > 1
        ):
            allowed |= (
                time[:, None].remainder(2) == 0
            ) & (time[None, :] == time[:, None] + 1)
        if config.sparse.local_blocks is not None:
            allowed &= key_group >= query_group - config.sparse.local_blocks + 1
    if config.sparse.local_range is not None:
        allowed &= _spatial_local_mask(grid_shape, config.sparse.local_range, device)
    return allowed


def _topk_count(config: AttentionConfig, grid_shape: tuple[int, int, int]) -> int:
    if config.sparse.topk is not None:
        return config.sparse.topk
    spatial = grid_shape[1] * grid_shape[2]
    ratio = config.sparse.topk_ratio if config.sparse.topk_ratio is not None else 2.5
    return max(int(spatial * spatial * ratio) - 1, 0)


def _grouped_topk(
    scores: torch.Tensor,
    candidates: torch.Tensor,
    temporal_windows: int,
    topk: int,
) -> torch.Tensor:
    """Match the original per-temporal-group draft-mask selection."""
    batch, heads, query_blocks, key_blocks = scores.shape
    if query_blocks % temporal_windows:
        raise ValueError("query block count must be divisible by temporal windows")
    spatial = query_blocks // temporal_windows
    masked = scores.masked_fill(~candidates, -torch.inf)
    grouped = masked.reshape(batch, heads, temporal_windows, spatial * key_blocks)
    candidate_group = candidates.reshape(batch, heads, temporal_windows, -1)
    # Match FlashVSR 1.1 exactly: take k+1 values and use a strict threshold.
    # This intentionally returns fewer than k entries when scores are tied.
    count = min(grouped.shape[-1] - 1, max(int(topk), 0))
    thresholds = grouped.topk(count + 1, dim=-1).values[..., -1]
    selected = grouped > thresholds[..., None]
    selected &= candidate_group
    return selected.reshape(batch, heads, query_blocks, key_blocks)


@torch.no_grad()
def build_sparse_layout(
    q_windows: torch.Tensor,
    k_windows: torch.Tensor,
    grid_shape: tuple[int, int, int],
    window: tuple[int, int, int],
    config: AttentionConfig,
) -> SparseAttentionLayout:
    """Build the draft block mask used by block-sparse/SDPA kernels."""
    if q_windows.shape != k_windows.shape:
        raise ValueError("Self-attention q/k sparse windows must have equal shapes")
    batch, heads, blocks, block_tokens, head_dim = q_windows.shape
    candidates = _candidate_mask(grid_shape, config, window[0], q_windows.device)
    candidates = candidates.view(1, 1, blocks, blocks).expand(batch, heads, -1, -1)
    if config.is_sparse:
        # Ask PyTorch to accumulate directly in FP32.  ``.float().mean``
        # materializes a full FP32 copy of every token first and costs about
        # 0.5 ms on the 4090 stream shape; ``mean(dtype=...)`` preserves the
        # exact result without that intermediate allocation.
        q_pooled = q_windows.mean(dim=3, dtype=torch.float32)
        k_pooled = k_windows.mean(dim=3, dtype=torch.float32)
        scores = torch.einsum("bhqd,bhkd->bhqk", q_pooled, k_pooled)
        scores = scores / math.sqrt(head_dim)
        attention_map = scores.masked_fill(~candidates, -torch.inf).softmax(dim=-1)
        mask = _grouped_topk(
            attention_map,
            candidates,
            grid_shape[0],
            _topk_count(config, grid_shape),
        )
    else:
        # Fixed causal/local-range layout: avoid pooled QK scoring, softmax,
        # and content-dependent top-k selection entirely.
        mask = candidates
    empty = ~mask.any(dim=-1, keepdim=True)
    if empty.any():
        fallback = torch.eye(blocks, dtype=torch.bool, device=mask.device)
        fallback = fallback.view(1, 1, blocks, blocks).expand_as(mask) & candidates
        mask = torch.where(empty, fallback, mask)
    padded_shape = tuple(size * win for size, win in zip(grid_shape, window))
    return SparseAttentionLayout(mask, padded_shape, grid_shape, block_tokens)


@torch.no_grad()
def build_temporal_local_sparse_layout(
    q_windows: torch.Tensor,
    k_windows: torch.Tensor,
    grid_shape: tuple[int, int, int],
    window: tuple[int, int, int],
    config: AttentionConfig,
) -> SparseAttentionLayout:
    """Draft sparse layout for fixed groups with optional short causal history.

    A group contains ``sliding.temporal_block_size`` partitioned temporal
    windows. ``local`` exposes only that bidirectional group. ``short_causal``
    additionally exposes a bounded number of temporal windows immediately
    before the group, matching the stream KV retention policy.
    """
    if q_windows.shape != k_windows.shape:
        raise ValueError("Temporal-local q/k sparse windows must have equal shapes")
    batch, heads, blocks, block_tokens, head_dim = q_windows.shape
    temporal, rows, cols = grid_shape
    group_size = config.sliding.temporal_block_size
    spatial = rows * cols
    local_range = (
        config.sliding.local_range
        if config.sliding.spatial_mode == "local"
        else config.sparse.local_range
    )
    device_index = q_windows.device.index
    if device_index is None:
        device_index = 0
    candidates = _temporal_local_candidate_mask(
        blocks,
        temporal,
        rows,
        cols,
        group_size,
        config.sliding.temporal_mode,
        config.sliding.cache_temporal_windows,
        local_range,
        q_windows.device.type,
        device_index,
    )
    candidates = candidates.view(1, 1, blocks, blocks).expand(batch, heads, -1, -1)
    if config.is_sparse:
        q_pooled = q_windows.mean(dim=3, dtype=torch.float32)
        k_pooled = k_windows.mean(dim=3, dtype=torch.float32)
        scores = torch.einsum("bhqd,bhkd->bhqk", q_pooled, k_pooled)
        scores = scores / math.sqrt(head_dim)
        attention_map = scores.masked_fill(~candidates, -torch.inf).softmax(dim=-1)
        topk = _topk_count(config, grid_shape)
        grouped_masks = []
        for group_start in range(0, temporal, group_size):
            query_start = group_start * spatial
            query_stop = min(group_start + group_size, temporal) * spatial
            grouped_masks.append(
                _grouped_topk(
                    attention_map[:, :, query_start:query_stop],
                    candidates[:, :, query_start:query_stop],
                    1,
                    topk,
                )
            )
        mask = torch.cat(grouped_masks, dim=2)
    else:
        mask = candidates
    empty = ~mask.any(dim=-1, keepdim=True)
    if empty.any():
        fallback = torch.eye(blocks, dtype=torch.bool, device=mask.device)
        fallback = fallback.view(1, 1, blocks, blocks).expand_as(mask) & candidates
        mask = torch.where(empty, fallback, mask)
    padded_shape = tuple(size * win for size, win in zip(grid_shape, window))
    return SparseAttentionLayout(mask, padded_shape, grid_shape, block_tokens)


@torch.no_grad()
def build_stream_sparse_layout(
    q_windows: torch.Tensor,
    k_windows: torch.Tensor,
    *,
    temporal_windows: int,
    spatial_grid: tuple[int, int],
    topk: int,
    cache_temporal_groups: int,
    temporal_block_size: int,
    local_range: int | None,
    is_sparse: bool = True,
) -> StreamSparseAttentionLayout:
    """Block-causal draft mask/cache policy used by FlashVSR 1.1 stream.

    The scheduler only supplies historical KV and the current block, so future
    chunks are structurally unavailable. No additional token-level triangular
    mask is required inside that block. Cached context is bounded in complete
    temporal windows; retaining only the last few raster-ordered spatial
    windows causes severe regional flicker.
    """
    if q_windows.shape[:2] != k_windows.shape[:2]:
        raise ValueError("stream q/k batch and head dimensions must match")
    batch, heads, query_blocks, block_tokens, head_dim = q_windows.shape
    key_blocks = k_windows.shape[2]
    rows, cols = spatial_grid
    spatial_blocks = rows * cols
    if query_blocks % temporal_windows:
        raise ValueError("stream query blocks must divide into temporal groups")
    if query_blocks // temporal_windows != spatial_blocks:
        raise ValueError("stream query layout does not match spatial grid")
    if key_blocks % spatial_blocks:
        raise ValueError("stream key cache must contain complete spatial grids")
    if temporal_block_size <= 0 or temporal_windows % temporal_block_size:
        raise ValueError("stream temporal block size must divide temporal windows")
    device_index = q_windows.device.index
    if device_index is None:
        device_index = 0
    candidate_base = _stream_candidate_mask(
        query_blocks,
        key_blocks,
        rows,
        cols,
        local_range,
        q_windows.device.type,
        device_index,
    )
    candidates = candidate_base.view(1, 1, query_blocks, key_blocks).expand(
        batch, heads, -1, -1
    )
    if is_sparse:
        q_pooled = q_windows.mean(dim=3, dtype=torch.float32)
        k_pooled = k_windows.mean(dim=3, dtype=torch.float32)
        scores = torch.einsum("bhqd,bhkd->bhqk", q_pooled, k_pooled)
        scores = scores / math.sqrt(head_dim)
        # The grouped selector ranks per-query softmax probabilities. Keep
        # that exact ordering, but normalize in-place so logits and probs do
        # not coexist as two full FP32 matrices on long 4K streams.
        scores.masked_fill_(~candidates, -torch.inf)
        scores.sub_(torch.logsumexp(scores, dim=-1, keepdim=True)).exp_()
        grouped_temporal_blocks = temporal_windows // temporal_block_size
        block_mask = _grouped_topk(
            scores, candidates, grouped_temporal_blocks, topk
        )
    else:
        block_mask = candidates
    # Strict top-k uses ``score > threshold`` and can theoretically produce an
    # empty row when all draft scores tie.  Resolve that rare case here, once,
    # so the production attention wrapper can skip cloning/scanning the mask
    # again on every layer.
    empty = ~block_mask.any(dim=-1, keepdim=True)
    if empty.any():
        first_candidate = candidates.to(torch.int32).argmax(dim=-1, keepdim=True)
        fallback = torch.zeros_like(block_mask).scatter_(-1, first_candidate, True)
        block_mask = torch.where(empty, fallback, block_mask)
    keep = min(
        max(int(cache_temporal_groups), 1) * spatial_blocks,
        key_blocks,
    )
    indices = torch.arange(key_blocks - keep, key_blocks, device=k_windows.device)
    indices = indices.view(1, 1, keep).expand(batch, heads, keep)
    return StreamSparseAttentionLayout(block_mask, indices, block_tokens)
