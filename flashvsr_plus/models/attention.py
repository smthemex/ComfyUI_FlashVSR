from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AttentionConfig
from .embeddings import apply_rope_3d
from .attention_backends import (
    build_sparse_layout,
    build_temporal_local_sparse_layout,
    run_attention,
)
from .attention_backends.sparse import partition_heads, reverse_heads
from .attention_backends.sparse import build_stream_sparse_layout

try:
    from flashvsr_attn import stream_block_sparse_attention
except ImportError:  # Optional inference-only backend.
    stream_block_sparse_attention = None


@dataclass
class AttentionRuntime:
    token_shape: tuple[int, int, int]
    frame_valid_mask: Optional[torch.Tensor] = None
    temporal_offset: int = 0
    stream_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    stream_topk: Optional[int] = None
    stream_cache_blocks: int = 3
    stream_block_size: int = 1
    stream_ffn_chunk_tokens: int | None = None
    stream_block_m: int = 128
    stream_compact_cache: bool = False
    stream_write_cache: bool = True
    stream_release_workspaces: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.float()
        value = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return value.to(dtype=x.dtype) * self.weight


def build_block_causal_mask(
    token_shape: tuple[int, int, int],
    block_size_frames: int,
    device: torch.device,
    include_current_block: bool = True,
) -> torch.Tensor:
    frames, height, width = token_shape
    frame_blocks = torch.arange(frames, device=device).div(block_size_frames, rounding_mode="floor")
    token_blocks = frame_blocks.repeat_interleave(height * width)
    if include_current_block:
        return token_blocks[:, None] >= token_blocks[None, :]
    return token_blocks[:, None] > token_blocks[None, :]


class DenseSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, config: AttentionConfig, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.config = config
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, runtime: AttentionRuntime) -> torch.Tensor:
        q = apply_rope_3d(self._heads(self.norm_q(self.q(x))), runtime.token_shape)
        k = apply_rope_3d(self._heads(self.norm_k(self.k(x))), runtime.token_shape)
        v = self._heads(self.v(x))
        mask = None
        if self.config.is_causal:
            mask = build_block_causal_mask(
                runtime.token_shape,
                self.config.causal.block_size_frames,
                x.device,
                self.config.causal.include_current_block,
            ).view(1, 1, x.shape[1], x.shape[1])
        if runtime.frame_valid_mask is not None:
            frames, height, width = runtime.token_shape
            valid = runtime.frame_valid_mask.bool().repeat_interleave(height * width, dim=-1)
            valid_mask = valid[:, None, :, None] & valid[:, None, None, :]
            mask = valid_mask if mask is None else (mask & valid_mask)
        output = run_attention(q, k, v, kernel=self.config.kernel, token_mask=mask)
        output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        return self.o(output)

    def forward_stream(
        self, x: torch.Tensor, runtime: AttentionRuntime
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Dense attention over the current chunk and its bounded KV history."""
        frames, height, width = runtime.token_shape
        q = apply_rope_3d(
            self._heads(self.norm_q(self.q(x))),
            runtime.token_shape,
            temporal_offset=runtime.temporal_offset,
        )
        k = apply_rope_3d(
            self._heads(self.norm_k(self.k(x))),
            runtime.token_shape,
            temporal_offset=runtime.temporal_offset,
        )
        v = self._heads(self.v(x))
        if runtime.stream_cache is not None:
            cache_k, cache_v = runtime.stream_cache
            k = torch.cat([cache_k.to(k), k], dim=2)
            v = torch.cat([cache_v.to(v), v], dim=2)
        output = run_attention(q, k, v, kernel=self.config.kernel)
        tokens_per_frame = height * width
        keep_frames = max(runtime.stream_cache_blocks, 1) * 2
        keep_tokens = min(k.shape[2], keep_frames * tokens_per_frame)
        next_cache = (k[:, :, -keep_tokens:].detach(), v[:, :, -keep_tokens:].detach())
        output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        return self.o(output), next_cache


class BlockSparseSelfAttention(DenseSelfAttention):
    """Wan-style 3D-window draft mask with selectable sparse kernel."""

    def _standard_layout(self, q_windows, k_windows, grid_shape, window):
        return build_sparse_layout(q_windows, k_windows, grid_shape, window, self.config)

    def _stream_local_range(self) -> int | None:
        return self.config.sparse.local_range

    def forward(self, x: torch.Tensor, runtime: AttentionRuntime) -> torch.Tensor:
        q = apply_rope_3d(self._heads(self.norm_q(self.q(x))), runtime.token_shape)
        k = apply_rope_3d(self._heads(self.norm_k(self.k(x))), runtime.token_shape)
        v = self._heads(self.v(x))
        window = tuple(self.config.sparse.window_size)
        q_windows, padded_shape, grid_shape = partition_heads(q, runtime.token_shape, window)
        k_windows, _, _ = partition_heads(k, runtime.token_shape, window)
        v_windows, _, _ = partition_heads(v, runtime.token_shape, window)
        layout = self._standard_layout(q_windows, k_windows, grid_shape, window)
        batch, heads, blocks, block_tokens, head_dim = q_windows.shape
        q_flat = q_windows.view(batch, heads, blocks * block_tokens, head_dim)
        k_flat = k_windows.view(batch, heads, blocks * block_tokens, head_dim)
        v_flat = v_windows.view(batch, heads, blocks * block_tokens, head_dim)
        output = run_attention(
            q_flat,
            k_flat,
            v_flat,
            kernel=self.config.kernel,
            block_mask=layout.block_mask,
            block_token_size=layout.block_token_size,
        )
        output = output.reshape(batch, heads, blocks, block_tokens, head_dim)
        output = reverse_heads(output, runtime.token_shape, window, padded_shape)
        if runtime.frame_valid_mask is not None:
            frames, height, width = runtime.token_shape
            valid = runtime.frame_valid_mask.bool().repeat_interleave(height * width, dim=-1)
            output = output * valid[:, None, :, None].to(dtype=output.dtype)
        output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        return self.o(output)

    def forward_stream(
        self, x: torch.Tensor, runtime: AttentionRuntime
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """FlashVSR 1.1 streaming attention with per-layer window KV cache."""
        return self._forward_stream_impl(x, runtime, use_cache=True)

    def _forward_stream_impl(
        self,
        x: torch.Tensor,
        runtime: AttentionRuntime,
        *,
        use_cache: bool,
        cache_temporal_windows: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        frames, height, width = runtime.token_shape
        expected = 4
        if frames != expected:
            raise ValueError(
                f"stream attention expects {expected} latent token frames, got {frames}"
            )
        window = tuple(self.config.sparse.window_size)
        if any(size % win for size, win in zip(runtime.token_shape, window)):
            raise ValueError(
                f"stream token shape {runtime.token_shape} must be divisible by {window}"
            )
        q = apply_rope_3d(
            self._heads(self.norm_q(self.q(x))),
            runtime.token_shape,
            temporal_offset=runtime.temporal_offset,
        )
        k = apply_rope_3d(
            self._heads(self.norm_k(self.k(x))),
            runtime.token_shape,
            temporal_offset=runtime.temporal_offset,
        )
        v = self._heads(self.v(x))
        q_windows, padded_shape, grid_shape = partition_heads(q, runtime.token_shape, window)
        k_windows, _, _ = partition_heads(k, runtime.token_shape, window)
        v_windows, _, _ = partition_heads(v, runtime.token_shape, window)
        if runtime.stream_release_workspaces:
            # Window partition has its own storage. The unpartitioned Q/K/V
            # are no longer consumed by this inference-only path.
            del q, k, v
        if use_cache and runtime.stream_cache is not None:
            cache_k, cache_v = runtime.stream_cache
            k_windows = torch.cat([cache_k.to(k_windows), k_windows], dim=2)
            v_windows = torch.cat([cache_v.to(v_windows), v_windows], dim=2)
        spatial = grid_shape[1] * grid_shape[2]
        ratio = self.config.sparse.topk_ratio or 2.0
        topk = runtime.stream_topk
        if topk is None:
            topk = int(spatial * spatial * ratio) - 1
        layout = build_stream_sparse_layout(
            q_windows,
            k_windows,
            temporal_windows=grid_shape[0],
            spatial_grid=(grid_shape[1], grid_shape[2]),
            topk=topk,
            cache_temporal_groups=(
                runtime.stream_cache_blocks
                if cache_temporal_windows is None
                else cache_temporal_windows
            ),
            temporal_block_size=runtime.stream_block_size,
            local_range=self._stream_local_range(),
            is_sparse=self.config.is_sparse,
        )
        batch, heads, query_blocks, block_tokens, head_dim = q_windows.shape
        key_blocks = k_windows.shape[2]
        # Keep the explicitly selected stream backends distinct:
        # ``block_sparse`` uses the BSA extension through run_attention(),
        # while ``triton_block_sparse`` uses the dedicated Triton kernel.
        use_triton_stream = self.config.kernel == "triton_block_sparse"
        if use_triton_stream:
            if stream_block_sparse_attention is None:
                raise ImportError(
                    "kernel='triton_block_sparse' requires the optional "
                    "flashvsr-attn package; install it with "
                    "pip install -e /path/to/flashvsr_attn "
                    "--no-build-isolation --no-deps"
                )
            output = stream_block_sparse_attention(
                q_windows,
                k_windows,
                v_windows,
                layout.block_mask,
                ensure_nonempty=False,
                block_m=runtime.stream_block_m,
            )
        else:
            output = run_attention(
                q_windows.reshape(batch, heads, query_blocks * block_tokens, head_dim),
                k_windows.reshape(batch, heads, key_blocks * block_tokens, head_dim),
                v_windows.reshape(batch, heads, key_blocks * block_tokens, head_dim),
                kernel=self.config.kernel,
                block_mask=layout.block_mask,
                block_token_size=layout.block_token_size,
            )
        output = output.reshape(batch, heads, query_blocks, block_tokens, head_dim)
        output = reverse_heads(output, runtime.token_shape, window, padded_shape)
        next_cache = None
        if use_cache and runtime.stream_write_cache and runtime.stream_compact_cache:
            # The layout retains a contiguous suffix of complete temporal grids.
            # Clone into compact storage: a view would retain the entire KV cat.
            keep = layout.cache_indices.shape[2]
            next_cache = tuple(
                value[:, :, -keep:].detach().clone(memory_format=torch.contiguous_format)
                for value in (k_windows, v_windows)
            )
        elif use_cache and runtime.stream_write_cache:
            gather = layout.cache_indices[..., None, None].expand(
                -1, -1, -1, block_tokens, head_dim
            )
            next_cache = (
                torch.gather(k_windows, 2, gather).detach(),
                torch.gather(v_windows, 2, gather).detach(),
            )
        output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.dim)
        return self.o(output), next_cache


class SlidingLocalSelfAttention(BlockSparseSelfAttention):
    """Draft sparse spatial range with bidirectional fixed temporal groups."""

    def _standard_layout(self, q_windows, k_windows, grid_shape, window):
        return build_temporal_local_sparse_layout(
            q_windows, k_windows, grid_shape, window, self.config
        )

    def _stream_local_range(self) -> int | None:
        if self.config.sliding.spatial_mode == "local":
            return self.config.sliding.local_range
        return self.config.sparse.local_range

    def forward_stream(
        self, x: torch.Tensor, runtime: AttentionRuntime
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        if runtime.stream_block_size != self.config.sliding.temporal_block_size:
            raise ValueError(
                "hybrid_sliding stream block size must match the training "
                "temporal block size"
            )
        if self.config.sliding.temporal_mode == "local":
            # Fully local replacement layers are self-contained in the
            # current four-latent group and do not persist KV.
            output, _ = self._forward_stream_impl(x, runtime, use_cache=False)
            return output, None
        if self.config.sliding.temporal_mode == "short_causal":
            return self._forward_stream_impl(
                x,
                runtime,
                use_cache=True,
                cache_temporal_windows=(
                    self.config.sliding.cache_temporal_windows
                ),
            )
        raise ValueError(
            "SlidingLocalSelfAttention requires temporal_mode local or "
            "short_causal"
        )


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(context_dim, dim)
        self.v = nn.Linear(context_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None = None):
        batch, tokens, dim = x.shape
        q = self.norm_q(self.q(x)).view(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.norm_k(self.k(context)).view(
            batch, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v(context).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        mask = None if context_mask is None else context_mask[:, None, None, :].bool()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(batch, tokens, dim))


def build_self_attention(dim: int, num_heads: int, config: AttentionConfig, eps: float):
    config.validate()
    if config.backend == "sliding_local":
        return SlidingLocalSelfAttention(dim, num_heads, config, eps)
    if config.is_sparse or config.backend == "sparse":
        return BlockSparseSelfAttention(dim, num_heads, config, eps)
    if config.backend == "dense":
        return DenseSelfAttention(dim, num_heads, config, eps)
    raise NotImplementedError(
        f"Attention backend={config.backend!r} is not implemented"
    )
