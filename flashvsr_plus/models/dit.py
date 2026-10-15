from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn

from .attention import AttentionRuntime
from .blocks import DiTBlock, OutputHead
from .conditioning import LQProjection
from .config import FlashVSRModelConfig
from .embeddings import sinusoidal_embedding_1d


class FlashVSRDiT(nn.Module):
    def __init__(self, config: FlashVSRModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        dit = config.dit
        self.dim = dit.dim
        self.freq_dim = dit.freq_dim
        self.patch_size = tuple(dit.patch_size)
        self.patch_embedding = nn.Conv3d(
            dit.in_channels, dit.dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(dit.text_dim, dit.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dit.dim, dit.dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(dit.freq_dim, dit.dim), nn.SiLU(), nn.Linear(dit.dim, dit.dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dit.dim, dit.dim * 6))

        base_attention = copy.deepcopy(config.attention)
        if base_attention.backend == "hybrid_sliding":
            base_attention.backend = (
                "sparse"
                if (
                    base_attention.is_sparse
                    or base_attention.sparse.local_range is not None
                )
                else "dense"
            )
        if base_attention.is_causal:
            patch_t = self.patch_size[0]
            if not base_attention.causal.pad_to_block and base_attention.causal.block_size_frames % patch_t:
                raise ValueError("causal block_size_frames must be divisible by temporal patch size")
            base_attention.causal.block_size_frames = math.ceil(
                base_attention.causal.block_size_frames / patch_t
            )

        replacement_layers = set(config.replacement.layers)
        attentions = []
        for layer_index in range(dit.num_layers):
            attention = copy.deepcopy(base_attention)
            if layer_index in replacement_layers:
                if attention.sliding.spatial_mode == "local":
                    attention.sparse.local_range = attention.sliding.local_range
                if attention.sliding.temporal_mode == "local":
                    attention.backend = "sliding_local"
                    # Temporal locality uses equality between fixed groups,
                    # not the original causal <= relation.
                    attention.is_causal = False
                    attention.causal.block_size_frames = 1
                elif attention.sliding.temporal_mode == "short_causal":
                    # Keep a bounded causal bridge into the previous
                    # chunk. The specialized layout matches the one-window
                    # stream cache used by this layer.
                    attention.backend = "sliding_local"
                else:
                    # Spatial-only ablation: change only local_range. Keep
                    # the original causal mask and normal stream KV cache.
                    attention.backend = "sparse"
            attentions.append(attention)
        self.blocks = nn.ModuleList(
            DiTBlock(dit.dim, dit.num_heads, dit.ffn_dim, attention, dit.eps)
            for attention in attentions
        )
        self.head = OutputHead(dit.dim, dit.out_channels, self.patch_size, dit.eps)
        self.LQ_proj_in = (
            LQProjection(config.lq_projection, dit.dim) if config.lq_projection.enabled else None
        )

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.patch_embedding(x)
        shape = tuple(x.shape[2:])
        return x.permute(0, 2, 3, 4, 1).flatten(1, 3), shape

    def unpatchify(self, x: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        batch = x.shape[0]
        frames, height, width = shape
        pt, ph, pw = self.patch_size
        channels = self.config.dit.out_channels
        x = x.view(batch, frames, height, width, pt, ph, pw, channels)
        return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            batch, channels, frames * pt, height * ph, width * pw
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        lq_video: Optional[torch.Tensor] = None,
        layer_conditions: Optional[list[torch.Tensor]] = None,
        frame_valid_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"DiT input must be [B,C,F,H,W], got {x.shape}")
        if timestep.ndim == 0:
            timestep = timestep.expand(x.shape[0])
        time_embedding = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=x.dtype))
        time_modulation = self.time_projection(time_embedding).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        x, token_shape = self.patchify(x)
        if lq_video is not None:
            if self.LQ_proj_in is None:
                raise ValueError("lq_video was provided but LQ projection is disabled")
            if layer_conditions is not None:
                raise ValueError("Provide lq_video or layer_conditions, not both")
            layer_conditions = self.LQ_proj_in(lq_video)
        if frame_valid_mask is not None and frame_valid_mask.shape[-1] != token_shape[0]:
            expected_frames = token_shape[0] * self.patch_size[0]
            if frame_valid_mask.shape[-1] != expected_frames:
                raise ValueError(
                    f"frame_valid_mask has {frame_valid_mask.shape[-1]} frames, expected "
                    f"{token_shape[0]} token frames or {expected_frames} input frames"
                )
            frame_valid_mask = frame_valid_mask.view(
                frame_valid_mask.shape[0], token_shape[0], self.patch_size[0]
            ).all(dim=-1)
        runtime = AttentionRuntime(
            token_shape=token_shape,
            frame_valid_mask=frame_valid_mask,
        )
        for block_index, block in enumerate(self.blocks):
            if layer_conditions is not None and block_index < len(layer_conditions):
                condition = layer_conditions[block_index].to(device=x.device, dtype=x.dtype)
                if condition.shape != x.shape:
                    raise ValueError(
                        f"LQ condition {block_index} has shape {condition.shape}, expected {x.shape}"
                    )
                x = x + condition
            x = block(x, context, time_modulation, runtime, context_mask)
        return self.unpatchify(self.head(x, time_embedding), token_shape)

    def forward_stream(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        layer_conditions: list[torch.Tensor],
        state,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Numerically compatible FlashVSR 1.1 streaming DiT forward."""
        if timestep.ndim == 0:
            timestep = timestep.expand(x.shape[0])
        time_embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=x.dtype)
        )
        time_modulation = self.time_projection(time_embedding).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        x, token_shape = self.patchify(x)
        expected_frames = 4
        if token_shape[0] != expected_frames:
            raise ValueError(
                f"FlashVSR stream chunk {state.chunk_index} requires {expected_frames} "
                f"latent frames, got {token_shape[0]}"
            )
        if len(state.attention_cache) not in {0, len(self.blocks)}:
            raise ValueError("attention cache must be empty or contain one entry per DiT block")
        old_caches = state.attention_cache
        old_ready_events = getattr(state, "attention_cache_ready_events", [])
        # Move ownership out of the long-lived state before updating layers.
        # Each old layer cache is released immediately after that layer has
        # consumed it, instead of keeping complete old and new cache sets
        # alive until the end of the DiT forward.
        state.attention_cache = []
        state.attention_cache_ready_events = []
        next_caches = []
        next_ready_events = []
        offload_layers = getattr(state, "attention_cache_offload_layers", frozenset())
        if offload_layers and max(offload_layers) >= len(self.blocks):
            raise ValueError(
                f"stream cache offload layer index exceeds DiT depth {len(self.blocks)}"
            )
        transfer_stream = getattr(state, "attention_cache_transfer_stream", None)
        for block_index, block in enumerate(self.blocks):
            if block_index < len(layer_conditions):
                condition = layer_conditions[block_index].to(device=x.device, dtype=x.dtype)
                if condition.shape != x.shape:
                    raise ValueError(
                        f"stream LQ condition {block_index} has {condition.shape}, expected {x.shape}"
                    )
                x = x + condition
            cache = (
                None
                if not old_caches
                else old_caches[block_index]
            )
            if old_caches:
                old_caches[block_index] = None
                if old_ready_events:
                    ready_event = old_ready_events[block_index]
                    old_ready_events[block_index] = None
                else:
                    ready_event = None
            else:
                ready_event = None
            if ready_event is not None and x.is_cuda:
                # The previous chunk's D2H copy may still be in flight on
                # the transfer stream. This wait orders the host source before
                # the H2D copy performed by attention on the main stream.
                torch.cuda.current_stream(x.device).wait_event(ready_event)
            runtime = AttentionRuntime(
                token_shape=token_shape,
                temporal_offset=4 * state.chunk_index,
                stream_cache=cache,
                stream_topk=state.attention_topk,
                stream_cache_blocks=state.attention_cache_blocks,
                stream_block_size=state.attention_block_size,
                stream_ffn_chunk_tokens=state.ffn_chunk_tokens,
                stream_block_m=state.attention_block_m,
                stream_compact_cache=getattr(state, "compact_cache", False),
                stream_write_cache=getattr(state, "write_cache", True),
                stream_release_workspaces=getattr(state, "release_attention_workspaces", False),
            )
            trace = state.memory_trace
            if trace is not None and x.is_cuda:
                def cache_stats(value):
                    if value is None:
                        return {"tokens": 0, "bytes": 0, "device": None}
                    k, v = value
                    return {
                        "tokens": int(k.shape[2]),
                        "bytes": int(k.numel() * k.element_size() + v.numel() * v.element_size()),
                        "device": str(k.device),
                    }

                before_cache = cache_stats(cache)
                trace.append({
                    "chunk": int(state.chunk_index),
                    "layer": int(block_index),
                    "stage": "before",
                    "attention": type(block.self_attn).__name__,
                    "alloc_mib": torch.cuda.memory_allocated(x.device) / 1024**2,
                    "reserved_mib": torch.cuda.memory_reserved(x.device) / 1024**2,
                    "peak_alloc_mib": torch.cuda.max_memory_allocated(x.device) / 1024**2,
                    "x_tokens": int(x.shape[1]),
                    "x_bytes": int(x.numel() * x.element_size()),
                    "old_cache_tokens": before_cache["tokens"],
                    "old_cache_bytes": before_cache["bytes"],
                    "old_cache_device": before_cache["device"],
                })
            try:
                x, next_cache = block.forward_stream(
                    x, context, time_modulation, runtime, context_mask
                )
            except Exception as error:
                if trace is not None and x.is_cuda:
                    trace.append({
                        "chunk": int(state.chunk_index),
                        "layer": int(block_index),
                        "stage": "error",
                        "error_type": type(error).__name__,
                        "error": str(error).splitlines()[0],
                        "alloc_mib": torch.cuda.memory_allocated(x.device) / 1024**2,
                        "reserved_mib": torch.cuda.memory_reserved(x.device) / 1024**2,
                        "peak_alloc_mib": torch.cuda.max_memory_allocated(x.device) / 1024**2,
                    })
                raise
            # A cache is consumed only by the same layer on the next stream
            # chunk. Moving selected caches to host memory changes residency,
            # not mask construction or arithmetic. Keep BF16/layout intact;
            # this opt-in path preserves deterministic cache ordering.
            if block_index in offload_layers and next_cache is not None:
                # Page-locked destinations make CUDA D2H copies asynchronous
                # and avoid the very slow pageable staging path. Copies are
                # enqueued on the transfer stream after block computation;
                # the next chunk waits on its completion event before H2D.
                host_cache = []
                transfer_stream = getattr(
                    state, "attention_cache_transfer_stream", None
                )
                current_stream = (
                    torch.cuda.current_stream(x.device) if x.is_cuda else None
                )
                if transfer_stream is not None and current_stream is not None:
                    transfer_stream.wait_stream(current_stream)
                for cache_index, value in enumerate(next_cache):
                    value = value.detach()
                    if getattr(state, "attention_cache_pin_memory", True):
                        try:
                            previous = None if cache is None else cache[cache_index]
                            if (
                                getattr(state, "reuse_host_cache", False)
                                and previous is not None
                                and previous.device.type == "cpu"
                                and previous.is_pinned()
                                and previous.shape == value.shape
                                and previous.dtype == value.dtype
                            ):
                                # H2D consumption precedes block completion on
                                # the compute stream. The transfer stream wait
                                # above prevents overwriting a still-read source.
                                host_value = previous
                            else:
                                host_value = torch.empty_like(
                                    value, device="cpu", pin_memory=True
                                )
                            if transfer_stream is not None:
                                value.record_stream(transfer_stream)
                                with torch.cuda.stream(transfer_stream):
                                    host_value.copy_(value, non_blocking=True)
                            else:
                                host_value.copy_(value, non_blocking=True)
                        except RuntimeError:
                            # Gracefully fall back on systems with an
                            # insufficient pinned-memory limit. Correctness
                            # is unchanged; only transfer throughput degrades.
                            if transfer_stream is not None:
                                transfer_stream.synchronize()
                                transfer_stream = None
                            # A failed pin allocation cannot be safely queued
                            # on the transfer stream; fall back synchronously.
                            host_value = value.to(device="cpu")
                    else:
                        host_value = value.to(device="cpu")
                    host_cache.append(host_value)
                next_cache = tuple(host_cache)
                ready_event = None
                if transfer_stream is not None:
                    with torch.cuda.stream(transfer_stream):
                        ready_event = transfer_stream.record_event()
                next_ready_events.append(ready_event)
            else:
                next_ready_events.append(None)
            if trace is not None and x.is_cuda:
                next_stats = cache_stats(next_cache)
                trace.append({
                    "chunk": int(state.chunk_index),
                    "layer": int(block_index),
                    "stage": "after",
                    "attention": type(block.self_attn).__name__,
                    "alloc_mib": torch.cuda.memory_allocated(x.device) / 1024**2,
                    "reserved_mib": torch.cuda.memory_reserved(x.device) / 1024**2,
                    "peak_alloc_mib": torch.cuda.max_memory_allocated(x.device) / 1024**2,
                    "next_cache_tokens": next_stats["tokens"],
                    "next_cache_bytes": next_stats["bytes"],
                    "next_cache_device": next_stats["device"],
                })
            next_caches.append(next_cache)
            del runtime, cache
        state.attention_cache = next_caches
        state.attention_cache_ready_events = next_ready_events
        return self.unpatchify(self.head(x, time_embedding), token_shape)
