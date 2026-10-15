from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamState:
    chunk_index: int = 0
    input_frame_offset: int = 0
    output_frame_offset: int = 0
    attention_cache: list[Any] = field(default_factory=list)
    attention_topk: int | None = None
    attention_cache_blocks: int = 3
    attention_block_size: int = 1
    decoder_state: Any = None
    # Filled only when InferenceConfig.profile_memory is enabled.
    memory_trace: list[dict[str, Any]] | None = None
    ffn_chunk_tokens: int | None = None
    attention_block_m: int = 128
    # Layer-local KV caches persisted on host memory between chunks.
    attention_cache_offload_layers: frozenset[int] = field(default_factory=frozenset)
    attention_cache_pin_memory: bool = True
    # Optional CUDA copy stream/event state used by pinned KV offload.
    attention_cache_ready_events: list[Any] = field(default_factory=list)
    attention_cache_transfer_stream: Any = None
    # Experimental exact-copy optimizations, selected by DiT-only benchmarks.
    compact_cache: bool = False
    reuse_host_cache: bool = False
    write_cache: bool = True
    release_attention_workspaces: bool = False
