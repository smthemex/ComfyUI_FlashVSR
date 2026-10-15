from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiTConfig:
    dim: int = 1536
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30
    patch_size: tuple[int, int, int] = (1, 2, 2)
    eps: float = 1e-6

    def validate(self) -> None:
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if any(value <= 0 for value in self.patch_size):
            raise ValueError("patch_size values must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


@dataclass
class CausalAttentionConfig:
    block_size_frames: int = 1
    include_current_block: bool = True
    pad_to_block: bool = True

    def validate(self) -> None:
        if self.block_size_frames <= 0:
            raise ValueError("causal block_size_frames must be positive")


@dataclass
class SparseAttentionConfig:
    window_size: tuple[int, int, int] = (2, 8, 8)
    topk: Optional[int] = None
    topk_ratio: Optional[float] = 2.5
    local_blocks: Optional[int] = None
    local_range: Optional[int] = None

    def validate(self) -> None:
        if len(self.window_size) != 3 or any(
            value <= 0 for value in self.window_size
        ):
            raise ValueError(
                "sparse.window_size must contain three positive values"
            )


@dataclass
class SlidingAttentionConfig:
    local_range: int = 3
    temporal_block_size: int = 2
    temporal_mode: str = "local"
    spatial_mode: str = "local"
    cache_temporal_windows: int = 1

    def validate(self) -> None:
        self.temporal_mode = self.temporal_mode.lower()
        self.spatial_mode = self.spatial_mode.lower()
        if self.local_range <= 0:
            raise ValueError("sliding.local_range must be positive")
        if self.temporal_block_size <= 0:
            raise ValueError("sliding.temporal_block_size must be positive")
        if self.cache_temporal_windows < 0:
            raise ValueError(
                "sliding.cache_temporal_windows cannot be negative"
            )
        if self.temporal_mode not in {"local", "causal", "short_causal"}:
            raise ValueError(
                "sliding.temporal_mode must be 'local', 'causal', or "
                "'short_causal'"
            )
        if (
            self.temporal_mode == "short_causal"
            and self.cache_temporal_windows == 0
        ):
            raise ValueError(
                "sliding short_causal mode requires at least one cached "
                "temporal window"
            )
        if self.spatial_mode not in {"local", "base"}:
            raise ValueError("sliding.spatial_mode must be 'local' or 'base'")


@dataclass
class AttentionConfig:
    backend: str = "dense"
    kernel: str = "auto"
    is_causal: bool = False
    is_sparse: bool = False
    is_hybrid_sliding: bool = False
    causal: CausalAttentionConfig = field(default_factory=CausalAttentionConfig)
    sparse: SparseAttentionConfig = field(default_factory=SparseAttentionConfig)
    sliding: SlidingAttentionConfig = field(default_factory=SlidingAttentionConfig)

    def validate(self) -> None:
        self.backend = self.backend.lower()
        if self.backend not in {"dense", "sparse", "sliding_local", "hybrid_sliding"}:
            raise ValueError(f"Unknown attention backend: {self.backend}")
        self.kernel = self.kernel.lower().replace("-", "_")
        aliases = {
            "fa2": "flash_attn2",
            "fa3": "flash_attn3",
            "flash2": "flash_attn2",
            "flash3": "flash_attn3",
            "triton_sparse": "triton_block_sparse",
        }
        self.kernel = aliases.get(self.kernel, self.kernel)
        if self.kernel not in {
            "auto",
            "sdpa",
            "flash_attn2",
            "flash_attn3",
            "block_sparse",
            "triton_block_sparse",
        }:
            raise ValueError(f"Unknown attention kernel: {self.kernel}")
        self.causal.validate()
        self.sparse.validate()
        self.sliding.validate()
        if (
            self.is_hybrid_sliding
            and not self.is_sparse
            and self.sparse.local_range is None
        ):
            raise ValueError(
                "hybrid_sliding requires draft attention or a fixed local_range"
            )
        if (
            self.is_hybrid_sliding
            and self.sliding.temporal_mode in {"causal", "short_causal"}
            and not self.is_causal
        ):
            raise ValueError(
                "hybrid_sliding causal temporal modes require causal attention"
            )
        if (
            self.is_hybrid_sliding
            and self.is_causal
            and self.causal.block_size_frames != self.sliding.temporal_block_size
        ):
            raise ValueError(
                "hybrid_sliding requires causal.block_size_frames to match "
                "sliding.temporal_block_size"
            )
        if not self.is_causal and self.causal.block_size_frames != 1:
            raise ValueError("block_size_frames is only meaningful when is_causal=True")
        if self.is_sparse and self.backend not in {"sparse", "sliding_local", "hybrid_sliding"}:
            raise ValueError("is_sparse=True requires a sparse attention backend")
        if self.sparse.topk is not None and self.sparse.topk <= 0:
            raise ValueError("sparse.topk must be positive")
        if self.sparse.topk_ratio is not None and self.sparse.topk_ratio <= 0:
            raise ValueError("sparse.topk_ratio must be positive")
        if self.sparse.local_blocks is not None and self.sparse.local_blocks <= 0:
            raise ValueError("sparse.local_blocks must be positive")
        if self.sparse.local_range is not None and self.sparse.local_range <= 0:
            raise ValueError("sparse.local_range must be positive")
        sparse_masked = self.backend in {"sparse", "sliding_local", "hybrid_sliding"} and (
            self.is_sparse or self.sparse.local_range is not None
        )
        if self.kernel in {"flash_attn2", "flash_attn3"} and (
            self.is_causal or sparse_masked
        ):
            raise ValueError(
                f"{self.kernel} cannot represent block causal/sparse masks; "
                "use kernel='block_sparse', 'triton_block_sparse', 'sdpa', or 'auto'"
            )
        if self.kernel in {"block_sparse", "triton_block_sparse"} and not sparse_masked:
            raise ValueError(
                "sparse attention kernels require draft sparse attention or local_range"
            )
        if self.kernel in {"block_sparse", "triton_block_sparse"}:
            block_tokens = (
                self.sparse.window_size[0]
                * self.sparse.window_size[1]
                * self.sparse.window_size[2]
            )
            if block_tokens != 128:
                raise ValueError(
                    "kernel='block_sparse' requires sparse.window_size to "
                    f"contain 128 tokens, got {self.sparse.window_size} "
                    f"({block_tokens} tokens)"
                )


@dataclass
class LQProjectionConfig:
    enabled: bool = True
    num_injection_layers: int = 30
    input_channels: int = 3
    spatial_factor: int = 16
    hidden_dim: int = 2048
    hidden_dim_2: int = 3072
    zero_init: bool = True
    pad_last_frames: int = 0

    def validate(self, dit: DiTConfig) -> None:
        if self.num_injection_layers < 0:
            raise ValueError("num_injection_layers cannot be negative")
        if self.pad_last_frames < 0:
            raise ValueError("pad_last_frames cannot be negative")
        if self.enabled and self.num_injection_layers > dit.num_layers:
            raise ValueError("LQ injection layers cannot exceed DiT layers")


@dataclass
class BlockReplacementConfig:
    layers: tuple[int, ...] = ()

    def validate(self) -> None:
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("replacement.layers cannot contain duplicates")


@dataclass
class FlashVSRModelConfig:
    dit: DiTConfig = field(default_factory=DiTConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    lq_projection: LQProjectionConfig = field(default_factory=LQProjectionConfig)
    replacement: BlockReplacementConfig = field(default_factory=BlockReplacementConfig)

    def validate(self) -> None:
        self.dit.validate()
        self.attention.validate()
        self.lq_projection.validate(self.dit)
        self.replacement.validate()
        if self.replacement.layers and not self.attention.is_hybrid_sliding:
            raise ValueError("replacement layers require hybrid_sliding attention")
        invalid = [index for index in self.replacement.layers if not 0 <= index < self.dit.num_layers]
        if invalid:
            raise ValueError(f"Replacement layer indices out of range: {invalid}")
