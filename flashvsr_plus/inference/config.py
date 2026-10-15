from __future__ import annotations

from dataclasses import dataclass, field

from ..models import (
    AttentionConfig,
    CausalAttentionConfig,
    SlidingAttentionConfig,
    SparseAttentionConfig,
)
from .randomness import DEFAULT_INFERENCE_SEED

TEMPORAL_DOWNSAMPLE = 4
INPUT_MULTIPLE = 128


# Uniformly spaced b=25 hybrid-sliding schedule. The explicit layer list keeps
# the periodic fallback parameters from changing the released inference layout.
HYBRID_B25_REPLACEMENT_LAYERS = (
    1, 2, 3, 4, 5,
    7, 8, 9, 10, 11,
    13, 14, 15, 16, 17,
    19, 20, 21, 22, 23,
    25, 26, 27, 28, 29,
)
HYBRID_B25_ORIGINAL_LAYERS = (0, 6, 12, 18, 24)


@dataclass
class StreamConfig:
    chunk_latents: int = 4

    def validate(self) -> None:
        if type(self.chunk_latents) is not int or self.chunk_latents != 4:
            raise ValueError("FlashVSR+ stream inference currently requires chunk_latents=4")


@dataclass
class DecoderConfig:
    kind: str = "tiny"
    tiled: bool = True
    tile_size: tuple[int, int] = (30, 52)
    tile_stride: tuple[int, int] = (15, 26)
    tiny_channels: tuple[int, ...] = (512, 256, 128, 128)
    tiny_latent_channels: int = 784
    tiny_decoder_parallel_mode: str = "auto"
    # Use NHWC storage for BF16 CUDA convolutions. This is mathematically
    # equivalent and can be disabled for strict contiguous-layout regression.
    tiny_decoder_channels_last: bool = True
    decode_every: int | str = "auto"
    # Deferred decoding: the stream runner stops after denoising and returns
    # latents plus the aligned LQ conditioning video instead of running the
    # TinyDecoder. Used by the ComfyUI node split (KSampler -> Decoder).
    deferred: bool = False

    def validate(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in {"vae", "tiny"}:
            raise ValueError(f"Unknown decoder: {self.kind}")
        if any(value <= 0 for value in self.tile_size + self.tile_stride):
            raise ValueError("decoder tile sizes and strides must be positive")
        if len(self.tiny_channels) != 4 or any(value <= 0 for value in self.tiny_channels):
            raise ValueError("tiny_channels must contain four positive values")
        if isinstance(self.decode_every, str):
            self.decode_every = self.decode_every.lower()
            if self.decode_every != "auto":
                try:
                    self.decode_every = int(self.decode_every)
                except ValueError as error:
                    raise ValueError("decoder.decode_every must be auto or a positive integer") from error
        if self.decode_every != "auto" and (type(self.decode_every) is not int or self.decode_every <= 0):
            raise ValueError("decoder.decode_every must be positive")
        if self.decode_every != "auto" and self.decode_every % 4:
            raise ValueError("decoder.decode_every must be a multiple of the fixed 4-latent stream chunk")
        if self.kind == "vae" and self.decode_every != "auto":
            raise ValueError("numeric decode_every requires the tiny decoder")
        if not isinstance(self.deferred, bool):
            raise TypeError("decoder.deferred must be a boolean")
        if self.deferred:
            if self.kind != "tiny":
                raise ValueError("deferred decoding requires the tiny decoder")
            if self.decode_every != "auto":
                raise ValueError("deferred decoding requires decode_every='auto'")
        if not isinstance(self.tiny_decoder_channels_last, bool):
            raise TypeError("decoder.tiny_decoder_channels_last must be a boolean")
        self.tiny_decoder_parallel_mode = self.tiny_decoder_parallel_mode.lower()
        if self.tiny_decoder_parallel_mode not in {"auto", "serial", "parallel"}:
            raise ValueError(
                "decoder.tiny_decoder_parallel_mode must be one of "
                "'auto', 'serial', or 'parallel'"
            )


@dataclass
class ColorCorrectionConfig:
    enabled: bool = False
    levels: int = 5
    chunk_size: int = 16

    def validate(self) -> None:
        if self.levels <= 0:
            raise ValueError("color correction levels must be positive")
        if self.chunk_size <= 0:
            raise ValueError("color correction chunk_size must be positive")


@dataclass
class InferenceConfig:
    is_causal: bool = True
    is_sparse: bool = False
    is_hybrid_sliding: bool = False
    attn_kernel: str = "auto"
    block_size: int = 1
    sparse_topk: int | None = None
    sparse_topk_ratio: float | None = 2.5
    sparse_local_blocks: int | None = None
    sparse_local_range: int | None = None
    sparse_window_size: tuple[int, int, int] = (2, 8, 8)
    sparse_kv_ratio: float = 3.0
    replacement_layers: tuple[int, ...] = ()
    sliding_local_range: int = 3
    sliding_temporal_block_size: int = 2
    sliding_temporal_mode: str = "local"
    sliding_spatial_mode: str = "local"
    sliding_cache_temporal_windows: int = 1
    lq_projection_layers: int = 1
    decoder: DecoderConfig | str = field(default_factory=DecoderConfig)
    color_correction: ColorCorrectionConfig | dict[str, object] = field(
        default_factory=ColorCorrectionConfig
    )
    streaming: StreamConfig = field(default_factory=StreamConfig)
    num_inference_steps: int = 1
    sigma_shift: float = 5.0
    seed: int = DEFAULT_INFERENCE_SEED
    dtype: str = "bf16"
    device: str = "cuda:0"
    # Fixed released-model geometry; these are not architecture overrides.
    latent_channels: int = 16
    spatial_downsample: int = 8
    temporal_compression: int = 4
    temporal_merge: str = "gaussian"
    temporal_gaussian_sigma: float = 1.0
    pad_mode: str = "repeat"
    # Optional diagnostics for stream-memory profiling. Disabled by default so
    # production inference does not retain per-layer records.
    profile_memory: bool = False
    # Optional stream-only FFN activation chunking. A value such as 32768
    # bounds the large [tokens, ffn_dim] intermediate without changing math.
    stream_ffn_chunk_tokens: int | None = None
    # Triton query tile size for the narrow stream kernel. 128 is faster on
    # Ada for the fixed 128-token sparse block; 64 remains available as a
    # compatibility/ablation path.
    stream_block_m: int = 128
    # Optional stream-only KV offload. Layer indices listed here keep their
    # detached K/V cache in host memory between chunks and copy it back before
    # that layer runs. Empty lets stream_cache_offload_mode choose the layers.
    stream_cache_offload_layers: tuple[int, ...] = ()
    # Cache residency policy for stream inference. ``auto`` selects offload
    # from the request resolution and available GPU capacity; ``never`` and
    # ``always`` are deterministic overrides useful for benchmarking/debugging.
    stream_cache_offload_mode: str = "auto"
    # Use page-locked host buffers for offloaded KV. This is substantially
    # faster for CUDA D2H/H2D transfers; disable only on systems that cannot
    # reserve enough pinned host memory.
    stream_cache_pin_memory: bool = True

    def validate(self) -> None:
        if isinstance(self.decoder, str):
            self.decoder = DecoderConfig(kind=self.decoder)
        self.decoder.validate()
        if isinstance(self.color_correction, dict):
            self.color_correction = ColorCorrectionConfig(**self.color_correction)
        self.color_correction.validate()
        self.streaming.validate()
        self.replacement_layers = tuple(int(layer) for layer in self.replacement_layers)
        if any(layer < 0 for layer in self.replacement_layers):
            raise ValueError("replacement_layers cannot contain negative indices")
        if self.replacement_layers and not self.is_hybrid_sliding:
            raise ValueError("replacement_layers require hybrid_sliding attention")
        if self.lq_projection_layers <= 0:
            raise ValueError("lq_projection_layers must be positive")
        if not self.is_causal:
            raise ValueError("stream inference requires is_causal=True")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        temporal_windows = self.streaming.chunk_latents // self.sparse_window_size[0]
        if self.streaming.chunk_latents % self.sparse_window_size[0]:
            raise ValueError(
                "stream chunk_latents must be divisible by sparse temporal window size"
            )
        if temporal_windows % self.block_size:
            raise ValueError(
                "stream block_size must divide the temporal windows in one chunk"
            )
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.num_inference_steps > 1:
            raise NotImplementedError(
                "Multi-step stream inference is not implemented yet. "
                "Set num_inference_steps=1."
            )
        if self.sigma_shift <= 0:
            raise ValueError("sigma_shift must be positive")
        if self.sparse_kv_ratio <= 0:
            raise ValueError("sparse_kv_ratio must be positive")
        if self.stream_ffn_chunk_tokens is not None and self.stream_ffn_chunk_tokens <= 0:
            raise ValueError("stream_ffn_chunk_tokens must be positive when set")
        if self.stream_block_m not in {64, 128}:
            raise ValueError("stream_block_m must be 64 or 128")
        self.stream_cache_offload_layers = tuple(
            int(layer) for layer in self.stream_cache_offload_layers
        )
        if any(layer < 0 for layer in self.stream_cache_offload_layers):
            raise ValueError("stream_cache_offload_layers cannot contain negative indices")
        if len(set(self.stream_cache_offload_layers)) != len(self.stream_cache_offload_layers):
            raise ValueError("stream_cache_offload_layers cannot contain duplicates")
        self.stream_cache_offload_mode = self.stream_cache_offload_mode.lower()
        if self.stream_cache_offload_mode not in {"auto", "never", "always"}:
            raise ValueError(
                "stream_cache_offload_mode must be 'auto', 'never', or 'always'"
            )
        for name, required in (
            ("latent_channels", 16),
            ("spatial_downsample", 8),
        ):
            value = getattr(self, name)
            if type(value) is not int or value != required:
                raise ValueError(f"FlashVSR+ inference requires {name}={required}")
        if type(self.temporal_compression) is not int or self.temporal_compression not in {1, 2, 4}:
            raise ValueError("temporal_compression must be one of 1, 2, or 4")
        self.temporal_merge = self.temporal_merge.lower()
        if self.temporal_merge not in {"gaussian", "first"}:
            raise ValueError("temporal_merge must be 'gaussian' or 'first'")
        if self.temporal_gaussian_sigma <= 0:
            raise ValueError("temporal_gaussian_sigma must be positive")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.pad_mode != "repeat":
            raise ValueError("FlashVSR+ inference requires pad_mode='repeat'")
        attention = self.attention_config()
        attention.validate()
        self.attn_kernel = attention.kernel

    def attention_config(self) -> AttentionConfig:
        backend = "hybrid_sliding" if self.is_hybrid_sliding else (
            "sparse" if self.is_sparse or self.sparse_local_range is not None else "dense"
        )
        return AttentionConfig(
            backend=backend,
            kernel=self.attn_kernel,
            # Streaming is block-causal: forward_stream only exposes cached
            # history and the current block, never a future chunk.
            is_causal=self.is_causal,
            is_sparse=self.is_sparse,
            is_hybrid_sliding=self.is_hybrid_sliding,
            causal=CausalAttentionConfig(block_size_frames=self.block_size),
            sparse=SparseAttentionConfig(
                window_size=tuple(self.sparse_window_size),
                topk=self.sparse_topk,
                topk_ratio=self.sparse_topk_ratio,
                local_blocks=self.sparse_local_blocks,
                local_range=self.sparse_local_range,
            ),
            sliding=SlidingAttentionConfig(
                local_range=self.sliding_local_range,
                temporal_block_size=self.sliding_temporal_block_size,
                temporal_mode=self.sliding_temporal_mode,
                spatial_mode=self.sliding_spatial_mode,
                cache_temporal_windows=self.sliding_cache_temporal_windows,
            ),
        )
