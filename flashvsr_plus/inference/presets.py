from __future__ import annotations

from pathlib import Path

from .config import (
    HYBRID_B25_ORIGINAL_LAYERS,
    HYBRID_B25_REPLACEMENT_LAYERS,
    DecoderConfig,
    ColorCorrectionConfig,
    InferenceConfig,
    StreamConfig,
)
from .loading import ModelPaths
from .hardware import detect_hardware, select_kernels


INFERENCE_MODES = ("turbo", "base", "v1")


def validate_inference_mode(mode: str) -> str:
    value = mode.lower()
    if value not in INFERENCE_MODES:
        raise ValueError(f"inference mode must be one of {INFERENCE_MODES}; got {mode!r}")
    return value


def checkpoint_dir(root: Path, mode: str) -> Path:
    return root / "ckpts" / validate_inference_mode(mode)


def model_paths_for_mode(root: Path, mode: str, ckpt_dir: str | Path | None = None) -> ModelPaths:
    directory = Path(ckpt_dir).expanduser() if ckpt_dir is not None else checkpoint_dir(root, mode)
    return ModelPaths(
        dit=str(directory / "dit.safetensors"),
        lq_projection=str(directory / "lq_proj_in.safetensors"),
        context=str(root / "prompt_tensor/posi_prompt.pth"),
        tiny_decoder=str(root / "ckpts/TCDecoder.ckpt"),
    )


def stream_config_for_mode(
    mode: str,
    *,
    device: str = "cuda:0",
    seed: int = 1,
    decode_every: int | str = "auto",
    tiny_decoder_mode: str = "auto",
    tiny_decoder_channels_last: bool = True,
    offload_original_cache: bool = False,
    cache_offload_mode: str = "auto",
    temporal_compression: int = 4,
    temporal_merge: str = "gaussian",
    temporal_gaussian_sigma: float = 1.0,
    attn_kernel: str = "auto",
    hardware_profile: str = "auto",
    color_correction: bool = False,
    color_correction_levels: int = 5,
    color_correction_chunk_size: int = 16,
) -> InferenceConfig:
    mode = validate_inference_mode(mode)
    use_turbo = mode == "turbo"
    if hardware_profile not in {"auto", "generic"}:
        raise ValueError("hardware_profile must be 'auto' or 'generic'")
    if hardware_profile == "generic":
        selected_attn = "sdpa"
    else:
        selected_attn = select_kernels(
            detect_hardware(device),
            attention=attn_kernel,
            sparse=True,
        )
    return InferenceConfig(
        is_causal=True,
        block_size=2,
        is_sparse=True,
        sparse_topk_ratio=2.5,
        sparse_local_range=11,
        sparse_window_size=(2, 8, 8),
        is_hybrid_sliding=use_turbo,
        attn_kernel=selected_attn,
        replacement_layers=HYBRID_B25_REPLACEMENT_LAYERS if use_turbo else (),
        sliding_local_range=3,
        sliding_spatial_mode="local",
        sliding_temporal_block_size=2,
        sliding_temporal_mode="short_causal",
        sliding_cache_temporal_windows=1,
        stream_cache_offload_layers=(
            HYBRID_B25_ORIGINAL_LAYERS
            if use_turbo and offload_original_cache
            else ()
        ),
        stream_cache_offload_mode=(
            "always" if use_turbo and offload_original_cache else cache_offload_mode
        ),
        decoder=DecoderConfig(
            kind="tiny",
            tiny_decoder_parallel_mode=tiny_decoder_mode,
            tiny_decoder_channels_last=tiny_decoder_channels_last,
            decode_every=decode_every,
        ),
        color_correction=ColorCorrectionConfig(
            enabled=color_correction,
            levels=color_correction_levels,
            chunk_size=color_correction_chunk_size,
        ),
        streaming=StreamConfig(chunk_latents=4),
        temporal_compression=temporal_compression,
        temporal_merge=temporal_merge,
        temporal_gaussian_sigma=temporal_gaussian_sigma,
        num_inference_steps=1,
        sigma_shift=5.0,
        seed=seed,
        dtype="bf16",
        device=device,
    )
