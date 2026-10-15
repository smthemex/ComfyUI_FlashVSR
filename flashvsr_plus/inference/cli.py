from __future__ import annotations

import argparse
import importlib

from .config import ColorCorrectionConfig, DecoderConfig, InferenceConfig, StreamConfig
from .console import print_banner, print_configuration, print_stage
from .loading import ComponentFactories, ModelPaths
from .pipeline import build_inference_pipeline
from .request import InferenceRequest


def _load_factory(value: str | None):
    if value is None:
        return None
    module_name, symbol = value.split(":", 1)
    return getattr(importlib.import_module(module_name), symbol)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FlashVSR+ stream video super-resolution inference", allow_abbrev=False,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dit", default=None)
    parser.add_argument("--student", default=None)
    parser.add_argument("--lq_projection", default=None)
    parser.add_argument("--context", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--tiny_decoder", default=None)
    parser.add_argument("--decoder", choices=["vae", "tiny"], default="tiny")
    parser.add_argument(
        "--decode-every", "--decode_every", default="auto",
        help="auto loads the full input; a positive multiple of 4 enables incremental reading and decoding",
    )
    parser.add_argument(
        "--tiny_decoder_parallel_mode",
        choices=["auto", "serial", "parallel"],
        default="auto",
        help="TinyDecoder mode: auto, serial, or full temporal parallel",
    )
    parser.add_argument(
        "--tiny_decoder_channels_last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use channels-last BF16 TinyDecoder convolutions (default: true)",
    )
    parser.add_argument("--is_sparse", action="store_true")
    parser.add_argument("--is_hybrid_sliding", action="store_true")
    parser.add_argument(
        "--attn_kernel",
        choices=[
            "auto",
            "sdpa",
            "block_sparse",
            "triton_block_sparse",
            "flash_attn2",
            "flash_attn3",
        ],
        default="auto",
    )
    parser.add_argument("--block_size", type=int, default=1)
    parser.add_argument("--sparse_topk", type=int, default=None)
    parser.add_argument("--sparse_topk_ratio", type=float, default=2.5)
    parser.add_argument("--sparse_local_blocks", type=int, default=None)
    parser.add_argument("--sparse_local_range", type=int, default=None)
    parser.add_argument(
        "--sparse_window_size",
        nargs=3,
        type=int,
        default=(2, 8, 8),
    )
    parser.add_argument(
        "--sparse_kv_ratio",
        type=float,
        default=3.0,
        help="stream KV temporal-group count; every group keeps the full spatial grid (default: 3.0)",
    )
    parser.add_argument(
        "--stream_cache_offload_layers",
        nargs="*",
        type=int,
        default=(),
        help="optional DiT layer indices whose stream KV cache is kept on CPU between chunks",
    )
    parser.add_argument(
        "--stream_cache_offload_mode",
        choices=["auto", "never", "always"],
        default="auto",
        help="stream KV residency policy: auto by resolution/VRAM, or force never/always",
    )
    parser.add_argument(
        "--stream_cache_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use pinned host buffers for stream KV offload (default: true)",
    )
    parser.add_argument("--replacement_layers", nargs="*", type=int, default=())
    parser.add_argument("--sliding_local_range", type=int, default=3)
    parser.add_argument("--sliding_temporal_block_size", type=int, default=2)
    parser.add_argument(
        "--sliding_temporal_mode",
        choices=["local", "causal", "short_causal"],
        default="local",
    )
    parser.add_argument(
        "--sliding_spatial_mode",
        choices=["local", "base"],
        default="local",
    )
    parser.add_argument("--sliding_cache_temporal_windows", type=int, default=1)
    parser.add_argument("--lq_projection_layers", type=int, default=1)
    parser.add_argument("--chunk_latents", type=int, default=4)
    # Accepted only so older launch scripts keep working. Stream scheduling is
    # now defined in latent space by --chunk_latents.
    parser.add_argument("--chunk_frames", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--temporal_compression",
        type=int,
        choices=[1, 2, 4],
        default=4,
        help="effective temporal compression exposed to the input/output timeline (default: 4)",
    )
    parser.add_argument(
        "--temporal_merge",
        choices=["gaussian", "first"],
        default="gaussian",
        help="how repeated decoded frames are collapsed for temporal compression 1 or 2",
    )
    parser.add_argument(
        "--temporal_gaussian_sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used by --temporal_merge gaussian",
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--color-correction",
        "--color_correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="match output low-frequency color to the LQ input (default: disabled)",
    )
    parser.add_argument(
        "--color-correction-levels", "--color_correction_levels", type=int, default=5
    )
    parser.add_argument(
        "--color-correction-chunk-size",
        "--color_correction_chunk_size",
        type=int,
        default=16,
    )
    parser.add_argument("--denoiser_factory", default=None, help="Denoiser factory in module:function format")
    parser.add_argument("--vae_factory", default=None, help="VAE factory in module:function format")
    parser.add_argument("--tiny_factory", default=None, help="Tiny decoder factory in module:function format")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig(
        is_sparse=args.is_sparse,
        is_hybrid_sliding=args.is_hybrid_sliding,
        attn_kernel=args.attn_kernel,
        block_size=args.block_size,
        sparse_topk=args.sparse_topk,
        sparse_topk_ratio=args.sparse_topk_ratio,
        sparse_local_blocks=args.sparse_local_blocks,
        sparse_local_range=args.sparse_local_range,
        sparse_window_size=tuple(args.sparse_window_size),
        sparse_kv_ratio=args.sparse_kv_ratio,
        stream_cache_offload_layers=tuple(args.stream_cache_offload_layers),
        stream_cache_offload_mode=args.stream_cache_offload_mode,
        stream_cache_pin_memory=args.stream_cache_pin_memory,
        replacement_layers=tuple(args.replacement_layers),
        sliding_local_range=args.sliding_local_range,
        sliding_temporal_block_size=args.sliding_temporal_block_size,
        sliding_temporal_mode=args.sliding_temporal_mode,
        sliding_spatial_mode=args.sliding_spatial_mode,
        sliding_cache_temporal_windows=args.sliding_cache_temporal_windows,
        lq_projection_layers=args.lq_projection_layers,
        decoder=DecoderConfig(
            kind=args.decoder,
            decode_every=args.decode_every,
            tiny_decoder_parallel_mode=args.tiny_decoder_parallel_mode,
            tiny_decoder_channels_last=args.tiny_decoder_channels_last,
        ),
        color_correction=ColorCorrectionConfig(
            enabled=args.color_correction,
            levels=args.color_correction_levels,
            chunk_size=args.color_correction_chunk_size,
        ),
        streaming=StreamConfig(args.chunk_latents),
        temporal_compression=args.temporal_compression,
        temporal_merge=args.temporal_merge,
        temporal_gaussian_sigma=args.temporal_gaussian_sigma,
        num_inference_steps=args.steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )
    config.validate()
    paths = ModelPaths(
        dit=args.dit,
        student=args.student,
        lq_projection=args.lq_projection,
        context=args.context,
        vae=args.vae,
        tiny_decoder=args.tiny_decoder,
    )
    factories = ComponentFactories(
        denoiser=_load_factory(args.denoiser_factory),
        vae=_load_factory(args.vae_factory),
        tiny_decoder=_load_factory(args.tiny_factory),
    )
    print_banner()
    print_configuration(config, args)
    print_stage("Loading model weights...")
    pipeline = build_inference_pipeline(config, paths, factories=factories)
    print_stage("Model loading complete", done=True)
    print_stage(f"Starting inference: {args.input}")
    result = pipeline.infer(
        InferenceRequest(
            input_path=args.input,
            output_path=args.output,
            height=args.height,
            width=args.width,
            max_frames=args.max_frames,
            fps=args.fps,
        )
    )
    print_stage(
        f"Inference complete · {result.frames} frames · Saved to {result.output_path}",
        done=True,
    )


if __name__ == "__main__":
    main()
