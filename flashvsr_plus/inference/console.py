from __future__ import annotations

import os
import sys
from pathlib import Path

from .config import InferenceConfig


_BANNER = """
███████╗██╗      █████╗ ███████╗██╗  ██╗██╗   ██╗███████╗██████╗      ██╗
██╔════╝██║     ██╔══██╗██╔════╝██║  ██║██║   ██║██╔════╝██╔══██╗     ██║
█████╗  ██║     ███████║███████╗███████║██║   ██║███████╗██████╔╝  ████████╗
██╔══╝  ██║     ██╔══██║╚════██║██╔══██║╚██╗ ██╔╝╚════██║██╔══██╗  ╚══██╔══╝
██║     ███████╗██║  ██║███████║██║  ██║ ╚████╔╝ ███████║██║  ██║     ██║
╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝     ╚═╝
""".strip("\n")


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _supports_color() else text


def _path_label(value: str | None) -> str:
    if not value:
        return "Not specified"
    path = Path(value)
    parts = path.parts
    if len(parts) <= 4:
        return str(path)
    return str(Path("…", *parts[-3:]))


def print_banner() -> None:
    gradient = ("1;36", "1;36", "1;34", "1;34", "1;35", "1;35")
    for line, color in zip(_BANNER.splitlines(), gradient):
        print(_paint(line, color))
    print(_paint("                 STRUCTURED VIDEO SUPER-RESOLUTION", "1;37"))
    print()


def print_configuration(config: InferenceConfig, args) -> None:
    mode = f"STREAM · {config.streaming.chunk_latents}-latent chunks"
    attention = config.attention_config()
    decoder = "Tiny Decoder" if config.decoder.kind == "tiny" else "VAE Decoder"
    sliding_range = (
        config.sliding_local_range
        if config.sliding_spatial_mode == "local"
        else config.sparse_local_range
    )
    rows = (
        ("Mode", mode),
        ("Attention", f"{attention.backend} / {attention.kernel}"),
        (
            "Sparse Top-k",
            str(config.sparse_topk or config.sparse_topk_ratio)
            if config.is_sparse
            else "Disabled",
        ),
        ("Local Range", str(config.sparse_local_range or "Disabled")),
        (
            "Hybrid Sliding",
            (
                "range "
                f"{sliding_range} / "
                f"spatial {config.sliding_spatial_mode} / "
                f"temporal {config.sliding_temporal_mode} / "
                f"cache {config.sliding_cache_temporal_windows} window(s) / "
                f"{config.sliding_temporal_block_size} temporal windows"
                if config.is_hybrid_sliding
                else "Disabled"
            ),
        ),
        ("KV Time Groups", str(int(config.sparse_kv_ratio))),
        (
            "KV Cache Policy",
            config.stream_cache_offload_mode,
        ),
        ("Decoder", decoder),
        (
            "Tiny Temporal Mode",
            (
                (f"{config.decoder.tiny_decoder_parallel_mode} · adaptive"
                 if config.decoder.tiny_decoder_parallel_mode == "auto"
                 else config.decoder.tiny_decoder_parallel_mode)
                + (
                    " · channels-last"
                    if config.decoder.tiny_decoder_channels_last
                    else " · contiguous"
                )
            )
            if config.decoder.kind == "tiny"
            else "N/A",
        ),
        ("Temporal Compression", f"{config.temporal_compression}×"),
        (
            "Color Correction",
            (
                f"Wavelet · {config.color_correction.levels} levels"
                if config.color_correction.enabled
                else "Disabled"
            ),
        ),
        (
            "Temporal Merge",
            config.temporal_merge if config.temporal_compression < 4 else "N/A",
        ),
        ("Precision / Device", f"{config.dtype} / {config.device}"),
        ("Output Size", f"{args.width or 'Original'} × {args.height or 'Original'}"),
        ("Maximum Frames", str(args.max_frames or "All")),
        ("Steps / Seed", f"{config.num_inference_steps} / {config.seed}"),
        ("Sigma Shift", str(config.sigma_shift)),
    )
    print(_paint("┌─ Runtime Configuration", "1;35"))
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"│  {label:<{width}}  {_paint(value, '1;37')}")
    print(_paint("└────────────────────────────────────────────────────", "2;35"))
    print()

    components = (
        ("DiT Model", args.dit),
        ("LQ Projection", args.lq_projection),
        ("Text Context", args.context),
        (
            "Tiny Decoder" if config.decoder.kind == "tiny" else "VAE Decoder",
            args.tiny_decoder if config.decoder.kind == "tiny" else args.vae,
        ),
    )
    print(_paint("Model Components", "1;35"))
    for name, path in components:
        print(f"  {_paint('◆', '36')} {name:<16} {_path_label(path)}")
    print()


def print_stage(message: str, *, done: bool = False) -> None:
    marker = _paint("✓", "1;32") if done else _paint("●", "1;36")
    print(f"{marker} {message}", flush=True)
