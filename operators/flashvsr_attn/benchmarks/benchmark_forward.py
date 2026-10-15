"""Benchmark the production Triton path against an eager reference.

Example:
  conda run -n flashvsr python benchmarks/benchmark_forward.py --heads 24
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch
import torch.nn.functional as F

from flashvsr_attn.reference import reference_block_sparse_attention
from flashvsr_attn.triton_kernel import stream_block_sparse_attention


def measure(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument(
        "--query-blocks",
        type=int,
        default=None,
        help="number of query blocks; defaults to 32 unless --input-height/width is used",
    )
    parser.add_argument(
        "--key-blocks",
        type=int,
        default=None,
        help="number of key blocks; defaults to 96 unless --input-height/width is used",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=None,
        help="FlashVSR input height; with --input-width, derive block counts (window=2x8x8)",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=None,
        help="FlashVSR input width; with --input-height, derive block counts",
    )
    parser.add_argument(
        "--query-temporal-windows",
        type=int,
        default=2,
        help="temporal query windows used when deriving counts from resolution",
    )
    parser.add_argument(
        "--key-temporal-windows",
        type=int,
        default=5,
        help="current+cache temporal windows used when deriving counts from resolution",
    )
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--block-m", type=int, choices=(64, 128), default=64,
        help="query rows per Triton program (128 tests one program per sparse block)",
    )
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=None)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--block-n", type=int, choices=(64, 128), default=64)
    parser.add_argument(
        "--measure-reference",
        action="store_true",
        help="also measure the eager reference; only use this for small shapes",
    )
    parser.add_argument(
        "--dense-baseline",
        action="store_true",
        help="measure unmasked dense PyTorch attention for a rough upper baseline",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    # FlashVSR's latent patch and sparse window make one spatial block cover
    # 128x128 input pixels.  The default temporal counts match a four-frame
    # stream with three cached groups (2 query windows, 5 key windows).
    if (args.input_height is None) != (args.input_width is None):
        raise SystemExit("--input-height and --input-width must be provided together")
    if args.input_height is not None:
        if args.input_height <= 0 or args.input_width <= 0:
            raise SystemExit("input resolution must be positive")
        if args.input_height % 128 or args.input_width % 128:
            raise SystemExit(
                "input resolution must be divisible by 128 for the FlashVSR window layout"
            )
        if args.query_temporal_windows <= 0 or args.key_temporal_windows <= 0:
            raise SystemExit("temporal window counts must be positive")
        spatial_blocks = (args.input_height // 128) * (args.input_width // 128)
        if args.query_blocks is None:
            args.query_blocks = args.query_temporal_windows * spatial_blocks
        if args.key_blocks is None:
            args.key_blocks = args.key_temporal_windows * spatial_blocks
    args.query_blocks = 32 if args.query_blocks is None else args.query_blocks
    args.key_blocks = 96 if args.key_blocks is None else args.key_blocks
    if args.query_blocks <= 0 or args.key_blocks <= 0:
        raise SystemExit("query/key block counts must be positive")
    if not 0.0 < args.density <= 1.0:
        raise SystemExit("density must be in (0, 1]")

    shape = (args.batch, args.heads, args.query_blocks, 128, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        (args.batch, args.heads, args.key_blocks, 128, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    mask = torch.rand(
        (args.batch, args.heads, args.query_blocks, args.key_blocks), device="cuda"
    ) < args.density
    # Match the stream scheduler's invariant: every query block has at least
    # one valid key.  This lets the benchmark measure the production path,
    # which skips the defensive empty-row copy.
    mask[..., :, 0] = True

    # Compile and warm the Triton specialization before measuring it. Peak
    # memory includes the packed block index and helps with long sequences.
    torch.cuda.reset_peak_memory_stats()
    triton_ms = measure(
        lambda: stream_block_sparse_attention(
            q, k, v, mask, ensure_nonempty=False, block_m=args.block_m,
            block_n=args.block_n,
            num_warps=args.num_warps, num_stages=args.num_stages,
        ),
        args.warmup,
        args.iters,
    )
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"shape: q={tuple(q.shape)}, k={tuple(k.shape)}, density={args.density:.3f}")
    ordered_triton = sorted(triton_ms)
    p90_index = min(len(ordered_triton) - 1, max(0, math.ceil(0.90 * len(ordered_triton)) - 1))
    print(
        "triton ms: median={:.3f}, p90={:.3f}".format(
            statistics.median(triton_ms), ordered_triton[p90_index]
        )
    )
    print(
        "triton peak allocated: {:.2f} GiB".format(
            torch.cuda.max_memory_allocated() / (1024**3)
        )
    )
    if args.dense_baseline:
        q_dense = q.reshape(args.batch, args.heads, args.query_blocks * 128, 128)
        k_dense = k.reshape(args.batch, args.heads, args.key_blocks * 128, 128)
        torch.cuda.reset_peak_memory_stats()
        dense_ms = measure(
            lambda: F.scaled_dot_product_attention(
                q_dense, k_dense, v.reshape(args.batch, args.heads, args.key_blocks * 128, 128),
                dropout_p=0.0,
            ),
            args.warmup,
            args.iters,
        )
        dense_median = statistics.median(dense_ms)
        print(
            "dense SDPA ms: median={:.3f}, triton speedup={:.2f}x".format(
                dense_median, dense_median / statistics.median(triton_ms)
            )
        )
        print(
            "dense peak allocated: {:.2f} GiB".format(
                torch.cuda.max_memory_allocated() / (1024**3)
            )
        )
    if args.measure_reference:
        if args.query_blocks > 8 or args.key_blocks > 8 or args.heads > 4:
            raise SystemExit(
                "--measure-reference is intentionally limited to "
                "query/key blocks <= 8 and heads <= 4"
            )
        ref_ms = measure(
            lambda: reference_block_sparse_attention(q, k, v, mask),
            1,
            min(args.iters, 5),
        )
        print("reference ms: median={:.3f}".format(statistics.median(ref_ms)))


if __name__ == "__main__":
    main()
