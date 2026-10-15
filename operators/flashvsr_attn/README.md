# flashvsr-attn

Triton block-sparse attention for FlashVSR+ streaming inference. This package
provides the optional `triton_block_sparse` backend.

The kernel accepts FP16 or BF16 CUDA tensors, with 128 tokens per block and
128 channels per head. It supports a block mask of shape
`[batch, heads, query_blocks, key_blocks]`. There is no backward implementation.

## Install

Install PyTorch, Triton, and FlashAttention as described in the
[FlashVSR+ setup guide](../../README.md#installation), then run from the
repository root:

```bash
pip install -e operators/flashvsr_attn --no-build-isolation --no-deps
```

## API

```python
import torch
from flashvsr_attn import stream_block_sparse_attention

with torch.inference_mode():
    output = stream_block_sparse_attention(q_windows, k_windows, v_windows, block_mask)
```

`q_windows` has shape `[B, H, query_blocks, 128, 128]`; `k_windows` and
`v_windows` have shape `[B, H, key_blocks, 128, 128]`. The output has the same
shape as `q_windows`. The default tile sizes are `block_m=128` and
`block_n=128`.

## Tests and benchmark

From `operators/flashvsr_attn`:

```bash
pip install -e '.[test]' --no-build-isolation
pytest -q
python benchmarks/benchmark_forward.py --heads 24 --query-blocks 32 --key-blocks 96 --density 0.25 --dense-baseline
```

Resolution-based runs:

```bash
python benchmarks/benchmark_forward.py --heads 24 --input-height 1024 --input-width 1920 \
  --density 0.50 --warmup 5 --iters 20 --dense-baseline

python benchmarks/benchmark_forward.py --heads 24 --input-height 2048 --input-width 3840 \
  --density 0.10 --warmup 2 --iters 5
```

Measured on an RTX 4090 with BF16 and 24 heads. Times are medians after JIT
compilation; the masks were generated at the listed densities.

| Approx. input | Query / key blocks | Mask density | Triton median | Peak allocated |
|---|---:|---:|---:|---:|
| 1024×1920 | 240 / 600 | 50% | 164.0 ms | 1.25 GiB |
| 2048×3840 | 960 / 2400 | 10% | 529.9 ms | 5.71 GiB |
| 2048×3840 | 960 / 2400 | 19% | 992.6 ms | 5.71 GiB |
| 2048×3840 | 960 / 2400 | 50% | 2,562.8 ms | 5.71 GiB |
