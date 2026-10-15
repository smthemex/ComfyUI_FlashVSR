"""Minimal standalone invocation of the stream backend."""

import torch

from flashvsr_attn import stream_block_sparse_attention


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    batch, heads, query_blocks, key_blocks = 1, 24, 32, 96
    q = torch.randn(
        batch, heads, query_blocks, 128, 128, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, heads, key_blocks, 128, 128, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    block_mask = torch.rand(
        batch, heads, query_blocks, key_blocks, device="cuda"
    ) < 0.25
    with torch.inference_mode():
        output = stream_block_sparse_attention(q, k, v, block_mask)
    print(output.shape, output.dtype)


if __name__ == "__main__":
    main()
