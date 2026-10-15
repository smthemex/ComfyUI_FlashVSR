from __future__ import annotations

import importlib
from functools import lru_cache

import torch
import torch.nn.functional as F


SUPPORTED_KERNELS = {
    "auto",
    "sdpa",
    "flash_attn2",
    "flash_attn3",
    "block_sparse",
    "triton_block_sparse",
}


@lru_cache(maxsize=None)
def _optional_module(name: str):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def attention_kernel_status() -> dict[str, bool]:
    return {
        "sdpa": True,
        "flash_attn2": _optional_module("flash_attn") is not None,
        "flash_attn3": _optional_module("flash_attn_interface") is not None,
        "block_sparse": _optional_module("block_sparse_attn") is not None,
        "triton_block_sparse": _optional_module("flashvsr_attn") is not None,
    }


def _require_kernel(name: str):
    module_name = {
        "flash_attn2": "flash_attn",
        "flash_attn3": "flash_attn_interface",
        "block_sparse": "block_sparse_attn",
    }[name]
    module = _optional_module(module_name)
    if module is None:
        raise ImportError(
            f"attention kernel {name!r} requires the optional module {module_name!r}"
        )
    return module


def _run_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel: str,
) -> torch.Tensor:
    module = _require_kernel(kernel)
    function = getattr(module, "flash_attn_func", None)
    if function is None:
        raise AttributeError(f"{module.__name__} does not expose flash_attn_func")
    # FlashAttention 2/3 both consume [B,S,H,D].
    output = function(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.transpose(1, 2)


def _expand_block_mask(
    block_mask: torch.Tensor,
    block_token_size: int,
    query_tokens: int,
    key_tokens: int,
) -> torch.Tensor:
    mask = block_mask.repeat_interleave(block_token_size, dim=-2)
    mask = mask.repeat_interleave(block_token_size, dim=-1)
    mask = mask[..., :query_tokens, :key_tokens].bool()
    empty = ~mask.any(dim=-1, keepdim=True)
    if empty.any():
        fallback = torch.zeros_like(mask)
        diagonal = torch.arange(min(query_tokens, key_tokens), device=mask.device)
        fallback[..., diagonal, diagonal] = True
        mask = torch.where(empty, fallback, mask)
    return mask


def _run_block_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
) -> torch.Tensor:
    module = _require_kernel("block_sparse")
    function = getattr(module, "block_sparse_attn_func", None)
    if function is None:
        raise AttributeError("block_sparse_attn does not expose block_sparse_attn_func")
    batch, heads, query_tokens, head_dim = q.shape
    key_tokens = k.shape[2]
    expected_query_blocks = (query_tokens + 127) // 128
    expected_key_blocks = (key_tokens + 127) // 128
    expected_shape = (batch, heads, expected_query_blocks, expected_key_blocks)
    if tuple(block_mask.shape) != expected_shape:
        raise ValueError(
            "block_sparse_attn requires a compact 128-token block mask with "
            f"shape {expected_shape}, got {tuple(block_mask.shape)}"
        )
    q_flat = q.transpose(1, 2).reshape(batch * query_tokens, heads, head_dim).contiguous()
    k_flat = k.transpose(1, 2).reshape(batch * key_tokens, heads, head_dim).contiguous()
    v_flat = v.transpose(1, 2).reshape(batch * key_tokens, heads, head_dim).contiguous()
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * query_tokens
    cu_k = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * key_tokens
    head_mask_type = torch.ones(heads, device=q.device, dtype=torch.int32)
    output = function(
        q_flat,
        k_flat,
        v_flat,
        cu_q,
        cu_k,
        head_mask_type,
        None,
        block_mask.contiguous(),
        query_tokens,
        key_tokens,
        0.0,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.reshape(batch, query_tokens, heads, head_dim).transpose(1, 2)


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel: str = "auto",
    token_mask: torch.Tensor | None = None,
    block_mask: torch.Tensor | None = None,
    block_token_size: int | None = None,
) -> torch.Tensor:
    """Run one explicitly selected attention implementation.

    Inputs and output use [B,H,S,D]. FlashAttention cannot consume arbitrary
    masks; block masks are routed to block-sparse attention or an exact SDPA
    expansion.
    """
    kernel = kernel.lower().replace("-", "_")
    aliases = {"fa2": "flash_attn2", "fa3": "flash_attn3"}
    aliases["triton_sparse"] = "triton_block_sparse"
    kernel = aliases.get(kernel, kernel)
    if kernel not in SUPPORTED_KERNELS:
        raise ValueError(f"Unsupported attention kernel: {kernel}")
    if token_mask is not None and block_mask is not None:
        raise ValueError("Provide token_mask or block_mask, not both")

    has_mask = token_mask is not None or block_mask is not None
    cuda_half = q.is_cuda and q.dtype in {torch.float16, torch.bfloat16}
    if kernel == "auto":
        if (
            block_mask is not None
            and block_token_size == 128
            and cuda_half
            and _optional_module("block_sparse_attn") is not None
        ):
            kernel = "block_sparse"
        elif has_mask:
            kernel = "sdpa"
        elif cuda_half and _optional_module("flash_attn_interface") is not None:
            kernel = "flash_attn3"
        elif cuda_half and _optional_module("flash_attn") is not None:
            kernel = "flash_attn2"
        else:
            kernel = "sdpa"

    if kernel in {"flash_attn2", "flash_attn3"}:
        if has_mask:
            raise ValueError(
                f"{kernel} supports only unmasked full attention in this API; "
                "use block_sparse or sdpa for causal/sparse masks"
            )
        if not cuda_half:
            raise ValueError(f"{kernel} requires CUDA FP16/BF16 tensors")
        return _run_flash_attention(q, k, v, kernel)

    if kernel == "block_sparse":
        if block_mask is None:
            raise ValueError("block_sparse requires a block_mask")
        if not cuda_half:
            raise ValueError("block_sparse requires CUDA FP16/BF16 tensors")
        if block_token_size != 128:
            raise ValueError(
                "block_sparse_attn requires 128 tokens per block; "
                f"the configured 3D window produces {block_token_size}"
            )
        # block_sparse_attn consumes the compact [B,H,Q_blocks,K_blocks]
        # layout directly. Each entry represents one 128-token window.
        return _run_block_sparse(q, k, v, block_mask)

    if kernel == "triton_block_sparse":
        raise ValueError(
            "triton_block_sparse accepts the unflattened stream window layout; "
            "call it through forward_stream()"
        )

    if block_mask is not None:
        if block_token_size is None:
            raise ValueError("block_token_size is required when SDPA expands a block mask")
        mask_entries = q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2]
        if mask_entries > 2_000_000_000:
            raise RuntimeError(
                "SDPA block-mask fallback would allocate too many mask entries; "
                "install block_sparse_attn or select a shorter sequence"
            )
        token_mask = _expand_block_mask(
            block_mask, block_token_size, q.shape[2], k.shape[2]
        )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=token_mask, dropout_p=0.0
    )
