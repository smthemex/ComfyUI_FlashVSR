"""Hardware capability detection and inference backend selection.

The selector only chooses implementations that are installed and supported by
the current device.  ``sdpa`` remains the correctness fallback for dense and
masked attention.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HardwareCapabilities:
    device: str
    device_name: str
    compute_capability: tuple[int, int] | None
    total_memory_gib: float
    cuda: bool
    bf16: bool
    flash_attn2: bool
    flash_attn3: bool
    block_sparse: bool
    triton_block_sparse: bool

    @property
    def sm(self) -> int | None:
        return None if self.compute_capability is None else self.compute_capability[0] * 10 + self.compute_capability[1]

    def supports(self, kernel: str) -> bool:
        return {
            "sdpa": True,
            "flash_attn2": self.cuda and self.flash_attn2,
            "flash_attn3": self.cuda and self.flash_attn3,
            "block_sparse": self.cuda and self.block_sparse,
            "triton_block_sparse": self.cuda and self.triton_block_sparse,
        }.get(kernel, False)


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def detect_hardware(device: str = "cuda:0") -> HardwareCapabilities:
    target = torch.device(device)
    cuda = target.type == "cuda" and torch.cuda.is_available()
    if not cuda:
        return HardwareCapabilities(
            device=str(target), device_name="CPU", compute_capability=None,
            total_memory_gib=0.0, cuda=False,
            bf16=False, flash_attn2=False, flash_attn3=False,
            block_sparse=False, triton_block_sparse=False,
        )
    index = target.index if target.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    capability = (props.major, props.minor)
    bf16 = bool(torch.cuda.is_bf16_supported(index))
    return HardwareCapabilities(
        device=str(target), device_name=props.name,
        compute_capability=capability,
        total_memory_gib=props.total_memory / (1024 ** 3), cuda=True,
        bf16=bf16,
        flash_attn2=_installed("flash_attn"),
        flash_attn3=_installed("flash_attn_interface"),
        block_sparse=_installed("block_sparse_attn"),
        triton_block_sparse=_installed("flashvsr_attn"),
    )


def select_kernels(
    capabilities: HardwareCapabilities,
    *,
    attention: str = "auto",
    sparse: bool = True,
) -> str:
    """Resolve user choices without silently selecting unavailable kernels."""
    if attention == "auto":
        if sparse:
            # Prefer the CUDA BSA backend on supported NVIDIA cards. The
            # in-tree Triton implementation remains the fallback sparse
            # backend when BSA is unavailable.
            attention = "block_sparse" if capabilities.supports("block_sparse") else (
                "triton_block_sparse" if capabilities.supports("triton_block_sparse") else "sdpa"
            )
        else:
            attention = "flash_attn3" if capabilities.supports("flash_attn3") else (
                "flash_attn2" if capabilities.supports("flash_attn2") else "sdpa"
            )
    if not capabilities.supports(attention):
        raise RuntimeError(
            f"attention kernel {attention!r} is unavailable on {capabilities.device_name}; "
            "install its optional package or use --attn-kernel auto"
        )
    return attention
