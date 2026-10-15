from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


# These constants are deliberately tied to the released FlashVSR+ hybrid
# checkpoint and its 1536x2560 RTX 4090 reference run.  The policy is not a
# generic PyTorch memory planner; it is a conservative guard for this stream
# inference shape.
REFERENCE_SPATIAL_BLOCKS = 240
REFERENCE_NO_OFFLOAD_PEAK_MIB = 17_394.0
REFERENCE_OFFLOAD_PEAK_MIB = 14_792.0
REFERENCE_STATIC_ALLOCATED_MIB = 10_000.0
TOTAL_MEMORY_SAFETY_FACTOR = 0.90


@dataclass(frozen=True)
class StreamOffloadDecision:
    mode: str
    enabled: bool
    layers: tuple[int, ...]
    spatial_blocks: int
    free_memory_mib: float | None
    total_memory_mib: float | None
    estimated_no_offload_peak_mib: float
    estimated_offload_peak_mib: float
    reason: str

    def metadata(self) -> dict[str, Any]:
        return {
            "cache_offload_mode": self.mode,
            "cache_offload_enabled": self.enabled,
            "cache_offload_layers": list(self.layers),
            "cache_offload_spatial_blocks": self.spatial_blocks,
            "cache_offload_free_memory_mib": self.free_memory_mib,
            "cache_offload_total_memory_mib": self.total_memory_mib,
            "cache_offload_estimated_no_offload_peak_mib": (
                self.estimated_no_offload_peak_mib
            ),
            "cache_offload_estimated_offload_peak_mib": (
                self.estimated_offload_peak_mib
            ),
            "cache_offload_decision_reason": self.reason,
        }


def _memory_info(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    try:
        free, total = torch.cuda.mem_get_info(device)
    except RuntimeError:
        return None, None
    return free / 1024**2, total / 1024**2


def _original_layers(denoiser: torch.nn.Module, config) -> tuple[int, ...]:
    blocks = list(getattr(denoiser, "blocks", ()))
    if not blocks:
        return ()
    # SlidingLocalSelfAttention subclasses BlockSparseSelfAttention, so use
    # the concrete class name rather than isinstance on the base class.
    if getattr(config, "is_hybrid_sliding", False):
        return tuple(
            index
            for index, block in enumerate(blocks)
            if type(block.self_attn).__name__ != "SlidingLocalSelfAttention"
        )
    return tuple(range(len(blocks)))


def _estimate_peak(spatial_blocks: int) -> tuple[float, float]:
    # The persistent original-layer cache scales approximately linearly with
    # spatial blocks, while draft/workspace pressure grows more steeply.  A
    # squared ratio is intentionally conservative at the 4K boundary.
    ratio = max(spatial_blocks, 1) / REFERENCE_SPATIAL_BLOCKS
    estimated_no_offload = REFERENCE_OFFLOAD_PEAK_MIB + (
        REFERENCE_NO_OFFLOAD_PEAK_MIB - REFERENCE_OFFLOAD_PEAK_MIB
    ) * ratio * ratio
    estimated_offload = REFERENCE_OFFLOAD_PEAK_MIB * max(ratio, 1.0)
    return estimated_no_offload, estimated_offload


def choose_stream_cache_offload(
    config,
    denoiser: torch.nn.Module,
    video: torch.Tensor,
    device: torch.device,
) -> StreamOffloadDecision:
    """Choose cache residency for the current request without running a probe.

    Explicit layer indices remain a backwards-compatible force-on override,
    except when ``never`` is explicitly requested. ``never`` and ``always``
    are intended for debugging/benchmarking; normal inference uses ``auto``.
    """
    height, width = video.shape[-2:]
    spatial_blocks = (height // 16 // 8) * (width // 16 // 8)
    free_mib, total_mib = _memory_info(device)
    estimated_no, estimated_offload = _estimate_peak(spatial_blocks)
    original_layers = _original_layers(denoiser, config)
    is_base = not getattr(config, "is_hybrid_sliding", False)
    if is_base:
        # Conservative base estimate: retain the reference workspace floor,
        # then budget every original layer's full K/V history separately.
        # This is an estimate, not a calibrated base benchmark. Do not scale
        # the hybrid five-layer cache delta as though it represented 30 layers.
        cached_frames = int(config.sparse_kv_ratio) * config.sparse_window_size[0]
        tokens = video.shape[0] * (height // 16) * (width // 16) * cached_frames
        cache_mib = sum(
            2 * tokens * block.self_attn.q.out_features
            * block.self_attn.q.weight.element_size() / 1024**2
            for block in getattr(denoiser, "blocks", ())
        )
        estimated_no = max(estimated_no, estimated_offload + cache_mib)
    explicit_layers = tuple(getattr(config, "stream_cache_offload_layers", ()))
    mode = getattr(config, "stream_cache_offload_mode", "auto")

    if mode == "never":
        enabled = False
        reason = "forced never"
    elif explicit_layers:
        return StreamOffloadDecision(
            mode="always",
            enabled=True,
            layers=explicit_layers,
            spatial_blocks=spatial_blocks,
            free_memory_mib=free_mib,
            total_memory_mib=total_mib,
            estimated_no_offload_peak_mib=estimated_no,
            estimated_offload_peak_mib=estimated_offload,
            reason="explicit stream_cache_offload_layers",
        )
    elif mode == "always":
        enabled = True
        reason = "forced always"
    else:
        reasons = []
        if is_base and device.type == "cuda" and (free_mib is None or total_mib is None):
            reasons.append("base: VRAM information unavailable; conservatively offload KV")
        # Compare both residency plans with the current GPU capacity.  The
        # reference peak includes roughly REFERENCE_STATIC_ALLOCATED_MIB of
        # model/static allocations; replace that part with the actual current
        # process allocation when CUDA reports it.  This keeps auto policy
        # tied to the installed GPU instead of a resolution-only hard limit.
        capacity_mib = None if total_mib is None else total_mib * TOTAL_MEMORY_SAFETY_FACTOR
        if free_mib is not None and total_mib is not None:
            allocated_mib = total_mib - free_mib
            estimated_no = allocated_mib + max(
                estimated_no - REFERENCE_STATIC_ALLOCATED_MIB, 0.0
            )
            estimated_offload = allocated_mib + max(
                estimated_offload - REFERENCE_STATIC_ALLOCATED_MIB, 0.0
            )
        no_fits = capacity_mib is not None and estimated_no <= capacity_mib
        offload_fits = capacity_mib is not None and estimated_offload <= capacity_mib
        if capacity_mib is not None and not no_fits:
            reasons.append(
                f"estimated_no_offload_peak={estimated_no:.0f}MiB exceeds "
                f"available VRAM ({capacity_mib:.0f}MiB)"
            )
        # If the no-offload estimate does not fit, offload is the only plan
        # that can reduce persistent KV residency.  The extrapolated
        # offload estimate can be pessimistic at resolutions beyond the
        # reference run, so do not disable the only mitigation merely because
        # that estimate is also above the safety budget.
        if capacity_mib is None:
            enabled = bool(reasons)
        else:
            enabled = not no_fits
        if capacity_mib is None and not reasons:
            reason = "VRAM information unavailable; keep cache on GPU"
        elif not no_fits and not reasons:
            reason = "estimated peak fits available VRAM"
        else:
            reason = "; ".join(reasons)
            if not offload_fits:
                reason += (
                    f"; offload estimate also exceeds safety budget "
                    f"({estimated_offload:.0f}MiB), enabling as best effort"
                )

    return StreamOffloadDecision(
        mode=mode,
        enabled=enabled,
        layers=original_layers if enabled else (),
        spatial_blocks=spatial_blocks,
        free_memory_mib=free_mib,
        total_memory_mib=total_mib,
        estimated_no_offload_peak_mib=estimated_no,
        estimated_offload_peak_mib=estimated_offload,
        reason=reason,
    )


__all__ = ["StreamOffloadDecision", "choose_stream_cache_offload"]
