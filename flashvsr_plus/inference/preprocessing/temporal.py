from __future__ import annotations

from dataclasses import dataclass

import torch

from .padding import pad_video_frames


@dataclass(frozen=True)
class TemporalExpansion:
    video: torch.Tensor
    source_frames: int
    effective_frames: int
    expanded_frames: int
    repeat_factor: int


def expand_for_temporal_compression(
    video: torch.Tensor,
    temporal_compression: int,
    *,
    model_compression: int = 4,
) -> TemporalExpansion:
    """Expand an input timeline onto the model's fixed temporal grid.

    The first frame remains unique. Every later frame is repeated
    ``model_compression // temporal_compression`` times.
    """
    if video.ndim != 5:
        raise ValueError("video must be [B,C,F,H,W]")
    if temporal_compression not in {1, 2, 4}:
        raise ValueError("temporal_compression must be one of 1, 2, or 4")
    if model_compression <= 0:
        raise ValueError("model_compression must be positive")
    if model_compression % temporal_compression:
        raise ValueError("model_compression must be divisible by temporal_compression")
    source_frames = video.shape[2]
    # Keep every requested source frame. The model still requires an aligned
    # temporal grid, so repeat the final frame only for model input and crop
    # the decoded result back to ``source_frames`` in the pipeline.
    video, _ = pad_video_frames(video, temporal_compression, mode="repeat")
    effective_frames = video.shape[2]
    repeat_factor = model_compression // temporal_compression
    if repeat_factor == 1 or effective_frames == 1:
        expanded = video
    else:
        expanded = torch.cat(
            [video[:, :, :1], video[:, :, 1:].repeat_interleave(repeat_factor, dim=2)],
            dim=2,
        )
    return TemporalExpansion(
        video=expanded,
        source_frames=source_frames,
        effective_frames=effective_frames,
        expanded_frames=expanded.shape[2],
        repeat_factor=repeat_factor,
    )


def _gaussian_weights(
    count: int,
    sigma: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(count, device=device, dtype=torch.float32)
    positions = positions - (count - 1) / 2
    weights = torch.exp(-0.5 * (positions / sigma).square())
    return weights / weights.sum()


def merge_temporal_output(
    video: torch.Tensor,
    expansion: TemporalExpansion,
    *,
    mode: str = "gaussian",
    gaussian_sigma: float = 1.0,
) -> torch.Tensor:
    """Collapse decoded duplicate groups back to the effective input length."""
    if video.ndim != 5:
        raise ValueError("video must be [B,C,F,H,W]")
    mode = mode.lower()
    if mode not in {"gaussian", "first"}:
        raise ValueError("temporal merge mode must be 'gaussian' or 'first'")
    if gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive")
    required = expansion.expanded_frames
    if video.shape[2] < required:
        raise ValueError(
            f"decoded video has {video.shape[2]} frames, expected at least {required}"
        )
    video = video[:, :, :required]
    factor = expansion.repeat_factor
    if factor == 1 or expansion.effective_frames == 1:
        return video[:, :, : expansion.effective_frames]

    first = video[:, :, :1]
    tail = video[:, :, 1:].reshape(
        video.shape[0],
        video.shape[1],
        expansion.effective_frames - 1,
        factor,
        video.shape[3],
        video.shape[4],
    )
    if mode == "first":
        merged = tail[:, :, :, 0]
    else:
        weights = _gaussian_weights(factor, gaussian_sigma, device=video.device)
        merged = (tail.float() * weights.view(1, 1, 1, factor, 1, 1)).sum(
            dim=3
        ).to(video.dtype)
    return torch.cat([first, merged], dim=2)
