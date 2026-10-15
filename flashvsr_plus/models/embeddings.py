from __future__ import annotations

import math

import torch


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("Sinusoidal embedding dimension must be even")
    # FlashVSR 1.1/Wan computes this in float64 and casts back to the
    # timestep dtype. Keeping that detail matters for BF16 checkpoint parity.
    frequencies = torch.outer(
        position.to(torch.float64).flatten(),
        torch.pow(
            10000.0,
            -torch.arange(dim // 2, device=position.device, dtype=torch.float64)
            / (dim // 2),
        ),
    )
    return torch.cat([frequencies.cos(), frequencies.sin()], dim=-1).to(position.dtype)


def _axis_pair_counts(num_pairs: int) -> tuple[int, int, int]:
    base = num_pairs // 3
    return num_pairs - 2 * base, base, base


def rope_angles_3d(
    shape: tuple[int, int, int],
    head_dim: int,
    device: torch.device,
    theta: float = 10000.0,
    temporal_offset: int = 0,
) -> torch.Tensor:
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    frames, height, width = shape
    pair_counts = _axis_pair_counts(head_dim // 2)
    axes = (
        torch.arange(
            temporal_offset, temporal_offset + frames, device=device
        ).view(frames, 1, 1).expand(frames, height, width),
        torch.arange(height, device=device).view(1, height, 1).expand(frames, height, width),
        torch.arange(width, device=device).view(1, 1, width).expand(frames, height, width),
    )
    chunks = []
    for positions, pairs in zip(axes, pair_counts):
        inverse = theta ** (-torch.arange(pairs, device=device).float() / max(pairs, 1))
        chunks.append(positions.reshape(-1, 1).float() * inverse)
    return torch.cat(chunks, dim=-1)


def apply_rope_3d(
    x: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    temporal_offset: int = 0,
) -> torch.Tensor:
    angles = rope_angles_3d(
        shape, x.shape[-1], x.device, temporal_offset=temporal_offset
    ).to(dtype=torch.float64)
    x_float = x.to(torch.float64).reshape(*x.shape[:-1], -1, 2)
    cos = angles.cos().view(1, 1, angles.shape[0], -1)
    sin = angles.sin().view(1, 1, angles.shape[0], -1)
    first, second = x_float[..., 0], x_float[..., 1]
    rotated = torch.stack((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return rotated.flatten(-2).to(dtype=x.dtype)
