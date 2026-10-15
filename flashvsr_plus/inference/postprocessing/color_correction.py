from __future__ import annotations

import torch
import torch.nn.functional as F


_GAUSSIAN_3X3 = (
    (0.0625, 0.125, 0.0625),
    (0.125, 0.25, 0.125),
    (0.0625, 0.125, 0.0625),
)


def _wavelet_blur(value: torch.Tensor, radius: int) -> torch.Tensor:
    channels = value.shape[1]
    kernel = value.new_tensor(_GAUSSIAN_3X3).view(1, 1, 3, 3)
    weight = kernel.expand(channels, 1, 3, 3)
    padded = F.pad(value, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(
        padded,
        weight,
        stride=1,
        padding=0,
        dilation=radius,
        groups=channels,
    )


def _wavelet_low_frequency(value: torch.Tensor, levels: int) -> torch.Tensor:
    low = value
    for level in range(levels):
        low = _wavelet_blur(low, 2**level)
    return low


class WaveletColorCorrector:
    """Match generated low-frequency color to LQ while retaining output detail."""

    def __init__(self, *, levels: int = 5, chunk_size: int = 16):
        if levels <= 0:
            raise ValueError("levels must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.levels = levels
        self.chunk_size = chunk_size

    @torch.no_grad()
    def __call__(self, generated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if generated.ndim != 5 or reference.ndim != 5:
            raise ValueError("generated and reference must be [B,C,F,H,W]")
        if generated.shape != reference.shape:
            raise ValueError(
                "generated and reference must have identical shapes, got "
                f"{tuple(generated.shape)} and {tuple(reference.shape)}"
            )
        if generated.shape[1] != 3:
            raise ValueError("wavelet color correction requires three RGB channels")

        output_dtype = generated.dtype
        corrected_video = torch.empty_like(generated)
        for start in range(0, generated.shape[2], self.chunk_size):
            stop = min(start + self.chunk_size, generated.shape[2])
            content = generated[:, :, start:stop].permute(0, 2, 1, 3, 4)
            content = content.reshape(-1, 3, *generated.shape[-2:]).float()
            style = reference[:, :, start:stop].to(
                device=generated.device, dtype=torch.float32
            )
            style = style.permute(0, 2, 1, 3, 4).reshape(
                -1, 3, *generated.shape[-2:]
            )

            content_low = _wavelet_low_frequency(content, self.levels)
            style_low = _wavelet_low_frequency(style, self.levels)
            corrected = (content - content_low + style_low).clamp_(-1.0, 1.0)
            corrected = corrected.reshape(
                generated.shape[0], stop - start, 3, *generated.shape[-2:]
            ).permute(0, 2, 1, 3, 4)
            corrected_video[:, :, start:stop] = corrected.to(dtype=output_dtype)

        return corrected_video
