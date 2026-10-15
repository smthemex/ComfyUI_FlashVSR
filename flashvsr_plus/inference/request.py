from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class InferenceRequest:
    input_path: str | Path | None = None
    lq_video: torch.Tensor | None = None
    output_path: str | Path | None = None
    context: torch.Tensor | None = None
    height: int | None = None
    width: int | None = None
    max_frames: int | None = None
    fps: float | None = None

    def validate(self) -> None:
        if (self.input_path is None) == (self.lq_video is None):
            raise ValueError("Provide exactly one of input_path or lq_video")
        if self.lq_video is not None and self.lq_video.ndim != 5:
            raise ValueError("lq_video must be [B,C,F,H,W] or [B,F,C,H,W]")
        if self.lq_video is not None and self.lq_video.shape[0] != 1:
            raise ValueError("inference currently supports exactly one video (batch size 1)")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive")
        if (self.height is None) != (self.width is None):
            raise ValueError("height and width must be provided together")


@dataclass
class InferenceResult:
    video: torch.Tensor | None
    output_path: str | None
    frames: int
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)
