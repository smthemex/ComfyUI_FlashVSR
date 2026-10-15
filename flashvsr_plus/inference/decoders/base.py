from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass
class DecodeResult:
    video: torch.Tensor
    state: Any = None


class Decoder(Protocol):
    def decode(
        self,
        latents: torch.Tensor,
        *,
        lq_video: torch.Tensor | None = None,
        state: Any = None,
        stream: bool = False,
    ) -> DecodeResult: ...

    def reset(self) -> None: ...

