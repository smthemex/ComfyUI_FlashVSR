from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import torch

from ..decoders import Decoder


class Scheduler(Protocol):
    def timesteps(self, steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor: ...
    def step(self, prediction: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor) -> torch.Tensor: ...


class FlowMatchInferenceScheduler:
    """Self-contained Flow-Matching scheduler used by FlashVSR inference."""

    def __init__(self, shift: float = 5.0):
        if shift <= 0:
            raise ValueError("scheduler shift must be positive")
        self.shift = float(shift)
        self._sigmas: torch.Tensor | None = None
        self._timesteps: torch.Tensor | None = None

    def timesteps(self, steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # The extra endpoint is used by the final Euler step but is not sent to
        # the denoiser. This matches FlashVSR's original one-step behavior.
        sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self._sigmas = sigmas
        self._timesteps = sigmas[:-1].mul(1000).to(dtype=dtype)
        return self._timesteps

    def step(self, prediction: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        if self._sigmas is None or self._timesteps is None:
            raise RuntimeError("timesteps() must be called before step()")
        scalar = timestep.detach().flatten()[0].to(
            device=self._timesteps.device, dtype=self._timesteps.dtype
        )
        index = int(torch.argmin((self._timesteps - scalar).abs()).item())
        delta = self._sigmas[index + 1] - self._sigmas[index]
        return latents + prediction * delta.to(device=latents.device, dtype=latents.dtype)


# Backward-compatible name for early framework users.
EulerInferenceScheduler = FlowMatchInferenceScheduler


@dataclass
class ModelPaths:
    dit: str | None = None
    student: str | None = None
    lq_projection: str | None = None
    context: str | None = None
    vae: str | None = None
    tiny_decoder: str | None = None


@dataclass
class ComponentFactories:
    denoiser: Callable | None = None
    vae: Callable | None = None
    tiny_decoder: Callable | None = None
    scheduler: Callable | None = None


@dataclass
class ModelBundle:
    denoiser: torch.nn.Module
    decoder: Decoder
    scheduler: Scheduler
    context: torch.Tensor | None = None

    def eval(self) -> "ModelBundle":
        self.denoiser.eval()
        module = getattr(self.decoder, "module", None)
        if isinstance(module, torch.nn.Module):
            module.eval()
        return self


def require_path(path: str | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} path is required")
    value = Path(path).expanduser()
    if not value.exists():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    return value
