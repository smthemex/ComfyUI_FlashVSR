from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import INPUT_MULTIPLE, TEMPORAL_DOWNSAMPLE, InferenceConfig
from ..loading import ModelBundle, dtype_from_name


@dataclass
class RunnerOutput:
    video: torch.Tensor
    metadata: dict


class BaseRunner:
    def __init__(self, config: InferenceConfig, bundle: ModelBundle):
        self.config = config
        self.bundle = bundle

    def _execution_dtype(self) -> torch.dtype:
        parameter = next(self.bundle.denoiser.parameters(), None)
        return parameter.dtype if parameter is not None else dtype_from_name(self.config.dtype)

    def _validate_video(self, video: torch.Tensor) -> None:
        height, width = video.shape[-2:]
        # Stream attention partitions the latent grid into 8x8 spatial
        # windows after the 2x2 DiT patch, so the pixel input must be a
        # multiple of 128.  Catch this before allocating model workspaces.
        multiple = INPUT_MULTIPLE
        if height % multiple or width % multiple:
            raise ValueError(
                f"Input height and width must be divisible by {multiple}, got {(height, width)}"
            )

    def _context(self, request_context: torch.Tensor | None, batch: int, device, dtype):
        context = request_context if request_context is not None else self.bundle.context
        if context is None:
            raise ValueError("Inference requires request.context or ModelBundle.context")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.shape[0] == 1 and batch > 1:
            context = context.expand(batch, -1, -1)
        return context.to(device=device, dtype=dtype)

    def _initial_latents(self, lq_video: torch.Tensor, seed: int) -> torch.Tensor:
        batch, _, frames, height, width = lq_video.shape
        # One extra terminal latent is required to decode all F=4n+1 input
        # frames. Its missing look-ahead conditioning is supplied by repeating
        # the last input frame in the runner.
        latent_frames = (frames - 1) // TEMPORAL_DOWNSAMPLE + 1
        generator = torch.Generator(device="cpu").manual_seed(seed)
        value = torch.randn(
            batch,
            self.config.latent_channels,
            latent_frames,
            height // self.config.spatial_downsample,
            width // self.config.spatial_downsample,
            generator=generator,
            dtype=torch.float32,
        )
        return value.to(device=lq_video.device, dtype=dtype_from_name(self.config.dtype))

    def _denoise(
        self,
        lq_video: torch.Tensor,
        context: torch.Tensor,
        *,
        seed: int,
        stream_state=None,
        lq_condition_video: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latents = self._initial_latents(lq_video, seed)
        lq_condition_video = (
            lq_video if lq_condition_video is None else lq_condition_video
        )
        timesteps = self.bundle.scheduler.timesteps(
            self.config.num_inference_steps, latents.device, latents.dtype
        )
        for timestep in timesteps:
            step = timestep.expand(latents.shape[0])
            if stream_state is not None and hasattr(self.bundle.denoiser, "forward_stream"):
                prediction = self.bundle.denoiser.forward_stream(
                    latents, step, context, lq_video=lq_video, state=stream_state
                )
            else:
                prediction = self.bundle.denoiser(
                    latents,
                    step,
                    context,
                    lq_video=lq_condition_video,
                )
            latents = self.bundle.scheduler.step(prediction, timestep, latents)
        return latents
