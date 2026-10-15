from __future__ import annotations

from typing import Any

import torch

from .base import DecodeResult


class TinyDecoder:
    """Adapter around the internal temporal-consistent Tiny Decoder.

    ``serial`` evaluates one frame at a time and ``parallel`` evaluates the
    whole stream chunk. ``auto`` selects between these two executors from
    resolution and available VRAM.
    """

    def __init__(
        self,
        module,
        *,
        parallel: bool = False,
        parallel_mode: str = "auto",
        channels_last: bool = True,
    ):
        if not hasattr(module, "decode_video"):
            raise TypeError("Tiny decoder module must expose decode_video")
        self.module = module
        # Ada's BF16 convolution kernels are faster in NHWC/channels-last
        # format. The in-tree TAEHV opts in via ``use_channels_last``; custom
        # decoder factories keep their original layout unless they expose the
        # same flag. This changes only storage/layout, not the graph or math.
        self.channels_last = bool(channels_last)
        if (
            self.channels_last
            and hasattr(module, "use_channels_last")
            and torch.cuda.is_available()
        ):
            try:
                parameter = next(module.parameters())
                if parameter.dtype in (torch.float16, torch.bfloat16):
                    module.to(memory_format=torch.channels_last)
                    module.use_channels_last = True
            except (AttributeError, StopIteration, TypeError, RuntimeError):
                pass
        self.parallel = bool(parallel)
        self.parallel_mode = str(parallel_mode).lower()
        if self.parallel_mode not in {"auto", "serial", "parallel"}:
            raise ValueError("parallel_mode must be 'auto', 'serial', or 'parallel'")
        self.last_execution_mode: str | None = None
        self.last_auto_fallbacks = 0

    def reset(self) -> None:
        if hasattr(self.module, "clean_mem"):
            self.module.clean_mem()

    @staticmethod
    def _decoder_workspace_mib(height: int, width: int) -> float:
        # Calibrated from the 2K/89-frame RTX 4090 run: full temporal parallel
        # decoding requested about 7.5 GiB for one four-latent stream chunk.
        # The estimate only orders the two candidates; auto still has an OOM
        # fallback to serial execution.
        area_ratio = (height * width) / (1536 * 2560)
        return 7680.0 * area_ratio

    def _auto_candidates(
        self,
        lq_video: torch.Tensor,
        device: torch.device,
    ) -> list[tuple[str, bool]]:
        height, width = lq_video.shape[-2:]
        area = height * width
        candidates: list[tuple[str, bool]] = []
        # The full parallel path is useful only for small latent feature maps;
        # on 24-GiB Ada, 1080p and larger inputs are more reliable and faster
        # with the O(1)-workspace serial executor.
        if area <= 1_000_000:
            free_mib = None
            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    free_mib = torch.cuda.mem_get_info(device)[0] / 1024**2
                except RuntimeError:
                    pass
            required = self._decoder_workspace_mib(height, width) + 768.0
            if free_mib is None or free_mib >= required:
                candidates.append(("parallel", True))
        # Strict serial is always the final safety net.
        candidates.append(("serial", False))
        return candidates

    def _decode_chunks(
        self,
        latents: torch.Tensor,
        lq_video: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        use_parallel: bool,
        first: bool = True,
    ) -> torch.Tensor:
        # Decode in bounded temporal segments. The decoder still uses its
        # internal four-latent temporal microsteps for correct memory updates.
        latent_chunk_size = 4
        outputs = []
        for latent_start in range(0, latents.shape[2], latent_chunk_size):
            latent_stop = min(latent_start + latent_chunk_size, latents.shape[2])
            current_latents = latents[:, :, latent_start:latent_stop].to(
                device=device, dtype=dtype
            )
            if first and latent_start == 0:
                frame_start = 0
                frame_count = 4 * (latent_stop - latent_start) - 3
            else:
                frame_start = 4 * latent_start - (3 if first else 0)
                frame_count = 4 * (latent_stop - latent_start)
            current_lq = lq_video[:, :, frame_start : frame_start + frame_count].to(
                device=device, dtype=dtype
            )
            current = self.module.decode_video(
                current_latents.transpose(1, 2),
                parallel=use_parallel,
                show_progress_bar=False,
                cond=current_lq,
            ).transpose(1, 2)
            outputs.append(current.detach().to("cpu"))
            del current_latents, current_lq, current
        value = torch.cat(outputs, dim=2)
        outputs.clear()
        return value

    @torch.inference_mode()
    def decode_stream_segment(self, latents, lq_video, *, first: bool = False):
        """Decode one contiguous segment while retaining TinyDecoder memory."""
        if first:
            self.reset()
            self.last_auto_fallbacks = 0
        parameter = next(self.module.parameters())
        device, dtype = parameter.device, parameter.dtype
        mode = self.parallel_mode
        if mode == "auto":
            candidates = self._auto_candidates(lq_video, device)
        elif mode == "parallel":
            candidates = [("parallel", True)]
        else:
            candidates = [("serial", False)]
        # Snapshot the incoming history so an OOM retry cannot lose preceding
        # segments or reuse partially updated temporal memory.
        saved = [None if x is None else x.detach().cpu().clone() for x in self.module.mem]
        for name, use_parallel in candidates:
            self.module.mem = [None if x is None else x.to(device) for x in saved]
            try:
                value = self._decode_chunks(
                    latents, lq_video, device=device, dtype=dtype,
                    use_parallel=use_parallel, first=first,
                )
            except torch.cuda.OutOfMemoryError:
                self.module.mem = saved
                if mode != "auto" or name == "serial":
                    raise
                self.last_auto_fallbacks += 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            self.last_execution_mode = name
            # No decoder activations remain on GPU while DiT resumes.
            self.module.mem = [None if x is None else x.detach().cpu().clone()
                               for x in self.module.mem]
            return DecodeResult(value.mul_(2).sub_(1), None)
        raise RuntimeError("Tiny Decoder produced no output")

    def decode(
        self,
        latents: torch.Tensor,
        *,
        lq_video: torch.Tensor | None = None,
        state: Any = None,
        stream: bool = False,
    ) -> DecodeResult:
        del state
        if lq_video is None:
            raise ValueError("Tiny decoder requires the aligned LQ conditioning video")
        try:
            parameter = next(self.module.parameters())
            device, dtype = parameter.device, parameter.dtype
        except (AttributeError, StopIteration):
            device, dtype = latents.device, latents.dtype
        requested_mode = self.parallel_mode
        if requested_mode == "auto" and self.parallel:
            # A legacy explicit parallel=True remains a full-parallel request.
            requested_mode = "parallel"
        if requested_mode == "auto":
            candidates = self._auto_candidates(lq_video, device)
        elif requested_mode == "serial":
            candidates = [("serial", False)]
        else:
            candidates = [("parallel", True)]

        last_error = None
        self.last_auto_fallbacks = 0
        try:
            for mode, use_parallel in candidates:
                self.reset()
                try:
                    value = self._decode_chunks(
                        latents,
                        lq_video,
                        device=device,
                        dtype=dtype,
                        use_parallel=use_parallel,
                    )
                except torch.cuda.OutOfMemoryError as error:
                    last_error = error
                    if requested_mode != "auto" or mode == "serial":
                        raise
                    self.last_auto_fallbacks += 1
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                self.last_execution_mode = mode
                # The original Tiny Decoder produces approximately [0,1].
                value = value.mul(2).sub(1)
                return DecodeResult(value, None)
            if last_error is not None:
                raise last_error
            raise RuntimeError("Tiny Decoder produced no output")
        finally:
            self.reset()
            if device.type == "cuda":
                torch.cuda.empty_cache()
