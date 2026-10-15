from __future__ import annotations

from typing import Callable

import torch

from ..config import TEMPORAL_DOWNSAMPLE
from ..preprocessing import TemporalVideoStream, repeat_last_frame, truncate_video_frames
from ..streaming import StreamState
from ..streaming.memory_policy import choose_stream_cache_offload
from ..postprocessing import WaveletColorCorrector

from .base import BaseRunner, RunnerOutput


def _move_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return value


def _offload_dit(bundle, state, device):
    if state.attention_cache_transfer_stream is not None:
        state.attention_cache_transfer_stream.synchronize()
    state.attention_cache_ready_events.clear()
    state.attention_cache = _move_tree(state.attention_cache, "cpu")
    projection = bundle.denoiser.LQ_proj_in
    for name in ("_stream_cache1", "_stream_cache2"):
        setattr(projection, name, _move_tree(getattr(projection, name, None), "cpu"))
    bundle.denoiser.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _restore_dit(bundle, state, device):
    bundle.denoiser.to(device)
    # Attention copies each CPU cache on demand. Restoring all layers eagerly
    # would defeat the base auto-offload policy and cause a transient VRAM peak.


def _merge_segment_groups(value, factor, *, first, mode, sigma):
    if factor == 1:
        return value
    if first:
        head, tail = value[:, :, :1], value[:, :, 1:]
    else:
        head, tail = None, value
    if tail.shape[2] % factor:
        raise RuntimeError("decode segment is not aligned to temporal merge groups")
    tail = tail.reshape(value.shape[0], value.shape[1], -1, factor, *value.shape[-2:])
    if mode == "first":
        merged = tail[:, :, :, 0]
    else:
        positions = torch.arange(factor, device=value.device, dtype=torch.float32)
        weights = torch.exp(-0.5 * ((positions - (factor - 1) / 2) / sigma).square())
        weights = weights / weights.sum()
        merged = (tail.float() * weights.view(1, 1, 1, factor, 1, 1)).sum(dim=3).to(value.dtype)
    return merged if head is None else torch.cat((head, merged), dim=2)


class StreamRunner(BaseRunner):
    def _stream_noise_chunk(
        self,
        lq_video: torch.Tensor,
        latent_frames: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, _, _, height, width = lq_video.shape
        return torch.randn(
            batch,
            self.config.latent_channels,
            latent_frames,
            height // self.config.spatial_downsample,
            width // self.config.spatial_downsample,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _append_conditions(
        accumulated: list[torch.Tensor] | None,
        current: list[torch.Tensor] | None,
    ) -> list[torch.Tensor] | None:
        if current is None:
            return accumulated
        if accumulated is None:
            return list(current)
        if len(accumulated) != len(current):
            raise ValueError("LQ projection returned inconsistent layer counts")
        return [torch.cat([old, new], dim=1) for old, new in zip(accumulated, current)]

    def _project_lq_chunk(
        self,
        lq_video: torch.Tensor,
        process_index: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        frame_offset: int = 0,
    ) -> list[torch.Tensor]:
        projection = getattr(self.bundle.denoiser, "LQ_proj_in", None)
        if projection is None or not hasattr(projection, "stream_forward"):
            raise TypeError("FlashVSR stream inference requires LQProjection.stream_forward")
        conditions = None
        if process_index == 0:
            first = lq_video[:, :, :1].to(device=device, dtype=dtype)
            conditions = self._append_conditions(
                conditions, projection.stream_forward(first)
            )
            del first
        chunk_latents = self.config.streaming.chunk_latents
        for inner in range(chunk_latents):
            latent_index = process_index * chunk_latents + inner
            start = 1 + latent_index * TEMPORAL_DOWNSAMPLE - frame_offset
            end = start + TEMPORAL_DOWNSAMPLE
            clip = lq_video[:, :, start:end].to(device=device, dtype=dtype)
            conditions = self._append_conditions(
                conditions, projection.stream_forward(clip)
            )
            del clip
        if conditions is None:
            raise RuntimeError("LQ projection produced no stream conditions")
        return conditions

    @torch.inference_mode()
    def run(
        self,
        lq_video: torch.Tensor | TemporalVideoStream,
        context: torch.Tensor | None,
        *,
        progress: Callable[[str], None] | None = None,
        segment_writer=None,
        output_frame_limit: int | None = None,
    ) -> RunnerOutput:
        input_stream = lq_video if isinstance(lq_video, TemporalVideoStream) else None
        if input_stream is not None:
            if self.config.decoder.decode_every == "auto":
                raise ValueError("Iterative input requires numeric decode_every")
            lq_video = input_stream.window(0, 1)
        self._validate_video(lq_video)
        if self.config.num_inference_steps != 1:
            raise ValueError("cached stream inference supports exactly one denoising step")
        source_frames = lq_video.shape[2]
        lq_video, frames = truncate_video_frames(
            lq_video, TEMPORAL_DOWNSAMPLE
        )
        device = torch.device(self.config.device)
        dtype = self._execution_dtype()
        embedded = self._context(context, lq_video.shape[0], device, dtype)
        spatial_windows = (lq_video.shape[-2] // 16 // 8) * (lq_video.shape[-1] // 16 // 8)
        topk = self.config.sparse_topk
        if topk is None:
            topk = int(spatial_windows * spatial_windows * self.config.sparse_topk_ratio) - 1
        offload_decision = choose_stream_cache_offload(
            self.config,
            self.bundle.denoiser,
            lq_video,
            device,
        )
        if progress is not None:
            progress(
                f"KV cache policy: {offload_decision.mode}; "
                f"offload {len(offload_decision.layers)} layers; {offload_decision.reason}"
            )
        offload_layers = frozenset(offload_decision.layers)
        state = StreamState(
            attention_topk=topk,
            attention_cache_blocks=int(self.config.sparse_kv_ratio),
            attention_block_size=self.config.block_size,
            memory_trace=[] if self.config.profile_memory else None,
            ffn_chunk_tokens=self.config.stream_ffn_chunk_tokens,
            attention_block_m=self.config.stream_block_m,
            attention_cache_offload_layers=offload_layers,
            attention_cache_pin_memory=self.config.stream_cache_pin_memory,
            # Q/K/V are released after window partitioning; this lowers the
            # live workspace during later sparse-mask and attention stages.
            # It does not alter any attention arithmetic or cache contents.
            release_attention_workspaces=True,
            attention_cache_transfer_stream=(
                torch.cuda.Stream(device=device)
                if device.type == "cuda"
                and offload_layers
                and self.config.stream_cache_pin_memory
                else None
            ),
        )
        # Retain a diagnostics handle for benchmark tooling. In normal mode
        # this is just the state object that is released at the end of run.
        self._last_state = state
        requested_latents = (frames - 1) // TEMPORAL_DOWNSAMPLE + 1
        chunk_latents = self.config.streaming.chunk_latents
        padded_latents = (
            (requested_latents + chunk_latents - 1) // chunk_latents
        ) * chunk_latents
        # LQ projection consumes four look-ahead pixel frames per latent. The
        # final incomplete region is aligned by repeating the last input frame.
        projection_frames = 1 + padded_latents * TEMPORAL_DOWNSAMPLE
        decoder_frames = padded_latents * TEMPORAL_DOWNSAMPLE - 3
        projection_video = None if input_stream is not None else repeat_last_frame(lq_video, projection_frames)
        decoder_video = None if input_stream is not None else projection_video[:, :, :decoder_frames]
        segment_color = None
        color_config = self.config.color_correction
        if self.config.decoder.decode_every != "auto" and color_config.enabled:
            segment_color = WaveletColorCorrector(
                levels=color_config.levels, chunk_size=color_config.chunk_size
            )
        process_count = padded_latents // chunk_latents
        deferred = bool(getattr(self.config.decoder, "deferred", False))
        deferred_latents = None
        timesteps = self.bundle.scheduler.timesteps(1, device, dtype)
        timestep = timesteps[0]
        generator = torch.Generator(device=device).manual_seed(self.config.seed)
        outputs = []
        pending_latents = []
        pending_count = 0
        decoded_latent_offset = 0
        decode_limit = self.config.decoder.decode_every
        total_decode_segments = (
            (padded_latents + decode_limit - 1) // decode_limit
            if decode_limit != "auto" and input_stream is None else None
        )
        if decode_limit == "auto":
            decode_limit = None
        projection = self.bundle.denoiser.LQ_proj_in
        projection.clear_cache()
        if self.bundle.decoder is not None:
            self.bundle.decoder.reset()
        chunks = (
            input_stream.projection_chunks(chunk_latents, TEMPORAL_DOWNSAMPLE)
            if input_stream is not None else (
                (index, projection_video, 0, index + 1 == process_count)
                for index in range(process_count)
            )
        )
        produced_frames = 0
        try:
            for process_index, projection_chunk, frame_offset, last_chunk in chunks:
                conditions = self._project_lq_chunk(
                    projection_chunk,
                    process_index,
                    device=device,
                    dtype=dtype,
                    frame_offset=frame_offset,
                )
                current = self._stream_noise_chunk(
                    lq_video,
                    chunk_latents,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                # No subsequent chunk can consume this cache. Avoid the final
                # KV gather/copy while preserving the denoiser output exactly.
                state.write_cache = not last_chunk
                prediction = self.bundle.denoiser.forward_stream(
                    current,
                    timestep.expand(current.shape[0]),
                    embedded,
                    layer_conditions=conditions,
                    state=state,
                )
                denoised_chunk = self.bundle.scheduler.step(
                    prediction, timestep, current
                )
                pending_latents.append(denoised_chunk.detach().to("cpu"))
                pending_count += denoised_chunk.shape[2]
                del conditions, current, prediction, denoised_chunk
                if decode_limit is not None and pending_count >= decode_limit and not last_chunk:
                    segment = torch.cat(pending_latents, dim=2)
                    start = max(0, 4 * decoded_latent_offset - 3)
                    stop = 4 * (decoded_latent_offset + segment.shape[2]) - 3
                    decoder_lq = (
                        input_stream.window(start, stop) if input_stream is not None
                        else decoder_video[:, :, start:stop]
                    )
                    _offload_dit(self.bundle, state, device)
                    try:
                        decoded = self.bundle.decoder.decode_stream_segment(
                            segment, decoder_lq,
                            first=(decoded_latent_offset == 0),
                        )
                    finally:
                        _restore_dit(self.bundle, state, device)
                    segment_video = decoded.video
                    reference = decoder_lq[:, :, :segment_video.shape[2]]
                    factor = TEMPORAL_DOWNSAMPLE // self.config.temporal_compression
                    if decode_limit is not None and factor > 1:
                        # LQ groups contain repeated source frames. Select one
                        # directly to preserve the exact color reference.
                        if decoded_latent_offset == 0:
                            segment_video = _merge_segment_groups(
                                segment_video, factor, first=True,
                                mode=self.config.temporal_merge,
                                sigma=self.config.temporal_gaussian_sigma)
                            reference = _merge_segment_groups(
                                reference, factor, first=True,
                                mode="first",
                                sigma=self.config.temporal_gaussian_sigma)
                        else:
                            segment_video = _merge_segment_groups(
                                segment_video, factor, first=False,
                                mode=self.config.temporal_merge,
                                sigma=self.config.temporal_gaussian_sigma)
                            reference = _merge_segment_groups(
                                reference, factor, first=False,
                                mode="first",
                                sigma=self.config.temporal_gaussian_sigma)
                    if output_frame_limit is not None:
                        segment_video = segment_video[:, :, :max(output_frame_limit - produced_frames, 0)]
                    if segment_color is not None and segment_video.shape[2]:
                        segment_video = segment_color(
                            segment_video, reference[:, :, :segment_video.shape[2]]
                        )
                    if segment_writer is not None:
                        segment_writer.append(segment_video)
                    else:
                        outputs.append(segment_video)
                    produced_frames += segment_video.shape[2]
                    if progress is not None and decode_limit is not None:
                        segment_number = decoded_latent_offset // decode_limit + 1
                        segment_label = str(segment_number)
                        if total_decode_segments is not None:
                            segment_label += f"/{total_decode_segments}"
                        progress(
                            f"Decode segment {segment_label} "
                            f"· {segment.shape[2]} latent · "
                            f"{segment_video.shape[2]} output frames"
                        )
                    decoded_latent_offset += segment.shape[2]
                    pending_latents.clear()
                    pending_count = 0
                    if input_stream is not None:
                        input_stream.release_before(stop)
                    del segment, decoded, decoder_lq, reference, segment_video
                state.chunk_index = process_index + 1
            if input_stream is not None:
                frames = input_stream.expanded_frames
                source_frames = frames
                process_count = state.chunk_index
                requested_latents = (frames - 1) // TEMPORAL_DOWNSAMPLE + 1
                padded_latents = process_count * chunk_latents
                output_frame_limit = (
                    input_stream.source_frames if output_frame_limit is None
                    else min(output_frame_limit, input_stream.source_frames)
                )
                total_decode_segments = (padded_latents + decode_limit - 1) // decode_limit
            # DiT KV is not used by the decoder. Drop all 30 layers before
            # allocating the first VAE tile.
            if state.attention_cache_transfer_stream is not None:
                state.attention_cache_transfer_stream.synchronize()
            state.attention_cache.clear()
            state.attention_cache_ready_events.clear()
            projection.clear_cache()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if progress is not None and decode_limit is None:
                progress("DiT stream inference complete")
            if pending_count:
                segment = torch.cat(pending_latents, dim=2)
                start = max(0, 4 * decoded_latent_offset - 3)
                stop = 4 * (decoded_latent_offset + segment.shape[2]) - 3
                decoder_lq = (
                    input_stream.window(start, stop) if input_stream is not None
                    else decoder_video[:, :, start:stop]
                )
                deferred_latents = None
                if decode_limit is None and getattr(self.config.decoder, "deferred", False):
                    # Deferred decoding: hand the latents plus the aligned LQ
                    # conditioning video back to the caller instead of running
                    # the TinyDecoder here (ComfyUI KSampler -> Decoder split).
                    deferred_latents = segment.detach().to("cpu")
                    segment_video = None
                elif decode_limit is None:
                    decoded = self.bundle.decoder.decode(
                        segment, lq_video=decoder_video, stream=True,
                    )
                    segment_video = decoded.video
                else:
                    _offload_dit(self.bundle, state, device)
                    try:
                        decoded = self.bundle.decoder.decode_stream_segment(
                            segment, decoder_lq,
                            first=(decoded_latent_offset == 0),
                        )
                    finally:
                        _restore_dit(self.bundle, state, device)
                    segment_video = decoded.video
                if segment_video is not None:
                    reference = decoder_lq[:, :, :segment_video.shape[2]]
                    factor = TEMPORAL_DOWNSAMPLE // self.config.temporal_compression
                    if decode_limit is not None and factor > 1:
                        if decoded_latent_offset == 0:
                            segment_video = _merge_segment_groups(segment_video, factor, first=True, mode=self.config.temporal_merge, sigma=self.config.temporal_gaussian_sigma)
                            reference = _merge_segment_groups(reference, factor, first=True, mode="first", sigma=self.config.temporal_gaussian_sigma)
                        else:
                            segment_video = _merge_segment_groups(segment_video, factor, first=False, mode=self.config.temporal_merge, sigma=self.config.temporal_gaussian_sigma)
                            reference = _merge_segment_groups(reference, factor, first=False, mode="first", sigma=self.config.temporal_gaussian_sigma)
                    if output_frame_limit is not None:
                        segment_video = segment_video[:, :, :max(output_frame_limit - produced_frames, 0)]
                    if segment_color is not None and segment_video.shape[2]:
                        segment_video = segment_color(
                            segment_video, reference[:, :, :segment_video.shape[2]]
                        )
                    if segment_writer is not None:
                        segment_writer.append(segment_video)
                    else:
                        outputs.append(segment_video)
                    produced_frames += segment_video.shape[2]
                    if progress is not None and decode_limit is not None:
                        segment_number = decoded_latent_offset // decode_limit + 1
                        progress(
                            f"Decode segment {segment_number}/{total_decode_segments} "
                            f"· {segment.shape[2]} latent · "
                            f"{segment_video.shape[2]} output frames"
                        )
                pending_latents.clear()
                pending_count = 0
                if input_stream is not None:
                    input_stream.release_before(stop)
                # Release branch-local references regardless of the taken path.
                segment = decoder_lq = segment_video = None
                decoded = reference = None
            if deferred and not outputs:
                video = None
            elif segment_writer is not None:
                video = None
            else:
                video = torch.cat(outputs, dim=2)
            outputs.clear()
            if progress is not None:
                progress(
                    "TinyDecoder complete" if video is not None
                    else "Deferred decode: latents returned without TinyDecoder"
                )
            if video is not None:
                video = video[:, :, :frames]
        finally:
            # Also release model-owned caches when a chunk or tile raises OOM;
            # empty_cache alone cannot free tensors still referenced here.
            if state.attention_cache_transfer_stream is not None:
                state.attention_cache_transfer_stream.synchronize()
            state.attention_cache.clear()
            state.attention_cache_ready_events.clear()
            projection.clear_cache()
            if self.bundle.decoder is not None:
                self.bundle.decoder.reset()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return RunnerOutput(
            video,
            {
                "runner": "stream",
                "decode_every": self.config.decoder.decode_every,
                "input_streaming": input_stream is not None,
                "input_buffer_peak_frames": input_stream.max_buffer_frames if input_stream is not None else projection_frames,
                "chunks": process_count,
                "latent_schedule": f"{chunk_latents}",
                "source_frames": source_frames,
                "input_frames": frames,
                "output_frames": frames,
                "requested_latents": requested_latents,
                "padded_latents": padded_latents,
                "attention_topk": topk,
                "attention_cache_blocks": state.attention_cache_blocks,
                "attention_block_size": state.attention_block_size,
                "attention_cached_temporal_windows": state.attention_cache_blocks,
                "attention_cached_latent_frames": (
                    state.attention_cache_blocks
                    * self.config.sparse_window_size[0]
                ),
                "attention_cache_offload_layers": sorted(
                    state.attention_cache_offload_layers
                ),
                **offload_decision.metadata(),
                "tiny_decoder_execution_mode": getattr(
                    self.bundle.decoder, "last_execution_mode", None
                ),
                "tiny_decoder_auto_fallbacks": getattr(
                    self.bundle.decoder, "last_auto_fallbacks", 0
                ),
                **(
                    {
                        # Deferred decoding payload: denoised latents plus the
                        # aligned LQ conditioning video for the Decoder node.
                        "latents": deferred_latents,
                        "decoder_lq": decoder_video,
                        "decoder_frames": decoder_video.shape[2],
                    }
                    if deferred and deferred_latents is not None
                    else {}
                ),
            },
        )
