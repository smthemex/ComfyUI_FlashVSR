from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

from .config import TEMPORAL_DOWNSAMPLE, InferenceConfig
from .loading import ComponentFactories, ModelBundle, ModelPaths, build_model_bundle
from .output import (
    IncrementalVideoWriter,
    mux_source_audio,
    save_tensor_video,
)
from .preprocessing import (
    expand_for_temporal_compression,
    load_lq_video,
    iter_lq_video,
    get_lq_video_fps,
    TemporalVideoStream,
    merge_temporal_output,
)
from .postprocessing import WaveletColorCorrector
from .request import InferenceRequest, InferenceResult
from   .runners import StreamRunner
from   .randomness import set_inference_seed


class FlashVSRInferencePipeline:
    def __init__(self, config: InferenceConfig, bundle: ModelBundle):
        config.validate()
        self.config = config
        self.bundle = bundle
        color_config = config.color_correction
        self.color_corrector = (
            WaveletColorCorrector(
                levels=color_config.levels,
                chunk_size=color_config.chunk_size,
            )
            if color_config.enabled
            else None
        )
        self.runner = StreamRunner(config, bundle)

    def infer(
        self,
        request: InferenceRequest | None = None,
        *,
        progress: Callable[[str], None] | None = None,
        video_preset: str = "veryfast",
        **kwargs,
    ) -> InferenceResult:
        request = request or InferenceRequest(**kwargs)
        request.validate()
        set_inference_seed(self.config.seed)
        input_stream = None
        expansion = None
        lq_video = None
        if self.config.decoder.decode_every != "auto":
            fps = get_lq_video_fps(request)
            chunk_frames = self.config.decoder.decode_every * self.config.temporal_compression
            input_stream = TemporalVideoStream(
                iter_lq_video(
                    request, chunk_frames=chunk_frames,
                    first_chunk_frames=chunk_frames + 1,
                ),
                self.config.temporal_compression,
                TEMPORAL_DOWNSAMPLE,
            )
            model_input = input_stream
        else:
            lq_video, fps = load_lq_video(request, stream=True)
            expansion = expand_for_temporal_compression(
                lq_video, self.config.temporal_compression,
                model_compression=TEMPORAL_DOWNSAMPLE,
            )
            model_input = expansion.video
        destination = Path(request.output_path) if request.output_path is not None else None
        audio_source = (
            Path(request.input_path)
            if destination is not None and request.input_path is not None
            else None
        )
        encoded_path = destination
        temporary_video = None
        if destination is not None and audio_source is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.stem}.",
                suffix=destination.suffix or ".mp4",
                dir=str(destination.parent),
            )
            os.close(fd)
            temporary_video = Path(temporary_name)
            temporary_video.unlink()
            encoded_path = temporary_video
        if progress is not None:
            progress("Iterative input ready" if input_stream is not None else "Input preprocessing complete")
        incremental = (
            request.output_path is not None
            and self.config.decoder.decode_every != "auto"
            and self.config.temporal_compression in (1, 2, 4)
        )
        writer = None
        if incremental:
            writer = IncrementalVideoWriter(
                encoded_path, fps, preset=video_preset
            )
        try:
            output = self.runner.run(
                model_input, request.context, progress=progress,
                segment_writer=writer,
                output_frame_limit=(expansion.source_frames if writer is not None and expansion is not None else None),
            )
        except Exception:
            if writer is not None:
                writer.close()
            if temporary_video is not None:
                temporary_video.unlink(missing_ok=True)
            raise
        finally:
            if input_stream is not None:
                input_stream.close()
        timeline = input_stream if input_stream is not None else expansion
        deferred = bool(getattr(self.config.decoder, "deferred", False))
        if writer is not None:
            writer.close()
            artifact = writer.verify()
            video = None
        else:
            if self.config.decoder.decode_every == "auto":
                if deferred:
                    # Decoding is deferred to the caller (ComfyUI Decoder
                    # node); the runner returned latents instead of video.
                    if output.video is not None:
                        raise RuntimeError(
                            "deferred decoding produced unexpected video output"
                        )
                    video = None
                else:
                    video = merge_temporal_output(
                        output.video,
                        expansion,
                        mode=self.config.temporal_merge,
                        gaussian_sigma=self.config.temporal_gaussian_sigma,
                    )
            else:
                # Numeric decode intervals already merge each segment onto
                # the source timeline inside StreamRunner.
                video = output.video
            # The model input may contain repeated tail frames used only to
            # satisfy its temporal grid. Never expose those synthetic frames.
            if video is not None:
                video = video[:, :, : timeline.source_frames]
        color_config = self.config.color_correction
        if video is not None and self.color_corrector is not None and self.config.decoder.decode_every == "auto":
            reference = lq_video[:, :, : video.shape[2]]
            video = self.color_corrector(video, reference)
            if progress is not None:
                progress("Color correction complete")
        elif self.color_corrector is not None and progress is not None:
            progress("Color correction complete (per decode segment)")
        output_path = None
        artifact = locals().get("artifact", None)
        if request.output_path is not None and video is not None:
            artifact = save_tensor_video(
                video[0].permute(1, 0, 2, 3),
                encoded_path,
                fps,
                preset=video_preset,
            )
            if progress is not None:
                progress("Video saved")
        if destination is not None and audio_source is not None and encoded_path is not None:
            mux_source_audio(encoded_path, audio_source, destination)
            if temporary_video is not None:
                temporary_video.unlink(missing_ok=True)
            if artifact is not None:
                artifact["path"] = str(destination)
            output_path = str(destination)
        elif destination is not None and artifact is not None:
            output_path = str(destination)
        runner_metadata = dict(output.metadata)
        if "input_frames" in runner_metadata:
            runner_metadata["model_input_frames"] = runner_metadata.pop("input_frames")
        if "output_frames" in runner_metadata:
            runner_metadata["model_output_frames"] = runner_metadata.pop(
                "output_frames"
            )
        metadata = {
            **runner_metadata,
            "stream": True,
            "deferred_decode": deferred,
            "decoder": self.config.decoder.kind,
            "tiny_decoder_parallel_mode": (
                self.config.decoder.tiny_decoder_parallel_mode
                if self.config.decoder.kind == "tiny"
                else None
            ),
            "tiny_decoder_channels_last": (
                self.config.decoder.tiny_decoder_channels_last
                if self.config.decoder.kind == "tiny"
                else None
            ),
            "is_causal": self.config.is_causal,
            "is_sparse": self.config.is_sparse,
            "is_hybrid_sliding": self.config.is_hybrid_sliding,
            "attn_kernel": self.config.attn_kernel,
            "block_size": self.config.block_size,
            "sparse_local_range": self.config.sparse_local_range,
            "sliding_local_range": self.config.sliding_local_range,
            "sliding_temporal_block_size": (
                self.config.sliding_temporal_block_size
            ),
            "sliding_temporal_mode": self.config.sliding_temporal_mode,
            "sliding_spatial_mode": self.config.sliding_spatial_mode,
            "sliding_cache_temporal_windows": (
                self.config.sliding_cache_temporal_windows
            ),
            "num_inference_steps": self.config.num_inference_steps,
            "sigma_shift": self.config.sigma_shift,
            "replacement_layers": self.config.replacement_layers,
            "source_frames": timeline.source_frames,
            "effective_input_frames": timeline.effective_frames,
            "expanded_model_frames": timeline.expanded_frames,
            "source_fps": fps,
            "output_frames": (video.shape[2] if video is not None else int(artifact.get("frames", timeline.effective_frames) if artifact else timeline.effective_frames)),
            "temporal_compression": self.config.temporal_compression,
            "temporal_repeat_factor": timeline.repeat_factor,
            "temporal_merge": self.config.temporal_merge,
            "color_correction": {
                "enabled": color_config.enabled,
                "method": "wavelet",
                "levels": color_config.levels,
                "chunk_size": color_config.chunk_size,
            },
            "artifact": artifact,
        }
        result_frames = video.shape[2] if video is not None else int(artifact.get("frames", 0) if artifact else 0)
        return InferenceResult(video, output_path or (str(request.output_path) if artifact else None), result_frames, fps, metadata)


def build_inference_pipeline(
    config: InferenceConfig,
    model_paths: ModelPaths | None = None,
    *,
    bundle: ModelBundle | None = None,
    factories: ComponentFactories | None = None,
) -> FlashVSRInferencePipeline:
    config.validate()
    if bundle is None:
        if model_paths is None:
            raise ValueError("Provide model_paths or a prebuilt ModelBundle")
        bundle = build_model_bundle(config, model_paths, factories)
    return FlashVSRInferencePipeline(config, bundle.eval())
