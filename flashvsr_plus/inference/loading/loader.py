from __future__ import annotations

import torch

from ...model_io import load_model_checkpoint
from ..config import InferenceConfig
from ..decoders import build_decoder
from ...models import (
    BlockReplacementConfig,
    FlashVSRModelConfig,
    LQProjectionConfig,
    build_model,
)

from .model_bundle import (
    ComponentFactories,
    FlowMatchInferenceScheduler,
    ModelBundle,
    ModelPaths,
    require_path,
)


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _load_context(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    value = torch.load(require_path(path, "context"), map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError("context checkpoint must contain a Tensor")
    return value.squeeze(0) if value.ndim == 3 and value.shape[0] == 1 else value


def _build_denoiser(config: InferenceConfig, paths: ModelPaths, factories: ComponentFactories):
    if factories.denoiser is not None:
        return factories.denoiser(config, paths.dit)
    model_config = FlashVSRModelConfig(
        attention=config.attention_config(),
        lq_projection=LQProjectionConfig(
            enabled=True,
            num_injection_layers=config.lq_projection_layers,
        ),
        replacement=BlockReplacementConfig(
            layers=config.replacement_layers,
        ),
    )
    model = build_model(model_config)
    if paths.dit:
        load_model_checkpoint(model, require_path(paths.dit, "DiT"), min_coverage=0.95)
    if paths.student:
        load_model_checkpoint(model, require_path(paths.student, "student"), min_coverage=0.95)
    if paths.lq_projection:
        load_model_checkpoint(
            model.LQ_proj_in,
            require_path(paths.lq_projection, "LQ projection"),
            component_prefix="LQ_proj_in.",
            min_coverage=0.95,
        )
    return model


def _build_decoder_module(config: InferenceConfig, paths: ModelPaths, factories: ComponentFactories):
    if config.decoder.kind == "vae":
        checkpoint = str(require_path(paths.vae, "VAE"))
        if factories.vae is not None:
            return factories.vae(checkpoint)
        from flashvsr_plus.models import build_vae

        return build_vae("wan", checkpoint)
    if factories.tiny_decoder is not None:
        module = factories.tiny_decoder(config.decoder, paths.tiny_decoder)
    else:
        from flashvsr_plus.inference.decoders.tiny_model import build_tcdecoder

        module = build_tcdecoder(
            new_channels=list(config.decoder.tiny_channels),
            device="cpu",
            dtype=dtype_from_name(config.dtype),
            new_latent_channels=config.decoder.tiny_latent_channels,
        )
        if paths.tiny_decoder:
            state = torch.load(
                require_path(paths.tiny_decoder, "Tiny decoder"),
                map_location="cpu",
                weights_only=True,
            )
            if isinstance(state, dict) and "decoder" in state:
                state = state["decoder"]
            module.load_state_dict(state, strict=False)
    return module


def build_model_bundle(
    config: InferenceConfig,
    paths: ModelPaths,
    factories: ComponentFactories | None = None,
) -> ModelBundle:
    config.validate()
    factories = factories or ComponentFactories()
    denoiser = _build_denoiser(config, paths, factories)
    if paths.vae is not None and paths.tiny_decoder is not None:
        decoder_module = _build_decoder_module(config, paths, factories)
        decoder = build_decoder(config.decoder, decoder_module)
    else:
        decoder = None
    scheduler = (
        factories.scheduler(config)
        if factories.scheduler
        else FlowMatchInferenceScheduler(shift=config.sigma_shift)
    )
    dtype = dtype_from_name(config.dtype)
    device = torch.device(config.device)
    denoiser.to(device=device, dtype=dtype)
    if paths.vae is not None and paths.tiny_decoder is not None:
        decoder_module.to(device=device, dtype=dtype)
    return ModelBundle(denoiser, decoder, scheduler, _load_context(paths.context)).eval()
