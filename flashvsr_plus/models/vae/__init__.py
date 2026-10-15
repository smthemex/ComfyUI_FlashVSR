from .wan import (
    WanVAE,
    WanVAEEncoder,
    build_wan_vae,
    build_wan_vae_encoder,
    load_wan_vae_checkpoint,
    load_wan_vae_encoder_checkpoint,
)


def build_vae_encoder(vae_type: str, checkpoint_path: str) -> WanVAEEncoder:
    vae_type = vae_type.lower()
    if vae_type != "wan":
        raise ValueError(f"Unknown VAE type: {vae_type}")
    return build_wan_vae_encoder(checkpoint_path)


def build_vae(vae_type: str, checkpoint_path: str) -> WanVAE:
    vae_type = vae_type.lower()
    if vae_type != "wan":
        raise ValueError(f"Unknown VAE type: {vae_type}")
    return build_wan_vae(checkpoint_path)


__all__ = [
    "WanVAE",
    "WanVAEEncoder",
    "build_vae",
    "build_vae_encoder",
    "build_wan_vae",
    "build_wan_vae_encoder",
    "load_wan_vae_checkpoint",
    "load_wan_vae_encoder_checkpoint",
]
