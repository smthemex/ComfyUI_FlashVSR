from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class LoadReport:
    source: str
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    skipped: tuple[str, ...]
    shape_mismatch: tuple[str, ...]

    @property
    def loaded_count(self) -> int:
        return len(self.loaded)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["loaded_count"] = self.loaded_count
        return result


def load_checkpoint_file(checkpoint: str | Path | Mapping) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        state = checkpoint
        source = "<mapping>"
    else:
        checkpoint = Path(checkpoint).expanduser()
        source = str(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if checkpoint.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as error:
                raise ImportError("Loading .safetensors requires safetensors") from error
            state = load_file(str(checkpoint), device="cpu")
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for container_key in ("state_dict", "module", "model", "model_state_dict"):
        if isinstance(state, Mapping) and container_key in state and isinstance(state[container_key], Mapping):
            state = state[container_key]
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint {source} does not contain a state dict")
    return {str(key): value for key, value in state.items() if torch.is_tensor(value)}


def normalize_key(key: str) -> str:
    prefixes = (
        "module.",
        "_forward_module.",
        "model.",
        "pipe.",
        "dit.",
        "pipe.denoising_model.",
        "pipe.dit.",
        "denoising_model.",
        "model.denoising_model.",
        "model.dit.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    key = key.replace("head.head.", "head.proj.")
    key = key.replace("norm1.gamma", "norm1.weight")
    key = key.replace("norm2.gamma", "norm2.weight")
    return key


def load_model_checkpoint(
    model,
    checkpoint: str | Path | Mapping,
    *,
    component_prefix: str | None = None,
    min_coverage: float = 0.0,
) -> LoadReport:
    state = load_checkpoint_file(checkpoint)
    source = str(checkpoint) if not isinstance(checkpoint, Mapping) else "<mapping>"
    own = model.state_dict()
    accepted: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    shape_mismatch: list[str] = []
    for source_key, value in state.items():
        key = normalize_key(source_key)
        if component_prefix:
            if key.startswith(component_prefix):
                key = key[len(component_prefix) :]
            elif source_key.startswith(component_prefix):
                key = source_key[len(component_prefix) :]
        if key not in own:
            skipped.append(source_key)
        elif own[key].shape != value.shape:
            shape_mismatch.append(source_key)
        else:
            accepted[key] = value
    result = model.load_state_dict(accepted, strict=False)
    coverage = len(accepted) / max(len(own), 1)
    if coverage < min_coverage:
        raise RuntimeError(
            f"Checkpoint coverage {coverage:.2%} is below required {min_coverage:.2%}: {source}"
        )
    return LoadReport(
        source=source,
        loaded=tuple(sorted(accepted)),
        missing=tuple(result.missing_keys),
        skipped=tuple(sorted(skipped)),
        shape_mismatch=tuple(sorted(shape_mismatch)),
    )
