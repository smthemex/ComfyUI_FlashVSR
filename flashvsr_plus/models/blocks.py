from __future__ import annotations

import torch
import torch.nn as nn

from .attention import AttentionRuntime, CrossAttention, build_self_attention
from .config import AttentionConfig


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        attention: AttentionConfig,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.self_attn = build_self_attention(dim, num_heads, attention, eps)
        self.cross_attn = CrossAttention(dim, dim, num_heads, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time_modulation: torch.Tensor,
        runtime: AttentionRuntime,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=time_modulation.dtype, device=time_modulation.device) + time_modulation
        ).chunk(6, dim=1)
        attention_output = self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), runtime)
        x = x + gate_msa * attention_output
        x = x + self.cross_attn(self.norm3(x), context, context_mask)
        ffn_output = self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + gate_mlp * ffn_output

    def forward_stream(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time_modulation: torch.Tensor,
        runtime: AttentionRuntime,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=time_modulation.dtype, device=time_modulation.device)
            + time_modulation
        ).chunk(6, dim=1)
        attention_input = modulate(self.norm1(x), shift_msa, scale_msa)
        if hasattr(self.self_attn, "forward_stream"):
            attention_output, next_cache = self.self_attn.forward_stream(
                attention_input, runtime
            )
        else:
            attention_output = self.self_attn(attention_input, runtime)
            next_cache = None
        x = x + gate_msa * attention_output
        x = x + self.cross_attn(self.norm3(x), context, context_mask)
        chunk_tokens = runtime.stream_ffn_chunk_tokens
        if chunk_tokens is not None and x.shape[1] > chunk_tokens:
            # The FFN expansion (8960 channels in Wan/FlashVSR+) is the
            # largest transient in long-sequence stream inference. Process
            # token slabs independently and concatenate only the final
            # model-width result. This is numerically the same inference
            # operation, with a bounded [chunk_tokens, ffn_dim] workspace.
            ffn_parts = []
            for start in range(0, x.shape[1], chunk_tokens):
                stop = min(start + chunk_tokens, x.shape[1])
                ffn_parts.append(
                    self.ffn(
                        modulate(
                            self.norm2(x[:, start:stop]),
                            shift_mlp,
                            scale_mlp,
                        )
                    )
                )
            ffn_output = torch.cat(ffn_parts, dim=1)
        else:
            ffn_output = self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + gate_mlp * ffn_output, next_cache


class OutputHead(nn.Module):
    def __init__(self, dim: int, out_channels: int, patch_size: tuple[int, int, int], eps: float):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(dim, out_channels * patch_size[0] * patch_size[1] * patch_size[2])
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor) -> torch.Tensor:
        shift, scale = (
            self.modulation.to(dtype=timestep_embedding.dtype, device=timestep_embedding.device)
            + timestep_embedding[:, None]
        ).chunk(2, dim=1)
        return self.proj(self.norm(x) * (1 + scale) + shift)
