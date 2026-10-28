# -*- coding: utf-8 -*-
"""FlashVSR+ (plus) native streaming sampler.

Drives the comfy native WanModel (plus DiT weights, `head.proj` remapped to
`head.head`) through the official FlashVSR+ stream schedule, replicating
flashvsr_plus/inference/runners/stream.py with comfy patch mechanisms:

  - per-layer LQ residual injection -> patches_replace["dit"][("double_block", i)]
  - per-layer streaming attention (sparse / sliding_local backends, window KV
    caches) -> optimized_attention_override, algorithm core imported VERBATIM
    from the plugin's flashvsr_plus.models.attention_backends (pure torch)
  - chunk RoPE offset (temporal_offset = 4*chunk_index) -> rope_options.shift_t
  - one-step flow at sigma=1.0 (t=1000); comfy apply_model already returns
    x - v which equals FlowMatchInferenceScheduler.step at 1 step / shift 5.

Deferred decode: returns DiT-space latents + aligned LQ conditioning video;
decoding is done by FlashVSR_Native_Decoder (TCDecoder path).
"""
import torch

import comfy.model_management
import comfy.utils

from ..flashvsr_plus.inference.config import (
    HYBRID_B25_REPLACEMENT_LAYERS, TEMPORAL_DOWNSAMPLE)
from ..flashvsr_plus.inference.presets import stream_config_for_mode
from ..flashvsr_plus.models.attention_backends.kernels import run_attention
from ..flashvsr_plus.models.attention_backends.sparse import (
    partition_heads,
    reverse_heads,
    build_stream_sparse_layout,
)


# ---------------------------------------------------------------------------
# LQ preprocessing (replicates the wrapper's run_plus_denoise input path:
# nearest-exact resize to (width,height), bicubic x scale, 128-aligned crop)
# ---------------------------------------------------------------------------
def prepare_lq_plus(image, width, height, scale=4, multiple=128):
    import numpy as np
    from PIL import Image
    from comfy.utils import common_upscale

    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.shape[1] != height or image.shape[2] != width:
        image = common_upscale(image.movedim(-1, 1), width, height,
                               "nearest-exact", "center").movedim(1, -1)
    h0, w0 = int(image.shape[1]), int(image.shape[2])
    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    left = max(0, (sW - tW) // 2)
    top = max(0, (sH - tH) // 2)
    u8 = (image.clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    frames = []
    for i in range(u8.shape[0]):
        img = Image.fromarray(u8[i]).convert("RGB")
        if (sW, sH) != (w0, h0):
            img = img.resize((sW, sH), Image.BICUBIC)
        img = img.crop((left, top, left + tW, top + tH))
        frames.append(torch.from_numpy(np.ascontiguousarray(np.asarray(img))).permute(2, 0, 1))
    lq = torch.stack(frames, 0).float().div(127.5).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0)
    return lq.to(torch.bfloat16), tH, tW


class PlusStreamState:
    def __init__(self, num_blocks, topk, layer_cfgs, cache_groups, sliding_cache_windows):
        self.layer_cfgs = layer_cfgs          # per-layer AttentionConfig
        self.caches = [None] * num_blocks     # per-layer (k_windows, v_windows)
        self.topk = topk
        self.cache_groups = cache_groups      # sparse layers: int(sparse_kv_ratio)
        self.sliding_cache_windows = sliding_cache_windows
        self.chunk_index = 0
        self.write_cache = True
        self.enabled = True
        self.lq_latents = None
        self.block_m = 128  # triton kernel block_m (AttentionRuntime default)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)


def _is_sliding_layer(cfg) -> bool:
    return getattr(cfg, "backend", "") == "sliding_local"


def plus_stream_attention(q, k, v, heads, transformer_options):
    """Replicates BlockSparseSelfAttention._forward_stream_impl with the
    plugin's own backend functions.  q/k are rope'd by comfy (temporal offset
    applied via rope_options.shift_t)."""
    st = transformer_options["flashvsr_plus_stream"]
    grid = transformer_options["grid_sizes"]
    token_shape = (int(grid[0]), int(grid[1]), int(grid[2]))
    block_index = transformer_options.get("block_index", 0)
    cfg = st.layer_cfgs[block_index]
    B = q.shape[0]

    frames, height, width = token_shape
    window = tuple(cfg.sparse.window_size)
    # q/k/v [B,S,H*D] -> [B,H,S,Dh]
    dh = q.shape[-1] // heads
    qh = q.view(B, frames, height, width, heads, dh).permute(0, 4, 1, 2, 3, 5).reshape(B, heads, frames * height * width, dh)
    kh = k.view(B, frames, height, width, heads, dh).permute(0, 4, 1, 2, 3, 5).reshape(B, heads, frames * height * width, dh)
    vh = v.view(B, frames, height, width, heads, dh).permute(0, 4, 1, 2, 3, 5).reshape(B, heads, frames * height * width, dh)

    q_windows, padded_shape, grid_shape = partition_heads(qh, token_shape, window)
    k_windows, _, _ = partition_heads(kh, token_shape, window)
    v_windows, _, _ = partition_heads(vh, token_shape, window)

    sliding = _is_sliding_layer(cfg)
    if sliding:
        # temporal_mode == "local": self-contained, no cache
        use_cache = cfg.sliding.temporal_mode == "short_causal"
        cache_temporal_windows = cfg.sliding.cache_temporal_windows if use_cache else None
    else:
        use_cache = True
        cache_temporal_windows = None  # -> runtime.stream_cache_blocks

    if use_cache and st.caches[block_index] is not None:
        cache_k, cache_v = st.caches[block_index]
        k_windows = torch.cat([cache_k.to(k_windows), k_windows], dim=2)
        v_windows = torch.cat([cache_v.to(v_windows), v_windows], dim=2)

    spatial = grid_shape[1] * grid_shape[2]
    topk = st.topk
    layout = build_stream_sparse_layout(
        q_windows, k_windows,
        temporal_windows=grid_shape[0],
        spatial_grid=(grid_shape[1], grid_shape[2]),
        topk=topk,
        cache_temporal_groups=(
            st.sliding_cache_windows if (sliding and cache_temporal_windows is not None)
            else st.cache_groups),
        temporal_block_size=cfg.sliding.temporal_block_size if sliding else 2,
        local_range=(cfg.sliding.local_range if (sliding and cfg.sliding.spatial_mode == "local")
                     else cfg.sparse.local_range),
        is_sparse=cfg.is_sparse,
    )
    batch, heads_, query_blocks, block_tokens, head_dim = q_windows.shape
    key_blocks = k_windows.shape[2]
    if cfg.kernel == "triton_block_sparse":
        # flashvsr-attn Triton kernel consumes the UNFLATTENED window layout
        # (same call site as flashvsr_plus BlockSparseSelfAttention).
        try:
            from flashvsr_attn import stream_block_sparse_attention
            output = stream_block_sparse_attention(
                q_windows, k_windows, v_windows, layout.block_mask,
                ensure_nonempty=False, block_m=st.block_m)
        except Exception as err:
            print(f"[FlashVSR Native+] triton_block_sparse failed ({type(err).__name__}); "
                  "falling back to sdpa")
            output = run_attention(
                q_windows.reshape(batch, heads_, query_blocks * block_tokens, head_dim),
                k_windows.reshape(batch, heads_, key_blocks * block_tokens, head_dim),
                v_windows.reshape(batch, heads_, key_blocks * block_tokens, head_dim),
                kernel="sdpa",
                block_mask=layout.block_mask,
                block_token_size=layout.block_token_size,
            )
    else:
        # kernel "block_sparse" (mit-han-lab BSA) or "sdpa" — routed inside
        # run_attention; auto resolution already guaranteed availability.
        output = run_attention(
            q_windows.reshape(batch, heads_, query_blocks * block_tokens, head_dim),
            k_windows.reshape(batch, heads_, key_blocks * block_tokens, head_dim),
            v_windows.reshape(batch, heads_, key_blocks * block_tokens, head_dim),
            kernel=cfg.kernel,
            block_mask=layout.block_mask,
            block_token_size=layout.block_token_size,
        )
    output = output.reshape(batch, heads_, query_blocks, block_tokens, head_dim)
    output = reverse_heads(output, token_shape, window, padded_shape)

    if use_cache and st.write_cache:
        gather = layout.cache_indices[..., None, None].expand(
            -1, -1, -1, block_tokens, head_dim)
        st.caches[block_index] = (
            torch.gather(k_windows, 2, gather).detach(),
            torch.gather(v_windows, 2, gather).detach(),
        )

    out = output.transpose(1, 2).reshape(B, frames * height * width, heads_ * dh)
    return out


def make_plus_attn_override():
    def override(func, q, k, v, heads=None, mask=None, skip_reshape=False,
                 transformer_options=None, **kwargs):
        if transformer_options is None or "flashvsr_plus_stream" not in transformer_options:
            return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
        st = transformer_options["flashvsr_plus_stream"]
        if st.enabled and k.shape[1] == q.shape[1]:
            return plus_stream_attention(q, k, v, heads, transformer_options)
        return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
    return override


def _project_lq_chunk(lq_proj, lq_video, process_index, chunk_latents, device, dtype):
    conditions = None
    if process_index == 0:
        first = lq_video[:, :, :1].to(device=device, dtype=dtype)
        cur = lq_proj.stream_forward(first)
        if cur is not None:
            conditions = list(cur)
    for inner in range(chunk_latents):
        latent_index = process_index * chunk_latents + inner
        start = 1 + latent_index * TEMPORAL_DOWNSAMPLE
        end = start + TEMPORAL_DOWNSAMPLE
        clip = lq_video[:, :, start:end].to(device=device, dtype=dtype)
        cur = lq_proj.stream_forward(clip)
        if cur is not None:
            conditions = list(cur) if conditions is None else [
                torch.cat([o, c], dim=1) for o, c in zip(conditions, cur)]
    return conditions


@torch.no_grad()
def run_flashvsr_plus_stream(patcher, lq_proj, posi_context, image, seed,
                             width, height, scale=4, mode="turbo", offload=False):
    """Full FlashVSR+ streaming one-step denoise on the native comfy model.

    Returns (z [1,16,N,h,w] bf16 DiT-space latents, decoder_lq
    [1,3,decoder_frames,tH,tW] bf16, meta dict).
    """
    device = comfy.model_management.get_torch_device()
    model = patcher.model
    num_blocks = len(model.diffusion_model.blocks)

    # kernel priority (official plus auto resolution): block_sparse_attn ->
    # flashvsr-attn (triton_block_sparse) -> sdpa.  select_kernels probes with
    # find_spec (no DLL pitfall) and refuses unavailable explicit choices.
    iconfig = stream_config_for_mode(mode, device=str(device), attn_kernel="auto")
    print(f"[FlashVSR Native+] attention kernel: {iconfig.attention_config().kernel}")
    attn_base = iconfig.attention_config()
    layer_cfgs = []
    for layer_index in range(num_blocks):
        import copy
        cfg = copy.deepcopy(attn_base)
        if cfg.backend == "hybrid_sliding":
            cfg.backend = "sparse" if (cfg.is_sparse or cfg.sparse.local_range is not None) else "dense"
        if layer_index in set(iconfig.replacement_layers):
            if cfg.sliding.spatial_mode == "local":
                cfg.sparse.local_range = cfg.sliding.local_range
            if cfg.sliding.temporal_mode == "local":
                cfg.backend = "sliding_local"
                cfg.is_causal = False
                cfg.causal.block_size_frames = 1
            elif cfg.sliding.temporal_mode == "short_causal":
                cfg.backend = "sliding_local"
            else:
                cfg.backend = "sparse"
        layer_cfgs.append(cfg)

    lq_proj = lq_proj.to(device=device, dtype=torch.bfloat16)
    lq_proj.clear_cache()

    LQ, tH, tW = prepare_lq_plus(image, width, height, scale=scale)

    frames_in = LQ.shape[2]
    frames = ((frames_in - 1) // TEMPORAL_DOWNSAMPLE) * TEMPORAL_DOWNSAMPLE + 1
    LQ = LQ[:, :, :frames]
    chunk_latents = iconfig.streaming.chunk_latents
    requested_latents = (frames - 1) // TEMPORAL_DOWNSAMPLE + 1
    padded_latents = ((requested_latents + chunk_latents - 1) // chunk_latents) * chunk_latents
    projection_frames = 1 + padded_latents * TEMPORAL_DOWNSAMPLE
    decoder_frames = padded_latents * TEMPORAL_DOWNSAMPLE - 3
    projection_video = LQ[:, :, -1:].repeat(1, 1, max(projection_frames - frames, 0), 1, 1)
    projection_video = torch.cat([LQ, projection_video], dim=2) if projection_frames > frames else LQ[:, :, :projection_frames]
    decoder_lq = projection_video[:, :, :decoder_frames]
    process_count = padded_latents // chunk_latents
    if process_count < 1:
        raise RuntimeError(f"Not enough frames for plus streaming: frames={frames}")

    spatial_windows = (tH // 16 // 8) * (tW // 16 // 8)
    topk = int(spatial_windows * spatial_windows * iconfig.sparse_topk_ratio) - 1

    cache_groups = int(iconfig.sparse_kv_ratio)
    st = PlusStreamState(num_blocks, topk, layer_cfgs, cache_groups,
                         iconfig.sliding_cache_temporal_windows)

    posi = posi_context.to(device=device, dtype=torch.bfloat16)
    if posi.ndim == 2:
        posi = posi.unsqueeze(0)
    sigma = torch.tensor([1.0], device=device)

    from .stream_sampler import prepare_lq_frames as _unused  # noqa: F401 (parity note)
    to_base = patcher.model_options["transformer_options"]
    to_base["optimized_attention_override"] = make_plus_attn_override()
    to_base["flashvsr_plus_stream"] = st

    def make_block_inject(block_id):
        def inject(args, opts):
            lq = st.lq_latents
            if lq is not None and block_id < len(lq):
                cond = lq[block_id].to(device=args["img"].device, dtype=args["img"].dtype)
                args["img"] = args["img"] + cond
            return opts["original_block"](args)
        return inject

    dr = to_base.setdefault("patches_replace", {}).setdefault("dit", {})
    for i in range(num_blocks):
        dr[("double_block", i)] = make_block_inject(i)

    generator = torch.Generator(device=device).manual_seed(int(seed))
    latents_total = []
    try:
        comfy.model_management.load_models_gpu([patcher])
        pbar = comfy.utils.ProgressBar(process_count)
        for process_index in range(process_count):
            conditions = _project_lq_chunk(lq_proj, projection_video, process_index,
                                           chunk_latents, device, torch.bfloat16)
            st.lq_latents = conditions
            current = torch.randn(
                1, 16, chunk_latents, tH // 8, tW // 8,
                generator=generator, device=device, dtype=torch.bfloat16)
            st.write_cache = process_index + 1 != process_count
            st.chunk_index = process_index
            to_base["rope_options"] = {"shift_t": float(4 * process_index),
                                       "scale_t": 1.0, "scale_x": 1.0, "scale_y": 1.0,
                                       "shift_x": 0.0, "shift_y": 0.0}
            denoised = model.apply_model(current, sigma, c_crossattn=posi,
                                         transformer_options=to_base)
            latents_total.append(denoised.to(torch.bfloat16))
            pbar.update(1)
        z = torch.cat(latents_total, dim=2)
    finally:
        st.enabled = False
        to_base.pop("flashvsr_plus_stream", None)
        to_base.pop("rope_options", None)
        to_base.pop("optimized_attention_override", None)
        to_base.get("patches_replace", {}).get("dit", {}).clear()
        st.caches = [None] * num_blocks
        if offload:
            comfy.model_management.unload_all_models()
            lq_proj = lq_proj.to("cpu")
        torch.cuda.empty_cache()

    meta = {"tH": tH, "tW": tW, "source_frames": int(LQ.shape[2]),
            "decoder_frames": int(decoder_frames), "mode": mode,
            "temporal_compression": int(iconfig.temporal_compression),
            "temporal_merge": iconfig.temporal_merge,
            "temporal_gaussian_sigma": iconfig.temporal_gaussian_sigma,
            "scale": scale}
    return z, decoder_lq, meta
