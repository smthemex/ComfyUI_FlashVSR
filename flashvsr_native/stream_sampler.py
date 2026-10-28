# -*- coding: utf-8 -*-
"""FlashVSR native streaming sampler — drives the comfy native WanModel
(loaded via comfy.sd.load_diffusion_model) through the official FlashVSR
window schedule using ONLY comfy patch mechanisms:

  - per-block LQ residual injection  -> patches_replace["dit"][("double_block", i)]
  - sparse streaming self-attention  -> optimized_attention_override
  - window temporal RoPE offset      -> rope_options {"shift_t": ...}
  - one-step flow at sigma=1.0 (t=1000) via BaseModel.apply_model
    (CONST / ModelSamplingDiscreteFlow: denoised = x - v, exactly the
    DiffSynth `cur_latents - noise_pred_posi` semantics)

Window schedule (verbatim from FlashVSRFullPipeline.__call__ / model_fn_wan_video):
  process_total_num = (num_frames - 1) // 8 - 2
  window 0: latent frames [0:6],  LQ clips [max(0,4i-3):(4i+1)-3] for i in 0..6
  window i: latent frames [4+2i : 6+2i], LQ clips [8i+17+4j : 8i+21+4j] j=0,1
  topk = int((2*h*w//128)**2 * topk_ratio) - 1, topk_ratio = sparse_ratio*768*1280/(th*tw)
  kv_len = int(kv_ratio)   (sliding KV window, groups of 2 latent frames)
"""
import torch

import comfy.model_management
import comfy.utils


# ---------------------------------------------------------------------------
# frame padding helpers (verbatim port of examples/WanVSR/utils/utils.py)
# ---------------------------------------------------------------------------
def calculate_output_frames_with_padding(input_frames):
    start_value = 5  # 8*1 - 3
    interval = 8
    if input_frames < start_value:
        return start_value
    group_index = (input_frames - start_value) // interval
    group_min = start_value + group_index * interval
    group_max = group_min + interval - 1
    if group_min <= input_frames <= group_max:
        return group_min
    return group_min + interval


def calculate_frame_adjustment_simple(original_frames):
    input_frames = original_frames
    while True:
        output_frames = calculate_output_frames_with_padding(input_frames)
        if output_frames >= original_frames:
            break
        input_frames += 1
    frames_to_add = input_frames + 4 - original_frames
    frames_to_remove = output_frames - original_frames
    return {
        'required_input_frames': input_frames,
        'frames_to_add': frames_to_add,
        'expected_output_frames': output_frames,
        'frames_to_remove': frames_to_remove,
    }


def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def prepare_lq_frames(image, scale=4, multiple=128):
    """comfy IMAGE [F,H,W,C] 0-1 -> LQ video [1,3,F,tH,tW] in [-1,1].

    Verbatim replica of prepare_input_tensor (tensor branch) +
    upscale_then_center_crop: PIL BICUBIC on the uint8 domain, centre crop to
    the 128-multiple target, 4 duplicated tail frames, largest 8n+1 frame count.
    Returns (vid, tH, tW, F, pad_frames)."""
    import numpy as np
    from PIL import Image

    total, h0, w0, _ = image.shape
    if total == 1:
        image = image.repeat(25, 1, 1, 1)
        total = 25
    adjustment = calculate_frame_adjustment_simple(total)
    pad_frames = adjustment['frames_to_remove']
    if adjustment['frames_to_add'] > 0:
        image = torch.cat([image, image[-1:].repeat(adjustment['frames_to_add'], 1, 1, 1)], dim=0)
        total = image.shape[0]

    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)

    idx = list(range(total)) + [total - 1] * 4
    F = largest_8n1_leq(len(idx))
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding: got {len(idx)}")
    idx = idx[:F]

    left = max(0, (sW - tW) // 2)
    top = max(0, (sH - tH) // 2)
    u8 = (image.clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    frames = []
    for i in idx:
        img = Image.fromarray(u8[i]).convert("RGB")
        if (sW, sH) != (w0, h0):
            img = img.resize((sW, sH), Image.BICUBIC)
        img = img.crop((left, top, left + tW, top + tH))
        arr = torch.from_numpy(np.ascontiguousarray(np.asarray(img)))
        frames.append(arr.permute(2, 0, 1).float().div(127.5).sub(1.0))
    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,F,tH,tW]
    vid = vid.to(torch.bfloat16)  # reference: pil_to_tensor_neg1_1(dtype=bf16)
    return vid, tH, tW, F, pad_frames


class FlashVSRStreamState:
    """Per-run state shared by the attention override and the block patch."""

    def __init__(self, num_blocks, topk, kv_len, local_range, chunk_q=0):
        self.caches = [None] * num_blocks
        self.local_masks = {}
        self.params = {"topk": topk, "kv_len": kv_len, "local_range": local_range}
        self.chunk_q = chunk_q
        self.lq_latents = None
        self.enabled = True

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)


def compute_stream_params(th, tw, sparse_ratio, kv_ratio):
    """topk / kv_len exactly as model_fn_wan_video computes them (th/tw: output res)."""
    h = th // 16  # token grid
    w = tw // 16
    window_size = 2 * h * w // 128
    square_num = window_size * window_size
    topk_ratio = sparse_ratio * 768 * 1280 / (th * tw)
    topk = int(square_num * topk_ratio) - 1
    kv_len = int(kv_ratio)
    return topk, kv_len


def generate_noise(shape, seed, device, dtype=torch.bfloat16):
    generator = torch.Generator(device).manual_seed(int(seed))
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


@torch.no_grad()
def run_flashvsr_stream(patcher, lq_proj, posi_context, image, seed,
                        width, height, scale=4, kv_ratio=3.5, local_range=11,
                        sparse_ratio=2.0, offload=False, chunk_q=0,
                        segmented=False):
    """Full FlashVSR streaming one-step denoise on the native comfy model.

    segmented=False: returns (z [1,16,N,h,w] bf16, LQ, pad_frames, meta).
    segmented=True  (tiny_long mode): returns (segments list of [1,16,n,h,w],
             lq_idx list of (LQ_cur_idx, LQ_pre_idx), LQ, pad_frames, meta) —
             identical schedule/cache to the joined run, just not concatenated.
    """
    device = comfy.model_management.get_torch_device()
    model = patcher.model
    num_blocks = len(model.diffusion_model.blocks)

    lq_proj = lq_proj.to(device=device, dtype=torch.bfloat16)
    lq_proj.clear_cache()
    lq_proj.clip_idx = 0

    LQ, tH, tW, F, pad_frames = prepare_lq_frames(image, scale=scale)
    LQ = LQ.to(device=device, dtype=torch.bfloat16)  # reference keeps LQ on cuda
    latent_frames = (F - 1) // 4
    process_total_num = (F - 1) // 8 - 2
    if process_total_num < 1:
        raise RuntimeError(f"Not enough frames for streaming: F={F}")

    topk, kv_len = compute_stream_params(tH, tW, sparse_ratio, kv_ratio)

    noise = generate_noise((1, 16, latent_frames, tH // 8, tW // 8), seed, device=device)

    posi = posi_context.to(device=device, dtype=torch.bfloat16)
    sigma = torch.tensor([1.0], device=device)

    # --- patches ---
    # NOTE: build patches_replace directly on ONE transformer_options dict.
    # patcher.set_model_patch_replace() copies+replaces model_options[
    # "transformer_options"], which would silently drop our other keys.
    from .sparse_attn import make_stream_attn_override
    to_base = patcher.model_options["transformer_options"]
    to_base["optimized_attention_override"] = make_stream_attn_override()

    st = FlashVSRStreamState(num_blocks, topk, kv_len, local_range, chunk_q=chunk_q)
    to_base["flashvsr_stream"] = st

    def make_block_inject(block_id):
        def inject(args, opts):
            lq = st.lq_latents
            if lq is not None and block_id < len(lq):
                args["img"] = args["img"] + lq[block_id]
            return opts["original_block"](args)
        return inject

    dr = to_base.setdefault("patches_replace", {}).setdefault("dit", {})
    for i in range(num_blocks):
        dr[("double_block", i)] = make_block_inject(i)

    latents_total = []
    lq_idx_list = []
    LQ_cur_idx = 0
    LQ_pre_idx = 0
    try:
        comfy.model_management.load_models_gpu([patcher])

        pbar = comfy.utils.ProgressBar(process_total_num)
        for cur_idx in range(process_total_num):
            # ---- window LQ clips (verbatim schedule) ----
            if cur_idx == 0:
                inner_loop_num = 7
                lq_latents = None
                for inner in range(inner_loop_num):
                    clip = LQ[:, :, max(0, inner * 4 - 3):(inner + 1) * 4 - 3, :, :]
                    cur = lq_proj.stream_forward(clip)
                    if cur is None:
                        continue
                    if lq_latents is None:
                        lq_latents = cur
                    else:
                        for li in range(len(lq_latents)):
                            lq_latents[li] = torch.cat([lq_latents[li], cur[li]], dim=1)
                st.lq_latents = lq_latents
                x_window = noise[:, :, :6, :, :]
                rope_shift = 0
                LQ_cur_idx = (inner_loop_num - 1) * 4 - 3
            else:
                inner_loop_num = 2
                lq_latents = None
                for inner in range(inner_loop_num):
                    clip = LQ[:, :, cur_idx * 8 + 17 + inner * 4:
                              cur_idx * 8 + 21 + inner * 4, :, :]
                    cur = lq_proj.stream_forward(clip)
                    if cur is None:
                        continue
                    if lq_latents is None:
                        lq_latents = cur
                    else:
                        for li in range(len(lq_latents)):
                            lq_latents[li] = torch.cat([lq_latents[li], cur[li]], dim=1)
                st.lq_latents = lq_latents
                x_window = noise[:, :, 4 + cur_idx * 2:6 + cur_idx * 2, :, :]
                rope_shift = 4 + cur_idx * 2
                LQ_cur_idx = cur_idx * 8 + 21

            to_base["rope_options"] = {"shift_t": float(rope_shift), "scale_t": 1.0,
                                       "scale_x": 1.0, "scale_y": 1.0,
                                       "shift_x": 0.0, "shift_y": 0.0}

            x_window = x_window.to(device=device, dtype=torch.bfloat16)
            denoised = model.apply_model(x_window, sigma, c_crossattn=posi,
                                         transformer_options=to_base)
            cur_latents = denoised.to(torch.bfloat16)
            latents_total.append(cur_latents)
            lq_idx_list.append((int(LQ_cur_idx), int(LQ_pre_idx)))
            LQ_pre_idx = LQ_cur_idx
            pbar.update(1)

        z = torch.cat(latents_total, dim=2) if not segmented else latents_total
    finally:
        st.enabled = False
        to_base.pop("flashvsr_stream", None)
        to_base.pop("rope_options", None)
        to_base.pop("optimized_attention_override", None)
        to_base.get("patches_replace", {}).get("dit", {}).clear()
        st.caches = [None] * num_blocks
        if offload:
            comfy.model_management.unload_all_models()
            lq_proj = lq_proj.to("cpu")  # LQ proj is small but owns ~0.5GB VRAM
        torch.cuda.empty_cache()

    meta = {"F": F, "tH": tH, "tW": tW, "LQ_cur_idx": LQ_cur_idx,
            "scale": scale, "pad_frames": pad_frames}
    if segmented:
        meta["lq_idx_list"] = lq_idx_list
        return latents_total, lq_idx_list, LQ, pad_frames, meta
    return z, LQ, pad_frames, meta
