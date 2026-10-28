# -*- coding: utf-8 -*-
"""FlashVSR native sparse-attention core (ported 1:1 from the official
DiffSynth implementation shipped inside this plugin: FlashVSR/diffsynth/
models/wan_video_dit.py + examples/WanVSR/utils/utils.py).

The reference semantics on machines WITHOUT the `block_sparse_attn` package
(which is the case here) are the fallback: block-level top-k mask -> nearest
upsample to token level -> SDPA with the 0/1 float mask applied ADDITIVELY
(+1 bias on selected blocks).  That quirk is intentional and is what the
released weights were verified against by the plugin author, so it is
replicated exactly.

Everything here is pure torch; no diffusers / transformers dependency.
"""
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

WIN = (2, 8, 8)  # official: temporal x 8 x 8 token windows


# ----------------------------------------------------------------------------
# Window partition (verbatim port of WindowPartition3D)
# ----------------------------------------------------------------------------
class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B,F,H,W,C)."""

    @staticmethod
    def partition(x: torch.Tensor, win: Tuple[int, int, int] = WIN):
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        assert F % wf == 0 and H % wh == 0 and W % ww == 0, "Dims must divide by window size."
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(windows: torch.Tensor, win: Tuple[int, int, int], orig: Tuple[int, int, int]):
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


# ----------------------------------------------------------------------------
# Local (sliding) block mask  (verbatim port)
# ----------------------------------------------------------------------------
@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(block_h: int,
                                                    block_w: int,
                                                    win_h: int = 6,
                                                    win_w: int = 6,
                                                    include_self: bool = True,
                                                    device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r = start_r + win_h - 1
    start_c = c_all - c_half
    end_c = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


# ----------------------------------------------------------------------------
# Draft block (top-k) mask  (verbatim port of generate_draft_block_mask)
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate_draft_block_mask(batch_size, nheads, seqlen, q_w, k_w, topk=10,
                              local_attn_mask=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    avgpool_q = torch.mean(q_w, dim=1)
    avgpool_k = torch.mean(k_w, dim=1)
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]
    local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    local_attn_mask = rearrange(local_attn_mask, 'x a y b -> (x a) (y b)')
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    local_attn_mask = local_attn_mask.to(torch.float32)
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == False, -float('inf'))
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
    scores = scores + local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    n = flat.shape[1]
    apply_topk = min(flat.shape[1] - 1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask


# ----------------------------------------------------------------------------
# Fallback block-sparse attention (verbatim semantics of
# examples/WanVSR/utils/utils.py::block_sparse_attn_func)
#
# Reference path: bool block mask [1,H,Nqw,Nkw] -> F.interpolate(nearest) to
# token level -> SDPA with the resulting 0/1 float mask (ADDITIVE bias).
# The chunked variant below computes exactly the same per-row math with a
# query-chunk loop so that production resolutions fit in VRAM.
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Official block-sparse kernel (mit-han-lab Block-Sparse-Attention), preferred
# exactly like the reference code: try to import it (torch must be imported
# first so its DLLs are on the search path), fall back to the SDPA expansion
# when unavailable.  Call convention replicated 1:1 from
# diffsynth/models/wan_video_dit.py::flash_attention (mask branch).
# ----------------------------------------------------------------------------
_BSA_FUNC = None
_BSA_PROBED = False


def _get_block_sparse_func():
    global _BSA_FUNC, _BSA_PROBED
    if not _BSA_PROBED:
        _BSA_PROBED = True
        try:
            import torch as _t  # noqa: F401  (DLL search path must include torch)
            from block_sparse_attn import block_sparse_attn_func as _f
            _BSA_FUNC = _f
            print("[FlashVSR Native] using official block_sparse_attn kernel")
        except Exception as e:
            _BSA_FUNC = None
            print(f"[FlashVSR Native] block_sparse_attn unavailable ({type(e).__name__}); "
                  "using SDPA fallback")
    return _BSA_FUNC


def _run_official_block_sparse(q_bhtd, k_bhtd, v_bhtd, block_mask):
    """q/k/v [B,H,S,D]; block_mask bool [1,H,Sq//128,Skv//128].
    Mirrors flash_attention's mask branch call exactly."""
    B, H, Sq, D = q_bhtd.shape
    Skv = k_bhtd.shape[2]
    fn = _get_block_sparse_func()
    if fn is None:
        return None
    if not (q_bhtd.is_cuda and q_bhtd.dtype in (torch.float16, torch.bfloat16)):
        return None
    if Sq % 128 or Skv % 128:
        return None
    q_p = q_bhtd.permute(2, 0, 1, 3).reshape(B * Sq, H, D)
    k_p = k_bhtd.permute(2, 0, 1, 3).reshape(B * Skv, H, D)
    v_p = v_bhtd.permute(2, 0, 1, 3).reshape(B * Skv, H, D)
    cu_q = torch.tensor([0, Sq], device=q_p.device, dtype=torch.int32)
    cu_k = torch.tensor([0, Skv], device=q_p.device, dtype=torch.int32)
    head_mask_type = torch.tensor([1] * H, device=q_p.device, dtype=torch.int32)
    out = fn(
        q_p, k_p, v_p,
        cu_q, cu_k,
        head_mask_type,
        None,                 # streaming_info
        block_mask,           # base_blockmask [1, H, Nq_win, Nkv_win]
        Sq, Skv,
        0.0,                  # p_dropout
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )
    return out.reshape(B, Sq, H, D).permute(0, 2, 1, 3)  # [B, H, Sq, D]


def _sdpa_with_token_mask(q_bhtd, k_bhtd, v_bhtd, block_mask, chunk_q=0):
    """q/k/v: [B,H,S,D]; block_mask: [1,H,Sqw//bw,Skw//bw] bool (bw=128).

    SEMANTICS (verified empirically): the reference fallback does
    `F.interpolate(mask.float(), ...).to(attn_mask.dtype)` where `attn_mask`
    still refers to the ORIGINAL BOOL tensor — so the token-level mask is a
    HARD BOOL mask (True=attend), not a float additive bias.
    chunk_q must be a multiple of bw(128) so the chunked nearest-upsample is
    exactly equivalent to the single-call reference.
    """
    B, H, Sq, D = q_bhtd.shape
    Skv = k_bhtd.shape[2]
    bw = Sq // block_mask.shape[2]
    assert Sq % bw == 0 and Skv % bw == 0

    # preferred: official mit-han-lab block-sparse kernel (same as reference)
    if chunk_q in (0, None):
        out = _run_official_block_sparse(q_bhtd, k_bhtd, v_bhtd, block_mask)
        if out is not None:
            return out

    if chunk_q and chunk_q > 0:
        chunk_q = max(bw, (chunk_q // bw) * bw)
    if not chunk_q or chunk_q >= Sq:
        mask_tok = F.interpolate(block_mask.float(), size=(Sq, Skv), mode='nearest')
        mask_tok = mask_tok.to(torch.bool)  # reference quirk: back to bool
        out = F.scaled_dot_product_attention(q_bhtd, k_bhtd, v_bhtd, attn_mask=mask_tok)
        return out

    outs = []
    for s in range(0, Sq, chunk_q):
        e = min(s + chunk_q, Sq)
        sub = block_mask[:, :, s // bw:e // bw, :]
        mask_tok = F.interpolate(sub.float(), size=(e - s, Skv), mode='nearest')
        mask_tok = mask_tok.to(torch.bool)
        outs.append(F.scaled_dot_product_attention(
            q_bhtd[:, :, s:e], k_bhtd, v_bhtd, attn_mask=mask_tok))
    return torch.cat(outs, dim=2)


def streaming_sparse_self_attention(q, k, v, heads, transformer_options):
    """FlashVSR streaming sparse self-attention.

    q, k, v: [B, S, heads*dim] — q/k are ALREADY rope'd by the comfy WanModel
    (freqs offset controlled via rope_options.shift_t by the stream sampler).
    Returns the attention output [B, S, heads*dim] (pre output-projection),
    and updates the per-block KV window cache held in the stream state.
    """
    st = transformer_options["flashvsr_stream"]
    grid = transformer_options["grid_sizes"]
    f, h, w = int(grid[0]), int(grid[1]), int(grid[2])
    B, L, D = q.shape
    block_id = transformer_options.get("block_index", 0)
    prm = st["params"]

    q = q.view(B, f, h, w, D)
    k = k.view(B, f, h, w, D)
    v = v.view(B, f, h, w, D)

    q_w = WindowPartition3D.partition(q)
    k_w = WindowPartition3D.partition(k)
    v_w = WindowPartition3D.partition(v)

    seqlen = f // WIN[0]
    one_len = k_w.shape[0] // B // seqlen

    cache = st["caches"][block_id]
    if cache is not None:
        k_w = torch.cat([cache[0], k_w], dim=0)
        v_w = torch.cat([cache[1], v_w], dim=0)

    block_n = q_w.shape[0] // B
    block_s = q_w.shape[1]

    reorder_q = rearrange(q_w, '(b block_n) (block_s) d -> b (block_n block_s) d',
                          block_n=block_n, block_s=block_s)
    reorder_k = rearrange(k_w, '(b block_n) (block_s) d -> b (block_n block_s) d',
                          block_n=k_w.shape[0] // B, block_s=block_s)
    reorder_v = rearrange(v_w, '(b block_n) (block_s) d -> b (block_n block_s) d',
                          block_n=k_w.shape[0] // B, block_s=block_s)

    # local mask (cached per resolution/local_range)
    lm_key = (h // 8, w // 8, prm["local_range"])
    lm = st["local_masks"].get(lm_key)
    if lm is None or lm.device != k_w.device:
        lm = build_local_block_mask_shifted_vec_normal_slide(
            h // 8, w // 8, prm["local_range"], prm["local_range"],
            include_self=True, device=k_w.device)
        st["local_masks"][lm_key] = lm

    attention_mask = generate_draft_block_mask(
        B, heads, seqlen, q_w, k_w, topk=prm["topk"], local_attn_mask=lm)

    # [B, tokens, D] -> [B, H, S, Dh] for SDPA
    n_hd = heads
    dh = D // n_hd
    qq = rearrange(reorder_q, 'b s (n d) -> b n s d', n=n_hd)
    kk = rearrange(reorder_k, 'b s (n d) -> b n s d', n=n_hd)
    vv = rearrange(reorder_v, 'b s (n d) -> b n s d', n=n_hd)
    x = _sdpa_with_token_mask(qq, kk, vv, attention_mask,
                              chunk_q=st.get("chunk_q", 0))
    x = rearrange(x, 'b n s d -> b s (n d)', n=n_hd)

    # sliding cache update (verbatim)
    cur_block_n, cur_block_s, _ = k_w.shape
    cache_num = cur_block_n // one_len
    if cache_num > prm["kv_len"]:
        st["caches"][block_id] = (k_w[one_len:, :, :], v_w[one_len:, :, :])
    else:
        st["caches"][block_id] = (k_w, v_w)

    x = rearrange(x, 'b (block_n block_s) d -> (b block_n) (block_s) d',
                  block_n=block_n, block_s=block_s)
    x = WindowPartition3D.reverse(x, WIN, (f, h, w))
    return x.view(B, f * h * w, D)


def make_stream_attn_override():
    """optimized_attention_override for comfy native wan self-attn.

    Only intercepts self-attention (q_len == k_len == grid tokens); everything
    else (cross-attn over the 512 context tokens) falls through to comfy.
    """
    def override(func, q, k, v, heads=None, mask=None, skip_reshape=False,
                 transformer_options=None, **kwargs):
        if transformer_options is None or "flashvsr_stream" not in transformer_options:
            return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
        st = transformer_options["flashvsr_stream"]
        if st.get("enabled", True) and k.shape[1] == q.shape[1]:
            return streaming_sparse_self_attention(q, k, v, heads, transformer_options)
        return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
    return override
