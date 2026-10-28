# -*- coding: utf-8 -*-
"""FlashVSR native nodes (comfy wan patches implementation).

Chain:  FlashVSR_Native_Model -> FlashVSR_Native_Sampler -> FlashVSR_Native_Decoder

- Model:    loads the FlashVSR DiT via comfy.sd.load_diffusion_model (native
            WanModel), plus the tiny LQ_proj_in module and the fixed posi_prompt
            embedding.  No diffusers / diffsynth involved.
- Sampler:  official streaming window schedule (one-step, t=1000) driving the
            native model through patches; outputs a standard LATENT whose
            "samples" are RAW Wan VAE latents (z*std+mean) so the standard
            VAEDecode node works.  Extra keys carry the DiT-space z + LQ video
            for the TCDecoder path and colour correction.
- Decoder:  comfy VAE (VAELoader, Wan2.1 VAE) or TCDecoder (tcd_encoder);
            optional wavelet / adain colour correction against the LQ video.
"""
import os

import torch
import folder_paths
from comfy_api.latest import io

import nodes as _comfy_nodes  # MAX_RESOLUTION

MAX_SEED = torch.iinfo(torch.int32).max

_FVSR_DIR = os.path.join(folder_paths.models_dir, "FlashVSR")
if not os.path.exists(_FVSR_DIR):
    os.makedirs(_FVSR_DIR)
folder_paths.add_model_folder_path("FlashVSR", _FVSR_DIR)

# Wan2.1 latent stats (comfy latent_formats.Wan21)
LATENTS_MEAN = torch.tensor([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
]).view(1, 16, 1, 1, 1)
LATENTS_STD = torch.tensor([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916
]).view(1, 16, 1, 1, 1)


def _fvsr_list(predicate):
    return ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if predicate(i)]


class FlashVSR_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_Model",
            display_name="FlashVSR  Model (wan patches)",
            category="FlashVSR",
            inputs=[
                io.Combo.Input("dit", options=_fvsr_list(
                    lambda i: ("dmd" in i.lower() or "dit" in i.lower()) and i.endswith(".safetensors")),
                    tooltip="FlashVSR streaming DMD DiT (native comfy WanModel); plus: turbo_dit_plus / base_dit_plus"),
                io.Combo.Input("lq_proj", options=_fvsr_list(lambda i: "proj" in i.lower()),
                    tooltip="LQ_proj_in ckpt; must match version (LQ_proj_in.ckpt=1.0, "
                            "LQ_proj_in_v11.ckpt=1.1, lq_proj_in_plus_turbo/base_lq_proj_in_plus=plus)"),
                io.Combo.Input("prompt_embed", options=_fvsr_list(lambda i: "prompt" in i.lower()),
                    tooltip="posi_prompt.pth fixed context embedding"),
                io.Combo.Input("version", options=["1.1", "1.0", "plus_turbo", "plus_base"],
                    tooltip="1.0=Buffer proj, 1.1=Causal proj; plus_turbo/plus_base=FlashVSR+ "
                            "(weights must match: turbo_dit_plus↔plus_turbo, base_dit_plus↔plus_base)"),
            ],
            outputs=[io.Custom("FVSRN_MODEL").Output(display_name="model")],
        )

    @classmethod
    def execute(cls, dit, lq_proj, prompt_embed, version) -> io.NodeOutput:
        import comfy.sd
        import comfy.utils

        dit_path = folder_paths.get_full_path("FlashVSR", dit)
        proj_path = folder_paths.get_full_path("FlashVSR", lq_proj)
        emb_path = folder_paths.get_full_path("FlashVSR", prompt_embed)
        assert dit_path and proj_path and emb_path, "select dit / lq_proj / prompt_embed files"

        posi = torch.load(emb_path, map_location="cpu", weights_only=False)
        if not torch.is_tensor(posi):
            posi = posi.get("context") if isinstance(posi, dict) else None
        assert torch.is_tensor(posi), "prompt_embed must be a context tensor"

        if str(version).startswith("plus"):
            mode = "turbo" if version == "plus_turbo" else "base"
            from ..flashvsr_plus.inference.presets import stream_config_for_mode
            from ..flashvsr_plus.models.config import LQProjectionConfig
            from ..flashvsr_plus.models.conditioning import LQProjection

            iconfig = stream_config_for_mode(mode, device="cpu", attn_kernel="auto")

            # ---- comfy META loading: remap head.proj -> head.head and hand the
            # state dict straight to comfy (NO real-module double build). ----
            sd = comfy.utils.load_torch_file(dit_path, safe_load=True)
            comfy_sd = {}
            for k, v in sd.items():
                if k.startswith("LQ_proj_in."):
                    continue  # separate checkpoint, loaded below
                if k.startswith("head.proj."):
                    comfy_sd["head.head." + k[len("head.proj."):]] = v
                else:
                    comfy_sd[k] = v
            del sd
            patcher = comfy.sd.load_diffusion_model_state_dict(
                comfy_sd, model_options={"dtype": torch.bfloat16})
            del comfy_sd
            if patcher is None:
                raise RuntimeError("comfy could not detect a wan model in the plus DiT weights")

            # ---- tiny LQ projection module (~25M params) ----
            lq = LQProjection(
                LQProjectionConfig(enabled=True,
                                   num_injection_layers=iconfig.lq_projection_layers),
                output_dim=1536)
            lq_sd = comfy.utils.load_torch_file(proj_path, safe_load=True)
            lq_report = lq.load_state_dict(lq_sd, strict=False)
            del lq_sd
            if lq_report.missing_keys:
                raise RuntimeError(f"plus LQ proj load missing keys: {lq_report.missing_keys[:5]}")
            lq = lq.to(dtype=torch.bfloat16)

            model_obj = {"patcher": patcher, "lq_proj": lq, "posi": posi,
                         "version": version, "mode": mode}
            return io.NodeOutput(model_obj)

        # ---- 1.0 / 1.1 ----
        from .lq_proj import build_lq_proj

        patcher = comfy.sd.load_diffusion_model(dit_path, model_options={"dtype": torch.bfloat16})
        if patcher is None:
            raise RuntimeError(f"comfy could not detect a wan model in {dit_path}")

        lq = build_lq_proj(version, device="cpu")
        sd = torch.load(proj_path, map_location="cpu", weights_only=False)
        lq.load_state_dict(sd, strict=True)
        del sd

        model_obj = {"patcher": patcher, "lq_proj": lq, "posi": posi,
                     "version": version}
        return io.NodeOutput(model_obj)


class FlashVSR_SM_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_Sampler",
            display_name="FlashVSR Sampler (streaming sparse)",
            category="FlashVSR",
            inputs=[
                io.Custom("FVSRN_MODEL").Input("model"),
                io.Image.Input("image"),
                io.Int.Input("width", default=1280, min=128, max=_comfy_nodes.MAX_RESOLUTION, step=64,
                             tooltip="LQ regularised width (nearest-exact resize before x-scale)"),
                io.Int.Input("height", default=768, min=128, max=_comfy_nodes.MAX_RESOLUTION, step=64,
                             tooltip="LQ regularised height"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("scale", default=4, min=1, max=4,
                             tooltip="upscale factor: LQ is bicubic-upscaled x scale, centre-cropped to 128-multiple"),
                io.Float.Input("kv_ratio", default=3.5, min=0.0, max=10.0, step=0.1, round=0.01),
                io.Int.Input("local_range", default=11, min=1, max=50, step=1,
                             tooltip="9 = sharper, 11 = more stable (official recommendation)"),
                io.Float.Input("sparse_ratio", default=2.0, min=0.0, max=10.0, step=0.1),
                io.Boolean.Input("offload", default=False,
                                 tooltip="unload the DiT after sampling"),
                io.Int.Input("chunk_q", default=0, min=0, max=8192, step=128,
                             tooltip="query-chunk size for the sparse attention (0=auto). "
                                     "Must be a multiple of 128. Lower = less VRAM."),
                io.Boolean.Input("long_mode", default=False,
                                 tooltip="tiny_long mode: emit per-window latent segments for "
                                         "segmented TCDecoder decoding (long videos); requires tcd_encoder"),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, model, image, width, height, seed, scale, kv_ratio,
                local_range, sparse_ratio, offload, chunk_q, long_mode=False) -> io.NodeOutput:
        from comfy.utils import common_upscale

        patcher = model["patcher"]
        lq_proj = model["lq_proj"]
        posi = model["posi"]
        version = str(model.get("version", "1.0"))

        # regularise LQ to width/height (nearest-exact, as the wrapper does)
        if image.shape[1] != height or image.shape[2] != width:
            image = common_upscale(image.movedim(-1, 1), width, height,
                                   "nearest-exact", "center").movedim(1, -1)

        if version.startswith("plus"):
            # FlashVSR+ streaming (fixed 1-step; kv_ratio/local_range/sparse_ratio/
            # chunk_q/long_mode are 1.x-only and ignored here)
            from .plus_sampler import run_flashvsr_plus_stream
            z, decoder_lq, meta = run_flashvsr_plus_stream(
                patcher, lq_proj, posi, image, seed=seed, width=width, height=height,
                scale=scale, mode=model["mode"], offload=offload)
            mean = LATENTS_MEAN.to(device=z.device, dtype=z.dtype)
            std = LATENTS_STD.to(device=z.device, dtype=z.dtype)
            raw = z * std + mean
            return io.NodeOutput({"samples": raw,
                                  "fvsr_z": z.cpu(),
                                  "fvsr_LQ": decoder_lq.cpu(),
                                  "fvsr_pad": 0,
                                  "fvsr_version": version,
                                  "fvsr_plus": True,
                                  "fvsr_plus_meta": meta})

        from .stream_sampler import run_flashvsr_stream
        result = run_flashvsr_stream(
            patcher, lq_proj, posi, image, seed=seed, width=width, height=height,
            scale=scale, kv_ratio=kv_ratio, local_range=local_range,
            sparse_ratio=sparse_ratio, offload=offload, chunk_q=chunk_q,
            segmented=bool(long_mode))

        if long_mode:
            segments, lq_idx, LQ, pad_frames, meta = result
            out = {"samples": [s.cpu() for s in segments],
                   "fvsr_long": True,
                   "fvsr_lq_cur_idx": lq_idx,
                   "fvsr_LQ": LQ.cpu(),
                   "fvsr_pad": int(pad_frames),
                   "fvsr_version": model["version"]}
            return io.NodeOutput(out)

        z, LQ, pad_frames, meta = result

        # raw VAE latent space for standard VAEDecode compatibility
        mean = LATENTS_MEAN.to(device=z.device, dtype=z.dtype)
        std = LATENTS_STD.to(device=z.device, dtype=z.dtype)
        raw = z * std + mean

        out = {"samples": raw,
               "fvsr_z": z.cpu(),
               "fvsr_LQ": LQ.cpu(),
               "fvsr_pad": int(pad_frames),
               "fvsr_version": model["version"],
               "fvsr_lq_cur_idx": int(meta["LQ_cur_idx"])}
        return io.NodeOutput(out)


class FlashVSR_SM_Vae(io.ComfyNode):
    """Wrapper-style VAE loader: pick the decode pipeline ONCE here, the
    Decoder node uses whatever you selected.  tcd_encoder takes priority."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_Vae",
            display_name="FlashVSR Vae Loader",
            category="FlashVSR",
            inputs=[
                io.Combo.Input("vae", options=["none"] + folder_paths.get_filename_list("vae"),
                    tooltip="comfy VAE (e.g. Wan2.1_VAE.pth) — standard decode, supports tiling"),
                io.Combo.Input("tcd_encoder", options=_fvsr_list(lambda i: "tcd" in i.lower()),
                    tooltip="TCDecoder.ckpt — FlashVSR official tiny conditional decoder "
                            "(takes priority over vae; REQUIRED for long_mode and for plus)"),
            ],
            outputs=[io.Custom("FVSRN_VAE").Output(display_name="vae")],
        )

    @classmethod
    def execute(cls, vae, tcd_encoder) -> io.NodeOutput:
        tcd_path = folder_paths.get_full_path("FlashVSR", tcd_encoder) if tcd_encoder != "none" else None
        if tcd_path is not None:
            import comfy.model_management
            from ..FlashVSR.examples.WanVSR.utils.TCDecoder import build_tcdecoder
            module = build_tcdecoder(new_channels=[512, 256, 128, 128],
                                     new_latent_channels=16 + 768)
            state = torch.load(tcd_path, weights_only=False)
            if isinstance(state, dict) and "decoder" in state:
                state = state["decoder"]  # plus TCDecoder.ckpt wrapper
            report = module.load_state_dict(state, strict=False)
            total = len(module.state_dict())
            miss = len(report.missing_keys)
            if miss > total * 0.05:
                print(f"[FlashVSR Native][WARN] TCDecoder coverage low: {total - miss}/{total} "
                      "— wrong checkpoint for this pipeline?")
            module = module.to(dtype=torch.bfloat16)
            module.eval()
            if hasattr(module, "clean_mem"):
                module.clean_mem()
            return io.NodeOutput({"kind": "tcd", "module": module})

        if vae != "none":
            import comfy.sd
            import comfy.utils
            path = folder_paths.get_full_path("vae", vae)
            sd = comfy.utils.load_torch_file(path, safe_load=True)
            vae_obj = comfy.sd.VAE(sd=sd)
            return io.NodeOutput({"kind": "comfy", "vae": vae_obj})

        raise ValueError("select a vae or tcd_encoder")


class FlashVSR_SM_Decoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_Decoder",
            display_name="FlashVSR Decoder",
            category="FlashVSR",
            inputs=[
                io.Latent.Input("latent"),
                io.Custom("FVSRN_VAE").Input("vae"),
                io.Boolean.Input("tiled", default=True,
                    tooltip="tile the comfy-VAE decode (Wan2.1 VAE full decode easily OOMs at 720p+). "
                            "TCDecoder path uses its official segment decode instead."),
                io.Int.Input("tile_size", default=64, min=16, max=4096, step=8,
                             tooltip="VAE spatial tile in LATENT pixels (output px = x8). "
                                     "2x-sr 1664x1024 -> latent 128x208, so 512 (SD default) never tiles! "
                                     "64 latent = 512x512 px per tile, ~wrapper's 60x104."),
                io.Int.Input("overlap", default=16, min=8, max=4096, step=8,
                             tooltip="VAE tile overlap in LATENT pixels"),
                io.Int.Input("temporal_size", default=64, min=8, max=4096, step=4,
                             tooltip="VAE frames per decode segment"),
                io.Int.Input("temporal_overlap", default=8, min=4, max=4096, step=4),
                io.Boolean.Input("color_fix", default=True),
                io.Combo.Input("fix_method", options=["wavelet", "adain"]),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, latent, vae, tiled=True, tile_size=512, overlap=64,
                temporal_size=64, temporal_overlap=8,
                color_fix=True, fix_method="wavelet") -> io.NodeOutput:
        pad_frames = int(latent.get("fvsr_pad", 0))
        LQ = latent.get("fvsr_LQ", None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kind = vae.get("kind") if isinstance(vae, dict) else None
        if kind == "tcd":
            module = vae["module"]
            if latent.get("fvsr_long"):
                images = _decode_tcd_long(latent, module, device)
            else:
                images = _decode_tcd(latent, module, device)   # [F,H,W,C] 0-1, cpu
        elif kind == "comfy":
            raw = latent["samples"]
            vae_obj = vae["vae"]
            import comfy.utils
            # comfy's tiled_scale_multidim runs under @torch.inference_mode() and
            # its process_output does INPLACE updates on the returned inference
            # tensor -> must call it inside inference_mode too, then clone out.
            pbar = comfy.utils.ProgressBar(1)
            with torch.inference_mode():
                if tiled:
                    pixel = vae_obj.decode_tiled(raw, tile_x=tile_size, tile_y=tile_size,
                                                 overlap=overlap, tile_t=temporal_size,
                                                 overlap_t=temporal_overlap)
                else:
                    pixel = vae_obj.decode(raw)          # [B,F,H,W,C] 0-1
            pixel = pixel.clone()
            pbar.update(1)
            if pixel.dim() == 5:
                pixel = pixel.reshape(-1, pixel.shape[-3], pixel.shape[-2], pixel.shape[-1])
            images = pixel.clamp(0, 1).cpu().float()
        else:
            raise ValueError("connect a FlashVSR Native Vae Loader output")

        # colour correction against the LQ video
        if color_fix and LQ is not None:
            from .color_fix import TorchColorCorrectorWavelet
            lq = LQ.float().cpu()                     # [1,3,F,tH,tW] in [-1,1]
            lq = ((lq + 1.0) * 0.5).clamp(0, 1)
            while lq.shape[2] < images.shape[0]:      # pad LQ frames if needed
                lq = torch.cat([lq, lq[:, :, -1:]], dim=2)
            hq = images.permute(3, 0, 1, 2).unsqueeze(0)   # [1,3,F,H,W]
            lq = lq[:, :, :hq.shape[2]]
            if lq.shape[-2:] != hq.shape[-2:]:
                lq = torch.nn.functional.interpolate(
                    lq, size=hq.shape[-2:], mode="bilinear", align_corners=False)
            hq = TorchColorCorrectorWavelet(levels=5)(hq, lq, clip_range=(0, 1),
                                                     method=fix_method, chunk_size=16)
            images = hq[0].permute(1, 2, 3, 0).clamp(0, 1)

        if pad_frames > 0:
            images = images[:-pad_frames]
        return io.NodeOutput(images.float())


def _decode_tcd_long(latent, vae, device):
    """tiny_long mode: per-segment TCDecoder decode (replicates the wrapper's
    decode_latents long-mode tcd branch: segments come with their own
    (LQ_cur_idx, LQ_pre_idx) conditioning windows)."""
    segments = latent["samples"]                 # list of [1,16,n,h,w]
    lq_idx = latent.get("fvsr_lq_cur_idx", [])
    LQ = latent.get("fvsr_LQ", None)
    pad_frames = int(latent.get("fvsr_pad", 0))
    if LQ is None:
        raise ValueError("latent missing fvsr_LQ")

    vae = vae.to(device=device)
    if hasattr(vae, "clean_mem"):
        vae.clean_mem()

    import comfy.utils
    frames_total = []
    pbar = comfy.utils.ProgressBar(len(segments))
    with torch.inference_mode():
        for seg, (cur_idx, pre_idx) in zip(segments, lq_idx):
            cur = seg.to(device=device, dtype=torch.bfloat16)
            cond = LQ[:, :, pre_idx:cur_idx, :, :].to(device=device, dtype=torch.bfloat16)
            out = vae.decode_video(cur.transpose(1, 2), parallel=False,
                                   show_progress_bar=False, cond=cond)
            out = out.transpose(1, 2).mul_(2.0).sub_(1.0)       # [1,3,f,H,W] in [-1,1]
            frames_total.append(out.to("cpu"))
            del cur, cond, out
            pbar.update(1)
    if hasattr(vae, "clean_mem"):
        vae.clean_mem()
    vae = vae.to("cpu")
    torch.cuda.empty_cache()

    images = torch.cat(frames_total, dim=2)                     # [1,3,F,H,W] in [-1,1]
    images = ((images[0].float() + 1.0) * 0.5).clamp(0, 1)      # [3,F,H,W] 0-1
    images = images.permute(1, 2, 3, 0)                         # [F,H,W,C]
    if pad_frames > 0:
        images = images[:-pad_frames]
    return images


def _decode_tcd(latent, vae, device):
    """TCDecoder path — replicates model_loader_utils.decode_latents tcd branch
    (1.x) and flashvsr_plus decode_plus_latents (plus, segment-wise)."""
    z = latent["fvsr_z"]                      # [1,16,N,h,w] DiT space, cpu bf16
    LQ = latent.get("fvsr_LQ", None)
    is_plus = bool(latent.get("fvsr_plus", False))
    lq_cur_idx = int(latent.get("fvsr_lq_cur_idx", 0))
    if LQ is None:
        raise ValueError("latent missing fvsr_LQ (sampler output required)")

    vae = vae.to(device=device)
    if hasattr(vae, "clean_mem"):
        vae.clean_mem()

    import comfy.utils
    dtype = torch.bfloat16
    zz = z.to(device=device, dtype=dtype)
    lq = LQ.to(device=device, dtype=dtype)
    outputs = []
    with torch.inference_mode():
        if is_plus:
            # segment-wise decode (official plus deferred-decode semantics)
            n = zz.shape[2]
            pbar = comfy.utils.ProgressBar((n + 3) // 4)
            start = 0
            while start < n:
                end = min(start + 4, n)
                first = start == 0
                frame_start = 0 if first else 4 * start - 3
                frame_count = 4 * (end - start) - (3 if first else 0)
                cur = zz[:, :, start:end]
                cur_lq = lq[:, :, frame_start:frame_start + frame_count]
                out = vae.decode_video(cur.transpose(1, 2), parallel=False,
                                       show_progress_bar=False, cond=cur_lq)
                outputs.append(out.transpose(1, 2).to("cpu"))
                del cur, cur_lq, out
                pbar.update(1)
                start = end
            video = torch.cat(outputs, dim=2).to(device)   # [1,3,F,H,W] in [0,1]
            frames = video.mul_(2.0).sub_(1.0)             # -> [-1,1]
        else:
            n = zz.shape[2]
            pbar = comfy.utils.ProgressBar(1)
            total_frames = 4 * n - 3
            cond = lq[:, :, :lq_cur_idx if lq_cur_idx > 0 else total_frames]
            if cond.shape[2] < total_frames:
                cond = torch.cat([cond, cond[:, :, -1:].repeat(1, 1, total_frames - cond.shape[2])], dim=2)
            frames = vae.decode_video(zz.transpose(1, 2), parallel=False,
                                      show_progress_bar=False, cond=cond)
            frames = frames.transpose(1, 2).mul_(2.0).sub_(1.0)   # [1,3,F,H,W] in [-1,1]
            pbar.update(1)
    if hasattr(vae, "clean_mem"):
        vae.clean_mem()
    vae = vae.to("cpu")
    torch.cuda.empty_cache()

    if is_plus:
        meta = latent.get("fvsr_plus_meta", {})
        src_frames = int(meta.get("source_frames", frames.shape[2]))
        repeat = src_frames  # placeholder; real merge below when compression < 4
        tc = int(meta.get("temporal_compression", 4))
        factor = 4 // tc
        if factor > 1:
            from ..flashvsr_plus.inference.preprocessing import merge_temporal_output, TemporalExpansion
            expansion = TemporalExpansion(
                video=None,
                source_frames=src_frames,
                effective_frames=src_frames,
                expanded_frames=frames.shape[2],
                repeat_factor=factor,
            )
            frames = merge_temporal_output(
                frames.float().cpu(), expansion,
                mode=meta.get("temporal_merge", "gaussian"),
                gaussian_sigma=float(meta.get("temporal_gaussian_sigma", 1.0)),
            ).to(frames)
        frames = frames[:, :, :src_frames]
    images = ((frames[0].float() + 1.0) * 0.5).clamp(0, 1)   # [3,F,H,W] 0-1
    return images.permute(1, 2, 3, 0).cpu()                  # [F,H,W,C]
