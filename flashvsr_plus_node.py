# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""FlashVSR+ (plus) 集成层。

职责：
1. 用包内 flashvsr_plus（命名空间化后的上游推理代码）构建推理 pipeline；
   权重全部由用户在节点入参上手选（dit / proj_pt / emb_pt / plus_tcd）；
2. run_plus_denoise：KSampler 阶段只做流式去噪，输出 latent + LQ 条件帧
   （延迟解码，与 1.1 的 Model -> KSampler -> Decoder 拆分结构对齐）；
3. decode_plus_latents：Decoder 阶段用 TinyDecoder 解码 latent 输出图像。

不动 ComfyUI 本体源码，也不改动原有 1.0 / 1.1 推理链路。
"""

import os

PLUS_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PLUS_CONTEXT_PATH = os.path.join(PLUS_PKG_DIR, "flashvsr_plus", "prompt_tensor", "posi_prompt.pth")


def is_plus_pipeline(model) -> bool:
    """判断 Model 节点输出是否为 FlashVSR+ pipeline（懒加载类型判断）。"""
    if model is None:
        return False
    try:
        from .flashvsr_plus import FlashVSRInferencePipeline
    except Exception:
        return False
    return isinstance(model, FlashVSRInferencePipeline)


def build_plus_pipeline(
    dit_path,
    proj_pt_path,
    plus_mode="turbo",
    plus_temporal="4",
    offload=False,
    plus_attn="sdpa",
    emb_pt_path=None,
    device="cuda:0",
):
    """构建 FlashVSR+ 推理 pipeline（只加载 denoiser，TinyDecoder 不在此反序列化）。

    权重由用户在节点上直接选择（与 1.0/1.1 共用同一套路径解析与断言）：
      dit_path      -> plus 的 dit.safetensors（对应节点 dit 入参）
      proj_pt_path  -> plus 的 lq_proj_in.safetensors（对应节点 proj_pt 入参）
      emb_pt_path   -> plus 的 posi_prompt.pth（对应节点 emb_pt 入参）
      plus_mode     -> 所选权重对应的 preset 配置（turbo=hybrid sliding 调度 /
                       base=sparse 调度），必须与权重匹配
    plus_attn: 注意力后端。默认 sdpa（所有 NVIDIA 卡通用、数值最稳）；
    block_sparse 仅建议 A100/H100/H200 等数据中心卡，triton_block_sparse 需
    安装 flashvsr_attn（Windows 一般不可用）。选了未安装的后端会在构建期报错。

    TinyDecoder（TCDecoder.ckpt）由 Vae 节点的 tcd_encoder 入参加载（与
    1.0/1.1 相同），解码在 Decoder 节点完成；因此 ModelPaths.tiny_decoder
    保持 None，跳过 decoder 反序列化。
    """
    import torch
    from .flashvsr_plus import ModelPaths, build_inference_pipeline, stream_config_for_mode

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FlashVSR+ (plus) 官方推理管线仅支持 NVIDIA CUDA GPU，"
            "当前环境没有可用的 CUDA 设备。"
        )

    if not dit_path or not os.path.isfile(dit_path):
        raise FileNotFoundError("plus 模式: dit 入参需选择 FlashVSR+ 的 dit.safetensors")
    if not proj_pt_path or not os.path.isfile(proj_pt_path):
        raise FileNotFoundError("plus 模式: proj_pt 入参需选择 FlashVSR+ 的 lq_proj_in.safetensors")
    if not emb_pt_path or not os.path.isfile(emb_pt_path):
        raise FileNotFoundError("plus 模式: emb_pt 入参需选择 FlashVSR+ 的 posi_prompt.pth")

    try:
        temporal = int(plus_temporal)
    except (TypeError, ValueError):
        temporal = 4
    if temporal not in (1, 2, 4):
        raise ValueError(f"plus_temporal 只支持 1/2/4，收到 {plus_temporal!r}")
    attn = str(plus_attn).lower()
    if attn not in ("auto", "sdpa", "block_sparse", "triton_block_sparse"):
        raise ValueError(f"plus_attn 不支持 {plus_attn!r}，可选 auto/sdpa/block_sparse/triton_block_sparse")

    print(
        f"[FlashVSR+] building pipeline: mode={plus_mode}, temporal_compression={temporal}, "
        f"attn={attn}, deferred_decode=True (TinyDecoder 由 Vae 节点加载)\n"
        f"  dit        = {dit_path}\n"
        f"  lq_proj_in = {proj_pt_path}\n"
        f"  posi_prompt= {emb_pt_path}"
    )
    config = stream_config_for_mode(
        plus_mode,
        device=device,
        temporal_compression=temporal,
        color_correction=False,  # 颜色校正走 Decoder 节点的 color_fix/fix_method 入参
        attn_kernel=attn,
    )
    config.decoder.deferred = True  # KSampler 只去噪，TinyDecoder 在 Decoder 节点执行
    paths = ModelPaths(
        dit=dit_path,
        lq_projection=proj_pt_path,
        context=emb_pt_path,
        tiny_decoder=None,  # 跳过 decoder 反序列化，TCDecoder 由 Vae 节点加载
    )
    pipeline = build_inference_pipeline(config, paths)
    # offload 语义与 1.0/1.1 一致：开启时加载节点只驻留 CPU，推理前上 CUDA、
    # 推理后卸回 CPU；关闭时 DiT 常驻 CUDA（多次推理不重复搬运）
    pipeline.offload = bool(offload)
    if pipeline.offload:
        pipeline.bundle.denoiser.to("cpu")
        decoder_module = getattr(pipeline.bundle.decoder, "module", None)
        if isinstance(decoder_module, torch.nn.Module):
            decoder_module.to("cpu")  # 随机初始化的占位 decoder 也不占显存
        torch.cuda.empty_cache()
        print("[FlashVSR+] denoiser loaded on CPU (offload=True), will move to CUDA on inference")
    else:
        print("[FlashVSR+] denoiser resident on CUDA (offload=False)")
    return pipeline


def run_plus_denoise(pipeline, image, width, height, seed, scale=4):
    """KSampler 阶段：图像张量 -> FlashVSR+ 流式去噪 -> 延迟解码 latent。

    尺寸语义与 1.0/1.1 完全一致（复刻 prepare_input_tensor）：
      1. width/height = 输入 LQ 的规整尺寸（tensor_upscale 到该尺寸）；
      2. scale = 放大倍数：LQ 先 bicubic 放大 scale 倍，再中心裁剪到 128 的倍数
         （向下取整，复刻 compute_scaled_and_target_dims + upscale_then_center_crop），
         模型在该分辨率上推理，输出即超分结果。

    返回 dict：
      samples                  [1,16,N,H,W] 去噪后 latent（cpu bf16）
      flashvsr_plus_deferred   True（Decoder 节点识别标记）
      flashvsr_plus_lq         [1,3,F,H,W] LQ 条件帧（cpu bf16，-1~1）
      flashvsr_plus_meta       时间轴元数据（merge/裁剪/颜色校正所需）
    TinyDecoder 不在此传递——由 Vae 节点（tcd_encoder）加载，Decoder 节点使用。
    """
    import torch
    from .flashvsr_plus import InferenceRequest
    from .model_loader_utils import tensor_upscale

    # 第一步：规整输入到用户设置的 LQ 尺寸
    if image.dim() == 3:
        image = image.unsqueeze(0)                           # 单图 [H,W,C] -> [1,H,W,C]
    image = tensor_upscale(image, int(width), int(height))   # [F,H,W,C] 0-1
    # 第二步：scale 倍放大 + 128 对齐中心裁剪（与 1.1 相同）
    try:
        scale_i = max(1, min(4, int(scale)))
    except (TypeError, ValueError):
        scale_i = 4
    h0, w0 = int(image.shape[1]), int(image.shape[2])
    sW, sH = w0 * scale_i, h0 * scale_i
    tW = max(128, (sW // 128) * 128)
    tH = max(128, (sH // 128) * 128)
    # scale 倍放大 + 中心裁剪：与 1.1 的 upscale_then_center_crop 完全一致（PIL BICUBIC、uint8 域）
    import numpy as np
    from PIL import Image
    u8 = (image.clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    left = max(0, (sW - tW) // 2)
    top = max(0, (sH - tH) // 2)
    frames = []
    for i in range(u8.shape[0]):
        img = Image.fromarray(u8[i]).convert("RGB")
        if (sW, sH) != (w0, h0):
            img = img.resize((sW, sH), Image.BICUBIC)
        img = img.crop((left, top, left + tW, top + tH))
        frames.append(torch.from_numpy(np.ascontiguousarray(np.asarray(img))).permute(2, 0, 1))
    lq = torch.stack(frames, 0).float().div_(127.5).sub_(1.0).permute(1, 0, 2, 3)  # [3,F,tH,tW] -1~1
    lq = lq.unsqueeze(0)                                     # [1,3,F,tH,tW]
    print(
        f"[FlashVSR+] input {w0}x{h0} (LQ) -> scale x{scale_i} -> {sW}x{sH} "
        f"-> crop {tW}x{tH} (128-aligned, 输出分辨率)"
    )

    pipeline.config.seed = int(seed) % (2 ** 31 - 1)
    device = torch.device(pipeline.config.device)
    from .model_loader_utils import clear_comfyui_cache
    clear_comfyui_cache()
    if getattr(pipeline, "offload", False):
        # offload 开启：推理前把 DiT 从 CPU 搬上 CUDA，推理完成卸载回 CPU
        print(f"[FlashVSR+] moving denoiser to {device} ...")
        pipeline.bundle.denoiser.to(device)
        try:
            request = InferenceRequest(lq_video=lq.contiguous(), fps=24.0)
            result = pipeline.infer(request, progress=lambda msg: print(f"[FlashVSR+] {msg}"))
        finally:
            pipeline.bundle.denoiser.to("cpu")
            torch.cuda.empty_cache()
            print("[FlashVSR+] denoiser offloaded back to CPU")
    else:
        # offload 关闭：DiT 常驻 CUDA，推理前后不搬运
        request = InferenceRequest(lq_video=lq.contiguous(), fps=24.0)
        result = pipeline.infer(request, progress=lambda msg: print(f"[FlashVSR+] {msg}"))
    meta = result.metadata
    latents = meta.get("latents")
    if latents is None:
        raise RuntimeError(
            "FlashVSR+ 未返回延迟解码 latent，请确认插件内 flashvsr_plus 副本已更新"
        )
    print(
        f"[FlashVSR+] denoise done: {latents.shape[2]} latent frames, "
        f"source_frames={meta.get('source_frames')} @ {tW}x{tH}, "
        f"attn_kernel={meta.get('attn_kernel')}"
    )
    return {
        "samples": latents,                                   # [1,16,N,H,W]
        "flashvsr_plus_deferred": True,
        "flashvsr_plus_lq": meta["decoder_lq"].to(torch.bfloat16),
        "flashvsr_plus_meta": {
            "source_frames": int(meta["source_frames"]),
            "effective_frames": int(meta["effective_input_frames"]),
            "expanded_frames": int(meta["expanded_model_frames"]),
            "repeat_factor": int(meta["temporal_repeat_factor"]),
            "temporal_merge": meta.get("temporal_merge", "gaussian"),
            "temporal_gaussian_sigma": float(meta.get("temporal_gaussian_sigma", 1.0)),
            # 注意：pipeline 层的 output_frames 在延迟解码时为 0，最终输出帧数取源帧数
            "frames": int(meta["source_frames"]),
            "fps": 24.0,
            "width": tW,
            "height": tH,
        },
    }


def decode_plus_latents(latent, vae, color_fix=False, fix_method="wavelet"):
    """Decoder 阶段：用 Vae 节点加载的 TCDecoder（tcd_encoder）解码 plus latent。

    执行方式与 1.0/1.1 的 decode_latents tcd 分支完全一致：逐 4-latent 段调用
    decode_video（mem 状态留在 GPU，cond 按段切片上卡，段输出即回 CPU）——
    不走 plus 包的 TinyDecoder wrapper（其分段间 mem 在 CPU/GPU 反复搬运，
    十几 GB 特征过 PCIe 导致分钟级解码与显存暴涨）。
    decode -> temporal merge -> 裁剪回源帧数 -> 可选低频小波颜色校正。
    输出 [F,H,W,C] 0-1 图像张量。
    """
    import torch
    from .flashvsr_plus.inference.preprocessing import (
        TemporalExpansion,
        merge_temporal_output,
    )
    from .flashvsr_plus.inference.postprocessing import WaveletColorCorrector

    latents = latent.get("samples")
    lq = latent.get("flashvsr_plus_lq")
    meta = latent.get("flashvsr_plus_meta") or {}
    if latents is None or lq is None:
        raise ValueError("latent 中缺少 FlashVSR+ 延迟解码所需数据")
    if vae is None:
        raise ValueError("plus 模式需要在 Vae 节点的 tcd_encoder 入参选择 TCDecoder.ckpt")
    if hasattr(vae, "module"):  # 兼容旧的 wrapper 包装
        vae = vae.module

    expansion = TemporalExpansion(
        video=None,  # merge 只需要时间轴标量，不占用 LQ 视频内存
        source_frames=meta["source_frames"],
        effective_frames=meta["effective_frames"],
        expanded_frames=meta["expanded_frames"],
        repeat_factor=meta["repeat_factor"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # plus 官方 TinyDecoder 即 bf16
    vae.to(device=device, dtype=dtype)
    vae.clean_mem()
    print(f"[FlashVSR+] TinyDecoder decoding {latents.shape[2]} latent frames (1.1-style, mem on GPU) ...")
    total_frames = 4 * latents.shape[2] - 3
    outputs = []
    start = 0
    try:
        with torch.inference_mode():
            while start < latents.shape[2]:
                end = min(start + 4, latents.shape[2])
                first = start == 0
                frame_start = 0 if first else 4 * start - 3
                frame_count = 4 * (end - start) - (3 if first else 0)
                cur = latents[:, :, start:end].to(device=device, dtype=dtype)
                cur_lq = lq[:, :, frame_start:frame_start + frame_count].to(device=device, dtype=dtype)
                out = vae.decode_video(
                    cur.transpose(1, 2), parallel=False, show_progress_bar=False, cond=cur_lq
                ).transpose(1, 2).mul_(2.0).sub_(1.0)
                outputs.append(out.to("cpu"))  # 段输出立即回 CPU，显存只留单段工作区
                del cur, cur_lq, out
                start = end
            video = torch.cat(outputs, dim=2)          # [1,3,F,tH,tW] -1~1 (cpu bf16)
            del outputs
    finally:
        vae.clean_mem()
        vae.to("cpu")                                   # 解码完成即卸载，不占显存
        torch.cuda.empty_cache()
    if video.shape[2] < total_frames:
        raise RuntimeError(f"TinyDecoder 输出帧数不足: {video.shape[2]} < {total_frames}")
    video = merge_temporal_output(
        video, expansion,
        mode=meta.get("temporal_merge", "gaussian"),
        gaussian_sigma=float(meta.get("temporal_gaussian_sigma", 1.0)),
    )
    video = video[:, :, : meta["frames"]].float()
    if color_fix:
        if str(fix_method).lower() != "wavelet":
            print("[FlashVSR+] plus 模式仅支持 wavelet 颜色校正，adain 已回退为 wavelet")
        video = WaveletColorCorrector(levels=5, chunk_size=16)(
            video, lq[:, :, : video.shape[2]].float()
        )
    frames = video[0].permute(1, 2, 3, 0).clamp(-1.0, 1.0)    # [F,H,W,C]
    frames = ((frames + 1.0) * 0.5).cpu().float()
    print(f"[FlashVSR+] decode done: {frames.shape[0]} frames @ {meta.get('width')}x{meta.get('height')}")
    return frames
