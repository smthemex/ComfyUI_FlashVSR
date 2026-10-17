 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
import math
import weakref
from .model_loader_utils import  tensor_upscale,decode_latents,clear_comfyui_cache,load_vae_model
from .FlashVSR.examples.WanVSR.infer_flashvsr_full import init_pipeline,run_inference
from .FlashVSR.examples.WanVSR.infer_flashvsr_tiny_long_video import init_pipeline_long
import folder_paths
from comfy_api.latest import  io, ui, Types
import nodes
from comfy_api.input_impl import VideoFromFile

MAX_SEED = np.iinfo(np.int32).max
node_cr_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

weigths_FlashVSR_current_path = os.path.join(folder_paths.models_dir, "FlashVSR")
if not os.path.exists(weigths_FlashVSR_current_path):
    os.makedirs(weigths_FlashVSR_current_path)

folder_paths.add_model_folder_path("FlashVSR", weigths_FlashVSR_current_path) #  FlashVSR dir


class FlashVSR_SM_Model(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        
        return io.Schema(
            node_id="FlashVSR_SM_Model",
            display_name="FlashVSR_SM_Model",
            category="FlashVSR",
            inputs=[
                io.Combo.Input("dit",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if ("dmd" in i.lower() or "dit" in i.lower())]),
                io.Combo.Input("proj_pt",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "proj" in i.lower()]),
                io.Combo.Input("emb_pt",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "prompt" in i.lower()]),
                io.Combo.Input("version",options= ["1.1","1.0","plus_turbo","plus_base"] ),
                io.Boolean.Input("tiny_long", default=False),
                io.Boolean.Input("offload", default=False),
                io.Combo.Input("plus_temporal",options= ["4","2","1"] ),
                io.Combo.Input("plus_attn",options= ["sdpa","auto","block_sparse","triton_block_sparse"] ),
            ],
            outputs=[
                io.Custom("FlashVSR_SM_Model").Output(),
                ],
            )
    @classmethod
    def execute(cls, dit,proj_pt,emb_pt,version,tiny_long,offload,plus_mode="turbo",plus_temporal="4",plus_attn="sdpa") -> io.NodeOutput:
        dit_path=folder_paths.get_full_path("FlashVSR", dit) if dit != "none" else None
        proj_pt_path=folder_paths.get_full_path("FlashVSR", proj_pt) if proj_pt != "none" else None
        prompt_path=folder_paths.get_full_path("FlashVSR", emb_pt) if emb_pt != "none" else None
        assert prompt_path is not None and dit_path is not None and proj_pt is not None  , "Please select the emb pt,dit,proj_pt file"
        if version == "plus_turbo" or version == "plus_base":
            # ---- FlashVSR+ (plus)：只加载 denoiser（TinyDecoder 由 Vae 节点 tcd_encoder 加载，
            # 与 1.0/1.1 的 tcd 模式一致），解码在 Decoder 节点完成 ----
            from .flashvsr_plus_node import build_plus_pipeline
            plus_mode="turbo" if version == "plus_turbo" else "base"
            model = build_plus_pipeline(dit_path, proj_pt_path, plus_mode, plus_temporal, offload, plus_attn, prompt_path)
            return io.NodeOutput(model)
        if tiny_long:
            model=init_pipeline_long(prompt_path,proj_pt_path,dit_path, device="cuda",offload=offload,version=version)
        else:
            model=init_pipeline(prompt_path,proj_pt_path,dit_path,node_cr_path ,device="cuda",offload=offload,version=version)
        return io.NodeOutput(model)
    

class FlashVSR_SM_KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_KSampler",
            display_name="FlashVSR_SM_KSampler",
            category="FlashVSR",
            inputs=[
                io.Custom("FlashVSR_SM_Model").Input("model"),
                io.Image.Input("image"),
                io.Int.Input("width", default=1280, min=128, max=nodes.MAX_RESOLUTION,step=64,display_mode=io.NumberDisplay.number),
                io.Int.Input("height", default=768, min=128, max=nodes.MAX_RESOLUTION,step=64,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("scale", default=4, min=1, max=4),
                io.Float.Input("kv_ratio", default=3.5, min=0.0, max=10.0, step=0.1, round=0.01,),
                io.Int.Input("local_range", default=11, min=1,step=1, max=50),
                io.Int.Input("steps", default=1, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,),
                io.Float.Input("sparse_ratio", default=2.0, min=0.0, max=10.0, step=0.1,), 
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )
    @classmethod
    def execute(cls, model,image,width,height,seed,scale,kv_ratio,local_range, steps, cfg,sparse_ratio,) -> io.NodeOutput:
        image=tensor_upscale(image,width, height)
        from .flashvsr_plus_node import is_plus_pipeline
        if is_plus_pipeline(model):
            # ---- FlashVSR+ (plus)：流式去噪 -> 延迟解码 latent，TinyDecoder 在 Decoder 节点执行 ----
            from .flashvsr_plus_node import run_plus_denoise
            print("[FlashVSR+] plus 模式: steps/cfg/sparse_ratio/kv_ratio/local_range/scale 参数不生效"
                  "(plus 为固定 1-step)")
            latent_out=run_plus_denoise(model,image,width,height,seed,scale)
            return io.NodeOutput(latent_out)
        
        offload=getattr(model,"offload",False)
        clear_comfyui_cache()
        print(f"start inference with seed:{seed}")
        images,LQ_cur_idx,LQ,pad_frames=run_inference(model,image,seed,scale,kv_ratio,local_range,steps,cfg,sparse_ratio,offload)
        outputs={"samples": images,"LQ_cur_idx":LQ_cur_idx,"LQ":LQ,"pad_frames":pad_frames}
        return io.NodeOutput(outputs)
    
class FlashVSR_SM_Vae(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_Vae",
            display_name="FlashVSR_SM_Vae",
            category="FlashVSR",
            inputs=[
                io.Combo.Input("vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
                io.Combo.Input("tcd_encoder",options= ["none"] + [i for i in folder_paths.get_filename_list("FlashVSR") if "tcd" in i.lower()] ),
                io.Combo.Input("lite_vae",options= ["none"] + folder_paths.get_filename_list("vae") ),
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
            ],
        )
    @classmethod
    def execute(cls, vae,tcd_encoder,lite_vae) -> io.NodeOutput:
        clear_comfyui_cache()
        vae=load_vae_model(vae,tcd_encoder,lite_vae, device)   
        return io.NodeOutput(vae)

class FlashVSR_SM_Decoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_Decoder",
            display_name="FlashVSR_SM_Decoder",
            category="FlashVSR",
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Boolean.Input("full_tiled", default=True),
                io.Boolean.Input("color_fix", default=True),
                io.Combo.Input("fix_method",options= ["wavelet","adain"]),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )
    @classmethod
    def execute(cls, vae,latent,full_tiled,color_fix,fix_method) -> io.NodeOutput:
        # ---- FlashVSR+ (plus) 分支：用 Vae 节点加载的 TCDecoder(tcd_encoder) 解码延迟 latent ----
        if isinstance(latent,dict) and latent.get("flashvsr_plus_deferred"):
            from .flashvsr_plus_node import decode_plus_latents
            images=decode_plus_latents(latent,vae,bool(color_fix),fix_method)
            return io.NodeOutput(images)
        images=decode_latents( vae,latent,color_fix,fix_method,full_tiled, device)
        return io.NodeOutput(images)


# ---- 切分预览缓存：同一 trimmed 视频对象复用同一段临时预览文件，避免重复转码 ----
_trim_preview_results: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _save_trim_preview(video) -> ui.PreviewVideo:
    """把迭代切分出的视频段落写入临时目录并返回预览（参考官方 VideoTrim 的 save_video_preview）"""
    cached = _trim_preview_results.get(video)
    if cached is not None and os.path.isfile(cached[0]):
        return ui.PreviewVideo([cached[1]])

    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        "FlashVSR_temp_video", folder_paths.get_temp_directory(), 0, 0
    )
    file = f"{filename}_{counter:05}_.{Types.VideoContainer.get_extension(Types.VideoContainer.MP4)}"
    full_path = os.path.join(full_output_folder, file)
    video.save_to(
        full_path,
        format=Types.VideoContainer.MP4,
        codec="auto",
        preset="ultrafast",
    )
    result = ui.SavedResult(file, subfolder, io.FolderType.temp)
    _trim_preview_results[video] = (full_path, result)
    return ui.PreviewVideo([result])


class FlashVSR_SM_VideoTrimLoop(io.ComfyNode):
    """
    可视化视频循环切分节点（参考官方 LoadVideo + VideoTrim）。

    直接在节点上选择一个完整视频文件（无需预先切分），
    每次迭代按 split_length（秒）输出下一段视频，输出到最后一段后从头循环。
    """
    @classmethod
    def __init__(cls):
        cls.counters = {}

    @classmethod
    def define_schema(cls):
        # 与官方 LoadVideo 一致：列出 input 目录下可播放的视频文件，并支持上传按钮
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="FlashVSR_SM_VideoTrimLoop",
            display_name="FlashVSR_SM_VideoTrimLoop",
            category="FlashVSR",
            has_intermediate_output=True,
            inputs=[
                io.Combo.Input("video_file", options=sorted(files), upload=io.UploadType.video,
                               tooltip="选择完整视频文件，无需预先切分"),
                io.Float.Input("split_length", default=5.0, min=0.1, max=3600.0, step=0.1,
                               tooltip="每次迭代输出的切分时长（秒）"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Int.Input("reset_bool", default=0, optional=True,
                             tooltip="输入值 >= 1 时，迭代计数器归零（从头开始）"),
            ],
            outputs=[
                io.Video.Output(),
                io.Custom("NUMBER").Output(display_name="number"),
                io.Int.Output(display_name="seed"),
                io.String.Output(display_name="filename"),
                io.Float.Output(display_name="start_time"),
            ],
        )

    @classmethod
    def execute(cls, video_file, split_length, seed, reset_bool=0) -> io.NodeOutput:
        video_path = folder_paths.get_annotated_filepath(video_file)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Invalid video file: {video_file}")

        # ---- 迭代计数器：以文件绝对路径为 key ----
        UNIQUE_ID = os.path.normpath(video_path)
        start = 0
        step = 1
        counter = start
        if cls.counters.__contains__(UNIQUE_ID):
            counter = cls.counters[UNIQUE_ID]
        if round(reset_bool) >= 1:
            counter = start
        counter += step
        cls.counters[UNIQUE_ID] = counter
        result = int(counter)

        source = VideoFromFile(video_path)
        total_duration = float(source.get_duration())
        split_length = max(float(split_length), 0.1)
        num_segments = max(1, math.ceil(total_duration / split_length))

        # ---- 本次迭代对应的切分段：超出总段数后回到第 1 段（循环） ----
        segment_index = (result - 1) % num_segments
        start_time = segment_index * split_length
        seg_duration = min(split_length, max(total_duration - start_time, 0.1))

        trimmed = source.as_trimmed(start_time, seg_duration, strict_duration=False)
        if trimmed is None:
            raise ValueError(
                f"Failed to trim video:\nSource duration: {total_duration}\n"
                f"Start time: {start_time}\nTarget duration: {seg_duration}"
            )

        print(f"[FlashVSR_SM_VideoTrimLoop] segment {segment_index + 1}/{num_segments}: "
              f"start={start_time:.3f}s duration={seg_duration:.3f}s ({video_file})")

        filename = os.path.basename(video_path)
        return io.NodeOutput(
            trimmed, result, seed, filename, float(start_time),
            ui=_save_trim_preview(trimmed),
        )

    @classmethod
    def validate_inputs(cls, video_file, **kwargs):
        if not folder_paths.exists_annotated_filepath(video_file):
            return "Invalid video file: {}".format(video_file)
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # 每次入队都重新执行，保证迭代计数器推进
        return ""





