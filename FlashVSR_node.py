 # !/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from .model_loader_utils import  tensor_upscale,decode_latents,get_video_files,clear_comfyui_cache,load_vae_model
from .FlashVSR.examples.WanVSR.infer_flashvsr_full import init_pipeline,run_inference
from .FlashVSR.examples.WanVSR.infer_flashvsr_tiny_long_video import init_pipeline_long
import folder_paths
from comfy_api.latest import  io
import nodes
from pathlib import PureWindowsPath
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
                  "(plus 为固定 1-step、固定 4x 超分)")
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


class FlashVSR_SM_VideoPathLoop(io.ComfyNode):
    @classmethod
    def __init__(cls):
        cls.counters = {}
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FlashVSR_SM_VideoPathLoop",
            display_name="FlashVSR_SM_VideoPathLoop",
            category="FlashVSR",
            inputs=[
                io.String.Input("video_dir", multiline=False, default="/video"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Combo.Input("video_file", options=['none', 'webm', 'mp4', 'mkv', 'gif', 'mov']),
                io.Int.Input("reset_bool",default=0,optional=True),
            ],
            outputs=[
                io.Video.Output(),
                io.Custom("NUMBER").Output(display_name="number"),
                io.Int.Output(display_name="seed"),
                io.String.Output(display_name="filename"),
            ],
        )
    
    @classmethod
    def execute(cls, video_dir,seed,  video_file,reset_bool,) -> io.NodeOutput:
        video_path = PureWindowsPath(video_dir).as_posix() if video_dir else None
        video_file = None if video_file == 'none' else video_file
        assert video_path is not None, "video_dir is not set"
        UNIQUE_ID = os.path.normpath(video_path) 
        start = 0
        step = 1 
        counter =start
        if cls.counters.__contains__(UNIQUE_ID):
            counter = cls.counters[UNIQUE_ID]
        if round(reset_bool) >= 1:
            counter = start

        counter += step
        cls.counters[UNIQUE_ID] = counter
        result = int(counter)

        video_list = get_video_files(video_path, video_file)
        rows = len(video_list) if video_list else 0
        if rows == 0:
            assert False, "no video found"

        if result == 0:
            selected_path = video_list[0]
        else:
            adjusted_index = (result - 1) % rows
            selected_path = video_list[adjusted_index]
            
        print(f"Selected video path: {selected_path}")
        filename=os.path.basename(selected_path)
        return io.NodeOutput(VideoFromFile(selected_path),result, seed,filename)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
       return ""





