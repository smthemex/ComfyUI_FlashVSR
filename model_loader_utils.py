# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
from PIL import Image
import numpy as np
import cv2
import comfy.model_management as mm
import torch.nn.functional as F
from comfy.utils import common_upscale,ProgressBar
import folder_paths
from diffusers import AutoencoderKLWan
from .FlashVSR.vae import WanVAE
from .FlashVSR.diffsynth.pipelines.flashvsr_full import TorchColorCorrectorWavelet
from .FlashVSR.examples.WanVSR.infer_flashvsr_full import upscale_lq_video_bilinear,dup_first_frame_1cthw_simple
from safetensors.torch import load_file
cur_path = os.path.dirname(os.path.abspath(__file__))


def load_vae_model(vae,tcd_encoder,lite_vae, device="cuda"):
    vae_path=folder_paths.get_full_path("vae", vae) if vae != "none" else None
    tcd_encoder_path=folder_paths.get_full_path("FlashVSR", tcd_encoder) if tcd_encoder != "none" else None
    lite_vae_path=folder_paths.get_full_path("vae", lite_vae) if lite_vae != "none" else "none"
    if vae_path is not None:
        if vae_path.endswith(".safetensors"):
            config=AutoencoderKLWan.load_config(os.path.join(cur_path,"FlashVSR/config.json"))
            vae=AutoencoderKLWan.from_config(config).to(device,dtype=torch.bfloat16)
            vae_dict=load_file(vae_path,device="cpu")
            vae.load_state_dict(vae_dict,strict=False)
            del vae_dict
            vae.vae_mode="diffusers"
        else:
            from .FlashVSR.diffsynth import ModelManager,FlashVSRFullPipeline
            mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
            mm.load_models([vae_path,])
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device="cuda")
            pipe.vae.model.encoder = None
            pipe.vae.model.conv1 = None
            pipe.enable_vram_management(num_persistent_param_in_dit=None,vae_only=True)
            pipe.load_models_to_device(["vae"])
            vae=pipe.vae
            vae.vae_mode="normal"
            del pipe
    elif tcd_encoder_path is not None:
        from .FlashVSR.examples.WanVSR.utils.TCDecoder import build_tcdecoder
        vae = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16+768)
        tcd_state=torch.load(tcd_encoder_path,weights_only=False,)
        if isinstance(tcd_state,dict) and "decoder" in tcd_state:
            tcd_state=tcd_state["decoder"]  # 兼容 FlashVSR+ 的 TCDecoder.ckpt 包装结构
        load_report=vae.load_state_dict(tcd_state, strict=False)
        total=len(vae.state_dict())
        miss=len(load_report.missing_keys); unexp=len(load_report.unexpected_keys)
        print(f"[FlashVSR] TCDecoder load: matched {total-miss}/{total}, missing={miss}, unexpected={unexp}")
        if miss > total*0.05:
            print("[FlashVSR][WARN] TCDecoder 权重覆盖率过低，解码质量会严重异常！"
                  "请确认 tcd_encoder 选择的是 FlashVSR+ 发布的 TCDecoder.ckpt")
        vae.vae_mode="tcd"
    elif  lite_vae_path is not None:
        if "light" in lite_vae_path.lower() or "tae" in lite_vae_path.lower():
            if os.path.basename(lite_vae_path).split(".")[0]=="lightvaew2_1":
                print("use lightvae decoder")
                vae = WanVAE(vae_path=lite_vae_path,dtype=torch.bfloat16,device=device,use_lightvae=True)
                vae.vae_mode="lightvae"
            elif os.path.basename(lite_vae_path).split(".")[0]=="taew2_1":
                from .FlashVSR.vae_tiny import WanVAE_tiny
                print("use vae_tiny decoder")
                vae = WanVAE_tiny(vae_path=lite_vae_path,dtype=torch.bfloat16,device=device,need_scaled=False)
                vae.vae_mode="tae"
            elif os.path.basename(lite_vae_path).split(".")[0]=="lighttaew2_1":
                from .FlashVSR.vae_tiny import WanVAE_tiny
                print("use vae_tiny light decoder")
                vae = WanVAE_tiny(vae_path=lite_vae_path,dtype=torch.bfloat16,device=device,need_scaled=True)
                vae.vae_mode="lighttae"
            else:
                raise ValueError(f"Unknown vae_name: {lite_vae_path},only support lightvae,tae,tae_tiny,lighttae_tiny")
        else:    
            print("use upscale2x decoder")
            config=AutoencoderKLWan.load_config(os.path.join(cur_path,"FlashVSR/examples/config.json"))
            vae=AutoencoderKLWan.from_config(config).to(device,dtype=torch.bfloat16)
            vae_dict=load_file(lite_vae_path,device="cpu")
            vae.load_state_dict(vae_dict,strict=False)              
            del vae_dict  
            vae.vae_mode="upscale2x"
    else:
        raise ValueError("vae_path,tcd_encoder_path,lite_vae_path is None")
    if not vae.vae_mode=="lightvae":
        vae.eval()
        vae.requires_grad_(False)
    return vae

def decode_latents( vae,latent,color_fix,fix_method,full_tiled, device):
    vae_mode=getattr(vae,"vae_mode","none")
    samples=latent["samples"]
    LQ_cur_idx=latent.get("LQ_cur_idx",0)
    pad_frames=latent.get("pad_frames",0)
    LQ=latent.get("LQ",None)
    if color_fix:
        ColorCorrector=TorchColorCorrectorWavelet(levels=5)
    pad_first_frame = True  if "wavelet"== fix_method and color_fix else False
    if isinstance(samples,list) and isinstance(LQ_cur_idx,list) and vae_mode!="tcd":
        raise ValueError("when use long mode  only  support tcd_encoder")
    #print(samples.shape) #torch.Size([1, 16, 22, 160, 96])
    if vae_mode=="none":
        samples=add_mean(samples)
        images=vae.decode(samples)
        #print(f"decode complete,images shape: {images.shape}") # torch.Size([1, 85, 1024, 768, 3])
        images=images.permute(0,4,1,2,3) # torch.Size([1, 3, 85, 1024, 768])
    else:
        if not vae_mode in ["normal","lightvae"]:
            vae.to(device)
        with torch.no_grad():
            if isinstance(vae,AutoencoderKLWan):
                samples=add_mean(samples)
                if full_tiled:
                    vae.enable_tiling()
                else:
                    vae.disable_tiling()
                images=vae.decode(samples,return_dict=False)[0]
                images = map_neg1_1_to_0_1(images.cpu().float())
                if vae_mode=="upscale2x":
                    ch = images.shape[1]
                    if ch == 3:
                        upscale = 1
                    else:  
                        if ch % 3 == 0:
                            upscale = round((ch // 3) ** 0.5)
                        else:
                            upscale = 1
                    images = F.pixel_shuffle(images.movedim(2, 1), upscale_factor=int(upscale)).movedim(1, 2) # pixel shuffle needs [..., C, H, W] format #torch.Size([1, 3, 77, 2048, 1536])
            else:
                if vae_mode=="tcd":
                    vae.clean_mem()
                    if isinstance(samples,list):
                        print("use tcd decoder to infer long mode")
                        frames_total = []
                        for i,(cur,pre) in zip(samples,LQ_cur_idx):
                            cur_LQ_frame = LQ[:,:,pre:cur,:,:]
                            cur_frames = vae.decode_video(i.transpose(1, 2),parallel=False, show_progress_bar=False, cond=LQ[:,:,pre:cur,:,:].to(device)).transpose(1, 2).mul_(2).sub_(1)
                            cur_frames = map_neg1_1_to_0_1(cur_frames.cpu().float())  #[21,8,8,8,8,8] #torch.Size([1, 3, 21, 1280, 768])
                            cur_LQ_frame = map_neg1_1_to_0_1(cur_LQ_frame.cpu().float()).to(device)
                            if color_fix:
                                try:
                                    if pad_first_frame: # 加帧
                                        cur_frames = dup_first_frame_1cthw_simple(cur_frames)
                                        cur_LQ_frame=dup_first_frame_1cthw_simple(cur_LQ_frame)
                                    cur_frames = ColorCorrector(
                                        cur_frames.to(device=device),
                                        cur_LQ_frame,
                                        clip_range=(-1, 1),
                                        chunk_size=None,
                                        method=fix_method,
                                    )
                                    if pad_first_frame: #减帧
                                        cur_frames = cur_frames[:, :, 1:, :, :] # remove first frame
                                except:
                                    print("color correction failed, return uncorrected images")
                            frames_total.append(cur_frames.to('cpu'))
                        images = torch.cat(frames_total, dim=2)
                        if pad_frames>0:
                            print(f"Remove{images.shape[2]} 's pad frames {pad_frames} --> original frames: {images.shape[2]-pad_frames}")
                            images = images[:, :, :-pad_frames, :, :]  #torch.Size([1, 3, 85, 1024, 768]) -> torch.Size([1, 3, 81, 1024, 768]) if 81 output -4 input + 8
                        return images.squeeze(0).permute(1,2,3,0).cpu()
                    else:
                        images = vae.decode_video(samples.transpose(1, 2),parallel=False, show_progress_bar=False, cond=LQ[:,:,:LQ_cur_idx,:,:]).transpose(1, 2).mul_(2).sub_(1)
                elif vae_mode=="normal":
                    vae.clear_cache()
                    images = vae.decode(samples, device=device, tiled=full_tiled, tile_size=(60, 104), tile_stride=(30, 52))
                else:
                    if isinstance(vae,WanVAE):
                        if full_tiled:
                            vae.use_tiling=True
                        else:
                            vae.use_tiling=False
                    images=vae.decode(samples.squeeze(0))
                images = map_neg1_1_to_0_1(images.cpu().float())
            if  not vae_mode=="lightvae":
                vae.to("cpu")
            torch.cuda.empty_cache()
    #print(f"decode complete,images shape: {images.shape}")
    
    if color_fix:
        LQ = map_neg1_1_to_0_1(LQ.cpu().float())
        try:
            if pad_first_frame:
                images = dup_first_frame_1cthw_simple(images)
                LQ=dup_first_frame_1cthw_simple(LQ)
            if vae_mode=="upscale2x" and LQ.shape[-1]!=images.shape[-1]:
                scale_=int(images.shape[-1]/LQ.shape[-1])
                LQ=upscale_lq_video_bilinear(LQ,scale_)
            images = ColorCorrector(
                images.to(device=device),
                LQ[:, :, :images.shape[2], :, :],
                clip_range=(-1, 1),
                chunk_size=16,
                method=fix_method
            )
            if pad_first_frame:
                images = images[:, :, 1:, :, :] # remove first  frame torch.Size([1, 3, 78, 1024, 768]) --> torch.Size([1, 3, 77, 1024, 768])
        except:
            print("color correction failed, return uncorrected images")
    
    if pad_frames>0:
        print(f"Remove{images.shape[2]} 's pad frames {pad_frames} --> original frames: {images.shape[2]-pad_frames}")
        images = images[:, :, :-pad_frames, :, :]  #torch.Size([1, 3, 85, 1024, 768]) -> torch.Size([1, 3, 81, 1024, 768]) if 81 output -4 input + 8
    return images.squeeze(0).permute(1,2,3,0).cpu()





def clear_comfyui_cache():
    cf_models=mm.loaded_models()
    try:
        for pipe in cf_models:
            pipe.unpatch_model(device_to=torch.device("cpu"))
            print(f"Unpatching models.{pipe}")
    except: pass
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    max_gpu_memory = torch.cuda.max_memory_allocated()
    print(f"After Max GPU memory allocated: {max_gpu_memory / 1000 ** 3:.2f} GB")

def add_mean(latents):
    vae_config={"latents_mean": [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921
        ],
        "latents_std": [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.916
        ],}
    latents_mean = (torch.tensor(vae_config["latents_mean"]).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype))
    latents_std = 1.0 / torch.tensor(vae_config["latents_std"]).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    return latents

def gc_cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def tensor2cv(tensor_image):
    if len(tensor_image.shape)==4:# b hwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def phi2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img

def tensor2image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2pillist(tensor_in):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [tensor2image(tensor_in)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[tensor2image(i) for i in tensor_list]
    return img_list

def tensor2pillist_upscale(tensor_in,width,height):
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        img_list = [nomarl_upscale(tensor_in,width,height)]
    else:
        tensor_list = torch.chunk(tensor_in, chunks=d1)
        img_list=[nomarl_upscale(i,width,height) for i in tensor_list]
    return img_list

def tensor2list(tensor_in,width,height):
    if tensor_in is None:
        return None
    d1, _, _, _ = tensor_in.size()
    if d1 == 1:
        tensor_list = [tensor_upscale(tensor_in,width,height)]
    else:
        tensor_list_ = torch.chunk(tensor_in, chunks=d1)
        tensor_list=[tensor_upscale(i,width,height) for i in tensor_list_]
    return tensor_list


def tensor_upscale(tensor, width, height):
    samples = tensor.movedim(-1, 1)
    samples = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = samples.movedim(1, -1)
    return samples

def nomarl_upscale(img, width, height):
    samples = img.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img = tensor2image(samples)
    return img



def cv2tensor(img,bgr2rgb=True):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).permute(1, 2, 0).unsqueeze(0)  



def images_generator(img_list: list, ):
    # get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_, Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_, np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in = img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in, np.ndarray):
            i = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            print(i.shape)
            return i
        else:
            raise "unsupport image list,must be pil,cv2 or tensor!!!"
    
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image
        
def map_neg1_1_to_0_1(t):
    """
    接受 torch.Tensor 或可转 torch.Tensor 的输入。
    可处理形状: H,W,C 或 B,H,W,C 或 B,T,H,W,C（会按元素处理）。
    返回 float tensor，范围 0..1。
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.float()
    # map -1..1 -> 0..1
    t = (t + 1.0) * 0.5
    # 限幅到 [0,1]
    t = t.clamp(0.0, 1.0)
    # 保持在 cpu 端，调用方可决定是否转 device/dtype
    return t.cpu()

def load_images_list(img_list: list, ):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images

def get_video_files(directory, extensions=None):
    if extensions is None:
        extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']
    extensions = [ext.lower() for ext in extensions]
    video_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            ext = ext.lower()[1:] 
            if ext in extensions:
                full_path = os.path.join(root, file)
                video_files.append(full_path)             
    return video_files
