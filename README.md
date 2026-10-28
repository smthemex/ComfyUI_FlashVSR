# ComfyUI_FlashVSR
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR): Towards Real-Time Diffusion-Based Streaming Video Super-Resolution,this node ,you can use it in comfyUI 
[FlashVSR_Plus](https://github.com/OpenImagingLab/FlashVSR_Plus),FlashVSR+ is an efficient one-step diffusion framework for high-quality video super-resolution, enabling spatially untiled 4K inference on a single 24GB RTX 4090 GPU. 

#  Update
* 使用comfyUI的核心封装flashVSR，去掉transformer的版本限制，利用comfyUI的加卸载机制防止OOM（推理超出硬件能力会很慢，但不会OOM）/ Use comfyUI core flashVSR,remove transformer version limit,use comfyUI's add/remove mechanism to prevent OOM (inference will be very slow if it exceeds the hardware capabilities, but it will not OOM) 
* Use new loop / 换一个loop切分超长视频
* 支持flashVSR-Plus 模型，plus模式 请在菜单手动选triton-flashvsr_attn，否则还会是 自动选择Block-Sparse-Attention  ，support flashVSR-Plus model，plus use triton attention，or auto will choice Block-Sparse-Attention


1.Installation  
-----
In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_FlashVSR
```

2.requirements  
----
```
pip install -r requirements.txt
```
* 2.1 Best quality need  最佳画质需要的依赖  
2.1.1 Block-Sparse-Attention wheel [Block-Sparse-Attentiontorch2.8 cu2.8 py311,torch2.12+cu130 py311 py313 wheel ](https://pan.quark.cn/s/c9ba067c89bc) or [CU128 toch2.7](https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper) or [Block-Sparse-Attention wheel](https://huggingface.co/smthem/AI-wheel)  
2.1.2 or install [flashvsr_attn origin](https://github.com/smthemex/Block-Sparse-Attention)  
```
git clone https://github.com/mit-han-lab/Block-Sparse-Attention  # if 5090 or newest cuda
# git clone https://github.com/smthemex/Block-Sparse-Attention # 无须梯子强制编译 可选
# git clone https://github.com/lihaoyun6/Block-Sparse-Attention # 须梯子 可选
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install
```

* 2.2 optional requirements (flashvsr_attn for plus version)  plus版本的可选依赖flashvsr_attn（只用plus且编译麻烦就用这个）  
in node dir 在插件目录下运行   
```
pip install -e operators/flashvsr_attn --no-build-isolation --no-deps

# if python_embeded/ 如果是便携包(也可以复制operators整个文件夹到你的便携python包下，下面的安装命令加上 python -m)

python_embeded/python -m pip install -e you_path/operators/flashvsr_attn --no-build-isolation --no-deps

```

3.checkpoints 
----

* 3.1 [FlashVSRv1.0](https://huggingface.co/JunhaoZhuang/FlashVSR/tree/main) ；[FlashVSRv1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/tree/main) ；[FlashVSRv-plus](https://huggingface.co/ruchangyao/FlashVSR_plus/tree/main) all checkpoints 所有模型，vae 用常规的wan2.1
* 3.2 version 1.0/1.1 emb  [posi_prompt.pth](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)  plus's emb  [posi_prompt.pth](https://github.com/OpenImagingLab/FlashVSR_Plus/blob/main/prompt_tensor/posi_prompt.pth)    

```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt # v1.1 or v1.0
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors #v1.1 or v1.0，rename 最好重命名
|     ├── posi_prompt.pth
|     ├── lq_proj_in_plus_turbo.safetensors   #rename turbo or base
|     ├── base_dit_plus.safetensors #rename base or turbo
├── ComfyUI/models/vae
|        ├──Wan2.1_VAE.pth # or safetensors  # optional/可选
|        ├──TCDecoder_plus.ckpt/TCDecoder.ckpt   # optional/可选 ,注意旧插件TCD不在vae目录下，需要手动剪切过去

```
  
# Example

![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example.png)


# Previous
*  如果觉得项目有用，请给官方项目[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) and [FlashVSR_Plus](https://github.com/OpenImagingLab/FlashVSR_Plus)打星；
* [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) 源码库已更新，支持5090系显卡及更新的设备。注意，需要cuda>=12.8 
*  编译Block-Sparse-Attention  window的轮子 可以使用 [ smthemex 强制编译版](https://github.com/smthemex/Block-Sparse-Attention) 或者 [lihaoyun6 要联网](https://github.com/lihaoyun6/Block-Sparse-Attention) 两个fork来  
* plus 和 long 模式仅支持tcd解码/plus and long mode only support tcd decoder


# Citation
```
@misc{yao2026flashvsrefficientvideosuperresolution,
      title={FlashVSR+: Efficient Video Super-Resolution via Hybrid Attention and Reward-Yield Distribution Matching}, 
      author={Ruchang Yao and Shi Guo and Junhao Zhuang and Mingde Yao and Xin Cai and Jiahao Zhan and Bin Fu and Xinyuan Chen and Yu Qiao and Tianfan Xue},
      year={2026},
      eprint={2610.17552},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.17552}, 
}

@article{zhuang2025flashvsr,
  title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution},
  author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
  journal={arXiv preprint arXiv:2510.12747},
  year={2025}
}


```
