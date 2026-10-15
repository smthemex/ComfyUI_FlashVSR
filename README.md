# ComfyUI_FlashVSR
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR): Towards Real-Time Diffusion-Based Streaming Video Super-Resolution,this node ,you can use it in comfyUI 
[FlashVSR_Plus](https://github.com/OpenImagingLab/FlashVSR_Plus),FlashVSR+ is an efficient one-step diffusion framework for high-quality video super-resolution, enabling spatially untiled 4K inference on a single 24GB RTX 4090 GPU. 

#  Update
* 支持flashVSR-Plus 模型，plus模式 请在菜单手动选triton-flashvsr_attn，否则还会是 自动选择Block-Sparse-Attention  ，support flashVSR-Plus model，plus use triton attention，or auto will choice Block-Sparse-Attention
* | `plus_temporal` | 4 | 每 latent 帧的输入帧数；大运动视频建议 2 或 1（更慢） |
* ~~简化代码， 分解推理和解码节点（tiny和full模式只是解码不同，最快是lightx2v的vae，其次是tcd，也可以使用原生comfyUI vae节点），修复视频分割bat文件的bug，新增单块卸载（offload按钮）以降低显存占用~~
* ~~Simplify the code, decompose inference and decoding nodes (the only difference between tiny and full modes is in decoding; the fastest is lightx2v's VAE, then tcd, and the native comfyUI VAE node can also be used), fix the bug in the video segmentation bat file, and add a single block offload button to reduce VRAM usage;~~

# Previous
*  如果觉得项目有用，请给官方项目[FlashVSR](https://github.com/OpenImagingLab/FlashVSR) and [FlashVSR_Plus](https://github.com/OpenImagingLab/FlashVSR_Plus)打星；
* [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) 源码库已更新，支持5090系显卡及更新的设备。注意，需要cuda>=12.8 
* ~~支持任意帧数输入不被裁切~~
* ~~新增lightx2v 加速vae decoder支持和Wan2.1-VAE-upscale2x 放大decoder支持，light的加速模型目前只支持（lightvaew2_1.pth  #32.2M,taew2_1.pth,lighttaew2_1.pth） 三个文件~~
*  ~~满足部分网友需要超分单张图片的奇怪要求,默认输出25帧1秒的视频，详见示例~~，
*  ~~新增切片视频路径加载节点，输入保存切片视频的路径，开启自动推理，即可推理完路径所有视频，切片的一键bat在插件目录下，window环境使用，点击即可~~；
*  ~~local_range=7这个是会最清晰，local_range=11会比较稳定，color fix 推荐用小波（没重影）；~~ 
*  编译Block-Sparse-Attention  window的轮子 可以使用 [ smthemex 强制编译版](https://github.com/smthemex/Block-Sparse-Attention) 或者 [lihaoyun6 要联网](https://github.com/lihaoyun6/Block-Sparse-Attention) 两个fork来，不推荐用官方的  
*  Block-Sparse-Attention 正确安装且能调用才是方法的完全体，当前的函数实现会更容易OOM,但是Block-Sparse-Attention轮子实在不好找，目前只有[CU128 toch2.7](https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper)的，我提供的（[cu128，torch2.8，py311单体,torch2.12+cu130 py311 py313](https://pan.quark.cn/s/c9ba067c89bc)）或者自己编译  
* tile关闭质量更高，需要VRam更高，corlor fix对于非模糊图片可以试试。
*  Choice vae infer full mode ，encoder infer tiny mode 选择vae跑full模式 效果最好，tiny则是速度，数据集基于4倍训练，所以1 scale是不推荐的； 

*  if you Like it ， star the official project： [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)  
* Good news, the [mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) library has been updated and now supports 5090 and updated graphics cards ,need cuda>=12.8; 
* Support any frame rate input without cropping
* Added support for lightx2v accelerated VAE decoder and Wan2.1-VAE-upscale2x upscaling decoder. The light acceleration model currently only supports three files: (lightvaew2_1.pth #32.2M, taew2_1.pth, lighttaew2_1.pth)
* Satisfying the strange requirement of some netizens to VSR a ‘single image’, the default output is 25 frames per second of video, you can find it in the example.
* Add a sliced video path loading node, enter the path to save the sliced video, enable automatic inference, and all videos in the path can be inferred. The one click ‘bat、 for slicing can be used in the node directory and window environment. Click to complete the inference;  
 * Local_range=7 will be the clearest, while local_range=11 will be more stable. For color fix, it is recommended to use wavelets (without ghosting); 
* The wheel for compiling the Block Spark Attention window can be used[ smthemex ](https://github.com/smthemex/Block-Sparse-Attention) 或者 [lihaoyun6](https://github.com/lihaoyun6/Block-Sparse-Attention)Two forks, not recommended to use official ones
* The correct installation and ability to call Block Spark Attention are the complete set of methods. The current function implementation will be easier to OOM, but the Block Spark Attention wheel is really hard to find. Currently, only [CU128 toch2.7]（ https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper ）Yes, I provided [cu128，torch2.8，py311,torch2.12+cu130 py311 py313](https://pan.quark.cn/s/c9ba067c89bc)Or compile it yourself
* Tile shutdown has higher quality and requires higher VRam. For non blurry images, corlor fix can be tried.    
* Choose VAE in full mode, encoder in tiny mode. Choosing VAE to run in full mode yields the best results, while tiny mode is for speed. The dataset is based on 4x training, so 1 scale is not recommended;   


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

* 1.optional requirements ( for flashvsr_attn)
if use triton flashvsr_attn ,in the node dir 
嫌弃编译麻烦的直接用这个，在插件目录下运行（如果是便携包，则复制operators整个文件夹到你的便携python包下，下面的安装命令加上 python -m） 
```
pip install -e operators/flashvsr_attn --no-build-isolation --no-deps
```
* 2.optional requirements ( for Block-Sparse-Attention)
if use Block-Sparse-Attention
要复现官方效果，可以安装Block-Sparse-Attention [torch2.8 cu2.8 py311,torch2.12+cu130 py311 py313 wheel ](https://pan.quark.cn/s/c9ba067c89bc) or [CU128 toch2.7](https://github.com/lihaoyun6/ComfyUI-WanVideoWrapper)  origin method need this 
```
git clone https://github.com/mit-han-lab/Block-Sparse-Attention  # if 5090 or newest cuda
# git clone https://github.com/smthemex/Block-Sparse-Attention # 无须梯子强制编译
# git clone https://github.com/lihaoyun6/Block-Sparse-Attention # 须梯子
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install
```

3.checkpoints 
----

* 3.1.2 [FlashVSRv1.0](https://huggingface.co/JunhaoZhuang/FlashVSR/tree/main)   all checkpoints 所有模型，vae 用常规的wan2.1
* 3.1.2 [FlashVSRv1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/tree/main) all checkpoints 所有模型，vae 用常规的wan2.1
* 3.1.3 [FlashVSRv-plus](https://huggingface.co/ruchangyao/FlashVSR_plus/tree/main) all checkpoints 所有模型，vae 用常规的wan2.1
* 3.2.1 1.0/1.1 emb  [posi_prompt.pth](https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/prompt_tensor)  4M而已
* 3.2.2 plus emb  [posi_prompt.pth](https://github.com/OpenImagingLab/FlashVSR_Plus/blob/main/prompt_tensor/posi_prompt.pth)  4M而已
* 3.3 [lightvaew2_1.pth](https://huggingface.co/lightx2v/Autoencoders/tree/main) and [diffusion_pytorch_model.safetensors](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/tree/main/diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1)
  
```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt # v1.1 or v1.0
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors #v1.1 or v1.0
|     ├── posi_prompt.pth
|     ├──lq_proj_in_plus_turbo.safetensors   #rename turbo or base
|     ├── base_dit_plus.safetensors #rename base or turbo
├── ComfyUI/models/vae
|        ├──Wan2.1_VAE.pth # or safetensors  optional/可选
|        ├──lightvaew2_1.pth  #32.2M  or taew2_1.pth,lighttaew2_1.pth optional/可选
|        ├──Wan2.1_VAE_upscale2x_imageonly_real_v1_diff.safetensors  # rename from diffusion_pytorch_model.safetensors optional/可选
```
  

# Example

![](https://github.com/smthemex/ComfyUI_FlashVSR/blob/main/example_workflows/example.png)


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
lightx2v
```
@misc{lightx2v,
 author = {LightX2V Contributors},
 title = {LightX2V: Light Video Generation Inference Framework},
 year = {2025},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/ModelTC/lightx2v}},
}
```
