# -*- coding: utf-8 -*-
"""FlashVSR+ (Plus) 推理包。

代码源自 OpenImagingLab/FlashVSR_Plus（inference / models / model_io 三个包），
为避免与其他 ComfyUI 插件的顶层包名冲突，统一收拢到 flashvsr_plus 命名空间下
（源码中的绝对导入已相应改写）。权重与提示张量随包内 prompt_tensor 提供。

公开 API 与上游一致：
    from flashvsr_plus import InferenceRequest, build_inference_pipeline, stream_config_for_mode
"""

from .inference import (
    ColorCorrectionConfig,
    DecoderConfig,
    FlashVSRInferencePipeline,
    InferenceConfig,
    InferenceRequest,
    InferenceResult,
    ModelBundle,
    ModelPaths,
    StreamConfig,
    build_inference_pipeline,
    stream_config_for_mode,
)

__all__ = [
    "ColorCorrectionConfig",
    "DecoderConfig",
    "FlashVSRInferencePipeline",
    "InferenceConfig",
    "InferenceRequest",
    "InferenceResult",
    "ModelBundle",
    "ModelPaths",
    "StreamConfig",
    "build_inference_pipeline",
    "stream_config_for_mode",
]
