from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .FlashVSR_node import FlashVSR_SM_Model, FlashVSR_SM_KSampler, FlashVSR_SM_Vae, FlashVSR_SM_Decoder, FlashVSR_SM_VideoTrimLoop
class FlashVSR_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            FlashVSR_SM_Model,
            FlashVSR_SM_KSampler,
            FlashVSR_SM_Vae,
            FlashVSR_SM_Decoder,
            FlashVSR_SM_VideoTrimLoop,
        ]
async def comfy_entrypoint() -> FlashVSR_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return FlashVSR_SM_Extension()
