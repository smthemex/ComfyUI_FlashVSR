from .config import FlashVSRModelConfig
from .dit import FlashVSRDiT


def build_model(config: FlashVSRModelConfig) -> FlashVSRDiT:
    config.validate()
    return FlashVSRDiT(config)
