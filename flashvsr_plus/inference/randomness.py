from __future__ import annotations

import random

import numpy as np
import torch


DEFAULT_INFERENCE_SEED = 2026


def set_inference_seed(seed: int = DEFAULT_INFERENCE_SEED) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
