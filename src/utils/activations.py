from typing import Callable

import torch
from torch import Tensor


def get_activation(name: str) -> Callable[[Tensor], Tensor]:
    alias = name.lower().strip()
    if alias == "tanh":
        return torch.tanh
    if alias == "relu":
        return torch.relu
    if alias == "sigmoid":
        return torch.sigmoid
    if alias in {"identity", "linear", "none", "id"}:
        return lambda x: x
    raise ValueError(f"Unsupported activation: {name}")


