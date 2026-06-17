from __future__ import annotations

import torch
from torch import nn


class SimNorm(nn.Module):
    """Simplicial normalization (TD-MPC2). Reshapes the last dim into groups of size V and
    applies a softmax within each group, yielding a concatenation of probability simplices.

    The output is non-negative and each group of V entries sums to 1, which bounds the latent
    and is a structural collapse guard for the JEPA target.
    """

    def __init__(self, group_size: int = 8) -> None:
        super().__init__()
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.group_size = int(group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.group_size != 0:
            raise ValueError(
                f"last dim {x.shape[-1]} not divisible by group_size {self.group_size}"
            )
        shape = x.shape
        x = x.reshape(*shape[:-1], shape[-1] // self.group_size, self.group_size)
        x = torch.softmax(x, dim=-1)
        return x.reshape(*shape)

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}"
