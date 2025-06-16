import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


@compile_mode('script')
class ChargeEquilibrium(nn.Module):
    """Placeholder charge equilibration module."""

    def __init__(
        self,
        data_key_x: str = KEY.NODE_FEATURE,
        data_key_out: str = KEY.PARTIAL_CHARGE,
    ) -> None:
        super().__init__()
        self.key_x = data_key_x
        self.key_out = data_key_out

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        x = data[self.key_x]
        data[self.key_out] = torch.zeros(
            x.shape[0], 1, dtype=x.dtype, device=x.device
        )
        return data


@compile_mode('script')
class GlobalScalarEmbedding(nn.Module):
    """Embed a scalar feature and concatenate to node features."""

    def __init__(
        self, data_key_scalar: str, data_key_x: str = KEY.NODE_FEATURE
    ) -> None:
        super().__init__()
        self.key_scalar = data_key_scalar
        self.key_x = data_key_x
        self.linear = nn.Linear(1, 1)

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        scalar = data[self.key_scalar]
        if scalar.dim() == 1:
            scalar = scalar.unsqueeze(-1)
        emb = self.linear(scalar)
        data[self.key_x] = torch.cat([data[self.key_x], emb], dim=1)
        return data
