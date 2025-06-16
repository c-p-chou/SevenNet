import torch
import torch.nn as nn

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType


class GlobalScalarEmbedding(nn.Module):
    """Append a graph-level scalar to each node feature."""

    def __init__(
        self,
        data_key_scalar: str,
        data_key_x: str = KEY.NODE_FEATURE,
    ) -> None:
        super().__init__()
        self.key_scalar = data_key_scalar
        self.key_x = data_key_x

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        scalar = data[self.key_scalar]
        if scalar.ndim == 0:
            scalar = scalar.unsqueeze(0)
        num_nodes = data[self.key_x].shape[0]
        expanded = scalar.view(1, -1).expand(num_nodes, -1)
        data[self.key_x] = torch.cat([data[self.key_x], expanded], dim=-1)
        return data
