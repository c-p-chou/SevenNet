import torch
import torch.nn as nn

import sevenn._keys as KEY
from sevenn._const import AtomGraphDataType
from sevenn.qeq import charge_equilibrium


class ChargeEquilibrium(nn.Module):
    """Compute partial charges using the analytic QEq update."""

    def __init__(
        self,
        total_charge: float | torch.Tensor = 0.0,
        data_key_chi: str = KEY.EFFECTIVE_ELECTRONEGATIVITY,
        data_key_jii: str = KEY.SELF_INTERACTION,
        data_key_charge: str = KEY.PARTIAL_CHARGE,
    ) -> None:
        super().__init__()
        self.total_charge = total_charge
        self.key_chi = data_key_chi
        self.key_jii = data_key_jii
        self.key_charge = data_key_charge

    def forward(self, data: AtomGraphDataType) -> AtomGraphDataType:
        q = charge_equilibrium(
            data[self.key_chi].squeeze(-1),
            data[self.key_jii].squeeze(-1),
            self.total_charge,
        )
        data[self.key_charge] = q.unsqueeze(-1)
        return data
