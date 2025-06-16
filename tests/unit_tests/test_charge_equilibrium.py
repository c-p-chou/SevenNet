import torch

from sevenn.qeq import charge_equilibrium


def test_charge_equilibrium_conserves_charge():
    chi = torch.tensor([1.0, 2.0, 3.0])
    jii = torch.tensor([0.5, 0.5, 0.5])
    q_tot = 1.0
    charges = charge_equilibrium(chi, jii, q_tot)
    assert torch.allclose(charges.sum(), torch.tensor(q_tot))


def test_charge_equilibrium_matches_formula():
    chi = torch.tensor([0.2, -0.3])
    jii = torch.tensor([0.4, 0.6])
    q_tot = 0.0
    charges = charge_equilibrium(chi, jii, q_tot)
    inv_j = 1 / jii
    lamb = (q_tot + (chi * inv_j).sum()) / inv_j.sum()
    expected = (lamb - chi) * inv_j
    assert torch.allclose(charges, expected)
