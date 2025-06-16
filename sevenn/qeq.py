import torch


def charge_equilibrium(
    chi_eff: torch.Tensor,
    j_ii: torch.Tensor,
    total_charge: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Return atomic charges using the QEq analytic solution.

    Parameters
    ----------
    chi_eff : torch.Tensor
        Effective electronegativity for each atom with shape ``(n_atoms,)``.
    j_ii : torch.Tensor
        Self interaction term ``J_ii`` for each atom with shape ``(n_atoms,)``.
    total_charge : float | torch.Tensor, optional
        Net charge of the system. Defaults to ``0.0``.

    Returns
    -------
    torch.Tensor
        Equilibrated atomic charges of shape ``(n_atoms,)``.
    """
    if chi_eff.ndim != 1 or j_ii.ndim != 1:
        raise ValueError('chi_eff and j_ii must be 1D tensors')
    if chi_eff.shape != j_ii.shape:
        raise ValueError('chi_eff and j_ii must have the same shape')

    total_charge = torch.as_tensor(
        total_charge, dtype=chi_eff.dtype, device=chi_eff.device
    )

    inv_j = 1.0 / j_ii
    lamb = (total_charge + (chi_eff * inv_j).sum()) / inv_j.sum()
    q = (lamb - chi_eff) * inv_j
    return q
