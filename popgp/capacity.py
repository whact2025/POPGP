"""
Cut-capacity functional for the finite distinguishability constraint (§4.4.2).

Implements:
- cut_capacity: Cap(∂R) = Σ_{i∈R, j∉R} κ(I_{ij}) for a region R
- check_capacity_bound: S_Araki(ω|_R ‖ ω^vac|_R) ≤ η · Cap(∂R)

The capacity bound is the pre-geometric version of the area law.  It is
imposed *before* any geometric embedding (§4.4.2), using the algebraic
cut-capacity on the cell net rather than a geometric boundary area.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch

from popgp.backend import Backend

log = logging.getLogger(__name__)


def cut_capacity(
    mi_matrix: torch.Tensor,
    region: list[int],
    kappa: Callable[[float], float],
) -> float:
    """Compute Cap(∂R) = Σ_{i∈R, j∉R} κ(I_{ij}) for a single region R (§4.4.2).

    Parameters
    ----------
    mi_matrix : Tensor [n_cells × n_cells]
        Pairwise mutual information between cells.
    region : list[int]
        Cell indices comprising region R.
    kappa : callable
        Monotone increasing weight function with κ(0)=0.

    Returns
    -------
    float
        Cut-capacity of the boundary ∂R.
    """
    n = mi_matrix.shape[0]
    region_set = set(region)
    complement = [j for j in range(n) if j not in region_set]

    cap = 0.0
    for i in region:
        for j in complement:
            cap += kappa(mi_matrix[i, j].item())
    return cap


def cut_capacity_all_regions(
    mi_matrix: torch.Tensor,
    kappa: Callable[[float], float],
    max_region_size: int | None = None,
) -> dict[frozenset[int], float]:
    """Compute Cap(∂R) for all non-trivial subsets R of the cell net.

    For small cell counts, this enumerates all 2^n − 2 non-trivial
    subsets.  For larger systems, ``max_region_size`` limits the search.

    Returns
    -------
    dict mapping frozenset of cell indices → capacity value.
    """
    from itertools import combinations as _combs

    n = mi_matrix.shape[0]
    if max_region_size is None:
        max_region_size = n - 1

    result: dict[frozenset[int], float] = {}
    for size in range(1, min(max_region_size, n - 1) + 1):
        for region_tuple in _combs(range(n), size):
            region = list(region_tuple)
            cap = cut_capacity(mi_matrix, region, kappa)
            result[frozenset(region)] = cap
    return result


def check_capacity_bound(
    state: torch.Tensor,
    cells: list[list[int]],
    region_cell_indices: list[int],
    rho_vac: torch.Tensor,
    backend: Backend,
    eta: float,
    mi_matrix: torch.Tensor,
    kappa: Callable[[float], float],
) -> tuple[bool, float, float]:
    """Check S_Araki(ω|_R ‖ ω^vac|_R) ≤ η · Cap(∂R) for a region (§4.4.2).

    Parameters
    ----------
    state : Tensor
        Full system density matrix.
    cells : list[list[int]]
        Cell decomposition (qubit indices per cell).
    region_cell_indices : list[int]
        Which cells form region R.
    rho_vac : Tensor
        Full-system vacuum/reference state.
    backend : Backend
        For computing reduced states and Araki relative entropy.
    eta : float
        Proportionality constant in the capacity bound.
    mi_matrix : Tensor
        Pairwise MI matrix (from Π_loc or pre-computed).
    kappa : callable
        Weight function for cut-capacity.

    Returns
    -------
    (passes, s_araki, cap) : (bool, float, float)
        Whether the bound holds, the Araki relative entropy of the region,
        and the cut-capacity.
    """
    region_qubits: list[int] = []
    for ci in region_cell_indices:
        region_qubits.extend(cells[ci])

    rho_R = backend.reduced_state(state, region_qubits)
    rho_vac_R = backend.reduced_state(rho_vac, region_qubits)

    s_araki = backend.araki_relative_entropy(rho_R, rho_vac_R)

    cap = cut_capacity(mi_matrix, region_cell_indices, kappa)
    bound = eta * cap

    passes = s_araki <= bound
    log.debug(
        "Capacity bound for R=%s: S_Araki=%.4f, η·Cap=%.4f → %s",
        region_cell_indices, s_araki, bound, "PASS" if passes else "FAIL",
    )
    return passes, s_araki, bound
