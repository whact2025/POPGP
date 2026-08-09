"""Finite-dimensional information-theory primitives used by POPGP.

These routines implement the finite-dimensional (Umegaki) representative of
Araki relative entropy.  In particular, they preserve the exact support
condition: D(rho || sigma) is infinite when rho has weight outside the support
of sigma.  Replacing zero eigenvalues with an arbitrary numerical floor would
turn a mathematical divergence into a parameter-dependent finite value.
"""

from __future__ import annotations

import math

import torch


def _as_complex128(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.to(dtype=torch.complex128)


def _eigh_density(
    matrix: torch.Tensor,
    *,
    name: str,
    validation_atol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = _as_complex128(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square density matrix")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    if torch.linalg.matrix_norm(matrix - hermitian).item() > validation_atol:
        raise ValueError(f"{name} must be Hermitian")
    trace = torch.trace(hermitian)
    if (
        abs(trace.imag.item()) > validation_atol
        or abs(trace.real.item() - 1.0) > validation_atol
    ):
        raise ValueError(f"{name} must have unit trace")
    evals, evecs = torch.linalg.eigh(hermitian)
    if evals.min().item() < -validation_atol:
        raise ValueError(f"{name} must be positive semidefinite")
    return evals, evecs


def von_neumann_entropy(rho: torch.Tensor, *, atol: float = 1e-15) -> float:
    """Return ``-Tr(rho log rho)`` while treating zero eigenvalues exactly."""
    evals, _ = _eigh_density(rho, name="rho")
    positive = evals.real[evals.real > atol]
    if positive.numel() == 0:
        return 0.0
    return float(-torch.sum(positive * torch.log(positive)).item())


def quantum_relative_entropy(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    *,
    support_atol: float = 1e-12,
    eigenvalue_atol: float = 1e-15,
) -> float:
    """Return finite-dimensional ``D(rho || sigma)`` with support checks.

    The result is ``math.inf`` unless ``supp(rho)`` is contained in
    ``supp(sigma)``.  Both inputs are assumed to be normalized density
    matrices; small Hermiticity noise is removed before diagonalization.
    """
    rho = _as_complex128(rho)
    sigma = _as_complex128(sigma)
    if rho.shape != sigma.shape:
        raise ValueError("rho and sigma must be square density matrices of equal shape")

    validation_atol = max(support_atol, 1e-10)
    evals_rho, _ = _eigh_density(
        rho, name="rho", validation_atol=validation_atol
    )
    evals_sigma, evecs_sigma = _eigh_density(
        sigma, name="sigma", validation_atol=validation_atol
    )
    evals_rho = evals_rho.real
    evals_sigma = evals_sigma.real

    numerical_leakage_atol = min(
        support_atol,
        32.0 * torch.finfo(torch.float64).eps * rho.shape[0],
    )
    sigma_kernel = evals_sigma <= eigenvalue_atol
    if sigma_kernel.any():
        kernel_vectors = evecs_sigma[:, sigma_kernel]
        leaked_weight = torch.trace(
            kernel_vectors.conj().T @ rho @ kernel_vectors
        ).real.item()
        # Support containment is exact.  The only ignored weight is a guard for
        # floating-point projection noise, deliberately much smaller than the
        # public matrix-validation tolerance.
        if leaked_weight > numerical_leakage_atol:
            return math.inf

    positive_rho = evals_rho > eigenvalue_atol
    rho_log_rho = torch.sum(
        evals_rho[positive_rho] * torch.log(evals_rho[positive_rho])
    ).item()

    log_sigma_evals = torch.zeros_like(evals_sigma)
    positive_sigma = evals_sigma > eigenvalue_atol
    log_sigma_evals[positive_sigma] = torch.log(evals_sigma[positive_sigma])
    log_sigma = (
        evecs_sigma
        @ torch.diag(log_sigma_evals.to(torch.complex128))
        @ evecs_sigma.conj().T
    )
    rho_log_sigma = torch.trace(rho @ log_sigma).real.item()

    result = float(rho_log_rho - rho_log_sigma)
    nonnegative_atol = max(
        support_atol,
        numerical_leakage_atol
        * abs(math.log(max(numerical_leakage_atol, torch.finfo(torch.float64).tiny))),
    )
    if -nonnegative_atol < result < 0.0:
        return 0.0
    if result < 0.0:
        raise ArithmeticError(
            "relative entropy became negative beyond the numerical tolerance"
        )
    return result


def modular_hamiltonian(sigma: torch.Tensor, *, eigenvalue_atol: float = 1e-15) -> torch.Tensor:
    """Return ``K_sigma = -log(sigma)`` for a faithful reference state."""
    evals, evecs = _eigh_density(sigma, name="sigma")
    evals = evals.real
    if (evals <= eigenvalue_atol).any():
        raise ValueError("the modular Hamiltonian requires a faithful reference state")
    return evecs @ torch.diag((-torch.log(evals)).to(torch.complex128)) @ evecs.conj().T


def modular_energy_delta(rho: torch.Tensor, sigma: torch.Tensor) -> float:
    """Return ``Tr[(rho - sigma) K_sigma]`` for a faithful reference state."""
    k_sigma = modular_hamiltonian(sigma)
    return float(torch.trace((_as_complex128(rho) - _as_complex128(sigma)) @ k_sigma).real.item())


def mix_states(sigma: torch.Tensor, excitation: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Construct ``(1-epsilon) sigma + epsilon excitation``."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    return (1.0 - epsilon) * _as_complex128(sigma) + epsilon * _as_complex128(excitation)
