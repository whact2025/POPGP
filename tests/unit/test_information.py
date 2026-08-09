import math

import pytest
import torch

from popgp.information import (
    finite_gibbs_state,
    mix_states,
    modular_energy_delta,
    quantum_relative_entropy,
    von_neumann_entropy,
)


def _diag(*values: float) -> torch.Tensor:
    return torch.diag(torch.tensor(values, dtype=torch.complex128))


def test_relative_entropy_enforces_support_condition() -> None:
    rho = _diag(0.0, 1.0)
    sigma = _diag(1.0, 0.0)

    assert math.isinf(quantum_relative_entropy(rho, sigma))


@pytest.mark.parametrize("leaked_weight", [1e-13, 1e-12, 1e-11])
def test_relative_entropy_rejects_subthreshold_support_leakage(
    leaked_weight: float,
) -> None:
    rho = _diag(1.0 - leaked_weight, leaked_weight)
    sigma = _diag(1.0, 0.0)

    assert math.isinf(quantum_relative_entropy(rho, sigma))


def test_relative_entropy_matches_commuting_closed_form() -> None:
    rho = _diag(0.6, 0.4)
    sigma = _diag(0.5, 0.5)
    expected = 0.6 * math.log(1.2) + 0.4 * math.log(0.8)

    assert quantum_relative_entropy(rho, sigma) == pytest.approx(expected, abs=1e-13)


def test_relative_entropy_clamps_distributed_kernel_roundoff() -> None:
    dimension = 200
    kernel_dimension = 100
    leaked_weight = 9.9e-13
    sigma_values = torch.zeros(dimension, dtype=torch.complex128)
    sigma_values[: dimension - kernel_dimension] = 1.0 / (
        dimension - kernel_dimension
    )
    rho_values = (1.0 - leaked_weight) * sigma_values
    rho_values[dimension - kernel_dimension :] = leaked_weight / kernel_dimension

    result = quantum_relative_entropy(
        torch.diag(rho_values), torch.diag(sigma_values)
    )

    assert result == 0.0


def test_entropy_treats_zero_eigenvalues_exactly() -> None:
    assert von_neumann_entropy(_diag(1.0, 0.0)) == pytest.approx(0.0)


def test_information_primitives_reject_non_normalized_input() -> None:
    invalid = torch.eye(2, dtype=torch.complex128)

    with pytest.raises(ValueError, match="unit trace"):
        von_neumann_entropy(invalid)


def test_modular_energy_affine_mixture_linearity_identity() -> None:
    sigma = _diag(0.7, 0.3)
    excitation = _diag(0.2, 0.8)
    first = modular_energy_delta(mix_states(sigma, excitation, 1e-3), sigma)
    second = modular_energy_delta(mix_states(sigma, excitation, 2e-3), sigma)

    assert second == pytest.approx(2.0 * first, rel=1e-10, abs=1e-14)


def test_finite_gibbs_state_matches_two_level_closed_form() -> None:
    hamiltonian = _diag(0.0, 2.0)
    state = finite_gibbs_state(hamiltonian, beta=0.7)
    excited_weight = math.exp(-1.4) / (1.0 + math.exp(-1.4))

    assert state[1, 1].real.item() == pytest.approx(excited_weight, abs=1e-14)
    assert torch.trace(state).real.item() == pytest.approx(1.0, abs=1e-14)


def test_finite_gibbs_state_rejects_nonhermitian_hamiltonian() -> None:
    hamiltonian = torch.tensor(
        [[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128
    )

    with pytest.raises(ValueError, match="Hermitian"):
        finite_gibbs_state(hamiltonian, beta=1.0)
