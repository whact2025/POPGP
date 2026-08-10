import pytest
import torch

from popgp import Simulator
from popgp.information import quantum_relative_entropy, von_neumann_entropy

pytestmark = pytest.mark.scientific


def _diag(*values: float) -> torch.Tensor:
    return torch.diag(torch.tensor(values, dtype=torch.complex128))


@pytest.mark.negative_control
def test_equal_energy_states_expose_entropy_source_confound() -> None:
    hamiltonian = torch.diag(torch.tensor([0.0, 1.0, 2.0], dtype=torch.complex128))
    pure_middle = _diag(0.0, 1.0, 0.0)
    mixed_extremes = _diag(0.5, 0.0, 0.5)
    thermal_weights = torch.softmax(
        -torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64), dim=0
    )
    reference = torch.diag(thermal_weights.to(torch.complex128))

    energy_pure = torch.trace(pure_middle @ hamiltonian).real.item()
    energy_mixed = torch.trace(mixed_extremes @ hamiltonian).real.item()
    source_pure = quantum_relative_entropy(pure_middle, reference)
    source_mixed = quantum_relative_entropy(mixed_extremes, reference)

    assert energy_pure == pytest.approx(energy_mixed)
    assert von_neumann_entropy(pure_middle) != pytest.approx(
        von_neumann_entropy(mixed_extremes)
    )
    assert source_pure != pytest.approx(source_mixed)


def test_negative_localized_source_produces_slower_source_clock() -> None:
    weights = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    source = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)

    phi, _, _, residual = Simulator._solve_clock_constraint(
        weights,
        source,
        mu=0.1,
        zero_mode_policy="subtract_mean",
        normalize_potential=True,
    )
    z = Simulator.gravitational_redshift(phi_emitter=phi[1], phi_observer=phi[0])

    assert phi[1] < phi[0]
    assert torch.exp(phi[1]) < torch.exp(phi[0])
    assert z > 0.0
    assert residual < 1e-12
