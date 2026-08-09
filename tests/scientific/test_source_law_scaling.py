import numpy as np
import pytest
import torch

from popgp import Simulator
from popgp.diagnostics import fit_power_law
from popgp.information import mix_states, modular_energy_delta, quantum_relative_entropy

pytestmark = pytest.mark.scientific


def _diag(*values: float) -> torch.Tensor:
    return torch.diag(torch.tensor(values, dtype=torch.complex128))


def _log_slope(epsilons: np.ndarray, values: np.ndarray) -> float:
    return fit_power_law(epsilons, np.abs(values)).slope


def test_relative_entropy_is_quadratic_near_faithful_reference() -> None:
    reference = _diag(0.7, 0.3)
    excitation = _diag(0.2, 0.8)
    epsilons = np.logspace(-5, -2, 10)
    values = np.array(
        [quantum_relative_entropy(mix_states(reference, excitation, eps), reference)
         for eps in epsilons]
    )

    assert _log_slope(epsilons, values) == pytest.approx(2.0, abs=0.02)


def test_modular_energy_obeys_affine_mixture_linearity_identity() -> None:
    reference = _diag(0.7, 0.3)
    excitation = _diag(0.2, 0.8)
    epsilons = np.logspace(-6, -2, 10)
    values = np.array(
        [modular_energy_delta(mix_states(reference, excitation, eps), reference)
         for eps in epsilons]
    )
    coefficient = modular_energy_delta(excitation, reference)

    assert values == pytest.approx(epsilons * coefficient, abs=1e-14)


def test_clock_potential_obeys_linear_solver_homogeneity_identity() -> None:
    reference = _diag(0.7, 0.3)
    excitation = _diag(0.2, 0.8)
    epsilons = np.logspace(-5, -2, 10)
    weights = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    relative_entropy_amplitudes = []
    modular_energy_amplitudes = []

    for epsilon in epsilons:
        state = mix_states(reference, excitation, epsilon)
        candidates = (
            (relative_entropy_amplitudes, quantum_relative_entropy(state, reference)),
            (modular_energy_amplitudes, abs(modular_energy_delta(state, reference))),
        )
        for amplitudes, source_strength in candidates:
            source = torch.tensor([0.0, -source_strength, 0.0], dtype=torch.float64)
            phi, _, _, _ = Simulator._solve_clock_constraint(
                weights,
                source,
                mu=0.1,
                zero_mode_policy="subtract_mean",
                normalize_potential=True,
            )
            amplitudes.append(float(phi.max() - phi.min()))

    relative_fit = fit_power_law(epsilons, np.array(relative_entropy_amplitudes))
    relative_ratios = np.array(relative_entropy_amplitudes) / np.array(
        [
            quantum_relative_entropy(
                mix_states(reference, excitation, epsilon), reference
            )
            for epsilon in epsilons
        ]
    )
    modular_ratios = np.array(modular_energy_amplitudes) / np.array(
        [
            abs(
                modular_energy_delta(
                    mix_states(reference, excitation, epsilon), reference
                )
            )
            for epsilon in epsilons
        ]
    )
    assert relative_fit.slope == pytest.approx(2.0, abs=0.02)
    assert relative_fit.slope_standard_error < 0.01
    assert np.ptp(relative_ratios) < 1e-10
    assert np.ptp(modular_ratios) < 1e-10
