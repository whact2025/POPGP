import numpy as np
import pytest
import torch

from popgp import Simulator, SimulatorConfig
from popgp.backend import ExactBackend
from popgp.diagnostics import (
    assess_quadratic_response,
    richardson_first_order_limit,
)
from popgp.information import (
    finite_gibbs_state,
    kubo_mori_covariance,
    mix_states,
    modular_energy_delta,
    modular_hamiltonian,
    quantum_relative_entropy,
    von_neumann_entropy,
)

pytestmark = pytest.mark.scientific


def _experiment() -> tuple[
    ExactBackend,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
]:
    config = SimulatorConfig.for_chain(
        n=5,
        beta=1.3,
        boundary="open",
        hamiltonian="heisenberg",
    )
    backend = ExactBackend(config)
    hamiltonian = backend.build_hamiltonian()
    reference = backend.prepare_state()
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    local_unitary = backend.site_operator(pauli_x, site=2)
    excitation = local_unitary @ reference @ local_unitary.conj().T
    return (
        backend,
        hamiltonian,
        reference,
        excitation,
        backend.build_local_energy_operators(),
    )


def _expectation_delta(
    state: torch.Tensor,
    reference: torch.Tensor,
    observable: torch.Tensor,
) -> float:
    return float(torch.trace((state - reference) @ observable).real.item())


def _precision_floor(
    reference: torch.Tensor,
    observable: torch.Tensor,
) -> float:
    baseline = abs(float(torch.trace(reference @ observable).real.item()))
    return float(np.finfo(float).eps * max(1.0, baseline))


@pytest.mark.parametrize(
    ("n_sites", "family", "beta"),
    [
        (5, "heisenberg", 0.3),
        (5, "heisenberg", 1.0),
        (5, "heisenberg", 2.0),
        (5, "heisenberg", 2.5),
        (5, "heisenberg", 3.0),
        (5, "ising", 0.3),
        (5, "ising", 1.0),
        (5, "ising", 2.0),
        (5, "ising", 2.5),
        (5, "ising", 3.0),
        (3, "heisenberg", 1.0),
        (7, "heisenberg", 1.0),
    ],
)
def test_nonaffine_kms_response_converges_across_declared_parameter_sweep(
    n_sites: int,
    family: str,
    beta: float,
) -> None:
    config = SimulatorConfig.for_chain(
        n=n_sites,
        beta=beta,
        boundary="open",
        hamiltonian=family,
    )
    backend = ExactBackend(config)
    hamiltonian = backend.build_hamiltonian()
    reference = backend.prepare_state()
    perturbation = -backend.build_local_energy_operators()[n_sites // 2]
    epsilons = np.logspace(-4.5, -3, 7)
    relative_entropy = []
    modular_energy = []

    for epsilon in epsilons:
        state = finite_gibbs_state(
            hamiltonian + float(epsilon) * perturbation,
            beta,
        )
        relative_entropy.append(quantum_relative_entropy(state, reference))
        modular_energy.append(modular_energy_delta(state, reference))

    modular_operator = modular_hamiltonian(reference)
    precision_floor = _precision_floor(reference, modular_operator)
    quadratic = assess_quadratic_response(
        epsilons,
        np.asarray(relative_entropy),
        absolute_precision_floor=precision_floor,
        lower_window_size=5,
    )
    susceptibility = richardson_first_order_limit(
        epsilons,
        np.asarray(modular_energy),
        absolute_precision_floor=precision_floor,
    )
    exact_coefficient = (
        0.5 * beta**2 * kubo_mori_covariance(reference, perturbation)
    )
    exact_susceptibility = -beta * kubo_mori_covariance(
        reference, modular_operator, perturbation
    )

    assert quadratic.passed is True
    assert susceptibility.passed is True
    assert quadratic.full_window.minimum_signal_to_floor >= 1000.0
    assert quadratic.full_window.coefficient == pytest.approx(
        exact_coefficient, rel=5e-4
    )
    assert susceptibility.estimate == pytest.approx(
        exact_susceptibility, rel=1e-6
    )


def test_nonaffine_kms_family_separates_linear_and_quadratic_orders() -> None:
    backend, hamiltonian, reference, _, local_energy = _experiment()
    beta = backend.config.substrate.beta
    perturbation = -local_energy[2]
    epsilons = np.logspace(-5, -3, 9)
    relative_entropy = []
    modular_energy = []
    total_energy = []

    for epsilon in epsilons:
        state = finite_gibbs_state(
            hamiltonian + float(epsilon) * perturbation,
            beta,
        )
        relative_entropy.append(quantum_relative_entropy(state, reference))
        modular_energy.append(modular_energy_delta(state, reference))
        total_energy.append(_expectation_delta(state, reference, hamiltonian))

        site_energy = np.array(
            [_expectation_delta(state, reference, term) for term in local_energy]
        )
        entropy_delta = von_neumann_entropy(state) - von_neumann_entropy(reference)
        assert site_energy.sum() == pytest.approx(total_energy[-1], abs=1e-14)
        assert modular_energy[-1] == pytest.approx(
            backend.config.substrate.beta * total_energy[-1], abs=2e-14
        )
        assert relative_entropy[-1] == pytest.approx(
            modular_energy[-1] - entropy_delta, abs=2e-14
        )

    modular_operator = modular_hamiltonian(reference)
    precision_floor = _precision_floor(reference, modular_operator)
    relative_assessment = assess_quadratic_response(
        epsilons,
        np.asarray(relative_entropy),
        absolute_precision_floor=precision_floor,
    )
    modular_limit = richardson_first_order_limit(
        epsilons,
        np.asarray(modular_energy),
        absolute_precision_floor=precision_floor,
    )
    nonaffinity_amplitude = 0.01
    endpoint = finite_gibbs_state(
        hamiltonian + nonaffinity_amplitude * perturbation, beta
    )
    midpoint = finite_gibbs_state(
        hamiltonian + (nonaffinity_amplitude / 2.0) * perturbation, beta
    )
    affine_midpoint = 0.5 * (reference + endpoint)

    assert relative_assessment.passed is True
    assert modular_limit.passed is True
    assert torch.linalg.matrix_norm(midpoint - affine_midpoint).item() > 1e-9


def test_isospectral_unitary_family_is_quadratic_and_has_zero_entropy_change() -> None:
    backend, _, reference, _, _ = _experiment()
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    generator = backend.site_operator(pauli_x, site=2)
    amplitudes = np.logspace(-4, -2, 9)
    relative_entropy = []
    modular_energy = []
    entropy_change = []

    for amplitude in amplitudes:
        unitary = torch.linalg.matrix_exp(-1j * float(amplitude) * generator)
        state = unitary @ reference @ unitary.conj().T
        relative_entropy.append(quantum_relative_entropy(state, reference))
        modular_energy.append(modular_energy_delta(state, reference))
        entropy_change.append(
            von_neumann_entropy(state) - von_neumann_entropy(reference)
        )

    identity_atol = np.finfo(float).eps * reference.shape[0]
    assert np.max(np.abs(np.asarray(entropy_change))) <= identity_atol
    assert np.max(
        np.abs(np.asarray(relative_entropy) - np.asarray(modular_energy))
    ) <= identity_atol


@pytest.mark.negative_control
def test_quadratic_gate_rejects_first_order_negative_control() -> None:
    amplitudes = np.logspace(-5, -3, 9)
    response = 0.4 * amplitudes + 0.7 * amplitudes**2

    assessment = assess_quadratic_response(amplitudes, response)

    assert assessment.passed is False


def test_local_energy_decomposition_sums_profile_spreads_and_clock_sign_is_correct() -> None:
    backend, hamiltonian, reference, excitation, local_energy = _experiment()
    profiles = []
    total_energies = []
    for time in (0.0, 0.2, 0.5, 1.0):
        state = backend.evolve(excitation, dt=time)
        profile = torch.tensor(
            [_expectation_delta(state, reference, term) for term in local_energy],
            dtype=torch.float64,
        )
        profiles.append(profile)
        total_energies.append(_expectation_delta(state, reference, hamiltonian))
        assert profile.sum().item() == pytest.approx(total_energies[-1], abs=1e-13)

    assert max(total_energies) - min(total_energies) < 1e-13
    assert torch.linalg.vector_norm(profiles[0][[0, 4]]).item() < 1e-13
    spread_fraction = (
        profiles[-1][[0, 4]].abs().sum() / profiles[-1].abs().sum()
    ).item()
    assert spread_fraction > 0.05

    epsilon = 0.01
    state = mix_states(reference, excitation, epsilon)
    energy_profile = torch.tensor(
        [_expectation_delta(state, reference, term) for term in local_energy],
        dtype=torch.float64,
    )
    weights = torch.zeros((5, 5), dtype=torch.float64)
    for i, j in backend.build_edges():
        weights[i, j] = weights[j, i] = 1.0
    phi, effective_source, background, residual = Simulator._solve_clock_constraint(
        weights,
        -energy_profile,
        mu=0.1,
        zero_mode_policy="subtract_mean",
        normalize_potential=True,
    )

    redshift = Simulator.gravitational_redshift(
        phi_emitter=phi[2], phi_observer=phi[0]
    )
    assert background < 0.0
    assert effective_source.sum().item() == pytest.approx(0.0, abs=1e-15)
    assert residual < 1e-12
    assert int(torch.argmin(phi).item()) == 2
    assert torch.exp(phi[2]) < torch.exp(phi[0])
    assert redshift > 0.0


def test_global_energy_drift_is_generator_observable_identity_not_local_gate() -> None:
    backend, hamiltonian, reference, excitation, local_energy = _experiment()
    mutated_decomposition = [term.clone() for term in local_energy]
    transferred_term = 0.25 * local_energy[0]
    mutated_decomposition[0] = mutated_decomposition[0] + transferred_term
    mutated_decomposition[1] = mutated_decomposition[1] - transferred_term
    mismatched_observable = hamiltonian + 0.25 * local_energy[0]
    times = (0.0, 0.2, 0.5, 1.0)
    arbitrary_vector = torch.arange(
        1,
        hamiltonian.shape[0] + 1,
        dtype=torch.float64,
    ).to(torch.complex128)
    arbitrary_vector += 1j * torch.flip(arbitrary_vector, dims=(0,))
    arbitrary_vector /= torch.linalg.vector_norm(arbitrary_vector)
    arbitrary_state = torch.outer(arbitrary_vector, arbitrary_vector.conj())

    canonical_totals = []
    mutated_totals = []
    mismatched_totals = []
    arbitrary_totals = []
    for time in times:
        state = backend.evolve(excitation, dt=time)
        canonical_totals.append(
            sum(
                _expectation_delta(state, reference, term)
                for term in local_energy
            )
        )
        mutated_totals.append(
            sum(
                _expectation_delta(state, reference, term)
                for term in mutated_decomposition
            )
        )
        mismatched_totals.append(
            _expectation_delta(state, reference, mismatched_observable)
        )
        arbitrary_totals.append(
            _expectation_delta(
                backend.evolve(arbitrary_state, dt=time),
                reference,
                hamiltonian,
            )
        )

    assert max(canonical_totals) - min(canonical_totals) < 1e-13
    assert max(mutated_totals) - min(mutated_totals) < 1e-13
    assert max(arbitrary_totals) - min(arbitrary_totals) < 1e-13
    assert mutated_totals == pytest.approx(canonical_totals, abs=1e-13)
    assert max(mismatched_totals) - min(mismatched_totals) > 1e-5


@pytest.mark.negative_control
def test_profile_spreading_is_not_automatic_in_commuting_ising_control() -> None:
    config = SimulatorConfig.for_chain(
        n=5,
        beta=1.0,
        boundary="open",
        hamiltonian="ising",
    )
    backend = ExactBackend(config)
    reference = backend.prepare_state()
    local_energy = backend.build_local_energy_operators()
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    local_unitary = backend.site_operator(pauli_x, site=2)
    excitation = local_unitary @ reference @ local_unitary.conj().T

    initial = torch.tensor(
        [_expectation_delta(excitation, reference, term) for term in local_energy]
    )
    evolved = backend.evolve(excitation, dt=1.0)
    evolved_profile = torch.tensor(
        [_expectation_delta(evolved, reference, term) for term in local_energy]
    )

    assert torch.allclose(evolved_profile, initial, atol=1e-13, rtol=0.0)


@pytest.mark.negative_control
def test_reduced_modular_source_is_blind_but_kms_energy_density_is_not() -> None:
    config = SimulatorConfig.for_chain(
        n=5,
        beta=1.3,
        boundary="open",
        hamiltonian="heisenberg",
    )
    simulator = Simulator(config)
    reference = simulator.prepare()
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    local_unitary = simulator.backend.site_operator(pauli_x, site=2)
    state = mix_states(
        reference,
        local_unitary @ reference @ local_unitary.conj().T,
        epsilon=0.01,
    )
    cells = [[site] for site in range(5)]

    config.pi_time.source_model = "negative_modular_energy_candidate"
    reduced_modular = simulator._compute_source_term(
        state,
        cells,
        config.pi_time,
        reference_state=reference,
    )
    config.pi_time.source_model = "negative_kms_energy_density_candidate"
    kms_energy_density = simulator._compute_source_term(
        state,
        cells,
        config.pi_time,
        reference_state=reference,
    )

    local_energy = simulator.backend.build_local_energy_operators()
    expected = -config.substrate.beta * torch.tensor(
        [_expectation_delta(state, reference, term) for term in local_energy],
        dtype=torch.float64,
    )
    global_modular = modular_energy_delta(state, reference)

    assert torch.linalg.vector_norm(reduced_modular).item() < 1e-12
    assert torch.linalg.vector_norm(kms_energy_density).item() > 1e-3
    assert torch.allclose(kms_energy_density, expected, atol=1e-14, rtol=0.0)
    assert kms_energy_density.sum().item() == pytest.approx(
        -global_modular, abs=2e-14
    )
