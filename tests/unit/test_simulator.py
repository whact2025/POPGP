import math

import pytest
import torch

from popgp import Simulator, SimulatorConfig
from popgp.simulator import PiLocResult


def test_grid_convenience_config_has_valid_single_qubit_cells() -> None:
    config = SimulatorConfig.for_grid(width=3, height=3)

    assert config.substrate.n_qubits == 9
    assert config.pi_res.cell_dim == 1


def test_unscreened_clock_solve_removes_constant_source_mode() -> None:
    weights = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    source = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)

    phi, effective, background, residual = Simulator._solve_clock_constraint(
        weights,
        source,
        mu=0.0,
        zero_mode_policy="subtract_mean",
        normalize_potential=True,
    )

    assert effective.sum().item() == pytest.approx(0.0, abs=1e-13)
    assert phi.mean().item() == pytest.approx(0.0, abs=1e-13)
    assert background == pytest.approx(1.0 / 3.0)
    assert residual < 1e-12


def test_unscreened_clock_solve_can_reject_incompatible_source() -> None:
    weights = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    source = torch.tensor([1.0, 0.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="must sum to zero"):
        Simulator._solve_clock_constraint(
            weights,
            source,
            mu=0.0,
            zero_mode_policy="require_zero_sum",
            normalize_potential=False,
        )


def test_redshift_sign_for_emitter_deeper_in_negative_well() -> None:
    z = Simulator.gravitational_redshift(phi_emitter=-0.1, phi_observer=0.0)

    assert z == pytest.approx(math.exp(0.1) - 1.0)
    assert z > 0.0


def test_geometry_reports_stress_separately_from_objective() -> None:
    config = SimulatorConfig.for_chain(n=2)
    config.pi_geom.lambda_dim = 0.5
    config.pi_geom.D_max = 2
    simulator = Simulator(config)
    distances = torch.tensor(
        [[0.0, 1.0, math.sqrt(2.0), 1.0],
         [1.0, 0.0, 1.0, math.sqrt(2.0)],
         [math.sqrt(2.0), 1.0, 0.0, 1.0],
         [1.0, math.sqrt(2.0), 1.0, 0.0]],
        dtype=torch.float64,
    )
    weights = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    pi_loc = PiLocResult(
        mi_matrix=weights,
        distance_matrix=distances,
        weight_matrix=weights,
        edges=[(0, 1), (1, 2), (2, 3), (0, 3)],
    )

    result = simulator.run_pi_geom(pi_loc)

    assert result.stress == pytest.approx(result.stress_by_dimension[result.D_star])
    assert result.objective == pytest.approx(result.objective_by_dimension[result.D_star])
    assert result.objective >= result.stress


def test_graph_geodesics_preserves_float64_distance_dtype() -> None:
    simulator = Simulator(SimulatorConfig.for_chain(n=4))
    weights = torch.tensor(
        [
            [0.0, 0.3, 0.0, 0.0],
            [0.3, 0.0, 0.2, 0.0],
            [0.0, 0.2, 0.0, 0.1],
            [0.0, 0.0, 0.1, 0.0],
        ],
        dtype=torch.float64,
    )
    distances = torch.full_like(weights, float("inf"))
    distances.fill_diagonal_(0.0)
    distances[weights > 0] = -torch.log(weights[weights > 0])

    graph_distances, *_ = simulator._graph_geodesics(
        distances,
        weights,
        4,
        method="adaptive_gap",
        k_nearest=3,
        minimum_gap_ratio=1.1,
    )

    assert graph_distances.dtype == torch.float64


def test_exact_path_has_negligible_one_dimensional_mds_stress() -> None:
    positions = torch.tensor(
        [[0.0], [1.0 / 3.0], [0.7], [1.1]], dtype=torch.float64
    )
    distances = torch.cdist(positions, positions)
    coords = Simulator._classical_mds(distances, D=1)

    assert Simulator._mds_stress(distances, coords) < 1e-12


def test_mds_canonical_frame_is_rotation_invariant() -> None:
    coords = torch.tensor(
        [[0.0, 0.0], [1.0, 0.1], [0.2, 1.2], [1.1, 0.9]],
        dtype=torch.float64,
    )
    angle = 0.491
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )

    canonical = Simulator._canonicalize_embedding(coords)
    rotated = Simulator._canonicalize_embedding(coords @ rotation)

    assert torch.allclose(canonical, rotated, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        torch.cdist(canonical, canonical),
        torch.cdist(coords, coords),
        atol=1e-12,
        rtol=1e-12,
    )


def test_singleton_resolution_reports_uncomputed_leakage_and_admissibility(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = SimulatorConfig.for_chain(n=2, beta=1.0)
    config.pi_res.cell_dim = 1
    config.pi_res.retention_epsilon = 0.0
    simulator = Simulator(config)

    result = simulator.run_pi_res(simulator.prepare())

    assert result.leakage is None
    assert result.retention_loss is not None
    assert result.retention_loss > 0.0
    assert result.admissible is False
    assert result.n_total == 1
    assert result.n_admissible == 0
    assert "returned an inadmissible decomposition" in caplog.text


def test_finite_graph_spectral_peak_ignores_float32_zero_mode() -> None:
    config = SimulatorConfig.for_chain(n=4)
    simulator = Simulator(config)
    weights = torch.tensor(
        [
            [0.0, 0.31, 0.0, 0.0],
            [0.31, 0.0, 0.27, 0.0],
            [0.0, 0.27, 0.0, 0.29],
            [0.0, 0.0, 0.29, 0.0],
        ],
        dtype=torch.float32,
    )

    peak, peak_time, curve = simulator._spectral_dimension(weights, n=4)

    assert 0.5 < peak < 1.5
    assert peak_time is not None
    assert curve


def test_candidate_source_requires_explicit_reference_state() -> None:
    config = SimulatorConfig.for_chain(n=2)
    config.pi_time.source_model = "negative_modular_energy_candidate"
    simulator = Simulator(config)
    state = simulator.prepare()

    with pytest.raises(ValueError, match="requires a reference_state"):
        simulator._compute_source_term(
            state,
            [[0], [1]],
            config.pi_time,
        )


def test_negative_candidate_source_models_are_available_explicitly() -> None:
    config = SimulatorConfig.for_chain(n=2)
    simulator = Simulator(config)
    reference = torch.diag(
        torch.tensor([0.42, 0.28, 0.18, 0.12], dtype=torch.complex128)
    )
    excitation = torch.diag(
        torch.tensor([0.20, 0.30, 0.30, 0.20], dtype=torch.complex128)
    )

    config.pi_time.source_model = "negative_relative_entropy_candidate"
    relative_source = simulator._compute_source_term(
        excitation,
        [[0], [1]],
        config.pi_time,
        reference_state=reference,
    )
    config.pi_time.source_model = "negative_modular_energy_candidate"
    modular_source = simulator._compute_source_term(
        excitation,
        [[0], [1]],
        config.pi_time,
        reference_state=reference,
    )

    assert torch.all(relative_source <= 0)
    assert torch.isfinite(modular_source).all()
