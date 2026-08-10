import math

import pytest
import torch

from popgp import Simulator, SimulatorConfig
from popgp.information import modular_energy_delta
from popgp.simulator import PiLocResult


def test_grid_convenience_config_has_valid_single_qubit_cells() -> None:
    config = SimulatorConfig.for_grid(width=3, height=3)

    assert config.substrate.n_qubits == 9
    assert config.pi_res.cell_dim == 1


def test_ising_pi_res_selects_contiguous_two_site_cells() -> None:
    config = SimulatorConfig.for_chain(n=4, beta=1.0)
    config.substrate.hamiltonian = "ising"
    config.pi_res.cell_dim = 2
    simulator = Simulator(config)

    result = simulator.run_pi_res(simulator.prepare())

    assert result.cells == [[0, 1], [2, 3]]
    assert result.leakage == pytest.approx(5.712651e-4, rel=1e-6)


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


@pytest.mark.parametrize(
    "source_model",
    [
        "negative_relative_entropy_candidate",
        "negative_modular_energy_candidate",
        "negative_kms_energy_density_candidate",
    ],
)
def test_candidate_source_requires_explicit_reference_state(source_model: str) -> None:
    config = SimulatorConfig.for_chain(n=2)
    config.pi_time.source_model = source_model
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


def test_kms_energy_density_candidate_aggregates_partitioned_sites() -> None:
    config = SimulatorConfig.for_chain(n=4, beta=0.7)
    config.pi_time.source_model = "negative_kms_energy_density_candidate"
    config.pi_time.source_scale = 1.7
    simulator = Simulator(config)
    reference = simulator.prepare()
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    local_unitary = simulator.backend.site_operator(pauli_x, site=1)
    state = 0.99 * reference + 0.01 * (
        local_unitary @ reference @ local_unitary.conj().T
    )

    source = simulator._compute_source_term(
        state,
        [[0, 1], [2, 3]],
        config.pi_time,
        reference_state=reference,
    )
    local_energy = simulator.backend.build_local_energy_operators()
    expected = []
    for cell in ([0, 1], [2, 3]):
        cell_operator = sum(
            (local_energy[site] for site in cell),
            torch.zeros_like(local_energy[0]),
        )
        expected.append(
            -config.pi_time.source_scale
            * config.substrate.beta
            * torch.trace((state - reference) @ cell_operator).real.item()
        )

    assert torch.allclose(
        source,
        torch.tensor(expected, dtype=torch.float64),
        atol=1e-14,
        rtol=0.0,
    )
    assert source.sum().item() == pytest.approx(
        -config.pi_time.source_scale
        * modular_energy_delta(state, reference),
        abs=2e-14,
    )


@pytest.mark.parametrize(
    "delta",
    [1e-3, 1e-5, 1e-6, 1e-7, 3e-8, 1.5e-8, 1.1e-8, 1e-9],
)
@pytest.mark.parametrize("padded", [False, True])
def test_mds_canonical_frame_is_isometric_near_rank_threshold(
    delta: float,
    padded: bool,
) -> None:
    first = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    second = torch.tensor([1.0, -2.0, 0.0, 2.0, -1.0], dtype=torch.float64)
    coords = torch.column_stack(
        [
            first / torch.linalg.vector_norm(first),
            delta * second / torch.linalg.vector_norm(second),
        ]
    )
    if padded:
        coords = torch.column_stack(
            [coords, torch.zeros(coords.shape[0], dtype=torch.float64)]
        )

    canonical = Simulator._canonicalize_embedding(coords)
    before = torch.cdist(coords, coords)
    after = torch.cdist(canonical, canonical)
    mask = torch.triu(torch.ones_like(before, dtype=torch.bool), diagonal=1)
    relative_change = torch.max(
        torch.abs(after[mask] - before[mask]) / before[mask]
    ).item()

    assert relative_change < 1e-14


def test_rank_deficient_mds_canonical_frame_fixes_degenerate_rotation() -> None:
    represented = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=torch.float64,
    )
    coords = torch.column_stack(
        [represented, torch.zeros(4, dtype=torch.float64)]
    )
    angle = 0.731
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    canonical = Simulator._canonicalize_embedding(coords)
    rotated = Simulator._canonicalize_embedding(coords @ rotation)

    assert torch.allclose(canonical, rotated, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(canonical[:, 2]).item() == 0


def _grid_hop_distances(width: int, height: int) -> torch.Tensor:
    coordinates = torch.tensor(
        [(x, y) for y in range(height) for x in range(width)],
        dtype=torch.float64,
    )
    return torch.cdist(coordinates, coordinates, p=1)


def _grid_edges(width: int, height: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            node = y * width + x
            if x + 1 < width:
                edges.add((node, node + 1))
            if y + 1 < height:
                edges.add((node, node + width))
    return edges


def _star_hop_distances(n: int) -> torch.Tensor:
    distances = torch.full((n, n), 2.0, dtype=torch.float64)
    distances.fill_diagonal_(0.0)
    distances[0, 1:] = 1.0
    distances[1:, 0] = 1.0
    return distances


def test_mds_canonicalization_terminates_when_first_label_is_symmetry_center() -> None:
    grid = _grid_hop_distances(3, 3)
    center_first = torch.tensor([4, 0, 1, 2, 3, 5, 6, 7, 8])
    fixtures = [
        _star_hop_distances(5),
        _star_hop_distances(6),
        grid[center_first][:, center_first],
    ]

    for distances in fixtures:
        for dimension in range(1, distances.shape[0]):
            coords = Simulator._classical_mds(distances, dimension)
            assert torch.isfinite(coords).all()


def test_mds_direct_anchor_scan_uses_global_rank_tolerance() -> None:
    coords = torch.tensor(
        [[1e-9, 0.0], [1.0, 1.0], [-1.0, 0.5], [-1e-9, -1.5]],
        dtype=torch.float64,
    )

    canonical = Simulator._canonicalize_embedding(coords)

    assert torch.isfinite(canonical).all()
    assert torch.allclose(
        torch.cdist(canonical, canonical),
        torch.cdist(coords, coords),
        atol=1e-14,
        rtol=1e-14,
    )


def test_mds_canonicalization_is_scale_covariant_below_unit_scale() -> None:
    chain = torch.cdist(
        torch.arange(6, dtype=torch.float64).reshape(-1, 1),
        torch.arange(6, dtype=torch.float64).reshape(-1, 1),
        p=1,
    )
    for scale in (1.0, 1e-4, 1e-8, 1e-10, 1e-11, 1e-13, 1e-15):
        for dimension in range(1, 6):
            embedded = Simulator._classical_mds(chain * scale, dimension)
            assert torch.isfinite(embedded).all()

    generator = torch.Generator().manual_seed(2718)
    raw = torch.randn((8, 3), dtype=torch.float64, generator=generator)
    orthonormal, _ = torch.linalg.qr(raw - raw.mean(dim=0, keepdim=True))
    for scale in (1e-8, 1e-9, 1e-10):
        for ratio in (1e-8, 1.5e-8, 1e-7):
            coords = orthonormal * torch.tensor(
                [scale, scale * math.sqrt(ratio), scale * ratio],
                dtype=torch.float64,
            )
            canonical = Simulator._canonicalize_embedding(coords)
            before = torch.cdist(coords, coords)
            after = torch.cdist(canonical, canonical)
            mask = torch.triu(torch.ones_like(before, dtype=torch.bool), diagonal=1)
            relative_change = torch.max(
                torch.abs(after[mask] - before[mask]) / before[mask]
            ).item()
            assert torch.isfinite(canonical).all()
            assert relative_change < 1e-14


@pytest.mark.parametrize(
    "coords",
    [
        torch.tensor(
            [(x, y) for y in range(3) for x in range(3)],
            dtype=torch.float64,
        ),
        torch.randn(
            (8, 3),
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(31415),
        ),
    ],
)
def test_mds_canonical_frame_is_positively_homogeneous(coords: torch.Tensor) -> None:
    coords = coords - coords.mean(dim=0, keepdim=True)
    canonical = Simulator._canonicalize_embedding(coords)
    magnitude = float(torch.max(torch.abs(canonical)).item())

    for scale in (1e20, 1e10, 1e-5, 1e-10, 1e-13, 1e-14, 1e-15):
        scaled = Simulator._canonicalize_embedding(scale * coords)
        maximum_deviation = torch.max(
            torch.abs(scaled - scale * canonical)
        ).item()
        assert maximum_deviation <= 1e-12 * scale * magnitude


def test_mds_anchor_scan_never_selects_a_zero_residual_row() -> None:
    coords = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-2.0, -1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    for scale in (1.0, 1e-6, 1e-15):
        scaled = scale * coords
        canonical = Simulator._canonicalize_embedding(scaled)
        assert torch.isfinite(canonical).all()
        assert torch.allclose(
            torch.cdist(canonical, canonical),
            torch.cdist(scaled, scaled),
            atol=1e-28,
            rtol=1e-14,
        )


def test_mds_canonicalization_docstring_names_maximum_volume_selection() -> None:
    summary = (Simulator._canonicalize_embedding.__doc__ or "").splitlines()[0]

    assert "maximum-volume anchors with label-ordered ties" in summary
    assert "using label-ordered anchors" not in summary


def test_mds_stress_and_geometry_status_are_scale_invariant() -> None:
    distance_matrix = _grid_hop_distances(3, 3)
    one_dimensional = Simulator._classical_mds(distance_matrix, 1)
    expected_stress = Simulator._mds_stress(distance_matrix, one_dimensional)
    assert expected_stress > 0

    edges = sorted(_grid_edges(3, 3))
    weights = torch.zeros_like(distance_matrix)
    for first, second in edges:
        weights[first, second] = weights[second, first] = 1.0
    simulator = Simulator(SimulatorConfig.for_grid(3, 3))
    baseline: tuple[int, str] | None = None

    for scale in (1.0, 1e-6, 1e-9, 1e-12):
        assert Simulator._mds_stress(
            distance_matrix * scale,
            one_dimensional * scale,
        ) == pytest.approx(expected_stress, rel=1e-13)
        locality = PiLocResult(
            mi_matrix=weights,
            distance_matrix=distance_matrix * scale,
            weight_matrix=weights,
            edges=edges,
            connectivity_method="scale_invariance_fixture",
        )
        geometry = simulator.run_pi_geom(locality)
        status = (geometry.D_star, geometry.embedding_status)
        if baseline is None:
            baseline = status
        assert status == baseline

    degenerate = PiLocResult(
        mi_matrix=weights,
        distance_matrix=torch.zeros_like(distance_matrix),
        weight_matrix=weights,
        edges=edges,
        connectivity_method="degenerate_distance_fixture",
    )
    degenerate_geometry = simulator.run_pi_geom(degenerate)
    assert math.isinf(degenerate_geometry.stress)
    assert degenerate_geometry.embedding_status == "poor_fit"


@pytest.mark.parametrize(
    ("distance_matrix", "maximum_dimension"),
    [
        (_grid_hop_distances(3, 3), 6),
        (
            torch.tensor(
                [
                    [0, 1, 2, 2, 1],
                    [1, 0, 1, 2, 2],
                    [2, 1, 0, 1, 2],
                    [2, 2, 1, 0, 1],
                    [1, 2, 2, 1, 0],
                ],
                dtype=torch.float64,
            ),
            3,
        ),
    ],
)
def test_classical_mds_is_invariant_to_degenerate_eigenbasis_for_every_dimension(
    monkeypatch: pytest.MonkeyPatch,
    distance_matrix: torch.Tensor,
    maximum_dimension: int,
) -> None:
    n = distance_matrix.shape[0]
    centering = torch.eye(n, dtype=torch.float64) - torch.ones(
        (n, n), dtype=torch.float64
    ) / n
    gram = -0.5 * centering @ distance_matrix.square() @ centering
    original_eigh = torch.linalg.eigh
    eigenvalues, eigenvectors = original_eigh(gram)
    alternative = eigenvectors.clone()
    tolerance = 1e-10
    for index in range(n - 1):
        if (
            eigenvalues[index] > tolerance
            and abs(float(eigenvalues[index + 1] - eigenvalues[index])) < tolerance
        ):
            angle = 0.731
            first = eigenvectors[:, index]
            second = eigenvectors[:, index + 1]
            alternative[:, index] = math.cos(angle) * first + math.sin(angle) * second
            alternative[:, index + 1] = (
                -math.sin(angle) * first + math.cos(angle) * second
            )
            break
    else:
        pytest.fail("fixture must contain a degenerate positive eigenspace")

    expected = {
        dimension: Simulator._classical_mds(distance_matrix, dimension)
        for dimension in range(1, maximum_dimension + 1)
    }

    def alternative_eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if matrix.shape == gram.shape and torch.allclose(
            matrix, gram, atol=1e-13, rtol=1e-13
        ):
            return eigenvalues.clone(), alternative.clone()
        return original_eigh(matrix)

    monkeypatch.setattr(torch.linalg, "eigh", alternative_eigh)
    for dimension in range(1, maximum_dimension + 1):
        actual = Simulator._classical_mds(distance_matrix, dimension)
        assert torch.allclose(
            actual,
            expected[dimension],
            atol=1e-11,
            rtol=1e-11,
        )


def test_gravitational_redshift_is_independent_of_default_dtype() -> None:
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        float64_result = Simulator.gravitational_redshift(
            -0.009963709390575564,
            0.0022578960410844567,
        )
        torch.set_default_dtype(torch.float32)
        float32_result = Simulator.gravitational_redshift(
            -0.009963709390575564,
            0.0022578960410844567,
        )
    finally:
        torch.set_default_dtype(original_dtype)

    assert float32_result == float64_result


def test_kms_energy_density_candidate_rejects_beta_mismatch() -> None:
    config = SimulatorConfig.for_chain(n=4, beta=1.3)
    config.pi_time.source_model = "negative_kms_energy_density_candidate"
    config.pi_time.beta_kms = 0.5
    simulator = Simulator(config)
    reference = simulator.prepare()

    with pytest.raises(ValueError, match="KMS_REFERENCE_TRACE_DISTANCE_TOLERANCE"):
        simulator._compute_source_term(
            reference,
            [[0, 1], [2, 3]],
            config.pi_time,
            reference_state=reference,
        )


def test_kms_energy_density_candidate_rejects_faithful_non_gibbs_reference() -> None:
    config = SimulatorConfig.for_chain(n=2, beta=0.7)
    config.pi_time.source_model = "negative_kms_energy_density_candidate"
    simulator = Simulator(config)
    state = simulator.prepare()
    non_gibbs = torch.diag(
        torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.complex128)
    )

    with pytest.raises(ValueError, match="KMS_REFERENCE_TRACE_DISTANCE_TOLERANCE"):
        simulator._compute_source_term(
            state,
            [[0], [1]],
            config.pi_time,
            reference_state=non_gibbs,
        )


def test_run_propagates_reference_state_to_kms_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimulatorConfig.for_chain(n=4, beta=0.7)
    config.pi_time.source_model = "negative_kms_energy_density_candidate"
    simulator = Simulator(config)
    reference = simulator.prepare()
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    local_unitary = simulator.backend.site_operator(pauli_x, site=1)
    state = 0.99 * reference + 0.01 * (
        local_unitary @ reference @ local_unitary.conj().T
    )
    monkeypatch.setattr(simulator, "prepare", lambda: state)

    result = simulator.run(reference_state=reference)

    assert result.pi_time is not None
    assert result.pi_time.source_model == "negative_kms_energy_density_candidate"
    assert torch.linalg.vector_norm(result.pi_time.delta_rho_raw).item() > 0.0
    assert result.pi_time.delta_rho_raw.sum().item() == pytest.approx(
        -modular_energy_delta(state, reference),
        abs=2e-14,
    )


@pytest.mark.parametrize("cells", [[[0, 1], [2]], [[0, 1, 2, 3], []]])
def test_kms_energy_density_candidate_requires_complete_partition(
    cells: list[list[int]],
) -> None:
    config = SimulatorConfig.for_chain(n=4, beta=0.7)
    config.pi_time.source_model = "negative_kms_energy_density_candidate"
    simulator = Simulator(config)
    reference = simulator.prepare()

    with pytest.raises(ValueError, match="disjoint partition"):
        simulator._compute_source_term(
            reference,
            cells,
            config.pi_time,
            reference_state=reference,
        )
