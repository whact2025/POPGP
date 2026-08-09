import pytest
import torch

from popgp import Simulator, SimulatorConfig
from popgp.diagnostics import edge_recovery_metrics
from popgp.simulator import PiLocResult, PiResResult

pytestmark = pytest.mark.scientific


def _grid_edges(width: int, height: int) -> set[tuple[int, int]]:
    edges = set()
    for y in range(height):
        for x in range(width):
            node = y * width + x
            if x + 1 < width:
                edges.add((node, node + 1))
            if y + 1 < height:
                edges.add((node, node + width))
    return edges


def _correlation_matrix(n: int, edges: set[tuple[int, int]]) -> torch.Tensor:
    matrix = torch.full((n, n), 0.02, dtype=torch.float64)
    matrix.fill_diagonal_(0.0)
    for i, j in edges:
        matrix[i, j] = matrix[j, i] = 0.25
    return matrix


def _infer_edges(weights: torch.Tensor) -> tuple[set[tuple[int, int]], bool | None]:
    n = weights.shape[0]
    distances = torch.zeros_like(weights)
    mask = ~torch.eye(n, dtype=torch.bool)
    distances[mask] = -torch.log(weights[mask] / (torch.e * weights.max()))
    config = SimulatorConfig.for_chain(n=n)
    simulator = Simulator(config)
    _, edges, _, _, separable = simulator._graph_geodesics(
        distances,
        weights,
        n,
        method="adaptive_gap",
        k_nearest=3,
        minimum_gap_ratio=1.5,
    )
    return set(edges), separable


def test_adaptive_gap_recovers_grid_without_reference_edges() -> None:
    reference = _grid_edges(3, 3)
    inferred, separable = _infer_edges(_correlation_matrix(9, reference))
    metrics = edge_recovery_metrics(inferred, reference)

    assert separable is True
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_adaptive_gap_is_permutation_equivariant() -> None:
    reference = _grid_edges(3, 3)
    weights = _correlation_matrix(9, reference)
    permutation = torch.tensor([4, 8, 0, 6, 2, 7, 1, 5, 3])
    permuted_weights = weights[permutation][:, permutation]

    permuted_edges, separable = _infer_edges(permuted_weights)
    mapped_back = {
        tuple(sorted((int(permutation[i]), int(permutation[j]))))
        for i, j in permuted_edges
    }

    assert separable is True
    assert mapped_back == reference


def test_nonseparable_correlations_are_not_called_identifiable() -> None:
    n = 6
    weights = torch.full((n, n), 0.2, dtype=torch.float64)
    weights.fill_diagonal_(0.0)

    inferred, separable = _infer_edges(weights)

    assert separable is False
    assert len(inferred) == n - 1


def test_adaptive_gap_rejects_non_separating_ratio() -> None:
    weights = _correlation_matrix(4, {(0, 1), (1, 2), (2, 3)})
    distances = torch.ones_like(weights)
    simulator = Simulator(SimulatorConfig.for_chain(n=4))

    with pytest.raises(ValueError, match="greater than 1"):
        simulator._graph_geodesics(
            distances,
            weights,
            4,
            method="adaptive_gap",
            k_nearest=3,
            minimum_gap_ratio=1.0,
        )


def _petersen_graph_fixture() -> PiLocResult:
    n = 10
    outer = {(i, (i + 1) % 5) for i in range(5)}
    spokes = {(i, i + 5) for i in range(5)}
    inner = {(5 + i, 5 + (i + 2) % 5) for i in range(5)}
    edges = {tuple(sorted(edge)) for edge in outer | spokes | inner}
    weights = torch.zeros((n, n), dtype=torch.float64)
    distances = torch.full((n, n), float("inf"), dtype=torch.float64)
    distances.fill_diagonal_(0.0)
    for i, j in edges:
        weights[i, j] = weights[j, i] = 1.0
        distances[i, j] = distances[j, i] = 1.0
    distances = Simulator._floyd_warshall(distances, n)
    return PiLocResult(
        mi_matrix=weights,
        distance_matrix=distances,
        weight_matrix=weights,
        edges=sorted(edges),
        connectivity_method="non_geometric_fixture",
    )


def test_petersen_control_exposes_dimension_penalty_sensitivity() -> None:
    fixture = _petersen_graph_fixture()
    weak_penalty_config = SimulatorConfig.for_chain(n=10)
    weak_penalty_config.pi_geom.lambda_dim = 0.01
    strong_penalty_config = SimulatorConfig.for_chain(n=10)
    strong_penalty_config.pi_geom.lambda_dim = 1.0

    weak_penalty = Simulator(weak_penalty_config).run_pi_geom(fixture)
    strong_penalty = Simulator(strong_penalty_config).run_pi_geom(fixture)

    assert weak_penalty.D_star == 4
    assert weak_penalty.embedding_status == "non_geometric_dimension"
    assert strong_penalty.D_star == 2
    assert strong_penalty.embedding_status == "poor_fit"
    assert strong_penalty.stress > 2.0 * weak_penalty.stress


def test_disjoint_bell_pairs_are_marked_nonseparable() -> None:
    config = SimulatorConfig.for_chain(n=4)
    config.pi_res.cell_dim = 1
    simulator = Simulator(config)
    state_vector = torch.zeros(16, dtype=torch.complex128)
    for basis in range(16):
        bits = [(basis >> (3 - qubit)) & 1 for qubit in range(4)]
        if bits[0] == bits[2] and bits[1] == bits[3]:
            state_vector[basis] = 0.5
    state = torch.outer(state_vector, state_vector.conj())
    resolution = PiResResult(
        cells=[[0], [1], [2], [3]],
        leakage=0.0,
    )

    locality = simulator.run_pi_loc(state, resolution)

    assert locality.connectivity_separable is False
    assert (0, 2) in locality.edges
    assert (1, 3) in locality.edges
