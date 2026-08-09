import math

import pytest
import torch

from popgp import Simulator, SimulatorConfig
from popgp.geometry import build_delaunay_proxy, vertex_deficits_2d


def test_planar_grid_proxy_has_flat_interior_and_disk_boundary_sum() -> None:
    coords = torch.tensor(
        [[x, y] for y in range(3) for x in range(3)], dtype=torch.float64
    )
    simplices = build_delaunay_proxy(coords)
    deficits, boundary = vertex_deficits_2d(coords, simplices)

    assert 4 not in boundary
    assert deficits[4] == pytest.approx(0.0, abs=1e-12)
    assert sum(deficits.values()) == pytest.approx(2.0 * math.pi, abs=1e-12)


def test_three_by_three_grid_corner_deficits_are_float64_symmetric() -> None:
    coords = torch.tensor(
        [[x, y] for y in range(3) for x in range(3)], dtype=torch.float64
    )
    simplices = build_delaunay_proxy(coords)
    deficits, _ = vertex_deficits_2d(coords, simplices)

    corners = torch.tensor(
        [deficits[index] for index in (0, 2, 6, 8)], dtype=torch.float64
    )
    assert torch.allclose(
        corners,
        torch.full((4,), math.pi / 2, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_graph_to_mds_to_regge_pipeline_preserves_square_corner_deficits() -> None:
    simulator = Simulator(SimulatorConfig.for_grid(width=3, height=3))
    source_coords = torch.tensor(
        [[x / 3.0, y / 3.0] for y in range(3) for x in range(3)],
        dtype=torch.float32,
    )
    pair_distances = torch.cdist(source_coords, source_coords)
    weights = torch.exp(-pair_distances)
    weights.fill_diagonal_(0.0)

    graph_distances, *_ = simulator._graph_geodesics(
        pair_distances,
        weights,
        9,
        method="knn_mst",
        k_nearest=8,
        minimum_gap_ratio=1.1,
    )
    embedded = simulator._classical_mds(graph_distances, D=2)
    simplices = build_delaunay_proxy(embedded)
    deficits, _ = vertex_deficits_2d(embedded, simplices)

    corners = torch.tensor(
        [deficits[index] for index in (0, 2, 6, 8)], dtype=torch.float64
    )
    assert graph_distances.dtype == torch.float32
    assert embedded.dtype == torch.float32
    assert torch.allclose(
        corners,
        torch.full((4,), math.pi / 2, dtype=torch.float64),
        atol=2e-6,
        rtol=0.0,
    )


def test_delaunay_proxy_rejects_non_2d_embedding() -> None:
    with pytest.raises(ValueError, match="requires 2D"):
        build_delaunay_proxy(torch.zeros((4, 3), dtype=torch.float64))
