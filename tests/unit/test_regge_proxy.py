import math

import pytest
import torch

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


def test_delaunay_proxy_rejects_non_2d_embedding() -> None:
    with pytest.raises(ValueError, match="requires 2D"):
        build_delaunay_proxy(torch.zeros((4, 3), dtype=torch.float64))
