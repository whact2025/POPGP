"""Explicit embedding-space triangulation and 2D angle-deficit proxies."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError


def build_delaunay_proxy(coords: torch.Tensor) -> list[tuple[int, int, int]]:
    """Triangulate 2D embedding coordinates as a declared visualization proxy."""
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("the Delaunay proxy currently requires 2D coordinates")
    if coords.shape[0] < 3:
        raise ValueError("at least three points are required for triangulation")
    try:
        triangulation = Delaunay(coords.detach().cpu().numpy())
    except QhullError as exc:
        raise ValueError("embedding points do not define a stable 2D triangulation") from exc
    return [tuple(sorted(map(int, simplex))) for simplex in triangulation.simplices]


def _triangle_angles(points: torch.Tensor) -> tuple[float, float, float]:
    angles = []
    for vertex in range(3):
        first = points[(vertex + 1) % 3] - points[vertex]
        second = points[(vertex + 2) % 3] - points[vertex]
        denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
        if denominator <= 0:
            raise ValueError("degenerate triangle in Regge proxy")
        cosine = torch.dot(first, second) / denominator
        angles.append(float(torch.acos(cosine.clamp(-1.0, 1.0)).item()))
    return angles[0], angles[1], angles[2]


def vertex_deficits_2d(
    coords: torch.Tensor,
    simplices: list[tuple[int, int, int]],
) -> tuple[dict[int, float], set[int]]:
    """Return 2D angle deficits with explicit disk-boundary treatment.

    Interior vertices use ``2π-sum(theta)`` and boundary vertices use
    ``π-sum(theta)``. These are embedding-space proxy diagnostics; intrinsic
    Regge curvature requires independently reconstructed edge lengths and a
    convergent complex.
    """
    coords = coords.to(dtype=torch.float64)
    angle_sums = torch.zeros(coords.shape[0], dtype=torch.float64)
    edge_counts: Counter[tuple[int, int]] = Counter()
    for simplex in simplices:
        if len(set(simplex)) != 3:
            raise ValueError("simplices must contain three distinct vertices")
        points = coords[torch.tensor(simplex, dtype=torch.long)]
        angles = _triangle_angles(points)
        for vertex, angle in zip(simplex, angles, strict=True):
            angle_sums[vertex] += angle
        for i, j in ((0, 1), (1, 2), (0, 2)):
            edge_counts[tuple(sorted((simplex[i], simplex[j])))] += 1

    boundary_vertices = {
        vertex
        for edge, count in edge_counts.items()
        if count == 1
        for vertex in edge
    }
    deficits = {}
    for vertex in range(coords.shape[0]):
        reference_angle = np.pi if vertex in boundary_vertices else 2.0 * np.pi
        deficits[vertex] = float(reference_angle - angle_sums[vertex].item())
    return deficits, boundary_vertices
