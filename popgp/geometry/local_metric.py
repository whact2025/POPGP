"""Regularized local metric fits on an already inferred embedding."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalMetricFit:
    """One node's SPD metric and numerical identifiability diagnostics."""

    metric: torch.Tensor
    condition_number: float
    relative_residual: float
    design_rank: int
    n_constraints: int
    underdetermined: bool


def _symmetric_design(displacements: torch.Tensor) -> torch.Tensor:
    """Return the Frobenius-orthonormal basis for symmetric quadratic forms."""
    dimension = displacements.shape[1]
    columns = []
    for a in range(dimension):
        columns.append(displacements[:, a] ** 2)
    for a in range(dimension):
        for b in range(a + 1, dimension):
            columns.append((2.0**0.5) * displacements[:, a] * displacements[:, b])
    return torch.stack(columns, dim=1)


def _identity_parameters(dimension: int, *, dtype: torch.dtype) -> torch.Tensor:
    n_parameters = dimension * (dimension + 1) // 2
    parameters = torch.zeros(n_parameters, dtype=dtype)
    parameters[:dimension] = 1.0
    return parameters


def _parameters_to_metric(parameters: torch.Tensor, dimension: int) -> torch.Tensor:
    metric = torch.zeros((dimension, dimension), dtype=parameters.dtype)
    metric.diagonal().copy_(parameters[:dimension])
    index = dimension
    for a in range(dimension):
        for b in range(a + 1, dimension):
            metric[a, b] = metric[b, a] = parameters[index] / (2.0**0.5)
            index += 1
    return metric


def reconstruct_local_metrics(
    coords: torch.Tensor,
    distance_matrix: torch.Tensor,
    edges: list[tuple[int, int]],
    *,
    regularization: float,
    eigenvalue_floor: float = 1e-8,
) -> list[LocalMetricFit]:
    """Fit ``v_ij^T h_i v_ij ≈ d_ij²`` at every embedded node.

    The fit is local to the MDS coordinates and inferred graph. It does not turn
    the embedding into an intrinsic or continuum metric. Ridge regularization
    toward the identity makes underdetermined boundary fits explicit and stable.
    """
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    if eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be positive")
    coords = coords.to(dtype=torch.float64)
    distance_matrix = distance_matrix.to(dtype=torch.float64)
    n_nodes, dimension = coords.shape
    if dimension < 1:
        raise ValueError("coords must have at least one embedding dimension")

    neighbors: list[list[int]] = [[] for _ in range(n_nodes)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    identity_parameters = _identity_parameters(dimension, dtype=torch.float64)
    identity_metric = torch.eye(dimension, dtype=torch.float64)
    fits = []
    for node, node_neighbors in enumerate(neighbors):
        if not node_neighbors:
            fits.append(
                LocalMetricFit(
                    metric=identity_metric.clone(),
                    condition_number=float("inf"),
                    relative_residual=float("inf"),
                    design_rank=0,
                    n_constraints=0,
                    underdetermined=True,
                )
            )
            continue

        neighbor_index = torch.tensor(node_neighbors, dtype=torch.long)
        displacements = coords[neighbor_index] - coords[node]
        targets = distance_matrix[node, neighbor_index] ** 2
        design = _symmetric_design(displacements)
        normal = design.T @ design
        regularized_normal = normal + regularization * torch.eye(
            normal.shape[0], dtype=torch.float64
        )
        rhs = design.T @ targets + regularization * identity_parameters
        parameters = torch.linalg.solve(regularized_normal, rhs)
        raw_metric = _parameters_to_metric(parameters, dimension)
        eigenvalues, eigenvectors = torch.linalg.eigh(raw_metric)
        eigenvalues = eigenvalues.clamp(min=eigenvalue_floor)
        metric = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T

        predicted = torch.einsum("ni,ij,nj->n", displacements, metric, displacements)
        target_norm = torch.linalg.vector_norm(targets)
        residual = torch.linalg.vector_norm(predicted - targets)
        relative_residual = (
            float((residual / target_norm).item())
            if target_norm > 0
            else float(residual.item())
        )
        condition_number = float(torch.linalg.cond(regularized_normal).item())
        design_rank = int(torch.linalg.matrix_rank(design, rtol=1e-8).item())
        underdetermined = design_rank < design.shape[1]
        fits.append(
            LocalMetricFit(
                metric=metric,
                condition_number=condition_number,
                relative_residual=relative_residual,
                design_rank=design_rank,
                n_constraints=len(node_neighbors),
                underdetermined=underdetermined,
            )
        )
    return fits
