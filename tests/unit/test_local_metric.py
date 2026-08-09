import pytest
import torch

from popgp.geometry import reconstruct_local_metrics


def test_local_metric_recovers_identity_on_euclidean_star() -> None:
    coords = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    distances = torch.cdist(coords, coords)
    edges = [(0, 1), (0, 2), (0, 3)]

    fit = reconstruct_local_metrics(
        coords,
        distances,
        edges,
        regularization=1e-8,
    )[0]

    assert torch.allclose(fit.metric, torch.eye(2, dtype=torch.float64), atol=1e-7)
    assert fit.relative_residual < 1e-7
    assert fit.design_rank == 3


def test_local_metric_rejects_zero_regularization() -> None:
    coords = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    distances = torch.cdist(coords, coords)

    with pytest.raises(ValueError, match="regularization must be positive"):
        reconstruct_local_metrics(
            coords,
            distances,
            [(0, 1)],
            regularization=0.0,
        )
