"""Neutral closure-residual interface for future discrete field equations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClosureMismatch:
    """Tensor mismatch without implying that either input is an Einstein tensor."""

    residual: torch.Tensor
    absolute_norm: float
    relative_norm: float


def closure_mismatch(
    geometric_side: torch.Tensor,
    source_side: torch.Tensor,
    *,
    coupling: float = 1.0,
    norm_floor: float = 1e-15,
) -> ClosureMismatch:
    """Measure ``geometric_side - coupling * source_side``.

    Callers must identify and validate the tensors they supply. This function is
    intentionally agnostic: it provides plumbing for convergence tests and is not
    evidence that an Einstein or Regge closure has been constructed.
    """
    if geometric_side.shape != source_side.shape:
        raise ValueError("closure tensors must have the same shape")
    if norm_floor <= 0.0:
        raise ValueError("norm_floor must be positive")
    geometric_side = geometric_side.to(dtype=torch.float64)
    source_side = source_side.to(dtype=torch.float64)
    residual = geometric_side - coupling * source_side
    absolute_norm = float(torch.linalg.vector_norm(residual).item())
    scale = max(
        float(torch.linalg.vector_norm(geometric_side).item()),
        abs(coupling) * float(torch.linalg.vector_norm(source_side).item()),
        norm_floor,
    )
    return ClosureMismatch(
        residual=residual,
        absolute_norm=absolute_norm,
        relative_norm=absolute_norm / scale,
    )
