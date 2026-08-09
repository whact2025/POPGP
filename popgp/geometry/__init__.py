"""Finite embedding-geometry diagnostics."""

from popgp.geometry.closure import ClosureMismatch, closure_mismatch
from popgp.geometry.local_metric import LocalMetricFit, reconstruct_local_metrics
from popgp.geometry.regge import build_delaunay_proxy, vertex_deficits_2d

__all__ = [
    "ClosureMismatch",
    "LocalMetricFit",
    "build_delaunay_proxy",
    "closure_mismatch",
    "reconstruct_local_metrics",
    "vertex_deficits_2d",
]
