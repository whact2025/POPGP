"""Scientific diagnostics that keep validation separate from inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerLawFit:
    """Log-log power-law fit with elementary uncertainty diagnostics."""

    slope: float
    intercept: float
    slope_standard_error: float
    r_squared: float


@dataclass(frozen=True)
class QuadraticAsymptoteFit:
    """Fit of ``response / amplitude**2`` to a finite intercept."""

    coefficient: float
    coefficient_standard_error: float
    linear_correction: float
    normalized_rmse: float


@dataclass(frozen=True)
class EdgeRecoveryMetrics:
    """Blind inferred-edge comparison against held-out reference edges."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def fit_power_law(amplitudes: np.ndarray, responses: np.ndarray) -> PowerLawFit:
    """Fit ``response = exp(intercept) * amplitude**slope`` in log space."""
    x = np.asarray(amplitudes, dtype=float)
    y = np.asarray(responses, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("amplitudes and responses must be equal 1D arrays of length >= 3")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("power-law inputs must be finite")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law inputs must be strictly positive")

    log_x = np.log(x)
    log_y = np.log(y)
    design = np.column_stack([log_x, np.ones_like(log_x)])
    slope, intercept = np.linalg.lstsq(design, log_y, rcond=None)[0]
    fitted = slope * log_x + intercept
    residual = log_y - fitted
    residual_sum = float(np.sum(residual**2))
    total_sum = float(np.sum((log_y - np.mean(log_y)) ** 2))
    degrees_of_freedom = x.size - 2
    centered_sum = float(np.sum((log_x - np.mean(log_x)) ** 2))
    standard_error = float(
        np.sqrt((residual_sum / degrees_of_freedom) / centered_sum)
    )
    r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    return PowerLawFit(
        slope=float(slope),
        intercept=float(intercept),
        slope_standard_error=standard_error,
        r_squared=r_squared,
    )


def fit_quadratic_asymptote(
    amplitudes: np.ndarray,
    responses: np.ndarray,
) -> QuadraticAsymptoteFit:
    """Fit ``response / amplitude**2 = c0 + c1 * amplitude``.

    A finite positive ``c0`` is the asymptotic statement that the response is
    quadratic.  Comparing ``c0`` across nested windows tests convergence
    without imposing an arbitrary band on a log-log slope.
    """
    x = np.asarray(amplitudes, dtype=float)
    y = np.asarray(responses, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("amplitudes and responses must be equal 1D arrays of length >= 3")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("quadratic-asymptote inputs must be finite")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("quadratic-asymptote inputs must be strictly positive")

    scaled = y / x**2
    design = np.column_stack([np.ones_like(x), x])
    coefficient, linear_correction = np.linalg.lstsq(design, scaled, rcond=None)[0]
    residual = scaled - design @ np.asarray([coefficient, linear_correction])
    degrees_of_freedom = x.size - 2
    residual_variance = float(np.sum(residual**2) / degrees_of_freedom)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    standard_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    scale = max(abs(float(coefficient)), np.finfo(float).tiny)
    normalized_rmse = float(np.sqrt(np.mean(residual**2)) / scale)
    return QuadraticAsymptoteFit(
        coefficient=float(coefficient),
        coefficient_standard_error=standard_error,
        linear_correction=float(linear_correction),
        normalized_rmse=normalized_rmse,
    )


def edge_recovery_metrics(
    inferred: set[tuple[int, int]], reference: set[tuple[int, int]]
) -> EdgeRecoveryMetrics:
    """Compare blind inference with a held-out edge set."""
    inferred = {tuple(sorted(edge)) for edge in inferred}
    reference = {tuple(sorted(edge)) for edge in reference}
    true_positives = len(inferred & reference)
    false_positives = len(inferred - reference)
    false_negatives = len(reference - inferred)
    precision = true_positives / len(inferred) if inferred else 0.0
    recall = true_positives / len(reference) if reference else 0.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return EdgeRecoveryMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )
