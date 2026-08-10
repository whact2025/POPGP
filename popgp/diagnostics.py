"""Scientific diagnostics that keep validation separate from inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerLawFit:
    """Descriptive log-log power-law fit with a residual slope scale."""

    slope: float
    intercept: float
    slope_residual_scale: float
    r_squared: float


@dataclass(frozen=True)
class QuadraticAsymptoteFit:
    """Descriptive fit of ``response / amplitude**2`` to a finite intercept."""

    coefficient: float
    coefficient_residual_scale: float
    linear_correction: float
    normalized_rmse: float
    absolute_precision_floor: float
    minimum_signal_to_floor: float


@dataclass(frozen=True)
class QuadraticResponseAssessment:
    """Direct, falsifiable assessment of second-order asymptotic behavior."""

    full_window: QuadraticAsymptoteFit
    lower_window: QuadraticAsymptoteFit
    power_law: PowerLawFit
    relative_coefficient_difference: float
    slope_deviation: float
    passed: bool


@dataclass(frozen=True)
class RichardsonLimit:
    """First-order limit with truncation and floating-point error estimates."""

    estimate: float
    truncation_error: float
    roundoff_error: float
    total_error: float
    significance_ratio: float
    minimum_signal_to_floor: float
    passed: bool


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
    residual_scale = float(
        np.sqrt((residual_sum / degrees_of_freedom) / centered_sum)
    )
    r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    return PowerLawFit(
        slope=float(slope),
        intercept=float(intercept),
        slope_residual_scale=residual_scale,
        r_squared=r_squared,
    )


def fit_quadratic_asymptote(
    amplitudes: np.ndarray,
    responses: np.ndarray,
    *,
    absolute_precision_floor: float = 0.0,
    minimum_signal_to_floor: float = 1000.0,
) -> QuadraticAsymptoteFit:
    """Fit ``response = c0*amplitude**2 + c1*amplitude**3``.

    The response-space least-squares fit corresponds to inverse-variance weighting
    after division by ``amplitude**2`` when the absolute numerical response floor is
    approximately constant. The coefficient residual scale is descriptive, not a
    sampling estimate. A quadratic claim also needs nested-window agreement, an
    absolute log-log slope band, residual control, and a first-order negative control;
    use :func:`assess_quadratic_response` for that combined gate.
    """
    x = np.asarray(amplitudes, dtype=float)
    y = np.asarray(responses, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("amplitudes and responses must be equal 1D arrays of length >= 3")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("quadratic-asymptote inputs must be finite")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("quadratic-asymptote inputs must be strictly positive")
    if absolute_precision_floor < 0.0:
        raise ValueError("absolute_precision_floor must be nonnegative")
    if minimum_signal_to_floor <= 0.0:
        raise ValueError("minimum_signal_to_floor must be positive")
    signal_to_floor = (
        float(np.min(y) / absolute_precision_floor)
        if absolute_precision_floor > 0.0
        else float("inf")
    )
    if signal_to_floor < minimum_signal_to_floor:
        raise ValueError(
            "smallest response does not clear the absolute precision floor by "
            f"the required factor {minimum_signal_to_floor:g}"
        )

    design = np.column_stack([x**2, x**3])
    coefficient, linear_correction = np.linalg.lstsq(design, y, rcond=None)[0]
    response_residual = y - design @ np.asarray([coefficient, linear_correction])
    degrees_of_freedom = x.size - 2
    residual_variance = float(np.sum(response_residual**2) / degrees_of_freedom)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    coefficient_residual_scale = float(np.sqrt(max(covariance[0, 0], 0.0)))
    scaled_residual = response_residual / x**2
    scale = max(abs(float(coefficient)), np.finfo(float).tiny)
    normalized_rmse = float(np.sqrt(np.mean(scaled_residual**2)) / scale)
    return QuadraticAsymptoteFit(
        coefficient=float(coefficient),
        coefficient_residual_scale=coefficient_residual_scale,
        linear_correction=float(linear_correction),
        normalized_rmse=normalized_rmse,
        absolute_precision_floor=float(absolute_precision_floor),
        minimum_signal_to_floor=signal_to_floor,
    )


def assess_quadratic_response(
    amplitudes: np.ndarray,
    responses: np.ndarray,
    *,
    absolute_precision_floor: float = 0.0,
    minimum_signal_to_floor: float = 1000.0,
    lower_window_size: int = 6,
    maximum_relative_coefficient_difference: float = 1e-3,
    maximum_slope_deviation: float = 0.02,
    maximum_normalized_rmse: float = 1e-2,
) -> QuadraticResponseAssessment:
    """Assess quadratic order with independent coefficient, slope, and fit gates."""
    x = np.asarray(amplitudes, dtype=float)
    y = np.asarray(responses, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 4:
        raise ValueError("amplitudes and responses must be equal 1D arrays of length >= 4")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("amplitudes must be strictly increasing")
    if not 3 <= lower_window_size < x.size:
        raise ValueError("lower_window_size must be between 3 and len(amplitudes)-1")
    full = fit_quadratic_asymptote(
        x,
        y,
        absolute_precision_floor=absolute_precision_floor,
        minimum_signal_to_floor=minimum_signal_to_floor,
    )
    lower = fit_quadratic_asymptote(
        x[:lower_window_size],
        y[:lower_window_size],
        absolute_precision_floor=absolute_precision_floor,
        minimum_signal_to_floor=minimum_signal_to_floor,
    )
    power = fit_power_law(x, y)
    relative_difference = abs(full.coefficient - lower.coefficient) / max(
        abs(full.coefficient), np.finfo(float).tiny
    )
    slope_deviation = abs(power.slope - 2.0)
    passed = (
        full.coefficient > 0.0
        and relative_difference <= maximum_relative_coefficient_difference
        and slope_deviation <= maximum_slope_deviation
        and max(full.normalized_rmse, lower.normalized_rmse)
        <= maximum_normalized_rmse
    )
    return QuadraticResponseAssessment(
        full_window=full,
        lower_window=lower,
        power_law=power,
        relative_coefficient_difference=float(relative_difference),
        slope_deviation=float(slope_deviation),
        passed=passed,
    )


def richardson_first_order_limit(
    amplitudes: np.ndarray,
    signed_responses: np.ndarray,
    *,
    absolute_precision_floor: float = 0.0,
    required_error_margin: float = 10.0,
) -> RichardsonLimit:
    """Estimate ``lim(response/amplitude)`` from the three smallest amplitudes."""
    x = np.asarray(amplitudes, dtype=float)
    y = np.asarray(signed_responses, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("amplitudes and responses must be equal 1D arrays of length >= 3")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Richardson inputs must be finite")
    if np.any(x <= 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("amplitudes must be positive and strictly increasing")
    if absolute_precision_floor < 0.0:
        raise ValueError("absolute_precision_floor must be nonnegative")
    if required_error_margin <= 0.0:
        raise ValueError("required_error_margin must be positive")

    quotient = y[:3] / x[:3]
    first = (x[1] * quotient[0] - x[0] * quotient[1]) / (x[1] - x[0])
    second = (x[2] * quotient[1] - x[1] * quotient[2]) / (x[2] - x[1])
    truncation_error = abs(first - second)
    roundoff_error = (
        absolute_precision_floor
        / (x[1] - x[0])
        * (x[1] / x[0] + x[0] / x[1])
    )
    total_error = truncation_error + roundoff_error
    significance_ratio = abs(first) / max(total_error, np.finfo(float).tiny)
    signal_to_floor = (
        float(np.min(np.abs(y)) / absolute_precision_floor)
        if absolute_precision_floor > 0.0
        else float("inf")
    )
    return RichardsonLimit(
        estimate=float(first),
        truncation_error=float(truncation_error),
        roundoff_error=float(roundoff_error),
        total_error=float(total_error),
        significance_ratio=float(significance_ratio),
        minimum_signal_to_floor=signal_to_floor,
        passed=bool(abs(first) > required_error_margin * total_error),
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
