import numpy as np
import pytest

from popgp.diagnostics import (
    assess_quadratic_response,
    fit_quadratic_asymptote,
    richardson_first_order_limit,
)


def test_quadratic_asymptote_recovers_nonzero_limit() -> None:
    amplitudes = np.logspace(-5, -2, 12)
    responses = 2.5 * amplitudes**2 + 0.75 * amplitudes**3

    fit = fit_quadratic_asymptote(amplitudes, responses)

    assert fit.coefficient == pytest.approx(2.5, abs=1e-12)
    assert fit.linear_correction == pytest.approx(0.75, abs=1e-10)
    assert fit.coefficient_residual_scale < 1e-12


def test_quadratic_asymptote_rejects_nonpositive_response() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        fit_quadratic_asymptote(
            np.asarray([1e-3, 2e-3, 3e-3]),
            np.asarray([1e-6, 0.0, 9e-6]),
        )


def test_quadratic_assessment_rejects_synthetic_first_order_response() -> None:
    amplitudes = np.logspace(-5, -3, 9)
    responses = 0.4 * amplitudes + 2.5 * amplitudes**2

    assessment = assess_quadratic_response(amplitudes, responses)

    assert assessment.passed is False
    assert assessment.slope_deviation > 0.9


def test_quadratic_asymptote_enforces_absolute_precision_floor() -> None:
    amplitudes = np.logspace(-5, -3, 9)
    responses = 2.5 * amplitudes**2

    with pytest.raises(ValueError, match="absolute precision floor"):
        fit_quadratic_asymptote(
            amplitudes,
            responses,
            absolute_precision_floor=responses[0] / 500.0,
        )


def test_richardson_limit_accepts_linear_and_rejects_quadratic_response() -> None:
    amplitudes = np.logspace(-5, -3, 9)
    linear = 0.7 * amplitudes + 0.2 * amplitudes**2
    quadratic = 0.7 * amplitudes**2

    linear_limit = richardson_first_order_limit(amplitudes, linear)
    quadratic_limit = richardson_first_order_limit(amplitudes, quadratic)

    assert linear_limit.estimate == pytest.approx(0.7, abs=1e-12)
    assert linear_limit.passed is True
    assert quadratic_limit.estimate == pytest.approx(0.0, abs=1e-15)
    assert quadratic_limit.passed is False
