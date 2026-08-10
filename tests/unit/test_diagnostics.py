import json
from pathlib import Path

import numpy as np
import pytest

from popgp.diagnostics import (
    assess_quadratic_response,
    fit_quadratic_asymptote,
    richardson_first_order_limit,
)

ROOT = Path(__file__).resolve().parents[2]


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


def test_quadratic_assessment_rejects_unsorted_amplitudes() -> None:
    amplitudes = np.logspace(-5, -3, 9)
    responses = 2.5 * amplitudes**2
    permutation = np.asarray([0, 3, 6, 1, 4, 7, 2, 5, 8])

    with pytest.raises(ValueError, match="strictly increasing"):
        assess_quadratic_response(amplitudes[permutation], responses[permutation])


@pytest.mark.negative_control
def test_quadratic_asymptote_enforces_absolute_precision_floor() -> None:
    amplitudes = np.logspace(-5, -3, 9)
    responses = 2.5 * amplitudes**2

    with pytest.raises(ValueError, match="absolute precision floor"):
        fit_quadratic_asymptote(
            amplitudes,
            responses,
            absolute_precision_floor=responses[0] / 500.0,
        )


def test_response_space_weighting_improves_committed_kms_coefficient() -> None:
    artifact = json.loads(
        (
            ROOT
            / "examples"
            / "physics_qg"
            / "source_law_many_body"
            / "results"
            / "validation.json"
        ).read_text(encoding="utf-8")
    )
    amplitudes = np.asarray(artifact["config"]["epsilons"], dtype=float)
    responses = np.asarray(artifact["measurements"]["relative_entropy"], dtype=float)
    check = next(
        item
        for item in artifact["checks"]
        if item["name"] == "nonaffine_kms_response_orders"
    )
    exact = check["value"]["exact_kubo_mori_quadratic_coefficient"]

    weighted = fit_quadratic_asymptote(amplitudes, responses)
    scaled = responses / amplitudes**2
    unweighted_design = np.column_stack([np.ones_like(amplitudes), amplitudes])
    unweighted_coefficient = np.linalg.lstsq(
        unweighted_design, scaled, rcond=None
    )[0][0]

    weighted_error = abs(weighted.coefficient - exact) / abs(exact)
    unweighted_error = abs(unweighted_coefficient - exact) / abs(exact)
    assert weighted_error < unweighted_error / 10.0


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
