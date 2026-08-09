import numpy as np
import pytest

from popgp.diagnostics import fit_quadratic_asymptote


def test_quadratic_asymptote_recovers_nonzero_limit() -> None:
    amplitudes = np.logspace(-5, -2, 12)
    responses = 2.5 * amplitudes**2 + 0.75 * amplitudes**3

    fit = fit_quadratic_asymptote(amplitudes, responses)

    assert fit.coefficient == pytest.approx(2.5, abs=1e-12)
    assert fit.linear_correction == pytest.approx(0.75, abs=1e-10)
    assert fit.coefficient_standard_error < 1e-12


def test_quadratic_asymptote_rejects_nonpositive_response() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        fit_quadratic_asymptote(
            np.asarray([1e-3, 2e-3, 3e-3]),
            np.asarray([1e-6, 0.0, 9e-6]),
        )
