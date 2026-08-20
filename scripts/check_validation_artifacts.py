"""Check regenerated validation artifacts against their committed contract.

The examples contain floating-point diagnostics produced by LAPACK-backed routines.
Those values can move slightly across platforms even when the scientific result is
unchanged.  This checker therefore compares JSON structure and semantics rather than
serialized bytes:

* keys, JSON types, list lengths, strings, integers, booleans, and nulls are exact;
* configuration floats allow only a few machine epsilons of serialization drift;
* diagnostic floats use a narrow default tolerance;
* a small named set of ill-conditioned fit diagnostics has an explicit wider policy;
* every numeric value must remain finite;
* every decision summary and Boolean is recomputed from lowest-level raw operands; and
* every declared visual artifact must satisfy global and locality-aware pixel bounds.

The change-boundary mode also compares the actual working tree to a fresh index loaded
from the frozen Git tree. It does not trust mutable index flags or omit ignored state.

The wider policies below correspond to fields that changed materially in the frozen
Linux CI diff for PR #2 while all associated scientific gates remained unchanged.
Keeping the list here makes that exception reviewable instead of silently weakening
the entire artifact comparison.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from popgp.diagnostics import (
    assess_quadratic_response,
    fit_power_law,
    richardson_first_order_limit,
)
from scripts.check_reproduction_boundary import check_repository_boundary

CONFIG_REL_TOL = 8 * sys.float_info.epsilon
DEFAULT_DIAGNOSTIC_REL_TOL = 1e-3
DEFAULT_DIAGNOSTIC_ABS_TOL = 5e-9

# (relative tolerance, absolute tolerance).  These are nuisance diagnostics from
# small-signal regressions/Richardson error estimates.  Their gate booleans and the
# check identity/criterion remain exact elsewhere in the document.
SENSITIVE_DIAGNOSTIC_TOLERANCES: dict[str, tuple[float, float]] = {
    "coefficient_residual_scale": (0.0, 5e-6),
    "linear_correction": (0.0, 2e-2),
    "normalized_rmse": (0.0, 2e-4),
    "quadratic_coefficient_relative_error": (0.0, 2e-4),
    "relative_coefficient_difference": (0.0, 2e-4),
    # This is a quotient whose denominator is an O(1e-10) cancellation/error
    # estimate.  Its precise large value is noise-sensitive; the exact `passed`
    # gate remains protected separately.  rel_tol=0.99 bounds accepted movement
    # to at most a factor of 100 in either direction.
    "significance_ratio": (0.99, 0.0),
    "slope_deviation": (0.0, 2e-4),
    "slope_residual_scale": (0.0, 1e-4),
}

STABLE_INPUT_KEYS = frozenset({"beta", "epsilons"})
VISUAL_SUFFIXES = frozenset({".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"})
VALIDATION_GLOB = "examples/physics_qg/*/results/validation.json"
# Retained exact-candidate Windows/Ubuntu calibration found a maximum honest
# per-channel raster delta of three after the near-zero chain legend is
# canonicalized.  A one-count guard band preserves that measured platform noise
# while making every rendered feature part of the contract; aggregate error budgets
# otherwise permit small labels and one-pixel curves to disappear completely.
VISUAL_MAXIMUM_CHANNEL_ERROR_LIMIT = 4
# Retained Windows/Ubuntu LAPACK fits differ by at most roughly 3e-15 in the
# recomputed sensitivity slope deviations.  These narrow equality tolerances admit
# that measured roundoff while remaining orders of magnitude below every registered
# decision margin and the 4e-9 adversarial raw-operand mutations.
RECOMPUTED_REL_TOL = 1e-9
RECOMPUTED_ABS_TOL = 2e-15
# Hosted Linux recomputation of the committed chain index moment differs from the
# Windows-generated value by 5.49e-19 because NumPy reduces the dot product in a
# different order.  The smallest registered sign/permutation attack changes the
# normalized moment by 4.46e-18, so this remains a measured fail-closed separation.
POTENTIAL_RECOMPUTED_ABS_TOL = 1e-18
INFORMATIONAL_CHECK_ALLOWLIST: dict[str, frozenset[str]] = {
    "examples/physics_qg/ca_model/results/validation.json": frozenset(
        {"survivor_entropy_filter_regression"}
    ),
    "examples/physics_qg/chain_1d/results/validation.json": frozenset(
        {"blind_edge_recovery"}
    ),
}
NONINFORMATIONAL_CHECK_ALLOWLIST: dict[str, frozenset[str]] = {
    "examples/physics_qg/ca_model/results/validation.json": frozenset(
        {"population_survival", "population_growth"}
    ),
    "examples/physics_qg/chain_1d/results/validation.json": frozenset(
        {
            "stability_selection",
            "contiguous_cells",
            "su2_equivariance_identity_regression",
            "geometry_1d_ordering",
            "dimension_selection",
            "placeholder_clock_constraint_solved",
        }
    ),
    "examples/physics_qg/gravity_well/results/validation.json": frozenset(
        {
            "pi_res_admissibility",
            "nonzero_source_constraint_residual",
            "monotonic_falloff",
            "grid_symmetry",
            "negative_well_at_source",
            "redshift_positive",
            "dimension_selection_2d",
        }
    ),
    "examples/physics_qg/grid_2d/results/validation.json": frozenset(
        {
            "pi_res_admissibility",
            "blind_edge_recovery",
            "dimension_selection",
            "topology_preservation",
            "finite_graph_spectral_peak",
            "mds_stress",
            "placeholder_source_degeneracy",
        }
    ),
    "examples/physics_qg/source_law/results/validation.json": frozenset(
        {
            "relative_entropy_is_quadratic",
            "affine_modular_linearity_identity_regression",
            "linear_solver_homogeneity_identity_regression",
            "equal_energy_entropy_confound",
        }
    ),
    "examples/physics_qg/source_law_many_body/results/validation.json": frozenset(
        {
            "nonaffine_kms_response_orders",
            "quadratic_gate_rejects_first_order_negative_control",
            "kms_and_local_decomposition_identities",
            "local_energy_decomposition_consistency_and_spreading",
            "nonaffine_kms_parameter_sensitivity",
            "isospectral_unitary_identity_regression",
            "spreading_requires_noncommuting_dynamics",
            "pipeline_reduced_modular_blindness_and_density_repair",
            "negative_energy_candidate_has_slower_source_clock",
        }
    ),
}


@dataclass
class ComparisonSummary:
    """Result of comparing one regenerated document to its committed reference."""

    errors: list[str] = field(default_factory=list)
    accepted_numeric_drifts: int = 0
    largest_absolute_drift: float = 0.0
    largest_relative_drift: float = 0.0

    @property
    def passed(self) -> bool:
        return not self.errors


def _json_path(parts: tuple[str | int, ...]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path


def _is_stable_input_path(parts: tuple[str | int, ...]) -> bool:
    return bool(parts) and (
        parts[0] == "config"
        or any(part in STABLE_INPUT_KEYS for part in parts if isinstance(part, str))
    )


def compare_validation_documents(reference: Any, candidate: Any) -> ComparisonSummary:
    """Compare two decoded JSON documents using the validation contract."""

    summary = ComparisonSummary()
    _compare_value(reference, candidate, (), summary)
    return summary


def _compare_value(
    reference: Any,
    candidate: Any,
    path: tuple[str | int, ...],
    summary: ComparisonSummary,
) -> None:
    location = _json_path(path)
    if type(reference) is not type(candidate):
        summary.errors.append(
            f"{location}: JSON type changed from {type(reference).__name__} "
            f"to {type(candidate).__name__}"
        )
        return

    if isinstance(reference, dict):
        reference_keys = set(reference)
        candidate_keys = set(candidate)
        if reference_keys != candidate_keys:
            missing = sorted(reference_keys - candidate_keys)
            added = sorted(candidate_keys - reference_keys)
            summary.errors.append(f"{location}: key set changed (missing={missing}, added={added})")
        for key in sorted(reference_keys & candidate_keys):
            _compare_value(reference[key], candidate[key], (*path, key), summary)
        return

    if isinstance(reference, list):
        if len(reference) != len(candidate):
            summary.errors.append(
                f"{location}: array length changed from {len(reference)} to {len(candidate)}"
            )
            return
        for index, (reference_item, candidate_item) in enumerate(zip(reference, candidate)):
            _compare_value(reference_item, candidate_item, (*path, index), summary)
        return

    if isinstance(reference, float):
        _compare_float(reference, candidate, path, summary)
        return

    # bool must be checked as an exact scalar before int because bool subclasses int.
    if isinstance(reference, (bool, int, str)) or reference is None:
        if reference != candidate:
            summary.errors.append(f"{location}: changed from {reference!r} to {candidate!r}")
        return

    summary.errors.append(f"{location}: unsupported decoded JSON type {type(reference).__name__}")


def _compare_float(
    reference: float,
    candidate: float,
    path: tuple[str | int, ...],
    summary: ComparisonSummary,
) -> None:
    location = _json_path(path)
    if not math.isfinite(reference) or not math.isfinite(candidate):
        summary.errors.append(
            f"{location}: non-finite number is forbidden "
            f"(reference={reference!r}, candidate={candidate!r})"
        )
        return

    if reference == candidate:
        return

    key = next((part for part in reversed(path) if isinstance(part, str)), "")
    if _is_stable_input_path(path):
        rel_tol, abs_tol = CONFIG_REL_TOL, 0.0
        policy = "stable input"
    elif key in SENSITIVE_DIAGNOSTIC_TOLERANCES:
        rel_tol, abs_tol = SENSITIVE_DIAGNOSTIC_TOLERANCES[key]
        policy = f"sensitive diagnostic {key!r}"
    else:
        rel_tol, abs_tol = DEFAULT_DIAGNOSTIC_REL_TOL, DEFAULT_DIAGNOSTIC_ABS_TOL
        policy = "diagnostic"

    absolute_drift = abs(candidate - reference)
    scale = max(abs(reference), abs(candidate))
    relative_drift = absolute_drift / scale if scale else 0.0
    if not math.isclose(reference, candidate, rel_tol=rel_tol, abs_tol=abs_tol):
        summary.errors.append(
            f"{location}: {policy} drift exceeds rel_tol={rel_tol:g}, abs_tol={abs_tol:g} "
            f"({reference!r} -> {candidate!r}, abs={absolute_drift:.6g}, "
            f"rel={relative_drift:.6g})"
        )
        return


    summary.accepted_numeric_drifts += 1
    summary.largest_absolute_drift = max(summary.largest_absolute_drift, absolute_drift)
    summary.largest_relative_drift = max(summary.largest_relative_drift, relative_drift)


def load_json_document(raw: str, *, source: str) -> Any:
    """Decode strict JSON, rejecting NaN and Infinity extensions."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        return json.loads(raw, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid strict JSON: {exc}") from exc


def _bind_exact(
    errors: list[str],
    observed: Any,
    authoritative: Any,
    *,
    location: str,
) -> None:
    """Require a serialized decision operand to equal its authoritative source."""

    if type(observed) is not type(authoritative) or observed != authoritative:
        errors.append(
            f"{location} differs from its authoritative retained field "
            f"(observed={observed!r}, authoritative={authoritative!r})"
        )


def _finite_array(value: Any, *, ndim: int, location: str) -> np.ndarray:
    """Decode a nonempty, finite numeric array without accepting booleans/strings."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} is not a rectangular numeric array") from exc
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{location} must contain only JSON numbers")
    if raw.ndim != ndim or any(size == 0 for size in raw.shape):
        raise ValueError(f"{location} must be a nonempty {ndim}-dimensional array")
    array = raw.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{location} must contain only finite values")
    return array


def _bind_recomputed(
    errors: list[str],
    observed: Any,
    expected: Any,
    *,
    location: str,
) -> None:
    """Bind a serialized derived value to an independently recomputed value."""

    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            errors.append(f"{location} must be an object recomputed from raw operands")
            return
        if set(observed) != set(expected):
            errors.append(
                f"{location} keys differ from recomputed keys "
                f"(observed={sorted(observed)}, expected={sorted(expected)})"
            )
            return
        for key in expected:
            _bind_recomputed(
                errors,
                observed[key],
                expected[key],
                location=f"{location}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            errors.append(f"{location} shape differs from recomputed raw operands")
            return
        for index, item in enumerate(expected):
            _bind_recomputed(
                errors,
                observed[index],
                item,
                location=f"{location}[{index}]",
            )
        return
    if isinstance(expected, bool):
        if type(observed) is not bool or observed is not expected:
            errors.append(
                f"{location} differs from recomputed raw operands "
                f"(observed={observed!r}, expected={expected!r})"
            )
        return
    if isinstance(expected, float):
        if type(observed) is not float or not math.isfinite(observed):
            errors.append(f"{location} must be a finite JSON float")
            return
        if not math.isclose(
            observed,
            expected,
            rel_tol=RECOMPUTED_REL_TOL,
            abs_tol=RECOMPUTED_ABS_TOL,
        ):
            errors.append(
                f"{location} differs from recomputed raw operands "
                f"(observed={observed!r}, expected={expected!r})"
            )
        return
    if type(observed) is not type(expected) or observed != expected:
        errors.append(
            f"{location} differs from recomputed raw operands "
            f"(observed={observed!r}, expected={expected!r})"
        )


def _bind_potential_summaries(
    errors: list[str],
    value: Any,
    *,
    location: str,
    include_mean_and_range: bool,
    include_max_absolute: bool = False,
) -> np.ndarray:
    """Bind a retained potential array to every serialized scalar summary."""

    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    phi = _finite_array(value["phi"], ndim=1, location=f"{location}.phi")
    expected = {
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
    }
    if include_mean_and_range:
        expected.update(
            {
                "phi_range": float(np.max(phi) - np.min(phi)),
                "phi_mean": float(np.mean(phi)),
            }
        )
    if include_max_absolute:
        expected["max_absolute_phi"] = float(np.max(np.abs(phi)))
    index_weights = np.arange(1, len(phi) + 1, dtype=float)
    expected["phi_index_moment"] = float(
        np.dot(index_weights / np.sum(index_weights), phi)
    )
    for key, recomputed in expected.items():
        observed = value[key]
        if type(observed) is not float or not math.isfinite(observed):
            errors.append(f"{location}.{key} must be a finite JSON float")
        elif not math.isclose(
            observed,
            recomputed,
            rel_tol=1e-12,
            abs_tol=POTENTIAL_RECOMPUTED_ABS_TOL,
        ):
            errors.append(
                f"{location}.{key} differs from recomputed raw potential "
                f"(observed={observed!r}, expected={recomputed!r})"
            )
    return phi


def _bind_clock_solver(
    errors: list[str],
    value: dict[str, Any],
    phi: np.ndarray,
    *,
    location: str,
) -> float:
    """Recompute the retained finite-graph clock equation from raw operands."""

    weights = _finite_array(
        value["weight_matrix"], ndim=2, location=f"{location}.weight_matrix"
    )
    source = _finite_array(
        value["effective_source"], ndim=1, location=f"{location}.effective_source"
    )
    if weights.shape != (len(phi), len(phi)) or source.shape != phi.shape:
        raise ValueError(f"{location} clock-solver operand shapes differ")
    mu = value["mu"]
    if type(mu) is not float or not math.isfinite(mu) or mu < 0.0:
        raise ValueError(f"{location}.mu must be a nonnegative finite JSON float")
    normalize_potential = value["normalize_potential"]
    if type(normalize_potential) is not bool:
        raise ValueError(f"{location}.normalize_potential must be a boolean")

    laplacian = np.diag(np.sum(weights, axis=1)) - weights
    operator = laplacian + mu**2 * np.eye(len(phi))
    expected_residual = float(np.linalg.norm(operator @ phi - source))
    _bind_recomputed(
        errors,
        value["constraint_residual"],
        expected_residual,
        location=f"{location}.constraint_residual",
    )
    if normalize_potential and abs(float(np.mean(phi))) >= 1e-12:
        errors.append(
            f"{location}.phi violates the normalized zero-mean potential gauge"
        )
    return expected_residual


def _retained_potential_errors(document: dict[str, Any]) -> list[str]:
    """Recompute potential summaries and gravity derivatives from raw arrays."""

    example = document.get("example")
    errors: list[str] = []
    try:
        if example in {"chain_1d", "grid_2d"}:
            pi_time = document["pipeline"]["pi_time"]
            phi = _bind_potential_summaries(
                errors,
                pi_time,
                location="pipeline.pi_time",
                include_mean_and_range=True,
                include_max_absolute=example == "grid_2d",
            )
            _bind_clock_solver(errors, pi_time, phi, location="pipeline.pi_time")
            if example == "grid_2d":
                source = _finite_array(
                    pi_time["effective_source"],
                    ndim=1,
                    location="pipeline.pi_time.effective_source",
                )
                _bind_recomputed(
                    errors,
                    pi_time["effective_source_norm"],
                    float(np.linalg.norm(source)),
                    location="pipeline.pi_time.effective_source_norm",
                )
            return errors

        if example != "gravity_well":
            return errors

        pipeline = document["pipeline"]
        natural = pipeline["pi_time_natural"]
        natural_phi = _bind_potential_summaries(
            errors,
            natural,
            location="pipeline.pi_time_natural",
            include_mean_and_range=False,
        )
        _bind_clock_solver(
            errors,
            natural,
            natural_phi,
            location="pipeline.pi_time_natural",
        )

        gravity = pipeline["gravity_test"]
        if not isinstance(gravity, dict):
            raise ValueError("pipeline.gravity_test must be an object")
        phi = _finite_array(
            gravity["phi_point"],
            ndim=1,
            location="pipeline.gravity_test.phi_point",
        )
        graph_distances_raw = gravity["graph_distances"]
        if not isinstance(graph_distances_raw, list) or any(
            type(item) is not int or item < 0 for item in graph_distances_raw
        ):
            raise ValueError(
                "pipeline.gravity_test.graph_distances must contain nonnegative integers"
            )
        graph_distances = np.asarray(graph_distances_raw, dtype=int)
        if graph_distances.shape != phi.shape:
            raise ValueError(
                "pipeline.gravity_test graph_distances and phi_point shapes differ"
            )
        center = document["config"]["center_cell"]
        if type(center) is not int or not 0 <= center < len(phi):
            raise ValueError("config.center_cell is not a valid phi_point index")

        for key, recomputed in {
            "phi_min": float(np.min(phi)),
            "phi_max": float(np.max(phi)),
            "phi_at_source": float(phi[center]),
            "phi_index_moment": float(
                np.dot(
                    np.arange(1, len(phi) + 1, dtype=float)
                    / np.sum(np.arange(1, len(phi) + 1, dtype=float)),
                    phi,
                )
            ),
        }.items():
            observed = gravity[key]
            if type(observed) is not float or not math.isfinite(observed):
                errors.append(f"pipeline.gravity_test.{key} must be a finite JSON float")
            elif not math.isclose(
                observed,
                recomputed,
                rel_tol=1e-12,
                abs_tol=POTENTIAL_RECOMPUTED_ABS_TOL,
            ):
                errors.append(
                    f"pipeline.gravity_test.{key} differs from recomputed raw potential "
                    f"(observed={observed!r}, expected={recomputed!r})"
                )

        unique_distances = np.unique(graph_distances)
        radial_means = np.asarray(
            [float(np.mean(phi[graph_distances == distance])) for distance in unique_distances]
        )
        radial_stds = np.asarray(
            [float(np.std(phi[graph_distances == distance])) for distance in unique_distances]
        )
        expected_radial = {
            "distances": [float(distance) for distance in unique_distances],
            "phi_avg": [float(item) for item in radial_means],
            "phi_std": [float(item) for item in radial_stds],
        }
        _bind_recomputed(
            errors,
            gravity["radial_profile"],
            expected_radial,
            location="pipeline.gravity_test.radial_profile",
        )

        fit_mask = unique_distances > 0
        if int(np.sum(fit_mask)) >= 2:
            log_distance = np.log(unique_distances[fit_mask].astype(float))
            fit_values = radial_means[fit_mask]
            design = np.column_stack([log_distance, np.ones_like(log_distance)])
            coefficients, *_ = np.linalg.lstsq(design, fit_values, rcond=None)
            prediction = design @ coefficients
            residual_sum = float(np.sum((fit_values - prediction) ** 2))
            total_sum = float(np.sum((fit_values - np.mean(fit_values)) ** 2))
            expected_log_fit = {
                "slope": float(coefficients[0]),
                "intercept": float(coefficients[1]),
                "r_squared": 1.0 - residual_sum / total_sum if total_sum > 0.0 else 0.0,
            }
        else:
            expected_log_fit = {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}
        _bind_recomputed(
            errors,
            gravity["log_fit"],
            expected_log_fit,
            location="pipeline.gravity_test.log_fit",
        )

        phi_source = float(phi[center])
        phi_boundary = float(radial_means[-1])
        one_plus_z = float(np.exp(phi_boundary - phi_source))
        expected_redshift = {
            "phi_source": phi_source,
            "phi_boundary": phi_boundary,
            "z": one_plus_z - 1.0,
            "one_plus_z": one_plus_z,
        }
        _bind_recomputed(
            errors,
            gravity["redshift"],
            expected_redshift,
            location="pipeline.gravity_test.redshift",
        )
        expected_residual = _bind_clock_solver(
            errors,
            gravity,
            phi,
            location="pipeline.gravity_test",
        )
        source = _finite_array(
            gravity["effective_source"],
            ndim=1,
            location="pipeline.gravity_test.effective_source",
        )
        source_norm = float(np.linalg.norm(source))
        if source_norm == 0.0:
            raise ValueError("pipeline.gravity_test.effective_source has zero norm")
        _bind_recomputed(
            errors,
            gravity["relative_constraint_residual"],
            expected_residual / source_norm,
            location="pipeline.gravity_test.relative_constraint_residual",
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(f"retained potential operands are invalid: {exc}")
    return errors


def _fit_dict_from_raw(
    amplitudes: Any,
    responses: Any,
    *,
    location: str,
    absolute_response: bool = False,
) -> dict[str, float]:
    x = _finite_array(amplitudes, ndim=1, location=f"{location}.amplitudes")
    y = _finite_array(responses, ndim=1, location=f"{location}.responses")
    if np.any(x <= 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError(f"{location}.amplitudes must be positive and increasing")
    if absolute_response:
        y = np.abs(y)
    fit = fit_power_law(x, y)
    return {
        "slope": fit.slope,
        "intercept": fit.intercept,
        "slope_residual_scale": fit.slope_residual_scale,
        "r_squared": fit.r_squared,
    }


def _assessment_dict_from_raw(
    amplitudes: Any,
    responses: Any,
    *,
    location: str,
    absolute_precision_floor: Any,
    minimum_signal_to_floor: Any,
    lower_window_size: int = 6,
    maximum_relative_coefficient_difference: Any = 1e-3,
    maximum_slope_deviation: Any = 0.02,
    maximum_normalized_rmse: Any = 1e-2,
) -> dict[str, Any]:
    x = _finite_array(amplitudes, ndim=1, location=f"{location}.amplitudes")
    y = _finite_array(responses, ndim=1, location=f"{location}.responses")
    assessment = assess_quadratic_response(
        x,
        y,
        absolute_precision_floor=float(absolute_precision_floor),
        minimum_signal_to_floor=float(minimum_signal_to_floor),
        lower_window_size=lower_window_size,
        maximum_relative_coefficient_difference=float(
            maximum_relative_coefficient_difference
        ),
        maximum_slope_deviation=float(maximum_slope_deviation),
        maximum_normalized_rmse=float(maximum_normalized_rmse),
    )

    def asymptote_dict(item: Any) -> dict[str, float]:
        return {
            "coefficient": item.coefficient,
            "coefficient_residual_scale": item.coefficient_residual_scale,
            "linear_correction": item.linear_correction,
            "normalized_rmse": item.normalized_rmse,
            "absolute_precision_floor": item.absolute_precision_floor,
            "minimum_signal_to_floor": item.minimum_signal_to_floor,
        }

    return {
        "full_window": asymptote_dict(assessment.full_window),
        "lower_window": asymptote_dict(assessment.lower_window),
        "power_law": {
            "slope": assessment.power_law.slope,
            "intercept": assessment.power_law.intercept,
            "slope_residual_scale": assessment.power_law.slope_residual_scale,
            "r_squared": assessment.power_law.r_squared,
        },
        "relative_coefficient_difference": (
            assessment.relative_coefficient_difference
        ),
        "slope_deviation": assessment.slope_deviation,
        "passed": assessment.passed,
    }


def _quadratic_assessment_outcome(
    assessment: Any,
    *,
    location: str,
    amplitudes: Any,
    responses: Any,
    absolute_precision_floor: Any,
    minimum_signal_to_floor: Any = 1000.0,
    lower_window_size: int = 6,
    maximum_relative_coefficient_difference: Any = 1e-3,
    maximum_slope_deviation: Any = 0.02,
    maximum_normalized_rmse: Any = 1e-2,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    expected_assessment = _assessment_dict_from_raw(
        amplitudes,
        responses,
        location=location,
        absolute_precision_floor=absolute_precision_floor,
        minimum_signal_to_floor=minimum_signal_to_floor,
        lower_window_size=lower_window_size,
        maximum_relative_coefficient_difference=(
            maximum_relative_coefficient_difference
        ),
        maximum_slope_deviation=maximum_slope_deviation,
        maximum_normalized_rmse=maximum_normalized_rmse,
    )
    _bind_recomputed(errors, assessment, expected_assessment, location=location)
    expected = expected_assessment["passed"]
    return expected, errors


def _richardson_outcome(
    limit: Any,
    *,
    location: str,
    amplitudes: Any,
    responses: Any,
    absolute_precision_floor: Any,
    required_error_margin: Any = 10.0,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    x = _finite_array(amplitudes, ndim=1, location=f"{location}.amplitudes")
    y = _finite_array(responses, ndim=1, location=f"{location}.responses")
    expected_limit = richardson_first_order_limit(
        x,
        y,
        absolute_precision_floor=float(absolute_precision_floor),
        required_error_margin=float(required_error_margin),
    )
    expected = {
        "estimate": expected_limit.estimate,
        "truncation_error": expected_limit.truncation_error,
        "roundoff_error": expected_limit.roundoff_error,
        "total_error": expected_limit.total_error,
        "significance_ratio": expected_limit.significance_ratio,
        "minimum_signal_to_floor": expected_limit.minimum_signal_to_floor,
        "passed": expected_limit.passed,
    }
    _bind_recomputed(errors, limit, expected, location=location)
    return expected_limit.passed, errors


def _decision_outcome(
    document: dict[str, Any],
    check: dict[str, Any],
    relative_path: str,
) -> tuple[bool | None, list[str]]:
    """Recompute a registered scientific decision from retained typed operands."""

    example = document.get("example")
    name = check["name"]
    value = check.get("value")
    errors: list[str] = []
    try:
        if example == "ca_model":
            dynamics = document["pipeline"]["dynamics"]
            if name == "population_survival":
                _bind_exact(
                    errors,
                    value,
                    dynamics["final_population"],
                    location=f"check {name!r}.value",
                )
                return value > 0, errors
            if name == "population_growth":
                _bind_exact(
                    errors,
                    value["initial"],
                    dynamics["initial_population"],
                    location=f"check {name!r}.value.initial",
                )
                _bind_exact(
                    errors,
                    value["final"],
                    dynamics["final_population"],
                    location=f"check {name!r}.value.final",
                )
                return value["final"] >= value["initial"], errors
            if name == "survivor_entropy_filter_regression":
                _bind_exact(
                    errors,
                    value,
                    dynamics["avg_entropy_last_5_steps"],
                    location=f"check {name!r}.value",
                )
                return value < check["threshold"], errors

        if example == "chain_1d":
            pipeline = document["pipeline"]
            if name == "stability_selection":
                return value["invalid"] > value["valid"], errors
            if name == "contiguous_cells":
                _bind_exact(
                    errors,
                    value,
                    pipeline["pi_res"]["cells"],
                    location=f"check {name!r}.value",
                )
                return value == [[0, 1], [2, 3], [4, 5], [6, 7]], errors
            if name == "su2_equivariance_identity_regression":
                _bind_exact(
                    errors,
                    value,
                    pipeline["pi_res"]["su2_equivariant"],
                    location=f"check {name!r}.value",
                )
                return value is True, errors
            if name == "blind_edge_recovery":
                _bind_exact(
                    errors,
                    value["precision"],
                    pipeline["pi_loc"]["edge_precision"],
                    location=f"check {name!r}.value.precision",
                )
                _bind_exact(
                    errors,
                    value["recall"],
                    pipeline["pi_loc"]["edge_recall"],
                    location=f"check {name!r}.value.recall",
                )
                _bind_exact(
                    errors,
                    value["mst_degenerate"],
                    pipeline["pi_loc"]["mst_alone_reproduces_inferred_edges"],
                    location=f"check {name!r}.value.mst_degenerate",
                )
                return value["precision"] == 1.0 and value["recall"] == 1.0, errors
            if name == "geometry_1d_ordering":
                _bind_exact(
                    errors,
                    value["coords"],
                    pipeline["pi_geom"]["coords"],
                    location=f"check {name!r}.value.coords",
                )
                rank = value["rank"]
                increasing = all(left <= right for left, right in zip(rank, rank[1:]))
                decreasing = all(left >= right for left, right in zip(rank, rank[1:]))
                return increasing or decreasing, errors
            if name == "dimension_selection":
                _bind_exact(
                    errors,
                    value,
                    pipeline["pi_geom"]["D_star"],
                    location=f"check {name!r}.value",
                )
                return value == 1, errors
            if name == "placeholder_clock_constraint_solved":
                _bind_exact(
                    errors,
                    value,
                    pipeline["pi_time"]["constraint_residual"],
                    location=f"check {name!r}.value",
                )
                return value < 1e-10, errors

        if example == "gravity_well":
            pipeline = document["pipeline"]
            gravity = pipeline["gravity_test"]
            if name == "pi_res_admissibility":
                _bind_exact(
                    errors,
                    value["admissible"],
                    pipeline["pi_res"]["admissible"],
                    location=f"check {name!r}.value.admissible",
                )
                _bind_exact(
                    errors,
                    value["retention_loss"],
                    pipeline["pi_res"]["retention_loss"],
                    location=f"check {name!r}.value.retention_loss",
                )
                _bind_exact(
                    errors,
                    value["retention_epsilon"],
                    document["config"]["retention_epsilon"],
                    location=f"check {name!r}.value.retention_epsilon",
                )
                return value["admissible"] is True, errors
            if name == "nonzero_source_constraint_residual":
                _bind_exact(
                    errors,
                    value,
                    gravity["relative_constraint_residual"],
                    location=f"check {name!r}.value",
                )
                return value < 1e-12, errors
            if name == "monotonic_falloff":
                radial = gravity["radial_profile"]
                authoritative = {
                    f"d={int(distance)}": phi
                    for distance, phi in zip(radial["distances"], radial["phi_avg"])
                }
                _bind_exact(
                    errors,
                    value,
                    authoritative,
                    location=f"check {name!r}.value",
                )
                ordered = [value[key] for key in sorted(value, key=lambda key: int(key[2:]))]
                return all(left < right for left, right in zip(ordered, ordered[1:])), errors
            if name == "grid_symmetry":
                phi = _finite_array(
                    gravity["phi_point"],
                    ndim=1,
                    location="pipeline.gravity_test.phi_point",
                )
                graph_distances = _finite_array(
                    gravity["graph_distances"],
                    ndim=1,
                    location="pipeline.gravity_test.graph_distances",
                )
                if graph_distances.shape != phi.shape:
                    raise ValueError("gravity phi and graph-distance shapes differ")
                expected_asymmetry = 0.0
                for distance in np.unique(graph_distances):
                    shell = phi[graph_distances == distance]
                    if len(shell) > 1:
                        scale = max(float(np.max(np.abs(shell))), 1e-12)
                        expected_asymmetry = max(
                            expected_asymmetry,
                            float((np.max(shell) - np.min(shell)) / scale * 100.0),
                        )
                _bind_recomputed(
                    errors,
                    value,
                    expected_asymmetry,
                    location=f"check {name!r}.value",
                )
                return expected_asymmetry < check["threshold"], errors
            if name == "negative_well_at_source":
                authoritative_argmin = min(
                    range(len(gravity["phi_point"])),
                    key=gravity["phi_point"].__getitem__,
                )
                _bind_exact(
                    errors,
                    value["argmin"],
                    authoritative_argmin,
                    location=f"check {name!r}.value.argmin",
                )
                _bind_exact(
                    errors,
                    value["center"],
                    document["config"]["center_cell"],
                    location=f"check {name!r}.value.center",
                )
                return value["argmin"] == value["center"], errors
            if name == "redshift_positive":
                _bind_exact(
                    errors,
                    value,
                    gravity["redshift"]["z"],
                    location=f"check {name!r}.value",
                )
                return value > 0.0, errors
            if name == "dimension_selection_2d":
                _bind_exact(
                    errors,
                    value,
                    pipeline["pi_geom"]["D_star"],
                    location=f"check {name!r}.value",
                )
                return value == 2, errors

        if example == "grid_2d":
            pipeline = document["pipeline"]
            if name == "pi_res_admissibility":
                _bind_exact(
                    errors,
                    value["admissible"],
                    pipeline["pi_res"]["admissible"],
                    location=f"check {name!r}.value.admissible",
                )
                _bind_exact(
                    errors,
                    value["retention_loss"],
                    pipeline["pi_res"]["retention_loss"],
                    location=f"check {name!r}.value.retention_loss",
                )
                _bind_exact(
                    errors,
                    value["retention_epsilon"],
                    document["config"]["retention_epsilon"],
                    location=f"check {name!r}.value.retention_epsilon",
                )
                return value["admissible"] is True, errors
            if name == "blind_edge_recovery":
                _bind_exact(
                    errors,
                    value["precision"],
                    pipeline["pi_loc"]["edge_precision"],
                    location=f"check {name!r}.value.precision",
                )
                _bind_exact(
                    errors,
                    value["recall"],
                    pipeline["pi_loc"]["edge_recall"],
                    location=f"check {name!r}.value.recall",
                )
                inferred = {tuple(edge) for edge in pipeline["pi_loc"]["inferred_edges"]}
                reference = {
                    tuple(edge) for edge in pipeline["pi_loc"]["held_out_reference_edges"]
                }
                _bind_exact(
                    errors,
                    value["false_positives"],
                    len(inferred - reference),
                    location=f"check {name!r}.value.false_positives",
                )
                _bind_exact(
                    errors,
                    value["false_negatives"],
                    len(reference - inferred),
                    location=f"check {name!r}.value.false_negatives",
                )
                return value["precision"] == 1.0 and value["recall"] == 1.0, errors
            if name == "dimension_selection":
                _bind_exact(
                    errors,
                    value,
                    pipeline["pi_geom"]["D_star"],
                    location=f"check {name!r}.value",
                )
                return value == 2, errors
            if name == "topology_preservation":
                return value["avg_dist_neighbors"] < value["avg_dist_non_neighbors"], errors
            if name == "finite_graph_spectral_peak":
                _bind_exact(
                    errors,
                    value,
                    pipeline["pi_geom"]["D_spectral"],
                    location=f"check {name!r}.value",
                )
                return 1.0 <= value <= 2.0, errors
            if name == "mds_stress":
                _bind_exact(
                    errors,
                    value,
                    pipeline["pi_geom"]["stress"],
                    location=f"check {name!r}.value",
                )
                return value < 0.5, errors
            if name == "placeholder_source_degeneracy":
                pipeline_value = document["pipeline"]["pi_time"]
                phi = _finite_array(
                    pipeline_value["phi"],
                    ndim=1,
                    location="pipeline.pi_time.phi",
                )
                expected_range = float(np.max(phi) - np.min(phi))
                expected_max_absolute = float(np.max(np.abs(phi)))
                for key in ("effective_source_norm", "constraint_residual"):
                    _bind_exact(
                        errors,
                        value[key],
                        pipeline_value[key],
                        location=f"check {name!r}.value.{key}",
                    )
                _bind_recomputed(
                    errors,
                    value["phi_range"],
                    expected_range,
                    location=f"check {name!r}.value.phi_range",
                )
                _bind_recomputed(
                    errors,
                    value["max_absolute_phi"],
                    expected_max_absolute,
                    location=f"check {name!r}.value.max_absolute_phi",
                )
                return (
                    value["effective_source_norm"] < 1e-12
                    and expected_range < 1e-12
                    and expected_max_absolute < 1e-12
                ), errors

        if example == "source_law":
            measurements = document["measurements"]
            config = document["config"]
            if name == "relative_entropy_is_quadratic":
                expected_fit = _fit_dict_from_raw(
                    config["epsilons"],
                    measurements["relative_entropy"],
                    location=f"check {name!r}.raw_fit",
                    absolute_response=True,
                )
                _bind_recomputed(
                    errors,
                    measurements["fits"]["relative_entropy"],
                    expected_fit,
                    location="measurements.fits.relative_entropy",
                )
                _bind_recomputed(
                    errors,
                    value,
                    expected_fit,
                    location=f"check {name!r}.value",
                )
                return abs(expected_fit["slope"] - 2.0) < 0.02, errors
            if name == "affine_modular_linearity_identity_regression":
                epsilons = _finite_array(
                    config["epsilons"], ndim=1, location="config.epsilons"
                )
                modular_energy = _finite_array(
                    measurements["modular_energy"],
                    ndim=1,
                    location="measurements.modular_energy",
                )
                relative_entropy = _finite_array(
                    measurements["relative_entropy"],
                    ndim=1,
                    location="measurements.relative_entropy",
                )
                entropy_change = _finite_array(
                    measurements["entropy_change"],
                    ndim=1,
                    location="measurements.entropy_change",
                )
                reference = _finite_array(
                    config["reference"], ndim=1, location="config.reference"
                )
                excitation = _finite_array(
                    config["excitation"], ndim=1, location="config.excitation"
                )
                if not (
                    epsilons.shape
                    == modular_energy.shape
                    == relative_entropy.shape
                    == entropy_change.shape
                ):
                    raise ValueError("source-law epsilon/identity operand shapes differ")
                if reference.shape != excitation.shape or np.any(reference <= 0.0):
                    raise ValueError("source-law reference/excitation operands are invalid")
                expected_fit = _fit_dict_from_raw(
                    epsilons,
                    modular_energy,
                    location=f"check {name!r}.raw_fit",
                    absolute_response=True,
                )
                modular_coefficient = float(
                    np.dot(excitation - reference, -np.log(reference))
                )
                identity_error = float(
                    np.max(np.abs(modular_energy - epsilons * modular_coefficient))
                )
                first_law_identity_error = float(
                    np.max(
                        np.abs(
                            relative_entropy - (modular_energy - entropy_change)
                        )
                    )
                )
                _bind_recomputed(
                    errors,
                    measurements["fits"]["modular_energy"],
                    expected_fit,
                    location="measurements.fits.modular_energy",
                )
                _bind_recomputed(
                    errors,
                    value["fit"],
                    expected_fit,
                    location=f"check {name!r}.value.fit",
                )
                _bind_recomputed(
                    errors,
                    value["max_absolute_identity_error"],
                    identity_error,
                    location=f"check {name!r}.value.max_absolute_identity_error",
                )
                _bind_recomputed(
                    errors,
                    value["max_absolute_first_law_identity_error"],
                    first_law_identity_error,
                    location=(
                        f"check {name!r}.value."
                        "max_absolute_first_law_identity_error"
                    ),
                )
                return (
                    identity_error < 1e-12
                    and first_law_identity_error < 1e-12
                ), errors
            if name == "linear_solver_homogeneity_identity_regression":
                epsilons = config["epsilons"]
                relative_fit = _fit_dict_from_raw(
                    epsilons,
                    measurements["relative_entropy_phi_amplitude"],
                    location=f"check {name!r}.relative_phi_fit",
                    absolute_response=True,
                )
                modular_fit = _fit_dict_from_raw(
                    epsilons,
                    measurements["modular_energy_phi_amplitude"],
                    location=f"check {name!r}.modular_phi_fit",
                    absolute_response=True,
                )
                for fit_key, expected_fit in (
                    ("relative_entropy_phi", relative_fit),
                    ("modular_energy_phi", modular_fit),
                ):
                    _bind_recomputed(
                        errors,
                        measurements["fits"][fit_key],
                        expected_fit,
                        location=f"measurements.fits.{fit_key}",
                    )
                _bind_recomputed(
                    errors,
                    value["relative_entropy_phi"],
                    relative_fit,
                    location=f"check {name!r}.value.relative_entropy_phi",
                )
                _bind_recomputed(
                    errors,
                    value["modular_energy_phi"],
                    modular_fit,
                    location=f"check {name!r}.value.modular_energy_phi",
                )
                relative_entropy = _finite_array(
                    measurements["relative_entropy"],
                    ndim=1,
                    location="measurements.relative_entropy",
                )
                modular_energy = _finite_array(
                    measurements["modular_energy"],
                    ndim=1,
                    location="measurements.modular_energy",
                )
                relative_phi = _finite_array(
                    measurements["relative_entropy_phi_amplitude"],
                    ndim=1,
                    location="measurements.relative_entropy_phi_amplitude",
                )
                modular_phi = _finite_array(
                    measurements["modular_energy_phi_amplitude"],
                    ndim=1,
                    location="measurements.modular_energy_phi_amplitude",
                )
                if not (
                    relative_entropy.shape
                    == modular_energy.shape
                    == relative_phi.shape
                    == modular_phi.shape
                ):
                    raise ValueError("source-law solver ratio shapes differ")
                if np.any(relative_entropy == 0.0) or np.any(modular_energy == 0.0):
                    raise ValueError("source-law solver ratio denominator is zero")
                ratio_spread = float(
                    max(
                        np.ptp(relative_phi / relative_entropy),
                        np.ptp(modular_phi / np.abs(modular_energy)),
                    )
                )
                _bind_recomputed(
                    errors,
                    value["max_ratio_spread"],
                    ratio_spread,
                    location=f"check {name!r}.value.max_ratio_spread",
                )
                return (
                    abs(relative_fit["slope"] - 2.0) < 0.02
                    and ratio_spread < 1e-10
                ), errors
            if name == "equal_energy_entropy_confound":
                control = measurements["equal_energy_control"]
                pure = control["pure_middle"]
                mixed = control["mixed_extremes"]
                expected_operands = {
                    "equal_energy": abs(pure["energy"] - mixed["energy"]) < 1e-12,
                    "equal_modular_energy": (
                        abs(pure["modular_energy"] - mixed["modular_energy"]) < 1e-12
                    ),
                    "different_relative_entropy": (
                        abs(pure["relative_entropy"] - mixed["relative_entropy"]) > 0.1
                    ),
                }
                _bind_exact(
                    errors,
                    value,
                    expected_operands,
                    location=f"check {name!r}.value",
                )
                return (
                    value["equal_energy"] is True
                    and value["equal_modular_energy"] is True
                    and value["different_relative_entropy"] is True
                ), errors

        if example == "source_law_many_body":
            measurements = document["measurements"]
            config = document["config"]
            if name == "nonaffine_kms_response_orders":
                expected_fits = {
                    "relative_entropy": _fit_dict_from_raw(
                        config["epsilons"],
                        measurements["relative_entropy"],
                        location=f"check {name!r}.fits.relative_entropy",
                    ),
                    "modular_energy": _fit_dict_from_raw(
                        config["epsilons"],
                        measurements["modular_energy"],
                        location=f"check {name!r}.fits.modular_energy",
                        absolute_response=True,
                    ),
                    "total_energy": _fit_dict_from_raw(
                        config["epsilons"],
                        measurements["total_energy_change"],
                        location=f"check {name!r}.fits.total_energy",
                        absolute_response=True,
                    ),
                    "potential_amplitude": _fit_dict_from_raw(
                        config["epsilons"],
                        measurements["potential_amplitudes"],
                        location=f"check {name!r}.fits.potential_amplitude",
                    ),
                }
                _bind_recomputed(
                    errors,
                    measurements["fits"],
                    expected_fits,
                    location="measurements.fits",
                )
                _bind_recomputed(
                    errors,
                    value["descriptive_power_law_fits"],
                    expected_fits,
                    location=f"check {name!r}.value.descriptive_power_law_fits",
                )
                for key, authoritative_key in (
                    (
                        "relative_entropy_quadratic_assessment",
                        "relative_entropy_quadratic_assessment",
                    ),
                    ("modular_susceptibility", "modular_susceptibility"),
                    (
                        "exact_kubo_mori_quadratic_coefficient",
                        "exact_kubo_mori_quadratic_coefficient",
                    ),
                    (
                        "quadratic_coefficient_relative_error",
                        "quadratic_coefficient_relative_error",
                    ),
                    (
                        "exact_kubo_mori_modular_susceptibility",
                        "exact_kubo_mori_modular_susceptibility",
                    ),
                    (
                        "modular_susceptibility_relative_error",
                        "modular_susceptibility_relative_error",
                    ),
                    ("nonaffine_midpoint_deviation", "nonaffine_midpoint_deviation"),
                ):
                    _bind_exact(
                        errors,
                        value[key],
                        measurements[authoritative_key],
                        location=f"check {name!r}.value.{key}",
                    )
                quadratic, nested = _quadratic_assessment_outcome(
                    value["relative_entropy_quadratic_assessment"],
                    location=f"check {name!r}.value.relative_entropy_quadratic_assessment",
                    amplitudes=config["epsilons"],
                    responses=measurements["relative_entropy"],
                    absolute_precision_floor=measurements["absolute_precision_floor"],
                    minimum_signal_to_floor=config[
                        "minimum_signal_to_precision_floor"
                    ],
                    maximum_relative_coefficient_difference=config[
                        "maximum_relative_quadratic_coefficient_difference"
                    ],
                    maximum_slope_deviation=config[
                        "maximum_quadratic_slope_deviation"
                    ],
                    maximum_normalized_rmse=config[
                        "maximum_quadratic_normalized_rmse"
                    ],
                )
                susceptibility, limit_errors = _richardson_outcome(
                    value["modular_susceptibility"],
                    location=f"check {name!r}.value.modular_susceptibility",
                    amplitudes=config["epsilons"],
                    responses=measurements["modular_energy"],
                    absolute_precision_floor=measurements["absolute_precision_floor"],
                    required_error_margin=config[
                        "minimum_susceptibility_error_margin"
                    ],
                )
                errors.extend([*nested, *limit_errors])
                expected_assessment = _assessment_dict_from_raw(
                    config["epsilons"],
                    measurements["relative_entropy"],
                    location=f"check {name!r}.raw_quadratic",
                    absolute_precision_floor=measurements[
                        "absolute_precision_floor"
                    ],
                    minimum_signal_to_floor=config[
                        "minimum_signal_to_precision_floor"
                    ],
                    maximum_relative_coefficient_difference=config[
                        "maximum_relative_quadratic_coefficient_difference"
                    ],
                    maximum_slope_deviation=config[
                        "maximum_quadratic_slope_deviation"
                    ],
                    maximum_normalized_rmse=config[
                        "maximum_quadratic_normalized_rmse"
                    ],
                )
                expected_quadratic_relative_error = abs(
                    expected_assessment["full_window"]["coefficient"]
                    - float(value["exact_kubo_mori_quadratic_coefficient"])
                ) / abs(float(value["exact_kubo_mori_quadratic_coefficient"]))
                raw_susceptibility = richardson_first_order_limit(
                    _finite_array(
                        config["epsilons"],
                        ndim=1,
                        location=f"check {name!r}.susceptibility.amplitudes",
                    ),
                    _finite_array(
                        measurements["modular_energy"],
                        ndim=1,
                        location=f"check {name!r}.susceptibility.responses",
                    ),
                    absolute_precision_floor=float(
                        measurements["absolute_precision_floor"]
                    ),
                    required_error_margin=float(
                        config["minimum_susceptibility_error_margin"]
                    ),
                )
                expected_susceptibility_relative_error = abs(
                    raw_susceptibility.estimate
                    - float(value["exact_kubo_mori_modular_susceptibility"])
                ) / abs(float(value["exact_kubo_mori_modular_susceptibility"]))
                _bind_recomputed(
                    errors,
                    value["quadratic_coefficient_relative_error"],
                    expected_quadratic_relative_error,
                    location=f"check {name!r}.value.quadratic_coefficient_relative_error",
                )
                _bind_recomputed(
                    errors,
                    value["modular_susceptibility_relative_error"],
                    expected_susceptibility_relative_error,
                    location=f"check {name!r}.value.modular_susceptibility_relative_error",
                )
                return (
                    quadratic
                    and susceptibility
                    and expected_quadratic_relative_error <= 5e-4
                    and expected_susceptibility_relative_error <= 1e-6
                    and value["nonaffine_midpoint_deviation"] > 1e-9
                ), errors
            if name == "quadratic_gate_rejects_first_order_negative_control":
                quadratic, nested = _quadratic_assessment_outcome(
                    value["assessment"],
                    location=f"check {name!r}.value.assessment",
                    amplitudes=config["epsilons"],
                    responses=value["synthetic_response"],
                    absolute_precision_floor=measurements["absolute_precision_floor"],
                    minimum_signal_to_floor=config[
                        "minimum_signal_to_precision_floor"
                    ],
                    maximum_relative_coefficient_difference=config[
                        "maximum_relative_quadratic_coefficient_difference"
                    ],
                    maximum_slope_deviation=config[
                        "maximum_quadratic_slope_deviation"
                    ],
                    maximum_normalized_rmse=config[
                        "maximum_quadratic_normalized_rmse"
                    ],
                )
                errors.extend(nested)
                return not quadratic, errors
            if name == "kms_and_local_decomposition_identities":
                relative_entropy = _finite_array(
                    measurements["relative_entropy"],
                    ndim=1,
                    location="measurements.relative_entropy",
                )
                modular_energy = _finite_array(
                    measurements["modular_energy"],
                    ndim=1,
                    location="measurements.modular_energy",
                )
                entropy_change = _finite_array(
                    measurements["entropy_change"],
                    ndim=1,
                    location="measurements.entropy_change",
                )
                total_energy = _finite_array(
                    measurements["total_energy_change"],
                    ndim=1,
                    location="measurements.total_energy_change",
                )
                local_profiles = _finite_array(
                    measurements["local_energy_profiles"],
                    ndim=2,
                    location="measurements.local_energy_profiles",
                )
                if not (
                    relative_entropy.shape
                    == modular_energy.shape
                    == entropy_change.shape
                    == total_energy.shape
                    == local_profiles.shape[:1]
                ):
                    raise ValueError("many-body identity operand shapes differ")
                expected = {
                    "kms_identity_error": float(
                        np.max(
                            np.abs(
                                modular_energy - float(config["beta"]) * total_energy
                            )
                        )
                    ),
                    "first_law_identity_error": float(
                        np.max(
                            np.abs(
                                relative_entropy
                                - (modular_energy - entropy_change)
                            )
                        )
                    ),
                    "local_decomposition_error": float(
                        np.max(np.abs(local_profiles.sum(axis=1) - total_energy))
                    ),
                }
                _bind_recomputed(
                    errors,
                    value,
                    expected,
                    location=f"check {name!r}.value",
                )
                return max(expected.values()) < 5e-13, errors
            if name == "local_energy_decomposition_consistency_and_spreading":
                evolved_total = _finite_array(
                    measurements["evolved_total_energy"],
                    ndim=1,
                    location="measurements.evolved_total_energy",
                )
                evolved_profiles = _finite_array(
                    measurements["evolved_local_energy_profiles"],
                    ndim=2,
                    location="measurements.evolved_local_energy_profiles",
                )
                times = _finite_array(
                    config["evolution_times"],
                    ndim=1,
                    location="config.evolution_times",
                )
                if (
                    evolved_profiles.shape[0] != evolved_total.size
                    or times.size != evolved_total.size
                ):
                    raise ValueError("evolved energy/profile/time shapes differ")
                t1_matches = np.flatnonzero(times == 1.0)
                if t1_matches.size != 1 or evolved_profiles.shape[1] < 2:
                    raise ValueError("evolution controls require one t=1 profile and endpoints")

                def endpoint_fraction(profile: np.ndarray) -> float:
                    denominator = float(np.sum(np.abs(profile)))
                    if denominator == 0.0:
                        raise ValueError("endpoint fraction denominator is zero")
                    return float(np.sum(np.abs(profile[[0, -1]])) / denominator)

                expected = {
                    "generator_observable_consistency_drift": float(
                        np.ptp(evolved_total)
                    ),
                    "evolved_local_decomposition_error": float(
                        np.max(
                            np.abs(evolved_profiles.sum(axis=1) - evolved_total)
                        )
                    ),
                    "initial_endpoint_fraction": endpoint_fraction(
                        evolved_profiles[0]
                    ),
                    "t1_endpoint_fraction": endpoint_fraction(
                        evolved_profiles[int(t1_matches[0])]
                    ),
                }
                _bind_recomputed(
                    errors,
                    value,
                    expected,
                    location=f"check {name!r}.value",
                )
                return (
                    expected["generator_observable_consistency_drift"] < 1e-12
                    and expected["evolved_local_decomposition_error"] < 5e-13
                    and expected["initial_endpoint_fraction"] < 1e-12
                    and expected["t1_endpoint_fraction"] > 0.05
                ), errors
            if name == "nonaffine_kms_parameter_sensitivity":
                _bind_exact(
                    errors,
                    value,
                    measurements["order_sensitivity"],
                    location=f"check {name!r}.value",
                )
                outcomes: list[bool] = []
                for index, case in enumerate(value):
                    expected_modular_fit = _fit_dict_from_raw(
                        case["epsilons"],
                        case["signed_modular_energy"],
                        location=f"check {name!r}.value[{index}].modular_fit",
                        absolute_response=True,
                    )
                    _bind_recomputed(
                        errors,
                        case["modular_energy_power_law"],
                        expected_modular_fit,
                        location=(
                            f"check {name!r}.value[{index}]."
                            "modular_energy_power_law"
                        ),
                    )
                    quadratic, nested = _quadratic_assessment_outcome(
                        case["relative_entropy_quadratic_assessment"],
                        location=f"check {name!r}.value[{index}].quadratic",
                        amplitudes=case["epsilons"],
                        responses=case["relative_entropy"],
                        absolute_precision_floor=case["absolute_precision_floor"],
                        minimum_signal_to_floor=config[
                            "minimum_signal_to_precision_floor"
                        ],
                        lower_window_size=5,
                        maximum_relative_coefficient_difference=config[
                            "maximum_relative_quadratic_coefficient_difference"
                        ],
                        maximum_slope_deviation=config[
                            "maximum_quadratic_slope_deviation"
                        ],
                        maximum_normalized_rmse=config[
                            "maximum_quadratic_normalized_rmse"
                        ],
                    )
                    susceptibility, limit_errors = _richardson_outcome(
                        case["modular_susceptibility"],
                        location=f"check {name!r}.value[{index}].susceptibility",
                        amplitudes=case["epsilons"],
                        responses=case["signed_modular_energy"],
                        absolute_precision_floor=case["absolute_precision_floor"],
                        required_error_margin=config[
                            "minimum_susceptibility_error_margin"
                        ],
                    )
                    errors.extend([*nested, *limit_errors])
                    raw_assessment = _assessment_dict_from_raw(
                        case["epsilons"],
                        case["relative_entropy"],
                        location=f"check {name!r}.value[{index}].raw_quadratic",
                        absolute_precision_floor=case[
                            "absolute_precision_floor"
                        ],
                        minimum_signal_to_floor=config[
                            "minimum_signal_to_precision_floor"
                        ],
                        lower_window_size=5,
                        maximum_relative_coefficient_difference=config[
                            "maximum_relative_quadratic_coefficient_difference"
                        ],
                        maximum_slope_deviation=config[
                            "maximum_quadratic_slope_deviation"
                        ],
                        maximum_normalized_rmse=config[
                            "maximum_quadratic_normalized_rmse"
                        ],
                    )
                    raw_limit = richardson_first_order_limit(
                        _finite_array(
                            case["epsilons"],
                            ndim=1,
                            location=f"check {name!r}.value[{index}].epsilons",
                        ),
                        _finite_array(
                            case["signed_modular_energy"],
                            ndim=1,
                            location=(
                                f"check {name!r}.value[{index}].signed_modular_energy"
                            ),
                        ),
                        absolute_precision_floor=float(
                            case["absolute_precision_floor"]
                        ),
                        required_error_margin=float(
                            config["minimum_susceptibility_error_margin"]
                        ),
                    )
                    exact_quadratic = float(
                        case["exact_kubo_mori_quadratic_coefficient"]
                    )
                    exact_susceptibility = float(
                        case["exact_kubo_mori_modular_susceptibility"]
                    )
                    expected_quadratic_error = abs(
                        raw_assessment["full_window"]["coefficient"]
                        - exact_quadratic
                    ) / abs(exact_quadratic)
                    expected_susceptibility_error = abs(
                        raw_limit.estimate - exact_susceptibility
                    ) / abs(exact_susceptibility)
                    _bind_recomputed(
                        errors,
                        case["quadratic_coefficient_relative_error"],
                        expected_quadratic_error,
                        location=(
                            f"check {name!r}.value[{index}]."
                            "quadratic_coefficient_relative_error"
                        ),
                    )
                    _bind_recomputed(
                        errors,
                        case["susceptibility_relative_error"],
                        expected_susceptibility_error,
                        location=(
                            f"check {name!r}.value[{index}]."
                            "susceptibility_relative_error"
                        ),
                    )
                    expected_case = bool(
                        quadratic
                        and susceptibility
                        and expected_quadratic_error <= 5e-4
                        and expected_susceptibility_error <= 1e-6
                    )
                    if case["passed"] is not expected_case:
                        errors.append(
                            f"check {name!r}.value[{index}].passed={case['passed']!r} "
                            f"but recomputed outcome is {expected_case}"
                        )
                    outcomes.append(expected_case)
                return all(outcomes), errors
            if name == "isospectral_unitary_identity_regression":
                control = measurements["isospectral_unitary_control"]
                relative_entropy = _finite_array(
                    control["relative_entropy"],
                    ndim=1,
                    location="measurements.isospectral_unitary_control.relative_entropy",
                )
                modular_energy = _finite_array(
                    control["modular_energy"],
                    ndim=1,
                    location="measurements.isospectral_unitary_control.modular_energy",
                )
                entropy_change = _finite_array(
                    control["entropy_change"],
                    ndim=1,
                    location="measurements.isospectral_unitary_control.entropy_change",
                )
                if not (
                    relative_entropy.shape
                    == modular_energy.shape
                    == entropy_change.shape
                ):
                    raise ValueError("isospectral control response shapes differ")
                expected_relative_fit = _fit_dict_from_raw(
                    control["amplitudes"],
                    relative_entropy,
                    location=f"check {name!r}.relative_entropy_fit",
                )
                expected_modular_fit = _fit_dict_from_raw(
                    control["amplitudes"],
                    modular_energy,
                    location=f"check {name!r}.modular_energy_fit",
                )
                for key, expected_fit in (
                    ("relative_entropy_fit", expected_relative_fit),
                    ("modular_energy_fit", expected_modular_fit),
                ):
                    _bind_recomputed(
                        errors,
                        control[key],
                        expected_fit,
                        location=f"measurements.isospectral_unitary_control.{key}",
                    )
                    _bind_recomputed(
                        errors,
                        value[key],
                        expected_fit,
                        location=f"check {name!r}.value.{key}",
                    )
                expected_identity_error = float(
                    np.max(np.abs(relative_entropy - modular_energy))
                )
                expected_entropy_error = float(np.max(np.abs(entropy_change)))
                _bind_recomputed(
                    errors,
                    value["max_D_minus_modular_energy"],
                    expected_identity_error,
                    location=f"check {name!r}.value.max_D_minus_modular_energy",
                )
                _bind_recomputed(
                    errors,
                    value["max_entropy_change"],
                    expected_entropy_error,
                    location=f"check {name!r}.value.max_entropy_change",
                )
                expected_tolerance = float(
                    sys.float_info.epsilon * (2 ** int(config["n_sites"]))
                )
                _bind_recomputed(
                    errors,
                    value["dimension_scaled_float64_tolerance"],
                    expected_tolerance,
                    location=(
                        f"check {name!r}.value.dimension_scaled_float64_tolerance"
                    ),
                )
                return (
                    expected_identity_error <= expected_tolerance
                    and expected_entropy_error <= expected_tolerance
                ), errors
            if name == "spreading_requires_noncommuting_dynamics":
                control = measurements["commuting_ising_control"]
                initial_profile = _finite_array(
                    control["initial_profile"],
                    ndim=1,
                    location="measurements.commuting_ising_control.initial_profile",
                )
                t1_profile = _finite_array(
                    control["t1_profile"],
                    ndim=1,
                    location="measurements.commuting_ising_control.t1_profile",
                )
                if initial_profile.shape != t1_profile.shape:
                    raise ValueError("commuting control profile shapes differ")
                expected_change = float(
                    np.max(np.abs(t1_profile - initial_profile))
                )
                _bind_recomputed(
                    errors,
                    control["maximum_profile_change"],
                    expected_change,
                    location=(
                        "measurements.commuting_ising_control.maximum_profile_change"
                    ),
                )
                _bind_recomputed(
                    errors,
                    value["maximum_profile_change_at_t1"],
                    expected_change,
                    location=f"check {name!r}.value.maximum_profile_change_at_t1",
                )
                return expected_change < 1e-12, errors
            if name == "pipeline_reduced_modular_blindness_and_density_repair":
                comparison = measurements["pipeline_source_comparison"]
                for key in (
                    "reduced_modular_source",
                    "kms_energy_density_source",
                ):
                    _bind_exact(
                        errors,
                        value[key],
                        comparison[key],
                        location=f"check {name!r}.value.{key}",
                    )
                for vector_key, norm_key in (
                    ("reduced_modular_source", "reduced_modular_source_norm"),
                    ("kms_energy_density_source", "kms_energy_density_source_norm"),
                ):
                    expected_norm = math.sqrt(
                        math.fsum(float(item) ** 2 for item in value[vector_key])
                    )
                    if not math.isclose(
                        float(value[norm_key]),
                        expected_norm,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    ):
                        errors.append(
                            f"check {name!r}.value.{norm_key} differs from the "
                            f"retained {vector_key} norm {expected_norm!r}"
                        )
                diagnostic_profile = _finite_array(
                    comparison["diagnostic_local_energy_profile"],
                    ndim=1,
                    location=(
                        "measurements.pipeline_source_comparison."
                        "diagnostic_local_energy_profile"
                    ),
                )
                kms_source = _finite_array(
                    comparison["kms_energy_density_source"],
                    ndim=1,
                    location=(
                        "measurements.pipeline_source_comparison."
                        "kms_energy_density_source"
                    ),
                )
                if diagnostic_profile.shape != kms_source.shape:
                    raise ValueError("KMS density/profile shapes differ")
                expected_density = -float(config["beta"]) * diagnostic_profile
                expected_match_error = float(
                    np.max(np.abs(kms_source - expected_density))
                )
                _bind_recomputed(
                    errors,
                    comparison["kms_density_match_error"],
                    expected_match_error,
                    location=(
                        "measurements.pipeline_source_comparison."
                        "kms_density_match_error"
                    ),
                )
                _bind_recomputed(
                    errors,
                    value["kms_density_match_error"],
                    expected_match_error,
                    location=f"check {name!r}.value.kms_density_match_error",
                )
                return (
                    value["reduced_modular_source_norm"] < 1e-12
                    and value["kms_energy_density_source_norm"] > 1e-3
                    and expected_match_error < 1e-14
                ), errors
            if name == "negative_energy_candidate_has_slower_source_clock":
                phi = _finite_array(
                    value["phi"], ndim=1, location=f"check {name!r}.value.phi"
                )
                effective_source = _finite_array(
                    value["effective_source"],
                    ndim=1,
                    location=f"check {name!r}.value.effective_source",
                )
                kms_source = _finite_array(
                    measurements["pipeline_source_comparison"][
                        "kms_energy_density_source"
                    ],
                    ndim=1,
                    location=(
                        "measurements.pipeline_source_comparison."
                        "kms_energy_density_source"
                    ),
                )
                if phi.shape != effective_source.shape or phi.shape != kms_source.shape:
                    raise ValueError("clock source/potential shapes differ")
                if phi.size != int(config["n_sites"]):
                    raise ValueError("clock source size differs from configured sites")
                expected_background = float(np.mean(kms_source))
                expected_effective = kms_source - expected_background
                _bind_recomputed(
                    errors,
                    value["effective_source"],
                    expected_effective.tolist(),
                    location=f"check {name!r}.value.effective_source",
                )
                _bind_recomputed(
                    errors,
                    value["source_background"],
                    expected_background,
                    location=f"check {name!r}.value.source_background",
                )
                weights = np.zeros((phi.size, phi.size), dtype=float)
                for index in range(phi.size - 1):
                    weights[index, index + 1] = 1.0
                    weights[index + 1, index] = 1.0
                laplacian = np.diag(weights.sum(axis=1)) - weights
                operator = laplacian + float(config["clock_mu"]) ** 2 * np.eye(
                    phi.size
                )
                expected_residual = float(
                    np.linalg.norm(operator @ phi - expected_effective)
                )
                expected_ratio = float(np.exp(phi[2] - phi[0]))
                expected_redshift = float(np.exp(phi[0] - phi[2]) - 1.0)
                for key, expected_value in (
                    ("constraint_residual", expected_residual),
                    ("center_to_edge_clock_rate_ratio", expected_ratio),
                    ("edge_observed_redshift", expected_redshift),
                ):
                    _bind_recomputed(
                        errors,
                        value[key],
                        expected_value,
                        location=f"check {name!r}.value.{key}",
                    )
                return (
                    int(np.argmin(phi)) == 2
                    and expected_ratio < 1.0
                    and expected_redshift > 0.0
                    and expected_residual < 1e-12
                ), errors
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return None, [f"check {name!r} decision operands are invalid: {exc}"]

    return None, [
        f"check {name!r} in {relative_path} has no registered executable decision predicate"
    ]


def check_validation_semantics(
    document: Any,
    relative_path: str,
) -> list[str]:
    """Validate headline/check consistency and the informational-check policy."""
    if not isinstance(document, dict):
        return ["validation document must be an object"]
    checks = document.get("checks")
    overall_pass = document.get("overall_pass")
    if not isinstance(checks, list) or not checks:
        return ["checks must be a nonempty array"]
    if not isinstance(overall_pass, bool):
        return ["overall_pass must be a boolean"]

    errors: list[str] = _retained_potential_errors(document)
    names: set[str] = set()
    informational_names: set[str] = set()
    noninformational_names: set[str] = set()
    noninformational_outcomes: list[bool] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"checks[{index}] must be an object")
            continue
        name = check.get("name")
        passed = check.get("passed")
        severity = check.get("severity")
        if not isinstance(name, str) or not name:
            errors.append(f"checks[{index}].name must be a nonempty string")
            continue
        if name in names:
            errors.append(f"duplicate check name {name!r}")
        names.add(name)
        if not isinstance(passed, bool):
            errors.append(f"check {name!r} must have a boolean passed value")
            continue
        if severity not in {None, "informational"}:
            errors.append(f"check {name!r} has unsupported severity {severity!r}")
            continue
        if severity == "informational":
            informational_names.add(name)
            if not passed:
                errors.append(f"failing check {name!r} cannot be informational")
        else:
            noninformational_names.add(name)
            noninformational_outcomes.append(passed)
        recomputed, decision_errors = _decision_outcome(
            document,
            check,
            relative_path,
        )
        errors.extend(decision_errors)
        if recomputed is not None and passed is not recomputed:
            errors.append(
                f"check {name!r} passed={passed} but retained operands recompute to "
                f"{recomputed}"
            )

    expected_informational = INFORMATIONAL_CHECK_ALLOWLIST.get(
        relative_path, frozenset()
    )
    if informational_names != expected_informational:
        errors.append(
            "informational check set changed "
            f"(expected={sorted(expected_informational)}, "
            f"actual={sorted(informational_names)})"
        )
    expected_noninformational = NONINFORMATIONAL_CHECK_ALLOWLIST.get(
        relative_path, frozenset()
    )
    if noninformational_names != expected_noninformational:
        errors.append(
            "non-informational check set is not explicitly registered "
            f"(missing={sorted(expected_noninformational - noninformational_names)}, "
            f"unregistered={sorted(noninformational_names - expected_noninformational)})"
        )
    expected_overall = all(noninformational_outcomes)
    if overall_pass is not expected_overall:
        errors.append(
            f"overall_pass={overall_pass} but non-informational conjunction is "
            f"{expected_overall}"
        )
    return errors


def check_required_visuals(
    document: Any,
    validation_path: Path,
    *,
    repo_root: Path,
    tracked_paths: set[str],
) -> list[str]:
    """Require every declared visual output to be tracked, present, and nonempty."""

    errors: list[str] = []
    artifacts = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(artifacts, list) or not all(isinstance(item, str) for item in artifacts):
        return [f"{validation_path}: artifacts must be an array of paths"]

    visual_artifacts = [item for item in artifacts if Path(item).suffix.lower() in VISUAL_SUFFIXES]
    if not visual_artifacts:
        return [f"{validation_path}: no required visual artifact is declared"]

    repo_root = repo_root.resolve()
    example_root = validation_path.parent.parent
    for artifact in visual_artifacts:
        visual_path = (example_root / artifact).resolve()
        try:
            relative_path = visual_path.relative_to(repo_root).as_posix()
        except ValueError:
            errors.append(f"{validation_path}: visual path escapes repository: {artifact!r}")
            continue
        if relative_path not in tracked_paths:
            errors.append(f"{relative_path}: required visual is not tracked by git")
        if not visual_path.is_file():
            errors.append(f"{relative_path}: required visual is missing")
        elif visual_path.stat().st_size == 0:
            errors.append(f"{relative_path}: required visual is empty")
    return errors


def compare_visual_artifact(
    reference_bytes: bytes,
    candidate_path: Path,
) -> list[str]:
    """Compare a regenerated raster visual to its committed semantic envelope.

    Encoded bytes are intentionally not compared: PNG metadata and compression can
    differ across platforms.  Geometry, mode, frame count, and bounded RGBA pixel
    differences are part of the portable contract instead.
    """

    try:
        with Image.open(io.BytesIO(reference_bytes)) as reference_image:
            with Image.open(candidate_path) as candidate_image:
                reference_format = reference_image.format
                candidate_format = candidate_image.format
                reference_frames = getattr(reference_image, "n_frames", 1)
                candidate_frames = getattr(candidate_image, "n_frames", 1)
                metadata = (
                    reference_format,
                    reference_image.size,
                    reference_image.mode,
                    reference_frames,
                )
                candidate_metadata = (
                    candidate_format,
                    candidate_image.size,
                    candidate_image.mode,
                    candidate_frames,
                )
                if metadata != candidate_metadata:
                    return [
                        "visual metadata changed "
                        f"(reference={metadata!r}, candidate={candidate_metadata!r})"
                    ]

                errors: list[str] = []
                for frame_index in range(reference_frames):
                    reference_image.seek(frame_index)
                    candidate_image.seek(frame_index)
                    reference_rgba = reference_image.convert("RGBA")
                    candidate_rgba = candidate_image.convert("RGBA")
                    reference_array = np.asarray(reference_rgba, dtype=np.int16)
                    candidate_array = np.asarray(candidate_rgba, dtype=np.int16)
                    absolute_error = np.abs(reference_array - candidate_array)
                    maximum_channel_error = int(absolute_error.max())
                    if maximum_channel_error > VISUAL_MAXIMUM_CHANNEL_ERROR_LIMIT:
                        errors.append(
                            f"frame {frame_index}: maximum per-channel pixel error "
                            f"{maximum_channel_error} exceeds calibrated limit "
                            f"{VISUAL_MAXIMUM_CHANNEL_ERROR_LIMIT}"
                        )
                return errors
    except (OSError, UnidentifiedImageError) as exc:
        return [f"could not decode visual artifact: {exc}"]


def _run_git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _run_git_bytes(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _tracked_paths(repo_root: Path) -> set[str]:
    return {line for line in _run_git(repo_root, "ls-files").splitlines() if line}


def check_repository(
    repo_root: Path,
    *,
    base_ref: str = "HEAD",
    enforce_change_boundary: bool = False,
) -> list[str]:
    """Check all working validation artifacts against ``base_ref``."""

    repo_root = repo_root.resolve()
    base_listing = _run_git(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        base_ref,
        "--",
        "examples/physics_qg",
    )
    base_paths = {
        path for path in base_listing.splitlines() if path.endswith("/results/validation.json")
    }
    working_paths = {
        path.relative_to(repo_root).as_posix() for path in repo_root.glob(VALIDATION_GLOB)
    }

    errors: list[str] = []
    if base_paths != working_paths:
        errors.append(
            "validation artifact set changed "
            f"(missing={sorted(base_paths - working_paths)}, "
            f"added={sorted(working_paths - base_paths)})"
        )

    tracked_paths = _tracked_paths(repo_root)
    declared_artifact_paths: set[str] = set()
    for relative_path in sorted(base_paths & working_paths):
        working_path = repo_root / relative_path
        try:
            reference = load_json_document(
                _run_git(repo_root, "show", f"{base_ref}:{relative_path}"),
                source=f"{base_ref}:{relative_path}",
            )
            candidate = load_json_document(
                working_path.read_text(encoding="utf-8"),
                source=relative_path,
            )
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            errors.append(str(exc))
            continue

        summary = compare_validation_documents(reference, candidate)
        errors.extend(f"{relative_path}: {error}" for error in summary.errors)
        errors.extend(
            f"{relative_path}: {error}"
            for error in check_validation_semantics(candidate, relative_path)
        )
        errors.extend(
            check_required_visuals(
                candidate,
                working_path,
                repo_root=repo_root,
                tracked_paths=tracked_paths,
            )
        )
        artifacts = candidate.get("artifacts") if isinstance(candidate, dict) else None
        if isinstance(artifacts, list):
            example_root = working_path.parent.parent
            for artifact in artifacts:
                if not isinstance(artifact, str):
                    continue
                artifact_path = (example_root / artifact).resolve()
                try:
                    artifact_relative = artifact_path.relative_to(repo_root).as_posix()
                except ValueError:
                    continue
                declared_artifact_paths.add(artifact_relative)
                if artifact_path.suffix.lower() not in VISUAL_SUFFIXES:
                    continue
                try:
                    reference_bytes = _run_git_bytes(
                        repo_root,
                        "show",
                        f"{base_ref}:{artifact_relative}",
                    )
                except subprocess.CalledProcessError as exc:
                    errors.append(f"{artifact_relative}: could not read committed visual: {exc}")
                    continue
                errors.extend(
                    f"{artifact_relative}: {error}"
                    for error in compare_visual_artifact(reference_bytes, artifact_path)
                )
        if summary.accepted_numeric_drifts:
            print(
                f"{relative_path}: accepted {summary.accepted_numeric_drifts} bounded numeric "
                f"drifts (max abs={summary.largest_absolute_drift:.6g}, "
                f"max rel={summary.largest_relative_drift:.6g})"
            )

    if enforce_change_boundary:
        errors.extend(
            check_repository_boundary(
                repo_root,
                declared_artifact_paths,
                base_ref=base_ref,
            )
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="HEAD", help="committed artifact reference")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--enforce-change-boundary",
        action="store_true",
        help="allow working-tree changes only for artifacts declared by validation JSON",
    )
    args = parser.parse_args()

    try:
        errors = check_repository(
            args.repo_root,
            base_ref=args.base_ref,
            enforce_change_boundary=args.enforce_change_boundary,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        print(f"validation artifact check could not run: {detail}", file=sys.stderr)
        return 2

    if errors:
        print("Validation artifact contract failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Validation artifact contracts and required visual outputs are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
