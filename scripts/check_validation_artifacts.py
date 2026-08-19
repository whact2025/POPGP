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
* every decision Boolean is recomputed from its retained typed operands; and
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
from PIL import Image, ImageChops, UnidentifiedImageError
from scipy import ndimage

from scripts.check_reproduction_boundary import (
    check_repository_boundary,
    is_known_runtime_ignored_path,
)

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
VISUAL_MEAN_ABSOLUTE_ERROR_LIMIT = 2.0 / 255.0
VISUAL_LARGE_ERROR_THRESHOLD = 32
VISUAL_LARGE_ERROR_FRACTION_LIMIT = 0.02
VISUAL_LOCAL_WINDOW_SIZE = 32
VISUAL_LOCAL_MEAN_ERROR_LIMIT = 0.25
VISUAL_LARGEST_HIGH_ERROR_COMPONENT_LIMIT = 768
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


def _quadratic_assessment_outcome(
    assessment: Any,
    *,
    location: str,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    full = assessment["full_window"]
    lower = assessment["lower_window"]
    expected = bool(
        full["coefficient"] > 0.0
        and assessment["relative_coefficient_difference"] <= 1e-3
        and assessment["slope_deviation"] <= 0.02
        and max(full["normalized_rmse"], lower["normalized_rmse"]) <= 1e-2
    )
    if assessment["passed"] is not expected:
        errors.append(
            f"{location}.passed={assessment['passed']!r} but recomputed outcome is {expected}"
        )
    return expected, errors


def _richardson_outcome(
    limit: Any,
    *,
    location: str,
) -> tuple[bool, list[str]]:
    expected = bool(abs(limit["estimate"]) > 10.0 * limit["total_error"])
    errors = []
    if limit["passed"] is not expected:
        errors.append(
            f"{location}.passed={limit['passed']!r} but recomputed outcome is {expected}"
        )
    return expected, errors


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
            if name == "population_survival":
                return value > 0, errors
            if name == "population_growth":
                return value["final"] >= value["initial"], errors
            if name == "survivor_entropy_filter_regression":
                return value < check["threshold"], errors

        if example == "chain_1d":
            if name == "stability_selection":
                return value["invalid"] > value["valid"], errors
            if name == "contiguous_cells":
                return value == [[0, 1], [2, 3], [4, 5], [6, 7]], errors
            if name == "su2_equivariance_identity_regression":
                return value is True, errors
            if name == "blind_edge_recovery":
                return value["precision"] == 1.0 and value["recall"] == 1.0, errors
            if name == "geometry_1d_ordering":
                rank = value["rank"]
                increasing = all(left <= right for left, right in zip(rank, rank[1:]))
                decreasing = all(left >= right for left, right in zip(rank, rank[1:]))
                return increasing or decreasing, errors
            if name == "dimension_selection":
                return value == 1, errors
            if name == "placeholder_clock_constraint_solved":
                return value < 1e-10, errors

        if example == "gravity_well":
            if name == "pi_res_admissibility":
                return value["admissible"] is True, errors
            if name == "nonzero_source_constraint_residual":
                return value < 1e-12, errors
            if name == "monotonic_falloff":
                ordered = [value[key] for key in sorted(value, key=lambda key: int(key[2:]))]
                return all(left < right for left, right in zip(ordered, ordered[1:])), errors
            if name == "grid_symmetry":
                return value < check["threshold"], errors
            if name == "negative_well_at_source":
                return value["argmin"] == value["center"], errors
            if name == "redshift_positive":
                return value > 0.0, errors
            if name == "dimension_selection_2d":
                return value == 2, errors

        if example == "grid_2d":
            if name == "pi_res_admissibility":
                return value["admissible"] is True, errors
            if name == "blind_edge_recovery":
                return value["precision"] == 1.0 and value["recall"] == 1.0, errors
            if name == "dimension_selection":
                return value == 2, errors
            if name == "topology_preservation":
                return value["avg_dist_neighbors"] < value["avg_dist_non_neighbors"], errors
            if name == "finite_graph_spectral_peak":
                return 1.0 <= value <= 2.0, errors
            if name == "mds_stress":
                return value < 0.5, errors
            if name == "placeholder_source_degeneracy":
                pipeline_value = document["pipeline"]["pi_time"]
                for key in ("effective_source_norm", "phi_range", "constraint_residual"):
                    if value[key] != pipeline_value[key]:
                        errors.append(
                            f"{name!r} operand {key!r} differs from pipeline.pi_time"
                        )
                return (
                    value["effective_source_norm"] < 1e-12
                    and value["phi_range"] < 1e-12
                ), errors

        if example == "source_law":
            if name == "relative_entropy_is_quadratic":
                return abs(value["slope"] - 2.0) < 0.02, errors
            if name == "affine_modular_linearity_identity_regression":
                return value["max_absolute_identity_error"] < 1e-12, errors
            if name == "linear_solver_homogeneity_identity_regression":
                return (
                    abs(value["relative_entropy_phi"]["slope"] - 2.0) < 0.02
                    and value["max_ratio_spread"] < 1e-10
                ), errors
            if name == "equal_energy_entropy_confound":
                return (
                    value["equal_energy"] is True
                    and value["equal_modular_energy"] is True
                    and value["different_relative_entropy"] is True
                ), errors

        if example == "source_law_many_body":
            if name == "nonaffine_kms_response_orders":
                quadratic, nested = _quadratic_assessment_outcome(
                    value["relative_entropy_quadratic_assessment"],
                    location=f"check {name!r}.value.relative_entropy_quadratic_assessment",
                )
                susceptibility, limit_errors = _richardson_outcome(
                    value["modular_susceptibility"],
                    location=f"check {name!r}.value.modular_susceptibility",
                )
                errors.extend([*nested, *limit_errors])
                return (
                    quadratic
                    and susceptibility
                    and value["quadratic_coefficient_relative_error"] <= 5e-4
                    and value["modular_susceptibility_relative_error"] <= 1e-6
                    and value["nonaffine_midpoint_deviation"] > 1e-9
                ), errors
            if name == "quadratic_gate_rejects_first_order_negative_control":
                quadratic, nested = _quadratic_assessment_outcome(
                    value["assessment"],
                    location=f"check {name!r}.value.assessment",
                )
                errors.extend(nested)
                return not quadratic, errors
            if name == "kms_and_local_decomposition_identities":
                return max(value.values()) < 5e-13, errors
            if name == "local_energy_decomposition_consistency_and_spreading":
                return (
                    value["generator_observable_consistency_drift"] < 1e-12
                    and value["initial_endpoint_fraction"] < 1e-12
                    and value["t1_endpoint_fraction"] > 0.05
                ), errors
            if name == "nonaffine_kms_parameter_sensitivity":
                outcomes: list[bool] = []
                for index, case in enumerate(value):
                    quadratic, nested = _quadratic_assessment_outcome(
                        case["relative_entropy_quadratic_assessment"],
                        location=f"check {name!r}.value[{index}].quadratic",
                    )
                    susceptibility, limit_errors = _richardson_outcome(
                        case["modular_susceptibility"],
                        location=f"check {name!r}.value[{index}].susceptibility",
                    )
                    errors.extend([*nested, *limit_errors])
                    expected_case = bool(
                        quadratic
                        and susceptibility
                        and case["quadratic_coefficient_relative_error"] <= 5e-4
                        and case["susceptibility_relative_error"] <= 1e-6
                    )
                    if case["passed"] is not expected_case:
                        errors.append(
                            f"check {name!r}.value[{index}].passed={case['passed']!r} "
                            f"but recomputed outcome is {expected_case}"
                        )
                    outcomes.append(expected_case)
                return all(outcomes), errors
            if name == "isospectral_unitary_identity_regression":
                tolerance = value["dimension_scaled_float64_tolerance"]
                return (
                    value["max_D_minus_modular_energy"] <= tolerance
                    and value["max_entropy_change"] <= tolerance
                ), errors
            if name == "spreading_requires_noncommuting_dynamics":
                return value["maximum_profile_change_at_t1"] < 1e-12, errors
            if name == "pipeline_reduced_modular_blindness_and_density_repair":
                return (
                    value["reduced_modular_source_norm"] < 1e-12
                    and value["kms_energy_density_source_norm"] > 1e-3
                    and value["kms_density_match_error"] < 1e-14
                ), errors
            if name == "negative_energy_candidate_has_slower_source_clock":
                return (
                    min(range(len(value["phi"])), key=value["phi"].__getitem__) == 2
                    and value["center_to_edge_clock_rate_ratio"] < 1.0
                    and value["edge_observed_redshift"] > 0.0
                    and value["constraint_residual"] < 1e-12
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

    errors: list[str] = []
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
                    difference = ImageChops.difference(reference_rgba, candidate_rgba)
                    reference_array = np.asarray(reference_rgba, dtype=np.int16)
                    candidate_array = np.asarray(candidate_rgba, dtype=np.int16)
                    absolute_error = np.abs(reference_array - candidate_array)
                    histogram = difference.histogram()
                    channel_samples = reference_rgba.width * reference_rgba.height * 4
                    total_error = sum(
                        (value % 256) * count for value, count in enumerate(histogram)
                    )
                    mean_absolute_error = total_error / (255.0 * channel_samples)
                    large_error_count = sum(
                        count
                        for value, count in enumerate(histogram)
                        if value % 256 > VISUAL_LARGE_ERROR_THRESHOLD
                    )
                    large_error_fraction = large_error_count / channel_samples
                    pixel_error = absolute_error.mean(axis=2) / 255.0
                    local_mean_error = float(
                        ndimage.uniform_filter(
                            pixel_error,
                            size=VISUAL_LOCAL_WINDOW_SIZE,
                            mode="constant",
                        ).max()
                    )
                    high_error_pixels = (
                        absolute_error.max(axis=2) > VISUAL_LARGE_ERROR_THRESHOLD
                    )
                    components, component_count = ndimage.label(
                        high_error_pixels,
                        structure=np.ones((3, 3), dtype=np.uint8),
                    )
                    component_sizes = (
                        np.bincount(components.ravel())[1:]
                        if component_count
                        else np.asarray([], dtype=int)
                    )
                    largest_component = (
                        int(component_sizes.max()) if component_sizes.size else 0
                    )
                    if mean_absolute_error > VISUAL_MEAN_ABSOLUTE_ERROR_LIMIT:
                        errors.append(
                            f"frame {frame_index}: normalized mean absolute pixel error "
                            f"{mean_absolute_error:.6g} exceeds "
                            f"{VISUAL_MEAN_ABSOLUTE_ERROR_LIMIT:.6g}"
                        )
                    if large_error_fraction > VISUAL_LARGE_ERROR_FRACTION_LIMIT:
                        errors.append(
                            f"frame {frame_index}: large-error channel fraction "
                            f"{large_error_fraction:.6g} exceeds "
                            f"{VISUAL_LARGE_ERROR_FRACTION_LIMIT:.6g}"
                        )
                    if local_mean_error > VISUAL_LOCAL_MEAN_ERROR_LIMIT:
                        errors.append(
                            f"frame {frame_index}: maximum local {VISUAL_LOCAL_WINDOW_SIZE}x"
                            f"{VISUAL_LOCAL_WINDOW_SIZE} normalized mean pixel error "
                            f"{local_mean_error:.6g} exceeds "
                            f"{VISUAL_LOCAL_MEAN_ERROR_LIMIT:.6g}"
                        )
                    if largest_component > VISUAL_LARGEST_HIGH_ERROR_COMPONENT_LIMIT:
                        errors.append(
                            f"frame {frame_index}: largest connected high-error region has "
                            f"{largest_component} pixels, exceeding "
                            f"{VISUAL_LARGEST_HIGH_ERROR_COMPONENT_LIMIT}"
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
                allowed_ignored_path=is_known_runtime_ignored_path,
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
