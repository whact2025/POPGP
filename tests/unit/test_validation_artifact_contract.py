from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.check_validation_artifacts import (
    check_required_visuals,
    check_validation_semantics,
    compare_validation_documents,
    compare_visual_artifact,
)


def _reference_document() -> dict:
    return {
        "example": "many_body_source_law",
        "scientific_status": "candidate_test",
        "config": {
            "beta": 1.3,
            "epsilons": [1e-5, 1.778279410038923e-5],
            "n_sites": 5,
        },
        "measurements": {
            "relative_entropy": [7.784306532698793e-11, 2.461622017335685e-10],
        },
        "checks": [
            {
                "name": "nonaffine_kms_response_orders",
                "criterion": "direct response gates pass",
                "value": {
                    "coefficient": 0.05183472812398261,
                    "coefficient_residual_scale": 5.133407823919096e-8,
                    "linear_correction": 0.003968513407430131,
                    "normalized_rmse": 1.8445130302159442e-6,
                    "relative_coefficient_difference": 3.0831377813706415e-7,
                    "significance_ratio": 1_212_944_752.3881521,
                },
                "passed": True,
            }
        ],
        "overall_pass": True,
        "artifacts": ["results/many_body_source.png", "results/validation.json"],
    }


def _linux_candidate() -> dict:
    candidate = _reference_document()
    candidate["config"]["epsilons"][0] = math.nextafter(1e-5, 0.0)
    candidate["measurements"]["relative_entropy"] = [
        7.784217714856823e-11,
        2.461630899119882e-10,
    ]
    value = candidate["checks"][0]["value"]
    value.update(
        {
            "coefficient": 0.05183533105485934,
            "coefficient_residual_scale": 6.297231712262574e-7,
            "linear_correction": -0.00045365154354176646,
            "normalized_rmse": 2.1736485163697116e-5,
            "relative_coefficient_difference": 8.402160087083213e-6,
            "significance_ratio": 1_756_962_845.6270688,
        }
    )
    return candidate


def test_representative_linux_windows_numeric_drift_is_accepted() -> None:
    summary = compare_validation_documents(_reference_document(), _linux_candidate())

    assert summary.passed
    assert summary.accepted_numeric_drifts > 0


def test_changed_gate_outcome_is_rejected() -> None:
    candidate = _linux_candidate()
    candidate["checks"][0]["passed"] = False

    summary = compare_validation_documents(_reference_document(), candidate)

    assert not summary.passed
    assert any("$.checks[0].passed" in error for error in summary.errors)


def test_changed_metadata_is_rejected() -> None:
    candidate = _linux_candidate()
    candidate["checks"][0]["criterion"] = "a weaker criterion"

    summary = compare_validation_documents(_reference_document(), candidate)

    assert not summary.passed
    assert any("$.checks[0].criterion" in error for error in summary.errors)


def test_changed_array_shape_is_rejected() -> None:
    candidate = _linux_candidate()
    candidate["measurements"]["relative_entropy"].append(1e-9)

    summary = compare_validation_documents(_reference_document(), candidate)

    assert not summary.passed
    assert any("array length changed" in error for error in summary.errors)


def test_meaningful_stable_config_change_is_rejected() -> None:
    candidate = _linux_candidate()
    candidate["config"]["beta"] = 1.31

    summary = compare_validation_documents(_reference_document(), candidate)

    assert not summary.passed
    assert any("stable input drift exceeds" in error for error in summary.errors)


def test_non_finite_diagnostic_is_rejected() -> None:
    candidate = _linux_candidate()
    candidate["checks"][0]["value"]["coefficient"] = math.nan

    summary = compare_validation_documents(_reference_document(), candidate)

    assert not summary.passed
    assert any("non-finite number is forbidden" in error for error in summary.errors)


def test_sensitive_diagnostic_drift_remains_bounded() -> None:
    candidate = _linux_candidate()
    candidate["checks"][0]["value"]["linear_correction"] = 0.05

    summary = compare_validation_documents(_reference_document(), candidate)

    assert not summary.passed
    assert any("sensitive diagnostic 'linear_correction'" in error for error in summary.errors)


def test_sensitive_relative_drift_remains_bounded() -> None:
    candidate = _linux_candidate()
    candidate["checks"][0]["value"]["significance_ratio"] = 1_000_000.0

    summary = compare_validation_documents(_reference_document(), candidate)

    assert not summary.passed
    assert any("sensitive diagnostic 'significance_ratio'" in error for error in summary.errors)


def test_noise_dominated_significance_ratio_accepts_observed_ci_drift() -> None:
    reference = _reference_document()
    candidate = _linux_candidate()
    reference["checks"][0]["value"]["significance_ratio"] = 934_024_743.6098032
    candidate["checks"][0]["value"]["significance_ratio"] = 371_279_242.7637979

    summary = compare_validation_documents(reference, candidate)

    assert summary.passed


def test_declared_tracked_nonempty_visual_is_accepted(tmp_path: Path) -> None:
    validation_path = tmp_path / "examples/physics_qg/demo/results/validation.json"
    visual_path = validation_path.parent / "plot.png"
    visual_path.parent.mkdir(parents=True)
    visual_path.write_bytes(b"nonempty image payload")
    document = {"artifacts": ["results/plot.png", "results/validation.json"]}

    errors = check_required_visuals(
        document,
        validation_path,
        repo_root=tmp_path,
        tracked_paths={"examples/physics_qg/demo/results/plot.png"},
    )

    assert errors == []


def test_empty_or_untracked_visual_is_rejected(tmp_path: Path) -> None:
    validation_path = tmp_path / "examples/physics_qg/demo/results/validation.json"
    visual_path = validation_path.parent / "plot.png"
    visual_path.parent.mkdir(parents=True)
    visual_path.write_bytes(b"")
    document = {"artifacts": ["results/plot.png", "results/validation.json"]}

    errors = check_required_visuals(
        document,
        validation_path,
        repo_root=tmp_path,
        tracked_paths=set(),
    )

    assert any("not tracked" in error for error in errors)
    assert any("empty" in error for error in errors)


def _png_bytes(color: tuple[int, int, int, int], *, size: tuple[int, int] = (32, 32)) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_visual_contract_accepts_equivalent_reencoding(tmp_path: Path) -> None:
    candidate_path = tmp_path / "plot.png"
    candidate_path.write_bytes(_png_bytes((10, 20, 30, 255)))

    errors = compare_visual_artifact(
        _png_bytes((10, 20, 30, 255)),
        candidate_path,
    )

    assert errors == []


def test_visual_contract_accepts_only_calibrated_channel_noise(tmp_path: Path) -> None:
    reference = _png_bytes((100, 100, 100, 255))
    candidate_path = tmp_path / "plot.png"
    candidate_path.write_bytes(_png_bytes((104, 96, 100, 255)))

    assert compare_visual_artifact(reference, candidate_path) == []

    candidate_path.write_bytes(_png_bytes((105, 100, 100, 255)))
    errors = compare_visual_artifact(reference, candidate_path)
    assert any("per-channel pixel error" in error for error in errors)


@pytest.mark.negative_control
def test_visual_contract_rejects_geometry_or_pixel_weakening(tmp_path: Path) -> None:
    candidate_path = tmp_path / "plot.png"
    reference = _png_bytes((10, 20, 30, 255))
    candidate_path.write_bytes(_png_bytes((255, 255, 255, 255), size=(32, 31)))
    metadata_errors = compare_visual_artifact(reference, candidate_path)
    assert any("metadata changed" in error for error in metadata_errors)

    candidate_path.write_bytes(_png_bytes((255, 255, 255, 255)))
    errors = compare_visual_artifact(reference, candidate_path)
    assert any("pixel error" in error or "large-error" in error for error in errors)


@pytest.mark.negative_control
def test_visual_contract_rejects_localized_structured_corruption(tmp_path: Path) -> None:
    source = Path(
        "examples/physics_qg/gravity_well/results/source_comparison.png"
    )
    reference = source.read_bytes()
    candidate_path = tmp_path / "source_comparison.png"
    with Image.open(BytesIO(reference)) as image:
        candidate = image.convert("RGBA")
    width, height = candidate.size
    ImageDraw.Draw(candidate).rectangle(
        (
            width // 2 - 90,
            height // 2 - 35,
            width // 2 + 89,
            height // 2 + 35,
        ),
        fill=(255, 0, 0, 255),
    )
    candidate.save(candidate_path)

    errors = compare_visual_artifact(reference, candidate_path)

    assert any("per-channel pixel error" in error for error in errors)


@pytest.mark.negative_control
@pytest.mark.parametrize("mutation", ["remove_curve", "move_curve", "remove_annotation"])
def test_visual_contract_rejects_thin_and_annotation_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    reference_image = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    candidate_image = reference_image.copy()
    reference_draw = ImageDraw.Draw(reference_image)
    candidate_draw = ImageDraw.Draw(candidate_image)
    if mutation == "remove_curve":
        reference_draw.line((50, 256, 462, 256), fill=(0, 0, 0, 255), width=3)
    elif mutation == "move_curve":
        reference_draw.line((50, 250, 462, 250), fill=(0, 0, 0, 255), width=3)
        candidate_draw.line((50, 256, 462, 256), fill=(0, 0, 0, 255), width=3)
    else:
        # A dense legend/annotation surrogate that occupies well below the old 2%
        # global-error allowance.
        reference_draw.rectangle((100, 220, 180, 240), fill=(0, 0, 0, 255))

    reference = BytesIO()
    reference_image.save(reference, format="PNG")
    candidate_path = tmp_path / f"{mutation}.png"
    candidate_image.save(candidate_path)

    errors = compare_visual_artifact(reference.getvalue(), candidate_path)

    assert any("per-channel pixel error" in error for error in errors)


@pytest.mark.negative_control
@pytest.mark.parametrize(
    "mutation",
    ["compact_feature", "one_pixel_curve", "dashed_curve", "rendered_text"],
)
def test_visual_contract_rejects_small_meaningful_features(
    tmp_path: Path,
    mutation: str,
) -> None:
    reference_image = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
    candidate_image = reference_image.copy()
    draw = ImageDraw.Draw(reference_image)
    if mutation == "compact_feature":
        draw.rectangle((247, 247, 264, 264), fill=(0, 0, 0, 255))
    elif mutation == "one_pixel_curve":
        draw.line((20, 256, 491, 256), fill=(0, 0, 0, 255), width=1)
    elif mutation == "dashed_curve":
        for start in range(20, 492, 16):
            draw.line((start, 256, min(start + 7, 491), 256), fill=(0, 0, 0, 255))
    else:
        draw.text((220, 250), "PASS=TRUE", fill=(0, 0, 0, 255))

    reference = BytesIO()
    reference_image.save(reference, format="PNG")
    candidate_path = tmp_path / f"{mutation}.png"
    candidate_image.save(candidate_path)

    errors = compare_visual_artifact(reference.getvalue(), candidate_path)

    assert any("per-channel pixel error" in error for error in errors)


@pytest.mark.negative_control
def test_visual_contract_rejects_actual_plot_annotation_removal(tmp_path: Path) -> None:
    source = Path(
        "examples/physics_qg/gravity_well/results/source_comparison.png"
    )
    reference = source.read_bytes()
    with Image.open(BytesIO(reference)) as image:
        candidate = image.convert("RGBA")
    # Remove the upper-left retained `0.0` annotation using the uniform heatmap
    # background sampled next to it.  This is the real reviewer counterexample,
    # not a dense annotation surrogate.
    fill = candidate.getpixel((150, 202))
    ImageDraw.Draw(candidate).rectangle((168, 187, 210, 216), fill=fill)
    candidate_path = tmp_path / "source_comparison.png"
    candidate.save(candidate_path)

    errors = compare_visual_artifact(reference, candidate_path)

    assert any("per-channel pixel error" in error for error in errors)


@pytest.mark.negative_control
def test_threshold_crossing_operand_recomputes_failed_gate() -> None:
    relative_path = "examples/physics_qg/grid_2d/results/validation.json"
    document = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    placeholder = next(
        check
        for check in document["checks"]
        if check["name"] == "placeholder_source_degeneracy"
    )
    placeholder["value"]["phi_range"] = 4e-9
    document["pipeline"]["pi_time"]["phi_range"] = 4e-9

    comparison = compare_validation_documents(
        json.loads(Path(relative_path).read_text(encoding="utf-8")),
        document,
    )
    semantic_errors = check_validation_semantics(document, relative_path)

    assert comparison.passed  # The bounded portability comparison is not the decision oracle.
    assert any(
        "phi_range differs from recomputed raw potential" in error
        for error in semantic_errors
    )


@pytest.mark.negative_control
@pytest.mark.parametrize(
    ("relative_path", "pipeline_path"),
    [
        (
            "examples/physics_qg/gravity_well/results/validation.json",
            ("pipeline", "gravity_test", "relative_constraint_residual"),
        ),
        (
            "examples/physics_qg/chain_1d/results/validation.json",
            ("pipeline", "pi_time", "constraint_residual"),
        ),
    ],
)
def test_threshold_crossing_pipeline_alias_cannot_leave_stale_check(
    relative_path: str,
    pipeline_path: tuple[str, ...],
) -> None:
    document = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    target = document
    for part in pipeline_path[:-1]:
        target = target[part]
    target[pipeline_path[-1]] = 4e-9

    errors = check_validation_semantics(document, relative_path)

    assert any("authoritative retained field" in error for error in errors), errors


POTENTIAL_ARRAY_PATHS: tuple[
    tuple[str, tuple[str, ...]], ...
] = (
    (
        "examples/physics_qg/chain_1d/results/validation.json",
        ("pipeline", "pi_time", "phi"),
    ),
    (
        "examples/physics_qg/grid_2d/results/validation.json",
        ("pipeline", "pi_time", "phi"),
    ),
    (
        "examples/physics_qg/gravity_well/results/validation.json",
        ("pipeline", "pi_time_natural", "phi"),
    ),
    (
        "examples/physics_qg/gravity_well/results/validation.json",
        ("pipeline", "gravity_test", "phi_point"),
    ),
)


@pytest.mark.negative_control
@pytest.mark.parametrize(("relative_path", "array_path"), POTENTIAL_ARRAY_PATHS)
def test_every_retained_potential_element_recomputes_summaries(
    relative_path: str,
    array_path: tuple[str, ...],
) -> None:
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    raw = reference
    for part in array_path:
        raw = raw[part]

    for index in range(len(raw)):
        candidate = json.loads(Path(relative_path).read_text(encoding="utf-8"))
        target = candidate
        for part in array_path:
            target = target[part]
        target[index] += 4e-9

        comparison = compare_validation_documents(reference, candidate)
        errors = check_validation_semantics(candidate, relative_path)

        assert comparison.passed, (relative_path, index, comparison.errors)
        assert any("recomputed raw potential" in error for error in errors), (
            relative_path,
            index,
            errors,
        )


@pytest.mark.negative_control
@pytest.mark.parametrize("transform", ["scale", "sign", "reverse", "roll", "shift"])
def test_grid_raw_potential_transformations_cannot_leave_stale_summaries(
    transform: str,
) -> None:
    relative_path = "examples/physics_qg/grid_2d/results/validation.json"
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    phi = candidate["pipeline"]["pi_time"]["phi"]
    if transform == "scale":
        mutated = [value * 1e7 for value in phi]
    elif transform == "sign":
        mutated = [-value for value in phi]
    elif transform == "reverse":
        mutated = list(reversed(phi))
    elif transform == "roll":
        mutated = [phi[-1], *phi[:-1]]
    else:
        mutated = [value + 4e-9 for value in phi]
    candidate["pipeline"]["pi_time"]["phi"] = mutated

    comparison = compare_validation_documents(reference, candidate)
    errors = check_validation_semantics(candidate, relative_path)

    assert comparison.passed, (transform, comparison.errors)
    assert any("recomputed raw potential" in error for error in errors), errors


@pytest.mark.negative_control
def test_potential_moment_tolerance_separates_honest_drift_from_attack() -> None:
    relative_path = "examples/physics_qg/chain_1d/results/validation.json"
    honest = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    honest["pipeline"]["pi_time"]["phi_index_moment"] += 5.49e-19
    attack = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    attack["pipeline"]["pi_time"]["phi_index_moment"] += 4.46e-18

    assert check_validation_semantics(honest, relative_path) == []
    assert any(
        "phi_index_moment differs from recomputed raw potential" in error
        for error in check_validation_semantics(attack, relative_path)
    )


@pytest.mark.negative_control
def test_grid_correlated_uniform_potential_shift_fails_absolute_gate() -> None:
    relative_path = "examples/physics_qg/grid_2d/results/validation.json"
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    pi_time = candidate["pipeline"]["pi_time"]
    phi = [value + 4e-9 for value in pi_time["phi"]]
    pi_time["phi"] = phi
    pi_time["phi_min"] = min(phi)
    pi_time["phi_max"] = max(phi)
    pi_time["phi_range"] = max(phi) - min(phi)
    pi_time["phi_mean"] = sum(phi) / len(phi)
    pi_time["max_absolute_phi"] = max(abs(value) for value in phi)
    pi_time["phi_index_moment"] = sum(
        (index + 1) * value for index, value in enumerate(phi)
    ) / sum(range(1, len(phi) + 1))
    placeholder = next(
        check
        for check in candidate["checks"]
        if check["name"] == "placeholder_source_degeneracy"
    )
    placeholder["value"]["phi_range"] = pi_time["phi_range"]
    placeholder["value"]["max_absolute_phi"] = pi_time["max_absolute_phi"]

    comparison = compare_validation_documents(reference, candidate)
    errors = check_validation_semantics(candidate, relative_path)

    assert comparison.passed, comparison.errors
    assert any("recompute to False" in error for error in errors), errors


@pytest.mark.negative_control
def test_grid_correlated_potential_scale_recomputes_solver_residual() -> None:
    relative_path = "examples/physics_qg/grid_2d/results/validation.json"
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    pi_time = candidate["pipeline"]["pi_time"]
    phi = [value * 1e7 for value in pi_time["phi"]]
    pi_time["phi"] = phi
    pi_time["phi_min"] = min(phi)
    pi_time["phi_max"] = max(phi)
    pi_time["phi_range"] = max(phi) - min(phi)
    pi_time["phi_mean"] = sum(phi) / len(phi)
    pi_time["max_absolute_phi"] = max(abs(value) for value in phi)
    pi_time["phi_index_moment"] = sum(
        (index + 1) * value for index, value in enumerate(phi)
    ) / sum(range(1, len(phi) + 1))
    placeholder = next(
        check
        for check in candidate["checks"]
        if check["name"] == "placeholder_source_degeneracy"
    )
    placeholder["value"]["phi_range"] = pi_time["phi_range"]
    placeholder["value"]["max_absolute_phi"] = pi_time["max_absolute_phi"]

    comparison = compare_validation_documents(reference, candidate)
    errors = check_validation_semantics(candidate, relative_path)

    assert comparison.passed, comparison.errors
    assert any(
        "constraint_residual differs from recomputed raw operands" in error
        for error in errors
    ), errors


@pytest.mark.negative_control
@pytest.mark.parametrize(
    "mutation",
    ["nonfinite_phi", "weight_shape", "source_shape", "negative_mu", "normalize_type"],
)
def test_clock_solver_raw_operands_are_fail_closed(mutation: str) -> None:
    relative_path = "examples/physics_qg/grid_2d/results/validation.json"
    candidate = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    pi_time = candidate["pipeline"]["pi_time"]
    if mutation == "nonfinite_phi":
        pi_time["phi"][0] = float("nan")
    elif mutation == "weight_shape":
        pi_time["weight_matrix"].pop()
    elif mutation == "source_shape":
        pi_time["effective_source"].pop()
    elif mutation == "negative_mu":
        pi_time["mu"] = -1.0
    elif mutation == "normalize_type":
        pi_time["normalize_potential"] = 1
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)

    errors = check_validation_semantics(candidate, relative_path)

    assert errors
    assert any(
        "retained potential operands are invalid" in error
        or "finite" in error
        for error in errors
    ), errors


_CLOCK_PROVENANCE_CASES = (
    ("chain_1d", "pi_time", "phi"),
    ("grid_2d", "pi_time", "phi"),
    ("gravity_well", "pi_time_natural", "phi"),
    ("gravity_well", "gravity_test", "phi_point"),
)


def _clock_case(example: str, clock_key: str) -> tuple[str, dict, dict]:
    relative_path = f"examples/physics_qg/{example}/results/validation.json"
    document = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    return relative_path, document, document["pipeline"][clock_key]


def _recomputed_clock_residual(clock: dict, phi_key: str) -> float:
    phi = clock[phi_key]
    source = clock["effective_source"]
    weights = clock["weight_matrix"]
    mu_squared = clock["mu"] ** 2
    residual = []
    for row, phi_row in enumerate(phi):
        operator_phi = mu_squared * phi_row
        for column, weight in enumerate(weights[row]):
            operator_phi += weight * (phi_row - phi[column])
        residual.append(operator_phi - source[row])
    return math.sqrt(sum(value * value for value in residual))


@pytest.mark.negative_control
@pytest.mark.parametrize("example,clock_key,phi_key", _CLOCK_PROVENANCE_CASES)
@pytest.mark.parametrize(
    "mutation",
    [
        "diagonal",
        "negative",
        "asymmetric",
        "new_edge",
        "correlated_matrix_source",
        "symmetric_correlated_matrix_source",
    ],
)
def test_clock_weight_matrix_is_bound_to_upstream_mi_graph(
    example: str,
    clock_key: str,
    phi_key: str,
    mutation: str,
) -> None:
    relative_path, candidate, clock = _clock_case(example, clock_key)
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    weights = clock["weight_matrix"]
    phi = clock[phi_key]
    if mutation == "diagonal":
        weights[0][0] += 4e-9
    elif mutation == "negative":
        weights[0][1] = -abs(weights[0][1])
    elif mutation == "asymmetric":
        weights[0][1] += 4e-9
    elif mutation == "new_edge":
        delta = 4e-9
        weights[0][2] += delta
        weights[2][0] += delta
        clock["effective_source"][0] += delta * (phi[0] - phi[2])
        clock["effective_source"][2] += delta * (phi[2] - phi[0])
        clock["constraint_residual"] = _recomputed_clock_residual(clock, phi_key)
    elif mutation == "correlated_matrix_source":
        delta = 4e-9
        weights[0][1] += delta
        clock["effective_source"][0] += delta * (phi[0] - phi[1])
        clock["constraint_residual"] = _recomputed_clock_residual(clock, phi_key)
    elif mutation == "symmetric_correlated_matrix_source":
        delta = 4e-9
        weights[0][1] += delta
        weights[1][0] += delta
        clock["effective_source"][0] += delta * (phi[0] - phi[1])
        clock["effective_source"][1] += delta * (phi[1] - phi[0])
        clock["constraint_residual"] = _recomputed_clock_residual(clock, phi_key)
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)

    errors = check_validation_semantics(candidate, relative_path)

    if mutation != "negative":
        assert compare_validation_documents(reference, candidate).passed
    assert any(
        "weight_matrix[" in error and "differs from recomputed raw operands" in error
        for error in errors
    ), (
        example,
        clock_key,
        mutation,
        errors,
    )


@pytest.mark.negative_control
@pytest.mark.parametrize("example,clock_key,phi_key", _CLOCK_PROVENANCE_CASES)
@pytest.mark.parametrize("mutation", ["mi_diagonal", "mi_asymmetry", "mi_weight", "edge_support"])
def test_clock_graph_provenance_is_fail_closed(
    example: str,
    clock_key: str,
    phi_key: str,
    mutation: str,
) -> None:
    del phi_key
    relative_path, candidate, _ = _clock_case(example, clock_key)
    pi_loc = candidate["pipeline"]["pi_loc"]
    if mutation == "mi_diagonal":
        pi_loc["mi_matrix"][0][0] += 4e-9
    elif mutation == "mi_asymmetry":
        pi_loc["mi_matrix"][0][1] += 4e-9
    elif mutation == "mi_weight":
        pi_loc["mi_matrix"][0][1] += 4e-9
        pi_loc["mi_matrix"][1][0] += 4e-9
    elif mutation == "edge_support":
        pi_loc["inferred_edges"].append([0, 2])
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)

    errors = check_validation_semantics(candidate, relative_path)

    assert errors, (example, clock_key, mutation)
    assert any(
        "mi_matrix" in error
        or "inferred_edges" in error
        or ("weight_matrix[" in error and "differs from recomputed raw operands" in error)
        for error in errors
    ), errors


@pytest.mark.negative_control
@pytest.mark.parametrize("example,clock_key,phi_key", _CLOCK_PROVENANCE_CASES)
@pytest.mark.parametrize(
    "mutation",
    [
        "background",
        "raw_source",
        "effective_source",
        "mu",
        "policy",
        "normalize",
        "nonzero_sum",
    ],
)
def test_clock_source_and_solver_policy_are_bound_to_configuration(
    example: str,
    clock_key: str,
    phi_key: str,
    mutation: str,
) -> None:
    relative_path, candidate, clock = _clock_case(example, clock_key)
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    if mutation == "background":
        clock["source_background"] += 4e-9
    elif mutation == "raw_source":
        clock["delta_rho_raw"][0] += 4e-9
    elif mutation == "effective_source":
        clock["effective_source"][0] += 4e-9
    elif mutation == "mu":
        old_mu = clock["mu"]
        clock["mu"] += 4e-9
        correction = clock["mu"] ** 2 - old_mu**2
        clock["effective_source"] = [
            source + correction * phi
            for source, phi in zip(clock["effective_source"], clock[phi_key])
        ]
        clock["constraint_residual"] = _recomputed_clock_residual(clock, phi_key)
    elif mutation == "policy":
        clock["zero_mode_policy"] = "require_zero_sum"
    elif mutation == "normalize":
        clock["normalize_potential"] = not clock["normalize_potential"]
    elif mutation == "nonzero_sum":
        clock["effective_source"] = [
            source + 1e-9 for source in clock["effective_source"]
        ]
        clock["constraint_residual"] = _recomputed_clock_residual(clock, phi_key)
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)

    errors = check_validation_semantics(candidate, relative_path)

    if mutation in {"background", "raw_source", "effective_source", "mu", "nonzero_sum"}:
        assert compare_validation_documents(reference, candidate).passed
    assert errors, (example, clock_key, mutation)
    assert any(
        "source_background" in error
        or "effective_source" in error
        or "delta_rho_raw" in error
        or ".mu differs from its authoritative" in error
        or ".zero_mode_policy differs from its authoritative" in error
        or ".normalize_potential differs from its authoritative" in error
        or "zero-sum invariant" in error
        for error in errors
    ), errors


@pytest.mark.negative_control
@pytest.mark.parametrize("mutation", ["raw_source", "point_strength", "center"])
def test_gravity_diagnostic_source_is_bound_to_center_and_strength(
    mutation: str,
) -> None:
    relative_path, candidate, gravity = _clock_case("gravity_well", "gravity_test")
    if mutation == "raw_source":
        gravity["delta_rho_raw"][candidate["config"]["center_cell"]] += 4e-9
    elif mutation == "point_strength":
        candidate["config"]["point_source_strength"] += 4e-9
    elif mutation == "center":
        candidate["config"]["center_cell"] = 0
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)

    errors = check_validation_semantics(candidate, relative_path)

    assert any(
        "pipeline.gravity_test.delta_rho_raw[" in error
        and "differs from recomputed raw operands" in error
        for error in errors
    ), errors


def _increment_path(document: dict, path: tuple[str | int, ...], delta: float) -> None:
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] += delta


@pytest.mark.negative_control
@pytest.mark.parametrize(
    "raw_key",
    [
        "relative_entropy",
        "modular_energy",
        "entropy_change",
        "relative_entropy_phi_amplitude",
        "modular_energy_phi_amplitude",
    ],
)
def test_source_law_raw_fit_and_identity_operands_recompute(raw_key: str) -> None:
    relative_path = "examples/physics_qg/source_law/results/validation.json"
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    for index in range(len(reference["measurements"][raw_key])):
        candidate = json.loads(Path(relative_path).read_text(encoding="utf-8"))
        candidate["measurements"][raw_key][index] += 4e-9

        comparison = compare_validation_documents(reference, candidate)
        errors = check_validation_semantics(candidate, relative_path)

        assert comparison.passed, (raw_key, index, comparison.errors)
        assert any("recomputed raw operands" in error for error in errors), (
            raw_key,
            index,
            errors,
        )


MANY_BODY_RAW_GATE_PATHS: tuple[tuple[str | int, ...], ...] = (
    ("measurements", "relative_entropy", 1),
    ("measurements", "modular_energy", 3),
    ("measurements", "entropy_change", 4),
    ("measurements", "total_energy_change", 0),
    ("measurements", "local_energy_profiles", 4, 0),
    ("measurements", "potential_amplitudes", 0),
    ("measurements", "evolved_total_energy", 1),
    ("measurements", "evolved_local_energy_profiles", 0, 0),
    (
        "measurements",
        "isospectral_unitary_control",
        "relative_entropy",
        0,
    ),
    ("measurements", "isospectral_unitary_control", "entropy_change", 0),
    ("measurements", "commuting_ising_control", "t1_profile", 0),
    (
        "measurements",
        "pipeline_source_comparison",
        "diagnostic_local_energy_profile",
        0,
    ),
)


@pytest.mark.negative_control
@pytest.mark.parametrize("raw_path", MANY_BODY_RAW_GATE_PATHS)
def test_many_body_raw_fit_identity_and_control_operands_recompute(
    raw_path: tuple[str | int, ...],
) -> None:
    relative_path = (
        "examples/physics_qg/source_law_many_body/results/validation.json"
    )
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    _increment_path(candidate, raw_path, 4e-9)

    comparison = compare_validation_documents(reference, candidate)
    errors = check_validation_semantics(candidate, relative_path)

    assert comparison.passed
    assert errors, raw_path
    assert any(
        "recomputed raw operands" in error or "recompute to False" in error
        for error in errors
    ), (raw_path, errors)


@pytest.mark.negative_control
def test_every_many_body_decision_bearing_raw_array_element_recomputes() -> None:
    relative_path = (
        "examples/physics_qg/source_law_many_body/results/validation.json"
    )
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    measurements = reference["measurements"]
    paths: list[tuple[str | int, ...]] = []
    for key in (
        "relative_entropy",
        "modular_energy",
        "entropy_change",
        "total_energy_change",
        "potential_amplitudes",
        "evolved_total_energy",
    ):
        paths.extend(("measurements", key, index) for index in range(len(measurements[key])))
    paths.extend(
        ("measurements", "local_energy_profiles", row, column)
        for row, profile in enumerate(measurements["local_energy_profiles"])
        for column in range(len(profile))
    )
    paths.extend(
        ("measurements", "evolved_local_energy_profiles", row, column)
        for row, profile in enumerate(measurements["evolved_local_energy_profiles"])
        for column in range(len(profile))
    )
    for key in ("relative_entropy", "modular_energy", "entropy_change"):
        paths.extend(
            ("measurements", "isospectral_unitary_control", key, index)
            for index in range(
                len(measurements["isospectral_unitary_control"][key])
            )
        )
    for key in ("initial_profile", "t1_profile"):
        paths.extend(
            ("measurements", "commuting_ising_control", key, index)
            for index in range(len(measurements["commuting_ising_control"][key]))
        )
    for key in (
        "diagnostic_local_energy_profile",
        "reduced_modular_source",
        "kms_energy_density_source",
    ):
        paths.extend(
            ("measurements", "pipeline_source_comparison", key, index)
            for index in range(len(measurements["pipeline_source_comparison"][key]))
        )

    for raw_path in paths:
        candidate = json.loads(Path(relative_path).read_text(encoding="utf-8"))
        _increment_path(candidate, raw_path, 4e-9)

        errors = check_validation_semantics(candidate, relative_path)

        assert errors, raw_path
        assert any(
            "recomputed raw operands" in error
            or "recompute to False" in error
            or "authoritative retained field" in error
            for error in errors
        ), (raw_path, errors)


@pytest.mark.negative_control
def test_many_body_correlated_global_energy_shift_recomputes_local_identity() -> None:
    relative_path = (
        "examples/physics_qg/source_law_many_body/results/validation.json"
    )
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    candidate["measurements"]["evolved_total_energy"] = [
        value + 4e-9
        for value in candidate["measurements"]["evolved_total_energy"]
    ]

    comparison = compare_validation_documents(reference, candidate)
    errors = check_validation_semantics(candidate, relative_path)

    assert comparison.passed, comparison.errors
    assert any("evolved_local_decomposition_error" in error for error in errors), errors


@pytest.mark.negative_control
@pytest.mark.parametrize(
    "source_key", ["reduced_modular_source", "kms_energy_density_source"]
)
def test_pipeline_source_paired_alias_mutation_recomputes(source_key: str) -> None:
    relative_path = (
        "examples/physics_qg/source_law_many_body/results/validation.json"
    )
    document = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    check = next(
        item
        for item in document["checks"]
        if item["name"]
        == "pipeline_reduced_modular_blindness_and_density_repair"
    )
    document["measurements"]["pipeline_source_comparison"][source_key][0] += 4e-9
    check["value"][source_key][0] += 4e-9

    errors = check_validation_semantics(document, relative_path)

    assert errors
    assert any(
        "differs from the retained" in error
        or "recomputed raw operands" in error
        or "recompute to False" in error
        for error in errors
    ), errors


@pytest.mark.negative_control
def test_every_sensitivity_raw_element_recomputes_paired_aliases() -> None:
    relative_path = (
        "examples/physics_qg/source_law_many_body/results/validation.json"
    )
    reference = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    reference_check = next(
        item
        for item in reference["checks"]
        if item["name"] == "nonaffine_kms_parameter_sensitivity"
    )
    for case_index, reference_case in enumerate(
        reference["measurements"]["order_sensitivity"]
    ):
        for raw_key in ("relative_entropy", "signed_modular_energy"):
            for response_index in range(len(reference_case[raw_key])):
                candidate = json.loads(
                    Path(relative_path).read_text(encoding="utf-8")
                )
                candidate_check = next(
                    item
                    for item in candidate["checks"]
                    if item["name"] == "nonaffine_kms_parameter_sensitivity"
                )
                candidate["measurements"]["order_sensitivity"][case_index][
                    raw_key
                ][response_index] += 4e-9
                candidate_check["value"][case_index][raw_key][response_index] += 4e-9

                comparison = compare_validation_documents(reference, candidate)
                errors = check_validation_semantics(candidate, relative_path)

                assert comparison.passed, (
                    case_index,
                    raw_key,
                    response_index,
                    comparison.errors,
                )
                assert errors, (case_index, raw_key, response_index)
                assert any(
                    "recomputed raw operands" in error
                    or "recompute to False" in error
                    for error in errors
                ), (case_index, raw_key, response_index, errors)
    assert len(reference_check["value"]) == 12


@pytest.mark.negative_control
@pytest.mark.parametrize(
    "mutation",
    ["nonfinite", "empty", "unordered_amplitudes", "zero_denominator"],
)
def test_raw_recomputation_is_fail_closed_on_invalid_operands(mutation: str) -> None:
    relative_path = "examples/physics_qg/source_law/results/validation.json"
    document = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    if mutation == "nonfinite":
        document["measurements"]["relative_entropy"][0] = float("nan")
    elif mutation == "empty":
        document["measurements"]["relative_entropy"] = []
    elif mutation == "unordered_amplitudes":
        document["config"]["epsilons"] = list(
            reversed(document["config"]["epsilons"])
        )
    elif mutation == "zero_denominator":
        document["measurements"]["modular_energy"][0] = 0.0
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(mutation)

    errors = check_validation_semantics(document, relative_path)

    assert errors
    assert any("invalid" in error or "finite" in error for error in errors), errors


@pytest.mark.negative_control
def test_many_body_precision_floor_recomputes_from_raw_response() -> None:
    relative_path = (
        "examples/physics_qg/source_law_many_body/results/validation.json"
    )
    document = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    check = next(
        item
        for item in document["checks"]
        if item["name"] == "nonaffine_kms_response_orders"
    )
    for assessment in (
        document["measurements"]["relative_entropy_quadratic_assessment"],
        check["value"]["relative_entropy_quadratic_assessment"],
    ):
        assessment["full_window"]["absolute_precision_floor"] = 1e-10
        assessment["lower_window"]["absolute_precision_floor"] = 1e-10

    errors = check_validation_semantics(document, relative_path)

    assert any("absolute_precision_floor" in error for error in errors), errors


@pytest.mark.negative_control
def test_many_body_consistent_but_insufficient_precision_floor_fails_gate() -> None:
    relative_path = (
        "examples/physics_qg/source_law_many_body/results/validation.json"
    )
    document = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    check = next(
        item
        for item in document["checks"]
        if item["name"] == "nonaffine_kms_response_orders"
    )
    floor = 1e-10
    ratio = min(abs(item) for item in document["measurements"]["relative_entropy"]) / floor
    document["measurements"]["absolute_precision_floor"] = floor
    for assessment in (
        document["measurements"]["relative_entropy_quadratic_assessment"],
        check["value"]["relative_entropy_quadratic_assessment"],
    ):
        for window in ("full_window", "lower_window"):
            assessment[window]["absolute_precision_floor"] = floor
            assessment[window]["minimum_signal_to_floor"] = ratio

    errors = check_validation_semantics(document, relative_path)

    assert any(
        "recomputed outcome is False" in error
        or "decision operands are invalid" in error
        for error in errors
    ), errors
    assert any(
        "recompute to False" in error or "decision operands are invalid" in error
        for error in errors
    ), errors


DECISION_MARGIN_CASES = [
    ("ca_model", "survivor_entropy_filter_regression"),
    ("chain_1d", "stability_selection"),
    ("chain_1d", "blind_edge_recovery"),
    ("chain_1d", "placeholder_clock_constraint_solved"),
    ("gravity_well", "nonzero_source_constraint_residual"),
    ("gravity_well", "monotonic_falloff"),
    ("gravity_well", "grid_symmetry"),
    ("gravity_well", "redshift_positive"),
    ("grid_2d", "blind_edge_recovery"),
    ("grid_2d", "topology_preservation"),
    ("grid_2d", "finite_graph_spectral_peak"),
    ("grid_2d", "mds_stress"),
    ("grid_2d", "placeholder_source_degeneracy"),
    ("source_law", "relative_entropy_is_quadratic"),
    ("source_law", "affine_modular_linearity_identity_regression"),
    ("source_law", "linear_solver_homogeneity_identity_regression"),
    ("source_law_many_body", "nonaffine_kms_response_orders"),
    ("source_law_many_body", "quadratic_gate_rejects_first_order_negative_control"),
    ("source_law_many_body", "kms_and_local_decomposition_identities"),
    (
        "source_law_many_body",
        "local_energy_decomposition_consistency_and_spreading",
    ),
    ("source_law_many_body", "nonaffine_kms_parameter_sensitivity"),
    ("source_law_many_body", "isospectral_unitary_identity_regression"),
    ("source_law_many_body", "spreading_requires_noncommuting_dynamics"),
    (
        "source_law_many_body",
        "pipeline_reduced_modular_blindness_and_density_repair",
    ),
    ("source_law_many_body", "negative_energy_candidate_has_slower_source_clock"),
]


def _cross_decision_margin(document: dict, check: dict) -> None:
    name = check["name"]
    value = check["value"]
    if name == "survivor_entropy_filter_regression":
        check["value"] = check["threshold"] * 2
    elif name == "stability_selection":
        value["invalid"] = value["valid"]
    elif name == "blind_edge_recovery":
        value["precision"] = 0.9995
    elif name == "placeholder_clock_constraint_solved":
        check["value"] = 1e-9
    elif name == "nonzero_source_constraint_residual":
        check["value"] = 1e-10
    elif name == "monotonic_falloff":
        value["d=1"] = value["d=0"]
    elif name == "grid_symmetry":
        check["value"] = check["threshold"]
    elif name == "redshift_positive":
        check["value"] = 0.0
    elif name == "topology_preservation":
        value["avg_dist_neighbors"] = value["avg_dist_non_neighbors"]
    elif name == "finite_graph_spectral_peak":
        check["value"] = 2.0005
    elif name == "mds_stress":
        check["value"] = 0.5
    elif name == "placeholder_source_degeneracy":
        value["phi_range"] = 4e-9
        document["pipeline"]["pi_time"]["phi_range"] = 4e-9
    elif name == "relative_entropy_is_quadratic":
        value["slope"] = 2.02
    elif name == "affine_modular_linearity_identity_regression":
        value["max_absolute_identity_error"] = 1e-10
    elif name == "linear_solver_homogeneity_identity_regression":
        value["max_ratio_spread"] = 1e-9
    elif name == "nonaffine_kms_response_orders":
        value["nonaffine_midpoint_deviation"] = 0.0
    elif name == "quadratic_gate_rejects_first_order_negative_control":
        assessment = value["assessment"]
        assessment["full_window"]["coefficient"] = 1.0
        assessment["relative_coefficient_difference"] = 0.0
        assessment["slope_deviation"] = 0.0
        assessment["full_window"]["normalized_rmse"] = 0.0
        assessment["lower_window"]["normalized_rmse"] = 0.0
    elif name == "kms_and_local_decomposition_identities":
        value["kms_identity_error"] = 1e-10
    elif name == "local_energy_decomposition_consistency_and_spreading":
        value["generator_observable_consistency_drift"] = 1e-10
    elif name == "nonaffine_kms_parameter_sensitivity":
        value[0]["quadratic_coefficient_relative_error"] = 1.0
    elif name == "isospectral_unitary_identity_regression":
        value["max_D_minus_modular_energy"] = (
            2 * value["dimension_scaled_float64_tolerance"]
        )
    elif name == "spreading_requires_noncommuting_dynamics":
        value["maximum_profile_change_at_t1"] = 1e-10
    elif name == "pipeline_reduced_modular_blindness_and_density_repair":
        value["kms_density_match_error"] = 1e-10
    elif name == "negative_energy_candidate_has_slower_source_clock":
        value["constraint_residual"] = 1e-10
    else:  # pragma: no cover - the parametrized matrix must stay exhaustive.
        raise AssertionError(f"missing decision-margin mutation for {name}")


@pytest.mark.negative_control
@pytest.mark.parametrize(("example", "check_name"), DECISION_MARGIN_CASES)
def test_every_float_decision_margin_is_recomputed(
    example: str,
    check_name: str,
) -> None:
    relative_path = f"examples/physics_qg/{example}/results/validation.json"
    document = json.loads(Path(relative_path).read_text(encoding="utf-8"))
    check = next(item for item in document["checks"] if item["name"] == check_name)
    original_passed = check["passed"]

    _cross_decision_margin(document, check)
    errors = check_validation_semantics(document, relative_path)

    assert check["passed"] is original_passed
    assert any(
        "recomputed outcome" in error or "recompute to" in error
        or "differs from recomputed raw operands" in error
        for error in errors
    ), errors


def test_headline_must_equal_noninformational_check_conjunction() -> None:
    document = _reference_document()
    document["checks"][0]["passed"] = False

    errors = check_validation_semantics(
        document,
        "examples/physics_qg/demo/results/validation.json",
    )

    assert any("overall_pass=True" in error for error in errors)


def test_failing_check_cannot_be_demoted_to_informational() -> None:
    document = _reference_document()
    document["checks"][0].update(
        {"name": "blind_edge_recovery", "severity": "informational", "passed": False}
    )
    document["overall_pass"] = True

    errors = check_validation_semantics(
        document,
        "examples/physics_qg/chain_1d/results/validation.json",
    )

    assert any("cannot be informational" in error for error in errors)


def test_unregistered_noninformational_check_is_rejected() -> None:
    root = Path(__file__).resolve().parents[2]
    relative_path = "examples/physics_qg/source_law/results/validation.json"
    document = json.loads((root / relative_path).read_text(encoding="utf-8"))
    document["checks"].append(
        {
            "name": "relative_entropy_fit_quality_floor",
            "criterion": "R squared exceeds 0.999",
            "value": 1.0,
            "passed": True,
        }
    )

    errors = check_validation_semantics(document, relative_path)

    assert any("unregistered" in error for error in errors)


def test_committed_validation_artifacts_are_internally_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = sorted(root.glob("examples/physics_qg/*/results/validation.json"))

    assert paths
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        document = json.loads(path.read_text(encoding="utf-8"))
        assert check_validation_semantics(document, relative_path) == []
