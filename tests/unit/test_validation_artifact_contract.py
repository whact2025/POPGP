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

    assert any("local" in error or "connected high-error" in error for error in errors)


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

    assert any("local" in error or "connected high-error" in error for error in errors)


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
    assert any("recompute to False" in error for error in semantic_errors)


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
