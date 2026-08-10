from __future__ import annotations

import math
from pathlib import Path

from scripts.check_validation_artifacts import (
    check_required_visuals,
    compare_validation_documents,
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
    candidate["checks"][0]["value"]["significance_ratio"] = 4_000_000_000.0

    summary = compare_validation_documents(_reference_document(), candidate)

    assert not summary.passed
    assert any("sensitive diagnostic 'significance_ratio'" in error for error in summary.errors)


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
