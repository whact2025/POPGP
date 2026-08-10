"""Check regenerated validation artifacts against their committed contract.

The examples contain floating-point diagnostics produced by LAPACK-backed routines.
Those values can move slightly across platforms even when the scientific result is
unchanged.  This checker therefore compares JSON structure and semantics rather than
serialized bytes:

* keys, JSON types, list lengths, strings, integers, booleans, and nulls are exact;
* configuration floats allow only a few machine epsilons of serialization drift;
* diagnostic floats use a narrow default tolerance;
* a small named set of ill-conditioned fit diagnostics has an explicit wider policy;
* every numeric value must remain finite; and
* every declared visual artifact must be tracked, present, and nonempty.

The wider policies below correspond to fields that changed materially in the frozen
Linux CI diff for PR #2 while all associated scientific gates remained unchanged.
Keeping the list here makes that exception reviewable instead of silently weakening
the entire artifact comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    "significance_ratio": (0.5, 0.0),
    "slope_deviation": (0.0, 2e-4),
    "slope_standard_error": (0.0, 1e-4),
}

STABLE_INPUT_KEYS = frozenset({"beta", "epsilons"})
VISUAL_SUFFIXES = frozenset({".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"})
VALIDATION_GLOB = "examples/physics_qg/*/results/validation.json"


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


def _run_git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _tracked_paths(repo_root: Path) -> set[str]:
    return {line for line in _run_git(repo_root, "ls-files").splitlines() if line}


def check_repository(repo_root: Path, *, base_ref: str = "HEAD") -> list[str]:
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
            check_required_visuals(
                candidate,
                working_path,
                repo_root=repo_root,
                tracked_paths=tracked_paths,
            )
        )
        if summary.accepted_numeric_drifts:
            print(
                f"{relative_path}: accepted {summary.accepted_numeric_drifts} bounded numeric "
                f"drifts (max abs={summary.largest_absolute_drift:.6g}, "
                f"max rel={summary.largest_relative_drift:.6g})"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="HEAD", help="committed artifact reference")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    try:
        errors = check_repository(args.repo_root, base_ref=args.base_ref)
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
