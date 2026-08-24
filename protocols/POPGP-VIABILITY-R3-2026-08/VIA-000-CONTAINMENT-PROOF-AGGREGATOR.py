"""Aggregate six non-scientific VIA-000 R3 hosted containment proof cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

PLATFORMS = ("ubuntu-latest-x86_64", "windows-x86_64")
STAGES = ("candidate", "pdf", "mutation")
EXPECTED_CELLS = {f"{platform}/{stage}" for platform in PLATFORMS for stage in STAGES}
WORKFLOW = ".github/workflows/via000-r3-containment-proof.yml"
TRUE_FIELDS = {
    "non_scientific",
    "receipt_bindings_verified",
    "descendants_quiescent",
    "os_process_tree_empty",
    "untrusted_identity_processes_empty",
    "untrusted_identity_retired",
    "child_of_child_observed_before_direct_exit",
    "protected_evidence_read_denied",
    "protected_evidence_write_denied",
    "protected_tool_write_denied",
    "replace_restore_denied",
    "hardlink_substitution_denied",
    "control_plane_environment_scrubbed",
    "delayed_descendant_write_absent",
    "closure_unchanged",
    "no_campaign_execution",
    "no_candidate_checkout",
    "no_lifecycle_mutation",
    "no_custody_access",
    "no_commitment_or_reveal",
}
HASH_FIELDS = {
    "production_helper_sha256",
    "proof_runner_sha256",
    "hostile_fixture_sha256",
    "proof_schema_sha256",
    "proof_aggregator_sha256",
    "proof_workflow_sha256",
    "protected_evidence_sha256",
    "protected_tool_sha256",
    "containment_result_sha256",
}
BUNDLE_HASH_FIELDS = {
    "production_helper_sha256",
    "proof_runner_sha256",
    "hostile_fixture_sha256",
    "proof_schema_sha256",
    "proof_aggregator_sha256",
    "proof_workflow_sha256",
}
CELL_FIELDS = {
    "schema_version",
    "proof_kind",
    "non_scientific",
    "repository",
    "workflow",
    "workflow_ref",
    "event_name",
    "source_ref",
    "source_sha",
    "run_id",
    "run_attempt",
    "platform_family",
    "stage_id",
    "cell",
    "artifact_name",
    *HASH_FIELDS,
    *TRUE_FIELDS,
    "primitive",
    "privilege_separation",
    "active_processes_after_teardown",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"proof is not one JSON object: {path}")
    return document


def _validate_cell(document: dict[str, Any], path: Path, identity: dict[str, str]) -> str:
    expected_files = {"proof.json", "containment-result.json", "stdout.txt", "stderr.txt"}
    observed_files = {item.name for item in path.parent.iterdir() if item.is_file()}
    if observed_files != expected_files:
        raise ValueError(f"proof artifact file set differs: {path.parent}")
    if set(document) != CELL_FIELDS:
        raise ValueError(f"proof fields differ from frozen cell schema: {path}")
    if document["schema_version"] != 1 or document["proof_kind"] != (
        "via000-r3-hosted-containment-cell"
    ):
        raise ValueError(f"proof kind/version differs: {path}")
    for field, expected in identity.items():
        if document[field] != expected:
            raise ValueError(f"proof {field} differs across cells: {path}")
    platform = document["platform_family"]
    stage = document["stage_id"]
    if platform not in PLATFORMS or stage not in STAGES:
        raise ValueError(f"proof platform/stage is outside the 2x3 matrix: {path}")
    cell = f"{platform}/{stage}"
    if document["cell"] != cell:
        raise ValueError(f"proof cell identity differs: {path}")
    artifact = f"via000-r3-containment-proof-{platform}-{stage}"
    if document["artifact_name"] != artifact or path.parent.name != artifact:
        raise ValueError(f"proof artifact directory differs from its cell: {path}")
    if any(document[field] is not True for field in TRUE_FIELDS):
        raise ValueError(f"proof contains a false security predicate: {path}")
    if document["active_processes_after_teardown"] != 0:
        raise ValueError(f"proof retains an active descendant: {path}")
    expected_primitive = (
        "windows-low-integrity-restricted-token-job-object"
        if platform == "windows-x86_64"
        else "ubuntu-systemd-ephemeral-user-control-group"
    )
    expected_privilege = (
        "low-integrity-restricted-token"
        if platform == "windows-x86_64"
        else "systemd-ephemeral-user"
    )
    if (
        document["primitive"] != expected_primitive
        or document["privilege_separation"] != expected_privilege
    ):
        raise ValueError(f"proof uses the wrong platform containment primitive: {path}")
    if any(
        not isinstance(document[field], str)
        or re.fullmatch(r"[0-9a-f]{64}", document[field]) is None
        for field in HASH_FIELDS
    ):
        raise ValueError(f"proof has a malformed SHA-256 binding: {path}")
    contained_path = path.parent / "containment-result.json"
    if _sha256(contained_path) != document["containment_result_sha256"]:
        raise ValueError(f"containment result hash differs: {path}")
    contained = _load(contained_path)
    if (
        contained.get("schema_version") != 1
        or contained.get("label") != f"proof-{stage}"
        or contained.get("contract_id") != "rr7-hosted-containment-proof"
        or contained.get("primitive") != expected_primitive
        or contained.get("privilege_separation") != expected_privilege
        or contained.get("descendants_quiescent") is not True
        or contained.get("active_processes_after_teardown") != 0
        or contained.get("exit_code") != 0
        or contained.get("timed_out") is not False
    ):
        raise ValueError(f"containment result predicates differ: {path}")
    if platform == "ubuntu-latest-x86_64" and (
        re.fullmatch(r"[1-9][0-9]*", contained.get("ephemeral_identity_uid", "")) is None
        or contained.get("ephemeral_identity_processes_empty") is not True
        or contained.get("ephemeral_identity_removed") is not True
    ):
        raise ValueError(f"ephemeral Ubuntu identity proof differs: {path}")
    if contained.get("stdout_sha256") != _sha256(path.parent / "stdout.txt") or contained.get(
        "stderr_sha256"
    ) != _sha256(path.parent / "stderr.txt"):
        raise ValueError(f"containment transcript hash differs: {path}")
    return cell


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", args.source_sha) is None:
        raise ValueError("aggregate source SHA is not one full commit")
    if re.fullmatch(r"[1-9][0-9]*", args.run_id) is None or re.fullmatch(
        r"[1-9][0-9]*", args.run_attempt
    ) is None:
        raise ValueError("aggregate run identity is malformed")
    if re.fullmatch(
        r"refs/heads/(?:campaign|review)/via000-r3-protocol-[A-Za-z0-9._/-]+",
        args.source_ref,
    ) is None:
        raise ValueError("aggregate source ref is outside the proof-only branch scope")
    identity = {
        "repository": "whact2025/POPGP",
        "workflow": WORKFLOW,
        "workflow_ref": args.workflow_ref,
        "event_name": "push",
        "source_ref": args.source_ref,
        "source_sha": args.source_sha,
        "run_id": args.run_id,
        "run_attempt": args.run_attempt,
    }
    paths = sorted(args.input_root.glob("*/proof.json"))
    if len(paths) != 6:
        raise ValueError(f"expected exactly six hosted proof fragments, observed {len(paths)}")
    cells: set[str] = set()
    hashes: dict[str, str] = {}
    bundle_hashes: dict[str, str] | None = None
    for path in paths:
        document = _load(path)
        cell = _validate_cell(document, path, identity)
        if cell in cells:
            raise ValueError(f"duplicate hosted containment proof cell: {cell}")
        observed_bundle = {field: document[field] for field in BUNDLE_HASH_FIELDS}
        if bundle_hashes is None:
            bundle_hashes = observed_bundle
        elif observed_bundle != bundle_hashes:
            raise ValueError(f"hosted containment proof bundle hashes differ: {cell}")
        cells.add(cell)
        hashes[cell] = _sha256(path)
    if cells != EXPECTED_CELLS:
        raise ValueError(f"hosted containment proof matrix differs: {sorted(cells)}")
    aggregate_document: dict[str, Any] = {
        "schema_version": 1,
        "proof_kind": "via000-r3-hosted-containment-aggregate",
        "non_scientific": True,
        **identity,
        "cell_count": 6,
        "cells": sorted(cells),
        "fragment_sha256": {cell: hashes[cell] for cell in sorted(hashes)},
        "bundle_sha256": {
            field: bundle_hashes[field] for field in sorted(bundle_hashes or {})
        },
        "all_cells_passed": True,
        "no_campaign_execution": True,
        "no_candidate_checkout": True,
        "no_lifecycle_mutation": True,
        "no_custody_access": True,
        "no_commitment_or_reveal": True,
    }
    _atomic_write(args.output, aggregate_document)
    return aggregate_document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--workflow-ref", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    args = parser.parse_args()
    try:
        aggregate(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"containment proof aggregation failed: {exc}", file=__import__("sys").stderr)
        args.output.unlink(missing_ok=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
