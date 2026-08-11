from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

from scripts.check_viability_campaign import (
    packet_rule_sha256,
    validate_campaign,
    validate_requirements,
)

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = json.loads(
    (ROOT / "schemas/viability/requirements-v2.json").read_text(encoding="utf-8")
)

CONTRACT_PATHS = (
    "docs/scientific_hardening/CLAIMS_MATRIX.md",
    "docs/scientific_hardening/GATE_TEST_REGISTRY.md",
    "schemas/viability/campaign-v2.schema.json",
    "schemas/viability/packet-v2.schema.json",
    "schemas/viability/protocol-manifest-v2.schema.json",
    "schemas/viability/independent-review-v1.schema.json",
    "schemas/viability/independent-rereview-v1.schema.json",
    "schemas/viability/review-response-v1.schema.json",
    "scripts/check_viability_campaign.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, document: Any, *, indent: int | None = None) -> None:
    path.write_bytes(
        (json.dumps(document, indent=indent, sort_keys=True) + "\n").encode("utf-8")
    )


def _seat(name: str, *, evaluator: bool = False) -> dict[str, Any]:
    return {
        "agent_identity": f"agent-{name}",
        "model_identity": "test-model",
        "model_version": "1",
        "operator": "test-operator",
        "session_id": f"session-{name}",
        "access_level": "test",
        "exposure": {
            "final_labels_seen": evaluator,
            "secret_seed_seen": evaluator,
            "private_evaluator_seen": evaluator,
        },
    }


def _receipt(
    receipt_id: str, kind: str, path: str, digest: str, media_type: str = "application/json"
) -> dict[str, str]:
    return {
        "id": receipt_id,
        "kind": kind,
        "path": path,
        "sha256": digest,
        "media_type": media_type,
    }


def _initial_review_document(packet_id: str, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "review_id": f"REVIEW-{packet_id}-1",
        "review_kind": "initial",
        "reviewer_seat": "independent-reviewer",
        "reviewer_model_identity": "test-reviewer-model",
        "reviewer_model_version": "1",
        "reviewer_operator": "test-reviewer-operator",
        "review_date": "2026-08-10",
        "commit_reviewed": hashes["candidate_commit"],
        "baseline_commit": hashes["baseline_commit"],
        "prior_review_ref": "",
        "builder_response_ref": "",
        "context_hash": hashes["tree_hash"],
        "context_hash_method": (
            f"git rev-parse \"{hashes['candidate_commit']}^{{tree}}\""
        ),
        "files_reviewed": ["candidate.txt"],
        "access_level": "test-repository-only",
        "independence_statement": "Fresh test reviewer session with disclosed access.",
        "independence_declaration": {
            "shared_operator": False,
            "shared_session": False,
            "shared_orchestrator": False,
            "builder_model_identity": "test-builder-model",
            "reviewer_model_differs_from_builder": True,
            "external_scientific_validation": False,
        },
        "hidden_access_declaration": {
            "final_labels_seen": False,
            "secret_seed_seen": False,
            "private_evaluator_seen": False,
        },
        "summary": "No blocking contract finding in the positive fixture.",
        "findings": [],
        "requested_tests": [],
        "prior_finding_results": [],
        "prior_requested_test_results": [],
        "predictions": {
            "experiment_id": "",
            "predicted_outcome": "",
            "predicted_failure_mode": "",
            "confidence_statement": "No prediction registered for this fixture.",
        },
        "recommendation": {
            "approve": True,
            "blocking_findings": 0,
            "rationale": "The fixture review has no unresolved blockers.",
        },
    }


def _protocol_document(packet_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "packet_id": packet_id,
        "parameters": {"threshold": 0.95, "replicates": 3},
        "measurement_procedure": "Measure the preregistered raw Boolean gates.",
        "uncertainty_procedure": "Report the preregistered interval and failure rate.",
        "statistical_analysis": "Apply the frozen threshold without reselection.",
        "resource_budget": {
            "wall_time_seconds": 3600,
            "memory_bytes": 1073741824,
            "accelerator_seconds": 0,
        },
        "commands": ["uv run python -m popgp.viability_fixture"],
        "mutation_plan": ["invert each required Boolean gate"],
    }


def _make_packet(
    packet_id: str,
    campaign_id: str,
    receipt_dir: Path,
    holdout_path: Path,
    seed_path: Path,
    hashes: dict[str, str],
    protocol_artifact: dict[str, str],
    initial_review_ref: str,
) -> dict[str, Any]:
    requirement = REQUIREMENTS["packets"][packet_id]
    declared_evidence = (
        "E4-convergent-replication"
        if packet_id == "VIA-000"
        else requirement["minimum_evidence"]
    )
    receipt_dir.mkdir(parents=True, exist_ok=True)
    generic_path = receipt_dir / "evidence.json"
    raw_path = receipt_dir / "raw-results.json"
    review_path = receipt_dir / "independent-review.json"
    protocol_path = receipt_dir / "protocol.json"
    output_commitment_path = receipt_dir / "output-commitment.json"
    reveal_record_path = receipt_dir / "reveal-record.json"
    generic_path.write_text('{"evidence": true}\n', encoding="utf-8")
    _write_json(protocol_path, _protocol_document(packet_id))
    raw_document = {
        "metric": 1,
        "passed": True,
        "failed": False,
        "blocked": False,
        "capabilities": {
            capability: True for capability in requirement["required_capabilities"]
        },
    }
    raw_path.write_text(json.dumps(raw_document, sort_keys=True) + "\n", encoding="utf-8")
    review_document = _initial_review_document(packet_id, hashes)
    _write_json(review_path, review_document)
    raw_hash = _sha256(raw_path)
    runner_identity = f"agent-{packet_id}-runner"
    custodian_identity = f"agent-{packet_id}-custodian"
    output_commitment_document = {
        "packet_id": packet_id,
        "committed_by": runner_identity,
        "committed_at": "2026-08-10T12:00:00Z",
        "output_receipt_id": "raw-results",
        "output_sha256": raw_hash,
    }
    output_commitment_path.write_text(
        json.dumps(output_commitment_document, sort_keys=True) + "\n", encoding="utf-8"
    )
    holdout_hash = _sha256(holdout_path)
    seed_hash = _sha256(seed_path)
    reveal_document = {
        "packet_id": packet_id,
        "authorized_by": custodian_identity,
        "revealed_at": "2026-08-10T13:00:00Z",
        "output_commitment_receipt_id": "output-commitment",
        "post_reveal_holdout_sha256": holdout_hash,
        "post_reveal_seed_sha256": seed_hash,
    }
    reveal_record_path.write_text(
        json.dumps(reveal_document, sort_keys=True) + "\n", encoding="utf-8"
    )
    relative_prefix = f"../receipts/{packet_id}"
    evidence_kinds = sorted(
        {
            kind
            for kinds in REQUIREMENTS["required_receipt_kinds"].values()
            for kind in kinds
        }
    )
    receipts = []
    for kind in evidence_kinds:
        if kind == "raw-results":
            path = raw_path
            relative = f"{relative_prefix}/raw-results.json"
        elif kind == "independent-review":
            path = review_path
            relative = f"{relative_prefix}/independent-review.json"
        elif kind == "protocol":
            path = protocol_path
            relative = f"{relative_prefix}/protocol.json"
        elif kind == "reveal-record":
            path = reveal_record_path
            relative = f"{relative_prefix}/reveal-record.json"
        else:
            path = generic_path
            relative = f"{relative_prefix}/evidence.json"
        receipts.append(_receipt(kind, kind, relative, _sha256(path)))
    receipts.extend(
        [
            _receipt(
                "output-commitment",
                "output-commitment",
                f"{relative_prefix}/output-commitment.json",
                _sha256(output_commitment_path),
            ),
            _receipt(
                "blockage-evidence",
                "blockage-evidence",
                f"{relative_prefix}/evidence.json",
                _sha256(generic_path),
            ),
            _receipt(
                "revealed-holdout",
                "revealed-manifest",
                "../receipts/holdout.json",
                holdout_hash,
            ),
            _receipt(
                "revealed-seed",
                "revealed-manifest",
                "../receipts/seed.json",
                seed_hash,
            ),
        ]
    )
    bindings = {
        "result_passed": {
            "receipt_id": "raw-results",
            "json_pointer": "/passed",
            "expected_type": "boolean",
        },
        "result_blocked": {
            "receipt_id": "raw-results",
            "json_pointer": "/blocked",
            "expected_type": "boolean",
        },
        "result_failed": {
            "receipt_id": "raw-results",
            "json_pointer": "/failed",
            "expected_type": "boolean",
        },
    }
    bindings.update(
        {
            capability: {
                "receipt_id": "raw-results",
                "json_pointer": f"/capabilities/{capability}",
                "expected_type": "boolean",
            }
            for capability in requirement["required_capabilities"]
        }
    )
    packet = {
        "schema_version": 2,
        "contract_version": "popgp-viability-contract-v2",
        "campaign_id": campaign_id,
        "packet_id": packet_id,
        **hashes,
        "protocol_rule_sha256": "0" * 64,
        "lifecycle_phase": "adjudicated",
        "claims": ["C01"],
        "existing_gates": [],
        "declared_evidence_requirement": declared_evidence,
        "capabilities": requirement["required_capabilities"],
        "capability_rules": {
            capability: {
                "compare": {
                    "left": {"binding": capability},
                    "op": "eq",
                    "right": {"value": True},
                }
            }
            for capability in requirement["required_capabilities"]
        },
        "hypothesis": "The frozen packet satisfies its preregistered gate.",
        "null_or_competitors": ["The gate does not discriminate the candidate."],
        "known_failure_to_retain": "none",
        "threat_model": ["post-selection", "hidden-label leakage"],
        "preregistration": {
            "parameters": {"threshold": 0.95, "replicates": 3},
            "measurement_procedure": "Measure the preregistered raw Boolean gates.",
            "uncertainty_procedure": (
                "Report the preregistered interval and failure rate."
            ),
            "statistical_analysis": "Apply the frozen threshold without reselection.",
            "resource_budget": {
                "wall_time_seconds": 3600,
                "memory_bytes": 1073741824,
                "accelerator_seconds": 0,
            },
            "commands": ["uv run python -m popgp.viability_fixture"],
            "mutation_plan": ["invert each required Boolean gate"],
            "protocol_artifacts": [
                {
                    "receipt_id": "protocol",
                    "content_role": "primary-protocol",
                    "campaign_path": f"{relative_prefix}/protocol.json",
                    "protocol_path": protocol_artifact["path"],
                    "sha256": protocol_artifact["sha256"],
                    "media_type": "application/json",
                }
            ],
        },
        "holdout_started": True,
        "seats": {
            "protocol_designer": _seat(f"{packet_id}-protocol"),
            "builder": _seat(f"{packet_id}-builder"),
            "falsifier": _seat(f"{packet_id}-falsifier"),
            "statistical_auditor": _seat(f"{packet_id}-statistics"),
            "reproduction_runner": _seat(f"{packet_id}-runner"),
            "claim_auditor": _seat(f"{packet_id}-claims"),
            "adjudicator": _seat(f"{packet_id}-adjudicator"),
            "evaluator_custodian": _seat(f"{packet_id}-custodian", evaluator=True),
        },
        "blind_custody": {
            "hash_algorithm": "sha256",
            "canonicalization": "raw-bytes-v1",
            "custodian_seat": "evaluator_custodian",
            "hidden_holdout_manifest": {
                "immutable_uri": "custody://holdout",
                "sha256": holdout_hash,
            },
            "secret_seed_manifest": {
                "immutable_uri": "custody://seed",
                "sha256": seed_hash,
            },
            "output_commitment": {
                "receipt_id": "output-commitment",
                "committed_by": runner_identity,
                "committed_at": "2026-08-10T12:00:00Z",
                "output_receipt_id": "raw-results",
                "output_sha256": raw_hash,
            },
            "reveal": {
                "status": "revealed",
                "authorized_by": custodian_identity,
                "revealed_at": "2026-08-10T13:00:00Z",
                "post_reveal_holdout_sha256": holdout_hash,
                "post_reveal_seed_sha256": seed_hash,
                "post_reveal_holdout_receipt_id": "revealed-holdout",
                "post_reveal_seed_receipt_id": "revealed-seed",
                "reveal_receipt_id": "reveal-record",
                "retention_policy": "retain immutable manifests with campaign receipts",
                "immutable_location": "archive://campaign/manifests",
            },
        },
        "outcome_rules": {
            "language": "popgp-bool-v2",
            "bindings": bindings,
            "pass": {
                "compare": {
                    "left": {"binding": "result_passed"},
                    "op": "eq",
                    "right": {"value": True},
                }
            },
            "fail": {
                "compare": {
                    "left": {"binding": "result_failed"},
                    "op": "eq",
                    "right": {"value": True},
                }
            },
            "blocked": {
                "compare": {
                    "left": {"binding": "result_blocked"},
                    "op": "eq",
                    "right": {"value": True},
                }
            },
        },
        "receipts": receipts,
        "review_chain": {
            "initial_review_receipt": "independent-review",
            "initial_review_ref": initial_review_ref,
            "response_receipts": [],
            "response_refs": [],
            "rereview_receipts": [],
            "rereview_refs": [],
            "findings": [],
            "requested_tests": [],
        },
        "adjudication": {
            "round_status": "valid",
            "packet_outcome": "passed",
            "cause_codes": ["pass-rule-satisfied"],
            "decisive_receipts": ["raw-results"],
            "achieved_evidence": declared_evidence,
        },
    }
    packet["protocol_rule_sha256"] = packet_rule_sha256(packet)
    return packet


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def _commit_json_artifact(
    root: Path, relative: str, document: dict[str, Any]
) -> tuple[str, bytes]:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, document)
    _git(root, "add", relative)
    _git(root, "commit", "-q", "-m", f"add {Path(relative).name}")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    return f"{commit}:{relative}", destination.read_bytes()


def _finding(finding_id: str = "F-1") -> dict[str, Any]:
    return {
        "id": finding_id,
        "severity": "high",
        "category": "governance",
        "location": "scripts/check_viability_campaign.py",
        "evidence": "The adversarial fixture reproduced the finding.",
        "finding": "A blocking fixture finding.",
        "failure_scenario": "The contract accepts an invalid state.",
        "consequence": "The campaign could pass incorrectly.",
        "required_action": "Reject the invalid state.",
        "verification": "confirmed-by-execution",
        "blocking": True,
    }


def _requested_test(test_id: str = "T-1") -> dict[str, Any]:
    return {
        "id": test_id,
        "description": "Run the adversarial contract mutation.",
        "rationale": "The mutation must fail closed.",
        "blocking": True,
    }


def _response_document(
    review: dict[str, Any], review_ref: str, *, include_test: bool = False
) -> dict[str, Any]:
    review_commit, review_artifact = review_ref.split(":", 1)
    return {
        "artifact_schema_version": 1,
        "response_id": "RESPONSE-VIA-000-1",
        "response_round": 1,
        "response_date": "2026-08-10",
        "builder_seat": "builder",
        "builder_model_identity": "test-builder-model",
        "builder_model_version": "1",
        "builder_operator": "test-builder-operator",
        "review_id": review["review_id"],
        "review_artifact": review_artifact,
        "review_commit": review_commit,
        "candidate_commit_reviewed": review["commit_reviewed"],
        "access_declaration": {
            "final_labels_seen": False,
            "secret_seed_seen": False,
            "private_evaluator_seen": False,
            "notes": "No hidden fixture inputs were exposed.",
        },
        "summary": "The fixture finding was remediated.",
        "finding_responses": [
            {
                "finding_id": "F-1",
                "blocking_as_reported": True,
                "disposition": "accepted",
                "implementation_status": "implemented",
                "rationale": "The mutation is valid and requires a fix.",
                "changed_files": ["scripts/check_viability_campaign.py"],
                "fix_commits": [],
                "verification": [{"command": "uv run pytest", "result": "passed"}],
                "residual_risk": "Synthetic fixture only.",
                "disagreement_ref": "",
            }
        ],
        "requested_test_responses": (
            [
                {
                    "requested_test_id": "T-1",
                    "disposition": "accepted",
                    "implementation_status": "implemented",
                    "test_locations": ["tests/unit/test_viability_campaign_contract.py"],
                    "verification": [
                        {"command": "uv run pytest", "result": "passed"}
                    ],
                    "rationale": "The regression is implemented.",
                    "disagreement_ref": "",
                }
            ]
            if include_test
            else []
        ),
        "new_or_changed_risks": [],
        "external_actions": [],
        "rereview_request": {
            "requested": True,
            "scope": "all findings, requested tests, regressions, and new findings",
            "handoff_commit": "recorded outside this artifact after it is committed",
            "notes": "Review the exact candidate handoff.",
        },
    }


def _rereview_document(
    hashes: dict[str, str],
    review_ref: str,
    response_ref: str,
    *,
    finding_outcome: Any = "verified-resolved",
    superseding_finding_id: str = "",
    include_test: bool = False,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "review_id": "REREVIEW-VIA-000-1",
        "review_kind": "re-review",
        "reviewer_seat": "independent-reviewer",
        "reviewer_model_identity": "test-reviewer-model",
        "reviewer_model_version": "1",
        "reviewer_operator": "test-reviewer-operator",
        "review_date": "2026-08-10",
        "commit_reviewed": hashes["candidate_commit"],
        "baseline_commit": hashes["baseline_commit"],
        "prior_review_ref": review_ref,
        "builder_response_ref": response_ref,
        "context_hash": hashes["tree_hash"],
        "context_hash_method": (
            f"git rev-parse \"{hashes['candidate_commit']}^{{tree}}\""
        ),
        "files_reviewed": ["scripts/check_viability_campaign.py"],
        "access_level": "test-repository-only",
        "independence_statement": "Fresh test re-review with disclosed access.",
        "independence_declaration": {
            "shared_operator": False,
            "shared_session": False,
            "shared_orchestrator": False,
            "builder_model_identity": "test-builder-model",
            "reviewer_model_differs_from_builder": True,
            "external_scientific_validation": False,
        },
        "hidden_access_declaration": {
            "final_labels_seen": False,
            "secret_seed_seen": False,
            "private_evaluator_seen": False,
        },
        "summary": "The fixture remediation was independently checked.",
        "findings": [],
        "requested_tests": [],
        "prior_finding_results": [
            {
                "finding_id": "F-1",
                "outcome": finding_outcome,
                "evidence": "The targeted mutation was independently rerun.",
                "verification": "confirmed-by-execution",
                "superseding_finding_id": superseding_finding_id,
                "notes": "Synthetic fixture result.",
            }
        ],
        "prior_requested_test_results": (
            [
                {
                    "requested_test_id": "T-1",
                    "outcome": "verified-satisfied",
                    "evidence": "The regression failed before and passes after remediation.",
                    "verification": "confirmed-by-execution",
                    "superseding_requested_test_id": "",
                    "notes": "Synthetic fixture result.",
                }
            ]
            if include_test
            else []
        ),
        "predictions": {
            "experiment_id": "T-1" if include_test else "",
            "predicted_outcome": "The invalid campaign is rejected.",
            "predicted_failure_mode": "A missing binding would remain accepted.",
            "confidence_statement": "High confidence in the fixture result.",
        },
        "recommendation": {
            "approve": True,
            "blocking_findings": 0,
            "rationale": "No unresolved blocker is declared by this fixture artifact.",
        },
    }


def _build_frozen_repo(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "repo"
    root.mkdir()
    for relative in (*CONTRACT_PATHS, "schemas/viability/requirements-v2.json"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "POPGP Test")
    _git(root, "config", "user.email", "popgp-test@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "baseline contract")
    baseline = _git(root, "rev-parse", "HEAD").decode().strip()
    (root / "candidate.txt").write_text("frozen candidate\n", encoding="utf-8")
    _git(root, "add", "candidate.txt")
    _git(root, "commit", "-q", "-m", "candidate")
    candidate = _git(root, "rev-parse", "HEAD").decode().strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    hashes = {
        "candidate_commit": candidate,
        "baseline_commit": baseline,
        "tree_hash": tree,
        "protocol_commit": "0" * 40,
    }

    review_paths: dict[str, str] = {}
    for packet_id in REQUIREMENTS["packets"]:
        relative = f"reviews/fixtures/{packet_id}-independent-review.json"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json(destination, _initial_review_document(packet_id, hashes))
        review_paths[packet_id] = relative
    _git(root, "add", "reviews/fixtures")
    _git(root, "commit", "-q", "-m", "add fixture independent reviews")
    review_commit = _git(root, "rev-parse", "HEAD").decode().strip()
    initial_review_refs = {
        packet_id: f"{review_commit}:{relative}"
        for packet_id, relative in review_paths.items()
    }

    protocol_artifacts: dict[str, dict[str, str]] = {}
    for packet_id in REQUIREMENTS["packets"]:
        relative = f"protocols/POPGP-VIABILITY-TEST/{packet_id}.json"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json(destination, _protocol_document(packet_id))
        protocol_artifacts[packet_id] = {
            "path": relative,
            "sha256": _sha256(destination),
        }

    scratch = tmp_path / "manifest-scratch"
    scratch.mkdir()
    holdout_path = scratch / "holdout.json"
    seed_path = scratch / "seed.json"
    holdout_path.write_text('{"labels": [0, 1]}\n', encoding="utf-8")
    seed_path.write_text('{"seeds": [17, 29]}\n', encoding="utf-8")
    packet_hashes = {}
    for packet_id in REQUIREMENTS["packets"]:
        packet = _make_packet(
            packet_id,
            "POPGP-VIABILITY-TEST",
            scratch / packet_id,
            holdout_path,
            seed_path,
            hashes,
            protocol_artifacts[packet_id],
            initial_review_refs[packet_id],
        )
        packet_hashes[packet_id] = packet["protocol_rule_sha256"]

    requirements_path = "schemas/viability/requirements-v2.json"
    requirements_bytes = _git(root, "show", f"HEAD:{requirements_path}")
    manifest = {
        "schema_version": 2,
        "contract_version": "popgp-viability-contract-v2",
        "requirements_version": "popgp-viability-requirements-v2",
        "campaign_id": "POPGP-VIABILITY-TEST",
        "candidate_commit": candidate,
        "baseline_commit": baseline,
        "tree_hash": tree,
        "packet_freeze_version": "popgp-packet-freeze-v2",
        "requirements": {
            "path": requirements_path,
            "sha256": hashlib.sha256(requirements_bytes).hexdigest(),
        },
        "contract_files": [
            {
                "path": relative,
                "sha256": hashlib.sha256(_git(root, "show", f"HEAD:{relative}")).hexdigest(),
            }
            for relative in CONTRACT_PATHS
        ],
        "packet_rule_sha256": packet_hashes,
    }
    manifest_path = root / "protocols/POPGP-VIABILITY-TEST.json"
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(root, "add", "protocols")
    _git(root, "commit", "-q", "-m", "freeze test protocol")
    protocol = _git(root, "rev-parse", "HEAD").decode().strip()
    manifest_bytes = _git(root, "show", "HEAD:protocols/POPGP-VIABILITY-TEST.json")
    return {
        "root": root,
        "candidate_commit": candidate,
        "baseline_commit": baseline,
        "tree_hash": tree,
        "protocol_commit": protocol,
        "requirements_path": requirements_path,
        "requirements_sha256": hashlib.sha256(requirements_bytes).hexdigest(),
        "protocol_manifest_path": "protocols/POPGP-VIABILITY-TEST.json",
        "protocol_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "protocol_artifacts": protocol_artifacts,
        "initial_review_refs": initial_review_refs,
    }


@pytest.fixture(scope="module")
def frozen_repo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _build_frozen_repo(tmp_path_factory.mktemp("viability-frozen-repo"))


def _make_campaign(
    tmp_path: Path, frozen: dict[str, Any], target_tier: str = "R"
) -> tuple[Path, dict[str, Path]]:
    campaign_dir = tmp_path / "campaign"
    packet_dir = campaign_dir / "packets"
    receipt_dir = campaign_dir / "receipts"
    packet_dir.mkdir(parents=True)
    receipt_dir.mkdir()
    holdout_path = receipt_dir / "holdout.json"
    seed_path = receipt_dir / "seed.json"
    holdout_path.write_text('{"labels": [0, 1]}\n', encoding="utf-8")
    seed_path.write_text('{"seeds": [17, 29]}\n', encoding="utf-8")

    campaign_id = "POPGP-VIABILITY-TEST"
    packet_paths: dict[str, Path] = {}
    packet_files: dict[str, str] = {}
    for packet_id in REQUIREMENTS["tiers"][target_tier]:
        packet_path = packet_dir / f"{packet_id}.yaml"
        _write_yaml(
            packet_path,
            _make_packet(
                packet_id,
                campaign_id,
                receipt_dir / packet_id,
                holdout_path,
                seed_path,
                {key: frozen[key] for key in (
                    "candidate_commit",
                    "baseline_commit",
                    "tree_hash",
                    "protocol_commit",
                )},
                frozen["protocol_artifacts"][packet_id],
                frozen["initial_review_refs"][packet_id],
            ),
        )
        packet_paths[packet_id] = packet_path
        packet_files[packet_id] = f"packets/{packet_id}.yaml"

    campaign = {
        "schema_version": 2,
        "contract_version": "popgp-viability-contract-v2",
        "requirements_version": "popgp-viability-requirements-v2",
        "campaign_id": campaign_id,
        "target_tier": target_tier,
        "repository": "https://github.com/whact2025/POPGP",
        **{key: frozen[key] for key in (
            "candidate_commit",
            "baseline_commit",
            "tree_hash",
            "protocol_commit",
            "requirements_path",
            "requirements_sha256",
            "protocol_manifest_path",
            "protocol_manifest_sha256",
        )},
        "packet_files": packet_files,
        "decision": {
            "outcome": "passed",
            "authorized_by": "maintainer",
            "decided_at": "2026-08-10T14:00:00Z",
        },
    }
    campaign_path = campaign_dir / "CAMPAIGN.yaml"
    _write_yaml(campaign_path, campaign)
    return campaign_path, packet_paths


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _assert_mutation_fails(
    campaign_path: Path,
    packet_path: Path,
    mutation: Any,
    expected: str,
    repo_root: Path,
) -> None:
    original = packet_path.read_text(encoding="utf-8")
    packet = yaml.safe_load(original)
    mutation(packet)
    _write_yaml(packet_path, packet)
    errors = validate_campaign(campaign_path, repo_root=repo_root)
    packet_path.write_text(original, encoding="utf-8")
    assert any(expected in error for error in errors), errors


def _set_campaign_outcome(campaign_path: Path, outcome: str) -> None:
    campaign = _load(campaign_path)
    campaign["decision"]["outcome"] = outcome
    if outcome == "pending":
        campaign["decision"]["authorized_by"] = None
        campaign["decision"]["decided_at"] = None
    _write_yaml(campaign_path, campaign)


def _mutate_receipt_json(packet_path: Path, receipt_id: str, mutation: Any) -> None:
    packet = _load(packet_path)
    receipt_by_id = {receipt["id"]: receipt for receipt in packet["receipts"]}
    receipt = receipt_by_id[receipt_id]
    receipt_path = (packet_path.parent / receipt["path"]).resolve()
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation(document)
    receipt_path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    receipt["sha256"] = _sha256(receipt_path)
    if receipt_id == "raw-results":
        commitment = packet["blind_custody"]["output_commitment"]
        commitment["output_sha256"] = receipt["sha256"]
        commitment_receipt = receipt_by_id["output-commitment"]
        commitment_path = (packet_path.parent / commitment_receipt["path"]).resolve()
        commitment_document = json.loads(commitment_path.read_text(encoding="utf-8"))
        commitment_document["output_sha256"] = receipt["sha256"]
        commitment_path.write_text(
            json.dumps(commitment_document, sort_keys=True) + "\n", encoding="utf-8"
        )
        commitment_receipt["sha256"] = _sha256(commitment_path)
    _write_yaml(packet_path, packet)


def _attach_review_round(
    packet_path: Path,
    frozen: dict[str, Any],
    namespace: str,
    *,
    minimal_artifacts: bool = False,
    malformed_outcome: Any | None = None,
) -> None:
    packet = _load(packet_path)
    receipt_by_id = {receipt["id"]: receipt for receipt in packet["receipts"]}
    review_path = (packet_path.parent / receipt_by_id["independent-review"]["path"]).resolve()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review.update(
        findings=[_finding()],
        requested_tests=[_requested_test()],
        recommendation={
            "approve": False,
            "blocking_findings": 1,
            "rationale": "The fixture blocker is unresolved.",
        },
    )
    review_ref, review_bytes = _commit_json_artifact(
        frozen["root"], f"reviews/fixtures/{namespace}-initial.json", review
    )
    review_path.write_bytes(review_bytes)
    receipt_by_id["independent-review"]["sha256"] = _sha256(review_path)

    if minimal_artifacts:
        response = {
            "response_id": "RESPONSE-VIA-000-1",
            "review_id": review["review_id"],
            "finding_responses": [{"finding_id": "F-1"}],
            "requested_test_responses": [{"requested_test_id": "T-1"}],
        }
    else:
        response = _response_document(review, review_ref, include_test=True)
    response_ref, response_bytes = _commit_json_artifact(
        frozen["root"], f"reviews/fixtures/{namespace}-response.json", response
    )

    if minimal_artifacts:
        rereview = {
            "review_id": "REREVIEW-VIA-000-1",
            "review_kind": "re-review",
            "prior_finding_results": [
                {"finding_id": "F-1", "outcome": "verified-resolved"}
            ],
            "prior_requested_test_results": [
                {"requested_test_id": "T-1", "outcome": "verified-satisfied"}
            ],
            "findings": [],
            "requested_tests": [],
            "recommendation": {"approve": True, "blocking_findings": 0},
        }
    else:
        rereview = _rereview_document(
            frozen,
            review_ref,
            response_ref,
            finding_outcome=(
                malformed_outcome
                if malformed_outcome is not None
                else "verified-resolved"
            ),
            include_test=True,
        )
    rereview_ref, rereview_bytes = _commit_json_artifact(
        frozen["root"], f"reviews/fixtures/{namespace}-rereview.json", rereview
    )

    receipt_dir = review_path.parent
    response_path = receipt_dir / f"{namespace}-response.json"
    rereview_path = receipt_dir / f"{namespace}-rereview.json"
    response_path.write_bytes(response_bytes)
    rereview_path.write_bytes(rereview_bytes)
    packet["receipts"].extend(
        [
            _receipt(
                f"{namespace}-response",
                "builder-response",
                f"../receipts/VIA-000/{response_path.name}",
                _sha256(response_path),
            ),
            _receipt(
                f"{namespace}-rereview",
                "independent-rereview",
                f"../receipts/VIA-000/{rereview_path.name}",
                _sha256(rereview_path),
            ),
        ]
    )
    packet["review_chain"].update(
        initial_review_ref=review_ref,
        response_receipts=[f"{namespace}-response"],
        response_refs=[response_ref],
        rereview_receipts=[f"{namespace}-rereview"],
        rereview_refs=[rereview_ref],
        findings=[
            {
                "id": "F-1",
                "blocking": True,
                "outcome": "verified-resolved",
                "superseding_id": None,
            }
        ],
        requested_tests=[
            {
                "id": "T-1",
                "blocking": True,
                "outcome": "verified-satisfied",
                "superseding_id": None,
            }
        ],
    )
    _write_yaml(packet_path, packet)


def test_shipped_templates_conform_to_versioned_schemas() -> None:
    campaign_schema = json.loads(
        (ROOT / "schemas/viability/campaign-v2.schema.json").read_text(encoding="utf-8")
    )
    packet_schema = json.loads(
        (ROOT / "schemas/viability/packet-v2.schema.json").read_text(encoding="utf-8")
    )
    manifest_schema = json.loads(
        (ROOT / "schemas/viability/protocol-manifest-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    campaign = yaml.safe_load(
        (ROOT / "docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml").read_text(encoding="utf-8")
    )
    packet = yaml.safe_load(
        (ROOT / "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(Draft202012Validator(campaign_schema).iter_errors(campaign)) == []
    assert list(Draft202012Validator(packet_schema).iter_errors(packet)) == []
    assert list(Draft202012Validator(manifest_schema).iter_errors(manifest)) == []


@pytest.mark.negative_control
def test_schema_contract_rejects_cross_field_and_receipt_mutations(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path, frozen_repo)
    assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []

    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet.update(lifecycle_phase="drafted"),
        "non-adjudicated packet must remain not-run/pending",
        frozen_repo["root"],
    )

    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["claims"].append("C99"),
        "unknown claim id C99",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["existing_gates"].append("GATE-UNKNOWN"),
        "unknown gate id GATE-UNKNOWN",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["receipts"][0].update(path="../receipts/missing.json"),
        "does not exist",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["receipts"][0].update(sha256="0" * 64),
        "hash mismatch",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["adjudication"].update(decisive_receipts=[]),
        "requires decisive receipts",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["outcome_rules"]["bindings"]["result_passed"].update(
            json_pointer="/missing"
        ),
        "binding 'result_passed' cannot resolve",
        frozen_repo["root"],
    )


@pytest.mark.negative_control
def test_schema_contract_rejects_unresolved_or_incomplete_review_chain(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path, frozen_repo)

    def unresolved(packet: dict[str, Any]) -> None:
        packet["review_chain"]["findings"] = [
            {"id": "F-1", "blocking": True, "outcome": "unresolved", "superseding_id": None}
        ]

    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        unresolved,
        "declared findings differ from hashed review artifacts",
        frozen_repo["root"],
    )


@pytest.mark.negative_control
def test_review_chain_is_reconciled_to_hashed_artifact_bytes(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path / "omitted", frozen_repo)
    packet_path = packets["VIA-000"]
    _mutate_receipt_json(
        packet_path,
        "independent-review",
        lambda review: review.update(
            findings=[{"id": "F-1", "blocking": True}],
            recommendation={"approve": False, "blocking_findings": 1},
        ),
    )
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any(
        "independent-review receipt bytes differ from immutable ref" in error
        for error in errors
    )

    campaign_path, packets = _make_campaign(tmp_path / "dangling", frozen_repo)
    packet_path = packets["VIA-000"]
    packet = _load(packet_path)
    receipt_by_id = {receipt["id"]: receipt for receipt in packet["receipts"]}
    review_path = (packet_path.parent / receipt_by_id["independent-review"]["path"]).resolve()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review.update(
        findings=[_finding()],
        recommendation={
            "approve": False,
            "blocking_findings": 1,
            "rationale": "The fixture blocker is unresolved.",
        },
    )
    review_ref, review_bytes = _commit_json_artifact(
        frozen_repo["root"], "reviews/fixtures/dangling-initial.json", review
    )
    review_path.write_bytes(review_bytes)
    receipt_by_id["independent-review"]["sha256"] = _sha256(review_path)
    packet["review_chain"]["initial_review_ref"] = review_ref

    receipt_dir = review_path.parent
    response_path = receipt_dir / "builder-response.json"
    rereview_path = receipt_dir / "independent-rereview.json"
    response = _response_document(review, review_ref)
    response_ref, response_bytes = _commit_json_artifact(
        frozen_repo["root"], "reviews/fixtures/dangling-response.json", response
    )
    response_path.write_bytes(response_bytes)
    rereview = _rereview_document(
        frozen_repo,
        review_ref,
        response_ref,
        finding_outcome="superseded",
        superseding_finding_id="F-999",
    )
    rereview_ref, rereview_bytes = _commit_json_artifact(
        frozen_repo["root"], "reviews/fixtures/dangling-rereview.json", rereview
    )
    rereview_path.write_bytes(rereview_bytes)
    packet["receipts"].extend(
        [
            _receipt(
                "builder-response-1",
                "builder-response",
                f"../receipts/VIA-000/{response_path.name}",
                _sha256(response_path),
            ),
            _receipt(
                "independent-rereview-1",
                "independent-rereview",
                f"../receipts/VIA-000/{rereview_path.name}",
                _sha256(rereview_path),
            ),
        ]
    )
    packet["review_chain"].update(
        response_receipts=["builder-response-1"],
        response_refs=[response_ref],
        rereview_receipts=["independent-rereview-1"],
        rereview_refs=[rereview_ref],
        findings=[
            {
                "id": "F-1",
                "blocking": True,
                "outcome": "superseded",
                "superseding_id": "F-999",
            }
        ],
    )
    _write_yaml(packet_path, packet)
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any(
        "superseded finding F-1 has no declared successor" in error for error in errors
    )

    def incomplete(packet: dict[str, Any]) -> None:
        packet["review_chain"]["findings"] = [
            {
                "id": "F-1",
                "blocking": True,
                "outcome": "verified-resolved",
                "superseding_id": None,
            }
        ]

    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        incomplete,
        "declared findings differ from hashed review artifacts",
        frozen_repo["root"],
    )


@pytest.mark.negative_control
def test_review_artifacts_are_schema_complete_immutable_and_total(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path / "complete", frozen_repo)
    _attach_review_round(packets["VIA-000"], frozen_repo, "complete")
    assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []

    packet = _load(packets["VIA-000"])
    packet["review_chain"]["response_refs"][0] = packet["review_chain"][
        "initial_review_ref"
    ]
    _write_yaml(packets["VIA-000"], packet)
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any(
        "builder-response receipt bytes differ from immutable ref" in error
        for error in errors
    )

    campaign_path, packets = _make_campaign(tmp_path / "minimal", frozen_repo)
    _attach_review_round(
        packets["VIA-000"], frozen_repo, "minimal", minimal_artifacts=True
    )
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any(
        "builder-response" in error and "required property" in error
        for error in errors
    )
    assert any(
        "independent-rereview" in error and "required property" in error
        for error in errors
    )

    campaign_path, packets = _make_campaign(tmp_path / "malformed", frozen_repo)
    _attach_review_round(
        packets["VIA-000"],
        frozen_repo,
        "malformed",
        malformed_outcome={},
    )
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any(
        "independent-rereview" in error
        and "prior_finding_results" in error
        and "is not one of" in error
        for error in errors
    )

    campaign_path, _ = _make_campaign(tmp_path / "duplicate-yaml", frozen_repo)
    campaign_path.write_text(
        campaign_path.read_text(encoding="utf-8")
        + "\ndecision:\n  outcome: passed\n  authorized_by: duplicate\n"
        + "  decided_at: '2026-08-10T14:00:00Z'\n",
        encoding="utf-8",
    )
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any("duplicate key 'decision'" in error for error in errors)

    campaign_path, packets = _make_campaign(tmp_path / "duplicate-json", frozen_repo)
    packet_path = packets["VIA-000"]
    packet = _load(packet_path)
    receipt_by_id = {receipt["id"]: receipt for receipt in packet["receipts"]}
    raw_receipt = receipt_by_id["raw-results"]
    raw_path = (packet_path.parent / raw_receipt["path"]).resolve()
    raw_text = raw_path.read_text(encoding="utf-8")
    raw_path.write_text(
        raw_text.replace('"passed": true', '"passed": true, "passed": false'),
        encoding="utf-8",
    )
    raw_receipt["sha256"] = _sha256(raw_path)
    packet["blind_custody"]["output_commitment"]["output_sha256"] = raw_receipt[
        "sha256"
    ]
    commitment_receipt = receipt_by_id["output-commitment"]
    commitment_path = (packet_path.parent / commitment_receipt["path"]).resolve()
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    commitment["output_sha256"] = raw_receipt["sha256"]
    _write_json(commitment_path, commitment)
    commitment_receipt["sha256"] = _sha256(commitment_path)
    _write_yaml(packet_path, packet)
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any("duplicate JSON key 'passed'" in error for error in errors)

    malformed_requirements = copy.deepcopy(REQUIREMENTS)
    malformed_requirements["packets"]["VIA-010"] = []
    errors = validate_requirements(malformed_requirements)
    assert any("VIA-010 must be an object" in error for error in errors)


@pytest.mark.negative_control
def test_protocol_content_and_budget_are_frozen_before_holdout(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path / "bytes", frozen_repo)
    packet_path = packets["VIA-000"]
    packet = _load(packet_path)
    protocol_receipt = next(
        receipt for receipt in packet["receipts"] if receipt["id"] == "protocol"
    )
    protocol_path = (packet_path.parent / protocol_receipt["path"]).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["parameters"]["threshold"] = 0.5
    _write_json(protocol_path, protocol)
    protocol_receipt["sha256"] = _sha256(protocol_path)
    _write_yaml(packet_path, packet)
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any(
        "protocol receipt 'protocol' sha256 differs from frozen preregistration" in error
        for error in errors
    )

    campaign_path, packets = _make_campaign(tmp_path / "path", frozen_repo)
    packet_path = packets["VIA-000"]
    packet = _load(packet_path)
    protocol_receipt = next(
        receipt for receipt in packet["receipts"] if receipt["id"] == "protocol"
    )
    original_path = (packet_path.parent / protocol_receipt["path"]).resolve()
    substituted_path = original_path.with_name("protocol-substituted.json")
    substituted = json.loads(original_path.read_text(encoding="utf-8"))
    substituted["resource_budget"]["wall_time_seconds"] = 7200
    _write_json(substituted_path, substituted)
    protocol_receipt.update(
        path=f"../receipts/VIA-000/{substituted_path.name}",
        sha256=_sha256(substituted_path),
    )
    packet["preregistration"]["protocol_artifacts"][0].update(
        campaign_path=protocol_receipt["path"],
        sha256=protocol_receipt["sha256"],
    )
    packet["protocol_rule_sha256"] = packet_rule_sha256(packet)
    _write_yaml(packet_path, packet)
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any("packet rules differ from protocol snapshot" in error for error in errors)
    assert any("preregistered protocol blob hash mismatch" in error for error in errors)

    campaign_path, packets = _make_campaign(tmp_path / "budget", frozen_repo)
    packet_path = packets["VIA-000"]
    packet = _load(packet_path)
    packet["preregistration"]["resource_budget"]["wall_time_seconds"] = 7200
    packet["protocol_rule_sha256"] = packet_rule_sha256(packet)
    _write_yaml(packet_path, packet)
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any("packet rules differ from protocol snapshot" in error for error in errors)


@pytest.mark.negative_control
def test_requirements_reject_missing_cycles_and_invalid_waves() -> None:
    assert validate_requirements(REQUIREMENTS) == []

    missing = copy.deepcopy(REQUIREMENTS)
    missing["packets"]["VIA-100"]["dependencies"].append("VIA-999")
    assert any("unknown dependency VIA-999" in error for error in validate_requirements(missing))

    cycle = copy.deepcopy(REQUIREMENTS)
    cycle["packets"]["VIA-000"]["dependencies"] = ["VIA-100"]
    assert any("dependency cycle" in error for error in validate_requirements(cycle))

    same_wave = copy.deepcopy(REQUIREMENTS)
    same_wave["packets"]["VIA-100"]["execution_wave"] = 1
    assert any(
        "must follow dependency VIA-010" in error
        for error in validate_requirements(same_wave)
    )


@pytest.mark.negative_control
def test_blind_custody_rejects_leaks_role_reuse_and_manifest_mutations(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path, frozen_repo)
    packet_path = packets["VIA-000"]

    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["seats"]["reproduction_runner"]["exposure"].update(
            final_labels_seen=True
        ),
        "blind seat reproduction_runner records prohibited exposure",
        frozen_repo["root"],
    )

    def shared_session(packet: dict[str, Any]) -> None:
        packet["seats"]["falsifier"]["session_id"] = packet["seats"]["builder"]["session_id"]

    _assert_mutation_fails(
        campaign_path,
        packet_path,
        shared_session,
        "prohibited shared session",
        frozen_repo["root"],
    )

    def reused_custodian(packet: dict[str, Any]) -> None:
        packet["seats"]["evaluator_custodian"]["agent_identity"] = packet["seats"][
            "builder"
        ]["agent_identity"]

    _assert_mutation_fails(
        campaign_path,
        packet_path,
        reused_custodian,
        "may not also be builder",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["blind_custody"]["reveal"].update(
            authorized_by=packet["seats"]["builder"]["agent_identity"]
        ),
        "reveal must be authorized by evaluator_custodian",
        frozen_repo["root"],
    )

    def premature_reveal(packet: dict[str, Any]) -> None:
        packet["lifecycle_phase"] = "preregistered"
        packet["holdout_started"] = False
        packet["adjudication"].update(
            round_status="not-run",
            packet_outcome="pending",
            cause_codes=["not-run"],
            decisive_receipts=[],
        )

    _assert_mutation_fails(
        campaign_path,
        packet_path,
        premature_reveal,
        "reveal cannot precede holdout execution",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["blind_custody"]["output_commitment"].update(
            committed_by=packet["seats"]["builder"]["agent_identity"]
        ),
        "output commitment must be made by reproduction_runner",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["blind_custody"]["output_commitment"].update(
            output_sha256="0" * 64
        ),
        "output commitment hash differs from runner output",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["blind_custody"].update(canonicalization="sorted-json-v1"),
        "raw-bytes-v1",
        frozen_repo["root"],
    )

    def reveal_early(packet: dict[str, Any]) -> None:
        packet["blind_custody"]["output_commitment"]["committed_at"] = (
            "2026-08-10T15:00:00Z"
        )

    _assert_mutation_fails(
        campaign_path,
        packet_path,
        reveal_early,
        "reveal must follow",
        frozen_repo["root"],
    )

    def post_reveal_substitution(packet: dict[str, Any]) -> None:
        packet["blind_custody"]["reveal"]["post_reveal_holdout_sha256"] = "0" * 64

    _assert_mutation_fails(
        campaign_path,
        packet_path,
        post_reveal_substitution,
        "differs from commitment",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["blind_custody"]["reveal"].update(retention_policy=""),
        "should be non-empty",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["seats"].pop("evaluator_custodian"),
        "evaluator_custodian",
        frozen_repo["root"],
    )


@pytest.mark.negative_control
def test_tier_g_countermodel_cannot_omit_minimum_physics_comparisons(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path, frozen_repo, target_tier="G")
    assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []

    def remove_framework_gates(packet: dict[str, Any]) -> None:
        omitted = {
            "three-dimensional-recovery",
            "acceleration-geodesic-response",
            "same-source-lensing",
            "same-source-shapiro-delay",
            "two-potential-consistency",
            "laboratory-interference-statistics",
            "laboratory-entanglement-statistics",
        }
        packet["capabilities"] = [
            capability for capability in packet["capabilities"] if capability not in omitted
        ]
        packet["capability_rules"] = {
            capability: expression
            for capability, expression in packet["capability_rules"].items()
            if capability not in omitted
        }

    _assert_mutation_fails(
        campaign_path,
        packets["VIA-700"],
        remove_framework_gates,
        "missing required capabilities",
        frozen_repo["root"],
    )

    packet_path = packets["VIA-700"]
    _mutate_receipt_json(
        packet_path,
        "raw-results",
        lambda document: document["capabilities"].update(
            {
                "three-dimensional-recovery": False,
                "same-source-lensing": False,
            }
        ),
    )
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any(
        "passing outcome has failed capabilities" in error
        and "three-dimensional-recovery" in error
        and "same-source-lensing" in error
        for error in errors
    )

    _mutate_receipt_json(
        packet_path,
        "raw-results",
        lambda document: (
            document["capabilities"].update(
                {
                    "three-dimensional-recovery": True,
                    "same-source-lensing": True,
                }
            ),
            document.update(passed=1),
        ),
    )
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any(
        "binding 'result_passed' cannot resolve" in error
        and "expected boolean, observed number" in error
        for error in errors
    )

    _assert_mutation_fails(
        campaign_path,
        packets["VIA-700"],
        lambda packet: packet["capability_rules"].update(
            {"same-source-lensing": {"literal": False}}
        ),
        "must use its canonical raw Boolean gate",
        frozen_repo["root"],
    )


def _set_packet_decision(
    packet_path: Path,
    *,
    pass_value: bool,
    fail_value: bool,
    blocked_value: bool,
    round_status: str,
    outcome: str,
    cause: str,
) -> None:
    _mutate_receipt_json(
        packet_path,
        "raw-results",
        lambda document: document.update(
            passed=pass_value, failed=fail_value, blocked=blocked_value
        ),
    )
    packet = _load(packet_path)
    packet["adjudication"]["round_status"] = round_status
    packet["adjudication"]["packet_outcome"] = outcome
    packet["adjudication"]["cause_codes"] = [cause]
    _write_yaml(packet_path, packet)


@pytest.mark.negative_control
def test_outcome_truth_table_is_deterministic(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path, frozen_repo)
    leaf = packets["VIA-400"]

    # Valid scientific negative.
    _set_packet_decision(
        leaf,
        pass_value=False,
        fail_value=True,
        blocked_value=False,
        round_status="valid",
        outcome="failed",
        cause="scientific-gate-failed",
    )
    _set_campaign_outcome(campaign_path, "failed")
    assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []

    # Candidate implementation defect that validly fails a capability claim.
    _set_packet_decision(
        leaf,
        pass_value=False,
        fail_value=True,
        blocked_value=False,
        round_status="valid",
        outcome="failed",
        cause="implementation-capability-failed",
    )
    assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []

    # Resource exhaustion is a failure when scalability is the tested capability.
    _set_packet_decision(
        leaf,
        pass_value=False,
        fail_value=True,
        blocked_value=False,
        round_status="valid",
        outcome="failed",
        cause="tested-capability-budget-exhausted",
    )
    assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []

    # Unavailable TeX, CUDA-class hardware, and external access are blockages.
    for cause in (
        "toolchain-unavailable",
        "hardware-unavailable",
        "external-access-unavailable",
    ):
        _set_packet_decision(
            leaf,
            pass_value=False,
            fail_value=False,
            blocked_value=True,
            round_status="valid",
            outcome="blocked",
            cause=cause,
        )
        _set_campaign_outcome(campaign_path, "blocked")
        assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []

    # A valid-round pending declaration must return errors, never raise.
    _set_packet_decision(
        leaf,
        pass_value=True,
        fail_value=False,
        blocked_value=False,
        round_status="valid",
        outcome="pending",
        cause="pass-rule-satisfied",
    )
    _set_campaign_outcome(campaign_path, "pending")
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any("declared outcome pending != computed passed" in error for error in errors)

    # A missing receipt fails closed.
    raw_receipt_path = (
        leaf.parent
        / next(
            receipt["path"]
            for receipt in _load(leaf)["receipts"]
            if receipt["id"] == "raw-results"
        )
    ).resolve()
    saved_raw = raw_receipt_path.read_bytes()
    raw_receipt_path.unlink()
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any("receipt 'raw-results' does not exist" in error for error in errors)
    raw_receipt_path.write_bytes(saved_raw)

    # Simultaneous failure/blockage predicates are rejected for a valid round.
    _set_packet_decision(
        leaf,
        pass_value=False,
        fail_value=True,
        blocked_value=True,
        round_status="valid",
        outcome="failed",
        cause="scientific-gate-failed",
    )
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any("outcome rules are nonexclusive" in error for error in errors)

    # The same ambiguity is representable only as an invalid round.
    packet = _load(leaf)
    packet["adjudication"].update(
        round_status="invalid",
        packet_outcome="pending",
        cause_codes=["rule-ambiguous"],
    )
    _write_yaml(leaf, packet)
    _set_campaign_outcome(campaign_path, "pending")
    assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []

    # Protocol-invalid and receipt-invalid rounds remain pending and deterministic.
    for invalid_cause in ("protocol-invalid", "receipt-invalid"):
        packet = _load(leaf)
        packet["adjudication"].update(
            round_status="invalid",
            packet_outcome="pending",
            cause_codes=[invalid_cause],
        )
        _write_yaml(leaf, packet)
        assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []


@pytest.mark.negative_control
def test_holdout_cannot_start_until_dependencies_pass(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path, frozen_repo)
    dependency = _load(packets["VIA-010"])
    dependency["lifecycle_phase"] = "preregistered"
    dependency["holdout_started"] = False
    dependency["adjudication"].update(
        round_status="not-run",
        packet_outcome="pending",
        cause_codes=["not-run"],
        decisive_receipts=[],
    )
    _write_yaml(packets["VIA-010"], dependency)
    _set_campaign_outcome(campaign_path, "pending")
    errors = validate_campaign(campaign_path, repo_root=frozen_repo["root"])
    assert any(
        "VIA-100: holdout started before dependency VIA-010 passed" in error
        for error in errors
    )
    assert any(
        "VIA-200: holdout started before dependency VIA-010 passed" in error
        for error in errors
    )


@pytest.mark.negative_control
def test_campaign_owned_evidence_floors_cannot_be_downgraded(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path, frozen_repo, target_tier="E")
    assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []

    _assert_mutation_fails(
        campaign_path,
        packets["VIA-300"],
        lambda packet: packet.update(declared_evidence_requirement="E3-adversarial-suite"),
        "below campaign floor E4-convergent-replication",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-400"],
        lambda packet: packet["adjudication"].update(
            achieved_evidence="E3-adversarial-suite"
        ),
        "below declared E4-convergent-replication",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-900"],
        lambda packet: packet.update(declared_evidence_requirement="E4-convergent-replication"),
        "below campaign floor E5-external-empirical",
        frozen_repo["root"],
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-900"],
        lambda packet: packet.update(declared_evidence_requirement="E9-unknown"),
        "is not one of",
        frozen_repo["root"],
    )

    assert _load(packets["VIA-000"])["declared_evidence_requirement"] == (
        "E4-convergent-replication"
    )
    assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []


@pytest.mark.negative_control
def test_campaign_is_bound_to_frozen_git_and_protocol_content(
    tmp_path: Path, frozen_repo: dict[str, Any]
) -> None:
    campaign_path, packets = _make_campaign(tmp_path, frozen_repo, target_tier="G")
    repo_root = frozen_repo["root"]
    assert validate_campaign(campaign_path, repo_root=repo_root) == []
    original_campaign = campaign_path.read_text(encoding="utf-8")

    for field in ("candidate_commit", "baseline_commit", "protocol_commit"):
        campaign = yaml.safe_load(original_campaign)
        campaign[field] = "f" * 40
        _write_yaml(campaign_path, campaign)
        errors = validate_campaign(campaign_path, repo_root=repo_root)
        assert any(f"{field} does not resolve to a Git commit" in error for error in errors)
    campaign_path.write_text(original_campaign, encoding="utf-8")

    campaign = yaml.safe_load(original_campaign)
    campaign["tree_hash"] = "0" * 40
    _write_yaml(campaign_path, campaign)
    errors = validate_campaign(campaign_path, repo_root=repo_root)
    assert any("tree_hash does not match candidate commit tree" in error for error in errors)
    campaign_path.write_text(original_campaign, encoding="utf-8")

    packet_path = packets["VIA-700"]
    packet = _load(packet_path)
    packet["outcome_rules"]["pass"]["compare"]["right"]["value"] = False
    packet["protocol_rule_sha256"] = packet_rule_sha256(packet)
    _write_yaml(packet_path, packet)
    errors = validate_campaign(campaign_path, repo_root=repo_root)
    assert any("packet rules differ from protocol snapshot" in error for error in errors)

    mutated_requirements = copy.deepcopy(REQUIREMENTS)
    mutated_requirements["packets"]["VIA-700"]["required_capabilities"].remove(
        "three-dimensional-recovery"
    )
    errors = validate_campaign(
        campaign_path,
        repo_root=repo_root,
        requirements_document=mutated_requirements,
    )
    assert any(
        "supplied requirements differ from frozen protocol requirements" in error
        for error in errors
    )

    protocol_script = repo_root / "scripts/check_viability_campaign.py"
    original_script = protocol_script.read_bytes()
    protocol_script.write_bytes(original_script + b"\n# post-freeze mutation\n")
    try:
        errors = validate_campaign(campaign_path, repo_root=repo_root)
        assert any(
            "executing contract file 'scripts/check_viability_campaign.py' differs"
            in error
            for error in errors
        )
    finally:
        protocol_script.write_bytes(original_script)
