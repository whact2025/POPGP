from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

from scripts.check_viability_campaign import validate_campaign, validate_requirements

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = json.loads(
    (ROOT / "schemas/viability/requirements-v1.json").read_text(encoding="utf-8")
)
HASHES = {
    "candidate_commit": "a" * 40,
    "baseline_commit": "b" * 40,
    "tree_hash": "c" * 40,
    "protocol_commit": "d" * 40,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


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


def _receipt(receipt_id: str, kind: str, path: str, digest: str) -> dict[str, str]:
    return {
        "id": receipt_id,
        "kind": kind,
        "path": path,
        "sha256": digest,
        "media_type": "application/json",
    }


def _make_packet(
    packet_id: str,
    campaign_id: str,
    evidence_path: Path,
    holdout_path: Path,
    seed_path: Path,
) -> dict[str, Any]:
    requirement = REQUIREMENTS["packets"][packet_id]
    evidence_hash = _sha256(evidence_path)
    holdout_hash = _sha256(holdout_path)
    seed_hash = _sha256(seed_path)
    evidence_kinds = sorted(
        {
            kind
            for kinds in REQUIREMENTS["required_receipt_kinds"].values()
            for kind in kinds
        }
    )
    receipts = [
        _receipt(kind, kind, "../receipts/evidence.json", evidence_hash)
        for kind in evidence_kinds
    ]
    receipts.extend(
        [
            _receipt(
                "output-commitment",
                "output-commitment",
                "../receipts/evidence.json",
                evidence_hash,
            ),
            _receipt(
                "blockage-evidence",
                "blockage-evidence",
                "../receipts/evidence.json",
                evidence_hash,
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
    return {
        "schema_version": 1,
        "contract_version": "popgp-viability-contract-v1",
        "campaign_id": campaign_id,
        "packet_id": packet_id,
        **HASHES,
        "lifecycle_phase": "adjudicated",
        "claims": ["C01"],
        "existing_gates": [],
        "declared_evidence_requirement": requirement["minimum_evidence"],
        "capabilities": requirement["required_capabilities"],
        "capability_rules": {
            capability: {"literal": True}
            for capability in requirement["required_capabilities"]
        },
        "hypothesis": "The frozen packet satisfies its preregistered gate.",
        "null_or_competitors": ["The gate does not discriminate the candidate."],
        "known_failure_to_retain": "none",
        "threat_model": ["post-selection", "hidden-label leakage"],
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
                "committed_at": "2026-08-10T12:00:00Z",
            },
            "reveal": {
                "status": "revealed",
                "authorized_by": "maintainer",
                "revealed_at": "2026-08-10T13:00:00Z",
                "post_reveal_holdout_sha256": holdout_hash,
                "post_reveal_seed_sha256": seed_hash,
                "post_reveal_holdout_receipt_id": "revealed-holdout",
                "post_reveal_seed_receipt_id": "revealed-seed",
                "retention_policy": "retain immutable manifests with campaign receipts",
                "immutable_location": "archive://campaign/manifests",
            },
        },
        "outcome_rules": {
            "language": "popgp-bool-v1",
            "bindings": {
                "result_passed": {
                    "receipt_id": "raw-results",
                    "json_pointer": "/passed",
                }
            },
            "pass": {
                "compare": {
                    "left": {"binding": "result_passed"},
                    "op": "eq",
                    "right": {"value": True},
                }
            },
            "fail": {
                "compare": {
                    "left": {"binding": "result_passed"},
                    "op": "eq",
                    "right": {"value": False},
                }
            },
            "blocked": {"literal": False},
        },
        "receipts": receipts,
        "review_chain": {
            "initial_review_receipt": "independent-review",
            "response_receipts": [],
            "rereview_receipts": [],
            "findings": [],
            "requested_tests": [],
        },
        "adjudication": {
            "round_status": "valid",
            "packet_outcome": "passed",
            "cause_codes": ["pass-rule-satisfied"],
            "decisive_receipts": ["raw-results"],
            "achieved_evidence": requirement["minimum_evidence"],
        },
    }


def _make_campaign(tmp_path: Path, target_tier: str = "R") -> tuple[Path, dict[str, Path]]:
    campaign_dir = tmp_path / "campaign"
    packet_dir = campaign_dir / "packets"
    receipt_dir = campaign_dir / "receipts"
    packet_dir.mkdir(parents=True)
    receipt_dir.mkdir()
    evidence_path = receipt_dir / "evidence.json"
    holdout_path = receipt_dir / "holdout.json"
    seed_path = receipt_dir / "seed.json"
    evidence_path.write_text('{"metric": 1, "passed": true}\n', encoding="utf-8")
    holdout_path.write_text('{"labels": [0, 1]}\n', encoding="utf-8")
    seed_path.write_text('{"seeds": [17, 29]}\n', encoding="utf-8")

    campaign_id = "POPGP-VIABILITY-TEST"
    packet_paths: dict[str, Path] = {}
    packet_files: dict[str, str] = {}
    for packet_id in REQUIREMENTS["tiers"][target_tier]:
        packet_path = packet_dir / f"{packet_id}.yaml"
        _write_yaml(
            packet_path,
            _make_packet(packet_id, campaign_id, evidence_path, holdout_path, seed_path),
        )
        packet_paths[packet_id] = packet_path
        packet_files[packet_id] = f"packets/{packet_id}.yaml"

    campaign = {
        "schema_version": 1,
        "contract_version": "popgp-viability-contract-v1",
        "requirements_version": "popgp-viability-requirements-v1",
        "campaign_id": campaign_id,
        "target_tier": target_tier,
        "repository": "https://github.com/whact2025/POPGP",
        **HASHES,
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
) -> None:
    original = packet_path.read_text(encoding="utf-8")
    packet = yaml.safe_load(original)
    mutation(packet)
    _write_yaml(packet_path, packet)
    errors = validate_campaign(campaign_path, repo_root=ROOT)
    packet_path.write_text(original, encoding="utf-8")
    assert any(expected in error for error in errors), errors


def _set_campaign_outcome(campaign_path: Path, outcome: str) -> None:
    campaign = _load(campaign_path)
    campaign["decision"]["outcome"] = outcome
    if outcome == "pending":
        campaign["decision"]["authorized_by"] = None
        campaign["decision"]["decided_at"] = None
    _write_yaml(campaign_path, campaign)


def test_shipped_templates_conform_to_versioned_schemas() -> None:
    campaign_schema = json.loads(
        (ROOT / "schemas/viability/campaign-v1.schema.json").read_text(encoding="utf-8")
    )
    packet_schema = json.loads(
        (ROOT / "schemas/viability/packet-v1.schema.json").read_text(encoding="utf-8")
    )
    campaign = yaml.safe_load(
        (ROOT / "docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml").read_text(encoding="utf-8")
    )
    packet = yaml.safe_load(
        (ROOT / "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml").read_text(encoding="utf-8")
    )
    assert list(Draft202012Validator(campaign_schema).iter_errors(campaign)) == []
    assert list(Draft202012Validator(packet_schema).iter_errors(packet)) == []


@pytest.mark.negative_control
def test_schema_contract_rejects_cross_field_and_receipt_mutations(tmp_path: Path) -> None:
    campaign_path, packets = _make_campaign(tmp_path)
    assert validate_campaign(campaign_path, repo_root=ROOT) == []

    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet.update(lifecycle_phase="drafted"),
        "non-adjudicated packet must remain not-run/pending",
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["claims"].append("C99"),
        "unknown claim id C99",
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["existing_gates"].append("GATE-UNKNOWN"),
        "unknown gate id GATE-UNKNOWN",
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["receipts"][0].update(path="../receipts/missing.json"),
        "does not exist",
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["receipts"][0].update(sha256="0" * 64),
        "hash mismatch",
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["adjudication"].update(decisive_receipts=[]),
        "requires decisive receipts",
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-000"],
        lambda packet: packet["outcome_rules"]["bindings"]["result_passed"].update(
            json_pointer="/missing"
        ),
        "binding 'result_passed' cannot resolve",
    )


@pytest.mark.negative_control
def test_schema_contract_rejects_unresolved_or_incomplete_review_chain(tmp_path: Path) -> None:
    campaign_path, packets = _make_campaign(tmp_path)

    def unresolved(packet: dict[str, Any]) -> None:
        packet["review_chain"]["findings"] = [
            {"id": "F-1", "blocking": True, "outcome": "unresolved", "superseding_id": None}
        ]

    _assert_mutation_fails(
        campaign_path, packets["VIA-000"], unresolved, "blocking finding F-1 is unresolved"
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
        "findings/tests require response and re-review receipts",
    )


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
def test_blind_custody_rejects_leaks_role_reuse_and_manifest_mutations(tmp_path: Path) -> None:
    campaign_path, packets = _make_campaign(tmp_path)
    packet_path = packets["VIA-000"]

    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["seats"]["reproduction_runner"]["exposure"].update(
            final_labels_seen=True
        ),
        "blind seat reproduction_runner records prohibited exposure",
    )

    def shared_session(packet: dict[str, Any]) -> None:
        packet["seats"]["falsifier"]["session_id"] = packet["seats"]["builder"]["session_id"]

    _assert_mutation_fails(
        campaign_path, packet_path, shared_session, "prohibited shared session"
    )

    def reused_custodian(packet: dict[str, Any]) -> None:
        packet["seats"]["evaluator_custodian"]["agent_identity"] = packet["seats"][
            "builder"
        ]["agent_identity"]

    _assert_mutation_fails(
        campaign_path, packet_path, reused_custodian, "may not also be builder"
    )
    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["blind_custody"].update(canonicalization="sorted-json-v1"),
        "raw-bytes-v1",
    )

    def reveal_early(packet: dict[str, Any]) -> None:
        packet["blind_custody"]["output_commitment"]["committed_at"] = (
            "2026-08-10T15:00:00Z"
        )

    _assert_mutation_fails(campaign_path, packet_path, reveal_early, "reveal must follow")

    def post_reveal_substitution(packet: dict[str, Any]) -> None:
        packet["blind_custody"]["reveal"]["post_reveal_holdout_sha256"] = "0" * 64

    _assert_mutation_fails(
        campaign_path, packet_path, post_reveal_substitution, "differs from commitment"
    )
    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["blind_custody"]["reveal"].update(retention_policy=""),
        "should be non-empty",
    )
    _assert_mutation_fails(
        campaign_path,
        packet_path,
        lambda packet: packet["seats"].pop("evaluator_custodian"),
        "evaluator_custodian",
    )


@pytest.mark.negative_control
def test_tier_g_countermodel_cannot_omit_minimum_physics_comparisons(tmp_path: Path) -> None:
    campaign_path, packets = _make_campaign(tmp_path, target_tier="G")
    assert validate_campaign(campaign_path, repo_root=ROOT) == []

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
    )

    _assert_mutation_fails(
        campaign_path,
        packets["VIA-700"],
        lambda packet: packet["capability_rules"].update(
            {"same-source-lensing": {"literal": False}}
        ),
        "passing outcome has failed capabilities ['same-source-lensing']",
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
    packet = _load(packet_path)
    packet["outcome_rules"] = {
        "language": "popgp-bool-v1",
        "bindings": {},
        "pass": {"literal": pass_value},
        "fail": {"literal": fail_value},
        "blocked": {"literal": blocked_value},
    }
    packet["adjudication"]["round_status"] = round_status
    packet["adjudication"]["packet_outcome"] = outcome
    packet["adjudication"]["cause_codes"] = [cause]
    _write_yaml(packet_path, packet)


@pytest.mark.negative_control
def test_outcome_truth_table_is_deterministic(tmp_path: Path) -> None:
    campaign_path, packets = _make_campaign(tmp_path)
    leaf = packets["VIA-400"]

    _set_packet_decision(
        leaf,
        pass_value=False,
        fail_value=True,
        blocked_value=False,
        round_status="valid",
        outcome="failed",
        cause="tested-capability-budget-exhausted",
    )
    _set_campaign_outcome(campaign_path, "failed")
    assert validate_campaign(campaign_path, repo_root=ROOT) == []

    _set_packet_decision(
        leaf,
        pass_value=False,
        fail_value=False,
        blocked_value=True,
        round_status="valid",
        outcome="blocked",
        cause="toolchain-unavailable",
    )
    _set_campaign_outcome(campaign_path, "blocked")
    assert validate_campaign(campaign_path, repo_root=ROOT) == []

    _set_packet_decision(
        leaf,
        pass_value=False,
        fail_value=True,
        blocked_value=True,
        round_status="valid",
        outcome="failed",
        cause="scientific-gate-failed",
    )
    errors = validate_campaign(campaign_path, repo_root=ROOT)
    assert any("outcome rules are nonexclusive" in error for error in errors)

    packet = _load(leaf)
    packet["outcome_rules"] = None
    packet["adjudication"].update(
        round_status="invalid",
        packet_outcome="pending",
        cause_codes=["protocol-invalid"],
    )
    _write_yaml(leaf, packet)
    _set_campaign_outcome(campaign_path, "pending")
    assert validate_campaign(campaign_path, repo_root=ROOT) == []


@pytest.mark.negative_control
def test_holdout_cannot_start_until_dependencies_pass(tmp_path: Path) -> None:
    campaign_path, packets = _make_campaign(tmp_path)
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
    errors = validate_campaign(campaign_path, repo_root=ROOT)
    assert any(
        "VIA-100: holdout started before dependency VIA-010 passed" in error
        for error in errors
    )
    assert any(
        "VIA-200: holdout started before dependency VIA-010 passed" in error
        for error in errors
    )


@pytest.mark.negative_control
def test_campaign_owned_evidence_floors_cannot_be_downgraded(tmp_path: Path) -> None:
    campaign_path, packets = _make_campaign(tmp_path, target_tier="E")
    assert validate_campaign(campaign_path, repo_root=ROOT) == []

    _assert_mutation_fails(
        campaign_path,
        packets["VIA-300"],
        lambda packet: packet.update(declared_evidence_requirement="E3-adversarial-suite"),
        "below campaign floor E4-convergent-replication",
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-400"],
        lambda packet: packet["adjudication"].update(
            achieved_evidence="E3-adversarial-suite"
        ),
        "below declared E4-convergent-replication",
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-900"],
        lambda packet: packet.update(declared_evidence_requirement="E4-convergent-replication"),
        "below campaign floor E5-external-empirical",
    )
    _assert_mutation_fails(
        campaign_path,
        packets["VIA-900"],
        lambda packet: packet.update(declared_evidence_requirement="E9-unknown"),
        "is not one of",
    )

    stricter = _load(packets["VIA-000"])
    stricter["declared_evidence_requirement"] = "E4-convergent-replication"
    stricter["adjudication"]["achieved_evidence"] = "E4-convergent-replication"
    _write_yaml(packets["VIA-000"], stricter)
    assert validate_campaign(campaign_path, repo_root=ROOT) == []
