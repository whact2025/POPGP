"""Validate a POPGP adversarial viability campaign and its packet receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker

CONTRACT_VERSION = "popgp-viability-contract-v1"
REQUIREMENTS_VERSION = "popgp-viability-requirements-v1"
RULE_LANGUAGE = "popgp-bool-v1"

LIFECYCLE_ORDER = {
    "drafted": 0,
    "preregistered": 1,
    "implemented": 2,
    "attacked": 3,
    "reproduced": 4,
    "adjudicated": 5,
}

INVALID_CAUSES = {
    "protocol-invalid",
    "receipt-invalid",
    "rule-ambiguous",
    "custody-invalid",
    "dependency-invalid",
}
FAILURE_CAUSES = {
    "scientific-gate-failed",
    "tested-capability-budget-exhausted",
    "implementation-capability-failed",
    "external-replication-disagreed",
}
BLOCKAGE_CAUSES = {
    "toolchain-unavailable",
    "hardware-unavailable",
    "external-access-unavailable",
    "resource-not-authorized",
}

SEPARATE_SESSION_SEATS = {
    "builder",
    "falsifier",
    "reproduction_runner",
    "adjudicator",
    "evaluator_custodian",
}
BLIND_SEATS = {
    "builder",
    "falsifier",
    "reproduction_runner",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_path(parts: list[Any]) -> str:
    if not parts:
        return "$"
    return "$" + "".join(f"[{part!r}]" if isinstance(part, str) else f"[{part}]" for part in parts)


def _schema_errors(document: Any, schema: Any, label: str) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        f"{label}{_format_path(list(error.absolute_path))}: {error.message}"
        for error in sorted(validator.iter_errors(document), key=lambda item: list(item.path))
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside(base: Path, relative: str, root: Path) -> Path | None:
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _canonical_claim_ids(root: Path) -> set[str]:
    text = (root / "docs/scientific_hardening/CLAIMS_MATRIX.md").read_text(encoding="utf-8")
    return set(re.findall(r"^\| (C[0-9]{2}) \|", text, flags=re.MULTILINE))


def _canonical_gate_ids(root: Path) -> set[str]:
    text = (root / "docs/scientific_hardening/GATE_TEST_REGISTRY.md").read_text(
        encoding="utf-8"
    )
    return set(re.findall(r"^\| `(GATE-[A-Z0-9-]+)` \|", text, flags=re.MULTILINE))


def validate_requirements(requirements: Any) -> list[str]:
    """Validate the campaign-owned packet DAG and evidence floors."""
    errors: list[str] = []
    if not isinstance(requirements, Mapping):
        return ["requirements: document must be an object"]
    if requirements.get("version") != REQUIREMENTS_VERSION:
        errors.append(f"requirements: version must be {REQUIREMENTS_VERSION!r}")

    evidence_order = requirements.get("evidence_order")
    if not isinstance(evidence_order, list) or not evidence_order:
        errors.append("requirements: evidence_order must be a nonempty array")
        evidence_order = []
    elif len(evidence_order) != len(set(evidence_order)):
        errors.append("requirements: evidence_order values must be unique")

    packets = requirements.get("packets")
    if not isinstance(packets, Mapping) or not packets:
        errors.append("requirements: packets must be a nonempty object")
        return errors

    for packet_id, requirement in packets.items():
        if not re.fullmatch(r"VIA-[0-9]{3}", str(packet_id)):
            errors.append(f"requirements: invalid packet id {packet_id!r}")
            continue
        if not isinstance(requirement, Mapping):
            errors.append(f"requirements: {packet_id} must be an object")
            continue
        dependencies = requirement.get("dependencies")
        if not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies)):
            errors.append(f"requirements: {packet_id} dependencies must be a unique array")
            dependencies = []
        wave = requirement.get("execution_wave")
        if not isinstance(wave, int) or wave < 0:
            errors.append(f"requirements: {packet_id} execution_wave must be nonnegative")
        minimum = requirement.get("minimum_evidence")
        if minimum not in evidence_order:
            errors.append(f"requirements: {packet_id} has unknown evidence floor {minimum!r}")
        capabilities = requirement.get("required_capabilities")
        if not isinstance(capabilities, list) or not capabilities:
            errors.append(f"requirements: {packet_id} requires capabilities")
        elif len(capabilities) != len(set(capabilities)):
            errors.append(f"requirements: {packet_id} capabilities must be unique")

        for dependency in dependencies:
            if dependency not in packets:
                errors.append(f"requirements: {packet_id} has unknown dependency {dependency}")
                continue
            dependency_wave = packets[dependency].get("execution_wave")
            if isinstance(wave, int) and isinstance(dependency_wave, int):
                if dependency_wave >= wave:
                    errors.append(
                        f"requirements: {packet_id} wave {wave} must follow dependency "
                        f"{dependency} wave {dependency_wave}"
                    )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(packet_id: str) -> None:
        if packet_id in visiting:
            errors.append(f"requirements: dependency cycle includes {packet_id}")
            return
        if packet_id in visited:
            return
        visiting.add(packet_id)
        requirement = packets.get(packet_id, {})
        for dependency in requirement.get("dependencies", []):
            if dependency in packets:
                visit(dependency)
        visiting.remove(packet_id)
        visited.add(packet_id)

    for packet_id in packets:
        visit(packet_id)

    tiers = requirements.get("tiers")
    if not isinstance(tiers, Mapping):
        errors.append("requirements: tiers must be an object")
    else:
        for tier in ("R", "G", "E", "extension"):
            packet_ids = tiers.get(tier)
            if not isinstance(packet_ids, list) or not packet_ids:
                errors.append(f"requirements: tier {tier} must name packets")
                continue
            if len(packet_ids) != len(set(packet_ids)):
                errors.append(f"requirements: tier {tier} packet ids must be unique")
            for packet_id in packet_ids:
                if packet_id not in packets:
                    errors.append(f"requirements: tier {tier} has unknown packet {packet_id}")

    receipt_requirements = requirements.get("required_receipt_kinds")
    if not isinstance(receipt_requirements, Mapping):
        errors.append("requirements: required_receipt_kinds must be an object")
    else:
        for level in evidence_order:
            kinds = receipt_requirements.get(level)
            if not isinstance(kinds, list):
                errors.append(f"requirements: evidence level {level} must define receipt kinds")
            elif len(kinds) != len(set(kinds)):
                errors.append(f"requirements: receipt kinds for {level} must be unique")
    return errors


def _receipt_map(
    packet: Mapping[str, Any], packet_path: Path, root: Path
) -> tuple[dict[str, Any], list[str]]:
    receipts: dict[str, Any] = {}
    errors: list[str] = []
    for index, receipt in enumerate(packet.get("receipts", [])):
        receipt_id = receipt.get("id")
        if receipt_id in receipts:
            errors.append(f"packet {packet.get('packet_id')}: duplicate receipt id {receipt_id!r}")
            continue
        receipts[receipt_id] = receipt
        resolved = _resolve_inside(packet_path.parent, receipt.get("path", ""), root)
        if resolved is None:
            errors.append(
                f"packet {packet.get('packet_id')}: receipt {receipt_id!r} path escapes repository"
            )
            continue
        receipt["_resolved_path"] = resolved
        if not resolved.is_file():
            message = (
                f"packet {packet.get('packet_id')}: receipt {receipt_id!r} "
                f"does not exist: {resolved}"
            )
            errors.append(message)
            continue
        observed = _sha256(resolved)
        if observed != receipt.get("sha256"):
            errors.append(
                f"packet {packet.get('packet_id')}: receipt {receipt_id!r} hash mismatch "
                f"({observed} != {receipt.get('sha256')})"
            )
    return receipts, errors


def _json_pointer(document: Any, pointer: str) -> Any:
    current = document
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, Mapping):
            current = current[token]
        else:
            raise KeyError(pointer)
    return current


def _binding_values(
    rules: Mapping[str, Any], receipts: Mapping[str, Any], packet_id: str
) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    errors: list[str] = []
    cache: dict[str, Any] = {}
    for name, binding in rules.get("bindings", {}).items():
        receipt_id = binding.get("receipt_id")
        receipt = receipts.get(receipt_id)
        if receipt is None:
            errors.append(
                f"packet {packet_id}: binding {name!r} has unknown receipt {receipt_id!r}"
            )
            continue
        path = receipt.get("_resolved_path")
        if not isinstance(path, Path) or not path.is_file():
            errors.append(f"packet {packet_id}: binding {name!r} receipt is unavailable")
            continue
        try:
            if receipt_id not in cache:
                cache[receipt_id] = _load_json(path)
            values[name] = _json_pointer(cache[receipt_id], binding.get("json_pointer", ""))
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"packet {packet_id}: binding {name!r} cannot resolve: {exc}")
    return values, errors


def _operand_value(operand: Mapping[str, Any], bindings: Mapping[str, Any]) -> Any:
    if "value" in operand:
        return operand["value"]
    return bindings[operand["binding"]]


def _evaluate_expression(expression: Mapping[str, Any], bindings: Mapping[str, Any]) -> bool:
    if "literal" in expression:
        return bool(expression["literal"])
    if "all" in expression:
        return all(_evaluate_expression(item, bindings) for item in expression["all"])
    if "any" in expression:
        return any(_evaluate_expression(item, bindings) for item in expression["any"])
    if "not" in expression:
        return not _evaluate_expression(expression["not"], bindings)
    comparison = expression["compare"]
    left = _operand_value(comparison["left"], bindings)
    right = _operand_value(comparison["right"], bindings)
    operation = comparison["op"]
    if operation == "eq":
        return left == right
    if operation == "ne":
        return left != right
    if operation == "gt":
        return left > right
    if operation == "ge":
        return left >= right
    if operation == "lt":
        return left < right
    if operation == "le":
        return left <= right
    raise ValueError(f"unsupported comparison operation {operation!r}")


def _required_receipt_kinds(requirements: Mapping[str, Any], achieved: str) -> set[str]:
    kinds: set[str] = set()
    for level in requirements["evidence_order"]:
        kinds.update(requirements["required_receipt_kinds"][level])
        if level == achieved:
            return kinds
    return kinds


def _validate_custody(
    packet: Mapping[str, Any], receipts: Mapping[str, Any], packet_id: str
) -> list[str]:
    errors: list[str] = []
    seats = packet["seats"]
    sessions: dict[str, str] = {}
    for name in SEPARATE_SESSION_SEATS:
        session = seats[name]["session_id"]
        if session in sessions:
            message = (
                f"packet {packet_id}: prohibited shared session between "
                f"{sessions[session]} and {name}"
            )
            errors.append(message)
        sessions[session] = name

    custodian_identity = seats["evaluator_custodian"]["agent_identity"]
    for name in BLIND_SEATS | {"adjudicator"}:
        if seats[name]["agent_identity"] == custodian_identity:
            errors.append(f"packet {packet_id}: evaluator_custodian may not also be {name}")

    custody = packet["blind_custody"]
    reveal = custody["reveal"]
    if reveal["status"] == "unrevealed":
        for field in (
            "authorized_by",
            "revealed_at",
            "post_reveal_holdout_sha256",
            "post_reveal_seed_sha256",
            "post_reveal_holdout_receipt_id",
            "post_reveal_seed_receipt_id",
        ):
            if reveal[field] is not None:
                errors.append(f"packet {packet_id}: unrevealed custody must leave {field} null")
    else:
        for field in (
            "authorized_by",
            "revealed_at",
            "post_reveal_holdout_sha256",
            "post_reveal_seed_sha256",
            "post_reveal_holdout_receipt_id",
            "post_reveal_seed_receipt_id",
        ):
            if reveal[field] is None:
                errors.append(f"packet {packet_id}: revealed custody requires {field}")
        if (
            reveal["post_reveal_holdout_sha256"]
            != custody["hidden_holdout_manifest"]["sha256"]
        ):
            errors.append(f"packet {packet_id}: revealed holdout manifest differs from commitment")
        if reveal["post_reveal_seed_sha256"] != custody["secret_seed_manifest"]["sha256"]:
            errors.append(f"packet {packet_id}: revealed seed manifest differs from commitment")
        for prefix, expected in (
            ("holdout", custody["hidden_holdout_manifest"]["sha256"]),
            ("seed", custody["secret_seed_manifest"]["sha256"]),
        ):
            receipt_id = reveal[f"post_reveal_{prefix}_receipt_id"]
            receipt = receipts.get(receipt_id)
            if receipt is None or receipt.get("kind") != "revealed-manifest":
                message = (
                    f"packet {packet_id}: post-reveal {prefix} manifest receipt "
                    "is missing or mistyped"
                )
                errors.append(message)
            elif receipt.get("sha256") != expected:
                errors.append(
                    f"packet {packet_id}: post-reveal {prefix} bytes differ from commitment"
                )

        commitment = custody["output_commitment"]
        if commitment is None:
            errors.append(f"packet {packet_id}: reveal requires a prior output commitment")
        else:
            receipt = receipts.get(commitment["receipt_id"])
            if receipt is None or receipt.get("kind") != "output-commitment":
                errors.append(
                    f"packet {packet_id}: output commitment receipt is missing or mistyped"
                )
            if reveal["revealed_at"] is not None:
                if _parse_datetime(commitment["committed_at"]) >= _parse_datetime(
                    reveal["revealed_at"]
                ):
                    errors.append(f"packet {packet_id}: reveal must follow output commitment")

    for name in BLIND_SEATS:
        exposure = seats[name]["exposure"]
        if any(exposure.values()):
            errors.append(f"packet {packet_id}: blind seat {name} records prohibited exposure")
    if reveal["status"] == "unrevealed":
        for name, seat in seats.items():
            if name == "evaluator_custodian":
                continue
            if any(seat["exposure"].values()):
                errors.append(f"packet {packet_id}: seat {name} saw hidden data before reveal")
    return errors


def _validate_review_chain(
    packet: Mapping[str, Any], receipts: Mapping[str, Any], packet_id: str
) -> list[str]:
    errors: list[str] = []
    chain = packet["review_chain"]
    finding_ids = [result["id"] for result in chain["findings"]]
    if len(finding_ids) != len(set(finding_ids)):
        errors.append(f"packet {packet_id}: review finding ids must be unique")
    test_ids = [result["id"] for result in chain["requested_tests"]]
    if len(test_ids) != len(set(test_ids)):
        errors.append(f"packet {packet_id}: requested test ids must be unique")
    initial = chain["initial_review_receipt"]
    if initial is None or initial not in receipts:
        errors.append(
            f"packet {packet_id}: a passing packet requires an independent review receipt"
        )
    elif receipts[initial].get("kind") != "independent-review":
        errors.append(f"packet {packet_id}: initial review receipt has the wrong kind")

    for result in chain["findings"]:
        if result["blocking"] and result["outcome"] == "unresolved":
            errors.append(f"packet {packet_id}: blocking finding {result['id']} is unresolved")
        if result["outcome"] == "superseded" and not result["superseding_id"]:
            errors.append(
                f"packet {packet_id}: superseded finding {result['id']} lacks a successor"
            )
    for result in chain["requested_tests"]:
        if result["blocking"] and result["outcome"] == "unresolved":
            errors.append(
                f"packet {packet_id}: blocking requested test {result['id']} is unresolved"
            )
        if result["outcome"] == "superseded" and not result["superseding_id"]:
            errors.append(
                f"packet {packet_id}: superseded requested test {result['id']} lacks a successor"
            )

    if chain["findings"] or chain["requested_tests"]:
        if not chain["response_receipts"] or not chain["rereview_receipts"]:
            errors.append(
                f"packet {packet_id}: findings/tests require response and re-review receipts"
            )
        if len(chain["response_receipts"]) != len(chain["rereview_receipts"]):
            errors.append(f"packet {packet_id}: response/re-review round counts differ")
    for receipt_id in chain["response_receipts"]:
        if receipt_id not in receipts or receipts[receipt_id].get("kind") != "builder-response":
            errors.append(f"packet {packet_id}: invalid builder response receipt {receipt_id!r}")
    for receipt_id in chain["rereview_receipts"]:
        if receipt_id not in receipts or receipts[receipt_id].get("kind") != "independent-rereview":
            errors.append(
                f"packet {packet_id}: invalid independent re-review receipt {receipt_id!r}"
            )
    return errors


def _validate_packet(
    packet: Mapping[str, Any],
    packet_path: Path,
    campaign: Mapping[str, Any],
    requirements: Mapping[str, Any],
    root: Path,
    campaign_base: Path,
    packet_outcomes: Mapping[str, str],
    packet_schema: Mapping[str, Any],
) -> list[str]:
    packet_id = str(packet.get("packet_id", "<unknown>"))
    errors = _schema_errors(packet, packet_schema, f"packet {packet_id}")
    if errors:
        return errors

    campaign_fields = (
        "campaign_id",
        "candidate_commit",
        "baseline_commit",
        "tree_hash",
        "protocol_commit",
    )
    for field in campaign_fields:
        if packet[field] != campaign[field]:
            errors.append(f"packet {packet_id}: {field} differs from campaign")

    canonical_claims = _canonical_claim_ids(root)
    for claim in packet["claims"]:
        if claim not in canonical_claims:
            errors.append(f"packet {packet_id}: unknown claim id {claim}")
    canonical_gates = _canonical_gate_ids(root)
    for gate in packet["existing_gates"]:
        if gate not in canonical_gates:
            errors.append(f"packet {packet_id}: unknown gate id {gate}")

    requirement = requirements["packets"].get(packet_id)
    if requirement is None:
        errors.append(f"packet {packet_id}: absent from campaign requirements")
        return errors

    evidence_order = requirements["evidence_order"]
    declared = packet["declared_evidence_requirement"]
    minimum = requirement["minimum_evidence"]
    if evidence_order.index(declared) < evidence_order.index(minimum):
        errors.append(
            f"packet {packet_id}: declared evidence {declared} is below campaign floor {minimum}"
        )

    phase = packet["lifecycle_phase"]
    if LIFECYCLE_ORDER[phase] >= LIFECYCLE_ORDER["preregistered"]:
        missing_capabilities = set(requirement["required_capabilities"]) - set(
            packet["capabilities"]
        )
        if missing_capabilities:
            errors.append(
                f"packet {packet_id}: missing required capabilities {sorted(missing_capabilities)}"
            )
        declared_capabilities = set(packet["capabilities"])
        bound_capabilities = set(packet["capability_rules"])
        if declared_capabilities != bound_capabilities:
            errors.append(
                f"packet {packet_id}: capability declarations and rules differ "
                f"({sorted(declared_capabilities)} != {sorted(bound_capabilities)})"
            )
        records_invalid_round = (
            phase == "adjudicated" and packet["adjudication"]["round_status"] == "invalid"
        )
        if packet["outcome_rules"] is None and not records_invalid_round:
            errors.append(f"packet {packet_id}: preregistration requires executable outcome rules")

    receipts, receipt_errors = _receipt_map(packet, packet_path, campaign_base)
    errors.extend(receipt_errors)
    errors.extend(_validate_custody(packet, receipts, packet_id))

    if packet["holdout_started"]:
        if LIFECYCLE_ORDER[phase] < LIFECYCLE_ORDER["attacked"]:
            errors.append(f"packet {packet_id}: holdout cannot start before attack phase")
        for dependency in requirement["dependencies"]:
            if packet_outcomes.get(dependency) != "passed":
                errors.append(
                    f"packet {packet_id}: holdout started before dependency {dependency} passed"
                )

    adjudication = packet["adjudication"]
    round_status = adjudication["round_status"]
    outcome = adjudication["packet_outcome"]
    causes = set(adjudication["cause_codes"])
    if phase != "adjudicated":
        if round_status != "not-run" or outcome != "pending" or causes != {"not-run"}:
            errors.append(f"packet {packet_id}: non-adjudicated packet must remain not-run/pending")
        return errors

    reveal_required = round_status == "valid" or packet["holdout_started"]
    if reveal_required and packet["blind_custody"]["reveal"]["status"] != "revealed":
        errors.append(f"packet {packet_id}: adjudication requires a recorded reveal")

    decisive = adjudication["decisive_receipts"]
    if not decisive:
        errors.append(f"packet {packet_id}: adjudication requires decisive receipts")
    for receipt_id in decisive:
        if receipt_id not in receipts:
            errors.append(f"packet {packet_id}: unknown decisive receipt {receipt_id!r}")

    if round_status == "invalid":
        if outcome != "pending" or not causes or not causes <= INVALID_CAUSES:
            errors.append(
                f"packet {packet_id}: invalid round requires pending outcome "
                "and invalid cause codes"
            )
        return errors
    if round_status != "valid":
        errors.append(f"packet {packet_id}: adjudicated packet must be valid or invalid")
        return errors

    rules = packet["outcome_rules"]
    if rules is None or rules.get("language") != RULE_LANGUAGE:
        errors.append(f"packet {packet_id}: valid adjudication requires {RULE_LANGUAGE} rules")
        return errors
    bindings, binding_errors = _binding_values(rules, receipts, packet_id)
    errors.extend(binding_errors)
    if binding_errors:
        return errors
    try:
        results = {
            name: _evaluate_expression(rules[name], bindings)
            for name in ("pass", "fail", "blocked")
        }
        capability_results = {
            name: _evaluate_expression(expression, bindings)
            for name, expression in packet["capability_rules"].items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"packet {packet_id}: outcome rule evaluation failed: {exc}")
        return errors
    true_outcomes = [name for name, value in results.items() if value]
    if len(true_outcomes) != 1:
        errors.append(
            f"packet {packet_id}: outcome rules are nonexclusive or incomplete: {results}; "
            "record an invalid round"
        )
        return errors
    computed = {"pass": "passed", "fail": "failed", "blocked": "blocked"}[
        true_outcomes[0]
    ]
    if outcome != computed:
        errors.append(f"packet {packet_id}: declared outcome {outcome} != computed {computed}")
    if outcome == "passed" and not all(capability_results.values()):
        failed_capabilities = sorted(
            name for name, passed in capability_results.items() if not passed
        )
        errors.append(
            f"packet {packet_id}: passing outcome has failed capabilities {failed_capabilities}"
        )

    allowed_causes = {
        "passed": {"pass-rule-satisfied"},
        "failed": FAILURE_CAUSES,
        "blocked": BLOCKAGE_CAUSES,
    }[outcome]
    if not causes or not causes <= allowed_causes:
        errors.append(f"packet {packet_id}: cause codes {sorted(causes)} do not match {outcome}")

    if outcome in {"passed", "failed"}:
        errors.extend(_validate_review_chain(packet, receipts, packet_id))
    if outcome == "passed":
        achieved = adjudication["achieved_evidence"]
        if evidence_order.index(achieved) < evidence_order.index(declared):
            errors.append(
                f"packet {packet_id}: achieved evidence {achieved} is below declared {declared}"
            )
        receipt_kinds = {receipt["kind"] for receipt in receipts.values()}
        required_kinds = _required_receipt_kinds(requirements, achieved)
        missing_kinds = required_kinds - receipt_kinds
        if missing_kinds:
            errors.append(
                f"packet {packet_id}: evidence {achieved} lacks receipt kinds "
                f"{sorted(missing_kinds)}"
            )
    if outcome == "blocked" and "blockage-evidence" not in {
        receipt["kind"] for receipt in receipts.values()
    }:
        errors.append(f"packet {packet_id}: blocked outcome requires blockage evidence")
    return errors


def _campaign_outcome(required_packets: list[str], outcomes: Mapping[str, str]) -> str:
    values = [outcomes.get(packet_id, "pending") for packet_id in required_packets]
    if any(value == "failed" for value in values):
        return "failed"
    if any(value == "blocked" for value in values):
        return "blocked"
    if values and all(value == "passed" for value in values):
        return "passed"
    return "pending"


def validate_campaign(
    campaign_path: Path | str,
    *,
    repo_root: Path | str | None = None,
    requirements_document: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return fail-closed validation errors for one campaign and its packets."""
    path = Path(campaign_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    campaign_schema = _load_json(root / "schemas/viability/campaign-v1.schema.json")
    packet_schema = _load_json(root / "schemas/viability/packet-v1.schema.json")
    requirements = (
        dict(requirements_document)
        if requirements_document is not None
        else _load_json(root / "schemas/viability/requirements-v1.json")
    )
    errors = validate_requirements(requirements)
    if errors:
        return errors
    try:
        campaign = _load_yaml(path)
    except (OSError, yaml.YAMLError) as exc:
        return [f"campaign: cannot load {path}: {exc}"]
    errors.extend(_schema_errors(campaign, campaign_schema, "campaign"))
    if errors:
        return errors

    if campaign["requirements_version"] != requirements["version"]:
        errors.append("campaign: requirements version does not match loaded requirements")
    if campaign["contract_version"] != CONTRACT_VERSION:
        errors.append("campaign: unsupported contract version")

    required_packets = requirements["tiers"][campaign["target_tier"]]
    packet_files = campaign["packet_files"]
    for packet_id in required_packets:
        if packet_id not in packet_files:
            errors.append(f"campaign: target tier lacks required packet {packet_id}")

    loaded: dict[str, tuple[Mapping[str, Any], Path]] = {}
    preliminary_outcomes: dict[str, str] = {}
    for packet_id, relative in packet_files.items():
        packet_path = _resolve_inside(path.parent, relative, path.parent)
        if packet_path is None:
            errors.append(f"campaign: packet path for {packet_id} escapes repository")
            continue
        if not packet_path.is_file():
            errors.append(f"campaign: packet file for {packet_id} does not exist: {packet_path}")
            continue
        try:
            packet = _load_yaml(packet_path)
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"campaign: cannot load packet {packet_id}: {exc}")
            continue
        if not isinstance(packet, Mapping):
            errors.append(f"campaign: packet {packet_id} must be an object")
            continue
        if packet.get("packet_id") != packet_id:
            errors.append(
                f"campaign: packet key {packet_id} != document id {packet.get('packet_id')!r}"
            )
        loaded[packet_id] = (packet, packet_path)
        preliminary_outcomes[packet_id] = packet.get("adjudication", {}).get(
            "packet_outcome", "pending"
        )

    for packet_id, (packet, packet_path) in loaded.items():
        errors.extend(
            _validate_packet(
                packet,
                packet_path,
                campaign,
                requirements,
                root,
                path.parent,
                preliminary_outcomes,
                packet_schema,
            )
        )

    computed = _campaign_outcome(required_packets, preliminary_outcomes)
    declared = campaign["decision"]["outcome"]
    if declared != computed:
        errors.append(f"campaign: declared outcome {declared} != computed {computed}")
    if declared == "pending":
        if campaign["decision"]["authorized_by"] is not None:
            errors.append("campaign: pending decision cannot be authorized")
        if campaign["decision"]["decided_at"] is not None:
            errors.append("campaign: pending decision cannot have decided_at")
    else:
        if not campaign["decision"]["authorized_by"] or not campaign["decision"]["decided_at"]:
            errors.append("campaign: terminal decision requires authorization and timestamp")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path, help="path to CAMPAIGN.yaml")
    args = parser.parse_args(argv)
    errors = validate_campaign(args.campaign)
    if errors:
        print("Viability campaign contract failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Viability campaign contract is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
