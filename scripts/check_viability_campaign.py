"""Validate a POPGP adversarial viability campaign and its packet receipts."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml
from jsonschema import Draft202012Validator, FormatChecker

CONTRACT_VERSION = "popgp-viability-contract-v2"
REQUIREMENTS_VERSION = "popgp-viability-requirements-v2"
RULE_LANGUAGE = "popgp-bool-v2"
PACKET_FREEZE_VERSION = "popgp-packet-freeze-v4"
KNOWN_REQUIREMENTS_SHA256 = (
    "632528e8c4b19d746253719e308b3a676b5a19cffc3a734a670d1c878c161d20"
)

CONTRACT_FILE_PATHS = {
    "docs/scientific_hardening/CLAIMS_MATRIX.md",
    "docs/scientific_hardening/GATE_TEST_REGISTRY.md",
    "schemas/viability/campaign-v2.schema.json",
    "schemas/viability/packet-v2.schema.json",
    "schemas/viability/protocol-manifest-v2.schema.json",
    "schemas/viability/primary-protocol-v1.schema.json",
    "schemas/viability/independent-review-v2.schema.json",
    "schemas/viability/independent-rereview-v2.schema.json",
    "schemas/viability/review-response-v2.schema.json",
    "scripts/check_viability_campaign.py",
}

PACKET_FREEZE_FIELDS = (
    "schema_version",
    "contract_version",
    "campaign_id",
    "packet_id",
    "candidate_commit",
    "baseline_commit",
    "tree_hash",
    "claims",
    "existing_gates",
    "declared_evidence_requirement",
    "capabilities",
    "capability_rules",
    "hypothesis",
    "null_or_competitors",
    "known_failure_to_retain",
    "threat_model",
    "preregistration",
    "external_replication",
    "outcome_rules",
)

REVIEW_SCHEMA_PATHS = {
    "independent-review": "schemas/viability/independent-review-v2.schema.json",
    "independent-rereview": "schemas/viability/independent-rereview-v2.schema.json",
    "builder-response": "schemas/viability/review-response-v2.schema.json",
}

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


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML safe loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_text(text: str) -> Any:
    return yaml.load(text, Loader=_UniqueKeyLoader)


def _load_yaml(path: Path) -> Any:
    return _load_yaml_text(path.read_text(encoding="utf-8"))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _load_json_bytes(content: bytes) -> Any:
    return json.loads(content, object_pairs_hook=_unique_json_object)


def _load_json_text(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_json_object)


def _load_json(path: Path) -> Any:
    return _load_json_text(path.read_text(encoding="utf-8"))


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


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(document: Any) -> bytes:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def packet_rule_sha256(packet: Mapping[str, Any]) -> str:
    """Return the v2 canonical hash of fields frozen before holdout execution."""
    frozen = {field: packet[field] for field in PACKET_FREEZE_FIELDS}
    frozen["seat_assignments"] = {
        seat_name: {
            field: seat[field]
            for field in (
                "agent_identity",
                "model_identity",
                "model_version",
                "operator",
                "session_id",
                "orchestrator_id",
                "organization",
                "access_level",
            )
        }
        for seat_name, seat in packet["seats"].items()
    }
    custody = packet["blind_custody"]
    frozen["blind_custody"] = {
        "hash_algorithm": custody["hash_algorithm"],
        "canonicalization": custody["canonicalization"],
        "custodian_seat": custody["custodian_seat"],
        "hidden_holdout_manifest": custody["hidden_holdout_manifest"],
        "secret_seed_manifest": custody["secret_seed_manifest"],
    }
    return _sha256_bytes(_canonical_json_bytes(frozen))


def _git_output(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        timeout=15,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(detail or f"git {' '.join(args)} failed")
    return result.stdout


@functools.lru_cache(maxsize=256)
def _git_commit_exists(root: Path, commit: str) -> bool:
    try:
        return _git_output(root, "cat-file", "-t", commit).strip() == b"commit"
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


@functools.lru_cache(maxsize=1024)
def _git_blob(root: Path, commit: str, relative_path: str) -> bytes:
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise ValueError(f"unsafe repository path {relative_path!r}")
    return _git_output(root, "show", f"{commit}:{relative_path}")


def _git_path_matches_commit(root: Path, commit: str, relative_path: str) -> bool:
    try:
        working = (root / relative_path).read_bytes().replace(b"\r\n", b"\n")
        frozen = _git_blob(root, commit, relative_path).replace(b"\r\n", b"\n")
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return working == frozen


@functools.lru_cache(maxsize=256)
def _git_tree(root: Path, commit: str) -> str:
    return _git_output(root, "rev-parse", f"{commit}^{{tree}}").decode("ascii").strip()


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        timeout=15,
    )
    return result.returncode == 0


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


def _validate_frozen_inputs(
    campaign: Mapping[str, Any],
    root: Path,
    manifest_schema: Mapping[str, Any],
    supplied_requirements: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, list[str]]:
    errors: list[str] = []
    for field in ("candidate_commit", "baseline_commit", "protocol_commit"):
        commit = campaign[field]
        if not _git_commit_exists(root, commit):
            errors.append(f"campaign: {field} does not resolve to a Git commit: {commit}")

    if _git_commit_exists(root, campaign["candidate_commit"]):
        try:
            observed_tree = _git_tree(root, campaign["candidate_commit"])
        except (OSError, subprocess.SubprocessError, UnicodeDecodeError, ValueError) as exc:
            errors.append(f"campaign: cannot resolve candidate tree: {exc}")
        else:
            if observed_tree != campaign["tree_hash"]:
                errors.append(
                    "campaign: tree_hash does not match candidate commit tree "
                    f"({campaign['tree_hash']} != {observed_tree})"
                )

    protocol_commit = campaign["protocol_commit"]
    if not _git_commit_exists(root, protocol_commit):
        return None, None, errors
    try:
        manifest_bytes = _git_blob(root, protocol_commit, campaign["protocol_manifest_path"])
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        errors.append(f"campaign: protocol manifest is absent from protocol commit: {exc}")
        return None, None, errors
    observed_manifest_hash = _sha256_bytes(manifest_bytes)
    if observed_manifest_hash != campaign["protocol_manifest_sha256"]:
        errors.append(
            "campaign: protocol manifest hash mismatch "
            f"({observed_manifest_hash} != {campaign['protocol_manifest_sha256']})"
        )
    try:
        manifest = _load_json_bytes(manifest_bytes)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"campaign: protocol manifest is not valid UTF-8 JSON: {exc}")
        return None, None, errors
    manifest_errors = _schema_errors(manifest, manifest_schema, "protocol_manifest")
    errors.extend(manifest_errors)
    if manifest_errors:
        return None, manifest, errors

    for field in (
        "campaign_id",
        "candidate_commit",
        "baseline_commit",
        "tree_hash",
        "contract_version",
        "requirements_version",
    ):
        if manifest[field] != campaign[field]:
            errors.append(f"campaign: protocol manifest {field} differs from campaign")
    if manifest["packet_freeze_version"] != PACKET_FREEZE_VERSION:
        errors.append("campaign: unsupported packet freeze version")

    contract_refs = manifest["contract_files"]
    contract_paths = [reference["path"] for reference in contract_refs]
    if len(contract_paths) != len(set(contract_paths)):
        errors.append("campaign: protocol manifest contract file paths must be unique")
    if set(contract_paths) != CONTRACT_FILE_PATHS:
        errors.append(
            "campaign: protocol manifest contract file set differs from v2 authority "
            f"({sorted(contract_paths)} != {sorted(CONTRACT_FILE_PATHS)})"
        )
    for reference in contract_refs:
        relative = reference["path"]
        try:
            frozen_bytes = _git_blob(root, protocol_commit, relative)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(f"campaign: frozen contract file {relative!r} is unavailable: {exc}")
            continue
        observed = _sha256_bytes(frozen_bytes)
        if observed != reference["sha256"]:
            errors.append(
                f"campaign: frozen contract file {relative!r} hash mismatch "
                f"({observed} != {reference['sha256']})"
            )
        if not _git_path_matches_commit(root, protocol_commit, relative):
            errors.append(
                f"campaign: executing contract file {relative!r} differs from protocol commit"
            )

    requirements_ref = manifest["requirements"]
    if campaign["requirements_path"] != requirements_ref["path"]:
        errors.append("campaign: requirements path differs from protocol manifest")
    if campaign["requirements_sha256"] != requirements_ref["sha256"]:
        errors.append("campaign: requirements hash differs from protocol manifest")
    try:
        requirements_bytes = _git_blob(root, protocol_commit, requirements_ref["path"])
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        errors.append(f"campaign: frozen requirements are unavailable: {exc}")
        return None, manifest, errors
    observed_requirements_hash = _sha256_bytes(requirements_bytes)
    if observed_requirements_hash != requirements_ref["sha256"]:
        errors.append(
            "campaign: frozen requirements hash mismatch "
            f"({observed_requirements_hash} != {requirements_ref['sha256']})"
        )
    if observed_requirements_hash != KNOWN_REQUIREMENTS_SHA256:
        errors.append(
            f"campaign: {REQUIREMENTS_VERSION} has noncanonical content hash "
            f"{observed_requirements_hash}; change the requirements version"
        )
    try:
        frozen_requirements = _load_json_bytes(requirements_bytes)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"campaign: frozen requirements are not valid UTF-8 JSON: {exc}")
        return None, manifest, errors
    if supplied_requirements is not None and supplied_requirements != frozen_requirements:
        errors.append("campaign: supplied requirements differ from frozen protocol requirements")
    return frozen_requirements, manifest, errors


def validate_requirements(requirements: Any) -> list[str]:
    """Validate the campaign-owned packet DAG and evidence floors."""
    errors: list[str] = []
    if not isinstance(requirements, Mapping):
        return ["requirements: document must be an object"]
    if requirements.get("version") != REQUIREMENTS_VERSION:
        errors.append(f"requirements: version must be {REQUIREMENTS_VERSION!r}")

    evidence_order = requirements.get("evidence_order")
    if (
        not isinstance(evidence_order, list)
        or not evidence_order
        or any(not isinstance(level, str) for level in evidence_order)
    ):
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
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(item, str) for item in dependencies)
            or len(dependencies) != len(set(dependencies))
        ):
            errors.append(f"requirements: {packet_id} dependencies must be a unique array")
            dependencies = []
        wave = requirement.get("execution_wave")
        if type(wave) is not int or wave < 0:
            errors.append(f"requirements: {packet_id} execution_wave must be nonnegative")
        minimum = requirement.get("minimum_evidence")
        if minimum not in evidence_order:
            errors.append(f"requirements: {packet_id} has unknown evidence floor {minimum!r}")
        capabilities = requirement.get("required_capabilities")
        if (
            not isinstance(capabilities, list)
            or not capabilities
            or any(not isinstance(item, str) for item in capabilities)
        ):
            errors.append(f"requirements: {packet_id} requires capabilities")
        elif len(capabilities) != len(set(capabilities)):
            errors.append(f"requirements: {packet_id} capabilities must be unique")

        for dependency in dependencies:
            if dependency not in packets:
                errors.append(f"requirements: {packet_id} has unknown dependency {dependency}")
                continue
            dependency_requirement = packets[dependency]
            if not isinstance(dependency_requirement, Mapping):
                continue
            dependency_wave = dependency_requirement.get("execution_wave")
            if type(wave) is int and type(dependency_wave) is int:
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
        if not isinstance(requirement, Mapping):
            visiting.remove(packet_id)
            visited.add(packet_id)
            return
        dependencies = requirement.get("dependencies", [])
        if not isinstance(dependencies, list):
            dependencies = []
        for dependency in dependencies:
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
            if (
                not isinstance(packet_ids, list)
                or not packet_ids
                or any(not isinstance(item, str) for item in packet_ids)
            ):
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
            if not isinstance(kinds, list) or any(not isinstance(kind, str) for kind in kinds):
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
        if receipt.get("kind") != "raw-results":
            errors.append(
                f"packet {packet_id}: binding {name!r} must use a raw-results receipt"
            )
            continue
        if receipt.get("media_type") != "application/json":
            errors.append(
                f"packet {packet_id}: binding {name!r} must use application/json"
            )
            continue
        path = receipt.get("_resolved_path")
        if not isinstance(path, Path) or not path.is_file():
            errors.append(f"packet {packet_id}: binding {name!r} receipt is unavailable")
            continue
        try:
            if receipt_id not in cache:
                cache[receipt_id] = _load_json(path)
            value = _json_pointer(cache[receipt_id], binding.get("json_pointer", ""))
            observed_type = _json_value_type(value)
            expected_type = binding.get("expected_type")
            if observed_type != expected_type:
                raise TypeError(
                    f"expected {expected_type}, observed {observed_type}"
                )
            if observed_type == "number" and not math.isfinite(value):
                raise TypeError("non-finite numeric binding")
            values[name] = value
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"packet {packet_id}: binding {name!r} cannot resolve: {exc}")
    return values, errors


def _json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) in {int, float}:
        return "number"
    if type(value) is str:
        return "string"
    return "array" if isinstance(value, list) else "object"


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
    left_type = _json_value_type(left)
    right_type = _json_value_type(right)
    if left_type != right_type:
        raise TypeError(
            f"comparison operands have different JSON types: {left_type} != {right_type}"
        )
    if operation in {"gt", "ge", "lt", "le"} and left_type != "number":
        raise TypeError(f"ordered comparison requires numbers, observed {left_type}")
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


def _expression_bindings(expression: Mapping[str, Any]) -> set[str]:
    if "literal" in expression:
        return set()
    if "all" in expression:
        return set().union(*(_expression_bindings(item) for item in expression["all"]))
    if "any" in expression:
        return set().union(*(_expression_bindings(item) for item in expression["any"]))
    if "not" in expression:
        return _expression_bindings(expression["not"])
    bindings: set[str] = set()
    for operand in (expression["compare"]["left"], expression["compare"]["right"]):
        if "binding" in operand:
            bindings.add(operand["binding"])
    return bindings


def _canonical_capability_expression(capability: str) -> Mapping[str, Any]:
    return {
        "compare": {
            "left": {"binding": capability},
            "op": "eq",
            "right": {"value": True},
        }
    }


def _required_receipt_kinds(requirements: Mapping[str, Any], achieved: str) -> set[str]:
    kinds: set[str] = set()
    for level in requirements["evidence_order"]:
        kinds.update(requirements["required_receipt_kinds"][level])
        if level == achieved:
            return kinds
    return kinds


def _structured_receipt_document(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    path = receipt.get("_resolved_path")
    if not isinstance(path, Path) or not path.is_file():
        raise ValueError("receipt file is unavailable")
    media_type = receipt.get("media_type")
    text = path.read_text(encoding="utf-8")
    if media_type == "application/json":
        document = _load_json_text(text)
    elif media_type in {"application/yaml", "text/yaml"}:
        document = _load_yaml_text(text)
    elif media_type in {"text/markdown", "text/x-markdown"}:
        match = re.search(r"```yaml\s*(.*?)\s*```", text, flags=re.DOTALL)
        if match is None:
            raise ValueError("Markdown receipt lacks a fenced YAML artifact")
        document = _load_yaml_text(match.group(1))
    else:
        raise ValueError(f"unsupported structured receipt media type {media_type!r}")
    if not isinstance(document, Mapping):
        raise ValueError("structured receipt must contain an object")
    return document


def _artifact_ref_parts(artifact_ref: str) -> tuple[str, str]:
    commit, separator, relative = artifact_ref.partition(":")
    if (
        not separator
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
    ):
        raise ValueError(f"invalid immutable artifact ref {artifact_ref!r}")
    return commit, relative


def _artifact_ref_errors(
    artifact_ref: str,
    receipt: Mapping[str, Any],
    root: Path,
    packet_id: str,
    label: str,
) -> list[str]:
    errors: list[str] = []
    try:
        commit, relative = _artifact_ref_parts(artifact_ref)
    except (TypeError, ValueError) as exc:
        return [f"packet {packet_id}: {label} has invalid immutable ref: {exc}"]
    if not _git_commit_exists(root, commit):
        return [f"packet {packet_id}: {label} ref commit does not exist: {commit}"]
    try:
        frozen_bytes = _git_blob(root, commit, relative)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return [f"packet {packet_id}: {label} ref blob is unavailable: {exc}"]
    observed = _sha256_bytes(frozen_bytes)
    if observed != receipt.get("sha256"):
        errors.append(
            f"packet {packet_id}: {label} receipt bytes differ from immutable ref "
            f"({receipt.get('sha256')} != {observed})"
        )
    return errors


def _review_commit_binding_errors(
    document: Mapping[str, Any], packet: Mapping[str, Any], root: Path, packet_id: str
) -> list[str]:
    errors: list[str] = []
    reviewed = document.get("commit_reviewed")
    if not isinstance(reviewed, str) or not _git_commit_exists(root, reviewed):
        return [f"packet {packet_id}: review commit_reviewed does not resolve: {reviewed!r}"]
    try:
        tree_hash = _git_tree(root, reviewed)
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"packet {packet_id}: cannot resolve review context tree: {exc}")
    else:
        if document.get("context_hash") != tree_hash:
            errors.append(f"packet {packet_id}: review context_hash differs from reviewed tree")
    if document.get("baseline_commit") != packet.get("baseline_commit"):
        errors.append(f"packet {packet_id}: review baseline differs from packet baseline")
    expected_method = f'git rev-parse "{reviewed}^{{tree}}"'
    if document.get("context_hash_method") != expected_method:
        errors.append(f"packet {packet_id}: review context_hash_method is not canonical")
    builder = packet["seats"]["builder"]
    declaration = document["independence_declaration"]
    if declaration["builder_model_identity"] != builder["model_identity"]:
        errors.append(
            f"packet {packet_id}: review builder model differs from packet builder"
        )
    model_differs = document["reviewer_model_identity"] != builder["model_identity"]
    if declaration["reviewer_model_differs_from_builder"] != model_differs:
        errors.append(
            f"packet {packet_id}: reviewer model-separation declaration is contradictory"
        )
    shared_operator = document["reviewer_operator"] == builder["operator"]
    if declaration["shared_operator"] != shared_operator:
        errors.append(
            f"packet {packet_id}: reviewer shared-operator declaration is contradictory"
        )
    if declaration["builder_session_id"] != builder["session_id"]:
        errors.append(f"packet {packet_id}: review builder session differs from packet builder")
    shared_session = document["reviewer_session_id"] == builder["session_id"]
    if declaration["shared_session"] != shared_session:
        errors.append(
            f"packet {packet_id}: reviewer shared-session declaration is contradictory"
        )
    if declaration["builder_orchestrator_id"] != builder["orchestrator_id"]:
        errors.append(
            f"packet {packet_id}: review builder orchestrator differs from packet builder"
        )
    shared_orchestrator = (
        document["reviewer_orchestrator_id"] == builder["orchestrator_id"]
    )
    if declaration["shared_orchestrator"] != shared_orchestrator:
        errors.append(
            f"packet {packet_id}: reviewer shared-orchestrator declaration is contradictory"
        )
    return errors


def _response_provenance_errors(
    response: Mapping[str, Any],
    rereview: Mapping[str, Any],
    packet: Mapping[str, Any],
    root: Path,
    packet_id: str,
    round_index: int,
) -> list[str]:
    errors: list[str] = []
    builder = packet["seats"]["builder"]
    if response.get("builder_model_identity") != builder["model_identity"]:
        errors.append(
            f"packet {packet_id}: response round {round_index} builder model is unrelated"
        )
    if response.get("builder_operator") != builder["operator"]:
        errors.append(
            f"packet {packet_id}: response round {round_index} builder operator is unrelated"
        )
    if response.get("builder_session_id") != builder["session_id"]:
        errors.append(
            f"packet {packet_id}: response round {round_index} builder session is unrelated"
        )
    if response.get("builder_orchestrator_id") != builder["orchestrator_id"]:
        errors.append(
            f"packet {packet_id}: response round {round_index} builder orchestrator is unrelated"
        )
    if response.get("builder_organization") != builder["organization"]:
        errors.append(
            f"packet {packet_id}: response round {round_index} builder organization is unrelated"
        )
    rereview_declaration = rereview["independence_declaration"]
    if rereview_declaration["builder_model_identity"] != response.get(
        "builder_model_identity"
    ):
        errors.append(
            f"packet {packet_id}: re-review round {round_index} builder identity differs "
            "from response"
        )
    reviewed_commit = rereview["commit_reviewed"]
    for item in response["finding_responses"]:
        for fix_commit in item["fix_commits"]:
            if not _git_commit_exists(root, fix_commit):
                errors.append(
                    f"packet {packet_id}: response round {round_index} fix commit "
                    f"does not exist: {fix_commit}"
                )
            elif not _git_is_ancestor(root, fix_commit, reviewed_commit):
                errors.append(
                    f"packet {packet_id}: response round {round_index} fix commit "
                    f"is not an ancestor of reviewed candidate: {fix_commit}"
                )
    return errors


def _validate_preregistration(
    packet: Mapping[str, Any],
    receipts: Mapping[str, Any],
    packet_id: str,
    root: Path,
) -> list[str]:
    errors: list[str] = []
    artifacts = packet["preregistration"]["protocol_artifacts"]
    receipt_ids = [artifact["receipt_id"] for artifact in artifacts]
    protocol_paths = [artifact["protocol_path"] for artifact in artifacts]
    if len(receipt_ids) != len(set(receipt_ids)):
        errors.append(f"packet {packet_id}: preregistered protocol receipt ids must be unique")
    if len(protocol_paths) != len(set(protocol_paths)):
        errors.append(f"packet {packet_id}: preregistered protocol paths must be unique")
    primary_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("content_role") == "primary-protocol"
    ]
    if len(primary_artifacts) != 1:
        errors.append(f"packet {packet_id}: preregistration requires one primary protocol")
    registered = set(receipt_ids)
    declared = {
        receipt_id
        for receipt_id, receipt in receipts.items()
        if receipt.get("kind") == "protocol"
    }
    if declared != registered:
        errors.append(
            f"packet {packet_id}: protocol receipts differ from frozen preregistration "
            f"({sorted(declared)} != {sorted(registered)})"
        )
    for artifact in artifacts:
        receipt_id = artifact["receipt_id"]
        receipt = receipts.get(receipt_id)
        if receipt is None:
            errors.append(
                f"packet {packet_id}: frozen protocol receipt {receipt_id!r} is missing"
            )
            continue
        expected = {
            "kind": "protocol",
            "path": artifact["campaign_path"],
            "sha256": artifact["sha256"],
            "media_type": artifact["media_type"],
        }
        for field, value in expected.items():
            if receipt.get(field) != value:
                errors.append(
                    f"packet {packet_id}: protocol receipt {receipt_id!r} {field} "
                    "differs from frozen preregistration"
                )
        try:
            frozen_bytes = _git_blob(
                root, packet["protocol_commit"], artifact["protocol_path"]
            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(
                f"packet {packet_id}: preregistered protocol blob "
                f"{artifact['protocol_path']!r} is unavailable: {exc}"
            )
            continue
        observed = _sha256_bytes(frozen_bytes)
        if observed != artifact["sha256"]:
            errors.append(
                f"packet {packet_id}: preregistered protocol blob hash mismatch "
                f"({observed} != {artifact['sha256']})"
            )
        if artifact.get("content_role") == "primary-protocol":
            if receipt.get("media_type") != "application/json":
                errors.append(
                    f"packet {packet_id}: primary protocol must use application/json"
                )
                continue
            try:
                document = _structured_receipt_document(receipt)
            except (
                OSError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
                yaml.YAMLError,
            ) as exc:
                errors.append(
                    f"packet {packet_id}: primary protocol cannot be parsed: {exc}"
                )
                continue
            try:
                primary_schema = _load_json(
                    root / "schemas/viability/primary-protocol-v1.schema.json"
                )
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                errors.append(f"packet {packet_id}: cannot load primary protocol schema: {exc}")
                continue
            protocol_schema_errors = _schema_errors(
                document, primary_schema, f"packet {packet_id} primary-protocol"
            )
            errors.extend(protocol_schema_errors)
            expected_document = {
                "schema_version": 1,
                "packet_id": packet_id,
                **{
                    field: packet["preregistration"][field]
                    for field in (
                        "parameters",
                        "measurement_procedure",
                        "uncertainty_procedure",
                        "statistical_analysis",
                        "resource_budget",
                        "commands",
                        "mutation_plan",
                    )
                },
            }
            if document != expected_document:
                errors.append(
                    f"packet {packet_id}: primary protocol differs from exact "
                    "frozen preregistration envelope"
                )
    return errors


def _structured_external_receipt(
    receipts: Mapping[str, Any],
    receipt_id: str,
    expected_kind: str,
    packet_id: str,
    label: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    receipt = receipts.get(receipt_id)
    if receipt is None or receipt.get("kind") != expected_kind:
        errors.append(
            f"packet {packet_id}: external replication {label} receipt is missing or mistyped"
        )
        return None
    if receipt.get("media_type") != "application/json":
        errors.append(
            f"packet {packet_id}: external replication {label} must use application/json"
        )
        return None
    try:
        return _structured_receipt_document(receipt)
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as exc:
        errors.append(
            f"packet {packet_id}: external replication {label} receipt cannot be parsed: {exc}"
        )
        return None


def _canonical_repository_identity(value: str) -> str:
    """Normalize common Git URL/path aliases for repository-identity comparisons."""
    raw = value.strip().replace("\\", "/")
    scp_match = re.fullmatch(r"(?:[^@/]+@)?([^:/]+):(.+)", raw)
    if scp_match and "://" not in raw:
        raw = f"ssh://{scp_match.group(1)}/{scp_match.group(2)}"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw.rstrip("/").removesuffix(".git").lower()
    if parsed.scheme and parsed.scheme != "file":
        authority = parsed.netloc.rsplit("@", 1)[-1].lower()
        path = unquote(parsed.path).replace("//", "/").rstrip("/")
        if path.lower().endswith(".git"):
            path = path[:-4]
        return f"{authority}{path}".lower()
    if parsed.scheme == "file":
        raw = unquote(parsed.path)
    try:
        normalized = str(Path(raw).resolve()).replace("\\", "/").rstrip("/")
    except OSError:
        normalized = raw.rstrip("/")
    if normalized.lower().endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def _external_git_bundle_errors(
    receipts: Mapping[str, Any],
    implementation: Mapping[str, Any],
    packet_id: str,
) -> list[str]:
    """Resolve the declared external commit/tree from a content-addressed Git bundle."""
    errors: list[str] = []
    receipt_id = implementation["repository_bundle_receipt_id"]
    receipt = receipts.get(receipt_id)
    if receipt is None or receipt.get("kind") != "independent-repository-bundle":
        return [f"packet {packet_id}: external repository bundle receipt is missing"]
    if receipt.get("media_type") != "application/x-git-bundle":
        errors.append(
            f"packet {packet_id}: external repository bundle has wrong media type"
        )
    if receipt.get("sha256") != implementation["repository_bundle_sha256"]:
        errors.append(f"packet {packet_id}: external repository bundle hash differs")
    bundle_path = receipt.get("_resolved_path")
    if not isinstance(bundle_path, Path) or not bundle_path.is_file():
        errors.append(f"packet {packet_id}: external repository bundle is unavailable")
        return errors
    try:
        with tempfile.TemporaryDirectory(prefix="popgp-external-bundle-") as temporary:
            checkout = Path(temporary) / "repository"
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-checkout",
                    str(bundle_path),
                    str(checkout),
                ],
                check=False,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace").strip()
                errors.append(
                    f"packet {packet_id}: external repository bundle cannot be cloned: "
                    f"{detail or result.returncode}"
                )
                return errors
            commit = implementation["commit_hash"]
            if not _git_commit_exists(checkout, commit):
                errors.append(
                    f"packet {packet_id}: external implementation commit is absent "
                    "from repository bundle"
                )
            else:
                observed_tree = _git_tree(checkout, commit)
                if observed_tree != implementation["tree_hash"]:
                    errors.append(
                        f"packet {packet_id}: external implementation tree differs "
                        "from repository bundle"
                    )
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"packet {packet_id}: external repository bundle is invalid: {exc}")
    return errors


def _validate_external_replication(
    packet: Mapping[str, Any],
    receipts: Mapping[str, Any],
    packet_id: str,
    campaign: Mapping[str, Any],
    internal_facts: Mapping[str, set[str]],
) -> list[str]:
    errors: list[str] = []
    contract = packet["external_replication"]
    if packet_id != "VIA-900":
        if contract is not None:
            errors.append(
                f"packet {packet_id}: external replication contract is reserved for VIA-900"
            )
        return errors
    if contract is None:
        if LIFECYCLE_ORDER[packet["lifecycle_phase"]] >= LIFECYCLE_ORDER["preregistered"]:
            errors.append(f"packet {packet_id}: Tier E requires external replication contract")
        return errors

    organization = contract["organization"]
    operator = contract["operator"]
    if operator["organization_id"] != organization["id"]:
        errors.append(f"packet {packet_id}: external operator organization id differs")
    comparisons = {
        "agent identity": (operator["agent_identity"], "agent_identities"),
        "operator": (operator["operator"], "operators"),
        "organization": (organization["id"], "organizations"),
        "organization name": (organization["name"], "organizations"),
        "model identity": (operator["model_identity"], "model_identities"),
        "session": (operator["session_id"], "session_ids"),
        "orchestrator": (operator["orchestrator_id"], "orchestrator_ids"),
    }
    for label, (value, fact_key) in comparisons.items():
        if value in internal_facts[fact_key]:
            errors.append(
                f"packet {packet_id}: external {label} is not distinct from internal campaign"
            )
    implementation = contract["implementation"]
    if _canonical_repository_identity(
        implementation["repository"]
    ) == _canonical_repository_identity(campaign["repository"]):
        errors.append(f"packet {packet_id}: external implementation reuses candidate repository")
    if implementation["commit_hash"] == campaign["candidate_commit"]:
        errors.append(f"packet {packet_id}: external implementation reuses candidate commit")
    if implementation["tree_hash"] == campaign["tree_hash"]:
        errors.append(f"packet {packet_id}: external implementation reuses candidate tree")
    errors.extend(_external_git_bundle_errors(receipts, implementation, packet_id))

    contract_document = _structured_external_receipt(
        receipts,
        contract["receipt_id"],
        "external-replication",
        packet_id,
        "contract",
        errors,
    )
    if contract_document is not None and contract_document != {
        "packet_id": packet_id,
        "contract": contract,
    }:
        errors.append(
            f"packet {packet_id}: external replication receipt differs from packet contract"
        )

    provenance_document = _structured_external_receipt(
        receipts,
        implementation["provenance_receipt_id"],
        "independent-implementation",
        packet_id,
        "implementation provenance",
        errors,
    )
    if provenance_document is not None and provenance_document != {
        "packet_id": packet_id,
        "organization": organization,
        "operator": operator,
        "implementation": implementation,
    }:
        errors.append(
            f"packet {packet_id}: independent implementation receipt differs from contract"
        )

    prediction = contract["prediction"]
    prediction_receipt = receipts.get(prediction["receipt_id"])
    prediction_document = _structured_external_receipt(
        receipts,
        prediction["receipt_id"],
        "blinded-prediction",
        packet_id,
        "blinded prediction",
        errors,
    )
    if prediction_receipt is not None and prediction_receipt.get("sha256") != prediction[
        "sha256"
    ]:
        errors.append(f"packet {packet_id}: blinded prediction hash differs from receipt")
    if prediction["committed_by"] != operator["agent_identity"]:
        errors.append(f"packet {packet_id}: blinded prediction has wrong committer")
    if prediction_document is not None:
        if set(prediction_document) != {
            "packet_id",
            "committed_by",
            "committed_at",
            "predictions",
        }:
            errors.append(f"packet {packet_id}: blinded prediction envelope is invalid")
        if any(
            prediction_document.get(field) != prediction[field]
            for field in ("committed_by", "committed_at")
        ) or prediction_document.get("packet_id") != packet_id:
            errors.append(f"packet {packet_id}: blinded prediction metadata differs")
        if not isinstance(prediction_document.get("predictions"), list):
            errors.append(f"packet {packet_id}: blinded predictions must be an array")

    reveal_document = _structured_external_receipt(
        receipts,
        prediction["reveal_receipt_id"],
        "reveal-record",
        packet_id,
        "prediction reveal",
        errors,
    )
    evaluator = packet["seats"]["evaluator_custodian"]["agent_identity"]
    if reveal_document is not None and reveal_document != {
        "packet_id": packet_id,
        "authorized_by": evaluator,
        "revealed_at": packet["blind_custody"]["reveal"]["revealed_at"],
        "prediction_receipt_id": prediction["receipt_id"],
        "prediction_sha256": prediction["sha256"],
    }:
        errors.append(f"packet {packet_id}: blinded prediction reveal differs from contract")

    reproduction = contract["reproduction"]
    output_receipt = receipts.get(reproduction["output_receipt_id"])
    if output_receipt is None or output_receipt.get("kind") != "raw-results":
        errors.append(f"packet {packet_id}: external raw output receipt is missing")
    elif output_receipt.get("sha256") != reproduction["output_sha256"]:
        errors.append(f"packet {packet_id}: external raw output hash differs")
    output_document = _structured_external_receipt(
        receipts,
        reproduction["output_receipt_id"],
        "raw-results",
        packet_id,
        "raw output",
        errors,
    )
    if reproduction["committed_by"] != operator["agent_identity"]:
        errors.append(f"packet {packet_id}: external output has wrong committer")
    commitment_document = _structured_external_receipt(
        receipts,
        reproduction["commitment_receipt_id"],
        "output-commitment",
        packet_id,
        "output commitment",
        errors,
    )
    if commitment_document is not None and commitment_document != {
        "packet_id": packet_id,
        "committed_by": reproduction["committed_by"],
        "committed_at": reproduction["committed_at"],
        "output_receipt_id": reproduction["output_receipt_id"],
        "output_sha256": reproduction["output_sha256"],
    }:
        errors.append(f"packet {packet_id}: external output commitment differs")

    comparison = contract["comparison"]
    candidate_receipt = receipts.get(comparison["candidate_output_receipt_id"])
    if candidate_receipt is None or candidate_receipt.get("kind") != "raw-results":
        errors.append(f"packet {packet_id}: comparison candidate output is missing")
    custody_commitment = packet["blind_custody"]["output_commitment"]
    if not isinstance(custody_commitment, Mapping):
        errors.append(f"packet {packet_id}: comparison requires candidate output commitment")
    else:
        if (
            comparison["candidate_output_receipt_id"]
            != custody_commitment["output_receipt_id"]
        ):
            errors.append(
                f"packet {packet_id}: comparison candidate output differs from custody output"
            )
        if comparison["candidate_output_sha256"] != custody_commitment["output_sha256"]:
            errors.append(
                f"packet {packet_id}: comparison candidate hash differs from custody output"
            )
    if candidate_receipt is not None and candidate_receipt.get("sha256") != comparison[
        "candidate_output_sha256"
    ]:
        errors.append(f"packet {packet_id}: comparison candidate output hash differs")
    if comparison["external_output_receipt_id"] != reproduction["output_receipt_id"]:
        errors.append(f"packet {packet_id}: comparison external output differs")
    if comparison["external_output_sha256"] != reproduction["output_sha256"]:
        errors.append(f"packet {packet_id}: comparison external output hash differs")
    if comparison["candidate_output_receipt_id"] == comparison[
        "external_output_receipt_id"
    ]:
        errors.append(f"packet {packet_id}: comparison reuses one output receipt")
    if candidate_receipt is not None and output_receipt is not None:
        if candidate_receipt.get("sha256") == output_receipt.get("sha256"):
            errors.append(f"packet {packet_id}: comparison output bytes are not distinct")
        if candidate_receipt.get("_resolved_path") == output_receipt.get("_resolved_path"):
            errors.append(f"packet {packet_id}: comparison output paths are not distinct")

    candidate_document = _structured_external_receipt(
        receipts,
        comparison["candidate_output_receipt_id"],
        "raw-results",
        packet_id,
        "candidate comparison output",
        errors,
    )
    candidate_value: int | float | None = None
    external_value: int | float | None = None
    agreement: bool | None = None
    if candidate_document is not None and output_document is not None:
        try:
            candidate_metric = _json_pointer(
                candidate_document, comparison["metric_json_pointer"]
            )
            external_metric = _json_pointer(
                output_document, comparison["metric_json_pointer"]
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"packet {packet_id}: comparison metric cannot resolve: {exc}")
        else:
            tolerance = comparison["absolute_tolerance"]
            numeric_values = (candidate_metric, external_metric, tolerance)
            if any(type(value) not in {int, float} for value in numeric_values) or any(
                type(value) is float and not math.isfinite(value)
                for value in numeric_values
            ):
                errors.append(f"packet {packet_id}: comparison metrics must be finite numbers")
            else:
                try:
                    difference = abs(
                        Decimal(str(candidate_metric)) - Decimal(str(external_metric))
                    )
                    agreement = difference <= Decimal(str(tolerance))
                except (InvalidOperation, ValueError):
                    errors.append(
                        f"packet {packet_id}: comparison metrics cannot be evaluated"
                    )
                else:
                    candidate_value = candidate_metric
                    external_value = external_metric

    adjudicator = packet["seats"]["adjudicator"]["agent_identity"]
    if comparison["compared_by"] != adjudicator:
        errors.append(f"packet {packet_id}: comparison must be made by adjudicator")
    comparison_document = _structured_external_receipt(
        receipts,
        comparison["receipt_id"],
        "cross-implementation-comparison",
        packet_id,
        "comparison",
        errors,
    )
    if agreement is not None:
        expected_comparison = {
            "packet_id": packet_id,
            "compared_by": comparison["compared_by"],
            "compared_at": comparison["compared_at"],
            "method": comparison["method"],
            "metric_json_pointer": comparison["metric_json_pointer"],
            "absolute_tolerance": comparison["absolute_tolerance"],
            "candidate_output_receipt_id": comparison["candidate_output_receipt_id"],
            "candidate_output_sha256": comparison["candidate_output_sha256"],
            "candidate_value": candidate_value,
            "external_output_receipt_id": comparison["external_output_receipt_id"],
            "external_output_sha256": comparison["external_output_sha256"],
            "external_value": external_value,
            "agreement": agreement,
        }
        if comparison_document is not None and comparison_document != expected_comparison:
            errors.append(
                f"packet {packet_id}: comparison receipt differs from computed outputs"
            )
        causes = set(packet["adjudication"]["cause_codes"])
        outcome = packet["adjudication"]["packet_outcome"]
        disagreement_cause = "external-replication-disagreed"
        if agreement and disagreement_cause in causes:
            errors.append(
                f"packet {packet_id}: agreement cannot claim external disagreement cause"
            )
        if not agreement and (
            outcome != "failed" or disagreement_cause not in causes
        ):
            errors.append(
                f"packet {packet_id}: external disagreement requires failed adjudication"
            )

    try:
        prediction_time = _parse_datetime(prediction["committed_at"])
        output_time = _parse_datetime(reproduction["committed_at"])
        reveal_time = _parse_datetime(packet["blind_custody"]["reveal"]["revealed_at"])
        comparison_time = _parse_datetime(comparison["compared_at"])
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(f"packet {packet_id}: external replication timestamps are invalid: {exc}")
    else:
        if not prediction_time < output_time < reveal_time < comparison_time:
            errors.append(
                f"packet {packet_id}: prediction, external output, reveal, and comparison "
                "order is invalid"
            )
    return errors


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
            "reveal_receipt_id",
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
            "reveal_receipt_id",
        ):
            if reveal[field] is None:
                errors.append(f"packet {packet_id}: revealed custody requires {field}")
        if reveal["authorized_by"] != custodian_identity:
            errors.append(
                f"packet {packet_id}: reveal must be authorized by evaluator_custodian"
            )
        if not packet["holdout_started"]:
            errors.append(f"packet {packet_id}: reveal cannot precede holdout execution")
        if LIFECYCLE_ORDER[packet["lifecycle_phase"]] < LIFECYCLE_ORDER["reproduced"]:
            errors.append(f"packet {packet_id}: reveal cannot precede reproduction")
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
            runner_identity = seats["reproduction_runner"]["agent_identity"]
            if commitment["committed_by"] != runner_identity:
                errors.append(
                    f"packet {packet_id}: output commitment must be made by reproduction_runner"
                )
            output_receipt = receipts.get(commitment["output_receipt_id"])
            if output_receipt is None or output_receipt.get("kind") != "raw-results":
                errors.append(
                    f"packet {packet_id}: output commitment must reference raw-results"
                )
            elif output_receipt.get("sha256") != commitment["output_sha256"]:
                errors.append(
                    f"packet {packet_id}: output commitment hash differs from runner output"
                )
            if receipt is not None and receipt.get("kind") == "output-commitment":
                try:
                    receipt_document = _structured_receipt_document(receipt)
                except (
                    OSError,
                    UnicodeDecodeError,
                    ValueError,
                    json.JSONDecodeError,
                    yaml.YAMLError,
                ) as exc:
                    errors.append(
                        f"packet {packet_id}: output commitment receipt cannot be parsed: {exc}"
                    )
                else:
                    expected_commitment = {
                        "packet_id": packet_id,
                        "committed_by": commitment["committed_by"],
                        "committed_at": commitment["committed_at"],
                        "output_receipt_id": commitment["output_receipt_id"],
                        "output_sha256": commitment["output_sha256"],
                    }
                    for field, expected in expected_commitment.items():
                        if receipt_document.get(field) != expected:
                            errors.append(
                                f"packet {packet_id}: output commitment receipt {field} "
                                "differs from packet record"
                            )
            if reveal["revealed_at"] is not None:
                if _parse_datetime(commitment["committed_at"]) >= _parse_datetime(
                    reveal["revealed_at"]
                ):
                    errors.append(f"packet {packet_id}: reveal must follow output commitment")

        reveal_receipt = receipts.get(reveal["reveal_receipt_id"])
        if reveal_receipt is None or reveal_receipt.get("kind") != "reveal-record":
            errors.append(f"packet {packet_id}: reveal record receipt is missing or mistyped")
        else:
            try:
                reveal_document = _structured_receipt_document(reveal_receipt)
            except (
                OSError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
                yaml.YAMLError,
            ) as exc:
                errors.append(f"packet {packet_id}: reveal record cannot be parsed: {exc}")
            else:
                expected_reveal = {
                    "packet_id": packet_id,
                    "authorized_by": reveal["authorized_by"],
                    "revealed_at": reveal["revealed_at"],
                    "output_commitment_receipt_id": (
                        commitment["receipt_id"] if commitment is not None else None
                    ),
                    "post_reveal_holdout_sha256": reveal[
                        "post_reveal_holdout_sha256"
                    ],
                    "post_reveal_seed_sha256": reveal["post_reveal_seed_sha256"],
                }
                for field, expected in expected_reveal.items():
                    if reveal_document.get(field) != expected:
                        errors.append(
                            f"packet {packet_id}: reveal record {field} differs from packet record"
                        )

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


def _artifact_items(
    document: Mapping[str, Any], key: str, id_key: str, packet_id: str, errors: list[str]
) -> list[Mapping[str, Any]]:
    items = document.get(key, [])
    if not isinstance(items, list) or any(not isinstance(item, Mapping) for item in items):
        errors.append(f"packet {packet_id}: review artifact {key} must be an object array")
        return []
    valid_items: list[Mapping[str, Any]] = []
    identifiers: list[str] = []
    for item in items:
        identifier = item.get(id_key)
        if not isinstance(identifier, str) or not identifier:
            errors.append(f"packet {packet_id}: review artifact {key} has an invalid id")
            continue
        identifiers.append(identifier)
        valid_items.append(item)
    if len(identifiers) != len(set(identifiers)):
        errors.append(f"packet {packet_id}: review artifact {key} ids must be unique")
    return valid_items


def _receipt_artifact(
    receipts: Mapping[str, Any],
    receipt_id: str | None,
    artifact_ref: str | None,
    expected_kind: str,
    packet_id: str,
    root: Path,
    errors: list[str],
) -> Mapping[str, Any] | None:
    receipt = receipts.get(receipt_id)
    if receipt is None or receipt.get("kind") != expected_kind:
        errors.append(
            f"packet {packet_id}: invalid {expected_kind} receipt {receipt_id!r}"
        )
        return None
    if not isinstance(artifact_ref, str):
        errors.append(f"packet {packet_id}: {expected_kind} receipt lacks immutable ref")
        return None
    errors.extend(
        _artifact_ref_errors(artifact_ref, receipt, root, packet_id, expected_kind)
    )
    try:
        document = _structured_receipt_document(receipt)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        errors.append(
            f"packet {packet_id}: {expected_kind} receipt {receipt_id!r} cannot be parsed: {exc}"
        )
        return None
    try:
        schema = _load_json(root / REVIEW_SCHEMA_PATHS[expected_kind])
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"packet {packet_id}: cannot load {expected_kind} schema: {exc}")
        return None
    schema_errors = _schema_errors(document, schema, f"packet {packet_id} {expected_kind}")
    errors.extend(schema_errors)
    if schema_errors:
        return None
    return document


def _review_recommendation_errors(
    document: Mapping[str, Any],
    findings: Mapping[str, Mapping[str, Any]],
    packet_id: str,
) -> list[str]:
    recommendation = document.get("recommendation")
    if not isinstance(recommendation, Mapping):
        return [f"packet {packet_id}: review artifact lacks a recommendation object"]
    unresolved = sum(
        1
        for result in findings.values()
        if result["blocking"] and result["outcome"] == "unresolved"
    )
    errors: list[str] = []
    blocker_count = recommendation.get("blocking_findings")
    if type(blocker_count) is not int or blocker_count != unresolved:
        errors.append(
            f"packet {packet_id}: review recommendation blocker count differs from artifact"
        )
    approve = recommendation.get("approve")
    if type(approve) is not bool or approve != (unresolved == 0):
        errors.append(f"packet {packet_id}: review approval differs from blocker state")
    return errors


def _validate_review_chain(
    packet: Mapping[str, Any], receipts: Mapping[str, Any], packet_id: str, root: Path
) -> list[str]:
    errors: list[str] = []
    chain = packet["review_chain"]
    initial = _receipt_artifact(
        receipts,
        chain["initial_review_receipt"],
        chain["initial_review_ref"],
        "independent-review",
        packet_id,
        root,
        errors,
    )
    if initial is None:
        return errors
    if initial.get("review_kind") != "initial" or not initial.get("review_id"):
        errors.append(f"packet {packet_id}: initial review artifact identity is invalid")
    errors.extend(_review_commit_binding_errors(initial, packet, root, packet_id))

    findings: dict[str, dict[str, Any]] = {}
    requested_tests: dict[str, dict[str, Any]] = {}
    for item in _artifact_items(initial, "findings", "id", packet_id, errors):
        finding_id = item.get("id")
        if not isinstance(finding_id, str) or not finding_id:
            continue
        if type(item.get("blocking")) is not bool:
            errors.append(f"packet {packet_id}: finding {finding_id} lacks Boolean blocking")
            continue
        findings[finding_id] = {
            "id": finding_id,
            "blocking": item["blocking"],
            "outcome": "unresolved",
            "superseding_id": None,
        }
    for item in _artifact_items(
        initial, "requested_tests", "id", packet_id, errors
    ):
        test_id = item.get("id")
        if not isinstance(test_id, str) or not test_id:
            continue
        if type(item.get("blocking")) is not bool:
            errors.append(
                f"packet {packet_id}: requested test {test_id} lacks Boolean blocking"
            )
            continue
        requested_tests[test_id] = {
            "id": test_id,
            "blocking": item["blocking"],
            "outcome": "unresolved",
            "superseding_id": None,
        }
    errors.extend(_review_recommendation_errors(initial, findings, packet_id))

    responses = chain["response_receipts"]
    response_refs = chain["response_refs"]
    rereviews = chain["rereview_receipts"]
    rereview_refs = chain["rereview_refs"]
    if not (
        len(responses)
        == len(response_refs)
        == len(rereviews)
        == len(rereview_refs)
    ):
        errors.append(f"packet {packet_id}: response/re-review receipt/ref counts differ")
    if (findings or requested_tests) and not responses:
        errors.append(f"packet {packet_id}: findings/tests require response and re-review receipts")

    prior_review_id = initial.get("review_id")
    prior_review_ref = chain["initial_review_ref"]
    prior_review_document = initial
    final_review_document = initial
    for round_index, (response_id, response_ref, rereview_id, rereview_ref) in enumerate(
        zip(responses, response_refs, rereviews, rereview_refs, strict=False), start=1
    ):
        active_finding_ids = {
            identifier
            for identifier, result in findings.items()
            if result["outcome"] == "unresolved"
        }
        active_test_ids = {
            identifier
            for identifier, result in requested_tests.items()
            if result["outcome"] == "unresolved"
        }
        response = _receipt_artifact(
            receipts,
            response_id,
            response_ref,
            "builder-response",
            packet_id,
            root,
            errors,
        )
        rereview = _receipt_artifact(
            receipts,
            rereview_id,
            rereview_ref,
            "independent-rereview",
            packet_id,
            root,
            errors,
        )
        if response is None or rereview is None:
            continue
        if response.get("review_id") != prior_review_id:
            errors.append(
                f"packet {packet_id}: response round {round_index} targets the wrong review"
            )
        if response.get("response_round") != round_index:
            errors.append(f"packet {packet_id}: response round number is not sequential")
        response_prior_ref = (
            f"{response.get('review_commit')}:{response.get('review_artifact')}"
        )
        if response_prior_ref != prior_review_ref:
            errors.append(
                f"packet {packet_id}: response round {round_index} does not bind prior review ref"
            )
        if response.get("candidate_commit_reviewed") != prior_review_document.get(
            "commit_reviewed"
        ):
            errors.append(
                f"packet {packet_id}: response round {round_index} targets wrong candidate"
            )
        if rereview.get("prior_review_ref") != prior_review_ref:
            errors.append(
                f"packet {packet_id}: re-review round {round_index} prior ref mismatch"
            )
        if rereview.get("builder_response_ref") != response_ref:
            errors.append(
                f"packet {packet_id}: re-review round {round_index} response ref mismatch"
            )
        errors.extend(_review_commit_binding_errors(rereview, packet, root, packet_id))
        errors.extend(
            _response_provenance_errors(
                response, rereview, packet, root, packet_id, round_index
            )
        )
        response_findings = _artifact_items(
            response, "finding_responses", "finding_id", packet_id, errors
        )
        response_tests = _artifact_items(
            response,
            "requested_test_responses",
            "requested_test_id",
            packet_id,
            errors,
        )
        if {item.get("finding_id") for item in response_findings} != active_finding_ids:
            errors.append(
                f"packet {packet_id}: response round {round_index} finding coverage is incomplete"
            )
        if {item.get("requested_test_id") for item in response_tests} != active_test_ids:
            errors.append(
                f"packet {packet_id}: response round {round_index} test coverage is incomplete"
            )
        if rereview.get("review_kind") != "re-review" or not rereview.get("review_id"):
            errors.append(f"packet {packet_id}: re-review round {round_index} identity is invalid")

        prior_findings = _artifact_items(
            rereview, "prior_finding_results", "finding_id", packet_id, errors
        )
        prior_tests = _artifact_items(
            rereview,
            "prior_requested_test_results",
            "requested_test_id",
            packet_id,
            errors,
        )
        if {item.get("finding_id") for item in prior_findings} != active_finding_ids:
            errors.append(
                f"packet {packet_id}: re-review round {round_index} prior finding "
                "coverage is incomplete"
            )
        if {item.get("requested_test_id") for item in prior_tests} != active_test_ids:
            errors.append(
                f"packet {packet_id}: re-review round {round_index} prior test "
                "coverage is incomplete"
            )

        new_findings = _artifact_items(rereview, "findings", "id", packet_id, errors)
        new_tests = _artifact_items(rereview, "requested_tests", "id", packet_id, errors)
        new_finding_ids = {item.get("id") for item in new_findings}
        new_test_ids = {item.get("id") for item in new_tests}
        if new_finding_ids & set(findings):
            errors.append(f"packet {packet_id}: re-review reuses an existing finding id")
        if new_test_ids & set(requested_tests):
            errors.append(f"packet {packet_id}: re-review reuses an existing requested test id")

        for result in prior_findings:
            finding_id = result.get("finding_id")
            if finding_id not in active_finding_ids:
                continue
            outcome = result.get("outcome")
            successor = result.get("superseding_finding_id") or None
            if not isinstance(outcome, str) or outcome not in {
                "verified-resolved",
                "unresolved",
                "superseded",
            }:
                errors.append(f"packet {packet_id}: finding {finding_id} has invalid outcome")
                continue
            if outcome == "superseded" and successor not in new_finding_ids:
                errors.append(
                    f"packet {packet_id}: superseded finding {finding_id} has no declared successor"
                )
            if outcome != "superseded" and successor is not None:
                errors.append(
                    f"packet {packet_id}: nonsuperseded finding {finding_id} names a successor"
                )
            findings[finding_id]["outcome"] = outcome
            findings[finding_id]["superseding_id"] = successor
        for result in prior_tests:
            test_id = result.get("requested_test_id")
            if test_id not in active_test_ids:
                continue
            outcome = result.get("outcome")
            successor = result.get("superseding_requested_test_id") or None
            if not isinstance(outcome, str) or outcome not in {
                "verified-satisfied",
                "unresolved",
                "superseded",
            }:
                errors.append(f"packet {packet_id}: requested test {test_id} has invalid outcome")
                continue
            if outcome == "superseded" and successor not in new_test_ids:
                errors.append(
                    f"packet {packet_id}: superseded requested test {test_id} "
                    "has no declared successor"
                )
            if outcome != "superseded" and successor is not None:
                errors.append(
                    f"packet {packet_id}: nonsuperseded requested test {test_id} names a successor"
                )
            requested_tests[test_id]["outcome"] = outcome
            requested_tests[test_id]["superseding_id"] = successor

        for item in new_findings:
            finding_id = item.get("id")
            if not isinstance(finding_id, str) or finding_id in findings:
                continue
            if type(item.get("blocking")) is not bool:
                errors.append(f"packet {packet_id}: finding {finding_id} lacks Boolean blocking")
                continue
            findings[finding_id] = {
                "id": finding_id,
                "blocking": item["blocking"],
                "outcome": "unresolved",
                "superseding_id": None,
            }
        for item in new_tests:
            test_id = item.get("id")
            if not isinstance(test_id, str) or test_id in requested_tests:
                continue
            if type(item.get("blocking")) is not bool:
                errors.append(
                    f"packet {packet_id}: requested test {test_id} lacks Boolean blocking"
                )
                continue
            requested_tests[test_id] = {
                "id": test_id,
                "blocking": item["blocking"],
                "outcome": "unresolved",
                "superseding_id": None,
            }
        errors.extend(_review_recommendation_errors(rereview, findings, packet_id))
        prior_review_id = rereview.get("review_id")
        prior_review_ref = rereview_ref
        prior_review_document = rereview
        final_review_document = rereview

    if final_review_document.get("commit_reviewed") != packet.get("candidate_commit"):
        errors.append(f"packet {packet_id}: final review does not audit packet candidate")

    declared_finding_ids = [item["id"] for item in chain["findings"]]
    declared_test_ids = [item["id"] for item in chain["requested_tests"]]
    if len(declared_finding_ids) != len(set(declared_finding_ids)):
        errors.append(f"packet {packet_id}: declared review finding ids must be unique")
    if len(declared_test_ids) != len(set(declared_test_ids)):
        errors.append(f"packet {packet_id}: declared requested test ids must be unique")
    declared_findings = {item["id"]: item for item in chain["findings"]}
    declared_tests = {item["id"]: item for item in chain["requested_tests"]}
    if declared_findings != findings:
        errors.append(f"packet {packet_id}: declared findings differ from hashed review artifacts")
    if declared_tests != requested_tests:
        errors.append(
            f"packet {packet_id}: declared requested tests differ from hashed review artifacts"
        )
    for result in findings.values():
        if result["blocking"] and result["outcome"] == "unresolved":
            errors.append(f"packet {packet_id}: blocking finding {result['id']} is unresolved")
    for result in requested_tests.values():
        if result["blocking"] and result["outcome"] == "unresolved":
            errors.append(
                f"packet {packet_id}: blocking requested test {result['id']} is unresolved"
            )
    return errors


def _validate_packet(
    packet: Mapping[str, Any],
    packet_path: Path,
    campaign: Mapping[str, Any],
    requirements: Mapping[str, Any],
    protocol_manifest: Mapping[str, Any],
    root: Path,
    campaign_base: Path,
    packet_outcomes: Mapping[str, str],
    packet_schema: Mapping[str, Any],
    internal_facts: Mapping[str, set[str]],
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

    try:
        computed_rule_hash = packet_rule_sha256(packet)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"packet {packet_id}: cannot compute frozen packet rules: {exc}")
    else:
        if packet["protocol_rule_sha256"] != computed_rule_hash:
            errors.append(
                f"packet {packet_id}: protocol_rule_sha256 differs from packet rules"
            )
        expected_rule_hash = protocol_manifest["packet_rule_sha256"].get(packet_id)
        if expected_rule_hash is None:
            errors.append(f"packet {packet_id}: absent from frozen protocol manifest")
        elif expected_rule_hash != computed_rule_hash:
            errors.append(f"packet {packet_id}: packet rules differ from protocol snapshot")

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
        rules = packet["outcome_rules"]
        if rules is not None:
            bindings = rules["bindings"]
            for rule_name in ("pass", "fail", "blocked"):
                referenced = _expression_bindings(rules[rule_name])
                if not referenced:
                    errors.append(
                        f"packet {packet_id}: {rule_name} rule must consume raw-result bindings"
                    )
                unknown = referenced - set(bindings)
                if unknown:
                    errors.append(
                        f"packet {packet_id}: {rule_name} rule has unknown bindings "
                        f"{sorted(unknown)}"
                    )
            for capability in packet["capabilities"]:
                binding = bindings.get(capability)
                if binding is None:
                    errors.append(
                        f"packet {packet_id}: capability {capability} lacks its raw-result binding"
                    )
                    continue
                if binding["expected_type"] != "boolean":
                    errors.append(
                        f"packet {packet_id}: capability {capability} binding must be Boolean"
                    )
                expected_pointer = f"/capabilities/{capability}"
                if binding["json_pointer"] != expected_pointer:
                    errors.append(
                        f"packet {packet_id}: capability {capability} binding must use "
                        f"{expected_pointer}"
                    )
                if packet["capability_rules"][capability] != (
                    _canonical_capability_expression(capability)
                ):
                    errors.append(
                        f"packet {packet_id}: capability {capability} must use its canonical "
                        "raw Boolean gate"
                    )

    receipts, receipt_errors = _receipt_map(packet, packet_path, campaign_base)
    errors.extend(receipt_errors)
    if LIFECYCLE_ORDER[phase] >= LIFECYCLE_ORDER["preregistered"]:
        errors.extend(_validate_preregistration(packet, receipts, packet_id, root))
        errors.extend(
            _validate_external_replication(
                packet, receipts, packet_id, campaign, internal_facts
            )
        )
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
    if computed == "passed" and not all(capability_results.values()):
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
    }[computed]
    if not causes or not causes <= allowed_causes:
        errors.append(
            f"packet {packet_id}: cause codes {sorted(causes)} do not match {computed}"
        )

    if computed in {"passed", "failed"}:
        errors.extend(_validate_review_chain(packet, receipts, packet_id, root))
    if computed == "passed":
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
    if computed == "blocked" and "blockage-evidence" not in {
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
    try:
        campaign_schema = _load_json(root / "schemas/viability/campaign-v2.schema.json")
        packet_schema = _load_json(root / "schemas/viability/packet-v2.schema.json")
        manifest_schema = _load_json(
            root / "schemas/viability/protocol-manifest-v2.schema.json"
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        return [f"campaign: cannot load contract schemas: {exc}"]
    errors: list[str] = []
    try:
        campaign = _load_yaml(path)
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        return [f"campaign: cannot load {path}: {exc}"]
    errors.extend(_schema_errors(campaign, campaign_schema, "campaign"))
    if errors:
        return errors

    requirements, protocol_manifest, freeze_errors = _validate_frozen_inputs(
        campaign, root, manifest_schema, requirements_document
    )
    errors.extend(freeze_errors)
    if requirements is None or protocol_manifest is None:
        return errors
    errors.extend(validate_requirements(requirements))
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
    extra_packets = set(packet_files) - set(required_packets)
    if extra_packets:
        errors.append(f"campaign: target tier has extra packet files {sorted(extra_packets)}")
    missing_frozen_rules = set(required_packets) - set(
        protocol_manifest["packet_rule_sha256"]
    )
    if missing_frozen_rules:
        errors.append(
            f"campaign: protocol manifest lacks packet rules {sorted(missing_frozen_rules)}"
        )

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
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
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
        adjudication = packet.get("adjudication")
        preliminary_outcomes[packet_id] = (
            adjudication.get("packet_outcome", "pending")
            if isinstance(adjudication, Mapping)
            else "pending"
        )

    internal_facts: dict[str, set[str]] = {
        "agent_identities": set(),
        "operators": set(),
        "organizations": set(),
        "model_identities": set(),
        "session_ids": set(),
        "orchestrator_ids": set(),
    }
    fact_fields = {
        "agent_identity": "agent_identities",
        "operator": "operators",
        "organization": "organizations",
        "model_identity": "model_identities",
        "session_id": "session_ids",
        "orchestrator_id": "orchestrator_ids",
    }
    for packet, _packet_path in loaded.values():
        seats = packet.get("seats")
        if not isinstance(seats, Mapping):
            continue
        for seat in seats.values():
            if not isinstance(seat, Mapping):
                continue
            for field, fact_key in fact_fields.items():
                value = seat.get(field)
                if isinstance(value, str):
                    internal_facts[fact_key].add(value)

    for packet_id, (packet, packet_path) in loaded.items():
        errors.extend(
            _validate_packet(
                packet,
                packet_path,
                campaign,
                requirements,
                protocol_manifest,
                root,
                path.parent,
                preliminary_outcomes,
                packet_schema,
                internal_facts,
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
    parser.add_argument("campaign", nargs="?", type=Path, help="path to CAMPAIGN.yaml")
    parser.add_argument(
        "--packet-rule-sha256",
        type=Path,
        metavar="PACKET.yaml",
        help="print the canonical preregistration hash for one packet",
    )
    parser.add_argument(
        "--git-blob-sha256",
        nargs=2,
        metavar=("COMMIT", "PATH"),
        help="print the SHA-256 of a repository blob at a frozen commit",
    )
    args = parser.parse_args(argv)
    if args.packet_rule_sha256 is not None:
        if args.campaign is not None or args.git_blob_sha256 is not None:
            parser.error("packet hashing cannot be combined with campaign validation")
        try:
            packet = _load_yaml(args.packet_rule_sha256)
            print(packet_rule_sha256(packet))
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            print(f"Cannot hash packet rules: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.git_blob_sha256 is not None:
        if args.campaign is not None:
            parser.error("Git blob hashing cannot be combined with campaign validation")
        commit, relative = args.git_blob_sha256
        try:
            print(_sha256_bytes(_git_blob(_repo_root(), commit, relative)))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            print(f"Cannot hash Git blob: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.campaign is None:
        parser.error("CAMPAIGN.yaml is required unless a hashing mode is selected")
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
