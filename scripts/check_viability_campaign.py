"""Validate a POPGP adversarial viability campaign and its packet receipts."""

from __future__ import annotations

import argparse
import functools
import hashlib
import ipaddress
import json
import math
import os
import posixpath
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from pypdf import PdfReader
from pypdf.errors import PdfReadError

CONTRACT_VERSION = "popgp-viability-contract-v2"
REQUIREMENTS_VERSION = "popgp-viability-requirements-v2"
RULE_LANGUAGE = "popgp-bool-v2"
PACKET_FREEZE_VERSION = "popgp-packet-freeze-v4"
MAX_STRUCTURED_NESTING = 128
MAX_STRUCTURED_EXPANDED_NODES = 100_000
MAX_STRUCTURED_INPUT_BYTES = 16 * 1024 * 1024
MAX_JSON_NUMBER_CHARACTERS = 256
GIT_BUNDLE_TIMEOUT_SECONDS = 30
PROCESS_TREE_CLEANUP_SECONDS = 5
GIT_BUNDLE_TOTAL_TIMEOUT_SECONDS = GIT_BUNDLE_TIMEOUT_SECONDS + PROCESS_TREE_CLEANUP_SECONDS
KNOWN_REQUIREMENTS_SHA256 = "632528e8c4b19d746253719e308b3a676b5a19cffc3a734a670d1c878c161d20"

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


def _reject_duplicate_explicit_keys(
    loader: _UniqueKeyLoader,
    node: yaml.Node,
    deep: bool,
    checked: set[int],
    active: set[int],
) -> None:
    marker = id(node)
    if marker in checked or marker in active:
        return
    active.add(marker)
    if isinstance(node, yaml.MappingNode):
        explicit_keys: set[Any] = set()
        for key_node, value_node in node.value:
            if key_node.tag != "tag:yaml.org,2002:merge":
                key = loader.construct_object(key_node, deep=deep)
                try:
                    duplicate = key in explicit_keys
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
                explicit_keys.add(key)
            _reject_duplicate_explicit_keys(loader, key_node, deep, checked, active)
            _reject_duplicate_explicit_keys(loader, value_node, deep, checked, active)
    elif isinstance(node, yaml.SequenceNode):
        for child in node.value:
            _reject_duplicate_explicit_keys(loader, child, deep, checked, active)
    active.remove(marker)
    checked.add(marker)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    checked = getattr(loader, "_duplicate_key_checked_nodes", None)
    if checked is None:
        checked = set()
        loader._duplicate_key_checked_nodes = checked
    _reject_duplicate_explicit_keys(loader, node, deep, checked, set())

    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _validate_structured_graph(document: Any) -> None:
    """Bound depth and logical expansion for every parsed structured input."""
    cache: dict[int, tuple[int, int]] = {}
    active: set[int] = set()

    def metrics(value: Any, depth: int) -> tuple[int, int]:
        if depth > MAX_STRUCTURED_NESTING:
            raise ValueError(f"structured input nesting exceeds limit {MAX_STRUCTURED_NESTING}")
        if type(value) is float and not math.isfinite(value):
            raise ValueError("structured input contains a non-finite number")
        if not isinstance(value, (Mapping, list)):
            return 1, 0
        marker = id(value)
        if marker in active:
            raise ValueError("structured input contains a cyclic YAML alias")
        if marker in cache:
            expanded_nodes, height = cache[marker]
            if depth + height > MAX_STRUCTURED_NESTING:
                raise ValueError(f"structured input nesting exceeds limit {MAX_STRUCTURED_NESTING}")
            return expanded_nodes, height

        active.add(marker)
        expanded_nodes = 1
        height = 0
        children = value.values() if isinstance(value, Mapping) else value
        for child in children:
            child_nodes, child_height = metrics(child, depth + 1)
            expanded_nodes += child_nodes
            if expanded_nodes > MAX_STRUCTURED_EXPANDED_NODES:
                raise ValueError(
                    "structured input expanded node count exceeds limit "
                    f"{MAX_STRUCTURED_EXPANDED_NODES}"
                )
            height = max(height, child_height + 1)
        active.remove(marker)
        cache[marker] = (expanded_nodes, height)
        return expanded_nodes, height

    metrics(document, 0)


def _load_yaml_text(text: str) -> Any:
    try:
        document = yaml.load(text, Loader=_UniqueKeyLoader)
    except RecursionError as exc:
        raise ValueError("YAML nesting exceeds parser limit") from exc
    _validate_structured_graph(document)
    return document


def _read_structured_text(path: Path) -> str:
    with path.open("rb") as stream:
        content = stream.read(MAX_STRUCTURED_INPUT_BYTES + 1)
    if len(content) > MAX_STRUCTURED_INPUT_BYTES:
        raise ValueError(f"structured input exceeds {MAX_STRUCTURED_INPUT_BYTES} bytes")
    return content.decode("utf-8")


def _load_yaml(path: Path) -> Any:
    return _load_yaml_text(_read_structured_text(path))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant {value!r}")


def _parse_json_integer(value: str) -> int:
    if len(value) > MAX_JSON_NUMBER_CHARACTERS:
        raise ValueError(f"JSON integer exceeds {MAX_JSON_NUMBER_CHARACTERS} characters")
    return int(value)


def _parse_json_float(value: str) -> float:
    if len(value) > MAX_JSON_NUMBER_CHARACTERS:
        raise ValueError(f"JSON number exceeds {MAX_JSON_NUMBER_CHARACTERS} characters")
    try:
        exact = Decimal(value)
        parsed = float(value)
    except (InvalidOperation, OverflowError, ValueError) as exc:
        raise ValueError(f"invalid JSON number {value!r}") from exc
    if not exact.is_finite() or not math.isfinite(parsed):
        raise ValueError(f"JSON number is outside the finite float range: {value!r}")
    if exact != 0 and parsed == 0:
        raise ValueError(f"JSON number underflows the finite float range: {value!r}")
    return parsed


def _check_json_nesting(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_STRUCTURED_NESTING:
                raise ValueError(f"JSON nesting exceeds limit {MAX_STRUCTURED_NESTING}")
        elif character in "]}":
            depth -= 1


def _load_json_bytes(content: bytes) -> Any:
    if len(content) > MAX_STRUCTURED_INPUT_BYTES:
        raise ValueError(f"structured input exceeds {MAX_STRUCTURED_INPUT_BYTES} bytes")
    return _load_json_text(content.decode("utf-8"))


def _load_json_text(text: str) -> Any:
    _check_json_nesting(text)
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
            parse_int=_parse_json_integer,
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds parser limit") from exc
    _validate_structured_graph(document)
    return document


def _load_json(path: Path) -> Any:
    return _load_json_text(_read_structured_text(path))


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
    _validate_structured_graph(packet)
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
    try:
        candidate = (base / relative).resolve()
        candidate.relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _canonical_claim_ids(root: Path) -> set[str]:
    text = (root / "docs/scientific_hardening/CLAIMS_MATRIX.md").read_text(encoding="utf-8")
    return set(re.findall(r"^\| (C[0-9]{2}) \|", text, flags=re.MULTILINE))


def _canonical_gate_ids(root: Path) -> set[str]:
    text = (root / "docs/scientific_hardening/GATE_TEST_REGISTRY.md").read_text(encoding="utf-8")
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
    try:
        _validate_structured_graph(requirements)
    except ValueError as exc:
        return [f"requirements: invalid structured graph: {exc}"]
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
            continue
        if receipt.get("media_type") in {
            "application/json",
            "application/schema+json",
            "application/yaml",
            "text/yaml",
            "text/markdown",
            "text/x-markdown",
        }:
            try:
                _structured_receipt_document(receipt)
            except (
                OSError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
                yaml.YAMLError,
            ) as exc:
                errors.append(
                    f"packet {packet.get('packet_id')}: receipt {receipt_id!r} "
                    f"cannot be parsed: {exc}"
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
            errors.append(f"packet {packet_id}: binding {name!r} must use a raw-results receipt")
            continue
        if receipt.get("media_type") != "application/json":
            errors.append(f"packet {packet_id}: binding {name!r} must use application/json")
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
                raise TypeError(f"expected {expected_type}, observed {observed_type}")
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
    text = _read_structured_text(path)
    if media_type in {"application/json", "application/schema+json"}:
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
        errors.append(f"packet {packet_id}: review builder model differs from packet builder")
    model_differs = document["reviewer_model_identity"] != builder["model_identity"]
    if declaration["reviewer_model_differs_from_builder"] != model_differs:
        errors.append(f"packet {packet_id}: reviewer model-separation declaration is contradictory")
    shared_operator = document["reviewer_operator"] == builder["operator"]
    if declaration["shared_operator"] != shared_operator:
        errors.append(f"packet {packet_id}: reviewer shared-operator declaration is contradictory")
    if declaration["builder_session_id"] != builder["session_id"]:
        errors.append(f"packet {packet_id}: review builder session differs from packet builder")
    shared_session = document["reviewer_session_id"] == builder["session_id"]
    if declaration["shared_session"] != shared_session:
        errors.append(f"packet {packet_id}: reviewer shared-session declaration is contradictory")
    if declaration["builder_orchestrator_id"] != builder["orchestrator_id"]:
        errors.append(
            f"packet {packet_id}: review builder orchestrator differs from packet builder"
        )
    shared_orchestrator = document["reviewer_orchestrator_id"] == builder["orchestrator_id"]
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
    if rereview_declaration["builder_model_identity"] != response.get("builder_model_identity"):
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
        artifact for artifact in artifacts if artifact.get("content_role") == "primary-protocol"
    ]
    if len(primary_artifacts) != 1:
        errors.append(f"packet {packet_id}: preregistration requires one primary protocol")
    registered = set(receipt_ids)
    declared = {
        receipt_id for receipt_id, receipt in receipts.items() if receipt.get("kind") == "protocol"
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
            errors.append(f"packet {packet_id}: frozen protocol receipt {receipt_id!r} is missing")
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
            frozen_bytes = _git_blob(root, packet["protocol_commit"], artifact["protocol_path"])
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
                errors.append(f"packet {packet_id}: primary protocol must use application/json")
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
                errors.append(f"packet {packet_id}: primary protocol cannot be parsed: {exc}")
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
            if not _strict_json_equal(document, expected_document):
                errors.append(
                    f"packet {packet_id}: primary protocol differs from exact "
                    "frozen preregistration envelope"
                )
    return errors


def _raw_command_matches_contract(
    contract_id: str,
    record: Mapping[str, Any],
    candidate_commit: str,
) -> bool:
    """Bind a retained command record to the frozen executable/argument contract."""

    file_name = Path(str(record.get("file", ""))).name.lower()
    if file_name.endswith(".exe"):
        file_name = file_name[:-4]
    arguments = record.get("arguments")
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        return False
    if contract_id == "git-clone":
        return (
            file_name == "git"
            and len(arguments) == 4
            and arguments[:3] == ["clone", "--no-checkout", "https://github.com/whact2025/POPGP"]
        )
    if contract_id == "git-checkout":
        return file_name == "git" and arguments == ["checkout", "--detach", candidate_commit]
    if contract_id == "uv-sync-frozen-no-editable":
        return file_name == "uv" and arguments == ["sync", "--frozen", "--no-editable"]
    if contract_id in {"pdflatex-pass-1", "pdflatex-pass-2"}:
        return (
            file_name == "pdflatex"
            and len(arguments) == 4
            and arguments[:2] == ["-interaction=nonstopmode", "-halt-on-error"]
            and arguments[2].startswith("-output-directory=")
            and arguments[3] == "docs/framework.tex"
        )
    expected_modules = {
        "trusted-python-ruff": ("ruff", ["check", "."]),
        "trusted-python-check-tex": ("scripts.check_tex", []),
        "trusted-python-pytest": ("pytest", ["-q", "-p", "no:cacheprovider"]),
        "trusted-python-chain-generator": ("examples.physics_qg.chain_1d", []),
        "trusted-python-grid-generator": ("examples.physics_qg.grid_2d", []),
        "trusted-python-gravity-generator": ("examples.physics_qg.gravity_well", []),
        "trusted-python-source-law-generator": ("examples.physics_qg.source_law", []),
        "trusted-python-many-body-generator": (
            "examples.physics_qg.source_law_many_body",
            [],
        ),
        "trusted-python-ca-generator": ("examples.physics_qg.ca_model", []),
        "trusted-python-artifact-boundary": (
            "scripts.check_validation_artifacts",
            ["--enforce-change-boundary"],
        ),
    }
    if contract_id == "trusted-python-environment-verify":
        return (
            file_name == "python"
            and arguments[:2] == ["-I", "-S"]
            and "check_reproduction_boundary.py" in arguments[2].replace("\\", "/")
            and "verify" in arguments
            and "--manifest" in arguments
            and "--expected-sha256" in arguments
        )
    expected = expected_modules.get(contract_id)
    if expected is None or file_name != "python" or arguments[:2] != ["-I", "-S"]:
        return False
    module, module_arguments = expected
    try:
        module_index = arguments.index("--module")
        separator_index = arguments.index("--", module_index + 2)
    except ValueError:
        return False
    return (
        "check_reproduction_boundary.py" in arguments[2].replace("\\", "/")
        and "run" in arguments
        and arguments[module_index + 1] == module
        and arguments[separator_index + 1 :] == module_arguments
        and "run_without_startup_hooks.py" in " ".join(arguments).replace("\\", "/")
    )


def _git_tree_entries(root: Path, commit: str) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for raw_entry in _git_output(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, _kind, object_id = metadata.decode("ascii").split()
        entries[raw_path.decode("utf-8")] = (mode, object_id)
    return entries


def _source_manifest_errors(
    document: Any,
    root: Path,
    commit: str,
    tree: str,
    label: str,
) -> list[str]:
    if not isinstance(document, Mapping) or set(document) != {
        "manifest_version",
        "base_ref",
        "base_commit",
        "base_tree",
        "entries",
    }:
        return [f"{label} has malformed source-manifest envelope"]
    if (
        document["manifest_version"] != 2
        or document["base_ref"] != commit
        or document["base_commit"] != commit
        or document["base_tree"] != tree
        or not isinstance(document["entries"], list)
    ):
        return [f"{label} source-manifest identity differs from frozen candidate"]
    expected = _git_tree_entries(root, commit)
    observed: dict[str, tuple[str, str, str]] = {}
    errors: list[str] = []
    for entry in document["entries"]:
        if not isinstance(entry, Mapping):
            return [f"{label} source-manifest entry is not an object"]
        try:
            path = entry["path"]
            value = (entry["mode"], entry["git_object_id"], entry["worktree_sha256"])
        except KeyError as exc:
            return [f"{label} source-manifest entry is missing {exc}"]
        if path in observed:
            errors.append(f"{label} source-manifest duplicates {path!r}")
        observed[path] = value
    if set(observed) != set(expected):
        errors.append(f"{label} source-manifest path set differs from candidate tree")
    for path in set(observed) & set(expected):
        mode, object_id = expected[path]
        observed_mode, observed_id, observed_sha = observed[path]
        if (observed_mode, observed_id) != (mode, object_id):
            errors.append(f"{label} source-manifest Git binding differs for {path!r}")
        if observed_sha != _sha256_bytes(_git_blob(root, commit, path)):
            errors.append(f"{label} source-manifest byte hash differs for {path!r}")
    return errors


def _environment_manifest_errors(document: Any, label: str) -> list[str]:
    if not isinstance(document, Mapping) or set(document) != {"manifest_version", "entries"}:
        return [f"{label} has malformed environment-manifest envelope"]
    entries = document["entries"]
    if document["manifest_version"] != 2 or not isinstance(entries, list) or not entries:
        return [f"{label} environment-manifest is empty or has the wrong version"]
    paths: set[str] = set()
    errors: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "kind",
            "mode",
            "size_bytes",
            "sha256",
        }:
            return [f"{label} environment-manifest entry is malformed"]
        path = entry["path"]
        if path in paths:
            errors.append(f"{label} environment-manifest duplicates {path!r}")
        paths.add(path)
        if (
            not isinstance(path, str)
            or not path
            or entry["kind"] not in {"file", "symlink"}
            or type(entry["mode"]) is not int
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
            or not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            errors.append(f"{label} environment-manifest has invalid typed data")
    return errors


def _validate_raw_evidence_contract(
    packet: Mapping[str, Any],
    receipts: Mapping[str, Any],
    packet_id: str,
    campaign_base: Path,
    root: Path,
) -> list[str]:
    """Validate typed, executable, hash-retained VIA raw evidence fail closed."""

    from scripts.check_validation_artifacts import (
        VISUAL_MAXIMUM_CHANNEL_ERROR_LIMIT,
        check_validation_semantics,
        compare_validation_documents,
        compare_visual_artifact,
    )

    parameters = packet["preregistration"]["parameters"]
    contract = parameters.get("raw_results_contract")
    if contract is None:
        return []
    required_fields = {
        "schema_receipt_id",
        "raw_results_receipt_id",
        "evidence_manifest_pointer",
        "required_platforms",
        "required_command_contracts",
        "required_artifact_paths",
        "required_mutation_ids",
        "required_mutation_oracles",
        "required_pdf_page_count",
    }
    if not isinstance(contract, Mapping) or set(contract) != required_fields:
        return [
            f"packet {packet_id}: raw_results_contract must contain exactly "
            f"{sorted(required_fields)}"
        ]
    required_platforms = contract["required_platforms"]
    command_contracts = contract["required_command_contracts"]
    artifact_paths = contract["required_artifact_paths"]
    mutation_ids = contract["required_mutation_ids"]
    mutation_oracles = contract["required_mutation_oracles"]
    if (
        not isinstance(required_platforms, list)
        or not required_platforms
        or len(required_platforms) != len(set(required_platforms))
        or required_platforms != parameters.get("platform_families")
        or not isinstance(command_contracts, Mapping)
        or not command_contracts
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in command_contracts.items()
        )
        or parameters.get("required_command_count") != len(command_contracts)
        or not isinstance(artifact_paths, list)
        or len(artifact_paths) != len(set(artifact_paths))
        or len(artifact_paths) != 18
        or not isinstance(mutation_ids, list)
        or len(mutation_ids) != len(set(mutation_ids))
        or not isinstance(mutation_oracles, Mapping)
        or set(mutation_oracles) != set(mutation_ids)
        or any(
            not isinstance(value, str) or not value for value in mutation_oracles.values()
        )
        or parameters.get("required_mutation_count") != len(mutation_ids)
        or type(contract["required_pdf_page_count"]) is not int
        or contract["required_pdf_page_count"] < 1
    ):
        return [f"packet {packet_id}: raw_results_contract has malformed or contradictory values"]

    schema_receipt_id = contract["schema_receipt_id"]
    raw_receipt_id = contract["raw_results_receipt_id"]
    schema_receipt = receipts.get(schema_receipt_id)
    if (
        schema_receipt is None
        or schema_receipt.get("kind") != "protocol"
        or schema_receipt.get("media_type") != "application/schema+json"
    ):
        return [f"packet {packet_id}: raw-results schema receipt is missing or mistyped"]
    try:
        raw_schema = _structured_receipt_document(schema_receipt)
        Draft202012Validator.check_schema(raw_schema)
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as exc:
        return [f"packet {packet_id}: raw-results schema is invalid: {exc}"]
    schema_platforms = raw_schema.get("properties", {}).get("platforms", {})
    if (
        schema_platforms.get("required") != required_platforms
        or set(schema_platforms.get("properties", {})) != set(required_platforms)
        or schema_platforms.get("additionalProperties") is not False
    ):
        return [f"packet {packet_id}: raw-results schema platform set differs from frozen contract"]

    raw_receipt = receipts.get(raw_receipt_id)
    raw_required = LIFECYCLE_ORDER[packet["lifecycle_phase"]] >= LIFECYCLE_ORDER["reproduced"]
    if raw_receipt is None:
        return (
            [
                f"packet {packet_id}: reproduced lifecycle requires raw-results "
                f"receipt {raw_receipt_id!r}"
            ]
            if raw_required
            else []
        )
    if (
        raw_receipt.get("kind") != "raw-results"
        or raw_receipt.get("media_type") != "application/json"
    ):
        return [f"packet {packet_id}: raw-results receipt {raw_receipt_id!r} is mistyped"]
    try:
        raw_document = _structured_receipt_document(raw_receipt)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return [f"packet {packet_id}: raw-results document cannot be parsed: {exc}"]
    errors = _schema_errors(raw_document, raw_schema, f"packet {packet_id} raw-results")
    if errors:
        return errors
    if raw_document["candidate_commit"] != packet["candidate_commit"]:
        errors.append(f"packet {packet_id}: raw-results candidate commit differs from packet")
    if raw_document["candidate_tree"] != packet["tree_hash"]:
        errors.append(f"packet {packet_id}: raw-results candidate tree differs from packet")
    if raw_document["blocked"] is not False:
        errors.append(f"packet {packet_id}: R2 raw results cannot self-declare blockage")

    try:
        evidence_manifest = _json_pointer(raw_document, contract["evidence_manifest_pointer"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return [f"packet {packet_id}: evidence manifest cannot resolve: {exc}"]
    raw_path = raw_receipt.get("_resolved_path")
    if not isinstance(raw_path, Path) or not isinstance(evidence_manifest, list):
        return [f"packet {packet_id}: raw-results path or evidence manifest is unavailable"]
    evidence_by_path: dict[str, Mapping[str, Any]] = {}
    evidence_files: dict[str, Path] = {}
    for entry in evidence_manifest:
        relative = entry["path"]
        if relative in evidence_by_path:
            errors.append(f"packet {packet_id}: duplicate evidence-manifest path {relative!r}")
            continue
        evidence_by_path[relative] = entry
        resolved = _resolve_inside(raw_path.parent, relative, campaign_base)
        if resolved is None or resolved == raw_path.resolve() or not resolved.is_file():
            errors.append(f"packet {packet_id}: evidence path is missing or unsafe: {relative!r}")
            continue
        evidence_files[relative] = resolved
        if resolved.stat().st_size != entry["byte_count"] or _sha256(resolved) != entry["sha256"]:
            errors.append(f"packet {packet_id}: evidence bytes differ from manifest: {relative!r}")

    expected_test_count = parameters["required_test_count"]
    expected_example_count = parameters["required_example_count"]
    expected_visual_count = parameters["required_visual_count"]
    expected_uv = parameters["uv_version"]
    expected_pages = contract["required_pdf_page_count"]
    expected_pdf_banner = parameters["pdf_engine_banner"]
    if (
        parameters.get("maximum_cross_platform_channel_delta")
        != VISUAL_MAXIMUM_CHANNEL_ERROR_LIMIT
    ):
        return [
            f"packet {packet_id}: cross-platform visual limit differs from the "
            "authoritative checker"
        ]
    platform_evidence: dict[str, bool] = {}
    platform_clean: dict[str, bool] = {}
    platform_mutations: dict[str, bool] = {}
    platform_visuals: dict[str, dict[str, Path]] = {}
    for platform_name in required_platforms:
        platform_errors_before = len(errors)
        platform = raw_document["platforms"].get(platform_name)
        if not isinstance(platform, Mapping):
            errors.append(f"packet {packet_id}: missing platform record {platform_name!r}")
            platform_evidence[platform_name] = False
            platform_clean[platform_name] = False
            platform_mutations[platform_name] = False
            continue
        label = f"packet {packet_id}: {platform_name}"
        if platform["platform_family"] != platform_name:
            errors.append(f"{label} platform record is under the wrong key")
        if platform["candidate_commit"] != packet["candidate_commit"]:
            errors.append(f"{label} candidate commit differs from packet")
        if platform["candidate_tree"] != packet["tree_hash"]:
            errors.append(f"{label} candidate tree differs from packet")

        command_results = platform["command_results"]
        if set(command_results) != set(command_contracts):
            errors.append(f"{label} command set differs from frozen contract")
        command_ok = True
        for command_id, expected_contract_id in command_contracts.items():
            command = command_results.get(command_id)
            if not isinstance(command, Mapping):
                command_ok = False
                continue
            result_path = command["result_path"]
            result_entry = evidence_by_path.get(result_path)
            result_file = evidence_files.get(result_path)
            stdout_entry = evidence_by_path.get(command["stdout_path"])
            stderr_entry = evidence_by_path.get(command["stderr_path"])
            if (
                result_entry is None
                or result_entry.get("role") != "command-result"
                or result_entry.get("platform_family") != platform_name
                or result_entry.get("sha256") != command["result_sha256"]
                or result_file is None
                or stdout_entry is None
                or stdout_entry.get("role") != "command-stdout"
                or stdout_entry.get("sha256") != command["stdout_sha256"]
                or stderr_entry is None
                or stderr_entry.get("role") != "command-stderr"
                or stderr_entry.get("sha256") != command["stderr_sha256"]
            ):
                command_ok = False
                errors.append(f"{label} command {command_id!r} is not typed and hash-bound")
                continue
            try:
                record = _load_json(result_file)
                expected_command = f"{record['file']} {' '.join(record['arguments'])}"
                matches = (
                    record["label"] == command_id
                    and record["contract_id"] == expected_contract_id
                    and command["contract_id"] == expected_contract_id
                    and record["exit_code"] == command["exit_code"]
                    and record["duration_seconds"] == command["duration_seconds"]
                    and record["stdout_sha256"] == command["stdout_sha256"]
                    and record["stderr_sha256"] == command["stderr_sha256"]
                    and command["command"] == expected_command
                    and _raw_command_matches_contract(
                        expected_contract_id, record, packet["candidate_commit"]
                    )
                )
            except (
                OSError,
                UnicodeDecodeError,
                ValueError,
                TypeError,
                KeyError,
                json.JSONDecodeError,
            ):
                matches = False
            if not matches or command["exit_code"] != 0:
                command_ok = False
                errors.append(f"{label} command {command_id!r} violates its executable contract")

        pytest_stdout = evidence_files.get(
            command_results.get("006-pytest", {}).get("stdout_path", "")
        )
        test_count_ok = False
        if pytest_stdout is not None:
            text = pytest_stdout.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"(?m)(\d+) passed in ", text)
            test_count_ok = match is not None and int(match.group(1)) == expected_test_count
        if platform["test_count"] != expected_test_count or not test_count_ok:
            errors.append(f"{label} retained pytest output does not prove the frozen test count")

        artifact_results = platform["artifact_results"]
        artifact_ok = set(artifact_results) == set(artifact_paths)
        if not artifact_ok:
            errors.append(f"{label} artifact set differs from frozen contract")
        semantic_ok = True
        visual_ok = True
        actual_visual_count = 0
        for source_path in artifact_paths:
            result = artifact_results.get(source_path)
            if not isinstance(result, Mapping):
                semantic_ok = False
                visual_ok = False
                continue
            entry = evidence_by_path.get(result["evidence_path"])
            evidence_file = evidence_files.get(result["evidence_path"])
            expected_media = (
                "application/json"
                if source_path.endswith(".json")
                else "image/gif"
                if source_path.endswith(".gif")
                else "image/png"
            )
            expected_role = "validation-json" if expected_media == "application/json" else "visual"
            bound = (
                result["source_path"] == source_path
                and result["media_type"] == expected_media
                and entry is not None
                and entry.get("platform_family") == platform_name
                and entry.get("role") == expected_role
                and entry.get("source_path") == source_path
                and entry.get("sha256") == result["sha256"]
                and evidence_file is not None
            )
            if not bound:
                errors.append(f"{label} artifact {source_path!r} is not typed and hash-bound")
                semantic_ok = False
                visual_ok = False
                continue
            try:
                reference = _git_blob(root, packet["candidate_commit"], source_path)
                if expected_media == "application/json":
                    reference_document = _load_json_bytes(reference)
                    candidate_document = _load_json(evidence_file)
                    comparison = compare_validation_documents(
                        reference_document, candidate_document
                    )
                    artifact_errors = [
                        *comparison.errors,
                        *check_validation_semantics(candidate_document, source_path),
                    ]
                    if artifact_errors:
                        semantic_ok = False
                        errors.append(
                            f"{label} validation artifact {source_path!r} is invalid: "
                            f"{artifact_errors[0]}"
                        )
                else:
                    actual_visual_count += 1
                    artifact_errors = compare_visual_artifact(reference, evidence_file)
                    if artifact_errors:
                        visual_ok = False
                        errors.append(
                            f"{label} visual artifact {source_path!r} is invalid: "
                            f"{artifact_errors[0]}"
                        )
                    else:
                        platform_visuals.setdefault(platform_name, {})[source_path] = evidence_file
            except (
                OSError,
                UnicodeDecodeError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(f"{label} artifact {source_path!r} cannot be validated: {exc}")
                semantic_ok = False
                visual_ok = False
        if (
            platform["visual_count"] != actual_visual_count
            or actual_visual_count != expected_visual_count
        ):
            visual_ok = False
            errors.append(f"{label} visual count is not derived from retained artifacts")

        role_entries = [
            (path, entry)
            for path, entry in evidence_by_path.items()
            if entry.get("platform_family") == platform_name
        ]
        environment_matches = [
            (path, entry)
            for path, entry in role_entries
            if entry.get("role") == "environment-manifest"
        ]
        source_matches = [
            (path, entry) for path, entry in role_entries if entry.get("role") == "source-manifest"
        ]
        pdf_matches = [(path, entry) for path, entry in role_entries if entry.get("role") == "pdf"]
        pdf_engine_matches = [
            (path, entry)
            for path, entry in role_entries
            if entry.get("role") == "pdf-engine"
        ]
        status_matches = [
            (path, entry)
            for path, entry in role_entries
            if entry.get("role") == "repository-status"
        ]
        environment_ok = len(environment_matches) == 1
        source_ok = len(source_matches) == 1
        pdf_ok = len(pdf_matches) == 1
        pdf_engine_ok = len(pdf_engine_matches) == 1
        status_ok = len(status_matches) == 2
        if environment_ok:
            path, entry = environment_matches[0]
            environment_ok = entry["sha256"] == platform["environment_manifest_sha256"]
            try:
                environment_document = _load_json(evidence_files[path])
                environment_errors = _environment_manifest_errors(environment_document, label)
                environment_ok = environment_ok and not environment_errors
                errors.extend(environment_errors)
            except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                environment_ok = False
                errors.append(f"{label} environment manifest cannot be validated: {exc}")
        if source_ok:
            path, entry = source_matches[0]
            source_ok = entry["sha256"] == platform["source_manifest_sha256"]
            try:
                source_document = _load_json(evidence_files[path])
                source_errors = _source_manifest_errors(
                    source_document,
                    root,
                    packet["candidate_commit"],
                    packet["tree_hash"],
                    label,
                )
                source_ok = source_ok and not source_errors
                errors.extend(source_errors)
            except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                source_ok = False
                errors.append(f"{label} source manifest cannot be validated: {exc}")
        if pdf_ok:
            path, entry = pdf_matches[0]
            pdf_file = evidence_files[path]
            pdf_bytes = pdf_file.read_bytes()
            try:
                pdf_reader = PdfReader(str(pdf_file), strict=True)
                parsed_pages = len(pdf_reader.pages)
            except (OSError, PdfReadError, TypeError, ValueError):
                parsed_pages = -1
            pdf_ok = (
                entry["sha256"] == platform["pdf_sha256"]
                and pdf_bytes.startswith(b"%PDF-")
                and b"%%EOF" in pdf_bytes[-1024:]
                and len(pdf_bytes) >= 100_000
                and parsed_pages == expected_pages
                and platform["pdf_page_count"] == parsed_pages
            )
            for command_id in ("014-pdflatex-1", "015-pdflatex-2"):
                stdout_path = command_results.get(command_id, {}).get("stdout_path", "")
                stdout_file = evidence_files.get(stdout_path)
                output = (
                    stdout_file.read_text(encoding="utf-8", errors="replace") if stdout_file else ""
                )
                pdf_ok = (
                    pdf_ok
                    and "Output written on" in output
                    and f"({expected_pages} pages" in output
                )
        if pdf_engine_ok:
            path, _entry = pdf_engine_matches[0]
            engine_file = evidence_files.get(path)
            pdf_engine_ok = (
                engine_file is not None
                and platform["pdf_engine"] == expected_pdf_banner
                and engine_file.read_text(encoding="utf-8", errors="strict").strip()
                == expected_pdf_banner
            )
        pdf_ok = pdf_ok and pdf_engine_ok
        if not environment_ok:
            errors.append(f"{label} environment evidence is incomplete or invalid")
        if not source_ok:
            errors.append(f"{label} source evidence is incomplete or invalid")
        if status_ok:
            status_files = {
                Path(path).name: evidence_files[path] for path, _entry in status_matches
            }
            generated_status = status_files.get("generated-status-with-ignored.txt")
            final_status = status_files.get("final-status-with-ignored.txt")
            if generated_status is None or final_status is None:
                status_ok = False
            else:
                generated_paths: set[str] = set()
                for line in generated_status.read_text(
                    encoding="utf-8", errors="strict"
                ).splitlines():
                    if len(line) < 4:
                        status_ok = False
                        break
                    generated_paths.add(line[3:].replace("\\", "/"))
                status_ok = (
                    status_ok
                    and generated_paths <= set(artifact_paths)
                    and final_status.read_text(encoding="utf-8", errors="strict").strip()
                    == ""
                )
        if not status_ok:
            errors.append(f"{label} retained repository-status evidence is incomplete or dirty")
        if not pdf_ok:
            errors.append(f"{label} PDF evidence is incomplete or invalid")

        mutation_results = platform["mutation_results"]
        mutation_ok = (
            len(mutation_results) == len(mutation_ids)
            and {item["mutation_id"] for item in mutation_results} == set(mutation_ids)
            and all(item["rejected"] is True for item in mutation_results)
        )
        for mutation in mutation_results:
            for evidence_path in mutation["evidence_paths"]:
                entry = evidence_by_path.get(evidence_path)
                evidence_file = evidence_files.get(evidence_path)
                if (
                    entry is None
                    or entry.get("platform_family") != platform_name
                    or entry.get("role") != "mutation-result"
                    or entry.get("media_type") != "application/json"
                    or evidence_file is None
                ):
                    mutation_ok = False
                    continue
                try:
                    mutation_document = _load_json(evidence_file)
                    execution = mutation_document.get("execution", {})
                    oracle_errors = mutation_document.get("oracle_errors", [])
                    mutation_ok = mutation_ok and (
                        isinstance(mutation_document, Mapping)
                        and set(mutation_document)
                        == {
                            "schema_version",
                            "mutation_id",
                            "platform_family",
                            "candidate_commit",
                            "candidate_tree",
                            "rejected",
                            "attack",
                            "oracle_id",
                            "oracle_errors",
                            "execution",
                        }
                        and mutation_document.get("schema_version") == 1
                        and mutation_document.get("mutation_id") == mutation["mutation_id"]
                        and mutation_document.get("platform_family") == platform_name
                        and mutation_document.get("candidate_commit") == packet["candidate_commit"]
                        and mutation_document.get("candidate_tree") == packet["tree_hash"]
                        and mutation_document.get("rejected") is True
                        and isinstance(mutation_document.get("attack"), str)
                        and bool(mutation_document.get("attack"))
                        and mutation_document.get("oracle_id")
                        == mutation_oracles[mutation["mutation_id"]]
                        and isinstance(oracle_errors, list)
                        and bool(oracle_errors)
                        and all(
                            isinstance(item, Mapping)
                            and set(item) == {"error_id", "message"}
                            and isinstance(item["error_id"], str)
                            and isinstance(item["message"], str)
                            and bool(item["message"])
                            for item in oracle_errors
                        )
                        and any(
                            item["error_id"] == mutation_oracles[mutation["mutation_id"]]
                            for item in oracle_errors
                        )
                        and isinstance(execution, Mapping)
                        and set(execution)
                        == {
                            "command",
                            "exit_code",
                            "started_at",
                            "finished_at",
                            "test_ids",
                            "passed_test_count",
                            "stdout_sha256",
                            "stderr_sha256",
                        }
                        and isinstance(execution.get("command"), str)
                        and bool(execution.get("command"))
                        and execution.get("exit_code") == 0
                        and isinstance(execution.get("started_at"), str)
                        and isinstance(execution.get("finished_at"), str)
                        and isinstance(execution.get("test_ids"), list)
                        and bool(execution.get("test_ids"))
                        and execution.get("passed_test_count") == len(execution["test_ids"])
                        and all(
                            isinstance(test_id, str) and bool(test_id)
                            for test_id in execution["test_ids"]
                        )
                        and re.fullmatch(r"[0-9a-f]{64}", execution.get("stdout_sha256", ""))
                        is not None
                        and re.fullmatch(r"[0-9a-f]{64}", execution.get("stderr_sha256", ""))
                        is not None
                    )
                except (
                    OSError,
                    AttributeError,
                    UnicodeDecodeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    mutation_ok = False
        if not mutation_ok:
            errors.append(f"{label} mutation evidence is incomplete, untyped, or accepted")

        declared_paths = set(platform["evidence_paths"])
        manifest_paths = {path for path, entry in role_entries}
        evidence_paths_ok = declared_paths == manifest_paths and bool(declared_paths)
        if not evidence_paths_ok:
            errors.append(f"{label} evidence_paths differ from manifest")
        uv_parts = platform["uv_version"].split()
        uv_ok = len(uv_parts) >= 2 and uv_parts[:2] == ["uv", expected_uv]
        example_count_ok = platform["example_count"] == expected_example_count
        fields = {
            "commands_passed": command_ok,
            "semantic_contract_passed": semantic_ok,
            "visual_contract_passed": visual_ok,
            "source_boundary_passed": source_ok and status_ok,
            "environment_boundary_passed": environment_ok,
            "pdf_passed": pdf_ok,
            "mutations_rejected": mutation_ok,
        }
        for field, expected in fields.items():
            if platform[field] is not expected:
                errors.append(f"{label} {field} differs from typed retained evidence")
        clean = all(
            (
                command_ok,
                test_count_ok,
                example_count_ok,
                semantic_ok,
                visual_ok,
                source_ok,
                status_ok,
                environment_ok,
                pdf_ok,
                uv_ok,
                evidence_paths_ok,
                platform["candidate_commit"] == packet["candidate_commit"],
                platform["candidate_tree"] == packet["tree_hash"],
            )
        )
        if platform["mutation_count"] != len(mutation_results):
            errors.append(f"{label} mutation_count differs from retained records")
        if platform["overall_passed"] is not (clean and mutation_ok):
            errors.append(f"{label} overall_passed differs from retained evidence")
        platform_evidence[platform_name] = (
            len(errors) == platform_errors_before
            and command_ok
            and artifact_ok
            and evidence_paths_ok
        )
        platform_clean[platform_name] = clean
        platform_mutations[platform_name] = mutation_ok

    if len(required_platforms) >= 2:
        reference_platform = required_platforms[0]
        reference_visuals = platform_visuals.get(reference_platform, {})
        for compared_platform in required_platforms[1:]:
            compared_visuals = platform_visuals.get(compared_platform, {})
            for source_path in artifact_paths:
                if not source_path.lower().endswith((".gif", ".jpeg", ".jpg", ".png", ".webp")):
                    continue
                reference_file = reference_visuals.get(source_path)
                compared_file = compared_visuals.get(source_path)
                if reference_file is None or compared_file is None:
                    continue
                cross_platform_errors = compare_visual_artifact(
                    reference_file.read_bytes(), compared_file
                )
                if cross_platform_errors:
                    errors.append(
                        f"packet {packet_id}: cross-platform visual {source_path!r} differs "
                        f"between {reference_platform} and {compared_platform}: "
                        f"{cross_platform_errors[0]}"
                    )
                    platform_clean[reference_platform] = False
                    platform_clean[compared_platform] = False

    expected_capabilities = {
        "evidence-contract": all(platform_evidence.values()),
        "cross-platform-reproduction": all(platform_clean.values()),
        "mutation-rejection": all(platform_mutations.values()),
    }
    if raw_document["capabilities"] != expected_capabilities:
        errors.append(
            f"packet {packet_id}: raw capability Booleans differ from retained evidence "
            f"({raw_document['capabilities']} != {expected_capabilities})"
        )
    if raw_document["failed"] is not (not all(expected_capabilities.values())):
        errors.append(f"packet {packet_id}: raw failed Boolean differs from retained evidence")
    return errors


def validate_via000_raw_results(
    protocol_path: Path | str,
    schema_path: Path | str,
    raw_results_path: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> list[str]:
    """Validate an assembled VIA-000 package before creating a commitment."""

    try:
        protocol_path = Path(protocol_path).resolve()
        schema_path = Path(schema_path).resolve()
        raw_results_path = Path(raw_results_path).resolve()
        root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
        protocol = _load_json(protocol_path)
        parameters = protocol["parameters"]
        contract = parameters["raw_results_contract"]
        packet = {
            "preregistration": {"parameters": parameters},
            "candidate_commit": parameters["candidate_commit"],
            "tree_hash": parameters["candidate_tree"],
            "lifecycle_phase": "reproduced",
        }
        receipts = {
            contract["schema_receipt_id"]: {
                "kind": "protocol",
                "media_type": "application/schema+json",
                "_resolved_path": schema_path,
            },
            contract["raw_results_receipt_id"]: {
                "kind": "raw-results",
                "media_type": "application/json",
                "_resolved_path": raw_results_path,
            },
        }
        return _validate_raw_evidence_contract(
            packet,
            receipts,
            protocol["packet_id"],
            raw_results_path.parent,
            root,
        )
    except Exception as exc:
        return [f"VIA-000 raw-results validation failed closed: {type(exc).__name__}: {exc}"]


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
        errors.append(f"packet {packet_id}: external replication {label} must use application/json")
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


def _repository_text_is_safe(value: str) -> bool:
    """Reject decoded control characters before any URL or filesystem operation."""
    if re.search(r"%(?![0-9A-Fa-f]{2})", value):
        return False
    try:
        decoded = unquote(value, errors="strict")
    except UnicodeDecodeError:
        return False
    return not any(ord(character) < 32 or ord(character) == 127 for character in decoded)


def _canonical_git_path(value: str) -> str | None:
    try:
        path = unquote(value, errors="strict").replace("\\", "/")
    except UnicodeDecodeError:
        return None
    if "%" in path or not _repository_text_is_safe(path):
        return None
    path = re.sub(r"/{2,}", "/", path)
    path = posixpath.normpath(path) if path else ""
    if path == ".":
        path = ""
    path = path.rstrip("/")
    while path.lower().endswith("/.git"):
        path = path[:-5].rstrip("/")
    if path.lower().endswith(".git"):
        path = path[:-4]
    return path.lower()


def _canonical_repository_identity(value: str) -> str | None:
    """Normalize Git URL/path aliases, returning ``None`` for unsafe identities."""
    raw = value.strip().replace("\\", "/")
    if not raw or not _repository_text_is_safe(raw):
        return None
    is_windows_drive = re.match(r"^[A-Za-z]:/", raw) is not None
    is_file_uri = raw.lower().startswith("file:")
    scp_match = re.fullmatch(r"(?:[^@/]+@)?([^:/]+):(.+)", raw)
    if scp_match and "://" not in raw and not is_windows_drive and not is_file_uri:
        raw = f"ssh://{scp_match.group(1)}/{scp_match.group(2)}"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme and parsed.scheme.lower() != "file" and not is_windows_drive:
        try:
            host = parsed.hostname
            port = parsed.port
        except ValueError:
            return None
        if not host:
            return None
        try:
            host = unquote(host, errors="strict")
        except UnicodeDecodeError:
            return None
        if (
            "%" in host
            or not _repository_text_is_safe(host)
            or any(character in host for character in "/\\?#@[]")
        ):
            return None
        host = host.rstrip(".").lower()
        if not host or len(host) > 253:
            return None
        dotted_decimal = re.fullmatch(r"\d+(?:\.\d+){3}", host)
        if dotted_decimal:
            parts = host.split(".")
            if any(len(part) > 3 for part in parts):
                return None
            octets = [int(part, 10) for part in parts]
            if any(octet > 255 for octet in octets):
                return None
            host = ".".join(str(octet) for octet in octets)
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            try:
                host = host.encode("idna").decode("ascii")
            except UnicodeError:
                return None
        else:
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                address = address.ipv4_mapped
            host = address.compressed
        scheme = parsed.scheme.lower()
        default_ports = {
            "http": 80,
            "https": 443,
            "ssh": 22,
            "git+ssh": 22,
            "git": 9418,
        }
        authority = f"[{host}]" if ":" in host else host
        if port is not None and port != default_ports.get(scheme):
            authority = f"{authority}:{port}"
        path = _canonical_git_path(parsed.path)
        if path is None:
            return None
        return f"{authority}{path}"
    if parsed.scheme.lower() == "file" and not is_windows_drive:
        try:
            local_path = unquote(parsed.path, errors="strict")
        except UnicodeDecodeError:
            return None
        if re.match(r"^/[A-Za-z]:/", local_path):
            local_path = local_path[1:]
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            local_path = f"//{parsed.netloc}{local_path}"
    else:
        local_path = raw
    if not _repository_text_is_safe(local_path):
        return None
    try:
        normalized = str(Path(local_path).resolve()).replace("\\", "/")
    except (OSError, RuntimeError, ValueError):
        return None
    return _canonical_git_path(normalized)


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's Boolean/number equality coercion."""
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            return False
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) or isinstance(expected, list):
        if not isinstance(actual, list) or not isinstance(expected, list):
            return False
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return type(actual) is type(expected) and actual == expected


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    """Terminate a bounded subprocess and every descendant it created."""
    if process.poll() is not None:
        return
    cleanup_deadline = time.monotonic() + PROCESS_TREE_CLEANUP_SECONDS

    def remaining() -> float:
        return max(0.0, cleanup_deadline - time.monotonic())

    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(0.1, remaining()),
            )
            if result.returncode != 0:
                process.kill()
        except (OSError, subprocess.SubprocessError):
            try:
                process.kill()
            except OSError:
                pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=remaining())
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=remaining())
        except subprocess.TimeoutExpired:
            pass


def _run_bounded_process(command: list[str], *, timeout: int | float) -> int:
    """Run without captured pipes and enforce one deadline over the process tree."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt",
        creationflags=(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        ),
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        raise


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
        errors.append(f"packet {packet_id}: external repository bundle has wrong media type")
    if receipt.get("sha256") != implementation["repository_bundle_sha256"]:
        errors.append(f"packet {packet_id}: external repository bundle hash differs")
    bundle_path = receipt.get("_resolved_path")
    if not isinstance(bundle_path, Path) or not bundle_path.is_file():
        errors.append(f"packet {packet_id}: external repository bundle is unavailable")
        return errors
    try:
        with tempfile.TemporaryDirectory(prefix="popgp-external-bundle-") as temporary:
            checkout = Path(temporary) / "repository"
            returncode = _run_bounded_process(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-checkout",
                    str(bundle_path),
                    str(checkout),
                ],
                timeout=GIT_BUNDLE_TIMEOUT_SECONDS,
            )
            if returncode != 0:
                errors.append(
                    f"packet {packet_id}: external repository bundle cannot be cloned: "
                    f"git exited with status {returncode}"
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
    external_repository = _canonical_repository_identity(implementation["repository"])
    candidate_repository = _canonical_repository_identity(campaign["repository"])
    if external_repository is None:
        errors.append(f"packet {packet_id}: external implementation repository identity is invalid")
    if candidate_repository is None:
        errors.append(f"packet {packet_id}: candidate repository identity is invalid")
    if (
        external_repository is not None
        and candidate_repository is not None
        and external_repository == candidate_repository
    ):
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
    if contract_document is not None and not _strict_json_equal(
        contract_document,
        {
            "packet_id": packet_id,
            "contract": contract,
        },
    ):
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
    if provenance_document is not None and not _strict_json_equal(
        provenance_document,
        {
            "packet_id": packet_id,
            "organization": organization,
            "operator": operator,
            "implementation": implementation,
        },
    ):
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
    if prediction_receipt is not None and prediction_receipt.get("sha256") != prediction["sha256"]:
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
        if (
            any(
                prediction_document.get(field) != prediction[field]
                for field in ("committed_by", "committed_at")
            )
            or prediction_document.get("packet_id") != packet_id
        ):
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
    if reveal_document is not None and not _strict_json_equal(
        reveal_document,
        {
            "packet_id": packet_id,
            "authorized_by": evaluator,
            "revealed_at": packet["blind_custody"]["reveal"]["revealed_at"],
            "prediction_receipt_id": prediction["receipt_id"],
            "prediction_sha256": prediction["sha256"],
        },
    ):
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
    if commitment_document is not None and not _strict_json_equal(
        commitment_document,
        {
            "packet_id": packet_id,
            "committed_by": reproduction["committed_by"],
            "committed_at": reproduction["committed_at"],
            "output_receipt_id": reproduction["output_receipt_id"],
            "output_sha256": reproduction["output_sha256"],
        },
    ):
        errors.append(f"packet {packet_id}: external output commitment differs")

    comparison = contract["comparison"]
    candidate_receipt = receipts.get(comparison["candidate_output_receipt_id"])
    if candidate_receipt is None or candidate_receipt.get("kind") != "raw-results":
        errors.append(f"packet {packet_id}: comparison candidate output is missing")
    custody_commitment = packet["blind_custody"]["output_commitment"]
    if not isinstance(custody_commitment, Mapping):
        errors.append(f"packet {packet_id}: comparison requires candidate output commitment")
    else:
        if comparison["candidate_output_receipt_id"] != custody_commitment["output_receipt_id"]:
            errors.append(
                f"packet {packet_id}: comparison candidate output differs from custody output"
            )
        if comparison["candidate_output_sha256"] != custody_commitment["output_sha256"]:
            errors.append(
                f"packet {packet_id}: comparison candidate hash differs from custody output"
            )
    if (
        candidate_receipt is not None
        and candidate_receipt.get("sha256") != comparison["candidate_output_sha256"]
    ):
        errors.append(f"packet {packet_id}: comparison candidate output hash differs")
    if comparison["external_output_receipt_id"] != reproduction["output_receipt_id"]:
        errors.append(f"packet {packet_id}: comparison external output differs")
    if comparison["external_output_sha256"] != reproduction["output_sha256"]:
        errors.append(f"packet {packet_id}: comparison external output hash differs")
    if comparison["candidate_output_receipt_id"] == comparison["external_output_receipt_id"]:
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
            candidate_metric = _json_pointer(candidate_document, comparison["metric_json_pointer"])
            external_metric = _json_pointer(output_document, comparison["metric_json_pointer"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"packet {packet_id}: comparison metric cannot resolve: {exc}")
        else:
            tolerance = comparison["absolute_tolerance"]
            numeric_values = (candidate_metric, external_metric, tolerance)
            if any(type(value) not in {int, float} for value in numeric_values) or any(
                type(value) is float and not math.isfinite(value) for value in numeric_values
            ):
                errors.append(f"packet {packet_id}: comparison metrics must be finite numbers")
            else:
                try:
                    difference = abs(Decimal(str(candidate_metric)) - Decimal(str(external_metric)))
                    agreement = difference <= Decimal(str(tolerance))
                except (InvalidOperation, ValueError):
                    errors.append(f"packet {packet_id}: comparison metrics cannot be evaluated")
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
        if comparison_document is not None and not _strict_json_equal(
            comparison_document, expected_comparison
        ):
            errors.append(f"packet {packet_id}: comparison receipt differs from computed outputs")
        causes = set(packet["adjudication"]["cause_codes"])
        outcome = packet["adjudication"]["packet_outcome"]
        disagreement_cause = "external-replication-disagreed"
        if agreement and disagreement_cause in causes:
            errors.append(f"packet {packet_id}: agreement cannot claim external disagreement cause")
        if not agreement and (outcome != "failed" or disagreement_cause not in causes):
            errors.append(f"packet {packet_id}: external disagreement requires failed adjudication")

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
            errors.append(f"packet {packet_id}: reveal must be authorized by evaluator_custodian")
        if not packet["holdout_started"]:
            errors.append(f"packet {packet_id}: reveal cannot precede holdout execution")
        if LIFECYCLE_ORDER[packet["lifecycle_phase"]] < LIFECYCLE_ORDER["reproduced"]:
            errors.append(f"packet {packet_id}: reveal cannot precede reproduction")
        if reveal["post_reveal_holdout_sha256"] != custody["hidden_holdout_manifest"]["sha256"]:
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
                errors.append(f"packet {packet_id}: output commitment must reference raw-results")
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
                    "post_reveal_holdout_sha256": reveal["post_reveal_holdout_sha256"],
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
        errors.append(f"packet {packet_id}: invalid {expected_kind} receipt {receipt_id!r}")
        return None
    if not isinstance(artifact_ref, str):
        errors.append(f"packet {packet_id}: {expected_kind} receipt lacks immutable ref")
        return None
    errors.extend(_artifact_ref_errors(artifact_ref, receipt, root, packet_id, expected_kind))
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
    for item in _artifact_items(initial, "requested_tests", "id", packet_id, errors):
        test_id = item.get("id")
        if not isinstance(test_id, str) or not test_id:
            continue
        if type(item.get("blocking")) is not bool:
            errors.append(f"packet {packet_id}: requested test {test_id} lacks Boolean blocking")
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
    if not (len(responses) == len(response_refs) == len(rereviews) == len(rereview_refs)):
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
        response_prior_ref = f"{response.get('review_commit')}:{response.get('review_artifact')}"
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
            errors.append(f"packet {packet_id}: re-review round {round_index} prior ref mismatch")
        if rereview.get("builder_response_ref") != response_ref:
            errors.append(
                f"packet {packet_id}: re-review round {round_index} response ref mismatch"
            )
        errors.extend(_review_commit_binding_errors(rereview, packet, root, packet_id))
        errors.extend(
            _response_provenance_errors(response, rereview, packet, root, packet_id, round_index)
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
            errors.append(f"packet {packet_id}: protocol_rule_sha256 differs from packet rules")
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
        for dependency in requirement["dependencies"]:
            if packet_outcomes.get(dependency) != "passed":
                errors.append(
                    f"packet {packet_id}: lifecycle {phase} started before dependency "
                    f"{dependency} passed"
                )
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
            _validate_external_replication(packet, receipts, packet_id, campaign, internal_facts)
        )
    errors.extend(_validate_custody(packet, receipts, packet_id))
    errors.extend(_validate_raw_evidence_contract(packet, receipts, packet_id, campaign_base, root))

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
    computed = {"pass": "passed", "fail": "failed", "blocked": "blocked"}[true_outcomes[0]]
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
        errors.append(f"packet {packet_id}: cause codes {sorted(causes)} do not match {computed}")

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


def _validate_campaign(
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
        manifest_schema = _load_json(root / "schemas/viability/protocol-manifest-v2.schema.json")
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
    if requirements_document is not None:
        try:
            _validate_structured_graph(requirements_document)
        except ValueError as exc:
            return [f"requirements: invalid structured graph: {exc}"]

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
    missing_frozen_rules = set(required_packets) - set(protocol_manifest["packet_rule_sha256"])
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


def validate_campaign(
    campaign_path: Path | str,
    *,
    repo_root: Path | str | None = None,
    requirements_document: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return fail-closed errors without propagating hostile-input failures."""
    try:
        return _validate_campaign(
            campaign_path,
            repo_root=repo_root,
            requirements_document=requirements_document,
        )
    except Exception as exc:
        # Campaigns and their referenced artifacts are untrusted review inputs.
        return [f"campaign: validation failed closed: {type(exc).__name__}: {exc}"]


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
