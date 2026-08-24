"""Validate canonical envelopes and aggregate six VIA-000 R3 containment cells."""

from __future__ import annotations

import argparse
import base64
import binascii
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
MEMBER_NAMES = (
    "containment-result.json",
    "proof.json",
    "stderr.txt",
    "stdout.txt",
)
MAX_MEMBER_BYTES = 262_144
MAX_TOTAL_DECODED_BYTES = 524_288
MAX_ENVELOPE_BYTES = 131_072
MAX_CACHE_KEY_CHARACTERS = 511
CACHE_NAMESPACE = "via000-r3-envelope-v1"
CACHE_ROOT = Path(".via000-r3-proof-cache")
DIGESTS = {
    "VIA000_DIGEST_UBUNTU_CANDIDATE": ("ubuntu-latest-x86_64", "candidate"),
    "VIA000_DIGEST_UBUNTU_PDF": ("ubuntu-latest-x86_64", "pdf"),
    "VIA000_DIGEST_UBUNTU_MUTATION": ("ubuntu-latest-x86_64", "mutation"),
    "VIA000_DIGEST_WINDOWS_CANDIDATE": ("windows-x86_64", "candidate"),
    "VIA000_DIGEST_WINDOWS_PDF": ("windows-x86_64", "pdf"),
    "VIA000_DIGEST_WINDOWS_MUTATION": ("windows-x86_64", "mutation"),
}
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
    "export_created_after_teardown",
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
    "proof_envelope_schema_sha256",
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
    "proof_envelope_schema_sha256",
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
    "token_restriction_flags",
    "token_integrity_sid",
    "enabled_privilege_count",
    "enabled_privileges",
    "protected_label_policy",
    "export_owner_sid",
    "export_dacl_policy",
    "export_integrity_sid",
    "export_mandatory_policy",
    "export_root_control_flags",
    "export_root_dacl_protected",
    "export_root_native_ace_count",
    "export_root_managed_ace_count",
    *HASH_FIELDS,
    *TRUE_FIELDS,
    "primitive",
    "privilege_separation",
    "active_processes_after_teardown",
}
ENVELOPE_FIELDS = {
    "schema_version",
    "envelope_kind",
    "identity",
    "members",
    "total_decoded_bytes",
}
IDENTITY_FIELDS = {
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
}
MEMBER_FIELDS = {"base64", "sha256", "size"}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    folded: set[str] = set()
    for key, value in pairs:
        folded_key = key.casefold()
        if key in document:
            raise ValueError(f"duplicate JSON object member: {key}")
        if folded_key in folded:
            raise ValueError(f"case-fold-colliding JSON object member: {key}")
        document[key] = value
        folded.add(folded_key)
    return document


def _parse_json(content: bytes, description: str) -> dict[str, Any]:
    if content.startswith(b"\xef\xbb\xbf") or b"\r" in content:
        raise ValueError(f"{description} is not UTF-8/LF/no-BOM JSON")
    try:
        document = json.loads(
            content.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{description} is not strict UTF-8") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{description} is not one JSON object")
    return document


def canonical_envelope_bytes(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def build_envelope(identity: dict[str, str], subjects: dict[str, bytes]) -> bytes:
    """Build the frozen envelope form used by tests and the PowerShell producer."""
    if set(identity) != IDENTITY_FIELDS:
        raise ValueError("envelope identity fields differ")
    if set(subjects) != set(MEMBER_NAMES):
        raise ValueError("envelope subject names differ")
    total = sum(len(content) for content in subjects.values())
    if total > MAX_TOTAL_DECODED_BYTES:
        raise ValueError("envelope decoded total exceeds the frozen limit")
    members: dict[str, dict[str, str | int]] = {}
    for name in MEMBER_NAMES:
        content = subjects[name]
        if len(content) > MAX_MEMBER_BYTES:
            raise ValueError(f"envelope member exceeds the frozen limit: {name}")
        members[name] = {
            "base64": base64.b64encode(content).decode("ascii"),
            "sha256": _sha256_bytes(content),
            "size": len(content),
        }
    document: dict[str, Any] = {
        "envelope_kind": "via000-r3-hosted-containment-envelope",
        "identity": identity,
        "members": members,
        "schema_version": 1,
        "total_decoded_bytes": total,
    }
    content = canonical_envelope_bytes(document)
    if len(content) > MAX_ENVELOPE_BYTES:
        raise ValueError("canonical envelope exceeds the frozen encoded limit")
    return content


def _decode_envelope(
    content: bytes, description: str
) -> tuple[dict[str, Any], dict[str, bytes], str]:
    if not content or len(content) > MAX_ENVELOPE_BYTES:
        raise ValueError(f"proof envelope encoded size is outside the frozen limit: {description}")
    document = _parse_json(content, f"proof envelope {description}")
    if canonical_envelope_bytes(document) != content:
        raise ValueError(f"proof envelope is not canonical JSON: {description}")
    if set(document) != ENVELOPE_FIELDS:
        raise ValueError(f"proof envelope fields differ: {description}")
    if document["schema_version"] != 1 or document["envelope_kind"] != (
        "via000-r3-hosted-containment-envelope"
    ):
        raise ValueError(f"proof envelope kind/version differs: {description}")
    identity = document["identity"]
    members = document["members"]
    if not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS:
        raise ValueError(f"proof envelope identity fields differ: {description}")
    if not isinstance(members, dict):
        raise ValueError(f"proof envelope members are not one object: {description}")
    member_names = list(members)
    if set(member_names) != set(MEMBER_NAMES) or len(
        {name.casefold() for name in member_names}
    ) != 4:
        raise ValueError(f"proof envelope member set differs: {description}")
    decoded: dict[str, bytes] = {}
    total = 0
    for name in MEMBER_NAMES:
        member = members[name]
        if not isinstance(member, dict) or set(member) != MEMBER_FIELDS:
            raise ValueError(f"proof envelope metadata differs for {name}: {description}")
        size = member["size"]
        digest = member["sha256"]
        encoded = member["base64"]
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or not (0 <= size <= MAX_MEMBER_BYTES)
        ):
            raise ValueError(f"proof envelope size differs for {name}: {description}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"proof envelope hash differs for {name}: {description}")
        if not isinstance(encoded, str) or len(encoded) > ((MAX_MEMBER_BYTES + 2) // 3) * 4:
            raise ValueError(f"proof envelope base64 size differs for {name}: {description}")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"proof envelope base64 differs for {name}: {description}") from exc
        if base64.b64encode(raw).decode("ascii") != encoded:
            raise ValueError(f"proof envelope base64 is noncanonical for {name}: {description}")
        if len(raw) != size or _sha256_bytes(raw) != digest:
            raise ValueError(f"proof envelope decoded size/hash differs for {name}: {description}")
        decoded[name] = raw
        total += len(raw)
    if total > MAX_TOTAL_DECODED_BYTES or document["total_decoded_bytes"] != total:
        raise ValueError(f"proof envelope decoded total differs: {description}")
    return document, decoded, _sha256_bytes(content)


def _load_envelope(path: Path) -> tuple[dict[str, Any], dict[str, bytes], str]:
    parent_items = list(path.parent.iterdir())
    if len(parent_items) != 1 or parent_items[0].name != "envelope.json":
        raise ValueError(f"proof artifact is not exactly one envelope: {path.parent}")
    stat = path.lstat()
    if not path.is_file() or path.is_symlink() or stat.st_nlink != 1:
        raise ValueError(f"proof envelope is not one regular single-link file: {path}")
    if stat.st_size <= 0 or stat.st_size > MAX_ENVELOPE_BYTES:
        raise ValueError(f"proof envelope encoded size is outside the frozen limit: {path}")
    return _decode_envelope(path.read_bytes(), str(path))


def _validate_cell(
    envelope: dict[str, Any],
    subjects: dict[str, bytes],
    path: Path,
    identity: dict[str, str],
) -> str:
    envelope_identity = envelope["identity"]
    for field, expected in identity.items():
        if envelope_identity.get(field) != expected:
            raise ValueError(f"envelope {field} differs across cells: {path}")
    platform = envelope_identity.get("platform_family")
    stage = envelope_identity.get("stage_id")
    if platform not in PLATFORMS or stage not in STAGES:
        raise ValueError(f"envelope platform/stage is outside the 2x3 matrix: {path}")
    cell = f"{platform}/{stage}"
    artifact = f"via000-r3-containment-proof-{platform}-{stage}"
    if (
        envelope_identity.get("cell") != cell
        or envelope_identity.get("artifact_name") != artifact
        or path.parent.name != artifact
    ):
        raise ValueError(f"envelope cell/artifact identity differs: {path}")

    document = _parse_json(subjects["proof.json"], f"inner proof {path}")
    if set(document) != CELL_FIELDS:
        raise ValueError(f"proof fields differ from frozen cell schema: {path}")
    if document["schema_version"] != 1 or document["proof_kind"] != (
        "via000-r3-hosted-containment-cell"
    ):
        raise ValueError(f"proof kind/version differs: {path}")
    for field, expected in envelope_identity.items():
        if document[field] != expected:
            raise ValueError(f"inner proof {field} differs from envelope: {path}")
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
    expected_token_flags = ["DISABLE_MAX_PRIVILEGE"] if platform == "windows-x86_64" else []
    expected_integrity_sid = "S-1-16-4096" if platform == "windows-x86_64" else ""
    expected_label_policy = (
        "medium-integrity-no-write-up-no-read-up"
        if platform == "windows-x86_64"
        else "owner-only-protected-root"
    )
    enabled_privileges = document.get("enabled_privileges")
    if (
        document.get("token_restriction_flags") != expected_token_flags
        or document.get("token_integrity_sid") != expected_integrity_sid
        or not isinstance(enabled_privileges, list)
        or enabled_privileges not in ([], ["SeChangeNotifyPrivilege"])
        or document.get("enabled_privilege_count") != len(enabled_privileges)
        or document.get("protected_label_policy") != expected_label_policy
    ):
        raise ValueError(f"proof token or protected-label policy differs: {path}")
    expected_export = (
        (
            r"S-1-(?:[0-9]+-)+[0-9]+",
            "protected-current-runner-full-control-v1",
            "S-1-16-8192",
            "NO_WRITE_UP",
            37892,
            True,
            1,
            1,
        )
        if platform == "windows-x86_64"
        else (r"", "owner-rwx-0700-v1", "", "owner-only", 0, False, 0, 0)
    )
    if (
        re.fullmatch(expected_export[0], document.get("export_owner_sid", "")) is None
        or document.get("export_dacl_policy") != expected_export[1]
        or document.get("export_integrity_sid") != expected_export[2]
        or document.get("export_mandatory_policy") != expected_export[3]
        or document.get("export_root_control_flags") != expected_export[4]
        or document.get("export_root_dacl_protected") is not expected_export[5]
        or document.get("export_root_native_ace_count") != expected_export[6]
        or document.get("export_root_managed_ace_count") != expected_export[7]
        or document.get("export_created_after_teardown") is not True
    ):
        raise ValueError(f"proof export security evidence differs: {path}")
    if any(
        not isinstance(document[field], str)
        or re.fullmatch(r"[0-9a-f]{64}", document[field]) is None
        for field in HASH_FIELDS
    ):
        raise ValueError(f"proof has a malformed SHA-256 binding: {path}")
    contained_bytes = subjects["containment-result.json"]
    if _sha256_bytes(contained_bytes) != document["containment_result_sha256"]:
        raise ValueError(f"containment result hash differs: {path}")
    contained = _parse_json(contained_bytes, f"inner containment result {path}")
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
        or contained.get("token_restriction_flags") != expected_token_flags
        or contained.get("token_integrity_sid") != expected_integrity_sid
        or contained.get("enabled_privileges") != enabled_privileges
        or contained.get("enabled_privilege_count") != len(enabled_privileges)
        or contained.get("protected_label_policy") != expected_label_policy
    ):
        raise ValueError(f"containment result predicates differ: {path}")
    if platform == "ubuntu-latest-x86_64" and (
        re.fullmatch(r"[1-9][0-9]*", contained.get("ephemeral_identity_uid", "")) is None
        or contained.get("ephemeral_identity_processes_empty") is not True
        or contained.get("ephemeral_identity_removed") is not True
    ):
        raise ValueError(f"ephemeral Ubuntu identity proof differs: {path}")
    if contained.get("stdout_sha256") != _sha256_bytes(subjects["stdout.txt"]) or contained.get(
        "stderr_sha256"
    ) != _sha256_bytes(subjects["stderr.txt"]):
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


def _expected_identity(args: argparse.Namespace) -> dict[str, str]:
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
    expected_workflow_ref = f"whact2025/POPGP/{WORKFLOW}@{args.source_ref}"
    if args.workflow_ref != expected_workflow_ref:
        raise ValueError("aggregate workflow ref differs from the exact source ref")
    return {
        "repository": "whact2025/POPGP",
        "workflow": WORKFLOW,
        "workflow_ref": args.workflow_ref,
        "event_name": "push",
        "source_ref": args.source_ref,
        "source_sha": args.source_sha,
        "run_id": args.run_id,
        "run_attempt": args.run_attempt,
    }


def _aggregate_envelopes(
    items: list[tuple[Path, bytes]], identity: dict[str, str]
) -> dict[str, Any]:
    cells: set[str] = set()
    hashes: dict[str, str] = {}
    bundle_hashes: dict[str, str] | None = None
    for path, content in items:
        envelope, subjects, envelope_hash = _decode_envelope(content, str(path))
        cell = _validate_cell(envelope, subjects, path, identity)
        if cell in cells:
            raise ValueError(f"duplicate hosted containment proof cell: {cell}")
        proof = _parse_json(subjects["proof.json"], f"inner proof {path}")
        observed_bundle = {field: proof[field] for field in BUNDLE_HASH_FIELDS}
        if bundle_hashes is None:
            bundle_hashes = observed_bundle
        elif observed_bundle != bundle_hashes:
            raise ValueError(f"hosted containment proof bundle hashes differ: {cell}")
        cells.add(cell)
        hashes[cell] = envelope_hash
    if cells != EXPECTED_CELLS:
        raise ValueError(f"hosted containment proof matrix differs: {sorted(cells)}")
    return {
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


def _aggregate_bytes(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _retained_name(platform: str, stage: str) -> str:
    return f"envelope-{platform}-{stage}.json"


def _load_cache_envelope(root: Path, platform: str, stage: str) -> bytes:
    cell_root = root / platform / stage
    if not cell_root.is_dir() or cell_root.is_symlink():
        raise ValueError(f"cache cell root is not one ordinary directory: {platform}/{stage}")
    children = list(cell_root.iterdir())
    if (
        len(children) != 1
        or children[0].name != "envelope.json"
        or len({child.name.casefold() for child in children}) != 1
    ):
        raise ValueError(f"cache cell root differs from one exact envelope: {platform}/{stage}")
    envelope = children[0]
    stat = envelope.lstat()
    if (
        not envelope.is_file()
        or envelope.is_symlink()
        or stat.st_nlink != 1
        or stat.st_size <= 0
        or stat.st_size > MAX_ENVELOPE_BYTES
        or stat.st_mode & 0o111
    ):
        raise ValueError(f"cache envelope metadata differs: {platform}/{stage}")
    return envelope.read_bytes()


def collect_cache_envelopes(
    args: argparse.Namespace, environment: dict[str, str] | os._Environ[str]
) -> dict[str, Any]:
    identity = _expected_identity(args)
    cache_root = Path.cwd() / CACHE_ROOT
    if not cache_root.is_dir() or cache_root.is_symlink():
        raise ValueError("cache transport root is not one ordinary workspace-relative directory")
    platform_entries = list(cache_root.iterdir())
    if (
        {entry.name for entry in platform_entries} != set(PLATFORMS)
        or len(platform_entries) != 2
        or len({entry.name.casefold() for entry in platform_entries}) != 2
        or any(not entry.is_dir() or entry.is_symlink() for entry in platform_entries)
    ):
        raise ValueError("cache platform root set differs")
    for platform_root in platform_entries:
        stage_entries = list(platform_root.iterdir())
        if (
            {entry.name for entry in stage_entries} != set(STAGES)
            or len(stage_entries) != 3
            or len({entry.name.casefold() for entry in stage_entries}) != 3
            or any(not entry.is_dir() or entry.is_symlink() for entry in stage_entries)
        ):
            raise ValueError(f"cache stage root set differs: {platform_root.name}")
    observed_digests: set[str] = set()
    retained: list[tuple[Path, bytes]] = []
    for variable, (platform, stage) in DIGESTS.items():
        digest = environment.get(variable, "")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"cache digest output differs: {variable}")
        if digest in observed_digests:
            raise ValueError(f"duplicate or overwritten cache digest: {variable}")
        observed_digests.add(digest)
        content = _load_cache_envelope(cache_root, platform, stage)
        if _sha256_bytes(content) != digest:
            raise ValueError(f"cache envelope digest differs from job output: {variable}")
        artifact = f"via000-r3-containment-proof-{platform}-{stage}"
        retained.append((Path(artifact) / "envelope.json", content))
    aggregate_document = _aggregate_envelopes(retained, identity)
    if args.output_root.exists():
        raise ValueError("consolidated proof output root already exists")
    args.output_root.mkdir(mode=0o700, parents=False)
    try:
        for logical_path, content in retained:
            envelope = _parse_json(content, f"cache transport {logical_path}")
            platform = envelope["identity"]["platform_family"]
            stage = envelope["identity"]["stage_id"]
            output = args.output_root / _retained_name(platform, stage)
            with output.open("xb") as stream:
                stream.write(content)
        with (args.output_root / "aggregate.json").open("xb") as stream:
            stream.write(_aggregate_bytes(aggregate_document))
        verify_retained(args.output_root, identity)
    except BaseException:
        for child in args.output_root.iterdir():
            child.unlink(missing_ok=True)
        args.output_root.rmdir()
        raise
    return aggregate_document


def verify_retained(root: Path, identity: dict[str, str]) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("retained proof root is not one ordinary directory")
    expected_names = {
        _retained_name(platform, stage)
        for platform in PLATFORMS
        for stage in STAGES
    } | {"aggregate.json"}
    children = list(root.iterdir())
    names = [child.name for child in children]
    if (
        set(names) != expected_names
        or len(names) != 7
        or len({name.casefold() for name in names}) != 7
    ):
        raise ValueError("retained proof file set differs from the exact seven files")
    items: list[tuple[Path, bytes]] = []
    aggregate_content = b""
    for child in children:
        stat = child.lstat()
        if (
            not child.is_file()
            or child.is_symlink()
            or stat.st_nlink != 1
            or stat.st_size <= 0
            or stat.st_size > MAX_ENVELOPE_BYTES
        ):
            raise ValueError(f"retained proof subject metadata differs: {child.name}")
        content = child.read_bytes()
        if child.name == "aggregate.json":
            aggregate_content = content
            continue
        match = re.fullmatch(
            r"envelope-(ubuntu-latest-x86_64|windows-x86_64)-(candidate|pdf|mutation)\.json",
            child.name,
        )
        if match is None:
            raise ValueError(f"retained envelope name differs: {child.name}")
        platform, stage = match.groups()
        artifact = f"via000-r3-containment-proof-{platform}-{stage}"
        items.append((Path(artifact) / "envelope.json", content))
    document = _aggregate_envelopes(items, identity)
    if aggregate_content != _aggregate_bytes(document):
        raise ValueError("retained aggregate bytes differ from the six exact envelopes")
    return document


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    identity = _expected_identity(args)
    if not args.input_root.is_dir() or args.input_root.is_symlink():
        raise ValueError("proof envelope input root is not one ordinary directory")
    roots = sorted(args.input_root.iterdir())
    if len(roots) != 6 or any(not root.is_dir() or root.is_symlink() for root in roots):
        raise ValueError("expected exactly six ordinary hosted proof artifact directories")
    paths = [root / "envelope.json" for root in roots]
    items: list[tuple[Path, bytes]] = []
    for path in paths:
        _load_envelope(path)
        items.append((path, path.read_bytes()))
    aggregate_document = _aggregate_envelopes(items, identity)
    _atomic_write(args.output, aggregate_document)
    return aggregate_document


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--input-root", type=Path)
    modes.add_argument("--collect-cache-envelopes", action="store_true")
    modes.add_argument("--verify-retained", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--workflow-ref", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    args = parser.parse_args()
    try:
        if args.collect_cache_envelopes:
            if args.output_root is None:
                raise ValueError("collect mode requires --output-root")
            collect_cache_envelopes(args, os.environ)
        elif args.verify_retained is not None:
            verify_retained(args.verify_retained, _expected_identity(args))
        else:
            if args.output is None:
                raise ValueError("legacy aggregate mode requires --output")
            aggregate(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"containment proof aggregation failed: {exc}", file=__import__("sys").stderr)
        if args.output is not None:
            args.output.unlink(missing_ok=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
