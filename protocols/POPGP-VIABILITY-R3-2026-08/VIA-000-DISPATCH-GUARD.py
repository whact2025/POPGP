"""Fail-closed campaign authorization guard for a VIA-000 R3 dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SHA1_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PROTOCOL_REF_PREFIX = "refs/tags/popgp-via000-r3-protocol-"
AUTHORIZATION_REF_PREFIX = "refs/tags/popgp-via000-r3-authorization-"
PRIMARY_PATH = "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
SIGNATURE_MARKER = b"-----BEGIN SSH SIGNATURE-----"


def _git(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        check=False,
        timeout=20,
    )
    if completed.returncode != 0:
        raise ValueError(f"Git authorization query failed: {' '.join(arguments)}")
    return completed.stdout


def _strict_json(content: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {label}")

    return json.loads(
        content.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _canonical_json(document: Any) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _top_level_scalar(content: bytes, key: str) -> str:
    prefix = f"{key}:".encode()
    matches = [line for line in content.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"authorization blob requires one top-level {key!r} field")
    value = matches[0][len(prefix) :].strip()
    if not value or value[:1] in (b"'", b'"') or b" #" in value:
        raise ValueError(f"authorization blob has unsupported scalar syntax for {key!r}")
    return value.decode("ascii")


def _protocol_commit(repo_root: Path, protocol_ref: str) -> str:
    if not protocol_ref.startswith(PROTOCOL_REF_PREFIX):
        raise ValueError("dispatch ref is not an R3 protocol snapshot tag")
    commit = protocol_ref.removeprefix(PROTOCOL_REF_PREFIX)
    if SHA1_RE.fullmatch(commit) is None:
        raise ValueError("dispatch ref must end in a full lowercase 40-hex commit")
    object_type = _git(repo_root, "cat-file", "-t", protocol_ref).decode("ascii").strip()
    if object_type != "commit":
        raise ValueError("protocol snapshot ref must be a lightweight tag to a commit")
    resolved = _git(repo_root, "rev-parse", "--verify", protocol_ref).decode("ascii").strip()
    if resolved != commit:
        raise ValueError("protocol snapshot ref resolves to a different commit")
    return commit


def _authorization_contract(repo_root: Path, protocol_commit: str) -> dict[str, Any]:
    primary = _strict_json(
        _git(repo_root, "cat-file", "blob", f"{protocol_commit}:{PRIMARY_PATH}"),
        PRIMARY_PATH,
    )
    contract = primary.get("parameters", {}).get("authorization_contract")
    required = {
        "schema_version",
        "authorization_ref_prefix",
        "tag_object_binding",
        "signature_format",
        "required_signer_principal",
        "allowed_signers_path",
        "allowed_signers_sha256",
        "campaign_path",
        "packet_path",
        "protocol_manifest_path",
        "required_packet_lifecycle",
        "validator_bundle",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError("frozen authorization contract is absent or malformed")
    if (
        contract["schema_version"] != 1
        or contract["authorization_ref_prefix"] != AUTHORIZATION_REF_PREFIX
        or contract["tag_object_binding"] != "captured-oid-with-final-ref-check-v1"
        or contract["signature_format"] != "ssh"
        or contract["required_packet_lifecycle"] != "preregistered"
    ):
        raise ValueError("unsupported frozen authorization contract")
    return contract


def _allowed_signers(
    repo_root: Path, protocol_commit: str, contract: dict[str, Any]
) -> bytes:
    path = contract["allowed_signers_path"]
    content = _git(repo_root, "cat-file", "blob", f"{protocol_commit}:{path}")
    if hashlib.sha256(content).hexdigest() != contract["allowed_signers_sha256"]:
        raise ValueError("frozen authorization signer hash mismatch")
    lines = [
        line.strip()
        for line in content.decode("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    principal = contract["required_signer_principal"]
    if len(lines) != 1 or not lines[0].startswith(principal + " "):
        raise ValueError("exactly one frozen authorization signer is required")
    if " ssh-ed25519 " not in f" {lines[0]} ":
        raise ValueError("the frozen authorization signer must use Ed25519")
    return content


def _tag_record(
    repo_root: Path, authorization_ref: str, expected_tag_oid: str
) -> tuple[dict[str, Any], str, str]:
    if not authorization_ref.startswith(AUTHORIZATION_REF_PREFIX):
        raise ValueError("authorization ref is not an R3 authorization tag")
    expected_record_sha = authorization_ref.removeprefix(AUTHORIZATION_REF_PREFIX)
    if SHA256_RE.fullmatch(expected_record_sha) is None:
        raise ValueError("authorization ref must end in a full lowercase SHA-256")
    tag_oid = _git(repo_root, "rev-parse", "--verify", authorization_ref).decode("ascii").strip()
    if SHA1_RE.fullmatch(tag_oid) is None or tag_oid != expected_tag_oid:
        raise ValueError("authorization ref differs from the captured tag object")
    if _git(repo_root, "cat-file", "-t", tag_oid).decode("ascii").strip() != "tag":
        raise ValueError("authorization object must be a signed annotated tag")
    raw = _git(repo_root, "cat-file", "tag", tag_oid)
    try:
        headers, message = raw.split(b"\n\n", 1)
    except ValueError as exc:
        raise ValueError("authorization tag object is malformed") from exc
    header_values: dict[str, str] = {}
    for line in headers.decode("utf-8").splitlines():
        key, separator, value = line.partition(" ")
        if not separator or key in header_values:
            raise ValueError("authorization tag headers are malformed")
        header_values[key] = value
    expected_tag_name = authorization_ref.removeprefix("refs/tags/")
    if (
        header_values.get("type") != "commit"
        or header_values.get("tag") != expected_tag_name
        or SHA1_RE.fullmatch(header_values.get("object", "")) is None
    ):
        raise ValueError("authorization tag target or name is malformed")
    marker_index = message.find(SIGNATURE_MARKER)
    if marker_index <= 0:
        raise ValueError("authorization tag has no SSH signature")
    record_bytes = message[:marker_index]
    if hashlib.sha256(record_bytes).hexdigest() != expected_record_sha:
        raise ValueError("authorization record hash differs from ref suffix")
    record = _strict_json(record_bytes, "authorization tag record")
    if _canonical_json(record) != record_bytes:
        raise ValueError("authorization record is not canonical JSON")
    return record, tag_oid, header_values["object"]


def verify_campaign_authorization(
    repo_root: Path,
    *,
    event_name: str,
    github_ref: str,
    github_sha: str,
    authorization_ref: str,
    expected_authorization_tag_oid: str,
    require_checkout_head: bool = True,
) -> dict[str, Any]:
    """Return the unique signed campaign authorization or fail closed."""

    repo_root = repo_root.resolve()
    if event_name != "workflow_dispatch":
        raise ValueError("R3 protocol permits workflow_dispatch only")
    protocol_commit = _protocol_commit(repo_root, github_ref)
    if github_sha != protocol_commit:
        raise ValueError("github.sha differs from the dispatched protocol snapshot")
    if require_checkout_head:
        head = _git(repo_root, "rev-parse", "HEAD").decode("ascii").strip()
        if head != protocol_commit:
            raise ValueError("checkout HEAD differs from the dispatched protocol snapshot")

    contract = _authorization_contract(repo_root, protocol_commit)
    signers = _allowed_signers(repo_root, protocol_commit, contract)
    if SHA1_RE.fullmatch(expected_authorization_tag_oid) is None:
        raise ValueError("captured authorization tag object must be full lowercase 40-hex")
    record, tag_oid, parsed_target = _tag_record(
        repo_root, authorization_ref, expected_authorization_tag_oid
    )
    required_record_fields = {
        "schema_version",
        "campaign_id",
        "packet_id",
        "authorization_commit",
        "protocol_commit",
        "campaign_path",
        "campaign_sha256",
        "packet_path",
        "packet_sha256",
        "protocol_manifest_path",
        "protocol_manifest_sha256",
    }
    if not isinstance(record, dict) or set(record) != required_record_fields:
        raise ValueError("authorization record fields are not exact")
    if (
        record["schema_version"] != 1
        or record["campaign_id"] != "POPGP-VIABILITY-R3-2026-08"
        or record["packet_id"] != "VIA-000"
        or record["protocol_commit"] != protocol_commit
        or record["campaign_path"] != contract["campaign_path"]
        or record["packet_path"] != contract["packet_path"]
        or record["protocol_manifest_path"] != contract["protocol_manifest_path"]
    ):
        raise ValueError("authorization record differs from the frozen R3 contract")
    authorization_commit = record["authorization_commit"]
    if SHA1_RE.fullmatch(authorization_commit) is None:
        raise ValueError("authorization commit must be full lowercase 40-hex")
    target = _git(repo_root, "rev-parse", "--verify", f"{tag_oid}^{{commit}}").decode(
        "ascii"
    ).strip()
    if parsed_target != authorization_commit or target != authorization_commit:
        raise ValueError("authorization tag target differs from its signed record")

    with tempfile.TemporaryDirectory(prefix="via000-r3-signers-") as temporary:
        signers_path = Path(temporary) / "allowed_signers"
        signers_path.write_bytes(signers)
        verified = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "-c",
                "gpg.format=ssh",
                "-c",
                f"gpg.ssh.allowedSignersFile={signers_path}",
                "verify-tag",
                "--raw",
                tag_oid,
            ],
            capture_output=True,
            check=False,
            timeout=20,
        )
    if verified.returncode != 0:
        raise ValueError("authorization tag signature is not from the frozen signer")

    blobs: dict[str, bytes] = {}
    for label in ("campaign", "packet", "protocol_manifest"):
        path = record[f"{label}_path"]
        content = _git(repo_root, "cat-file", "blob", f"{authorization_commit}:{path}")
        if hashlib.sha256(content).hexdigest() != record[f"{label}_sha256"]:
            raise ValueError(f"authorized {label} blob hash mismatch")
        blobs[label] = content
    campaign_commit = _top_level_scalar(blobs["campaign"], "protocol_commit")
    packet_commit = _top_level_scalar(blobs["packet"], "protocol_commit")
    if campaign_commit != protocol_commit or packet_commit != protocol_commit:
        raise ValueError("authorized campaign and packet do not select this protocol snapshot")
    if _top_level_scalar(blobs["campaign"], "campaign_id") != record["campaign_id"]:
        raise ValueError("authorized campaign identity mismatch")
    if (
        _top_level_scalar(blobs["packet"], "campaign_id") != record["campaign_id"]
        or _top_level_scalar(blobs["packet"], "packet_id") != record["packet_id"]
    ):
        raise ValueError("authorized packet identity mismatch")
    if _top_level_scalar(blobs["packet"], "lifecycle_phase") != contract[
        "required_packet_lifecycle"
    ]:
        raise ValueError("authorized packet is not preregistered")
    if _top_level_scalar(blobs["packet"], "holdout_started") != "false":
        raise ValueError("authorized packet has already started holdout")
    if (
        _top_level_scalar(blobs["campaign"], "protocol_manifest_path")
        != record["protocol_manifest_path"]
        or _top_level_scalar(blobs["campaign"], "protocol_manifest_sha256")
        != record["protocol_manifest_sha256"]
    ):
        raise ValueError("authorized campaign manifest binding mismatch")

    final_tag_oid = _git(repo_root, "rev-parse", "--verify", authorization_ref).decode(
        "ascii"
    ).strip()
    if final_tag_oid != tag_oid:
        raise ValueError("authorization ref changed during immutable-object verification")

    return {
        "protocol_commit": protocol_commit,
        "source_ref": github_ref,
        "authorization_ref": authorization_ref,
        "authorization_tag_oid": tag_oid,
        "authorization_commit": authorization_commit,
        "authorization_record_sha256": authorization_ref.removeprefix(
            AUTHORIZATION_REF_PREFIX
        ),
        "campaign_sha256": record["campaign_sha256"],
        "packet_sha256": record["packet_sha256"],
        "protocol_manifest_sha256": record["protocol_manifest_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--github-ref", required=True)
    parser.add_argument("--github-sha", required=True)
    parser.add_argument("--authorization-ref", required=True)
    parser.add_argument("--expected-authorization-tag-oid", required=True)
    parser.add_argument("--allow-non-snapshot-worktree", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = verify_campaign_authorization(
        args.repo_root,
        event_name=args.event_name,
        github_ref=args.github_ref,
        github_sha=args.github_sha,
        authorization_ref=args.authorization_ref,
        expected_authorization_tag_oid=args.expected_authorization_tag_oid,
        require_checkout_head=not args.allow_non_snapshot_worktree,
    )
    content = json.dumps(result, allow_nan=False, sort_keys=True)
    if args.output_json is None:
        print(content)
    else:
        args.output_json.write_text(content + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
