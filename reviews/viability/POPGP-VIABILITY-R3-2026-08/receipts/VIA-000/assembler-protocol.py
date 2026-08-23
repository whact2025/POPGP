"""Fail-closed assembler for the frozen VIA-000 R3 evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

SNAPSHOT_REF_PREFIX = "refs/tags/popgp-via000-r3-protocol-"


def _load_json(path: Path) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _load_json_bytes(content: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {label}")

    return json.loads(
        content.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: Any) -> None:
    path.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _safe_source(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"unsafe evidence path {relative!r}")
    candidate = (root / relative).resolve()
    candidate.relative_to(root.resolve())
    if not candidate.is_file():
        raise ValueError(f"evidence file is missing: {relative}")
    return candidate


def _parse_bindings(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use PLATFORM=PATH: {value!r}")
        platform, raw_path = value.split("=", 1)
        if platform in result or not platform or not raw_path:
            raise ValueError(f"duplicate or empty {label}: {value!r}")
        result[platform] = Path(raw_path).resolve()
    return result


def _prefixed(platform: str, relative: str) -> str:
    return f"evidence/{platform}/{relative}"


def _git_output(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        check=False,
        timeout=20,
    )
    if completed.returncode != 0:
        raise ValueError(f"Git identity query failed: {' '.join(arguments)}")
    return completed.stdout


def _verify_protocol_identity(
    args: argparse.Namespace, protocol: dict[str, Any]
) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    if not args.protocol_source_ref.startswith(SNAPSHOT_REF_PREFIX):
        raise ValueError("protocol source ref is not an R3 snapshot tag")
    protocol_source_commit = args.protocol_source_ref.removeprefix(SNAPSHOT_REF_PREFIX)
    if re.fullmatch(r"[0-9a-f]{40}", protocol_source_commit) is None:
        raise ValueError("protocol source ref does not end in a full Git commit")
    object_type = _git_output(repo_root, "cat-file", "-t", args.protocol_source_ref).decode(
        "ascii"
    ).strip()
    if object_type != "commit":
        raise ValueError("protocol source ref must be a lightweight commit tag")
    resolved = _git_output(repo_root, "rev-parse", "--verify", args.protocol_source_ref).decode(
        "ascii"
    ).strip()
    if resolved != protocol_source_commit:
        raise ValueError("protocol source ref resolves to a different commit")

    for path in (args.protocol.resolve(), args.schema.resolve()):
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"protocol artifact is outside repository: {path}") from exc
        frozen = _git_output(
            repo_root, "cat-file", "blob", f"{protocol_source_commit}:{relative}"
        )
        if path.read_bytes() != frozen:
            raise ValueError(f"protocol artifact differs from snapshot bytes: {relative}")

    parameters = protocol["parameters"]
    artifacts = (
        ("runner_protocol_path", "runner_protocol_sha256"),
        ("raw_results_schema_path", "raw_results_schema_sha256"),
        ("assembler_protocol_path", "assembler_protocol_sha256"),
        ("mutation_runner_protocol_path", "mutation_runner_protocol_sha256"),
        ("dispatch_guard_protocol_path", "dispatch_guard_protocol_sha256"),
        ("workflow_protocol_path", "workflow_protocol_sha256"),
        ("validator_package_init_path", "validator_package_init_sha256"),
    )
    for path_field, hash_field in artifacts:
        relative = parameters[path_field]
        frozen = _git_output(
            repo_root, "cat-file", "blob", f"{protocol_source_commit}:{relative}"
        )
        observed = hashlib.sha256(frozen).hexdigest()
        if observed != parameters[hash_field]:
            raise ValueError(f"frozen protocol artifact hash mismatch: {relative}")

    guard_path = parameters["dispatch_guard_protocol_path"]
    guard_blob = _git_output(
        repo_root, "cat-file", "blob", f"{protocol_source_commit}:{guard_path}"
    )
    captured_authorization_tag_oid = _git_output(
        repo_root, "rev-parse", "--verify", args.authorization_ref
    ).decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40}", captured_authorization_tag_oid) is None:
        raise ValueError("authorization ref did not resolve to a full Git object")
    with tempfile.TemporaryDirectory(prefix="via000-r3-guard-") as temporary:
        frozen_guard = Path(temporary) / "dispatch_guard.py"
        frozen_guard.write_bytes(guard_blob)
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(frozen_guard),
                "--repo-root",
                str(repo_root),
                "--event-name",
                "workflow_dispatch",
                "--github-ref",
                args.protocol_source_ref,
                "--github-sha",
                protocol_source_commit,
                "--authorization-ref",
                args.authorization_ref,
                "--expected-authorization-tag-oid",
                captured_authorization_tag_oid,
                "--allow-non-snapshot-worktree",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    if completed.returncode != 0:
        raise ValueError(f"campaign authorization failed: {completed.stderr.strip()}")
    try:
        authorization = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("campaign authorization guard returned invalid JSON") from exc
    if authorization.get("protocol_commit") != protocol_source_commit:
        raise ValueError("campaign authorization selected a different protocol snapshot")
    if authorization.get("authorization_tag_oid") != captured_authorization_tag_oid:
        raise ValueError("campaign authorization returned a different tag object")
    return authorization


def _expected_dispatch_identity(
    args: argparse.Namespace, authorization: dict[str, Any]
) -> dict[str, Any]:
    return {
        "event_name": "workflow_dispatch",
        "source_ref": authorization["source_ref"],
        "protocol_snapshot_commit": authorization["protocol_commit"],
        "authorization_ref": authorization["authorization_ref"],
        "authorization_tag_oid": authorization["authorization_tag_oid"],
        "authorization_commit": authorization["authorization_commit"],
        "authorization_record_sha256": authorization["authorization_record_sha256"],
        "producer_run_id": args.producer_run_id,
        "producer_run_attempt": args.producer_run_attempt,
    }


def _verify_attestation(
    subject: Path,
    bundle: Path,
    *,
    repository: str,
    signer_workflow: str,
    source_commit: str,
    predicate_type: str,
    minimum_gh_version: str,
) -> None:
    version = subprocess.run(
        ["gh", "--version"], capture_output=True, text=True, check=False
    )
    match = re.search(r"(?m)^gh version (\d+)\.(\d+)\.(\d+)", version.stdout)
    expected = tuple(int(item) for item in minimum_gh_version.split("."))
    if version.returncode != 0 or match is None or tuple(map(int, match.groups())) < expected:
        raise ValueError(
            f"GitHub CLI {minimum_gh_version}+ is required for attestation verification"
        )
    completed = subprocess.run(
        [
            "gh",
            "attestation",
            "verify",
            str(subject),
            "--bundle",
            str(bundle),
            "--repo",
            repository,
            "--signer-workflow",
            signer_workflow,
            "--source-digest",
            source_commit,
            "--predicate-type",
            predicate_type,
            "--deny-self-hosted-runners",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"producer attestation verification failed for {subject.name}: "
            f"{completed.stderr.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("GitHub attestation verifier returned malformed JSON") from exc
    if not isinstance(result, (list, dict)) or not result:
        raise ValueError("GitHub attestation verifier returned no verified attestation")


def _copy_manifest_evidence(
    platform: str,
    workspace: Path,
    destination: Path,
    attestation_contract: dict[str, Any],
    protocol_source_commit: str,
    dispatch_identity: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence_root = workspace / "evidence"
    summary_path = evidence_root / "platform-summary.json"
    manifest_path = evidence_root / "evidence-manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"{platform}: platform summary or evidence manifest is missing")
    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path)
    if not isinstance(summary, dict) or not isinstance(manifest, list):
        raise ValueError(f"{platform}: malformed platform evidence documents")
    producer = summary.get("producer_attestation")
    expected_producer = {
        "repository": attestation_contract["repository"],
        "signer_workflow": attestation_contract["signer_workflow"],
        "source_commit": protocol_source_commit,
        "bundle_path": attestation_contract["bundle_path"],
        "subject_paths": attestation_contract["subject_paths"],
    }
    if (
        summary.get("protocol_source_commit") != protocol_source_commit
        or summary.get("dispatch_identity") != dispatch_identity
        or producer != expected_producer
    ):
        raise ValueError(f"{platform}: producer-attestation identity differs from frozen contract")
    bundle_path = _safe_source(workspace, producer["bundle_path"])
    _verify_attestation(
        summary_path,
        bundle_path,
        repository=attestation_contract["repository"],
        signer_workflow=attestation_contract["signer_workflow"],
        source_commit=protocol_source_commit,
        predicate_type=attestation_contract["predicate_type"],
        minimum_gh_version=attestation_contract["minimum_gh_version"],
    )
    _verify_attestation(
        manifest_path,
        bundle_path,
        repository=attestation_contract["repository"],
        signer_workflow=attestation_contract["signer_workflow"],
        source_commit=protocol_source_commit,
        predicate_type=attestation_contract["predicate_type"],
        minimum_gh_version=attestation_contract["minimum_gh_version"],
    )

    observed_paths: set[str] = set()
    rewritten_manifest: list[dict[str, Any]] = []
    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError(f"{platform}: malformed evidence entry")
        relative = entry.get("path")
        if relative in observed_paths:
            raise ValueError(f"{platform}: duplicate evidence path {relative!r}")
        observed_paths.add(relative)
        source = _safe_source(workspace, relative)
        if source.stat().st_size != entry.get("byte_count") or _sha256(source) != entry.get(
            "sha256"
        ):
            raise ValueError(f"{platform}: evidence bytes disagree with manifest: {relative}")
        target_relative = _prefixed(platform, relative)
        target = destination / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rewritten = dict(entry)
        rewritten["path"] = target_relative
        rewritten_manifest.append(rewritten)

    if set(summary.get("evidence_paths", [])) != observed_paths:
        raise ValueError(f"{platform}: summary paths differ from evidence manifest")

    rewritten_summary = dict(summary)
    rewritten_summary["evidence_paths"] = sorted(
        _prefixed(platform, path) for path in observed_paths
    )
    for command in rewritten_summary.get("command_results", {}).values():
        for field in ("result_path", "stdout_path", "stderr_path"):
            command[field] = _prefixed(platform, command[field])
    for artifact in rewritten_summary.get("artifact_results", {}).values():
        artifact["evidence_path"] = _prefixed(platform, artifact["evidence_path"])
    for mutation in rewritten_summary.get("mutation_results", []):
        mutation["evidence_paths"] = [
            _prefixed(platform, path) for path in mutation["evidence_paths"]
        ]
    provenance_root = destination / "evidence" / platform / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    provenance_sources = (
        (summary_path, "platform-summary.json", "producer-summary", "application/json"),
        (manifest_path, "evidence-manifest.json", "producer-manifest", "application/json"),
        (
            bundle_path,
            "producer-attestation.sigstore.json",
            "producer-attestation",
            "application/json",
        ),
    )
    provenance_paths: dict[str, str] = {}
    for source, name, role, media_type in provenance_sources:
        target = provenance_root / name
        shutil.copyfile(source, target)
        relative = target.relative_to(destination).as_posix()
        provenance_paths[name] = relative
        rewritten_manifest.append(
            {
                "platform_family": platform,
                "path": relative,
                "sha256": _sha256(target),
                "byte_count": target.stat().st_size,
                "media_type": media_type,
                "role": role,
            }
        )
    rewritten_summary["producer_attestation"] = {
        "repository": attestation_contract["repository"],
        "signer_workflow": attestation_contract["signer_workflow"],
        "source_commit": protocol_source_commit,
        "bundle_path": provenance_paths["producer-attestation.sigstore.json"],
        "subject_paths": [
            provenance_paths["platform-summary.json"],
            provenance_paths["evidence-manifest.json"],
        ],
    }
    return rewritten_summary, rewritten_manifest


def _run_frozen_precommit_validator(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    authorization: dict[str, Any],
    raw_path: Path,
) -> None:
    repo_root = args.repo_root.resolve()
    protocol_commit = authorization["protocol_commit"]
    authorization_commit = authorization["authorization_commit"]
    contract = protocol["parameters"]["authorization_contract"]
    manifest_path = contract["protocol_manifest_path"]
    manifest_bytes = _git_output(
        repo_root, "cat-file", "blob", f"{authorization_commit}:{manifest_path}"
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != authorization[
        "protocol_manifest_sha256"
    ]:
        raise ValueError("authorized protocol manifest differs before validation")
    manifest = _load_json_bytes(manifest_bytes, manifest_path)
    manifest_hashes = {
        item["path"]: item["sha256"] for item in manifest.get("contract_files", [])
    }
    if len(manifest_hashes) != len(manifest.get("contract_files", [])):
        raise ValueError("authorized protocol manifest has duplicate contract paths")

    bundle_contract = contract["validator_bundle"]
    if not isinstance(bundle_contract, list) or not bundle_contract:
        raise ValueError("frozen precommit validator bundle is empty")
    bundle_paths: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="via000-r3-validator-") as temporary:
        bundle_root = Path(temporary)
        for item in bundle_contract:
            if not isinstance(item, dict) or set(item) != {
                "source_path",
                "bundle_path",
                "sha256",
            }:
                raise ValueError("frozen validator bundle entry is malformed")
            source_path = item["source_path"]
            bundle_path = item["bundle_path"]
            if (
                not isinstance(source_path, str)
                or not isinstance(bundle_path, str)
                or not source_path
                or not bundle_path
                or "\\" in source_path
                or "\\" in bundle_path
                or Path(source_path).is_absolute()
                or Path(bundle_path).is_absolute()
                or ".." in Path(source_path).parts
                or ".." in Path(bundle_path).parts
                or bundle_path in bundle_paths
            ):
                raise ValueError("unsafe or duplicate frozen validator bundle path")
            bundle_paths.add(bundle_path)
            frozen = _git_output(
                repo_root, "cat-file", "blob", f"{protocol_commit}:{source_path}"
            )
            observed = hashlib.sha256(frozen).hexdigest()
            if observed != item["sha256"] or manifest_hashes.get(source_path) != observed:
                raise ValueError(f"validator dependency is not manifest-bound: {source_path}")
            worktree_source = (repo_root / source_path).resolve()
            try:
                worktree_source.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError("validator dependency escapes repository") from exc
            if not worktree_source.is_file() or worktree_source.read_bytes() != frozen:
                raise ValueError(f"validator worktree source differs from snapshot: {source_path}")
            destination = bundle_root / bundle_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(frozen)

        packet_path = contract["packet_path"]
        packet_bytes = _git_output(
            repo_root, "cat-file", "blob", f"{authorization_commit}:{packet_path}"
        )
        if hashlib.sha256(packet_bytes).hexdigest() != authorization["packet_sha256"]:
            raise ValueError("authorized packet differs before precommit validation")
        protocol_path = protocol["parameters"].get(
            "primary_protocol_path",
            "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json",
        )
        schema_path = protocol["parameters"]["raw_results_schema_path"]
        frozen_protocol = _git_output(
            repo_root, "cat-file", "blob", f"{protocol_commit}:{protocol_path}"
        )
        frozen_schema = _git_output(
            repo_root, "cat-file", "blob", f"{protocol_commit}:{schema_path}"
        )
        input_root = bundle_root / "inputs"
        input_root.mkdir()
        packet_file = input_root / "packet.yaml"
        protocol_file = input_root / "protocol.json"
        schema_file = input_root / "schema.json"
        packet_file.write_bytes(packet_bytes)
        protocol_file.write_bytes(frozen_protocol)
        schema_file.write_bytes(frozen_schema)
        validator = bundle_root / "check_viability_campaign.py"
        environment = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"):
            environment.pop(name, None)
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                str(validator),
                "--via000-raw-results",
                str(raw_path),
                "--via000-protocol",
                str(protocol_file),
                "--via000-schema",
                str(schema_file),
                "--via000-packet",
                str(packet_file),
                "--via000-repo-root",
                str(repo_root),
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
            timeout=600,
        )
        try:
            result = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "frozen precommit validator returned invalid JSON: "
                + completed.stderr.strip()
            ) from exc
        errors = result.get("errors") if isinstance(result, dict) else None
        if completed.returncode != 0 or errors != []:
            detail = errors[0] if isinstance(errors, list) and errors else completed.stderr.strip()
            raise ValueError(f"assembled raw results fail frozen validation: {detail}")


def assemble(args: argparse.Namespace) -> None:
    protocol = _load_json(args.protocol)
    schema = _load_json(args.schema)
    authorization = _verify_protocol_identity(args, protocol)
    args.protocol_source_commit = authorization["protocol_commit"]
    dispatch_identity = _expected_dispatch_identity(args, authorization)
    contract = protocol["parameters"]["raw_results_contract"]
    required_platforms = contract["required_platforms"]
    platform_roots = _parse_bindings(args.platform_root, "platform root")
    if set(platform_roots) != set(required_platforms):
        raise ValueError("platform inputs must exactly match frozen platforms")
    if args.output_dir.exists():
        raise ValueError(f"output directory must not exist: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent)
    )
    try:
        platforms: dict[str, Any] = {}
        manifest: list[dict[str, Any]] = []
        expected_mutations = set(contract["required_mutation_ids"])
        for platform in required_platforms:
            summary, platform_manifest = _copy_manifest_evidence(
                platform,
                platform_roots[platform],
                temporary,
                contract["producer_attestation"],
                args.protocol_source_commit,
                dispatch_identity,
            )
            if summary.get("candidate_commit") != protocol["parameters"]["candidate_commit"]:
                raise ValueError(f"{platform}: candidate commit mismatch")
            if summary.get("candidate_tree") != protocol["parameters"]["candidate_tree"]:
                raise ValueError(f"{platform}: candidate tree mismatch")
            if set(summary.get("command_results", {})) != set(
                contract["required_command_contracts"]
            ):
                raise ValueError(f"{platform}: incomplete command set")
            if any(item.get("exit_code") != 0 for item in summary["command_results"].values()):
                raise ValueError(f"{platform}: nonzero command result")
            if set(summary.get("artifact_results", {})) != set(contract["required_artifact_paths"]):
                raise ValueError(f"{platform}: incomplete artifact set")
            mutation_results = summary.get("mutation_results", [])
            if (
                {item["mutation_id"] for item in mutation_results} != expected_mutations
                or any(item.get("rejected") is not True for item in mutation_results)
            ):
                raise ValueError(f"{platform}: incomplete mutation set")
            if summary.get("mutation_count") != len(mutation_results):
                raise ValueError(f"{platform}: mutation count differs from retained results")
            if summary.get("mutations_rejected") is not True:
                raise ValueError(f"{platform}: mutation rejection was not derived as true")
            summary["overall_passed"] = all(
                summary[field]
                for field in (
                    "commands_passed",
                    "semantic_contract_passed",
                    "visual_contract_passed",
                    "source_boundary_passed",
                    "environment_boundary_passed",
                    "pdf_passed",
                    "mutations_rejected",
                )
            )
            platforms[platform] = summary
            manifest.extend(platform_manifest)

        raw_document = {
            "schema_version": 1,
            "campaign_id": "POPGP-VIABILITY-R3-2026-08",
            "packet_id": "VIA-000",
            "candidate_commit": protocol["parameters"]["candidate_commit"],
            "candidate_tree": protocol["parameters"]["candidate_tree"],
            "protocol_source_commit": args.protocol_source_commit,
            "dispatch_identity": dispatch_identity,
            "platforms": platforms,
            "evidence_manifest": sorted(manifest, key=lambda item: item["path"]),
            "capabilities": {
                "evidence-contract": all(item["commands_passed"] for item in platforms.values()),
                "cross-platform-reproduction": all(
                    all(
                        item[field]
                        for field in (
                            "commands_passed",
                            "semantic_contract_passed",
                            "visual_contract_passed",
                            "source_boundary_passed",
                            "environment_boundary_passed",
                            "pdf_passed",
                        )
                    )
                    for item in platforms.values()
                ),
                "mutation-rejection": all(
                    item["mutations_rejected"] for item in platforms.values()
                ),
            },
            "blocked": False,
        }
        raw_document["failed"] = not all(raw_document["capabilities"].values())
        schema_errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(raw_document),
            key=lambda error: list(error.path),
        )
        if schema_errors:
            raise ValueError("assembled raw results fail schema: " + schema_errors[0].message)
        raw_path = temporary / "raw-results.json"
        _write_json(raw_path, raw_document)
        _run_frozen_precommit_validator(args, protocol, authorization, raw_path)
        commitment = {
            "packet_id": raw_document["packet_id"],
            "committed_by": args.committed_by,
            "committed_at": args.committed_at,
            "output_receipt_id": "raw-results",
            "output_sha256": _sha256(raw_path),
        }
        _write_json(temporary / "output-commitment.json", commitment)
        os.replace(temporary, args.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--protocol", type=Path, default=here / "VIA-000.json")
    parser.add_argument("--schema", type=Path, default=here / "VIA-000-RAW-RESULTS.schema.json")
    parser.add_argument("--repo-root", type=Path, default=here.parents[1])
    parser.add_argument("--platform-root", action="append", default=[], required=True)
    parser.add_argument("--protocol-source-ref", required=True)
    parser.add_argument("--authorization-ref", required=True)
    parser.add_argument("--producer-run-id", required=True)
    parser.add_argument("--producer-run-attempt", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--committed-by", required=True)
    parser.add_argument("--committed-at", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[1-9][0-9]*", args.producer_run_id) is None:
        parser.error("--producer-run-id must be a positive decimal identifier")
    if args.producer_run_attempt < 1:
        parser.error("--producer-run-attempt must be positive")
    assemble(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
