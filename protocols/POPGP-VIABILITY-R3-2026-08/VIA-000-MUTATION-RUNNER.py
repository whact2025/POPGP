"""Prepare and finalize the contained VIA-000 R3 mutation-test matrix.

This trusted helper never executes candidate Python. The workflow prepares an
immutable plan, executes it through VIA-000-CONTAINMENT.ps1, proves complete
descendant quiescence, and only then invokes ``finalize`` to create evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any


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


def _write_json(path: Path, document: Any) -> None:
    path.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_explicit_file(path: Path, expected_root: Path, expected_sha256: str) -> Path:
    if not path.is_absolute() or not expected_root.is_absolute():
        raise ValueError("tool paths and roots must be absolute")
    cursor = path
    while True:
        info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError("environment Python has symlink or reparse ancestry")
        if cursor == expected_root:
            break
        if cursor.parent == cursor:
            raise ValueError("environment Python is outside the locked environment")
        cursor = cursor.parent
    path = path.resolve(strict=True)
    expected_root = expected_root.resolve(strict=True)
    if not path.is_relative_to(expected_root):
        raise ValueError("environment Python is outside the locked environment")
    if not path.is_file() or path.suffix.lower() in {".cmd", ".bat", ".ps1"}:
        raise ValueError("environment Python is not one regular executable")
    if _sha256(path) != expected_sha256:
        raise ValueError("environment Python bytes changed before mutation execution")
    return path


def _passed_nodes(stdout: str) -> list[str]:
    nodes: list[str] = []
    for line in stdout.splitlines():
        match = re.match(r"^(tests/\S+::\S+)\s+PASSED(?:\s+\[.*\])?$", line.strip())
        if match is not None:
            nodes.append(match.group(1))
    return nodes


def _entry(workspace: Path, path: Path, platform: str, role: str) -> dict[str, Any]:
    return {
        "platform_family": platform,
        "path": path.relative_to(workspace).as_posix(),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
        "media_type": "application/json" if path.suffix == ".json" else "text/plain",
        "role": role,
    }


def _identity(
    args: argparse.Namespace, protocol: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    contract = protocol["parameters"]["raw_results_contract"]
    tests = contract["required_mutation_tests"]
    mutation_ids = contract["required_mutation_ids"]
    if set(tests) != set(mutation_ids):
        raise ValueError("mutation-test mapping differs from frozen mutation IDs")
    expected_ref = "refs/tags/popgp-via000-r3-protocol-" + args.protocol_source_commit
    if args.dispatch_ref != expected_ref:
        raise ValueError("dispatch ref differs from the content-addressed snapshot tag")
    dispatch = {
        "event_name": "workflow_dispatch",
        "source_ref": args.dispatch_ref,
        "protocol_snapshot_commit": args.protocol_source_commit,
        "authorization_ref": args.authorization_ref,
        "authorization_tag_oid": args.authorization_tag_oid,
        "authorization_commit": args.authorization_commit,
        "authorization_record_sha256": args.authorization_record_sha256,
        "producer_run_id": args.producer_run_id,
        "producer_run_attempt": args.producer_run_attempt,
    }
    selectors: list[str] = []
    for mutation_id in mutation_ids:
        for item in tests[mutation_id]:
            if item["test_prefix"] not in selectors:
                selectors.append(item["test_prefix"])
    return dispatch, selectors


def prepare(args: argparse.Namespace) -> None:
    workspace = args.workspace_root.resolve(strict=True)
    evidence = workspace / "evidence"
    environment_root = (workspace / "tool-closure/python-environment").resolve(strict=True)
    environment_python = _regular_explicit_file(
        args.environment_python, environment_root, args.environment_python_sha256
    )
    repo_root = args.repo_root.resolve(strict=True)
    bootstrap = (repo_root / "scripts/run_without_startup_hooks.py").resolve(strict=True)
    if not bootstrap.is_file() or bootstrap.is_symlink():
        raise ValueError("mutation bootstrap is not one regular source file")
    protocol = _load_json(args.protocol.resolve(strict=True))
    dispatch, selectors = _identity(args, protocol)
    if args.plan.parent.resolve() != evidence.resolve():
        raise ValueError("mutation plan must be written directly in trusted evidence")
    _write_json(
        args.plan,
        {
            "schema_version": 1,
            "platform_family": args.platform_family,
            "candidate_commit": protocol["parameters"]["candidate_commit"],
            "candidate_tree": protocol["parameters"]["candidate_tree"],
            "protocol_source_commit": args.protocol_source_commit,
            "dispatch_identity": dispatch,
            "environment_python": str(environment_python),
            "environment_python_sha256": args.environment_python_sha256,
            "working_directory": str(repo_root),
            "arguments": [
                "-I", "-S", "-X",
                f"pycache_prefix={workspace / 'mutable/mutation-python-cache'}",
                str(bootstrap), "--repo-root", str(repo_root),
                "--module", "pytest", "--", "-vv", "-p", "no:cacheprovider",
                *selectors,
            ],
            "selectors": selectors,
        },
    )


def finalize(args: argparse.Namespace) -> None:
    workspace = args.workspace_root.resolve(strict=True)
    evidence = workspace / "evidence"
    plan = _load_json(args.plan.resolve(strict=True))
    contained = _load_json(args.contained_result.resolve(strict=True))
    protocol = _load_json(args.protocol.resolve(strict=True))
    dispatch, selectors = _identity(args, protocol)
    if plan.get("selectors") != selectors or plan.get("dispatch_identity") != dispatch:
        raise ValueError("mutation plan identity differs from the frozen protocol")
    expected_token_flags = (
        ["DISABLE_MAX_PRIVILEGE"] if args.platform_family == "windows-x86_64" else []
    )
    expected_integrity_sid = (
        "S-1-16-4096" if args.platform_family == "windows-x86_64" else ""
    )
    expected_label_policy = (
        "medium-integrity-no-write-up-no-read-up"
        if args.platform_family == "windows-x86_64"
        else "owner-only-protected-root"
    )
    enabled_privileges = contained.get("enabled_privileges")
    if (
        contained.get("exit_code") != 0
        or contained.get("timed_out") is not False
        or contained.get("descendants_quiescent") is not True
        or contained.get("active_processes_after_teardown") != 0
        or contained.get("privilege_separation")
        not in {"low-integrity-restricted-token", "systemd-ephemeral-user"}
        or contained.get("token_restriction_flags") != expected_token_flags
        or contained.get("token_integrity_sid") != expected_integrity_sid
        or enabled_privileges not in ([], ["SeChangeNotifyPrivilege"])
        or contained.get("enabled_privilege_count") != len(enabled_privileges or [])
        or contained.get("protected_label_policy") != expected_label_policy
    ):
        raise ValueError("mutation containment proof is absent or unsuccessful")
    stdout_path = args.stdout.resolve(strict=True)
    stderr_path = args.stderr.resolve(strict=True)
    if contained.get("stdout_sha256") != _sha256(stdout_path):
        raise ValueError("contained mutation stdout differs from trusted capture")
    if contained.get("stderr_sha256") != _sha256(stderr_path):
        raise ValueError("contained mutation stderr differs from trusted capture")

    contract = protocol["parameters"]["raw_results_contract"]
    tests = contract["required_mutation_tests"]
    mutation_ids = contract["required_mutation_ids"]
    oracles = contract["required_mutation_oracles"]
    passed_nodes = _passed_nodes(stdout_path.read_text(encoding="utf-8", errors="replace"))
    if not passed_nodes:
        raise ValueError("frozen mutation suite retained no passed test nodes")

    mutation_root = evidence / "mutations"
    mutation_root.mkdir(parents=True, exist_ok=False)
    retained_stdout = mutation_root / "mutation-suite.stdout.txt"
    retained_stderr = mutation_root / "mutation-suite.stderr.txt"
    retained_result = mutation_root / "mutation-suite.result.json"
    retained_stdout.write_bytes(stdout_path.read_bytes())
    retained_stderr.write_bytes(stderr_path.read_bytes())
    retained_result.write_bytes(args.contained_result.read_bytes())
    stdout_path.unlink()
    stderr_path.unlink()
    args.contained_result.unlink()
    mutation_results: list[dict[str, Any]] = []
    new_entries = [
        _entry(workspace, retained_stdout, args.platform_family, "mutation-suite-stdout"),
        _entry(workspace, retained_stderr, args.platform_family, "mutation-suite-stderr"),
        _entry(workspace, retained_result, args.platform_family, "mutation-suite-result"),
        _entry(workspace, args.plan, args.platform_family, "mutation-plan"),
    ]
    for mutation_id in mutation_ids:
        matched_nodes: list[str] = []
        oracle_errors: list[dict[str, str]] = []
        for requirement in tests[mutation_id]:
            prefix = requirement["test_prefix"]
            matches = [node for node in passed_nodes if node.startswith(prefix)]
            if len(matches) != requirement["expected_passed_count"]:
                raise ValueError(
                    f"{mutation_id}: expected {requirement['expected_passed_count']} passed "
                    f"tests for {prefix!r}, observed {len(matches)}"
                )
            matched_nodes.extend(matches)
            oracle_errors.append(
                {
                    "error_id": oracles[mutation_id],
                    "message": (
                        f"contained workflow executed {len(matches)} frozen rejection "
                        f"test(s) under {prefix}"
                    ),
                }
            )
        receipt_path = mutation_root / f"{mutation_id}.json"
        _write_json(
            receipt_path,
            {
                "schema_version": 1,
                "mutation_id": mutation_id,
                "platform_family": args.platform_family,
                "candidate_commit": protocol["parameters"]["candidate_commit"],
                "candidate_tree": protocol["parameters"]["candidate_tree"],
                "rejected": True,
                "attack": f"frozen {mutation_id} adversarial mutation-test family",
                "oracle_id": oracles[mutation_id],
                "oracle_errors": oracle_errors,
                "execution": {
                    "command": " ".join([plan["environment_python"], *plan["arguments"]]),
                    "exit_code": 0,
                    "test_ids": sorted(matched_nodes),
                    "passed_test_count": len(matched_nodes),
                    "stdout_sha256": _sha256(retained_stdout),
                    "stderr_sha256": _sha256(retained_stderr),
                    "containment_primitive": contained["primitive"],
                    "descendants_quiescent": True,
                },
            },
        )
        relative = receipt_path.relative_to(workspace).as_posix()
        mutation_results.append(
            {"mutation_id": mutation_id, "rejected": True, "evidence_paths": [relative]}
        )
        new_entries.append(_entry(workspace, receipt_path, args.platform_family, "mutation-result"))

    manifest_path = evidence / "evidence-manifest.json"
    summary_path = evidence / "stage-summary.json"
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    if not isinstance(manifest, list) or not isinstance(summary, dict):
        raise ValueError("runner evidence manifest or summary is malformed")
    if summary.get("stage_id") != "mutation" or summary.get("dispatch_identity") != dispatch:
        raise ValueError("runner mutation-stage identity differs from finalization")
    observed = {entry["path"] for entry in manifest}
    if any(entry["path"] in observed for entry in new_entries):
        raise ValueError("mutation evidence collides with runner evidence")
    manifest.extend(new_entries)
    manifest.sort(key=lambda item: item["path"])
    summary["mutation_results"] = mutation_results
    summary["mutation_count"] = len(mutation_results)
    summary["mutations_rejected"] = True
    boundary = summary.get("execution_boundary")
    if not isinstance(boundary, dict) or not isinstance(
        boundary.get("contained_command_count"), int
    ):
        raise ValueError("mutation summary lacks the runner containment proof")
    boundary["contained_command_count"] += 1
    summary["overall_passed"] = all(
        summary[field]
        for field in (
            "commands_passed", "semantic_contract_passed", "visual_contract_passed",
            "source_boundary_passed", "environment_boundary_passed", "pdf_passed",
            "mutations_rejected",
        )
    )
    summary["evidence_paths"] = sorted(entry["path"] for entry in manifest)
    _write_json(manifest_path, manifest)
    _write_json(summary_path, summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("operation", choices=("prepare", "finalize"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=here / "VIA-000.json")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--environment-python", type=Path, required=True)
    parser.add_argument("--environment-python-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stdout", type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--contained-result", type=Path)
    parser.add_argument(
        "--platform-family",
        choices=("ubuntu-latest-x86_64", "windows-x86_64"),
        required=True,
    )
    parser.add_argument("--protocol-source-commit", required=True)
    parser.add_argument("--dispatch-ref", required=True)
    parser.add_argument("--authorization-ref", required=True)
    parser.add_argument("--authorization-tag-oid", required=True)
    parser.add_argument("--authorization-commit", required=True)
    parser.add_argument("--authorization-record-sha256", required=True)
    parser.add_argument("--producer-run-id", required=True)
    parser.add_argument("--producer-run-attempt", type=int, required=True)
    args = parser.parse_args()
    for name in ("protocol_source_commit", "authorization_tag_oid", "authorization_commit"):
        if re.fullmatch(r"[0-9a-f]{40}", getattr(args, name)) is None:
            parser.error(f"--{name.replace('_', '-')} must be one lowercase Git object")
    if re.fullmatch(r"[0-9a-f]{64}", args.authorization_record_sha256) is None:
        parser.error("--authorization-record-sha256 must be a lowercase SHA-256")
    if re.fullmatch(r"[0-9a-f]{64}", args.environment_python_sha256) is None:
        parser.error("--environment-python-sha256 must be a lowercase SHA-256")
    if re.fullmatch(r"[1-9][0-9]*", args.producer_run_id) is None or args.producer_run_attempt < 1:
        parser.error("producer run identity must be positive")
    if args.operation == "prepare":
        prepare(args)
    else:
        if None in (args.stdout, args.stderr, args.contained_result):
            parser.error("finalize requires stdout, stderr, and contained result paths")
        finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
