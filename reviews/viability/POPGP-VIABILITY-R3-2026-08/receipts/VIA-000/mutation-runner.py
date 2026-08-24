"""Execute and retain the frozen VIA-000 R3 mutation-test matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import UTC, datetime
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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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
    relative = path.relative_to(workspace).as_posix()
    return {
        "platform_family": platform,
        "path": relative,
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
        "media_type": "application/json" if path.suffix == ".json" else "text/plain",
        "role": role,
    }


def run(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    workspace = args.workspace_root.resolve()
    evidence = workspace / "evidence"
    if not workspace.is_dir() or workspace.is_symlink() or not evidence.is_dir():
        raise ValueError("runner workspace and evidence must already exist")
    environment = workspace / "python-environment"
    environment_python = _regular_explicit_file(
        args.environment_python, environment, args.environment_python_sha256
    )
    bootstrap = (repo_root / "scripts/run_without_startup_hooks.py").resolve(strict=True)
    if not bootstrap.is_file() or bootstrap.is_symlink():
        raise ValueError("mutation bootstrap is not one regular source file")
    protocol = _load_json(args.protocol.resolve())
    contract = protocol["parameters"]["raw_results_contract"]
    test_contracts = contract["required_mutation_tests"]
    mutation_ids = contract["required_mutation_ids"]
    mutation_oracles = contract["required_mutation_oracles"]
    expected_dispatch = {
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
    expected_ref = "refs/tags/popgp-via000-r3-protocol-" + args.protocol_source_commit
    if args.dispatch_ref != expected_ref:
        raise ValueError("dispatch ref differs from the content-addressed snapshot tag")
    if set(test_contracts) != set(mutation_ids):
        raise ValueError("mutation-test mapping differs from frozen mutation IDs")

    selectors: list[str] = []
    for mutation_id in mutation_ids:
        for item in test_contracts[mutation_id]:
            selector = item["test_prefix"]
            if selector not in selectors:
                selectors.append(selector)

    mutation_root = evidence / "mutations"
    mutation_root.mkdir(parents=True, exist_ok=False)
    stdout_path = mutation_root / "mutation-suite.stdout.txt"
    stderr_path = mutation_root / "mutation-suite.stderr.txt"
    result_path = mutation_root / "mutation-suite.result.json"
    python_cache = workspace / "mutation-python-cache"
    python_cache.mkdir(parents=True, exist_ok=False)
    command = [
        str(environment_python),
        "-I",
        "-S",
        "-X",
        f"pycache_prefix={python_cache}",
        str(bootstrap),
        "--repo-root",
        str(repo_root),
        "--module",
        "pytest",
        "--",
        "-vv",
        "-p",
        "no:cacheprovider",
        *selectors,
    ]
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("GIT_") or name in {
            "PATH",
            "PATHEXT",
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "UV_PROJECT_ENVIRONMENT",
            "GITHUB_ENV",
            "BASH_ENV",
            "ENV",
        }:
            environment.pop(name, None)
    environment.update(
        {
            "UV_CACHE_DIR": str(workspace / "mutation-uv-cache"),
            "PYTHONPYCACHEPREFIX": str(python_cache),
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUFF_CACHE_DIR": str(workspace / "mutation-ruff-cache"),
            "MPLCONFIGDIR": str(workspace / "mutation-matplotlib-cache"),
            "XDG_CACHE_HOME": str(workspace / "mutation-general-cache"),
        }
    )
    started_at = _utc_now()
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    finished_at = _utc_now()
    stdout_path.write_text(completed.stdout, encoding="utf-8", newline="\n")
    stderr_path.write_text(completed.stderr, encoding="utf-8", newline="\n")
    _write_json(
        result_path,
        {
            "schema_version": 1,
            "candidate_commit": protocol["parameters"]["candidate_commit"],
            "candidate_tree": protocol["parameters"]["candidate_tree"],
            "platform_family": args.platform_family,
            "protocol_source_commit": args.protocol_source_commit,
            "dispatch_identity": expected_dispatch,
            "command": command,
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": completed.returncode,
            "stdout_sha256": _sha256(stdout_path),
            "stderr_sha256": _sha256(stderr_path),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(f"frozen mutation suite failed with exit code {completed.returncode}")

    passed_nodes = _passed_nodes(completed.stdout)
    if not passed_nodes:
        raise ValueError("frozen mutation suite retained no passed test nodes")
    mutation_results: list[dict[str, Any]] = []
    new_entries = [
        _entry(workspace, stdout_path, args.platform_family, "mutation-suite-stdout"),
        _entry(workspace, stderr_path, args.platform_family, "mutation-suite-stderr"),
        _entry(workspace, result_path, args.platform_family, "mutation-suite-result"),
    ]
    for mutation_id in mutation_ids:
        requirements = test_contracts[mutation_id]
        matched_nodes: list[str] = []
        oracle_errors: list[dict[str, str]] = []
        for requirement in requirements:
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
                    "error_id": mutation_oracles[mutation_id],
                    "message": (
                        f"trusted workflow executed {len(matches)} frozen rejection test(s) "
                        f"under {prefix}"
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
                "oracle_id": mutation_oracles[mutation_id],
                "oracle_errors": oracle_errors,
                "execution": {
                    "command": " ".join(command),
                    "exit_code": completed.returncode,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "test_ids": sorted(matched_nodes),
                    "passed_test_count": len(matched_nodes),
                    "stdout_sha256": _sha256(stdout_path),
                    "stderr_sha256": _sha256(stderr_path),
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
    if summary.get("stage_id") != "mutation":
        raise ValueError("mutation runner requires a fresh mutation-stage workspace")
    if summary.get("dispatch_identity") != expected_dispatch:
        raise ValueError("runner summary dispatch identity differs from mutation execution")
    observed = {entry["path"] for entry in manifest}
    if any(entry["path"] in observed for entry in new_entries):
        raise ValueError("mutation evidence collides with runner evidence")
    manifest.extend(new_entries)
    manifest.sort(key=lambda item: item["path"])
    summary["mutation_results"] = mutation_results
    summary["mutation_count"] = len(mutation_results)
    summary["mutations_rejected"] = True
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
    summary["evidence_paths"] = sorted(entry["path"] for entry in manifest)
    _write_json(manifest_path, manifest)
    _write_json(summary_path, summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--repo-root", type=Path, default=here.parents[1])
    parser.add_argument("--protocol", type=Path, default=here / "VIA-000.json")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--environment-python", type=Path, required=True)
    parser.add_argument("--environment-python-sha256", required=True)
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
    if re.fullmatch(r"[0-9a-f]{40}", args.protocol_source_commit) is None:
        parser.error("--protocol-source-commit must be a full lowercase Git commit")
    if re.fullmatch(r"[0-9a-f]{40}", args.authorization_tag_oid) is None:
        parser.error("--authorization-tag-oid must be a full lowercase Git object")
    if re.fullmatch(r"[0-9a-f]{40}", args.authorization_commit) is None:
        parser.error("--authorization-commit must be a full lowercase Git commit")
    if re.fullmatch(r"[0-9a-f]{64}", args.authorization_record_sha256) is None:
        parser.error("--authorization-record-sha256 must be a lowercase SHA-256")
    if re.fullmatch(r"[1-9][0-9]*", args.producer_run_id) is None:
        parser.error("--producer-run-id must be a positive decimal identifier")
    if args.producer_run_attempt < 1:
        parser.error("--producer-run-attempt must be positive")
    if re.fullmatch(r"[0-9a-f]{64}", args.environment_python_sha256) is None:
        parser.error("--environment-python-sha256 must be a lowercase SHA-256")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
