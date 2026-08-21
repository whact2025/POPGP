"""Execute and retain the frozen VIA-000 R2 mutation-test matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    protocol = _load_json(args.protocol.resolve())
    contract = protocol["parameters"]["raw_results_contract"]
    test_contracts = contract["required_mutation_tests"]
    mutation_ids = contract["required_mutation_ids"]
    mutation_oracles = contract["required_mutation_oracles"]
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
    command = [
        "uv",
        "run",
        "--isolated",
        "--frozen",
        "--no-editable",
        "python",
        "-m",
        "pytest",
        "-vv",
        "-p",
        "no:cacheprovider",
        *selectors,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "UV_CACHE_DIR": str(workspace / "mutation-uv-cache"),
            "PYTHONPYCACHEPREFIX": str(workspace / "mutation-python-cache"),
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
    summary_path = evidence / "platform-summary.json"
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    if not isinstance(manifest, list) or not isinstance(summary, dict):
        raise ValueError("runner evidence manifest or summary is malformed")
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
    parser.add_argument(
        "--platform-family",
        choices=("ubuntu-latest-x86_64", "windows-x86_64"),
        required=True,
    )
    parser.add_argument("--protocol-source-commit", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.protocol_source_commit) is None:
        parser.error("--protocol-source-commit must be a full lowercase Git commit")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
