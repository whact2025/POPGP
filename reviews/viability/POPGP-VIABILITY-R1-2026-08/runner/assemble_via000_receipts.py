#!/usr/bin/env python3
"""Assemble hash-closed VIA-000 runner receipts without custody access."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

CAMPAIGN_ID = "POPGP-VIABILITY-R1-2026-08"
PACKET_ID = "VIA-000"
CANDIDATE_COMMIT = "9a29e05f803666bf0e3a28417ea399e3e26769fc"
CANDIDATE_TREE = "358fb1af6ca587b6c71ff2ef0fb87e335163eeaf"
ATTACKED_HANDOFF = "b7b0d12ddfbbfb1268cdf22db2349a10bf7ed96d"
PROTOCOL_COMMIT = "72bfcfbb5ab5a0fee3449510f8868b8bb19be805"
RUNNER_IDENTITY = "codex-via000-runner"
RUNNER_SESSION = "popgp-viability-r1-2026-08-via000-runner-session"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_by_id(platform: dict[str, Any], command_id: str) -> dict[str, Any]:
    matches = [
        command for command in platform["protocol"]["commands"] if command["id"] == command_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {command_id!r} command, found {len(matches)}")
    return matches[0]


def command_output(evidence_dir: Path, command: dict[str, Any], stream: str) -> str:
    return (evidence_dir / command[f"{stream}_path"]).read_text(encoding="utf-8", errors="replace")


def receipt_hash(path: Path, receipt_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(receipt_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def validate_identity(document: dict[str, Any], platform: str) -> None:
    expected = {
        "campaign_id": CAMPAIGN_ID,
        "packet_id": PACKET_ID,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "runner_identity": RUNNER_IDENTITY,
        "runner_session_id": RUNNER_SESSION,
        "platform_family": platform,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise ValueError(f"{platform}: {field} mismatch: {document.get(field)!r} != {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt_dir", type=Path)
    args = parser.parse_args()
    receipt_dir = args.receipt_dir.resolve()
    runner_dir = receipt_dir / "runner"

    windows_dir = runner_dir / "windows"
    windows_mutation_dir = runner_dir / "windows-mutation-continuation"
    linux_dir = runner_dir / "linux-attempt-1"
    linux_mutation_dir = runner_dir / "linux-mutation-continuation"
    github_runs_path = runner_dir / "github-runs.json"
    required_paths = (
        windows_dir / "platform-results.json",
        windows_mutation_dir / "platform-results.json",
        linux_dir / "platform-results.json",
        linux_mutation_dir / "platform-results.json",
        github_runs_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing runner evidence: {missing}")

    windows = read_json(windows_dir / "platform-results.json")
    windows_mutations = read_json(windows_mutation_dir / "platform-results.json")
    linux = read_json(linux_dir / "platform-results.json")
    linux_mutations = read_json(linux_mutation_dir / "platform-results.json")
    validate_identity(windows, "windows-x86_64")
    validate_identity(windows_mutations, "windows-x86_64")
    validate_identity(linux, "ubuntu-latest-x86_64")
    validate_identity(linux_mutations, "ubuntu-latest-x86_64")

    windows_git_diff = command_by_id(windows, "git-diff")
    windows_postflight = command_by_id(windows, "postflight")
    linux_git_diff = command_by_id(linux, "git-diff")
    linux_postflight = command_by_id(linux, "postflight")
    linux_diff_text = command_output(linux_dir, linux_git_diff, "stdout")
    dirty_paths = sorted(
        set(re.findall(r"^diff --git a/(.+?) b/", linux_diff_text, flags=re.MULTILINE))
    )
    linux_postflight_text = command_output(linux_dir, linux_postflight, "stdout")
    linux_startup_surfaces = sorted(
        line.strip() for line in linux_postflight_text.splitlines() if line.strip().endswith(".pth")
    )

    windows_main_success = (
        windows["engine_exact"] is True
        and windows["protocol"]["all_commands_succeeded"] is True
        and windows["protocol"]["test_count"] == 187
        and windows["protocol"]["required_example_count_met"] is True
        and windows["pdf"]["framework_pdf_nonempty"] is True
        and windows_git_diff["exit_code"] == 0
        and windows_postflight["exit_code"] == 0
    )
    linux_main_success = (
        linux["engine_exact"] is True
        and linux["protocol"]["all_commands_succeeded"] is True
        and linux["protocol"]["test_count"] == 187
        and linux["protocol"]["required_example_count_met"] is True
        and linux["pdf"]["framework_pdf_nonempty"] is True
        and linux_git_diff["exit_code"] == 0
        and linux_postflight["exit_code"] == 0
    )
    windows_mutation_success = (
        windows_mutations["required_mutation_count_met"] is True
        and windows_mutations["mutations_rejected"] is True
        and windows_mutations["harness_error"] is None
    )
    linux_mutation_success = (
        linux_mutations["required_mutation_count_met"] is True
        and linux_mutations["mutations_rejected"] is True
        and linux_mutations["harness_error"] is None
    )
    evidence_contract = windows_main_success and linux_main_success
    cross_platform = windows_main_success and linux_main_success
    mutation_rejection = windows_mutation_success and linux_mutation_success
    failed = not (evidence_contract and cross_platform and mutation_rejection)

    environment_document = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "packet_id": PACKET_ID,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "attacked_handoff_commit": ATTACKED_HANDOFF,
        "protocol_commit": PROTOCOL_COMMIT,
        "runner": {
            "agent_identity": RUNNER_IDENTITY,
            "model_identity": "unknown",
            "model_version": "unknown",
            "operator": "fuocor",
            "session_id": RUNNER_SESSION,
            "orchestrator_id": "codex-desktop",
            "organization": "popgp-internal",
            "access_level": "public-calibration-only",
        },
        "hidden_access_declaration": {
            "final_labels_seen": False,
            "secret_seed_seen": False,
            "private_evaluator_seen": False,
            "custody_path_accessed": False,
            "sealed_manifest_contents_seen": False,
            "local_handoff_file_seen": False,
        },
        "incidental_public_calibration_exposure": {
            "seen": True,
            "scope": "brief rg search-result snippets after the public attack plan was committed",
            "used_as_execution_evidence": False,
        },
        "platform_environment_receipts": {
            "windows_full": receipt_hash(windows_dir / "environment.json", receipt_dir),
            "windows_mutations": receipt_hash(
                windows_mutation_dir / "environment.json", receipt_dir
            ),
            "linux_full": receipt_hash(linux_dir / "environment.json", receipt_dir),
            "linux_mutations": receipt_hash(linux_mutation_dir / "environment.json", receipt_dir),
        },
        "github_runs": receipt_hash(github_runs_path, receipt_dir),
    }
    environment_path = receipt_dir / "environment.json"
    write_json(environment_path, environment_document)

    attempts = []
    for name, evidence_dir in (
        ("windows-full", windows_dir),
        ("windows-mutation-continuation", windows_mutation_dir),
        ("linux-full", linux_dir),
        ("linux-mutation-continuation", linux_mutation_dir),
    ):
        index_path = evidence_dir / "command-index.json"
        commands = read_json(index_path)
        attempts.append(
            {
                "id": name,
                "command_count": len(commands),
                "command_index": receipt_hash(index_path, receipt_dir),
                "platform_results": receipt_hash(
                    evidence_dir / "platform-results.json", receipt_dir
                ),
            }
        )
    run_log_document = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "packet_id": PACKET_ID,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "attempts": attempts,
        "primary_protocol_attempts": {
            "windows": "windows-full",
            "linux": "linux-full",
        },
        "primary_mutation_attempts": {
            "windows": "windows-mutation-continuation",
            "linux": "linux-mutation-continuation",
        },
        "windows_protocol_exit_codes": {
            command["id"]: command["exit_code"] for command in windows["protocol"]["commands"]
        },
        "linux_protocol_exit_codes": {
            command["id"]: command["exit_code"] for command in linux["protocol"]["commands"]
        },
    }
    run_log_path = receipt_dir / "run-log.json"
    write_json(run_log_path, run_log_document)

    mutation_document = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "packet_id": PACKET_ID,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "required_mutation_count": 10,
        "windows": {
            "evidence_path": (windows_mutation_dir / "mutation-results.json")
            .relative_to(receipt_dir)
            .as_posix(),
            "results": windows_mutations["mutations"],
            "all_rejected": windows_mutation_success,
        },
        "linux": {
            "evidence_path": (linux_mutation_dir / "mutation-results.json")
            .relative_to(receipt_dir)
            .as_posix(),
            "clean_baseline_postflight_valid": linux_mutations["protocol"]["mutation_baseline"][
                "frozen_postflight_valid"
            ],
            "results": linux_mutations["mutations"],
            "all_rejected": linux_mutation_success,
        },
        "mutation_rejection": mutation_rejection,
        "failure_detail": (
            "Linux G10 removed the editable carrier and isolated Python printed None, "
            "but the frozen final postflight rejected the lock-installed "
            "_cuda_bindings_redirector.pth startup surface."
        ),
    }
    mutation_path = receipt_dir / "mutation-results.json"
    write_json(mutation_path, mutation_document)

    evidence_manifest = [
        receipt_hash(path, receipt_dir) for path in sorted(runner_dir.rglob("*")) if path.is_file()
    ]
    manifest_bytes = json.dumps(
        evidence_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    raw_document = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "packet_id": PACKET_ID,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "attacked_handoff_commit": ATTACKED_HANDOFF,
        "protocol_commit": PROTOCOL_COMMIT,
        "runner_identity": RUNNER_IDENTITY,
        "runner_session_id": RUNNER_SESSION,
        "capabilities": {
            "evidence-contract": evidence_contract,
            "cross-platform-reproduction": cross_platform,
            "mutation-rejection": mutation_rejection,
        },
        "failed": failed,
        "blocked": False,
        "failure_classification": {
            "cause_class": "tested-capability-failure",
            "cause_codes": [
                "linux-dirty-regeneration",
                "linux-unexpected-startup-surface",
                "linux-mutation-boundary-failure",
            ],
            "infrastructure_available": True,
            "post_output_threshold_selection": False,
        },
        "platforms": {
            "windows-x86_64": {
                "main_protocol_succeeded": windows_main_success,
                "test_count": windows["protocol"]["test_count"],
                "example_count": 6,
                "engine_exact": windows["engine_exact"],
                "pdf_nonempty": windows["pdf"]["framework_pdf_nonempty"],
                "git_diff_exit_code": windows_git_diff["exit_code"],
                "postflight_exit_code": windows_postflight["exit_code"],
                "mutation_count": len(windows_mutations["mutations"]),
                "mutation_rejection": windows_mutation_success,
            },
            "ubuntu-latest-x86_64": {
                "main_protocol_succeeded": linux_main_success,
                "test_count": linux["protocol"]["test_count"],
                "example_count": 6,
                "engine_exact": linux["engine_exact"],
                "pdf_nonempty": linux["pdf"]["framework_pdf_nonempty"],
                "git_diff_exit_code": linux_git_diff["exit_code"],
                "postflight_exit_code": linux_postflight["exit_code"],
                "dirty_regeneration_path_count": len(dirty_paths),
                "dirty_regeneration_paths": dirty_paths,
                "observed_startup_surfaces": linux_startup_surfaces,
                "mutation_count": len(linux_mutations["mutations"]),
                "mutation_rejection": linux_mutation_success,
            },
        },
        "aggregate_receipts": {
            "environment": receipt_hash(environment_path, receipt_dir),
            "run-log": receipt_hash(run_log_path, receipt_dir),
            "mutation-results": receipt_hash(mutation_path, receipt_dir),
        },
        "runner_evidence_manifest": evidence_manifest,
        "runner_evidence_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "generated_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
    }
    raw_path = receipt_dir / "raw-results.json"
    write_json(raw_path, raw_document)

    committed_at = dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")
    commitment_document = {
        "packet_id": PACKET_ID,
        "committed_by": RUNNER_IDENTITY,
        "committed_at": committed_at,
        "output_receipt_id": "raw-results",
        "output_sha256": sha256(raw_path),
    }
    write_json(receipt_dir / "output-commitment.json", commitment_document)
    print(json.dumps(commitment_document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
