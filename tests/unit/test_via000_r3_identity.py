from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jsonschema import Draft202012Validator

from tests.unit import test_via000_r2_assembler as r2_assembler

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = ROOT / "protocols/POPGP-VIABILITY-R3-2026-08"
PROTOCOL = PROTOCOL_DIR / "VIA-000.json"
SCHEMA = PROTOCOL_DIR / "VIA-000-RAW-RESULTS.schema.json"
ASSEMBLER = PROTOCOL_DIR / "VIA-000-ASSEMBLER.py"
GUARD = PROTOCOL_DIR / "VIA-000-DISPATCH-GUARD.py"
RUNNER = PROTOCOL_DIR / "VIA-000-RUNNER.ps1"
CONTAINMENT = PROTOCOL_DIR / "VIA-000-CONTAINMENT.ps1"
CONTAINMENT_PROOF_RUNNER = PROTOCOL_DIR / "VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
CONTAINMENT_PROOF_FIXTURE = PROTOCOL_DIR / "VIA-000-CONTAINMENT-HOSTILE.ps1"
CONTAINMENT_PROOF_SCHEMA = PROTOCOL_DIR / "VIA-000-CONTAINMENT-PROOF.schema.json"
CONTAINMENT_PROOF_ENVELOPE_SCHEMA = (
    PROTOCOL_DIR / "VIA-000-CONTAINMENT-PROOF-ENVELOPE.schema.json"
)
CONTAINMENT_PROOF_AGGREGATOR = PROTOCOL_DIR / "VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
MUTATION_RUNNER = PROTOCOL_DIR / "VIA-000-MUTATION-RUNNER.py"
R3_POWERSHELL_RELATIVE_PATHS = (
    "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-HOSTILE.ps1",
    "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1",
    "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1",
    "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1",
    "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-hostile.ps1",
    "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1",
    "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-protocol.ps1",
    "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1",
)
CAMPAIGN_CHECKER = ROOT / "scripts/check_viability_campaign.py"
WORKFLOW = ROOT / ".github/workflows/via000-r3-protocol.yml"
CONTAINMENT_PROOF_WORKFLOW = ROOT / ".github/workflows/via000-r3-containment-proof.yml"
FAKE_COMMIT = "a" * 40
FAKE_REF = f"refs/tags/popgp-via000-r3-protocol-{FAKE_COMMIT}"
FAKE_AUTHORIZATION_SHA256 = "b" * 64
FAKE_AUTHORIZATION_REF = "refs/tags/popgp-via000-r3-authorization-" + FAKE_AUTHORIZATION_SHA256
FAKE_AUTHORIZATION = {
    "protocol_commit": FAKE_COMMIT,
    "source_ref": FAKE_REF,
    "authorization_ref": FAKE_AUTHORIZATION_REF,
    "authorization_tag_oid": "c" * 40,
    "authorization_commit": "d" * 40,
    "authorization_record_sha256": FAKE_AUTHORIZATION_SHA256,
    "campaign_sha256": "e" * 64,
    "packet_sha256": "f" * 64,
    "protocol_manifest_sha256": "1" * 64,
}
DISPATCH_IDENTITY = {
    "event_name": "workflow_dispatch",
    "source_ref": FAKE_REF,
    "protocol_snapshot_commit": FAKE_COMMIT,
    "authorization_ref": FAKE_AUTHORIZATION_REF,
    "authorization_tag_oid": "c" * 40,
    "authorization_commit": "d" * 40,
    "authorization_record_sha256": FAKE_AUTHORIZATION_SHA256,
    "producer_run_id": "424242",
    "producer_run_attempt": 1,
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    ).stdout


def _git_bytes_input(repo: Path, content: bytes, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=content,
        check=True,
        capture_output=True,
    ).stdout


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_yaml(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _powershell_function_text(source: str, name: str) -> str:
    start = source.index(f"function {name} {{")
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"PowerShell function {name} is unterminated")


def _token_evidence(platform: str) -> dict[str, object]:
    if platform == "windows-x86_64":
        return {
            "token_restriction_flags": ["DISABLE_MAX_PRIVILEGE"],
            "token_integrity_sid": "S-1-16-4096",
            "enabled_privilege_count": 0,
            "enabled_privileges": [],
            "protected_label_policy": "medium-integrity-no-write-up-no-read-up",
        }
    return {
        "token_restriction_flags": [],
        "token_integrity_sid": "",
        "enabled_privilege_count": 0,
        "enabled_privileges": [],
        "protected_label_policy": "owner-only-protected-root",
    }


def _export_evidence(platform: str) -> dict[str, object]:
    if platform == "windows-x86_64":
        return {
            "export_owner_sid": "S-1-5-21-1-2-3-1001",
            "export_dacl_policy": "protected-current-runner-full-control-v1",
            "export_integrity_sid": "S-1-16-8192",
            "export_mandatory_policy": "NO_WRITE_UP",
            "export_root_control_flags": 37892,
            "export_root_dacl_protected": True,
            "export_root_native_ace_count": 1,
            "export_root_managed_ace_count": 1,
            "export_created_after_teardown": True,
        }
    return {
        "export_owner_sid": "",
        "export_dacl_policy": "owner-rwx-0700-v1",
        "export_integrity_sid": "",
        "export_mandatory_policy": "owner-only",
        "export_root_control_flags": 0,
        "export_root_dacl_protected": False,
        "export_root_native_ace_count": 0,
        "export_root_managed_ace_count": 0,
        "export_created_after_teardown": True,
    }


def _overlay(source: Path, destination: Path) -> None:
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.rglob("*"):
            if item.is_file():
                relative = item.relative_to(source)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(item.read_bytes().replace(b"\r\n", b"\n"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))


def _authorized_repo(tmp_path: Path) -> tuple[Path, str, str, str, str]:
    repo = tmp_path / "authorized-repository"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            "--no-checkout",
            str(ROOT),
            str(repo),
        ],
        check=True,
    )
    _git(repo, "config", "core.longpaths", "true")
    _git(repo, "sparse-checkout", "init", "--no-cone")
    _git(
        repo,
        "sparse-checkout",
        "set",
        "/.gitattributes",
        "/.github/workflows/via000-r3-protocol.yml",
        "/.github/workflows/via000-r3-containment-proof.yml",
        "/protocols/POPGP-VIABILITY-R3-2026-08/**",
        "/reviews/viability/POPGP-VIABILITY-R3-2026-08/**",
        "/docs/scientific_hardening/**",
        "/schemas/viability/**",
        "/scripts/check_viability_campaign.py",
        "/scripts/check_validation_artifacts.py",
        "/scripts/check_reproduction_boundary.py",
        "/popgp/diagnostics.py",
        "/tests/unit/test_via000_r3_identity.py",
    )
    _git(repo, "checkout", "HEAD")
    _git(repo, "config", "user.name", "R3 authorization test")
    _git(repo, "config", "user.email", "r3-authorization@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    for relative in (
        Path(".gitattributes"),
        Path(".github/workflows/via000-r3-protocol.yml"),
        Path(".github/workflows/via000-r3-containment-proof.yml"),
        Path("protocols/POPGP-VIABILITY-R3-2026-08"),
        Path("reviews/viability/POPGP-VIABILITY-R3-2026-08"),
        Path("scripts/check_viability_campaign.py"),
        Path("scripts/check_validation_artifacts.py"),
        Path("scripts/check_reproduction_boundary.py"),
        Path("popgp/diagnostics.py"),
        Path("tests/unit/test_via000_r3_identity.py"),
    ):
        _overlay(ROOT / relative, repo / relative)

    key = tmp_path / "authorization-key"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        check=True,
    )
    public_key = key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    signers_path = (
        repo / "reviews/viability/POPGP-VIABILITY-R3-2026-08/authorization/"
        "VIA-000-AUTHORIZED-SIGNERS"
    )
    signers_path.write_text(
        f"popgp-via000-r3-authorizer {public_key}\n",
        encoding="utf-8",
        newline="\n",
    )

    primary_path = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    parameters = primary["parameters"]
    for path_field, hash_field in (
        ("runner_protocol_path", "runner_protocol_sha256"),
        ("containment_protocol_path", "containment_protocol_sha256"),
        ("containment_proof_runner_path", "containment_proof_runner_sha256"),
        ("containment_proof_fixture_path", "containment_proof_fixture_sha256"),
        ("containment_proof_schema_path", "containment_proof_schema_sha256"),
        (
            "containment_proof_envelope_schema_path",
            "containment_proof_envelope_schema_sha256",
        ),
        ("containment_proof_aggregator_path", "containment_proof_aggregator_sha256"),
        ("containment_proof_workflow_path", "containment_proof_workflow_sha256"),
        ("raw_results_schema_path", "raw_results_schema_sha256"),
        ("assembler_protocol_path", "assembler_protocol_sha256"),
        ("mutation_runner_protocol_path", "mutation_runner_protocol_sha256"),
        ("dispatch_guard_protocol_path", "dispatch_guard_protocol_sha256"),
        ("workflow_protocol_path", "workflow_protocol_sha256"),
        ("validator_package_init_path", "validator_package_init_sha256"),
    ):
        parameters[hash_field] = _sha(repo / parameters[path_field])
    authorization_contract = parameters["authorization_contract"]
    authorization_contract["allowed_signers_sha256"] = _sha(
        repo / authorization_contract["allowed_signers_path"]
    )
    for item in authorization_contract["validator_bundle"]:
        item["sha256"] = _sha(repo / item["source_path"])
    _write_json(primary_path, primary)

    packet_path = repo / authorization_contract["packet_path"]
    packet = yaml.safe_load(packet_path.read_text(encoding="utf-8"))
    for field in (
        "parameters",
        "measurement_procedure",
        "uncertainty_procedure",
        "statistical_analysis",
        "resource_budget",
        "commands",
        "mutation_plan",
    ):
        packet["preregistration"][field] = primary[field]
    _write_yaml(packet_path, packet)

    manifest_path = repo / authorization_contract["protocol_manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_contracts = {
        ".gitattributes",
        "popgp/diagnostics.py",
        "scripts/check_reproduction_boundary.py",
        "scripts/check_validation_artifacts.py",
        "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-VALIDATOR-PACKAGE-INIT.py",
        authorization_contract["allowed_signers_path"],
    }
    by_path = {item["path"]: item for item in manifest["contract_files"]}
    for relative in required_contracts:
        by_path.setdefault(relative, {"path": relative, "sha256": ""})
    for item in by_path.values():
        item["sha256"] = _sha(repo / item["path"])
    manifest["contract_files"] = sorted(by_path.values(), key=lambda item: item["path"])
    _write_json(manifest_path, manifest)

    campaign_path = repo / authorization_contract["campaign_path"]
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign["protocol_manifest_sha256"] = _sha(manifest_path)
    _write_yaml(campaign_path, campaign)

    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "authorized protocol snapshot")
    snapshot = _git(repo, "rev-parse", "HEAD")
    protocol_ref = f"refs/tags/popgp-via000-r3-protocol-{snapshot}"
    _git(repo, "tag", protocol_ref.removeprefix("refs/tags/"), snapshot)

    packet = yaml.safe_load(packet_path.read_text(encoding="utf-8"))
    packet["protocol_commit"] = snapshot
    packet["lifecycle_phase"] = "preregistered"
    _write_yaml(packet_path, packet)
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign["protocol_commit"] = snapshot
    _write_yaml(campaign_path, campaign)
    _git(repo, "add", str(packet_path), str(campaign_path))
    _git(repo, "commit", "-m", "bind authorized campaign packet")
    authorization_commit = _git(repo, "rev-parse", "HEAD")

    def frozen_sha(relative: str) -> str:
        return hashlib.sha256(
            _git_bytes(repo, "cat-file", "blob", f"{authorization_commit}:{relative}")
        ).hexdigest()

    record = {
        "schema_version": 1,
        "campaign_id": "POPGP-VIABILITY-R3-2026-08",
        "packet_id": "VIA-000",
        "authorization_commit": authorization_commit,
        "protocol_commit": snapshot,
        "campaign_path": authorization_contract["campaign_path"],
        "campaign_sha256": frozen_sha(authorization_contract["campaign_path"]),
        "packet_path": authorization_contract["packet_path"],
        "packet_sha256": frozen_sha(authorization_contract["packet_path"]),
        "protocol_manifest_path": authorization_contract["protocol_manifest_path"],
        "protocol_manifest_sha256": frozen_sha(authorization_contract["protocol_manifest_path"]),
    }
    record_bytes = (
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    record_path = tmp_path / "authorization-record.json"
    record_path.write_bytes(record_bytes)
    authorization_ref = (
        "refs/tags/popgp-via000-r3-authorization-" + hashlib.sha256(record_bytes).hexdigest()
    )
    _git(repo, "config", "gpg.format", "ssh")
    _git(repo, "config", "user.signingkey", str(key))
    _git(
        repo,
        "tag",
        "-s",
        "-F",
        str(record_path),
        authorization_ref.removeprefix("refs/tags/"),
        authorization_commit,
    )
    _git(repo, "checkout", "--detach", snapshot)
    return repo, snapshot, protocol_ref, authorization_ref, authorization_commit


def _authorization_race_objects(
    repo: Path, authorization_ref: str, authorization_commit: str
) -> tuple[str, str, str]:
    """Return original-valid, invalid-record, and unrelated-valid tag object IDs."""

    original_oid = _git(repo, "rev-parse", authorization_ref)
    original = _git_bytes(repo, "cat-file", "tag", original_oid)
    marker = b"-----BEGIN SSH SIGNATURE-----\n"
    prefix, signature = original.split(marker, 1)
    signature_lines = signature.splitlines(keepends=True)
    for index, line in enumerate(signature_lines):
        if line.strip() and not line.startswith(b"-----END"):
            replacement = b"A" if line[:1] != b"A" else b"B"
            signature_lines[index] = replacement + line[1:]
            break
    invalid = prefix + marker + b"".join(signature_lines)
    invalid_oid = (
        _git_bytes_input(repo, invalid, "hash-object", "-w", "-t", "tag", "--stdin")
        .decode("ascii")
        .strip()
    )

    tag_name = authorization_ref.removeprefix("refs/tags/")
    _git(
        repo,
        "tag",
        "-f",
        "-s",
        "-m",
        "signed unrelated message",
        tag_name,
        authorization_commit,
    )
    unrelated_valid_oid = _git(repo, "rev-parse", authorization_ref)
    assert unrelated_valid_oid not in {original_oid, invalid_oid}
    _git(repo, "update-ref", authorization_ref, invalid_oid, unrelated_valid_oid)
    return original_oid, invalid_oid, unrelated_valid_oid


def _replace_object(repo: Path, original_oid: str, replacement_oid: str) -> None:
    _git(repo, "update-ref", f"refs/replace/{original_oid}", replacement_oid)


def _hash_blob(repo: Path, content: bytes) -> str:
    return _git_bytes_input(repo, content, "hash-object", "-w", "--stdin").decode("ascii").strip()


def _workflow_step_script(step_name: str) -> str:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = document["jobs"]["platform-fragment"]["steps"]
    matches = [step["run"] for step in steps if step.get("name") == step_name]
    assert len(matches) == 1
    return matches[0]


def _path_shim(directory: Path, name: str, marker: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        path = directory / f"{name}.cmd"
        path.write_text(
            f'@echo off\r\necho invoked>>"{marker}"\r\nexit /b 99\r\n',
            encoding="utf-8",
        )
    else:
        path = directory / name
        path.write_text(
            f"#!/bin/sh\nprintf invoked >> '{marker}'\nexit 99\n",
            encoding="utf-8",
        )
        path.chmod(0o755)


def _snapshot_repo(tmp_path: Path, *, protocol_tree: bool = False) -> tuple[Path, str, str]:
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "R3 identity test")
    _git(repo, "config", "user.email", "r3-identity@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    if protocol_tree:
        target = repo / "protocols/POPGP-VIABILITY-R3-2026-08"
        target.parent.mkdir(parents=True)
        shutil.copytree(PROTOCOL_DIR, target)
        workflow = repo / ".github/workflows/via000-r3-protocol.yml"
        workflow.parent.mkdir(parents=True)
        shutil.copyfile(WORKFLOW, workflow)
    else:
        (repo / "snapshot.txt").write_text("snapshot\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "snapshot")
    commit = _git(repo, "rev-parse", "HEAD")
    ref = f"refs/tags/popgp-via000-r3-protocol-{commit}"
    _git(repo, "tag", ref.removeprefix("refs/tags/"), commit)
    return repo, commit, ref


def _r3_platform_roots(tmp_path: Path) -> dict[str, Path]:
    roots, _packet, _receipts = r2_assembler._assembler_inputs(tmp_path)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    contract = protocol["parameters"]["raw_results_contract"]
    signer = contract["producer_attestation"]
    selectors: list[str] = []
    for mutation_id in contract["required_mutation_ids"]:
        for requirement in contract["required_mutation_tests"][mutation_id]:
            if requirement["test_prefix"] not in selectors:
                selectors.append(requirement["test_prefix"])
    for platform_name, workspace in roots.items():
        manifest_path = workspace / "evidence/evidence-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if platform_name == "windows-x86_64":
            tools = {
                "git": {
                    "path": "C:/Program Files/Git/cmd/git.exe",
                    "sha256": "1" * 64,
                    "version": "git version 2.51.0.windows.1",
                },
                "ssh_keygen": {
                    "path": "C:/Windows/System32/OpenSSH/ssh-keygen.exe",
                    "sha256": "2" * 64,
                    "version": "OpenSSH system ssh-keygen",
                },
                "base_python": {
                    "path": "C:/hostedtoolcache/windows/Python/3.11.15/x64/python.exe",
                    "sha256": "3" * 64,
                    "version": "3.11.15",
                },
                "uv": {
                    "path": "C:/hostedtoolcache/windows/Python/3.11.15/x64/Scripts/uv.exe",
                    "sha256": "4" * 64,
                    "version": "uv 0.11.11",
                },
                "pdflatex": {
                    "path": "C:/runner/_temp/via000-r3-texlive/2026/bin/windows/pdftex.exe",
                    "sha256": "5" * 64,
                    "version": "pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)",
                },
                "powershell": {
                    "path": "C:/Program Files/PowerShell/7/pwsh.exe",
                    "sha256": "6" * 64,
                    "version": "7.5.3",
                },
                "environment_python": {
                    "path": (
                        "C:/runner/_temp/via000-r3-platform/tool-closure/python-environment/Scripts/python.exe"
                    ),
                    "sha256": "3" * 64,
                    "version": "3.11.15",
                },
                "containment_protocol": {
                    "path": (
                        "C:/runner/work/POPGP/protocols/"
                        "POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
                    ),
                    "sha256": "7" * 64,
                    "version": (
                        "VIA-000 R3 restricted-token/job-object plus systemd "
                        "ephemeral-user/control-group"
                    ),
                },
            }
            runner_label, image_os = "windows-2025", "win25"
        else:
            tools = {
                "git": {
                    "path": "/usr/bin/git",
                    "sha256": "1" * 64,
                    "version": "git version 2.51.0",
                },
                "ssh_keygen": {
                    "path": "/usr/bin/ssh-keygen",
                    "sha256": "2" * 64,
                    "version": "OpenSSH system ssh-keygen",
                },
                "base_python": {
                    "path": "/opt/hostedtoolcache/Python/3.11.15/x64/bin/python3.11",
                    "sha256": "3" * 64,
                    "version": "3.11.15",
                },
                "uv": {
                    "path": "/opt/hostedtoolcache/Python/3.11.15/x64/bin/uv",
                    "sha256": "4" * 64,
                    "version": "uv 0.11.11",
                },
                "pdflatex": {
                    "path": (
                        "/home/runner/work/_temp/via000-r3-texlive/2026/bin/x86_64-linux/pdftex"
                    ),
                    "sha256": "5" * 64,
                    "version": "pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)",
                },
                "powershell": {
                    "path": "/opt/microsoft/powershell/7/pwsh",
                    "sha256": "6" * 64,
                    "version": "7.5.3",
                },
                "environment_python": {
                    "path": (
                        "/home/runner/work/_temp/via000-r3-platform/tool-closure/python-environment/bin/python"
                    ),
                    "sha256": "3" * 64,
                    "version": "3.11.15",
                },
                "containment_protocol": {
                    "path": (
                        "/home/runner/work/POPGP/protocols/"
                        "POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
                    ),
                    "sha256": "7" * 64,
                    "version": (
                        "VIA-000 R3 restricted-token/job-object plus systemd "
                        "ephemeral-user/control-group"
                    ),
                },
                "sudo": {
                    "path": "/usr/bin/sudo",
                    "sha256": "8" * 64,
                    "version": "Ubuntu 24.04 hosted system containment primitive",
                },
                "systemd_run": {
                    "path": "/usr/bin/systemd-run",
                    "sha256": "9" * 64,
                    "version": "Ubuntu 24.04 hosted system containment primitive",
                },
                "systemctl": {
                    "path": "/usr/bin/systemctl",
                    "sha256": "a" * 64,
                    "version": "Ubuntu 24.04 hosted system containment primitive",
                },
            }
            runner_label, image_os = "ubuntu-24.04", "ubuntu24"
        tool_identity_path = workspace / "evidence/tool-identity-manifest.json"
        tool_identity_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "platform_family": platform_name,
                    "runner_label": runner_label,
                    "image_os": image_os,
                    "image_version": "20260823.1.0",
                    "runner_arch": "X64",
                    "tools": tools,
                }
            ),
            encoding="utf-8",
        )
        manifest.append(
            {
                "platform_family": platform_name,
                "path": "evidence/tool-identity-manifest.json",
                "sha256": hashlib.sha256(tool_identity_path.read_bytes()).hexdigest(),
                "byte_count": tool_identity_path.stat().st_size,
                "media_type": "application/json",
                "role": "tool-identity-manifest",
            }
        )
        command = [
            tools["environment_python"]["path"],
            "-I",
            "-S",
            "-X",
            "pycache_prefix=/runner/_temp/via000-r3-platform/mutation-python-cache",
            "/runner/work/POPGP/scripts/run_without_startup_hooks.py",
            "--repo-root",
            "/runner/work/POPGP",
            "--module",
            "pytest",
            "--",
            "-vv",
            "-p",
            "no:cacheprovider",
            *selectors,
        ]
        stdout_entry = next(entry for entry in manifest if entry["role"] == "mutation-suite-stdout")
        stderr_entry = next(entry for entry in manifest if entry["role"] == "mutation-suite-stderr")
        suite_entry = next(entry for entry in manifest if entry["role"] == "mutation-suite-result")
        stdout_path = workspace / stdout_entry["path"]
        stdout = stdout_path.read_text(encoding="utf-8")
        for requirements in contract["required_mutation_tests"].values():
            for requirement in requirements:
                prefix = requirement["test_prefix"]
                if not prefix.startswith("tests/unit/test_via000_r3_identity.py::"):
                    continue
                for index in range(requirement["expected_passed_count"]):
                    suffix = "" if requirement["expected_passed_count"] == 1 else f"[case{index}]"
                    stdout += f"\n{prefix}{suffix} PASSED\n"
        stdout_path.write_text(stdout, encoding="utf-8")
        stdout_digest = hashlib.sha256(stdout_path.read_bytes()).hexdigest()
        stdout_entry["sha256"] = stdout_digest
        stdout_entry["byte_count"] = stdout_path.stat().st_size

        suite_path = workspace / suite_entry["path"]
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        suite["protocol_source_commit"] = FAKE_COMMIT
        suite["dispatch_identity"] = DISPATCH_IDENTITY
        suite["command"] = command
        suite["stdout_sha256"] = stdout_digest
        primitive = (
            "windows-low-integrity-restricted-token-job-object"
            if platform_name == "windows-x86_64"
            else "ubuntu-systemd-ephemeral-user-control-group"
        )
        separation = (
            "low-integrity-restricted-token"
            if platform_name == "windows-x86_64"
            else "systemd-ephemeral-user"
        )
        suite.update(
            {
                "primitive": primitive,
                "privilege_separation": separation,
                "descendants_quiescent": True,
                "active_processes_after_teardown": 0,
                **_token_evidence(platform_name),
            }
        )
        suite_path.write_text(json.dumps(suite), encoding="utf-8")
        suite_entry["sha256"] = hashlib.sha256(suite_path.read_bytes()).hexdigest()
        suite_entry["byte_count"] = suite_path.stat().st_size

        passed_nodes = [
            match.group(1)
            for line in stdout.splitlines()
            if (match := re.match(r"^(tests/\S+::\S+)\s+PASSED(?:\s+\[.*\])?$", line.strip()))
        ]
        mutation_entries = {
            Path(entry["path"]).stem: entry
            for entry in manifest
            if entry["role"] == "mutation-result"
        }
        for mutation_id in contract["required_mutation_ids"]:
            entry = mutation_entries[mutation_id]
            receipt_path = workspace / entry["path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected_test_ids = sorted(
                node
                for requirement in contract["required_mutation_tests"][mutation_id]
                for node in passed_nodes
                if node.startswith(requirement["test_prefix"])
            )
            receipt["oracle_errors"] = [
                {
                    "error_id": contract["required_mutation_oracles"][mutation_id],
                    "message": (
                        f"trusted workflow executed {requirement['expected_passed_count']} "
                        f"frozen rejection test(s) under {requirement['test_prefix']}"
                    ),
                }
                for requirement in contract["required_mutation_tests"][mutation_id]
            ]
            execution = receipt["execution"]
            execution["command"] = " ".join(command)
            execution["test_ids"] = expected_test_ids
            execution["passed_test_count"] = len(expected_test_ids)
            execution["stdout_sha256"] = stdout_digest
            execution["stderr_sha256"] = stderr_entry["sha256"]
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            entry["sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            entry["byte_count"] = receipt_path.stat().st_size
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        containment_count = 1
        for entry in manifest:
            if entry["role"] != "command-result":
                continue
            result_path = workspace / entry["path"]
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result.update(
                {
                    "primitive": primitive,
                    "privilege_separation": separation,
                    "descendants_quiescent": True,
                    "active_processes_after_teardown": 0,
                    **_token_evidence(platform_name),
                }
            )
            if platform_name == "ubuntu-latest-x86_64":
                result.update(
                    {
                        "ephemeral_identity_uid": "999",
                        "ephemeral_identity_processes_empty": True,
                        "ephemeral_identity_removed": True,
                    }
                )
            result_path.write_text(json.dumps(result), encoding="utf-8")
            entry["sha256"] = _sha(result_path)
            entry["byte_count"] = result_path.stat().st_size
            containment_count += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        summary_path = workspace / "evidence/platform-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["campaign_id"] = "POPGP-VIABILITY-R3-2026-08"
        summary["protocol_source_commit"] = FAKE_COMMIT
        summary["dispatch_identity"] = DISPATCH_IDENTITY
        summary["producer_attestation"] = {
            "repository": signer["repository"],
            "signer_workflow": signer["signer_workflow"],
            "source_commit": FAKE_COMMIT,
            "bundle_path": signer["bundle_path"],
            "subject_paths": signer["subject_paths"],
        }
        summary["tool_identity_manifest_sha256"] = hashlib.sha256(
            tool_identity_path.read_bytes()
        ).hexdigest()
        summary["execution_boundary"] = {
            "primitive": primitive,
            "privilege_separation": separation,
            "all_commands_contained": True,
            "descendants_quiescent": True,
            "active_processes_after_teardown": 0,
            "untrusted_identity_processes_empty": True,
            "untrusted_identity_retired": True,
            "trusted_evidence_unreadable_unwritable": True,
            "mutable_root_separate": True,
            "attestation_subjects_captured_after_quiescence": True,
            "contained_command_count": containment_count,
            **_token_evidence(platform_name),
        }
        summary["evidence_paths"] = sorted(
            [*summary["evidence_paths"], "evidence/tool-identity-manifest.json"]
        )
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    staged_roots: dict[str, Path] = {}
    for platform_name, combined in roots.items():
        platform_root = tmp_path / "staged" / platform_name
        for stage in ("candidate", "pdf", "mutation"):
            stage_root = platform_root / stage
            shutil.copytree(combined, stage_root)
            summary_path = stage_root / "evidence/platform-summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["stage_id"] = stage
            summary["producer_attestation"]["subject_paths"] = [
                "evidence/stage-summary.json",
                "evidence/evidence-manifest.json",
            ]
            summary_path.rename(stage_root / "evidence/stage-summary.json")
            (stage_root / "evidence/stage-summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            tool_path = stage_root / "evidence/tool-identity-manifest.json"
            tool = json.loads(tool_path.read_text(encoding="utf-8"))
            tool["stage_id"] = stage
            tool_path.write_text(json.dumps(tool), encoding="utf-8")
            manifest_path = stage_root / "evidence/evidence-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tool_entry = next(
                entry
                for entry in manifest
                if entry["path"] == "evidence/tool-identity-manifest.json"
            )
            tool_entry["sha256"] = _sha(tool_path)
            tool_entry["byte_count"] = tool_path.stat().st_size
            summary["tool_identity_manifest_sha256"] = _sha(tool_path)
            (stage_root / "evidence/stage-summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        staged_roots[platform_name] = platform_root
    return staged_roots


def _run_assembler(
    roots: dict[str, Path],
    output: Path,
    *,
    identity_error: bool = False,
    attestation_error: bool = False,
) -> subprocess.CompletedProcess[str]:
    identity_patch = (
        "(_ for _ in ()).throw(ValueError('protocol identity mismatch'))"
        if identity_error
        else repr(FAKE_AUTHORIZATION)
    )
    attestation_patch = (
        "(_ for _ in ()).throw(ValueError('attestation rejected'))" if attestation_error else "None"
    )
    bootstrap = (
        "import importlib.util,sys;"
        "path=sys.argv.pop(1);"
        "spec=importlib.util.spec_from_file_location('r3_assembler_test',path);"
        "module=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(module);"
        f"module._verify_protocol_identity=lambda *args,**kwargs:{identity_patch};"
        f"module._verify_attestation=lambda *args,**kwargs:{attestation_patch};"
        "module._run_frozen_precommit_validator=lambda *args,**kwargs:None;"
        "raise SystemExit(module.main())"
    )
    command = [
        sys.executable,
        "-c",
        bootstrap,
        str(ASSEMBLER),
        "--protocol",
        str(PROTOCOL),
        "--schema",
        str(SCHEMA),
    ]
    for platform in r2_assembler.PLATFORMS:
        command.extend(["--platform-root", f"{platform}={roots[platform]}"])
    command.extend(
        [
            "--protocol-source-ref",
            FAKE_REF,
            "--authorization-ref",
            FAKE_AUTHORIZATION_REF,
            "--producer-run-id",
            DISPATCH_IDENTITY["producer_run_id"],
            "--producer-run-attempt",
            str(DISPATCH_IDENTITY["producer_run_attempt"]),
            "--output-dir",
            str(output),
            "--committed-by",
            "r3-test-runner",
            "--committed-at",
            "2026-08-23T00:00:00Z",
        ]
    )
    return subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)


def test_dispatch_guard_accepts_exact_snapshot_ref(tmp_path: Path) -> None:
    guard = _load_module(GUARD, "r3_dispatch_guard_happy")
    repo, commit, ref, authorization_ref, _authorization_commit = _authorized_repo(tmp_path)
    authorization_tag_oid = _git(repo, "rev-parse", authorization_ref)
    result = guard.verify_campaign_authorization(
        repo,
        event_name="workflow_dispatch",
        github_ref=ref,
        github_sha=commit,
        authorization_ref=authorization_ref,
        expected_authorization_tag_oid=authorization_tag_oid,
    )
    assert result["protocol_commit"] == commit
    assert result["authorization_ref"] == authorization_ref


@pytest.mark.negative_control
@pytest.mark.parametrize("mutation", ["event", "ref", "sha", "head", "authorization"])
def test_dispatch_guard_rejects_wrong_event_ref_sha_or_head(tmp_path: Path, mutation: str) -> None:
    guard = _load_module(GUARD, f"r3_dispatch_guard_{mutation}")
    repo, commit, ref, authorization_ref, _authorization_commit = _authorized_repo(tmp_path)
    authorization_tag_oid = _git(repo, "rev-parse", authorization_ref)
    event_name = "workflow_dispatch"
    github_ref = ref
    github_sha = commit
    if mutation == "event":
        event_name = "push"
    elif mutation == "ref":
        github_ref = "refs/heads/master"
    elif mutation == "sha":
        github_sha = "f" * 40
    elif mutation == "head":
        (repo / "lifecycle.txt").write_text("later lifecycle head\n", encoding="utf-8")
        _git(repo, "add", "--sparse", "lifecycle.txt")
        _git(repo, "commit", "-m", "lifecycle handoff")
    else:
        authorization_ref = "refs/tags/popgp-via000-r3-authorization-" + "0" * 64
    with pytest.raises(ValueError):
        guard.verify_campaign_authorization(
            repo,
            event_name=event_name,
            github_ref=github_ref,
            github_sha=github_sha,
            authorization_ref=authorization_ref,
            expected_authorization_tag_oid=authorization_tag_oid,
        )


@pytest.mark.negative_control
def test_r3_authorization_rejects_self_consistent_later_lifecycle_and_substitution(
    tmp_path: Path,
) -> None:
    guard = _load_module(GUARD, "r3_dispatch_guard_later_lifecycle")
    repo, snapshot, _ref, authorization_ref, _authorization_commit = _authorized_repo(tmp_path)
    authorization_tag_oid = _git(repo, "rev-parse", authorization_ref)
    lifecycle = repo / "protocols/POPGP-VIABILITY-R3-2026-08/later-lifecycle.txt"
    lifecycle.write_text("later\n", encoding="utf-8", newline="\n")
    _git(repo, "add", str(lifecycle))
    _git(repo, "commit", "-m", "later lifecycle")
    later = _git(repo, "rev-parse", "HEAD")
    later_ref = f"refs/tags/popgp-via000-r3-protocol-{later}"
    _git(repo, "tag", later_ref.removeprefix("refs/tags/"), later)
    with pytest.raises(ValueError, match="campaign|packet|contract"):
        guard.verify_campaign_authorization(
            repo,
            event_name="workflow_dispatch",
            github_ref=later_ref,
            github_sha=later,
            authorization_ref=authorization_ref,
            expected_authorization_tag_oid=authorization_tag_oid,
        )
    assert snapshot != later
    output = tmp_path / "unauthorized-output"
    command = [
        sys.executable,
        str(repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"),
        "--repo-root",
        str(repo),
        "--protocol",
        str(repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"),
        "--schema",
        str(repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"),
        "--platform-root",
        f"ubuntu-latest-x86_64={tmp_path / 'missing-linux'}",
        "--platform-root",
        f"windows-x86_64={tmp_path / 'missing-windows'}",
        "--protocol-source-ref",
        later_ref,
        "--authorization-ref",
        authorization_ref,
        "--producer-run-id",
        "42",
        "--producer-run-attempt",
        "1",
        "--output-dir",
        str(output),
        "--committed-by",
        "test",
        "--committed-at",
        "2026-08-23T00:00:00Z",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode != 0
    assert not output.exists()
    assert not list(tmp_path.glob(".unauthorized-output-*"))


@pytest.mark.negative_control
@pytest.mark.parametrize(
    "mutation", ["deleted-authorization", "moved-authorization", "wrong-suffix", "annotated-source"]
)
def test_r3_authorization_rejects_moved_deleted_or_wrong_kind_refs(
    tmp_path: Path, mutation: str
) -> None:
    guard = _load_module(GUARD, f"r3_dispatch_guard_ref_{mutation}")
    repo, snapshot, protocol_ref, authorization_ref, _authorization_commit = _authorized_repo(
        tmp_path
    )
    authorization_tag_oid = _git(repo, "rev-parse", authorization_ref)
    if mutation == "deleted-authorization":
        _git(repo, "tag", "-d", authorization_ref.removeprefix("refs/tags/"))
    elif mutation == "moved-authorization":
        _git(repo, "update-ref", authorization_ref, snapshot)
    elif mutation == "wrong-suffix":
        tag_oid = _git(repo, "rev-parse", authorization_ref)
        authorization_ref = "refs/tags/popgp-via000-r3-authorization-" + "0" * 64
        _git(repo, "update-ref", authorization_ref, tag_oid)
    else:
        _git(repo, "tag", "-d", protocol_ref.removeprefix("refs/tags/"))
        _git(
            repo,
            "tag",
            "-a",
            "-m",
            "annotated source tags are forbidden",
            protocol_ref.removeprefix("refs/tags/"),
            snapshot,
        )
    with pytest.raises(ValueError):
        guard.verify_campaign_authorization(
            repo,
            event_name="workflow_dispatch",
            github_ref=protocol_ref,
            github_sha=snapshot,
            authorization_ref=authorization_ref,
            expected_authorization_tag_oid=authorization_tag_oid,
        )


@pytest.mark.negative_control
def test_r3_authorization_rejects_invalid_captured_object_after_valid_ref_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _load_module(GUARD, "r3_dispatch_guard_captured_invalid_swap")
    repo, snapshot, protocol_ref, authorization_ref, authorization_commit = _authorized_repo(
        tmp_path
    )
    _original_oid, invalid_oid, unrelated_valid_oid = _authorization_race_objects(
        repo, authorization_ref, authorization_commit
    )
    original_git = guard._git
    swapped = False

    def swap_after_parse(repo_root: Path, *arguments: str) -> bytes:
        nonlocal swapped
        result = original_git(repo_root, *arguments)
        if arguments == ("cat-file", "tag", invalid_oid) and not swapped:
            _git(
                repo,
                "update-ref",
                authorization_ref,
                unrelated_valid_oid,
                invalid_oid,
            )
            swapped = True
        return result

    monkeypatch.setattr(guard, "_git", swap_after_parse)
    output_json = tmp_path / "authorization.json"
    with pytest.raises(ValueError, match="signature|changed|captured"):
        guard.verify_campaign_authorization(
            repo,
            event_name="workflow_dispatch",
            github_ref=protocol_ref,
            github_sha=snapshot,
            authorization_ref=authorization_ref,
            expected_authorization_tag_oid=invalid_oid,
        )
    assert swapped
    assert _git(repo, "rev-parse", authorization_ref) == unrelated_valid_oid
    assert not output_json.exists()


@pytest.mark.negative_control
@pytest.mark.parametrize(
    ("stage", "mutation"),
    [("parse", "delete"), ("peel", "move"), ("verify", "swap")],
)
def test_r3_authorization_rejects_ref_change_during_exact_object_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    mutation: str,
) -> None:
    guard = _load_module(GUARD, f"r3_dispatch_guard_race_{stage}")
    repo, snapshot, protocol_ref, authorization_ref, authorization_commit = _authorized_repo(
        tmp_path
    )
    original_oid, _invalid_oid, unrelated_valid_oid = _authorization_race_objects(
        repo, authorization_ref, authorization_commit
    )
    _git(repo, "update-ref", authorization_ref, original_oid)
    original_git = guard._git
    original_run = guard.subprocess.run
    changed = False

    def mutate_ref() -> None:
        nonlocal changed
        if changed:
            return
        if mutation == "delete":
            command = ["update-ref", "-d", authorization_ref, original_oid]
        elif mutation == "move":
            command = ["update-ref", authorization_ref, authorization_commit, original_oid]
        else:
            command = ["update-ref", authorization_ref, unrelated_valid_oid, original_oid]
        original_run(["git", "-C", str(repo), *command], check=True, capture_output=True)
        changed = True

    def git_with_race(repo_root: Path, *arguments: str) -> bytes:
        result = original_git(repo_root, *arguments)
        if stage == "parse" and arguments == ("cat-file", "tag", original_oid):
            mutate_ref()
        elif stage == "peel" and arguments == (
            "rev-parse",
            "--verify",
            f"{original_oid}^{{commit}}",
        ):
            mutate_ref()
        return result

    def run_with_race(*args: object, **kwargs: object):
        command = args[0] if args else kwargs.get("args")
        if stage == "verify" and isinstance(command, list) and "verify-tag" in command:
            mutate_ref()
        return original_run(*args, **kwargs)

    monkeypatch.setattr(guard, "_git", git_with_race)
    monkeypatch.setattr(guard.subprocess, "run", run_with_race)
    with pytest.raises(ValueError, match="changed|Git authorization query failed"):
        guard.verify_campaign_authorization(
            repo,
            event_name="workflow_dispatch",
            github_ref=protocol_ref,
            github_sha=snapshot,
            authorization_ref=authorization_ref,
            expected_authorization_tag_oid=original_oid,
        )
    assert changed
    assert not (tmp_path / "authorization.json").exists()


@pytest.mark.negative_control
def test_r3_assembler_extracted_guard_rejects_ref_swap_after_oid_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assembler = _load_module(ASSEMBLER, "r3_assembler_extracted_guard_ref_swap")
    repo, _snapshot, protocol_ref, authorization_ref, authorization_commit = _authorized_repo(
        tmp_path
    )
    _original_oid, invalid_oid, unrelated_valid_oid = _authorization_race_objects(
        repo, authorization_ref, authorization_commit
    )
    protocol_path = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
    schema_path = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(
        repo_root=repo,
        protocol=protocol_path,
        schema=schema_path,
        protocol_source_ref=protocol_ref,
        authorization_ref=authorization_ref,
    )
    original_git_output = assembler._git_output
    swapped = False

    def capture_then_swap(repo_root: Path, *arguments: str) -> bytes:
        nonlocal swapped
        result = original_git_output(repo_root, *arguments)
        if arguments == ("rev-parse", "--verify", authorization_ref) and not swapped:
            assert result.decode("ascii").strip() == invalid_oid
            _git(
                repo,
                "update-ref",
                authorization_ref,
                unrelated_valid_oid,
                invalid_oid,
            )
            swapped = True
        return result

    monkeypatch.setattr(assembler, "_git_output", capture_then_swap)
    with pytest.raises(ValueError, match="authorization failed"):
        assembler._verify_protocol_identity(args, protocol)
    assert swapped
    assert not (tmp_path / "assembled").exists()
    assert not list(tmp_path.glob(".assembled-*"))


@pytest.mark.negative_control
def test_r3_guard_rejects_invalid_tag_hidden_by_default_replacement(
    tmp_path: Path,
) -> None:
    guard = _load_module(GUARD, "r3_dispatch_guard_default_replace")
    repo, snapshot, protocol_ref, authorization_ref, authorization_commit = _authorized_repo(
        tmp_path
    )
    original_oid, invalid_oid, _unrelated_oid = _authorization_race_objects(
        repo, authorization_ref, authorization_commit
    )
    _replace_object(repo, invalid_oid, original_oid)
    assert _git_bytes(repo, "cat-file", "tag", invalid_oid) == _git_bytes(
        repo, "--no-replace-objects", "cat-file", "tag", original_oid
    )
    assert _git_bytes(repo, "--no-replace-objects", "cat-file", "tag", invalid_oid) != _git_bytes(
        repo, "cat-file", "tag", invalid_oid
    )

    with pytest.raises(ValueError, match="signature"):
        guard.verify_campaign_authorization(
            repo,
            event_name="workflow_dispatch",
            github_ref=protocol_ref,
            github_sha=snapshot,
            authorization_ref=authorization_ref,
            expected_authorization_tag_oid=invalid_oid,
        )
    assert not (tmp_path / "authorization.json").exists()


@pytest.mark.negative_control
def test_r3_unmodified_assembler_rejects_invalid_tag_hidden_by_replacement(
    tmp_path: Path,
) -> None:
    repo, _snapshot, protocol_ref, authorization_ref, authorization_commit = _authorized_repo(
        tmp_path
    )
    original_oid, invalid_oid, _unrelated_oid = _authorization_race_objects(
        repo, authorization_ref, authorization_commit
    )
    _replace_object(repo, invalid_oid, original_oid)
    output = tmp_path / "replacement-output"
    command = [
        sys.executable,
        str(repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"),
        "--repo-root",
        str(repo),
        "--protocol",
        str(repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"),
        "--schema",
        str(repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"),
        "--platform-root",
        f"ubuntu-latest-x86_64={tmp_path / 'missing-linux'}",
        "--platform-root",
        f"windows-x86_64={tmp_path / 'missing-windows'}",
        "--protocol-source-ref",
        protocol_ref,
        "--authorization-ref",
        authorization_ref,
        "--producer-run-id",
        "42",
        "--producer-run-attempt",
        "1",
        "--output-dir",
        str(output),
        "--committed-by",
        "test",
        "--committed-at",
        "2026-08-23T00:00:00Z",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "signature" in completed.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".replacement-output-*"))


@pytest.mark.negative_control
def test_r3_guard_scrubs_custom_replace_object_repository_and_config_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _load_module(GUARD, "r3_dispatch_guard_git_environment")
    repo, snapshot, protocol_ref, authorization_ref, _authorization_commit = _authorized_repo(
        tmp_path
    )
    authorization_oid = _git(repo, "rev-parse", authorization_ref)
    missing = tmp_path / "caller-controlled-missing"
    _git(repo, "config", "gpg.ssh.program", str(missing / "local-ssh-keygen"))
    injected = {
        "GIT_REPLACE_REF_BASE": "refs/caller-replacements/",
        "GIT_OBJECT_DIRECTORY": str(missing / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(missing / "alternates"),
        "GIT_DIR": str(missing / "repository.git"),
        "GIT_WORK_TREE": str(missing / "worktree"),
        "GIT_INDEX_FILE": str(missing / "index"),
        "GIT_COMMON_DIR": str(missing / "common"),
        "GIT_NAMESPACE": "caller-controlled",
        "GIT_CEILING_DIRECTORIES": str(repo.parent),
        "GIT_DISCOVERY_ACROSS_FILESYSTEM": "0",
        "GIT_EXEC_PATH": str(missing / "exec"),
        "GIT_SSH": str(missing / "ssh"),
        "GIT_SSH_COMMAND": str(missing / "ssh-command"),
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "gpg.ssh.program",
        "GIT_CONFIG_VALUE_0": str(missing / "ssh-keygen"),
        "GIT_CONFIG_KEY_1": "core.useReplaceRefs",
        "GIT_CONFIG_VALUE_1": "true",
    }
    for name, value in injected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", str(missing / "programs"))

    result = guard.verify_campaign_authorization(
        repo,
        event_name="workflow_dispatch",
        github_ref=protocol_ref,
        github_sha=snapshot,
        authorization_ref=authorization_ref,
        expected_authorization_tag_oid=authorization_oid,
    )
    assert result["authorization_tag_oid"] == authorization_oid
    clean_environment = guard._git_environment()
    assert all(name not in clean_environment for name in injected)
    assert clean_environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert Path(guard.GIT_EXECUTABLE).is_file()


@pytest.mark.negative_control
def test_r3_custom_replace_namespace_cannot_authenticate_invalid_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _load_module(GUARD, "r3_dispatch_guard_custom_replace")
    repo, snapshot, protocol_ref, authorization_ref, authorization_commit = _authorized_repo(
        tmp_path
    )
    original_oid, invalid_oid, _unrelated_oid = _authorization_race_objects(
        repo, authorization_ref, authorization_commit
    )
    replacement_base = "refs/caller-replacements/"
    _git(repo, "update-ref", replacement_base + invalid_oid, original_oid)
    monkeypatch.setenv("GIT_REPLACE_REF_BASE", replacement_base)
    assert _git_bytes(repo, "cat-file", "tag", invalid_oid) == _git_bytes(
        repo, "--no-replace-objects", "cat-file", "tag", original_oid
    )

    with pytest.raises(ValueError, match="signature"):
        guard.verify_campaign_authorization(
            repo,
            event_name="workflow_dispatch",
            github_ref=protocol_ref,
            github_sha=snapshot,
            authorization_ref=authorization_ref,
            expected_authorization_tag_oid=invalid_oid,
        )


@pytest.mark.negative_control
def test_r3_source_campaign_packet_manifest_and_validator_reads_ignore_replacements(
    tmp_path: Path,
) -> None:
    assembler = _load_module(ASSEMBLER, "r3_assembler_source_replacements")
    repo, snapshot, protocol_ref, authorization_ref, authorization_commit = _authorized_repo(
        tmp_path
    )
    protocol_path = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
    schema_path = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    contract = protocol["parameters"]["authorization_contract"]

    original_primary = protocol_path.read_bytes()
    protocol_path.write_bytes(original_primary + b"\n")
    _git(repo, "add", str(protocol_path))
    _git(repo, "commit", "-m", "caller-controlled replacement snapshot")
    replacement_commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--detach", snapshot)
    _replace_object(repo, snapshot, replacement_commit)

    substituted_blob = _hash_blob(repo, b"caller-controlled replacement bytes\n")
    for commit, relative in (
        (authorization_commit, contract["campaign_path"]),
        (authorization_commit, contract["packet_path"]),
        (authorization_commit, contract["protocol_manifest_path"]),
        (snapshot, "scripts/check_viability_campaign.py"),
    ):
        original_blob = _git(repo, "--no-replace-objects", "rev-parse", f"{commit}:{relative}")
        _replace_object(repo, original_blob, substituted_blob)
        assert _git_bytes(repo, "cat-file", "blob", original_blob) == (
            b"caller-controlled replacement bytes\n"
        )

    assert (
        _git_bytes(
            repo,
            "show",
            f"{snapshot}:protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json",
        )
        != original_primary
    )
    args = SimpleNamespace(
        repo_root=repo,
        protocol=protocol_path,
        schema=schema_path,
        protocol_source_ref=protocol_ref,
        authorization_ref=authorization_ref,
    )
    authorization = assembler._verify_protocol_identity(args, protocol)
    assert authorization["protocol_commit"] == snapshot
    raw_path = tmp_path / "invalid-raw-results.json"
    raw_path.write_text("{}\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="frozen validation"):
        assembler._run_frozen_precommit_validator(args, protocol, authorization, raw_path)
    assert not (tmp_path / "assembled").exists()


@pytest.mark.negative_control
@pytest.mark.parametrize(
    "dependency",
    [
        "scripts/check_viability_campaign.py",
        "scripts/check_validation_artifacts.py",
        "scripts/check_reproduction_boundary.py",
        "popgp/diagnostics.py",
    ],
)
def test_r3_frozen_validator_rejects_worktree_dependency_substitution_without_output(
    tmp_path: Path, dependency: str
) -> None:
    assembler = _load_module(ASSEMBLER, "r3_frozen_validator_source_closure")
    repo, _snapshot, protocol_ref, authorization_ref, _authorization_commit = _authorized_repo(
        tmp_path
    )
    protocol_path = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
    schema_path = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(
        repo_root=repo,
        protocol=protocol_path,
        schema=schema_path,
        protocol_source_ref=protocol_ref,
        authorization_ref=authorization_ref,
    )
    authorization = assembler._verify_protocol_identity(args, protocol)
    raw_path = tmp_path / "raw-results.json"
    raw_path.write_text("{}\n", encoding="utf-8", newline="\n")
    changed = repo / dependency
    changed.write_bytes(changed.read_bytes() + b"\n# substituted only in worktree\n")
    with pytest.raises(ValueError, match="worktree source differs"):
        assembler._run_frozen_precommit_validator(args, protocol, authorization, raw_path)
    assert not (tmp_path / "assembled").exists()


@pytest.mark.negative_control
def test_r3_frozen_validator_executes_snapshot_bundle_and_real_packet(tmp_path: Path) -> None:
    assembler = _load_module(ASSEMBLER, "r3_frozen_validator_bundle_execution")
    repo, _snapshot, protocol_ref, authorization_ref, _authorization_commit = _authorized_repo(
        tmp_path
    )
    protocol_path = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
    schema_path = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(
        repo_root=repo,
        protocol=protocol_path,
        schema=schema_path,
        protocol_source_ref=protocol_ref,
        authorization_ref=authorization_ref,
    )
    authorization = assembler._verify_protocol_identity(args, protocol)
    raw_path = tmp_path / "invalid-raw-results.json"
    raw_path.write_text("{}\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="frozen validation") as raised:
        assembler._run_frozen_precommit_validator(args, protocol, authorization, raw_path)
    assert "raw-results" in str(raised.value)
    assert not (tmp_path / "assembled").exists()


@pytest.mark.parametrize("autocrlf", ["true", "false"])
def test_r3_cross_platform_git_blob_identity_with_autocrlf(tmp_path: Path, autocrlf: str) -> None:
    origin = tmp_path / "blob-origin"
    origin.mkdir()
    _git(origin, "init")
    _git(origin, "config", "user.name", "R3 blob test")
    _git(origin, "config", "user.email", "r3-blob@example.invalid")
    _git(origin, "config", "core.autocrlf", "false")
    _overlay(ROOT / ".gitattributes", origin / ".gitattributes")
    _overlay(WORKFLOW, origin / ".github/workflows/via000-r3-protocol.yml")
    _git(origin, "add", ".")
    _git(origin, "commit", "-m", "blob fixture")
    clone = tmp_path / f"checkout-{autocrlf}"
    subprocess.run(
        [
            "git",
            "-c",
            f"core.autocrlf={autocrlf}",
            "clone",
            "--quiet",
            str(origin),
            str(clone),
        ],
        check=True,
    )
    relative = ".github/workflows/via000-r3-protocol.yml"
    blob = _git_bytes(origin, "cat-file", "blob", f"HEAD:{relative}")
    assert clone.joinpath(relative).read_bytes() == blob
    assert b"\r\n" not in blob
    assert _git(clone, "check-attr", "eol", "--", relative).endswith("eol: lf")


@pytest.mark.negative_control
def test_r3_workflow_is_manual_only_and_binds_exact_identity() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    document = yaml.safe_load(workflow)
    trigger = workflow.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger
    assert "push:" not in trigger
    run_sources = [
        step["run"] for step in document["jobs"]["platform-fragment"]["steps"] if "run" in step
    ]
    assert all("${{" not in source for source in run_sources)
    authorization_script = _workflow_step_script("Reject non-snapshot lifecycle refs")
    assert "Get-Command" not in authorization_script
    for token in (
        "${{ github.ref }}",
        "${{ github.sha }}",
        "${{ inputs.authorization_ref }}",
        "--expected-authorization-tag-oid",
        "--no-replace-objects",
        'Where-Object Name -Like "GIT_*"',
        'GIT_NO_REPLACE_OBJECTS = "1"',
        'GIT_CONFIG_NOSYSTEM = "1"',
        "VIA000_AUTHORIZATION_REF: ${{ inputs.authorization_ref }}",
        "VIA000_BASE_PYTHON: ${{ steps.base-python.outputs.python-path }}",
        '"C:\\Program Files\\Git\\cmd\\git.exe"',
        '"C:\\Windows\\System32\\OpenSSH\\ssh-keygen.exe"',
        '"/usr/bin/git"',
        '"/usr/bin/ssh-keygen"',
        '"--end-of-options"',
        "authorization ref changed before platform execution",
        "${{ github.run_id }}",
        "${{ github.run_attempt }}",
        "VIA-000-DISPATCH-GUARD.py",
    ):
        assert token in workflow


@pytest.mark.negative_control
def test_r3_workflow_command_boundary_rejects_dispatch_payloads_as_inert_data(
    tmp_path: Path,
) -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    script_path = tmp_path / "authorization-step.ps1"
    script_path.write_text(
        _workflow_step_script("Reject non-snapshot lifecycle refs"),
        encoding="utf-8",
        newline="\n",
    )
    marker = tmp_path / "workflow-input-marker"
    reviewer_payload = (
        '"; Write-Output VIA000_AUTH_INPUT_CODE_EXECUTED; '
        '$capturedAuthorizationTagOid=("b"*40); $global:LASTEXITCODE=0; #'
    )
    side_effect_payload = (
        f"\"; Set-Content -LiteralPath '{marker}' -Value injected; $global:LASTEXITCODE=0; #"
    )
    payloads = [
        reviewer_payload,
        side_effect_payload,
        "refs/tags/popgp-via000-r3-authorization-" + "a" * 63 + "\n",
        "`Write-Output injected",
        "$(Write-Output injected)",
        "; Write-Output injected",
        "| Write-Output injected",
        "& Write-Output injected",
        '"quoted"',
        "'quoted'",
        "authorization-\N{SNOWMAN}",
        "authorization-\x01-control",
        " authorization ",
        "--help",
    ]
    for index, payload in enumerate(payloads):
        runner_temp = tmp_path / f"runner-{index}"
        runner_temp.mkdir()
        environment = os.environ.copy()
        environment.update(
            {
                "VIA000_AUTHORIZATION_REF": payload,
                "VIA000_EVENT_NAME": "workflow_dispatch",
                "VIA000_GITHUB_REF": FAKE_REF,
                "VIA000_GITHUB_SHA": FAKE_COMMIT,
                "VIA000_BASE_PYTHON": sys.executable,
                "VIA000_BASE_PYTHON_VERSION": "3.11.15",
                "VIA000_RUNNER_ARCH": "X64",
                "GITHUB_WORKSPACE": str(tmp_path),
                "RUNNER_TEMP": str(runner_temp),
            }
        )
        completed = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script_path)],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert "VIA000_AUTH_INPUT_CODE_EXECUTED" not in completed.stdout
        assert not marker.exists()
        assert not any(runner_temp.iterdir())


@pytest.mark.negative_control
def test_r3_workflow_command_boundary_ignores_path_tool_shims(
    tmp_path: Path,
) -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    repo, snapshot, protocol_ref, authorization_ref, _authorization_commit = _authorized_repo(
        tmp_path
    )
    script_path = tmp_path / "authorization-step.ps1"
    script_path.write_text(
        _workflow_step_script("Reject non-snapshot lifecycle refs"),
        encoding="utf-8",
        newline="\n",
    )
    marker = tmp_path / "path-shim-marker"
    shim_one = tmp_path / "shim-one"
    shim_two = tmp_path / "shim-two"
    for directory in (shim_one, shim_two):
        for name in ("git", "ssh-keygen", "python"):
            _path_shim(directory, name, marker)
    path_sets = [
        "",
        str(shim_one),
        os.pathsep.join((str(shim_one), str(shim_two), os.environ.get("PATH", ""))),
    ]
    for index, path_value in enumerate(path_sets):
        runner_temp = tmp_path / f"trusted-runner-{index}"
        runner_temp.mkdir()
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": path_value,
                "VIA000_AUTHORIZATION_REF": authorization_ref,
                "VIA000_EVENT_NAME": "workflow_dispatch",
                "VIA000_GITHUB_REF": protocol_ref,
                "VIA000_GITHUB_SHA": snapshot,
                "VIA000_BASE_PYTHON": sys.executable,
                "VIA000_BASE_PYTHON_VERSION": "3.11.15",
                "VIA000_RUNNER_ARCH": "X64",
                "GITHUB_WORKSPACE": str(repo),
                "RUNNER_TEMP": str(runner_temp),
            }
        )
        completed = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script_path)],
            cwd=repo,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        authorization_output = runner_temp / "via000-r3-authorization.json"
        assert not authorization_output.exists()
        assert not marker.exists()


@pytest.mark.negative_control
def test_r3_complete_execution_toolchain_has_no_path_resolved_commands() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    mutation_runner = MUTATION_RUNNER.read_text(encoding="utf-8")
    document = yaml.safe_load(workflow)
    assert {
        item["os"]
        for item in document["jobs"]["platform-fragment"]["strategy"]["matrix"]["include"]
    } == {
        "ubuntu-24.04",
        "windows-2025",
    }
    matrix = document["jobs"]["platform-fragment"]["strategy"]["matrix"]["include"]
    assert all("shell" not in item for item in matrix)
    trusted_runs = [
        step
        for step in document["jobs"]["platform-fragment"]["steps"]
        if "run" in step
    ]
    assert len(trusted_runs) == 8
    assert all(step["shell"] == "pwsh" for step in trusted_runs)
    assert 'update-environment: "false"' in workflow
    assert "if: always()" not in workflow
    assert "Validate complete trusted toolchain" in workflow
    assert "Assert-RegularExecutable" in workflow
    assert "Get-FileHash -Algorithm SHA256" in workflow
    assert "via000-r3-tool-identity.json" in workflow
    assert "C:\\hostedtoolcache\\windows\\Python\\3.11.15\\x64\\python.exe" in workflow
    assert "/opt/hostedtoolcache/Python/3.11.15/x64/bin/python3.11" in workflow
    assert "via000-r3-texlive/2026/bin/x86_64-linux/pdftex" in workflow
    assert "-TrustedGitPath" in workflow and "$TrustedGitPath" in runner
    assert "-TrustedBasePythonPath" in workflow and "$TrustedBasePythonPath" in runner
    assert "-TrustedUvPath" in workflow and "$TrustedUvPath" in runner
    assert "-TrustedPdfLatexPath" in workflow and "$TrustedPdfLatexPath" in runner
    assert "-TrustedPowerShellPath" in workflow and "$TrustedPowerShellPath" in runner
    assert "tool_identity_manifest_sha256" in runner
    assert (
        re.search(
            r'(?im)(?:&|FilePath\s+)\s*["\']?(?:git|python|uv|pdflatex|pwsh)["\']?(?:\s|$)',
            runner,
        )
        is None
    )
    assert '"uv",\n        "run"' not in mutation_runner
    assert '"python",\n        "-m"' not in mutation_runner
    assert "subprocess" not in mutation_runner
    assert "never executes candidate Python" in mutation_runner
    assert "Invoke-Via000ContainedCommand" in workflow
    assert "AssignProcessToJobObject before resume" in runner + (
        PROTOCOL_DIR / "VIA-000-CONTAINMENT.ps1"
    ).read_text(encoding="utf-8")


@pytest.mark.negative_control
def test_r3_workflow_rejects_setup_python_output_substitution_before_execution(
    tmp_path: Path,
) -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    script_path = tmp_path / "authorization-step.ps1"
    script_path.write_text(
        _workflow_step_script("Reject non-snapshot lifecycle refs"),
        encoding="utf-8",
        newline="\n",
    )
    marker = tmp_path / "setup-python-substitution-marker"
    malicious_python = tmp_path / "python.cmd"
    malicious_python.write_text(
        f'@echo executed>"{marker}"\r\n@exit /b 0\r\n',
        encoding="utf-8",
    )
    runner_temp = tmp_path / "setup-substitution-runner"
    runner_temp.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "VIA000_AUTHORIZATION_REF": FAKE_AUTHORIZATION_REF,
            "VIA000_EVENT_NAME": "workflow_dispatch",
            "VIA000_GITHUB_REF": FAKE_REF,
            "VIA000_GITHUB_SHA": FAKE_COMMIT,
            "VIA000_BASE_PYTHON": str(malicious_python),
            "VIA000_BASE_PYTHON_VERSION": "3.11.15",
            "VIA000_RUNNER_ARCH": "X64",
            "GITHUB_WORKSPACE": str(tmp_path),
            "RUNNER_TEMP": str(runner_temp),
        }
    )
    completed = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script_path)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert not marker.exists()
    assert not (runner_temp / "via000-r3-authorization.json").exists()


@pytest.mark.negative_control
def test_r3_tool_identity_manifest_rejects_substituted_paths_versions_and_hashes(
    tmp_path: Path,
) -> None:
    checker = _load_module(CAMPAIGN_CHECKER, "r3_tool_identity_checker")
    roots = _r3_platform_roots(tmp_path)
    for platform, workspace in roots.items():
        manifest_path = workspace / "candidate/evidence/tool-identity-manifest.json"
        valid = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid.pop("stage_id")
        assert checker._tool_identity_manifest_errors(valid, platform, platform) == []
        mutations = []
        for mutate in (
            lambda item: item["tools"]["base_python"].update(
                path="C:/untrusted/python.cmd" if os.name == "nt" else "/tmp/python"
            ),
            lambda item: item["tools"]["base_python"].update(version="3.11.14"),
            lambda item: item["tools"]["uv"].update(version="uv 0.11.12"),
            lambda item: item["tools"]["pdflatex"].update(version="forged banner"),
            lambda item: item["tools"]["environment_python"].update(sha256="9" * 64),
            lambda item: item.update(runner_arch="ARM64"),
        ):
            changed = json.loads(json.dumps(valid))
            mutate(changed)
            mutations.append(changed)
        for changed in mutations:
            assert checker._tool_identity_manifest_errors(changed, platform, platform)


@pytest.mark.negative_control
def test_r3_unmodified_runner_rejects_explicit_shim_before_workspace(
    tmp_path: Path,
) -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    marker = tmp_path / "runner-tool-shim-marker"
    shim_dir = tmp_path / "path-shims"
    for name in ("git", "python", "uv", "pdflatex", "pwsh"):
        _path_shim(shim_dir, name, marker)
    suffix = ".cmd" if os.name == "nt" else ""
    git_shim = shim_dir / f"git{suffix}"
    manifest = tmp_path / "tool-manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "runner-workspace"
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(shim_dir), environment.get("PATH", "")))
    environment["PATHEXT"] = ".CMD;.EXE"
    environment["RUNNER_TEMP"] = str(tmp_path)
    arguments = [
        pwsh,
        "-NoProfile",
        "-NonInteractive",
        "-File",
        str(RUNNER),
        "-WorkspaceRoot",
        str(workspace),
        "-PlatformFamily",
        "windows-x86_64" if os.name == "nt" else "ubuntu-latest-x86_64",
        "-ProtocolSourceCommit",
        FAKE_COMMIT,
        "-DispatchRef",
        FAKE_REF,
        "-AuthorizationRef",
        FAKE_AUTHORIZATION_REF,
        "-AuthorizationTagOid",
        "c" * 40,
        "-AuthorizationCommit",
        "d" * 40,
        "-AuthorizationRecordSha256",
        FAKE_AUTHORIZATION_SHA256,
        "-ProducerRunId",
        "424242",
        "-ProducerRunAttempt",
        "1",
        "-TrustedGitPath",
        str(git_shim),
        "-TrustedBasePythonPath",
        sys.executable,
        "-TrustedUvPath",
        str(shim_dir / f"uv{suffix}"),
        "-TrustedPdfLatexPath",
        str(shim_dir / f"pdflatex{suffix}"),
        "-TrustedPowerShellPath",
        str(shim_dir / f"pwsh{suffix}"),
        "-ToolIdentityManifestPath",
        str(manifest),
    ]
    for name, path in (
        ("TrustedGitSha256", git_shim),
        ("TrustedBasePythonSha256", Path(sys.executable)),
        ("TrustedUvSha256", shim_dir / f"uv{suffix}"),
        ("TrustedPdfLatexSha256", shim_dir / f"pdflatex{suffix}"),
        ("TrustedPowerShellSha256", shim_dir / f"pwsh{suffix}"),
        ("ToolIdentityManifestSha256", manifest),
    ):
        arguments.extend((f"-{name}", hashlib.sha256(path.read_bytes()).hexdigest()))
    completed = subprocess.run(
        arguments,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert not marker.exists()
    assert not workspace.exists()


@pytest.mark.negative_control
def test_r3_unmodified_mutation_runner_rejects_command_shim_before_output(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "via000-r3-platform"
    evidence = workspace / "evidence"
    environment_root = workspace / "python-environment"
    evidence.mkdir(parents=True)
    environment_root.mkdir()
    marker = tmp_path / "mutation-tool-shim-marker"
    shim_dir = tmp_path / "mutation-path-shims"
    for name in ("git", "python", "uv", "pdflatex", "pwsh"):
        _path_shim(shim_dir, name, marker)
    suffix = ".cmd" if os.name == "nt" else ""
    python_shim = environment_root / f"python{suffix}"
    _path_shim(environment_root, "python", marker)
    initial_entries = sorted(evidence.iterdir())
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(shim_dir), environment.get("PATH", "")))
    environment["PATHEXT"] = ".CMD;.EXE"
    completed = subprocess.run(
        [
            sys.executable,
            str(MUTATION_RUNNER),
            "--repo-root",
            str(ROOT),
            "--workspace-root",
            str(workspace),
            "--environment-python",
            str(python_shim),
            "--environment-python-sha256",
            hashlib.sha256(python_shim.read_bytes()).hexdigest(),
            "--platform-family",
            "windows-x86_64" if os.name == "nt" else "ubuntu-latest-x86_64",
            "--protocol-source-commit",
            FAKE_COMMIT,
            "--dispatch-ref",
            FAKE_REF,
            "--authorization-ref",
            FAKE_AUTHORIZATION_REF,
            "--authorization-tag-oid",
            "c" * 40,
            "--authorization-commit",
            "d" * 40,
            "--authorization-record-sha256",
            FAKE_AUTHORIZATION_SHA256,
            "--producer-run-id",
            "424242",
            "--producer-run-attempt",
            "1",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert not marker.exists()
    assert sorted(evidence.iterdir()) == initial_entries


@pytest.mark.negative_control
@pytest.mark.parametrize("mutation", ["identity", "attestation", "cross-run"])
def test_r3_assembler_rejects_identity_attestation_and_cross_run_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    roots = _r3_platform_roots(tmp_path)
    if mutation == "cross-run":
        changed = roots[r2_assembler.PLATFORMS[1]] / "candidate/evidence/stage-summary.json"
        summary = json.loads(changed.read_text(encoding="utf-8"))
        summary["dispatch_identity"]["producer_run_id"] = "424243"
        changed.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "assembled"
    result = _run_assembler(
        roots,
        output,
        identity_error=mutation == "identity",
        attestation_error=mutation == "attestation",
    )
    assert result.returncode != 0
    assert not output.exists()


def test_r3_assembler_exact_snapshot_ref_happy_path(tmp_path: Path) -> None:
    assembler = _load_module(ASSEMBLER, "r3_assembler_identity_happy")
    repo, commit, ref, authorization_ref, _authorization_commit = _authorized_repo(tmp_path)
    protocol = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
    schema = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
    args = SimpleNamespace(
        repo_root=repo,
        protocol=protocol,
        schema=schema,
        protocol_source_ref=ref,
        authorization_ref=authorization_ref,
    )
    authorization = assembler._verify_protocol_identity(
        args, json.loads(protocol.read_text(encoding="utf-8"))
    )
    assert authorization["protocol_commit"] == commit

    roots = _r3_platform_roots(tmp_path / "roundtrip")
    output = tmp_path / "assembled"
    result = _run_assembler(roots, output)
    assert result.returncode == 0, result.stderr
    assert (output / "raw-results.json").is_file()
    assert (output / "output-commitment.json").is_file()


def test_r3_workflow_separates_candidate_pdf_and_mutation_execution_contexts() -> None:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    matrix = document["jobs"]["platform-fragment"]["strategy"]["matrix"]["include"]
    assert {(item["platform"], item["stage"]) for item in matrix} == {
        (platform, stage)
        for platform in ("ubuntu-latest-x86_64", "windows-x86_64")
        for stage in ("candidate", "pdf", "mutation")
    }
    steps = document["jobs"]["platform-fragment"]["steps"]
    by_name = {step["name"]: step for step in steps}
    assert by_name["Set up exact TeX Live 2026"]["if"] == "matrix.stage == 'pdf'"
    assert by_name["Execute frozen mutation-test matrix"]["if"] == "matrix.stage == 'mutation'"
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "evidence/stage-summary.json" in workflow
    assert "${{ matrix.stage }}-run-${{ github.run_id }}" in workflow
    assert '[ValidateSet("candidate", "pdf", "mutation")]' in runner
    assert 'if ($ExecutionStage -eq "pdf")' in runner
    assert 'if ($ExecutionStage -in @("candidate", "mutation"))' in runner
    assert "texlive-closure-before.json" in runner
    assert "texlive-closure-after.json" in runner
    assert '"-no-shell-escape"' in runner
    for token in ("TEXMF*", "KPATHSEA*", "FONTCONFIG*", "LD_*", "DYLD_*"):
        assert token in runner


@pytest.mark.negative_control
def test_r3_workflow_requires_production_containment_and_atomic_subject_capture() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    containment = CONTAINMENT.read_text(encoding="utf-8")
    for token in (
        "CreateProcessAsUser",
        "CREATE_SUSPENDED",
        "AssignProcessToJobObject(job, pi.hProcess)",
        "TerminateJobObject",
        "ActiveProcessesAfterTermination",
        "User=$serviceUser",
        "KillMode=control-group",
        "NoNewPrivileges=yes",
        "cgroup.procs",
    ):
        assert token in containment
    assert "Prove production containment against detached replacement payload" in workflow
    assert "child-of-child running" in workflow
    assert "replace-then-restore" in workflow
    assert "Capture exact attestation subjects after containment teardown" in workflow
    assert workflow.count("attestation subjects changed") == 2
    assert workflow.index("Capture exact attestation subjects after containment teardown") < (
        workflow.index("Attest exact platform subjects")
    )


@pytest.mark.negative_control
def test_r3_safe_hosted_containment_proof_path_is_bound_and_exact_2x3(
    tmp_path: Path,
) -> None:
    workflow_text = CONTAINMENT_PROOF_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    trigger = workflow[True]
    assert set(trigger) == {"push"}
    assert trigger["push"]["branches"] == [
        "campaign/via000-r3-protocol-*",
        "review/via000-r3-protocol-*",
    ]
    assert workflow["permissions"] == {"contents": "read"}
    assert "workflow_dispatch" not in workflow_text
    for forbidden in (
        "id-token:",
        "secrets.",
        "environment:",
        "via-000-assembler",
        "via-000-dispatch-guard",
        "authorization",
        "custody",
        "output-commitment",
        "reveal",
        "5be3c38a0822d49953d0933f14ccab32ca12c896",
        "9a29e05f803666bf0e3a28417ea399e3e26769fc",
    ):
        assert forbidden not in workflow_text.lower()
    proof_jobs = {
        "ubuntu-candidate": ("ubuntu-24.04", "ubuntu_candidate_digest"),
        "ubuntu-pdf": ("ubuntu-24.04", "ubuntu_pdf_digest"),
        "ubuntu-mutation": ("ubuntu-24.04", "ubuntu_mutation_digest"),
        "windows-candidate": ("windows-2025", "windows_candidate_digest"),
        "windows-pdf": ("windows-2025", "windows_pdf_digest"),
        "windows-mutation": ("windows-2025", "windows_mutation_digest"),
    }
    for job_name, (runner, output_name) in proof_jobs.items():
        job = workflow["jobs"][job_name]
        assert job["runs-on"] == runner
        assert set(job["outputs"]) == {output_name}
        assert job["outputs"][output_name] == f"${{{{ steps.digest.outputs.{output_name} }}}}"
    assert "strategy" not in workflow_text
    assert set(workflow["jobs"]["aggregate"]["needs"]) == set(proof_jobs)
    for action in (
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
        "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
    ):
        assert action in workflow_text
    assert workflow_text.count("shell: pwsh") == 15
    assert "WindowsPowerShell" not in workflow_text
    assert "-NonInteractive -Command \". '{0}'\"" not in workflow_text
    assert workflow_text.count('$workspace = "/tmp/via000-proof-workspace"') == 1
    assert workflow_text.count('$workspace = Join-Path $root "via000-proof-workspace"') == 1
    assert workflow_text.count('$relative = ".via000-r3-proof-cache/') >= 3
    assert "-OutputName" not in workflow_text
    assert "VIA000_ENVELOPE_" not in workflow_text
    assert workflow_text.count(
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
    ) == 1
    assert "--collect-cache-envelopes" in workflow_text
    assert "--verify-retained" in workflow_text
    assert "via000-r3-containment-proof-retained" in workflow_text
    assert "via000-proof-output" not in workflow_text
    assert "if ((Test-Path -LiteralPath $WorkspaceRoot) -or" in (
        CONTAINMENT_PROOF_RUNNER.read_text(encoding="utf-8")
    )
    runner_text = CONTAINMENT_PROOF_RUNNER.read_text(encoding="utf-8")
    fixture_text = CONTAINMENT_PROOF_FIXTURE.read_text(encoding="utf-8")
    assert "Invoke-Via000ContainedCommand" in runner_text
    assert "mutable hostile fixture copy differs from the frozen Git bytes" in runner_text
    assert "synthetic containment diagnostic" not in runner_text
    assert "SerializeToUtf8Bytes" in runner_text
    assert 'Join-Path $repoItem.FullName ".via000-r3-proof-cache"' in runner_text
    assert "live proof subject inner hash binding differs before export" in runner_text
    assert "Get-Item -LiteralPath $full -Stream *" in runner_text
    assert "LinkCount($full)" in runner_text
    assert "Set-Via000WindowsExportSecurity -Path $OutputRoot" in runner_text
    assert "Set-Via000WindowsExportFileOwner -Path $verifiedEnvelope" in runner_text
    assert "Windows export native root/file descriptor differs after hash capture" in runner_text
    assert "MaximumProofEnvelopeBytes" in runner_text
    assert "Protect-Via000ReadOnlyClosure -Path $OutputRoot" not in runner_text
    assert "Assert-FrozenProofBundle" in runner_text
    assert "child-of-child-ready" in runner_text
    assert "delayed-descendant-survived" in runner_text
    assert "replace-restore-succeeded" in fixture_text
    assert "hardlink-substitution-succeeded" in fixture_text
    assert '"GITHUB_*", "ACTIONS_*", "RUNNER_*"' in CONTAINMENT.read_text(encoding="utf-8")
    assert '$active -in @("inactive", "failed")' in CONTAINMENT.read_text(
        encoding="utf-8"
    )
    containment_text = CONTAINMENT.read_text(encoding="utf-8")
    assert '"--property=PrivateTmp=no"' in containment_text
    assert '"--property=ProtectSystem=no"' in containment_text
    assert '"--property=ProtectHome=no"' in containment_text
    assert "could not create fresh untrusted service identity" in containment_text
    assert "fresh untrusted service identity retained a process" in containment_text
    assert "could not remove fresh untrusted service identity" in containment_text
    evidence_write = containment_text.index(
        "Write-Via000CanonicalJsonObject -Document $record -Path $ResultPath"
    )
    final_uid_check = containment_text.index(
        "Assert-Via000UidQuiescent -Uid $serviceUid", evidence_write
    )
    identity_delete = containment_text.index(
        "([string]$SystemTools.userdel) $serviceUser", final_uid_check
    )
    assert evidence_write < final_uid_check < identity_delete
    assert "Start-Descendant -ChildMode \"relay\"" in fixture_text
    assert "Start-Descendant -ChildMode \"writer\"" in fixture_text

    source_sha = "a" * 40
    source_ref = "refs/heads/campaign/via000-r3-protocol-proof-test"
    workflow_ref = f"whact2025/POPGP/.github/workflows/via000-r3-containment-proof.yml@{source_ref}"
    fragments = tmp_path / "fragments"
    schema = json.loads(CONTAINMENT_PROOF_SCHEMA.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    envelope_schema = json.loads(CONTAINMENT_PROOF_ENVELOPE_SCHEMA.read_text(encoding="utf-8"))
    envelope_validator = Draft202012Validator(envelope_schema)
    aggregator_module = _load_module(CONTAINMENT_PROOF_AGGREGATOR, "r3_proof_aggregator")
    true_fields = {
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
    hash_fields = {
        "production_helper_sha256",
        "proof_runner_sha256",
        "hostile_fixture_sha256",
        "proof_schema_sha256",
        "proof_envelope_schema_sha256",
        "proof_aggregator_sha256",
        "proof_workflow_sha256",
        "protected_evidence_sha256",
        "protected_tool_sha256",
    }
    for platform in ("ubuntu-latest-x86_64", "windows-x86_64"):
        primitive = (
            "ubuntu-systemd-ephemeral-user-control-group"
            if platform.startswith("ubuntu")
            else "windows-low-integrity-restricted-token-job-object"
        )
        privilege = (
            "systemd-ephemeral-user"
            if platform.startswith("ubuntu")
            else "low-integrity-restricted-token"
        )
        for stage in ("candidate", "pdf", "mutation"):
            artifact_name = f"via000-r3-containment-proof-{platform}-{stage}"
            artifact = fragments / artifact_name
            artifact.mkdir(parents=True)
            subjects = {"stdout.txt": b"", "stderr.txt": b""}
            contained = {
                "schema_version": 1,
                "label": f"proof-{stage}",
                "contract_id": "rr7-hosted-containment-proof",
                "primitive": primitive,
                "privilege_separation": privilege,
                "descendants_quiescent": True,
                "active_processes_after_teardown": 0,
                "exit_code": 0,
                "timed_out": False,
                **_token_evidence(platform),
                "stdout_sha256": hashlib.sha256(subjects["stdout.txt"]).hexdigest(),
                "stderr_sha256": hashlib.sha256(subjects["stderr.txt"]).hexdigest(),
            }
            if platform.startswith("ubuntu"):
                contained.update(
                    {
                        "ephemeral_identity_uid": "999",
                        "ephemeral_identity_processes_empty": True,
                        "ephemeral_identity_removed": True,
                    }
                )
            subjects["containment-result.json"] = (
                json.dumps(contained, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            proof = {
                "schema_version": 1,
                "proof_kind": "via000-r3-hosted-containment-cell",
                "repository": "whact2025/POPGP",
                "workflow": ".github/workflows/via000-r3-containment-proof.yml",
                "workflow_ref": workflow_ref,
                "event_name": "push",
                "source_ref": source_ref,
                "source_sha": source_sha,
                "run_id": "424242",
                "run_attempt": "1",
                "platform_family": platform,
                "stage_id": stage,
                "cell": f"{platform}/{stage}",
                "artifact_name": artifact_name,
                "primitive": primitive,
                "privilege_separation": privilege,
                **_token_evidence(platform),
                **_export_evidence(platform),
                "active_processes_after_teardown": 0,
                "containment_result_sha256": hashlib.sha256(
                    subjects["containment-result.json"]
                ).hexdigest(),
                **{field: True for field in true_fields},
                **{field: "b" * 64 for field in hash_fields},
            }
            validator.validate(proof)
            subjects["proof.json"] = (
                json.dumps(proof, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            identity = {
                field: proof[field]
                for field in (
                    "artifact_name",
                    "cell",
                    "event_name",
                    "platform_family",
                    "repository",
                    "run_attempt",
                    "run_id",
                    "source_ref",
                    "source_sha",
                    "stage_id",
                    "workflow",
                    "workflow_ref",
                )
            }
            envelope = aggregator_module.build_envelope(identity, subjects)
            envelope_document = json.loads(envelope)
            envelope_validator.validate(envelope_document)
            (artifact / "envelope.json").write_bytes(envelope)

    output = tmp_path / "aggregate.json"
    command = [
        sys.executable,
        "-I",
        "-S",
        str(CONTAINMENT_PROOF_AGGREGATOR),
        "--input-root",
        str(fragments),
        "--output",
        str(output),
        "--source-sha",
        source_sha,
        "--source-ref",
        source_ref,
        "--workflow-ref",
        workflow_ref,
        "--run-id",
        "424242",
        "--run-attempt",
        "1",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    validator.validate(aggregate)
    assert aggregate["cell_count"] == 6
    assert len(aggregate["fragment_sha256"]) == 6

    ubuntu_artifact = fragments / "via000-r3-containment-proof-ubuntu-latest-x86_64-candidate"
    original_envelope = (ubuntu_artifact / "envelope.json").read_bytes()

    def reject_envelope(content: bytes) -> None:
        (ubuntu_artifact / "envelope.json").write_bytes(content)
        output.unlink(missing_ok=True)
        malformed = subprocess.run(command, check=False, capture_output=True, text=True)
        assert malformed.returncode != 0
        assert not output.exists()
        (ubuntu_artifact / "envelope.json").write_bytes(original_envelope)

    reject_envelope(b"{")
    reject_envelope(b"\xef\xbb\xbf" + original_envelope)
    reject_envelope(original_envelope.replace(b"\n", b"\r\n"))
    reject_envelope(original_envelope + b" " * 1_048_576)
    document = json.loads(original_envelope)
    reject_envelope((json.dumps(document, indent=2) + "\n").encode())
    reject_envelope(
        original_envelope.replace(
            b'{"envelope_kind":', b'{"schema_version":1,"envelope_kind":', 1
        )
    )
    for mutation in ("extra", "missing", "case", "base64", "size", "hash", "total"):
        document = json.loads(original_envelope)
        if mutation == "extra":
            document["members"]["extra.txt"] = document["members"]["stdout.txt"]
        elif mutation == "missing":
            del document["members"]["stderr.txt"]
        elif mutation == "case":
            document["members"]["Proof.json"] = document["members"]["proof.json"]
        elif mutation == "base64":
            document["members"]["stdout.txt"]["base64"] = "!"
        elif mutation == "size":
            document["members"]["stdout.txt"]["size"] = 1
        elif mutation == "hash":
            document["members"]["stdout.txt"]["sha256"] = "0" * 64
        else:
            document["total_decoded_bytes"] += 1
        reject_envelope(aggregator_module.canonical_envelope_bytes(document))

    document = json.loads(original_envelope)
    proof_bytes = base64.b64decode(document["members"]["proof.json"]["base64"])
    inner_proof = json.loads(proof_bytes)
    inner_proof["containment_result_sha256"] = "0" * 64
    bad_proof = json.dumps(inner_proof, separators=(",", ":")).encode() + b"\n"
    document["members"]["proof.json"] = {
        "base64": base64.b64encode(bad_proof).decode(),
        "sha256": hashlib.sha256(bad_proof).hexdigest(),
        "size": len(bad_proof),
    }
    document["total_decoded_bytes"] += len(bad_proof) - len(proof_bytes)
    reject_envelope(aggregator_module.canonical_envelope_bytes(document))

    hardlink_source = tmp_path / "hardlink-envelope.json"
    hardlink_source.write_bytes(original_envelope)
    (ubuntu_artifact / "envelope.json").unlink()
    os.link(hardlink_source, ubuntu_artifact / "envelope.json")
    output.unlink(missing_ok=True)
    hardlink = subprocess.run(command, check=False, capture_output=True, text=True)
    assert hardlink.returncode != 0
    assert not output.exists()
    (ubuntu_artifact / "envelope.json").unlink()
    hardlink_source.unlink()
    (ubuntu_artifact / "envelope.json").write_bytes(original_envelope)

    envelope_document = json.loads(original_envelope)
    envelope_document["identity"]["source_sha"] = "c" * 40
    (ubuntu_artifact / "envelope.json").write_bytes(
        aggregator_module.canonical_envelope_bytes(envelope_document)
    )
    output.unlink(missing_ok=True)
    stale_identity = subprocess.run(command, check=False, capture_output=True, text=True)
    assert stale_identity.returncode != 0
    assert not output.exists()
    (ubuntu_artifact / "envelope.json").write_bytes(original_envelope)

    windows_artifact = (
        fragments / "via000-r3-containment-proof-windows-x86_64-candidate"
    )
    windows_original = (windows_artifact / "envelope.json").read_bytes()
    for field, value in (
        ("token_restriction_flags", []),
        ("token_integrity_sid", "S-1-16-8192"),
        ("enabled_privilege_count", 1),
        ("enabled_privileges", ["SeDebugPrivilege"]),
        ("enabled_privileges", ["SeChangeNotifyPrivilege", "SeDebugPrivilege"]),
        ("protected_label_policy", "mutable"),
        ("export_owner_sid", ""),
        ("export_dacl_policy", "owner-rwx-0700-v1"),
        ("export_integrity_sid", "S-1-16-4096"),
        ("export_mandatory_policy", "owner-only"),
        ("export_root_control_flags", 33796),
        ("export_root_dacl_protected", False),
        ("export_root_native_ace_count", 2),
        ("export_root_managed_ace_count", 2),
    ):
        document = json.loads(windows_original)
        proof_bytes = base64.b64decode(document["members"]["proof.json"]["base64"])
        proof = json.loads(proof_bytes)
        proof[field] = value
        replacement = json.dumps(proof, separators=(",", ":")).encode() + b"\n"
        document["members"]["proof.json"] = {
            "base64": base64.b64encode(replacement).decode(),
            "sha256": hashlib.sha256(replacement).hexdigest(),
            "size": len(replacement),
        }
        document["total_decoded_bytes"] += len(replacement) - len(proof_bytes)
        (windows_artifact / "envelope.json").write_bytes(
            aggregator_module.canonical_envelope_bytes(document)
        )
        output.unlink(missing_ok=True)
        forged = subprocess.run(command, check=False, capture_output=True, text=True)
        assert forged.returncode != 0
        assert not output.exists()
        (windows_artifact / "envelope.json").write_bytes(windows_original)

    removed = next(fragments.glob("*/envelope.json"))
    removed.unlink()
    rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert not output.exists()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["parameters"]["containment_protocol_sha256"] = "0" * 64
    bad_protocol = tmp_path / "bad-protocol.json"
    _write_json(bad_protocol, protocol)
    workspace = tmp_path / "should-not-exist-workspace"
    proof_runner_temp = tmp_path / "proof-runner-temp"
    proof_runner_temp.mkdir()
    proof_output = ROOT / ".via000-r3-proof-cache/windows-x86_64/candidate"
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    bundle_rejection = subprocess.run(
        [
            pwsh,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(CONTAINMENT_PROOF_RUNNER),
            "-PlatformFamily",
            "windows-x86_64",
            "-StageId",
            "candidate",
            "-RepoRoot",
            str(ROOT),
            "-ProtocolPath",
            str(bad_protocol),
            "-WorkspaceRoot",
            str(workspace),
            "-RunnerTemp",
            str(proof_runner_temp),
            "-OutputRoot",
            str(proof_output),
            "-PowerShellPath",
            str(Path(pwsh).resolve()),
            "-Repository",
            "whact2025/POPGP",
            "-EventName",
            "push",
            "-SourceRef",
            source_ref,
            "-SourceSha",
            source_sha,
            "-WorkflowRef",
            workflow_ref,
            "-RunId",
            "424242",
            "-RunAttempt",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert bundle_rejection.returncode != 0
    assert not workspace.exists()
    assert not proof_output.exists()


@pytest.mark.negative_control
def test_r3_rr8_windows_proof_export_is_canonical_and_exact_2x3(tmp_path: Path) -> None:
    """Stable RR8 entry point for the production-path envelope adversarial suite."""
    test_r3_safe_hosted_containment_proof_path_is_bound_and_exact_2x3(tmp_path)


def _exercise_r3_digest_cache_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_r3_safe_hosted_containment_proof_path_is_bound_and_exact_2x3(tmp_path)
    module = _load_module(CONTAINMENT_PROOF_AGGREGATOR, "r3_rr9_transport")
    fragments = tmp_path / "fragments"
    missing = next(path for path in fragments.iterdir() if not (path / "envelope.json").exists())
    donor_path = next(fragments.glob("*/envelope.json"))
    donor = json.loads(donor_path.read_bytes())
    platform_stage = missing.name.removeprefix("via000-r3-containment-proof-")
    if platform_stage.startswith("ubuntu-latest-x86_64-"):
        platform = "ubuntu-latest-x86_64"
    else:
        platform = "windows-x86_64"
    stage = platform_stage.removeprefix(f"{platform}-")
    subjects = {
        name: base64.b64decode(metadata["base64"])
        for name, metadata in donor["members"].items()
    }
    contained = json.loads(subjects["containment-result.json"])
    contained.update(
        {
            "label": f"proof-{stage}",
            "primitive": (
                "ubuntu-systemd-ephemeral-user-control-group"
                if platform.startswith("ubuntu")
                else "windows-low-integrity-restricted-token-job-object"
            ),
            "privilege_separation": (
                "systemd-ephemeral-user"
                if platform.startswith("ubuntu")
                else "low-integrity-restricted-token"
            ),
            **_token_evidence(platform),
        }
    )
    if platform.startswith("ubuntu"):
        contained.update(
            {
                "ephemeral_identity_uid": "999",
                "ephemeral_identity_processes_empty": True,
                "ephemeral_identity_removed": True,
            }
        )
    else:
        for field in (
            "ephemeral_identity_uid",
            "ephemeral_identity_processes_empty",
            "ephemeral_identity_removed",
        ):
            contained.pop(field, None)
    subjects["containment-result.json"] = (
        json.dumps(contained, separators=(",", ":")).encode() + b"\n"
    )
    proof = json.loads(subjects["proof.json"])
    artifact = f"via000-r3-containment-proof-{platform}-{stage}"
    proof.update(
        {
            "artifact_name": artifact,
            "cell": f"{platform}/{stage}",
            "platform_family": platform,
            "stage_id": stage,
            "primitive": contained["primitive"],
            "privilege_separation": contained["privilege_separation"],
            **_token_evidence(platform),
            **_export_evidence(platform),
            "containment_result_sha256": hashlib.sha256(
                subjects["containment-result.json"]
            ).hexdigest(),
        }
    )
    subjects["proof.json"] = (
        json.dumps(proof, separators=(",", ":")).encode() + b"\n"
    )
    identity = {
        field: proof[field]
        for field in (
            "artifact_name",
            "cell",
            "event_name",
            "platform_family",
            "repository",
            "run_attempt",
            "run_id",
            "source_ref",
            "source_sha",
            "stage_id",
            "workflow",
            "workflow_ref",
        )
    }
    (missing / "envelope.json").write_bytes(module.build_envelope(identity, subjects))

    cache_root = tmp_path / ".via000-r3-proof-cache"
    environment: dict[str, str] = {}
    for variable, (cell_platform, cell_stage) in module.DIGESTS.items():
        source = (
            fragments
            / f"via000-r3-containment-proof-{cell_platform}-{cell_stage}"
            / "envelope.json"
        )
        target_root = cache_root / cell_platform / cell_stage
        target_root.mkdir(parents=True)
        target = target_root / "envelope.json"
        shutil.copyfile(source, target)
        environment[variable] = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        source_sha="a" * 40,
        source_ref="refs/heads/campaign/via000-r3-protocol-proof-test",
        workflow_ref=(
            "whact2025/POPGP/.github/workflows/via000-r3-containment-proof.yml@"
            "refs/heads/campaign/via000-r3-protocol-proof-test"
        ),
        run_id="424242",
        run_attempt="1",
        output_root=tmp_path / "retained",
    )
    aggregate = module.collect_cache_envelopes(args, environment)
    assert aggregate["cell_count"] == 6
    assert {path.name for path in args.output_root.iterdir()} == {
        "aggregate.json",
        *{
            f"envelope-{cell_platform}-{cell_stage}.json"
            for cell_platform in module.PLATFORMS
            for cell_stage in module.STAGES
        },
    }
    assert module.verify_retained(args.output_root, module._expected_identity(args)) == aggregate
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    for envelope_path in args.output_root.glob("envelope-*.json"):
        envelope = json.loads(envelope_path.read_bytes())
        for transcript in ("stdout.txt", "stderr.txt"):
            assert envelope["members"][transcript] == {
                "base64": "",
                "sha256": empty_sha256,
                "size": 0,
            }

    def reject_transport(label: str, changed: dict[str, str]) -> None:
        rejected_args = SimpleNamespace(**vars(args))
        rejected_args.output_root = tmp_path / f"rejected-{label}"
        with pytest.raises(ValueError):
            module.collect_cache_envelopes(rejected_args, changed)
        assert not rejected_args.output_root.exists()

    variables = list(module.DIGESTS)
    missing_output = dict(environment)
    del missing_output[variables[0]]
    reject_transport("missing", missing_output)
    for label, value in (
        ("empty", ""),
        ("truncated", environment[variables[0]][:-1]),
        ("masked", "***"),
        ("newline", environment[variables[0]] + "\n"),
        ("injected", environment[variables[0]] + ";marker"),
        ("nonhex", environment[variables[0]][:-1] + "!"),
        ("uppercase", environment[variables[0]].upper()),
    ):
        changed = dict(environment)
        changed[variables[0]] = value
        reject_transport(label, changed)
    duplicate = dict(environment)
    duplicate[variables[1]] = duplicate[variables[0]]
    reject_transport("duplicate", duplicate)
    crossed = dict(environment)
    crossed[variables[0]], crossed[variables[-1]] = crossed[variables[-1]], crossed[variables[0]]
    reject_transport("cross-cell", crossed)

    first_platform, first_stage = module.DIGESTS[variables[0]]
    first_cache = cache_root / first_platform / first_stage / "envelope.json"
    original_cache = first_cache.read_bytes()

    def reject_cache(label: str, mutate: object, changed: dict[str, str] | None = None) -> None:
        mutate()
        try:
            reject_transport(label, environment if changed is None else changed)
        finally:
            if first_cache.is_symlink() or first_cache.exists():
                first_cache.unlink()
            first_cache.write_bytes(original_cache)

    reject_cache("corrupt-cache", lambda: first_cache.write_bytes(b"{}\n"))
    reject_cache("oversized-cache", lambda: first_cache.write_bytes(b"x" * 131073))
    hardlink_cache_source = tmp_path / "cache-hardlink-source.json"
    hardlink_cache_source.write_bytes(original_cache)

    def hardlink_cache() -> None:
        first_cache.unlink()
        os.link(hardlink_cache_source, first_cache)

    reject_cache("hardlink-cache", hardlink_cache)
    extra_cache = first_cache.parent / "extra.json"

    def extra_case_collision() -> None:
        extra_cache.write_bytes(original_cache)

    with pytest.raises(ValueError):
        extra_case_collision()
        module.collect_cache_envelopes(
            SimpleNamespace(**{**vars(args), "output_root": tmp_path / "rejected-case-cache"}),
            environment,
        )
    extra_cache.unlink()

    last_platform, last_stage = module.DIGESTS[variables[-1]]
    last_cache = cache_root / last_platform / last_stage / "envelope.json"
    last_bytes = last_cache.read_bytes()
    first_cache.write_bytes(last_bytes)
    last_cache.write_bytes(original_cache)
    swapped = dict(environment)
    swapped[variables[0]] = hashlib.sha256(last_bytes).hexdigest()
    swapped[variables[-1]] = hashlib.sha256(original_cache).hexdigest()
    reject_transport("cache-cross-cell", swapped)
    first_cache.write_bytes(original_cache)
    last_cache.write_bytes(last_bytes)

    def reject_retained(label: str, mutate: object) -> None:
        root = tmp_path / f"retained-{label}"
        shutil.copytree(args.output_root, root)
        mutate(root)
        with pytest.raises(ValueError):
            module.verify_retained(root, module._expected_identity(args))

    envelope_name = next(args.output_root.glob("envelope-*.json")).name
    reject_retained("missing", lambda root: (root / envelope_name).unlink())
    reject_retained("extra", lambda root: (root / "extra.json").write_bytes(b"{}\n"))
    reject_retained(
        "case-collision", lambda root: (root / "Aggregate.json").write_bytes(b"{}\n")
    )
    reject_retained(
        "corrupt", lambda root: (root / envelope_name).write_bytes(b"{\"bad\":true}\n")
    )
    reject_retained(
        "oversized", lambda root: (root / envelope_name).write_bytes(b"x" * 131073)
    )
    reject_retained(
        "aggregate-hash",
        lambda root: (root / "aggregate.json").write_bytes(
            (root / "aggregate.json").read_bytes() + b" "
        ),
    )
    hardlink_source = tmp_path / "retained-hardlink-source.json"
    hardlink_source.write_bytes((args.output_root / envelope_name).read_bytes())

    def substitute_hardlink(root: Path) -> None:
        target = root / envelope_name
        target.unlink()
        os.link(hardlink_source, target)

    reject_retained("hardlink", substitute_hardlink)

    workflow_text = CONTAINMENT_PROOF_WORKFLOW.read_text(encoding="utf-8")
    runner_text = CONTAINMENT_PROOF_RUNNER.read_text(encoding="utf-8")
    assert workflow_text.count("outputs:\n") == 7
    assert "needs.containment" not in workflow_text
    assert workflow_text.count("actions/upload-artifact@") == 1
    assert "VIA000_ENVELOPE_" not in workflow_text
    assert "ToBase64String($envelope)" not in workflow_text
    assert workflow_text.count(
        "actions/cache/save@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"
    ) == 6
    assert workflow_text.count(
        "actions/cache/restore@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"
    ) == 12
    assert workflow_text.count("enableCrossOsArchive: true") == 18
    assert workflow_text.count("lookup-only: true") == 6
    assert workflow_text.count("fail-on-cache-miss: true") == 6
    assert "restore-keys:" not in workflow_text
    assert "cache-primary-key" in workflow_text
    assert "cache-matched-key" in workflow_text
    assert "exact digest-bound cache key already exists" in workflow_text
    assert "six-cache proof consolidation failed" in workflow_text
    assert r"shell: C:\Program Files\PowerShell" not in workflow_text
    assert "WindowsPowerShell" not in workflow_text
    workflow_document = yaml.safe_load(workflow_text)
    for job_name in ("windows-candidate", "windows-pdf", "windows-mutation"):
        trusted_steps = [
            step for step in workflow_document["jobs"][job_name]["steps"] if "run" in step
        ]
        assert len(trusted_steps) == 5
        assert all(step["shell"] == "pwsh" for step in trusted_steps)
        assert all(
            "trusted Windows built-in pwsh identity, profile, version, or PATH differs"
            in step["run"]
            for step in trusted_steps
        )
    assert "ConvertFrom-Json -AsHashtable" not in workflow_text
    assert "[Security.Cryptography.SHA256]::HashData" not in workflow_text
    assert workflow_text.index("$tarExit = $LASTEXITCODE") < workflow_text.index(
        "$zstdOutput = @(& $zstd --version 2>&1)"
    )
    assert workflow_text.index("$zstdOutput = @(& $zstd --version 2>&1)") < (
        workflow_text.index("$zstdExit = $LASTEXITCODE")
    )
    tar_banner = re.compile(r"^tar \(GNU tar\) [0-9]+\.[0-9]+(?:\.[0-9]+)?")
    zstd_banner = re.compile(r"Zstandard.*\bv?[0-9]+\.[0-9]+\.[0-9]+\b")
    assert tar_banner.search("tar (GNU tar) 1.35")
    assert zstd_banner.search("*** Zstandard CLI (64-bit) v1.5.7, by Yann Collet ***")
    assert "${{ github.workflow_sha }}" in workflow_text
    assert "${{ github.repository_id }}" in workflow_text
    assert ".via000-r3-proof-cache/windows-x86_64/candidate" in workflow_text
    assert "[AllowEmptyCollection()]" in runner_text
    assert "$script:MaximumProofEnvelopeBytes = 131072" in runner_text
    assert "MaximumProofEnvelopeBase64Characters" not in runner_text
    assert "Write-ProofJobOutput" not in runner_text
    assert 'Join-Path $repoItem.FullName ".via000-r3-proof-cache"' in runner_text
    assert "exact cell-specific workspace-relative path" in runner_text
    assert '"GITHUB_*", "ACTIONS_*", "RUNNER_*"' in CONTAINMENT.read_text(
        encoding="utf-8"
    )

    if os.name == "nt":
        pwsh = shutil.which("pwsh")
        assert pwsh is not None
        local_protocol_dir = tmp_path / "protocols/POPGP-VIABILITY-R3-2026-08"
        local_protocol_dir.mkdir(parents=True)
        local_helper = local_protocol_dir / "VIA-000-CONTAINMENT.ps1"
        shutil.copyfile(CONTAINMENT, local_helper)
        (local_protocol_dir / "VIA-000.json").write_text(
            json.dumps(
                {
                    "parameters": {
                        "containment_protocol_sha256": hashlib.sha256(
                            local_helper.read_bytes()
                        ).hexdigest()
                    }
                }
            ),
            encoding="utf-8",
        )
        live_export = cache_root / "windows-x86_64/candidate"
        live_envelope = live_export / "envelope.json"
        live_bytes = live_envelope.read_bytes()
        live_envelope.unlink()
        secure_script = tmp_path / "rr16-secure-live-export.ps1"
        secure_script.write_text(
            "param([string]$Helper,[string]$Root)\n"
            + "$ErrorActionPreference='Stop'; . $Helper\n"
            + "$null=Set-Via000WindowsExportSecurity -Path $Root\n",
            encoding="utf-8",
            newline="\n",
        )
        secured = subprocess.run(
            [pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(secure_script),
             str(local_helper), str(live_export)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert secured.returncode == 0, secured.stderr
        live_envelope.write_bytes(live_bytes)
        local_trusted_path = f"{Path(pwsh).parent};C:\\Windows\\System32;C:\\Windows"
        digest_step = next(
            step
            for step in workflow_document["jobs"]["windows-candidate"]["steps"]
            if step.get("id") == "digest"
        )
        digest_script = tmp_path / "rr10-outer-digest.ps1"
        digest_script.write_text(
            digest_step["run"]
            .replace(r"C:\Program Files\PowerShell\7\pwsh.exe", pwsh)
            .replace(
                r"C:\Program Files\PowerShell\7;C:\tools\zstd;"
                r"C:\Program Files\Git\usr\bin;"
                r"C:\Program Files\Git\mingw64\bin;C:\Program Files\Git\cmd;"
                r"C:\Windows\System32;C:\Windows",
                local_trusted_path,
            ),
            encoding="utf-8",
            newline="\n",
        )
        github_output = tmp_path / "rr10-github-output.txt"
        github_output.write_bytes(b"")
        live_environment = os.environ.copy()
        live_environment.update(
            {
                "PATH": local_trusted_path,
                "GITHUB_WORKSPACE": str(tmp_path),
                "GITHUB_OUTPUT": str(github_output),
                "VIA000_PLATFORM_FAMILY": "windows-x86_64",
                "VIA000_STAGE_ID": "candidate",
                "VIA000_DIGEST_OUTPUT_NAME": "windows_candidate_digest",
                "VIA000_SOURCE_SHA": "a" * 40,
                "VIA000_RUN_ID": "424242",
                "VIA000_RUN_ATTEMPT": "1",
            }
        )
        completed = subprocess.run(
            [
                pwsh,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-File",
                str(digest_script),
            ],
            cwd=tmp_path,
            env=live_environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert github_output.read_bytes() == (
            f"windows_candidate_digest={environment['VIA000_DIGEST_WINDOWS_CANDIDATE']}\n"
        ).encode("ascii")


@pytest.mark.negative_control
def test_r3_rr9_canonical_envelope_transport_is_exact_and_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prior RR9 behavior remains covered through the superseding cache transport."""
    _exercise_r3_digest_cache_transport(tmp_path, monkeypatch)


@pytest.mark.negative_control
def test_r3_rr10_job_output_cache_transport_is_digest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001."""
    _exercise_r3_digest_cache_transport(tmp_path, monkeypatch)


@pytest.mark.negative_control
def test_r3_rr12_windows_builtin_pwsh_execution_is_observable_and_persistent(
    tmp_path: Path,
) -> None:
    """TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001."""
    proof_text = CONTAINMENT_PROOF_WORKFLOW.read_text(encoding="utf-8")
    production_text = WORKFLOW.read_text(encoding="utf-8")
    proof = yaml.safe_load(proof_text)
    production = yaml.safe_load(production_text)
    proof_expected_path = (
        r"C:\Program Files\PowerShell\7;C:\tools\zstd;"
        r"C:\Program Files\Git\usr\bin;"
        r"C:\Program Files\Git\mingw64\bin;C:\Program Files\Git\cmd;"
        r"C:\Windows\System32;C:\Windows"
    )
    production_expected_path = (
        r"C:\Program Files\PowerShell\7;C:\Program Files\Git\usr\bin;"
        r"C:\Program Files\Git\mingw64\bin;C:\Program Files\Git\cmd;"
        r"C:\Windows\System32;C:\Windows"
    )
    forbidden_patterns = (
        "WindowsPowerShell",
        r'"C:\Program Files\PowerShell\7\pwsh.exe"',
        "-ExecutionPolicy Bypass -File {0}",
        "-NonInteractive -Command \". '{0}'\"",
    )
    proof_shell_lines = "\n".join(
        line for line in proof_text.splitlines() if "shell:" in line
    )
    production_shell_lines = "\n".join(
        line for line in production_text.splitlines() if "shell:" in line
    )
    for pattern in forbidden_patterns:
        assert pattern not in proof_shell_lines
        assert pattern not in production_shell_lines

    proof_step_names = {
        "Prove contained Windows candidate and stage one envelope",
        "Prove contained Windows pdf and stage one envelope",
        "Prove contained Windows mutation and stage one envelope",
        "Validate staged envelope and emit digest only",
        "Assert exact cross-OS archive tools",
        "Require fresh exact cache key",
        "Recheck archive tool and staged envelope after cache save",
    }
    proof_step_count = 0
    for job_name in ("windows-candidate", "windows-pdf", "windows-mutation"):
        job = proof["jobs"][job_name]
        assert job["env"]["PATH"] == proof_expected_path
        trusted_steps = [step for step in job["steps"] if "run" in step]
        assert len(trusted_steps) == 5
        for step in trusted_steps:
            proof_step_count += 1
            assert step["shell"] == "pwsh"
            assert step["name"] in proof_step_names
            assert (
                "trusted Windows built-in pwsh identity, profile, version, or PATH differs"
                in step["run"]
            )
    assert proof_step_count == 15

    windows_matrix = [
        entry
        for entry in production["jobs"]["platform-fragment"]["strategy"]["matrix"][
            "include"
        ]
        if entry["platform"] == "windows-x86_64"
    ]
    assert len(windows_matrix) == 3
    assert all("shell" not in entry for entry in windows_matrix)
    assert all(
        entry["trusted_path"] == production_expected_path for entry in windows_matrix
    )
    ubuntu_matrix = [
        entry
        for entry in production["jobs"]["platform-fragment"]["strategy"]["matrix"][
            "include"
        ]
        if entry["platform"] == "ubuntu-latest-x86_64"
    ]
    assert len(ubuntu_matrix) == 3
    assert all("shell" not in entry for entry in ubuntu_matrix)
    assert all(
        entry["trusted_path"] == "/opt/microsoft/powershell/7:/usr/bin:/bin"
        for entry in ubuntu_matrix
    )
    assert production["jobs"]["platform-fragment"]["env"]["PATH"] == (
        "${{ matrix.trusted_path }}"
    )
    production_step_names = {
        "Reject non-snapshot lifecycle refs",
        "Validate complete trusted toolchain",
        "Prove production containment against detached replacement payload",
        "Execute frozen clean platform protocol",
        "Execute frozen mutation-test matrix",
        "Capture exact attestation subjects after containment teardown",
        "Retain producer attestation bundle",
        "Remove rejected platform workspace",
    }
    production_steps = [
        step
        for step in production["jobs"]["platform-fragment"]["steps"]
        if step.get("name") in production_step_names
    ]
    assert {step["name"] for step in production_steps} == production_step_names
    assert len(production_steps) == 8
    assert all(step["shell"] == "pwsh" for step in production_steps)
    first_material_token = {
        "Reject non-snapshot lifecycle refs": "$authorizationRef =",
        "Validate complete trusted toolchain": "if ([string]$env:VIA000_BASE_PYTHON_VERSION",
        "Prove production containment against detached replacement payload": "$platform =",
        "Execute frozen clean platform protocol": "$platform =",
        "Execute frozen mutation-test matrix": "$platform =",
        "Capture exact attestation subjects after containment teardown": "$evidence =",
        "Retain producer attestation bundle": "$evidence =",
        "Remove rejected platform workspace": "$workspace =",
    }
    for step in production_steps:
        run = step["run"]
        assert "if ($IsWindows)" in run
        assert "} elseif ($IsLinux) {" in run
        assert "$expectedShell = '/opt/microsoft/powershell/7/pwsh'" in run
        assert (
            "trusted hosted built-in pwsh identity, profile, version, or PATH differs"
            in run
        )
        assert run.index("if ($IsWindows)") < run.index(first_material_token[step["name"]])
        assert run.index("} elseif ($IsLinux) {") < run.index(
            first_material_token[step["name"]]
        )

    if os.name != "nt":
        return
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    sentinel = tmp_path / "rr12-built-in-pwsh-sentinel.txt"
    payload = b"rr12-built-in-pwsh-executed\n"
    expected_sha = hashlib.sha256(payload).hexdigest()
    local_path = f"{Path(pwsh).parent};C:\\Windows\\System32;C:\\Windows"
    script = tmp_path / "rr12-built-in-pwsh.ps1"
    script.write_text(
        r"""
$ErrorActionPreference = 'Stop'
$expectedShell = [IO.Path]::GetFullPath([string]$env:VIA000_EXPECTED_SHELL)
$expectedPath = [string]$env:VIA000_EXPECTED_PATH
$processModule = [Diagnostics.Process]::GetCurrentProcess().MainModule
$observedShell = if ($null -eq $processModule) {
  ''
} else {
  [IO.Path]::GetFullPath($processModule.FileName)
}
$psHomeShell = [IO.Path]::GetFullPath((Join-Path $PSHOME 'pwsh.exe'))
$profileFiles = @(
  $PROFILE.AllUsersAllHosts,
  $PROFILE.AllUsersCurrentHost,
  $PROFILE.CurrentUserAllHosts,
  $PROFILE.CurrentUserCurrentHost
) | Select-Object -Unique
if (-not $observedShell.Equals($expectedShell, [StringComparison]::OrdinalIgnoreCase) -or
    -not $psHomeShell.Equals($expectedShell, [StringComparison]::OrdinalIgnoreCase) -or
    $PSVersionTable.PSVersion.ToString() -cnotmatch '^7\.[0-9]+\.[0-9]+$' -or
    @($profileFiles | Where-Object { Test-Path -LiteralPath $_ }).Count -ne 0 -or
    [string]$env:PATH -cne $expectedPath) {
  throw 'trusted Windows built-in pwsh identity, profile, version, or PATH differs'
}
$bytes = [Text.UTF8Encoding]::new($false).GetBytes("rr12-built-in-pwsh-executed`n")
[IO.File]::WriteAllBytes([string]$env:VIA000_SENTINEL, $bytes)
$item = Get-Item -LiteralPath ([string]$env:VIA000_SENTINEL) -Force -ErrorAction Stop
if (-not ($item -is [IO.FileInfo]) -or
    ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
    $item.Length -ne 28 -or
    (Get-FileHash -Algorithm SHA256 -LiteralPath $item.FullName).Hash.ToLowerInvariant() `
      -cne [string]$env:VIA000_EXPECTED_SHA256) {
  throw 'same-step sentinel postcondition failed'
}
""".lstrip(),
        encoding="utf-8",
        newline="\n",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": local_path,
            "VIA000_EXPECTED_PATH": local_path,
            "VIA000_EXPECTED_SHELL": pwsh,
            "VIA000_EXPECTED_SHA256": expected_sha,
            "VIA000_SENTINEL": str(sentinel),
        }
    )
    command = [pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(script)]
    completed = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert sentinel.read_bytes() == payload
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == expected_sha

    sentinel.unlink()
    hostile_path = f"{tmp_path};{local_path}"
    hostile_environment = dict(environment)
    hostile_environment["PATH"] = hostile_path
    rejected = subprocess.run(
        command, env=hostile_environment, capture_output=True, text=True
    )
    assert rejected.returncode != 0
    assert not sentinel.exists()


@pytest.mark.skipif(os.name != "nt", reason="exact Windows Job Object control")
@pytest.mark.negative_control
def test_r3_windows_production_containment_kills_detached_replace_restore_tree(
    tmp_path: Path,
) -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    mutable = tmp_path / "mutable"
    trusted = tmp_path / "trusted"
    mutable.mkdir()
    trusted.mkdir()
    protected_tool = trusted / "tool.bin"
    protected_tool.write_bytes(b"trusted-tool-closure")
    payload = mutable / "hostile.py"
    payload.write_text(
        "import os, subprocess, sys, time\n"
        "mode, tool, trusted, mutable = sys.argv[1:]\n"
        "if mode == 'writer':\n"
        "    time.sleep(1.5)\n"
        "    for target in (os.path.join(trusted, 'delayed-marker'), tool):\n"
        "        try:\n"
        "            old = open(target, 'rb').read() if os.path.exists(target) else b''\n"
        "            if target == tool and old: open(os.path.join(mutable, "
        "'delayed-protected-read-succeeded'), 'wb').write(old)\n"
        "            open(target, 'wb').write(b'replaced')\n"
        "            open(target, 'wb').write(old)\n"
        "        except OSError:\n"
        "            pass\n"
        "    raise SystemExit(0)\n"
        "if mode == 'relay':\n"
        "    subprocess.Popen([sys.executable, '-I', '-S', __file__, 'writer', "
        "tool, trusted, mutable], creationflags=subprocess.DETACHED_PROCESS | "
        "subprocess.CREATE_NEW_PROCESS_GROUP, close_fds=True)\n"
        "    open(os.path.join(mutable, 'ready'), 'w').write('ready')\n"
        "    raise SystemExit(0)\n"
        "open(os.path.join(mutable, 'mutable-write-allowed'), 'w').write('allowed')\n"
        "for target in (os.path.join(trusted, 'direct-marker'), tool):\n"
        "    try:\n"
        "        old = open(target, 'rb').read() if os.path.exists(target) else b''\n"
        "        if target == tool and old: open(os.path.join(mutable, "
        "'protected-read-succeeded'), 'wb').write(old)\n"
        "        open(target, 'wb').write(b'replace-then-restore')\n"
        "        open(target, 'wb').write(old)\n"
        "    except OSError:\n"
        "        pass\n"
        "try:\n"
        "    link = os.path.join(mutable, 'tool-hardlink')\n"
        "    os.link(tool, link)\n"
        "    os.replace(link, tool)\n"
        "except OSError:\n"
        "    pass\n"
        "subprocess.Popen([sys.executable, '-I', '-S', __file__, 'relay', tool, "
        "trusted, mutable], close_fds=True)\n"
        "deadline = time.monotonic() + 2\n"
        "while not os.path.exists(os.path.join(mutable, 'ready')) and "
        "time.monotonic() < deadline:\n"
        "    time.sleep(.01)\n"
        "raise SystemExit(0 if os.path.exists(os.path.join(mutable, 'ready')) else 76)\n",
        encoding="utf-8",
    )
    harness = tmp_path / "invoke.ps1"
    result_path = trusted / "result.json"
    harness.write_text(
        "param([string]$Boundary,[string]$Python,[string]$Payload,[string]$Mutable,[string]$Trusted)\n"
        "$ErrorActionPreference='Stop'\n"
        ". $Boundary\n"
        "$tools=@{}\n"
        "Test-Via000ContainmentAvailability -PlatformFamily windows-x86_64 -SystemTools $tools\n"
        "Set-Via000RootIntegrity -Path $Mutable -Kind mutable -SystemTools $tools\n"
            "Set-Via000RootIntegrity -Path $Trusted -Kind protected -SystemTools $tools\n"
            "[Via000R3.NativeContainment]::SetIntegrityLabel("
            "(Join-Path $Trusted 'tool.bin'),$false,$true)\n"
        "$closure=@{}; $closure[$Python]=(Get-FileHash -Algorithm SHA256 "
        "-LiteralPath $Python).Hash.ToLowerInvariant(); "
        "$closure[$Boundary]=(Get-FileHash -Algorithm SHA256 -LiteralPath "
        "$Boundary).Hash.ToLowerInvariant()\n"
        "$closure[(Join-Path $Trusted 'tool.bin')]=(Get-FileHash -Algorithm SHA256 "
        "-LiteralPath (Join-Path $Trusted 'tool.bin')).Hash.ToLowerInvariant()\n"
        "Invoke-Via000ContainedCommand -Label hostile -ContractId "
        "rr6-production-hostile -PlatformFamily windows-x86_64 -FilePath $Python "
        "-Arguments @('-I','-S',$Payload,'attack',"
        "(Join-Path $Trusted 'tool.bin'),$Trusted,$Mutable) "
        "-WorkingDirectory $Mutable -MutableRoot $Mutable -TrustedRoot $Trusted "
        "-StdoutPath (Join-Path $Trusted 'stdout.txt') -StderrPath (Join-Path "
        "$Trusted 'stderr.txt') -ResultPath (Join-Path $Trusted 'result.json') "
        "-Environment @{} -Closure $closure -SystemTools $tools -TimeoutSeconds 30\n",
        encoding="utf-8",
    )
    tool_sha = _sha(protected_tool)
    completed = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-File",
            str(harness),
            str(CONTAINMENT),
            sys.executable,
            str(payload),
            str(mutable),
            str(trusted),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["primitive"] == "windows-low-integrity-restricted-token-job-object"
    assert result["descendants_quiescent"] is True
    assert result["active_processes_after_teardown"] == 0
    assert result["token_restriction_flags"] == ["DISABLE_MAX_PRIVILEGE"]
    assert result["token_integrity_sid"] == "S-1-16-4096"
    assert result["enabled_privileges"] in ([], ["SeChangeNotifyPrivilege"])
    assert result["enabled_privilege_count"] == len(result["enabled_privileges"])
    assert result["protected_label_policy"] == "medium-integrity-no-write-up-no-read-up"
    import time

    time.sleep(2)
    assert not (trusted / "direct-marker").exists()
    assert not (trusted / "delayed-marker").exists()
    assert not (mutable / "protected-read-succeeded").exists()
    assert not (mutable / "delayed-protected-read-succeeded").exists()
    assert (mutable / "mutable-write-allowed").read_text(encoding="utf-8") == "allowed"
    assert _sha(protected_tool) == tool_sha


@pytest.mark.negative_control
def test_r3_rr14_no_lua_low_il_production_identity_is_bound() -> None:
    """TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001."""
    source = CONTAINMENT.read_text(encoding="utf-8")
    receipt = (
        ROOT
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-protocol.ps1"
    ).read_text(encoding="utf-8")
    for frozen in (source, receipt):
        assert "LUA_TOKEN" not in frozen
        assert "CreateRestrictedToken(current, DISABLE_MAX_PRIVILEGE," in frozen
        assert 'String.Equals(integritySid, "S-1-16-4096"' in frozen
        assert '"SeChangeNotifyPrivilege"' in frozen
        assert "EnabledPrivilegeNames(restricted)" in frozen
    for consumer in (
        CONTAINMENT_PROOF_RUNNER.read_text(encoding="utf-8"),
        CONTAINMENT_PROOF_AGGREGATOR.read_text(encoding="utf-8"),
        RUNNER.read_text(encoding="utf-8"),
        MUTATION_RUNNER.read_text(encoding="utf-8"),
        ASSEMBLER.read_text(encoding="utf-8"),
    ):
        for token in (
            "DISABLE_MAX_PRIVILEGE",
            "S-1-16-4096",
            "enabled_privileges",
            "protected_label_policy",
        ):
            assert token in consumer


@pytest.mark.negative_control
def test_r3_rr15_powershell_parse_closure_and_exact_array_comparison(
    tmp_path: Path,
) -> None:
    """TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001."""
    protocol_paths = sorted(PROTOCOL_DIR.glob("*.ps1"))
    receipt_paths = sorted(
        (
            ROOT
            / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000"
        ).glob("*.ps1")
    )
    paths = sorted(protocol_paths + receipt_paths)
    observed_paths = tuple(path.relative_to(ROOT).as_posix() for path in paths)
    assert observed_paths == R3_POWERSHELL_RELATIVE_PATHS

    pwsh = shutil.which("pwsh")
    assert pwsh is not None, "PowerShell 7 is mandatory for the R3 parser gate"
    version = subprocess.run(
        [pwsh, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert version.startswith("PowerShell 7."), version

    parser_gate = tmp_path / "rr15-parse-gate.ps1"
    parser_gate.write_text(
        """param(
    [Parameter(Mandatory = $true, Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Paths
)
$failed = $false
foreach ($path in $Paths) {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $path, [ref]$tokens, [ref]$errors
    ) | Out-Null
    if ($errors.Count -ne 0) {
        $failed = $true
        foreach ($parseError in $errors) {
            [Console]::Error.WriteLine("{0}: {1}", $path, $parseError.Message)
        }
    }
}
if ($failed) { exit 1 }
""",
        encoding="utf-8",
        newline="\n",
    )
    parser_command = [
        pwsh,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-File",
        str(parser_gate),
    ]
    parsed = subprocess.run(
        [*parser_command, *(str(path) for path in paths)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert parsed.returncode == 0, parsed.stderr
    assert parsed.stderr == ""

    workflow_blocks: list[Path] = []
    for workflow_path in (WORKFLOW, CONTAINMENT_PROOF_WORKFLOW):
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job_name, job in workflow["jobs"].items():
            for step_index, step in enumerate(job.get("steps", [])):
                shell = str(step.get("shell", "")).lower()
                if not isinstance(step.get("run"), str) or not (
                    "pwsh" in shell or "powershell" in shell
                ):
                    continue
                block_path = tmp_path / (
                    f"{workflow_path.stem}-{job_name}-{step_index:02d}.ps1"
                )
                block_path.write_text(
                    step["run"], encoding="utf-8", newline="\n"
                )
                workflow_blocks.append(block_path)
    assert len(workflow_blocks) == 40
    parsed_blocks = subprocess.run(
        [*parser_command, *(str(path) for path in workflow_blocks)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert parsed_blocks.returncode == 0, parsed_blocks.stderr
    assert parsed_blocks.stderr == ""

    malformed = tmp_path / "rr15-invalid-operator.ps1"
    malformed.write_text(
        'if (($value -cjoin "\\n")) { $true }\n',
        encoding="utf-8",
        newline="\n",
    )
    rejected = subprocess.run(
        [*parser_command, *(str(path) for path in paths), str(malformed)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert malformed.name in rejected.stderr
    assert "Unexpected token '-cjoin'" in rejected.stderr

    affected_paths = (
        CONTAINMENT_PROOF_RUNNER,
        RUNNER,
        ROOT
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/"
        "containment-proof-runner.ps1",
        ROOT
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1",
    )
    affected_sources = [path.read_text(encoding="utf-8") for path in affected_paths]
    assert all("-cjoin" not in source for source in affected_sources)
    predicates = {
        _powershell_function_text(source, "Test-ExactOrdinalStringArray")
        for source in affected_sources
    }
    assert len(predicates) == 1
    predicate = predicates.pop()
    assert "[StringComparison]::Ordinal" in predicate
    assert "[string]::Equals" in predicate
    assert "$Left.Count -ne $Right.Count" in predicate
    assert "$null -eq $Left" in predicate
    assert "$null -eq $leftValue" in predicate
    assert "$leftValue -isnot [string]" in predicate

    predicate_gate = tmp_path / "rr15-exact-array-gate.ps1"
    predicate_gate.write_text(
        predicate
        + """
function Assert-ExactArrayCase {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Invocation,
        [Parameter(Mandatory = $true)][bool]$Expected
    )
    $result = @(& $Invocation)
    if ($result.Count -ne 1 -or $result[0] -isnot [bool] -or
        [bool]$result[0] -ne $Expected) {
        throw 'exact ordinal string-array predicate returned a wrong or non-Boolean result'
    }
}
Assert-ExactArrayCase { Test-ExactOrdinalStringArray -Left @() -Right @() } $true
Assert-ExactArrayCase {
    Test-ExactOrdinalStringArray `
        -Left @('DISABLE_MAX_PRIVILEGE') -Right @('DISABLE_MAX_PRIVILEGE')
} $true
Assert-ExactArrayCase {
    Test-ExactOrdinalStringArray `
        -Left @('DISABLE_MAX_PRIVILEGE') -Right @('disable_max_privilege')
} $false
Assert-ExactArrayCase { Test-ExactOrdinalStringArray -Left @('a', 'b') -Right @('b', 'a') } $false
Assert-ExactArrayCase { Test-ExactOrdinalStringArray -Left @('a') -Right @('a', 'b') } $false
Assert-ExactArrayCase {
    Test-ExactOrdinalStringArray `
        -Left ([object[]]@($null)) -Right ([object[]]@($null))
} $false
Assert-ExactArrayCase {
    Test-ExactOrdinalStringArray -Left ([object[]]@(1)) -Right ([object[]]@(1))
} $false
Assert-ExactArrayCase {
    Test-ExactOrdinalStringArray -Left @('a', "b`nc") -Right @("a`nb", 'c')
} $false
Assert-ExactArrayCase { Test-ExactOrdinalStringArray -Left @() -Right @('') } $false
Assert-ExactArrayCase { Test-ExactOrdinalStringArray -Left @('') -Right @('') } $true
""",
        encoding="utf-8",
        newline="\n",
    )
    predicate_result = subprocess.run(
        [
            pwsh,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(predicate_gate),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert predicate_result.returncode == 0, predicate_result.stderr
    assert predicate_result.stdout == ""
    assert predicate_result.stderr == ""


@pytest.mark.negative_control
def test_r3_rr16_ubuntu_canonical_empty_array_and_windows_export_source(
    tmp_path: Path,
) -> None:
    """TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001."""
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    affected = (
        CONTAINMENT_PROOF_RUNNER,
        RUNNER,
        ROOT
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/"
        "containment-proof-runner.ps1",
        ROOT
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1",
    )
    for path in affected:
        source = path.read_text(encoding="utf-8")
        assert "[object[]]$expectedTokenFlags = @()" in source
        assert '$expectedTokenFlags = [object[]]@("DISABLE_MAX_PRIVILEGE")' in source
        assert '$expectedTokenFlags = if ($PlatformFamily' not in source

    predicate = _powershell_function_text(
        CONTAINMENT_PROOF_RUNNER.read_text(encoding="utf-8"),
        "Test-ExactOrdinalStringArray",
    )
    script = tmp_path / "canonical-empty-array.ps1"
    script.write_text(
        predicate
        + "\n[object[]]$expected=@()\n"
        + "if ($null -eq $expected -or $expected.Count -ne 0) { exit 71 }\n"
        + "if (-not (Test-ExactOrdinalStringArray -Left ([object[]]@()) "
        + "-Right $expected)) { exit 72 }\n"
        + "if (Test-ExactOrdinalStringArray -Left $null -Right $expected) { exit 73 }\n",
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        [pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    for path in (
        CONTAINMENT,
        ROOT
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-protocol.ps1",
    ):
        source = path.read_text(encoding="utf-8")
        for token in (
            "Set-Via000WindowsExportSecurity",
            "Assert-Via000WindowsExportSecurity",
            "protected-current-runner-full-control-v1",
            "S-1-16-8192",
            "NO_WRITE_UP",
            "GetMandatoryLabel",
            "SetExportRootDescriptor",
            "SetExportFileOwner",
            "PROTECTED_DACL_SECURITY_INFORMATION",
        ):
            assert token in source
        assert "Set-Acl -LiteralPath $item.FullName" not in source


@pytest.mark.skipif(os.name != "nt", reason="exact Windows low-integrity export boundary")
@pytest.mark.negative_control
def test_r3_rr16_windows_medium_export_boundary_denies_low_integrity_writes(
    tmp_path: Path,
) -> None:
    """TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001."""
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    mutable, trusted, workspace = (
        tmp_path / name for name in ("mutable", "trusted", "workspace")
    )
    for directory in (mutable, trusted, workspace):
        directory.mkdir()
    export = workspace / ".via000-r3-proof-cache/windows-x86_64/candidate"
    export.mkdir(parents=True)
    envelope = export / "envelope.json"
    envelope.write_bytes(b'{"safe":true}\n')
    expected_sha = _sha(envelope)
    (mutable / "replacement.json").write_bytes(b"replacement\n")
    payload = mutable / "export-attack.py"
    payload.write_text(
        "import os, sys\n"
        "export, envelope, mutable = sys.argv[1:]\n"
        "open(os.path.join(mutable, 'mutable-write-allowed'), 'w').write('allowed')\n"
        "operations = {\n"
        " 'create': lambda: open(os.path.join(export, 'created'), 'wb').write(b'x'),\n"
        " 'write': lambda: open(envelope, 'wb').write(b'replaced'),\n"
        " 'hardlink': lambda: os.link(envelope, os.path.join(export, 'hardlink')),\n"
        " 'rename': lambda: os.rename(envelope, envelope + '.moved'),\n"
        " 'delete': lambda: os.unlink(envelope),\n"
        " 'reparse': lambda: os.symlink(mutable, os.path.join(export, 'link'), "
        "target_is_directory=True),\n"
        " 'replace': lambda: os.replace(os.path.join(mutable, 'replacement.json'), envelope),\n"
        "}\n"
        "for name, operation in operations.items():\n"
        " try:\n"
        "  operation(); open(os.path.join(mutable, name + '-succeeded'), 'w').write(name)\n"
        " except OSError:\n"
        "  pass\n",
        encoding="utf-8",
        newline="\n",
    )
    harness = tmp_path / "invoke-export.ps1"
    harness.write_text(
        "param([string]$Boundary,[string]$Python,[string]$Payload,[string]$Mutable,[string]$Trusted,[string]$Export,[string]$ExpectedSha)\n"
        "$ErrorActionPreference='Stop'; . $Boundary; $tools=@{}\n"
        "Test-Via000ContainmentAvailability -PlatformFamily windows-x86_64 -SystemTools $tools\n"
        "Set-Via000RootIntegrity -Path $Mutable -Kind mutable -SystemTools $tools\n"
        "Set-Via000RootIntegrity -Path $Trusted -Kind protected -SystemTools $tools\n"
        "$preAcl=Get-Acl -LiteralPath $Export\n"
        "$world=[Security.Principal.SecurityIdentifier]::new('S-1-1-0')\n"
        "$preRule=[Security.AccessControl.FileSystemAccessRule]::new($world,"
        "[Security.AccessControl.FileSystemRights]::Modify,"
        "([Security.AccessControl.InheritanceFlags]::ContainerInherit -bor "
        "[Security.AccessControl.InheritanceFlags]::ObjectInherit),"
        "[Security.AccessControl.PropagationFlags]::None,"
        "[Security.AccessControl.AccessControlType]::Allow)\n"
        "[void]$preAcl.AddAccessRule($preRule)\n"
        "Set-Acl -LiteralPath $Export -AclObject $preAcl\n"
        "$rootSecurity=Set-Via000WindowsExportSecurity -Path $Export\n"
        "$fileSecurity=Set-Via000WindowsExportFileOwner `\n"
        " -Path (Join-Path $Export 'envelope.json') `\n"
        " -ExpectedSha256 $ExpectedSha\n"
        "if ($rootSecurity.owner_sid -cne $fileSecurity.owner_sid -or `\n"
        " -not $fileSecurity.inherited_file_dacl -or `\n"
        " [uint16]$rootSecurity.control_flags -ne 37892 -or `\n"
        " [uint16]$fileSecurity.control_flags -ne 33796) { `\n"
        " throw 'initial export descriptor differs' }\n"
        "$closure=@{}\n"
        "foreach($path in @($Python,$Boundary,(Join-Path $Export 'envelope.json'))){\n"
        " $closure[$path]=(Get-FileHash -Algorithm SHA256 `\n"
        "  -LiteralPath $path).Hash.ToLowerInvariant()\n"
        "}\n"
        "Invoke-Via000ContainedCommand -Label rr16-export `\n"
        " -ContractId rr16-export-boundary -PlatformFamily windows-x86_64 `\n"
        " -FilePath $Python -Arguments @('-I','-S',$Payload,$Export,`\n"
        "  (Join-Path $Export 'envelope.json'),$Mutable) `\n"
        " -WorkingDirectory $Mutable -MutableRoot $Mutable -TrustedRoot $Trusted `\n"
        " -StdoutPath (Join-Path $Trusted 'stdout.txt') `\n"
        " -StderrPath (Join-Path $Trusted 'stderr.txt') `\n"
        " -ResultPath (Join-Path $Trusted 'result.json') `\n"
        " -Environment @{} -Closure $closure -SystemTools $tools -TimeoutSeconds 30\n"
        "$rootCheck=Assert-Via000WindowsExportSecurity -Path $Export -Kind root\n"
        "$fileCheck=Assert-Via000WindowsExportSecurity `\n"
        " -Path (Join-Path $Export 'envelope.json') -Kind file `\n"
        " -ExpectedSha256 $ExpectedSha\n"
        "[IO.File]::WriteAllText((Join-Path $Export 'trusted-write.txt'),`\n"
        " 'trusted',[Text.UTF8Encoding]::new($false))\n"
        "if ([IO.File]::ReadAllText((Join-Path $Export 'trusted-write.txt')) `\n"
        " -cne 'trusted') { throw 'trusted runner export access failed' }\n",
        encoding="utf-8",
        newline="\n",
    )
    completed = subprocess.run(
        [pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(harness),
         str(CONTAINMENT), sys.executable, str(payload), str(mutable), str(trusted),
         str(export), expected_sha],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads((trusted / "result.json").read_text(encoding="utf-8"))
    assert result["descendants_quiescent"] is True
    assert result["active_processes_after_teardown"] == 0
    assert (mutable / "mutable-write-allowed").read_text(encoding="utf-8") == "allowed"
    for name in (
        "create", "write", "hardlink", "rename", "delete", "reparse", "replace"
    ):
        assert not (mutable / f"{name}-succeeded").exists()
    assert _sha(envelope) == expected_sha
    assert (export / "trusted-write.txt").read_text(encoding="utf-8") == "trusted"


@pytest.mark.negative_control
def test_r3_rr18_native_root_file_descriptor_and_workspace_hostile_replay(
    tmp_path: Path,
) -> None:
    """TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001."""
    containment_text = CONTAINMENT.read_text(encoding="utf-8")
    runner_text = CONTAINMENT_PROOF_RUNNER.read_text(encoding="utf-8")
    workflow_text = CONTAINMENT_PROOF_WORKFLOW.read_text(encoding="utf-8")
    for token in (
        "OWNER_SECURITY_INFORMATION",
        "DACL_SECURITY_INFORMATION",
        "PROTECTED_DACL_SECURITY_INFORMATION",
        "SetExportRootDescriptor",
        "SetExportFileOwner",
        "GetExportDescriptor",
        "DaclDefaulted",
        "DaclNull",
        "AccessMask -ne 2032127",
        "AceFlags -ne $expectedAceFlags",
        "$expectedControlFlags",
        "GetMandatoryLabel",
    ):
        assert token in containment_text
    assert "Set-Acl -LiteralPath $item.FullName -AclObject $security" not in containment_text
    assert "Set-Via000WindowsExportFileOwner -Path $verifiedEnvelope" in runner_text
    assert runner_text.index("$stream.Dispose()") < runner_text.index(
        "Set-Via000WindowsExportFileOwner -Path $verifiedEnvelope"
    )
    assert workflow_text.count("[uint16]$rootSecurity.control_flags -ne 37892") == 2
    assert workflow_text.count("[uint16]$fileSecurity.control_flags -ne 33796") == 2
    if os.name == "nt":
        test_r3_rr16_windows_medium_export_boundary_denies_low_integrity_writes(tmp_path)


@pytest.mark.negative_control
def test_r3_rr19_windows_cache_zstd_identity_is_exact_and_rechecked() -> None:
    """TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001."""
    workflow_text = CONTAINMENT_PROOF_WORKFLOW.read_text(encoding="utf-8")
    receipt_path = (
        ROOT
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/"
        "containment-proof-workflow.yml"
    )
    expected_path = (
        r"C:\Program Files\PowerShell\7;C:\tools\zstd;"
        r"C:\Program Files\Git\usr\bin;C:\Program Files\Git\mingw64\bin;"
        r"C:\Program Files\Git\cmd;C:\Windows\System32;C:\Windows"
    )
    exact_zstd = r"C:\tools\zstd\zstd.exe"

    def assert_contract(source: str) -> None:
        document = yaml.safe_load(source)
        assert r"C:\Program Files\Git\mingw64\bin\zstd.exe" not in source
        assert "$zstd = 'C:\\tools\\zstd\\zstd.exe'" in source
        assert "$zstdRoot = 'C:\\tools\\zstd'" in source
        assert "@('C:\\', 'C:\\tools', $zstdRoot, $zstd)" in source
        assert "[IO.FileAttributes]::ReparsePoint" in source
        assert "Get-Item -LiteralPath $zstd -Stream *" in source
        assert "hardlink list $zstd" in source
        assert "Get-Command zstd.exe -All -CommandType Application" in source
        assert "$commands.Count -ne 1" in source
        assert "Groups[1].Value -cne '1.5.7'" in source
        assert "Get-FileHash -Algorithm SHA256 -LiteralPath $zstd" in source
        assert "$observedZstdSha256 -cne $expectedZstdSha256" in source
        assert "windows_zstd_sha256=$zstdSha256" in source
        assert source.count("C:\\tools\\zstd") >= 12
        for job_name in ("windows-candidate", "windows-pdf", "windows-mutation"):
            job = document["jobs"][job_name]
            assert job["env"]["PATH"] == expected_path
            steps = job["steps"]
            archive_index = next(
                index
                for index, step in enumerate(steps)
                if step.get("name") == "Assert exact cross-OS archive tools"
            )
            preflight_index = next(
                index
                for index, step in enumerate(steps)
                if step.get("name") == "Require fresh exact cache key"
            )
            save_index = next(
                index
                for index, step in enumerate(steps)
                if step.get("name") == "Save one exact digest-bound envelope cache"
            )
            post_index = next(
                index
                for index, step in enumerate(steps)
                if step.get("name")
                == "Recheck archive tool and staged envelope after cache save"
            )
            archive = steps[archive_index]
            preflight = steps[preflight_index]
            post = steps[post_index]
            assert archive["id"] == "archive_tools"
            assert archive["shell"] == "pwsh"
            assert exact_zstd in archive["run"]
            assert preflight_index + 1 == save_index
            assert save_index + 1 == post_index
            assert archive_index < preflight_index
            assert preflight["env"]["VIA000_ZSTD_SHA256"] == (
                "${{ steps.archive_tools.outputs.windows_zstd_sha256 }}"
            )
            assert post["env"]["VIA000_ZSTD_SHA256"] == (
                "${{ steps.archive_tools.outputs.windows_zstd_sha256 }}"
            )
            assert "$observedZstdSha256 -cne $expectedZstdSha256" in preflight["run"]
            assert "$observedZstdSha256 -cne $expectedZstdSha256" in post["run"]
            assert steps[save_index]["uses"] == (
                "actions/cache/save@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"
            )

    assert receipt_path.read_bytes() == CONTAINMENT_PROOF_WORKFLOW.read_bytes()
    assert_contract(workflow_text)

    mutations = {
        "missing": workflow_text.replace(exact_zstd, r"C:\missing\zstd.exe"),
        "wrong-version": workflow_text.replace("1.5.7", "1.5.6"),
        "changed-hash": workflow_text.replace(
            "$observedZstdSha256 -cne $expectedZstdSha256", "$false"
        ),
        "path-shadow": workflow_text.replace(
            r"C:\Program Files\PowerShell\7;C:\tools\zstd;",
            r"C:\shadow;C:\Program Files\PowerShell\7;C:\tools\zstd;",
        ),
        "reparse": workflow_text.replace(
            "[IO.FileAttributes]::ReparsePoint", "[IO.FileAttributes]::Hidden"
        ),
        "alias": workflow_text.replace(
            exact_zstd, r"C:\tools\zstd\..\zstd\zstd.exe"
        ),
        "git-relative": workflow_text.replace(
            exact_zstd, r"C:\Program Files\Git\mingw64\bin\zstd.exe"
        ),
        "multi-allowlist": workflow_text.replace(
            "$zstd = 'C:\\tools\\zstd\\zstd.exe'",
            "$zstd = @('C:\\tools\\zstd\\zstd.exe','C:\\other\\zstd.exe')[0]",
        ),
    }
    for mutation in mutations.values():
        with pytest.raises((AssertionError, KeyError, StopIteration, yaml.YAMLError)):
            assert_contract(mutation)


@pytest.mark.negative_control
def test_r3_rr20_canonical_inner_json_bytes(tmp_path: Path) -> None:
    """TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001."""
    containment_text = CONTAINMENT.read_text(encoding="utf-8")
    runner_text = CONTAINMENT_PROOF_RUNNER.read_text(encoding="utf-8")
    containment_receipt = (
        ROOT
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/"
        "containment-protocol.ps1"
    )
    runner_receipt = (
        ROOT
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/"
        "containment-proof-runner.ps1"
    )
    aggregator_receipt = (
        ROOT
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/"
        "containment-proof-aggregator.py"
    )
    assert containment_receipt.read_bytes() == CONTAINMENT.read_bytes()
    assert runner_receipt.read_bytes() == CONTAINMENT_PROOF_RUNNER.read_bytes()
    assert aggregator_receipt.read_bytes() == CONTAINMENT_PROOF_AGGREGATOR.read_bytes()
    assert "function Write-Via000CanonicalJsonObject" in containment_text
    assert "[Collections.IDictionary]$Document" in containment_text
    assert "[IO.FileMode]::CreateNew" in containment_text
    assert "[IO.FileMode]::Open" in containment_text
    assert "[IO.FileShare]::None" in containment_text
    assert "[IO.FileOptions]::WriteThrough" in containment_text
    assert "$stream.Flush($true)" in containment_text
    assert "[Text.UTF8Encoding]::new($false, $true)" in containment_text
    assert "[Security.Cryptography.SHA256]::HashData($readBack)" in containment_text
    assert containment_text.count("Write-Via000CanonicalJsonObject -Document $record") == 2
    assert "Write-Via000CanonicalJsonObject -Document $proof" in runner_text
    assert "ConvertTo-Json -Depth 8 | Set-Content" not in containment_text
    assert "ConvertTo-Json -Depth 8 | Set-Content" not in runner_text

    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    module = _load_module(CONTAINMENT_PROOF_AGGREGATOR, "r3_rr20_aggregator")
    source_sha = "a" * 40
    source_ref = "refs/heads/campaign/via000-r3-protocol-proof-test"
    workflow_ref = (
        "whact2025/POPGP/.github/workflows/via000-r3-containment-proof.yml@"
        + source_ref
    )
    subject_root = tmp_path / "production-subjects"
    subject_root.mkdir()
    writer_script = tmp_path / "rr20-writer.ps1"
    writer_script.write_text(
        "param([string]$Helper,[string]$DocumentPath,[string]$Root,[switch]$ReplaceOne)\n"
        "$ErrorActionPreference='Stop'; . $Helper\n"
        "$documents=Get-Content -LiteralPath $DocumentPath -Raw | "
        "ConvertFrom-Json -AsHashtable\n"
        "$first=$true\n"
        "foreach($entry in $documents.GetEnumerator()){\n"
        " if([string]$entry.Key -cnotmatch '^[a-z0-9._-]+$'){throw 'bad name'}\n"
        " $path=Join-Path $Root ([string]$entry.Key)\n"
        " Write-Via000CanonicalJsonObject -Document $entry.Value -Path $path\n"
        " if($ReplaceOne -and $first){\n"
        "  $before=(Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash\n"
        "  Write-Via000CanonicalJsonObject -Document $entry.Value -Path $path "
        "-ReplaceExisting\n"
        "  $after=(Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash\n"
        "  if($before -cne $after){throw 'replacement bytes changed'}\n"
        "  $first=$false\n"
        " }\n"
        "}\n",
        encoding="utf-8",
        newline="\n",
    )

    def write_documents(documents: dict[str, object], label: str, replace: bool) -> None:
        source = tmp_path / f"{label}.json"
        source.write_text(
            json.dumps(documents, separators=(",", ":")),
            encoding="utf-8",
            newline="\n",
        )
        command = [
            pwsh,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(writer_script),
            str(CONTAINMENT),
            str(source),
            str(subject_root),
        ]
        if replace:
            command.append("-ReplaceOne")
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr

    result_documents: dict[str, object] = {}
    cell_metadata: dict[tuple[str, str], tuple[str, str, str]] = {}
    for platform in module.PLATFORMS:
        primitive = (
            "windows-low-integrity-restricted-token-job-object"
            if platform == "windows-x86_64"
            else "ubuntu-systemd-ephemeral-user-control-group"
        )
        privilege = (
            "low-integrity-restricted-token"
            if platform == "windows-x86_64"
            else "systemd-ephemeral-user"
        )
        for stage in module.STAGES:
            stem = f"{platform}-{stage}"
            result_name = f"{stem}-containment-result.json"
            proof_name = f"{stem}-proof.json"
            artifact = f"via000-r3-containment-proof-{platform}-{stage}"
            contained: dict[str, object] = {
                "schema_version": 1,
                "label": f"proof-{stage}",
                "contract_id": "rr7-hosted-containment-proof",
                "primitive": primitive,
                "privilege_separation": privilege,
                "descendants_quiescent": True,
                "active_processes_after_teardown": 0,
                "exit_code": 0,
                "timed_out": False,
                **_token_evidence(platform),
                "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "writer_control": "escaped\ncontrol",
            }
            if platform == "ubuntu-latest-x86_64":
                contained.update(
                    {
                        "ephemeral_identity_uid": "999",
                        "ephemeral_identity_processes_empty": True,
                        "ephemeral_identity_removed": True,
                    }
                )
            result_documents[result_name] = contained
            cell_metadata[(platform, stage)] = (result_name, proof_name, artifact)
    write_documents(result_documents, "results", True)

    proof_documents: dict[str, object] = {}
    for (platform, stage), (result_name, proof_name, artifact) in cell_metadata.items():
        result_bytes = (subject_root / result_name).read_bytes()
        primitive = result_documents[result_name]["primitive"]
        privilege = result_documents[result_name]["privilege_separation"]
        proof_documents[proof_name] = {
            "schema_version": 1,
            "proof_kind": "via000-r3-hosted-containment-cell",
            "repository": "whact2025/POPGP",
            "workflow": ".github/workflows/via000-r3-containment-proof.yml",
            "workflow_ref": workflow_ref,
            "event_name": "push",
            "source_ref": source_ref,
            "source_sha": source_sha,
            "run_id": "424242",
            "run_attempt": "1",
            "platform_family": platform,
            "stage_id": stage,
            "cell": f"{platform}/{stage}",
            "artifact_name": artifact,
            "primitive": primitive,
            "privilege_separation": privilege,
            **_token_evidence(platform),
            **_export_evidence(platform),
            "active_processes_after_teardown": 0,
            **{field: True for field in module.TRUE_FIELDS},
            **{field: "b" * 64 for field in module.HASH_FIELDS},
            "containment_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
    write_documents(proof_documents, "proofs", False)

    for path in subject_root.iterdir():
        content = path.read_bytes()
        assert content.startswith(b"{") and content.endswith(b"}\n")
        assert not content.startswith(b"\xef\xbb\xbf")
        assert b"\r" not in content
        assert content.count(b"\n") == 1
        assert not content.endswith(b"\n\n")
        assert isinstance(json.loads(content.decode("utf-8", errors="strict")), dict)

    fragments = tmp_path / "fragments"
    for (platform, stage), (result_name, proof_name, artifact) in cell_metadata.items():
        subjects = {
            "containment-result.json": (subject_root / result_name).read_bytes(),
            "proof.json": (subject_root / proof_name).read_bytes(),
            "stderr.txt": b"",
            "stdout.txt": b"",
        }
        proof = proof_documents[proof_name]
        identity = {field: proof[field] for field in module.IDENTITY_FIELDS}
        artifact_root = fragments / artifact
        artifact_root.mkdir(parents=True)
        (artifact_root / "envelope.json").write_bytes(
            module.build_envelope(identity, subjects)
        )

    output = tmp_path / "aggregate.json"
    command = [
        sys.executable,
        "-I",
        "-S",
        str(CONTAINMENT_PROOF_AGGREGATOR),
        "--input-root",
        str(fragments),
        "--output",
        str(output),
        "--source-sha",
        source_sha,
        "--source-ref",
        source_ref,
        "--workflow-ref",
        workflow_ref,
        "--run-id",
        "424242",
        "--run-attempt",
        "1",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_bytes())["cell_count"] == 6

    target = fragments / (
        "via000-r3-containment-proof-windows-x86_64-candidate/envelope.json"
    )
    original = target.read_bytes()
    original_document = json.loads(original)
    proof_bytes = base64.b64decode(
        original_document["members"]["proof.json"]["base64"]
    )
    mutations = {
        "bom": b"\xef\xbb\xbf" + proof_bytes,
        "crlf": proof_bytes[:-1] + b"\r\n",
        "missing-final-lf": proof_bytes[:-1],
        "double-final-lf": proof_bytes + b"\n",
    }
    for label, replacement in mutations.items():
        document = json.loads(original)
        previous_size = document["members"]["proof.json"]["size"]
        document["members"]["proof.json"] = {
            "base64": base64.b64encode(replacement).decode("ascii"),
            "sha256": hashlib.sha256(replacement).hexdigest(),
            "size": len(replacement),
        }
        document["total_decoded_bytes"] += len(replacement) - previous_size
        target.write_bytes(module.canonical_envelope_bytes(document))
        output.unlink(missing_ok=True)
        rejected = subprocess.run(command, check=False, capture_output=True, text=True)
        assert rejected.returncode != 0, label
        assert not output.exists()
    target.write_bytes(original)

    nonobject_input = tmp_path / "nonobject.json"
    nonobject_input.write_text('[1,2]', encoding="utf-8", newline="\n")
    rejected_path = subject_root / "nonobject.json"
    nonobject_script = tmp_path / "rr20-nonobject.ps1"
    nonobject_script.write_text(
        "param($Helper,$DocumentPath,$OutputPath)\n"
        "$ErrorActionPreference='Stop'; . $Helper\n"
        "$value=Get-Content -LiteralPath $DocumentPath -Raw | "
        "ConvertFrom-Json -AsHashtable\n"
        "Write-Via000CanonicalJsonObject -Document $value -Path $OutputPath\n",
        encoding="utf-8",
        newline="\n",
    )
    nonobject = subprocess.run(
        [
            pwsh,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(nonobject_script),
            str(CONTAINMENT),
            str(nonobject_input),
            str(rejected_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert nonobject.returncode != 0
    assert not rejected_path.exists()


@pytest.mark.negative_control
@pytest.mark.parametrize(
    "field,value",
    [
        ("descendants_quiescent", False),
        ("active_processes_after_teardown", 1),
        ("trusted_evidence_unreadable_unwritable", False),
        ("token_restriction_flags", ["DISABLE_MAX_PRIVILEGE", "LUA_TOKEN"]),
        ("token_integrity_sid", "S-1-16-8192"),
        ("enabled_privilege_count", 1),
        ("enabled_privileges", ["SeDebugPrivilege"]),
        ("protected_label_policy", "mutable"),
    ],
)
def test_r3_assembler_rejects_forged_execution_boundary(
    tmp_path: Path, field: str, value: object
) -> None:
    roots = _r3_platform_roots(tmp_path)
    summary_path = roots["windows-x86_64"] / "candidate/evidence/stage-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["execution_boundary"][field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "boundary-output"
    result = _run_assembler(roots, output)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.negative_control
@pytest.mark.parametrize(
    "payload_name",
    [
        "self-restored-pdftex.exe",
        "self-restored-texmf.cnf",
        "background-watcher-marker.txt",
        "runner-temp-closure.py",
        "cross-stage-payload.dll",
    ],
)
def test_r3_stage_isolation_rejects_cross_stage_state_injection(
    tmp_path: Path, payload_name: str
) -> None:
    roots = _r3_platform_roots(tmp_path)
    platform = r2_assembler.PLATFORMS[0]
    candidate = roots[platform] / "candidate"
    pdf_summary = roots[platform] / "pdf/evidence/stage-summary.json"
    mutation_summary = roots[platform] / "mutation/evidence/stage-summary.json"
    later_stage_hashes = (_sha(pdf_summary), _sha(mutation_summary))

    tool_manifest = candidate / "evidence/tool-identity-manifest.json"
    original_tool_bytes = tool_manifest.read_bytes()
    tool_manifest.write_bytes(b"candidate attempted a same-path replacement\n")
    tool_manifest.write_bytes(original_tool_bytes)
    (candidate / "evidence" / payload_name).write_bytes(b"untrusted cross-stage state\n")

    output = tmp_path / "stage-injection-output"
    result = _run_assembler(roots, output)
    assert result.returncode != 0
    assert not output.exists()
    assert not list(tmp_path.glob(".stage-injection-output-*"))
    assert (_sha(pdf_summary), _sha(mutation_summary)) == later_stage_hashes
