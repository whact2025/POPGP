from __future__ import annotations

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

from tests.unit import test_via000_r2_assembler as r2_assembler

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = ROOT / "protocols/POPGP-VIABILITY-R3-2026-08"
PROTOCOL = PROTOCOL_DIR / "VIA-000.json"
SCHEMA = PROTOCOL_DIR / "VIA-000-RAW-RESULTS.schema.json"
ASSEMBLER = PROTOCOL_DIR / "VIA-000-ASSEMBLER.py"
GUARD = PROTOCOL_DIR / "VIA-000-DISPATCH-GUARD.py"
RUNNER = PROTOCOL_DIR / "VIA-000-RUNNER.ps1"
MUTATION_RUNNER = PROTOCOL_DIR / "VIA-000-MUTATION-RUNNER.py"
CAMPAIGN_CHECKER = ROOT / "scripts/check_viability_campaign.py"
WORKFLOW = ROOT / ".github/workflows/via000-r3-protocol.yml"
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
                        "C:/runner/_temp/via000-r3-platform/python-environment/Scripts/python.exe"
                    ),
                    "sha256": "3" * 64,
                    "version": "3.11.15",
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
                        "/home/runner/work/_temp/via000-r3-platform/python-environment/bin/python"
                    ),
                    "sha256": "3" * 64,
                    "version": "3.11.15",
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
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return roots


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
    assert all(item["shell"].startswith(("/opt/", '"C:\\Program Files')) for item in matrix)
    assert all(
        step.get("shell") == "${{ matrix.shell }}"
        for step in document["jobs"]["platform-fragment"]["steps"]
        if "run" in step
    )
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
    assert "str(environment_python)" in mutation_runner
    assert "environment.pop(name, None)" in mutation_runner


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
        manifest_path = workspace / "evidence/tool-identity-manifest.json"
        valid = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        changed = roots[r2_assembler.PLATFORMS[1]] / "evidence/platform-summary.json"
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
