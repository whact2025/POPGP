from __future__ import annotations

import hashlib
import importlib.util
import json
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
WORKFLOW = ROOT / ".github/workflows/via000-r3-protocol.yml"
FAKE_COMMIT = "a" * 40
FAKE_REF = f"refs/tags/popgp-via000-r3-protocol-{FAKE_COMMIT}"
FAKE_AUTHORIZATION_SHA256 = "b" * 64
FAKE_AUTHORIZATION_REF = (
    "refs/tags/popgp-via000-r3-authorization-" + FAKE_AUTHORIZATION_SHA256
)
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
        repo
        / "reviews/viability/POPGP-VIABILITY-R3-2026-08/authorization/"
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
        "protocol_manifest_sha256": frozen_sha(
            authorization_contract["protocol_manifest_path"]
        ),
    }
    record_bytes = (
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    record_path = tmp_path / "authorization-record.json"
    record_path.write_bytes(record_bytes)
    authorization_ref = (
        "refs/tags/popgp-via000-r3-authorization-"
        + hashlib.sha256(record_bytes).hexdigest()
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
    invalid_oid = _git_bytes_input(
        repo, invalid, "hash-object", "-w", "-t", "tag", "--stdin"
    ).decode("ascii").strip()

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
    for workspace in roots.values():
        manifest_path = workspace / "evidence/evidence-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stdout_entry = next(
            entry for entry in manifest if entry["role"] == "mutation-suite-stdout"
        )
        stderr_entry = next(
            entry for entry in manifest if entry["role"] == "mutation-suite-stderr"
        )
        suite_entry = next(
            entry for entry in manifest if entry["role"] == "mutation-suite-result"
        )
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
            if (
                match := re.match(
                    r"^(tests/\S+::\S+)\s+PASSED(?:\s+\[.*\])?$", line.strip()
                )
            )
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
        "(_ for _ in ()).throw(ValueError('attestation rejected'))"
        if attestation_error
        else "None"
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
def test_dispatch_guard_rejects_wrong_event_ref_sha_or_head(
    tmp_path: Path, mutation: str
) -> None:
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
    repo, snapshot, protocol_ref, authorization_ref, _authorization_commit = (
        _authorized_repo(tmp_path)
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
    repo, snapshot, protocol_ref, authorization_ref, authorization_commit = (
        _authorized_repo(tmp_path)
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
    repo, snapshot, protocol_ref, authorization_ref, authorization_commit = (
        _authorized_repo(tmp_path)
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
        original_run(
            ["git", "-C", str(repo), *command], check=True, capture_output=True
        )
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
        if (
            stage == "verify"
            and isinstance(command, list)
            and "verify-tag" in command
        ):
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
    repo, _snapshot, protocol_ref, authorization_ref, authorization_commit = (
        _authorized_repo(tmp_path)
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
    repo, _snapshot, protocol_ref, authorization_ref, _authorization_commit = (
        _authorized_repo(tmp_path)
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
        assembler._run_frozen_precommit_validator(
            args, protocol, authorization, raw_path
        )
    assert not (tmp_path / "assembled").exists()


@pytest.mark.negative_control
def test_r3_frozen_validator_executes_snapshot_bundle_and_real_packet(tmp_path: Path) -> None:
    assembler = _load_module(ASSEMBLER, "r3_frozen_validator_bundle_execution")
    repo, _snapshot, protocol_ref, authorization_ref, _authorization_commit = (
        _authorized_repo(tmp_path)
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
def test_r3_cross_platform_git_blob_identity_with_autocrlf(
    tmp_path: Path, autocrlf: str
) -> None:
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
    trigger = workflow.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger
    assert "push:" not in trigger
    for token in (
        "${{ github.ref }}",
        "${{ github.sha }}",
        "${{ inputs.authorization_ref }}",
        "--expected-authorization-tag-oid",
        "authorization ref changed before platform execution",
        "${{ github.run_id }}",
        "${{ github.run_attempt }}",
        "VIA-000-DISPATCH-GUARD.py",
    ):
        assert token in workflow


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
