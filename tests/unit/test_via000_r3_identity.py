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
DISPATCH_IDENTITY = {
    "event_name": "workflow_dispatch",
    "source_ref": FAKE_REF,
    "protocol_snapshot_commit": FAKE_COMMIT,
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
        else "None"
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
        "import scripts.check_viability_campaign as campaign;"
        "campaign._github_attestation_errors=lambda *args,**kwargs:[];"
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
            "--protocol-source-commit",
            FAKE_COMMIT,
            "--protocol-source-ref",
            FAKE_REF,
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
    repo, commit, ref = _snapshot_repo(tmp_path)
    guard.verify_workflow_dispatch(
        repo,
        event_name="workflow_dispatch",
        github_ref=ref,
        github_sha=commit,
        expected_commit=commit,
    )


@pytest.mark.negative_control
@pytest.mark.parametrize("mutation", ["event", "ref", "sha", "head"])
def test_dispatch_guard_rejects_wrong_event_ref_sha_or_head(
    tmp_path: Path, mutation: str
) -> None:
    guard = _load_module(GUARD, f"r3_dispatch_guard_{mutation}")
    repo, commit, ref = _snapshot_repo(tmp_path)
    event_name = "workflow_dispatch"
    github_ref = ref
    github_sha = commit
    if mutation == "event":
        event_name = "push"
    elif mutation == "ref":
        github_ref = "refs/heads/master"
    elif mutation == "sha":
        github_sha = "f" * 40
    else:
        (repo / "lifecycle.txt").write_text("later lifecycle head\n", encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "lifecycle handoff")
    with pytest.raises(ValueError):
        guard.verify_workflow_dispatch(
            repo,
            event_name=event_name,
            github_ref=github_ref,
            github_sha=github_sha,
            expected_commit=commit,
        )


@pytest.mark.negative_control
def test_r3_workflow_is_manual_only_and_binds_exact_identity() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    trigger = workflow.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger
    assert "push:" not in trigger
    for token in (
        "${{ github.ref }}",
        "${{ github.sha }}",
        "${{ inputs.protocol_snapshot_commit }}",
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
    repo, commit, ref = _snapshot_repo(tmp_path, protocol_tree=True)
    protocol = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
    schema = repo / "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
    args = SimpleNamespace(
        repo_root=repo,
        protocol=protocol,
        schema=schema,
        protocol_source_commit=commit,
        protocol_source_ref=ref,
    )
    assembler._verify_protocol_identity(args, json.loads(protocol.read_text(encoding="utf-8")))

    roots = _r3_platform_roots(tmp_path / "roundtrip")
    output = tmp_path / "assembled"
    result = _run_assembler(roots, output)
    assert result.returncode == 0, result.stderr
    assert (output / "raw-results.json").is_file()
    assert (output / "output-commitment.json").is_file()
