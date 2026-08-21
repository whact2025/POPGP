from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.check_viability_campaign import _validate_custody
from tests.unit.test_viability_raw_evidence_contract import (
    CONTRACT,
    PLATFORMS,
    PROTOCOL_PATH,
    PROTOCOL_SOURCE_COMMIT,
    ROOT,
    SCHEMA_SOURCE,
    _fixture,
    _validate,
)

ASSEMBLER = ROOT / "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-ASSEMBLER.py"
PACKET = ROOT / "reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"


@pytest.fixture(autouse=True)
def _trusted_attestation_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.check_viability_campaign._github_attestation_errors",
        lambda *_args, **_kwargs: [],
    )


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _assembler_inputs(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, object], dict[str, object]]:
    packet, receipts, context = _fixture(tmp_path / "fixture")
    document = context["raw_document"]
    receipt_dir = context["raw_path"].parent
    roots: dict[str, Path] = {}
    for platform in PLATFORMS:
        workspace = tmp_path / f"runner-{platform}"
        roots[platform] = workspace
        prefix = f"evidence/{platform}/"
        entries = []
        for entry in document["evidence_manifest"]:
            if entry["platform_family"] != platform or entry["role"].startswith("producer-"):
                continue
            source = receipt_dir / entry["path"]
            original = copy.deepcopy(entry)
            original["path"] = original["path"].removeprefix(prefix)
            entries.append(original)
            target = workspace / original["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for entry in entries:
            assert (workspace / entry["path"]).is_file()
        provenance = receipt_dir / f"evidence/{platform}/provenance"
        summary = json.loads((provenance / "platform-summary.json").read_text())
        _write_json(workspace / "evidence/evidence-manifest.json", entries)
        _write_json(workspace / "evidence/platform-summary.json", summary)
        shutil.copyfile(
            provenance / "producer-attestation.sigstore.json",
            workspace / "evidence/producer-attestation.sigstore.json",
        )
    return roots, packet, receipts


def _run_assembler(
    roots: dict[str, Path],
    output: Path,
    *,
    committed_by: str = "test-runner",
) -> subprocess.CompletedProcess[str]:
    bootstrap = (
        "import importlib.util,sys;"
        "path=sys.argv.pop(1);"
        "spec=importlib.util.spec_from_file_location('via000_assembler_test',path);"
        "module=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(module);"
        "module._verify_attestation=lambda *args,**kwargs:None;"
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
        str(PROTOCOL_PATH),
        "--schema",
        str(SCHEMA_SOURCE),
    ]
    for platform in CONTRACT["required_platforms"]:
        command.extend(["--platform-root", f"{platform}={roots[platform]}"])
    command.extend(
        [
            "--protocol-source-commit",
            PROTOCOL_SOURCE_COMMIT,
            "--output-dir",
            str(output),
            "--committed-by",
            committed_by,
            "--committed-at",
            "2026-08-20T00:00:00Z",
        ]
    )
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_assembler_roundtrip_produces_valid_raw_results_and_commitment(
    tmp_path: Path,
) -> None:
    roots, packet, receipts = _assembler_inputs(tmp_path)
    output = tmp_path / "assembled"
    result = _run_assembler(roots, output)
    assert result.returncode == 0, result.stderr
    raw_path = output / "raw-results.json"
    commitment = json.loads((output / "output-commitment.json").read_text())
    assert commitment == {
        "packet_id": "VIA-000",
        "committed_by": "test-runner",
        "committed_at": "2026-08-20T00:00:00Z",
        "output_receipt_id": "raw-results",
        "output_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    context = {
        "campaign_base": tmp_path,
        "raw_path": raw_path,
        "raw_document": json.loads(raw_path.read_text()),
    }
    receipts["raw-results"]["_resolved_path"] = raw_path
    receipts["raw-results-schema"]["_resolved_path"] = SCHEMA_SOURCE
    assert _validate(packet, receipts, context) == []


def test_assembler_commitment_roundtrips_through_custody_reveal(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    tmp_path = tmp_path_factory.mktemp("custody")
    roots, _packet, _receipts = _assembler_inputs(tmp_path)
    packet = yaml.safe_load(PACKET.read_text(encoding="utf-8"))
    runner_identity = packet["seats"]["reproduction_runner"]["agent_identity"]
    custodian_identity = packet["seats"]["evaluator_custodian"]["agent_identity"]
    output = tmp_path / "custody-output"
    result = _run_assembler(
        roots,
        output,
        committed_by=runner_identity,
    )
    assert result.returncode == 0, result.stderr

    raw_path = output / "raw-results.json"
    commitment_path = output / "output-commitment.json"
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    holdout_path = tmp_path / "revealed-holdout.json"
    seed_path = tmp_path / "revealed-seed.json"
    holdout_path.write_text('{"labels": [0, 1]}\n', encoding="utf-8")
    seed_path.write_text('{"seeds": [17, 29]}\n', encoding="utf-8")
    holdout_hash = hashlib.sha256(holdout_path.read_bytes()).hexdigest()
    seed_hash = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    reveal_path = tmp_path / "reveal-record.json"
    reveal_document = {
        "packet_id": "VIA-000",
        "authorized_by": custodian_identity,
        "revealed_at": "2026-08-20T01:00:00Z",
        "output_commitment_receipt_id": "output-commitment",
        "post_reveal_holdout_sha256": holdout_hash,
        "post_reveal_seed_sha256": seed_hash,
    }
    _write_json(reveal_path, reveal_document)

    packet["lifecycle_phase"] = "reproduced"
    packet["holdout_started"] = True
    custody = packet["blind_custody"]
    custody["hidden_holdout_manifest"]["sha256"] = holdout_hash
    custody["secret_seed_manifest"]["sha256"] = seed_hash
    custody["output_commitment"] = {
        "receipt_id": "output-commitment",
        **commitment,
    }
    custody["reveal"] = {
        "status": "revealed",
        "authorized_by": custodian_identity,
        "revealed_at": reveal_document["revealed_at"],
        "post_reveal_holdout_sha256": holdout_hash,
        "post_reveal_seed_sha256": seed_hash,
        "post_reveal_holdout_receipt_id": "revealed-holdout",
        "post_reveal_seed_receipt_id": "revealed-seed",
        "reveal_receipt_id": "reveal-record",
    }

    receipts = {
        "raw-results": {
            "kind": "raw-results",
            "media_type": "application/json",
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "_resolved_path": raw_path,
        },
        "output-commitment": {
            "kind": "output-commitment",
            "media_type": "application/json",
            "sha256": hashlib.sha256(commitment_path.read_bytes()).hexdigest(),
            "_resolved_path": commitment_path,
        },
        "revealed-holdout": {
            "kind": "revealed-manifest",
            "media_type": "application/json",
            "sha256": holdout_hash,
            "_resolved_path": holdout_path,
        },
        "revealed-seed": {
            "kind": "revealed-manifest",
            "media_type": "application/json",
            "sha256": seed_hash,
            "_resolved_path": seed_path,
        },
        "reveal-record": {
            "kind": "reveal-record",
            "media_type": "application/json",
            "sha256": hashlib.sha256(reveal_path.read_bytes()).hexdigest(),
            "_resolved_path": reveal_path,
        },
    }
    assert _validate_custody(packet, receipts, "VIA-000") == []


def test_assembler_rejects_partial_or_failed_fragments_without_commitment(
    tmp_path: Path,
) -> None:
    roots, _packet, _receipts = _assembler_inputs(tmp_path)
    platform = PLATFORMS[0]
    summary_path = roots[platform] / "evidence/platform-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["command_results"]["006-pytest"]["exit_code"] = 1
    _write_json(summary_path, summary)
    output = tmp_path / "failed-output"
    result = _run_assembler(roots, output)
    assert result.returncode != 0
    assert not output.exists()

    roots, _packet, _receipts = _assembler_inputs(tmp_path / "missing")
    roots.pop(PLATFORMS[1])
    output = tmp_path / "missing-output"
    command = [
        sys.executable,
        str(ASSEMBLER),
        "--platform-root",
        f"{PLATFORMS[0]}={roots[PLATFORMS[0]]}",
        "--protocol-source-commit",
        PROTOCOL_SOURCE_COMMIT,
        "--output-dir",
        str(output),
        "--committed-by",
        "test-runner",
        "--committed-at",
        "2026-08-20T00:00:00Z",
    ]
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert not output.exists()


def test_assembler_semantic_gate(tmp_path: Path) -> None:
    roots, _packet, _receipts = _assembler_inputs(tmp_path)
    platform = PLATFORMS[0]
    workspace = roots[platform]
    manifest_path = workspace / "evidence/evidence-manifest.json"
    summary_path = workspace / "evidence/platform-summary.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    pdf_entry = next(entry for entry in manifest if entry["role"] == "pdf")
    pdf_path = workspace / pdf_entry["path"]
    pdf_path.write_bytes(b"%PDF-1.4\n" + b"not-a-pdf-object\n" * 7_000 + b"%%EOF\n")
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    pdf_entry["sha256"] = digest
    pdf_entry["byte_count"] = pdf_path.stat().st_size
    summary["pdf_sha256"] = digest
    _write_json(manifest_path, manifest)
    _write_json(summary_path, summary)

    output = tmp_path / "o"
    result = _run_assembler(roots, output)
    assert result.returncode != 0
    assert "authoritative validation" in result.stderr
    assert not output.exists()
