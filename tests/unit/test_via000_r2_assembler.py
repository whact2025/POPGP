from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tests.unit.test_viability_raw_evidence_contract import (
    CANDIDATE_COMMIT,
    CANDIDATE_TREE,
    CONTRACT,
    PLATFORMS,
    PROTOCOL_PATH,
    ROOT,
    SCHEMA_SOURCE,
    _fixture,
    _validate,
)

ASSEMBLER = ROOT / "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-ASSEMBLER.py"


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _assembler_inputs(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, object], dict[str, object]]:
    packet, receipts, context = _fixture(tmp_path / "fixture")
    document = context["raw_document"]
    receipt_dir = context["raw_path"].parent
    roots: dict[str, Path] = {}
    mutation_files: dict[str, Path] = {}
    for platform in PLATFORMS:
        workspace = tmp_path / f"runner-{platform}"
        roots[platform] = workspace
        entries = [
            copy.deepcopy(entry)
            for entry in document["evidence_manifest"]
            if entry["platform_family"] == platform and entry["role"] != "mutation-result"
        ]
        for entry in entries:
            source = receipt_dir / entry["path"]
            target = workspace / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        summary = copy.deepcopy(document["platforms"][platform])
        summary["mutation_results"] = []
        summary["mutation_count"] = 0
        summary["mutations_rejected"] = False
        summary["overall_passed"] = False
        summary["evidence_paths"] = sorted(entry["path"] for entry in entries)
        _write_json(workspace / "evidence/evidence-manifest.json", entries)
        _write_json(workspace / "evidence/platform-summary.json", summary)

        mutation_root = tmp_path / f"mutations-{platform}"
        mutations = []
        for mutation in document["platforms"][platform]["mutation_results"]:
            source = receipt_dir / mutation["evidence_paths"][0]
            target = mutation_root / f"{mutation['mutation_id']}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            mutations.append(
                {
                    "mutation_id": mutation["mutation_id"],
                    "rejected": True,
                    "evidence": [
                        {
                            "path": target.name,
                            "media_type": "application/json",
                        }
                    ],
                }
            )
        mutation_file = mutation_root / "mutations.json"
        _write_json(
            mutation_file,
            {
                "schema_version": 1,
                "platform_family": platform,
                "candidate_commit": CANDIDATE_COMMIT,
                "candidate_tree": CANDIDATE_TREE,
                "mutations": mutations,
            },
        )
        mutation_files[platform] = mutation_file
    return roots, mutation_files, packet, receipts


def _run_assembler(
    roots: dict[str, Path], mutation_files: dict[str, Path], output: Path
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ASSEMBLER),
        "--protocol",
        str(PROTOCOL_PATH),
        "--schema",
        str(SCHEMA_SOURCE),
    ]
    for platform in CONTRACT["required_platforms"]:
        command.extend(["--platform-root", f"{platform}={roots[platform]}"])
        command.extend(["--mutation-file", f"{platform}={mutation_files[platform]}"])
    command.extend(
        [
            "--output-dir",
            str(output),
            "--committed-by",
            "test-runner",
            "--committed-at",
            "2026-08-20T00:00:00Z",
        ]
    )
    return subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)


def test_assembler_roundtrip_produces_valid_raw_results_and_commitment(
    tmp_path: Path,
) -> None:
    roots, mutation_files, packet, receipts = _assembler_inputs(tmp_path)
    output = tmp_path / "assembled"
    result = _run_assembler(roots, mutation_files, output)
    assert result.returncode == 0, result.stderr
    raw_path = output / "raw-results.json"
    commitment = json.loads((output / "output-commitment.json").read_text())
    assert commitment["raw_results_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    context = {
        "campaign_base": tmp_path,
        "raw_path": raw_path,
        "raw_document": json.loads(raw_path.read_text()),
    }
    receipts["raw-results"]["_resolved_path"] = raw_path
    receipts["raw-results-schema"]["_resolved_path"] = SCHEMA_SOURCE
    assert _validate(packet, receipts, context) == []


def test_assembler_rejects_partial_or_failed_fragments_without_commitment(
    tmp_path: Path,
) -> None:
    roots, mutation_files, _packet, _receipts = _assembler_inputs(tmp_path)
    platform = PLATFORMS[0]
    summary_path = roots[platform] / "evidence/platform-summary.json"
    summary = json.loads(summary_path.read_text())
    summary["command_results"]["006-pytest"]["exit_code"] = 1
    _write_json(summary_path, summary)
    output = tmp_path / "failed-output"
    result = _run_assembler(roots, mutation_files, output)
    assert result.returncode != 0
    assert not output.exists()

    roots, mutation_files, _packet, _receipts = _assembler_inputs(tmp_path / "missing")
    mutation_files.pop(PLATFORMS[1])
    output = tmp_path / "missing-output"
    command = [
        sys.executable,
        str(ASSEMBLER),
        "--platform-root",
        f"{PLATFORMS[0]}={roots[PLATFORMS[0]]}",
        "--platform-root",
        f"{PLATFORMS[1]}={roots[PLATFORMS[1]]}",
        "--mutation-file",
        f"{PLATFORMS[0]}={mutation_files[PLATFORMS[0]]}",
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
