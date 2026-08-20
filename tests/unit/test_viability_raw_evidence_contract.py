from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.check_viability_campaign import _validate_raw_evidence_contract

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_SOURCE = (
    ROOT
    / "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json"
)
CANDIDATE_COMMIT = "5be3c38a0822d49953d0933f14ccab32ca12c896"
CANDIDATE_TREE = "6ad387f9f4e0bab7f97df1bb54a03177887f0707"
PLATFORMS = ("ubuntu-latest-x86_64", "windows-x86_64")
COMMAND_IDS = tuple(f"{index:03d}-command" for index in range(1, 17))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
    }


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    campaign_base = tmp_path / "campaign"
    receipt_dir = campaign_base / "receipts/VIA-000"
    evidence_dir = receipt_dir / "evidence"
    receipt_dir.mkdir(parents=True)
    schema_path = receipt_dir / "raw-results.schema.json"
    schema_path.write_bytes(SCHEMA_SOURCE.read_bytes())
    raw_path = receipt_dir / "raw-results.json"
    evidence_manifest: list[dict[str, Any]] = []
    platforms: dict[str, Any] = {}

    for platform_name in PLATFORMS:
        command_results: dict[str, Any] = {}
        platform_paths: list[str] = []
        for command_id in COMMAND_IDS:
            for stream in ("stdout", "stderr"):
                file_path = evidence_dir / platform_name / f"{command_id}.{stream}.txt"
                metadata = _write(
                    file_path, f"{platform_name} {command_id} {stream}\n".encode()
                )
                relative = file_path.relative_to(receipt_dir).as_posix()
                evidence_manifest.append(
                    {
                        "platform_family": platform_name,
                        "path": relative,
                        "sha256": metadata["sha256"],
                        "byte_count": metadata["byte_count"],
                        "media_type": "text/plain",
                    }
                )
                platform_paths.append(relative)
                command_results.setdefault(
                    command_id,
                    {
                        "command": f"trusted-command {command_id}",
                        "exit_code": 0,
                        "duration_seconds": 0.1,
                    },
                )[f"{stream}_path"] = relative
                command_results[command_id][f"{stream}_sha256"] = metadata["sha256"]
            result_path = evidence_dir / platform_name / f"{command_id}.result.json"
            result_document = {
                "label": command_id,
                "file": "trusted-command",
                "arguments": [command_id],
                "exit_code": 0,
                "duration_seconds": 0.1,
                "stdout_sha256": command_results[command_id]["stdout_sha256"],
                "stderr_sha256": command_results[command_id]["stderr_sha256"],
            }
            result_metadata = _write(
                result_path, json.dumps(result_document).encode("utf-8")
            )
            result_relative = result_path.relative_to(receipt_dir).as_posix()
            evidence_manifest.append(
                {
                    "platform_family": platform_name,
                    "path": result_relative,
                    "sha256": result_metadata["sha256"],
                    "byte_count": result_metadata["byte_count"],
                    "media_type": "application/json",
                }
            )
            platform_paths.append(result_relative)
            command_results[command_id]["result_path"] = result_relative
            command_results[command_id]["result_sha256"] = result_metadata["sha256"]

        retained_hashes: dict[str, str] = {}
        for label, media_type in (
            ("environment-manifest.json", "application/json"),
            ("source-manifest.json", "application/json"),
            ("framework.pdf", "application/pdf"),
        ):
            file_path = evidence_dir / platform_name / label
            metadata = _write(file_path, f"retained {platform_name} {label}\n".encode())
            relative = file_path.relative_to(receipt_dir).as_posix()
            evidence_manifest.append(
                {
                    "platform_family": platform_name,
                    "path": relative,
                    "sha256": metadata["sha256"],
                    "byte_count": metadata["byte_count"],
                    "media_type": media_type,
                }
            )
            platform_paths.append(relative)
            retained_hashes[label] = metadata["sha256"]

        platforms[platform_name] = {
            "platform_family": platform_name,
            "candidate_commit": CANDIDATE_COMMIT,
            "candidate_tree": CANDIDATE_TREE,
            "uv_version": "uv 0.11.11 (test build metadata)",
            "pdf_engine": "pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)",
            "command_results": command_results,
            "test_count": 366,
            "example_count": 6,
            "visual_count": 12,
            "mutation_count": 18,
            "commands_passed": True,
            "semantic_contract_passed": True,
            "visual_contract_passed": True,
            "source_boundary_passed": True,
            "environment_boundary_passed": True,
            "pdf_passed": True,
            "mutations_rejected": True,
            "overall_passed": True,
            "evidence_paths": platform_paths,
            "environment_manifest_sha256": retained_hashes[
                "environment-manifest.json"
            ],
            "source_manifest_sha256": retained_hashes["source-manifest.json"],
            "pdf_sha256": retained_hashes["framework.pdf"],
        }

    raw_document = {
        "schema_version": 1,
        "campaign_id": "POPGP-VIABILITY-R2-2026-08",
        "packet_id": "VIA-000",
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_tree": CANDIDATE_TREE,
        "platforms": platforms,
        "evidence_manifest": evidence_manifest,
        "capabilities": {
            "evidence-contract": True,
            "cross-platform-reproduction": True,
            "mutation-rejection": True,
        },
        "failed": False,
        "blocked": False,
    }
    raw_path.write_text(json.dumps(raw_document), encoding="utf-8")
    packet = {
        "packet_id": "VIA-000",
        "candidate_commit": CANDIDATE_COMMIT,
        "tree_hash": CANDIDATE_TREE,
        "lifecycle_phase": "reproduced",
        "preregistration": {
            "parameters": {
                "required_test_count": 366,
                "required_example_count": 6,
                "required_visual_count": 12,
                "required_mutation_count": 18,
                "required_command_count": 16,
                "uv_version": "0.11.11",
                "pdf_engine": "pdfTeX-1.40.29-TeX-Live-2026",
                "raw_results_contract": {
                    "schema_receipt_id": "raw-results-schema",
                    "raw_results_receipt_id": "raw-results",
                    "evidence_manifest_pointer": "/evidence_manifest",
                    "required_platforms": list(PLATFORMS),
                    "required_command_ids": list(COMMAND_IDS),
                },
            }
        },
    }
    receipts = {
        "raw-results-schema": {
            "kind": "protocol",
            "media_type": "application/schema+json",
            "_resolved_path": schema_path,
        },
        "raw-results": {
            "kind": "raw-results",
            "media_type": "application/json",
            "_resolved_path": raw_path,
        },
    }
    context = {
        "campaign_base": campaign_base,
        "raw_path": raw_path,
        "raw_document": raw_document,
        "evidence_manifest": evidence_manifest,
    }
    return packet, receipts, context


def _validate(
    packet: dict[str, Any], receipts: dict[str, Any], context: dict[str, Any]
) -> list[str]:
    context["raw_path"].write_text(
        json.dumps(context["raw_document"]), encoding="utf-8"
    )
    return _validate_raw_evidence_contract(
        packet, receipts, "VIA-000", context["campaign_base"]
    )


def test_raw_evidence_contract_accepts_hash_closed_complete_results(
    tmp_path: Path,
) -> None:
    packet, receipts, context = _fixture(tmp_path)

    assert _validate(packet, receipts, context) == []


def test_raw_evidence_contract_rejects_summary_only_and_stale_results(
    tmp_path: Path,
) -> None:
    packet, receipts, context = _fixture(tmp_path)

    summary_only = {
        "capabilities": {
            "evidence-contract": True,
            "cross-platform-reproduction": True,
            "mutation-rejection": True,
        },
        "failed": False,
        "blocked": False,
    }
    original = context["raw_document"]
    context["raw_document"] = summary_only
    errors = _validate(packet, receipts, context)
    assert any("is a required property" in error for error in errors)

    context["raw_document"] = copy.deepcopy(original)
    context["raw_document"]["capabilities"]["evidence-contract"] = False
    errors = _validate(packet, receipts, context)
    assert any("raw capability Booleans differ" in error for error in errors)


def test_raw_evidence_contract_rejects_missing_or_changed_evidence(
    tmp_path: Path,
) -> None:
    packet, receipts, context = _fixture(tmp_path)
    document = context["raw_document"]
    first_entry = document["evidence_manifest"][0]

    first_entry["sha256"] = "0" * 64
    errors = _validate(packet, receipts, context)
    assert any("hash mismatch" in error for error in errors)
    assert any("is not bound to matching retained evidence" in error for error in errors)

    packet, receipts, context = _fixture(tmp_path / "missing-platform")
    del context["raw_document"]["platforms"]["windows-x86_64"]
    errors = _validate(packet, receipts, context)
    assert any("is a required property" in error for error in errors)


def test_raw_evidence_contract_recomputes_commands_counts_and_outcome(
    tmp_path: Path,
) -> None:
    packet, receipts, context = _fixture(tmp_path)
    platform = context["raw_document"]["platforms"]["windows-x86_64"]
    platform["command_results"][COMMAND_IDS[0]]["exit_code"] = 1
    errors = _validate(packet, receipts, context)
    assert any("commands_passed is stale" in error for error in errors)
    assert any("overall_passed is stale" in error for error in errors)
    assert any("raw capability Booleans differ" in error for error in errors)

    packet, receipts, context = _fixture(tmp_path / "counts")
    context["raw_document"]["platforms"]["ubuntu-latest-x86_64"][
        "test_count"
    ] = 365
    errors = _validate(packet, receipts, context)
    assert any("overall_passed is stale" in error for error in errors)
