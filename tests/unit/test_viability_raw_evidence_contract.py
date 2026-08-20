from __future__ import annotations

import copy
import functools
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.check_viability_campaign import _validate_raw_evidence_contract

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json"
SCHEMA_SOURCE = ROOT / "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json"
PDF_SOURCE = (
    ROOT
    / "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner"
    / "windows/pdf/framework.pdf"
)
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
PARAMETERS = PROTOCOL["parameters"]
CONTRACT = PARAMETERS["raw_results_contract"]
CANDIDATE_COMMIT = PARAMETERS["candidate_commit"]
CANDIDATE_TREE = PARAMETERS["candidate_tree"]
PLATFORMS = tuple(CONTRACT["required_platforms"])
COMMAND_CONTRACTS = CONTRACT["required_command_contracts"]
ARTIFACT_PATHS = tuple(CONTRACT["required_artifact_paths"])
MUTATION_IDS = tuple(CONTRACT["required_mutation_ids"])
MUTATION_ORACLES = CONTRACT["required_mutation_oracles"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.cache
def _git_bytes(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{CANDIDATE_COMMIT}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _entry(
    receipt_dir: Path,
    path: Path,
    platform: str,
    media_type: str,
    role: str,
    *,
    source_path: str | None = None,
) -> dict[str, Any]:
    value = {
        "platform_family": platform,
        "path": path.relative_to(receipt_dir).as_posix(),
        "sha256": _sha256(path),
        "byte_count": path.stat().st_size,
        "media_type": media_type,
        "role": role,
    }
    if source_path is not None:
        value["source_path"] = source_path
    return value


def _command_record(contract_id: str, workspace: Path) -> tuple[str, list[str]]:
    repo = workspace / "candidate"
    environment = workspace / "environment"
    boundary = workspace / "evidence/check_reproduction_boundary.py"
    module_map = {
        "trusted-python-ruff": ("ruff", ["check", "."]),
        "trusted-python-check-tex": ("scripts.check_tex", []),
        "trusted-python-pytest": ("pytest", ["-q", "-p", "no:cacheprovider"]),
        "trusted-python-chain-generator": ("examples.physics_qg.chain_1d", []),
        "trusted-python-grid-generator": ("examples.physics_qg.grid_2d", []),
        "trusted-python-gravity-generator": ("examples.physics_qg.gravity_well", []),
        "trusted-python-source-law-generator": ("examples.physics_qg.source_law", []),
        "trusted-python-many-body-generator": ("examples.physics_qg.source_law_many_body", []),
        "trusted-python-ca-generator": ("examples.physics_qg.ca_model", []),
        "trusted-python-artifact-boundary": (
            "scripts.check_validation_artifacts",
            ["--enforce-change-boundary"],
        ),
    }
    if contract_id == "git-clone":
        return "git", ["clone", "--no-checkout", "https://github.com/whact2025/POPGP", str(repo)]
    if contract_id == "git-checkout":
        return "git", ["checkout", "--detach", CANDIDATE_COMMIT]
    if contract_id == "uv-sync-frozen-no-editable":
        return "uv", ["sync", "--frozen", "--no-editable"]
    if contract_id.startswith("pdflatex-pass-"):
        return "pdflatex", [
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={workspace / 'pdf'}",
            "docs/framework.tex",
        ]
    if contract_id == "trusted-python-environment-verify":
        return "python", [
            "-I",
            "-S",
            str(boundary),
            "--repo-root",
            str(repo),
            "--environment",
            str(environment),
            "verify",
            "--manifest",
            str(workspace / "evidence/environment-manifest.json"),
            "--expected-sha256",
            "0" * 64,
        ]
    module, module_arguments = module_map[contract_id]
    return "python", [
        "-I",
        "-S",
        str(boundary),
        "--repo-root",
        str(repo),
        "--environment",
        str(environment),
        "run",
        "--manifest",
        str(workspace / "evidence/environment-manifest.json"),
        "--expected-sha256",
        "0" * 64,
        "--",
        str(environment / "python"),
        "-I",
        "-S",
        str(repo / "scripts/run_without_startup_hooks.py"),
        "--repo-root",
        str(repo),
        "--module",
        module,
        "--",
        *module_arguments,
    ]


@functools.lru_cache(maxsize=1)
def _source_manifest() -> dict[str, Any]:
    raw = subprocess.run(
        ["git", "ls-tree", "-r", "-z", CANDIDATE_COMMIT],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    entries = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, raw_path = item.split(b"\t", 1)
        mode, _kind, object_id = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        content = _git_bytes(path)
        entries.append(
            {
                "path": path,
                "kind": "symlink" if mode == "120000" else "file",
                "mode": mode,
                "git_object_id": object_id,
                "size_bytes": len(content),
                "worktree_sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "manifest_version": 2,
        "base_ref": CANDIDATE_COMMIT,
        "base_commit": CANDIDATE_COMMIT,
        "base_tree": CANDIDATE_TREE,
        "entries": entries,
    }


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    campaign_base = tmp_path / "campaign"
    receipt_dir = campaign_base / "receipts/VIA-000"
    evidence_dir = receipt_dir / "evidence"
    receipt_dir.mkdir(parents=True)
    schema_path = receipt_dir / "raw-results.schema.json"
    schema_path.write_bytes(SCHEMA_SOURCE.read_bytes())
    raw_path = receipt_dir / "raw-results.json"
    source_manifest_bytes = json.dumps(_source_manifest()).encode()
    environment_manifest_bytes = json.dumps(
        {
            "manifest_version": 2,
            "entries": [
                {
                    "path": "bin/python",
                    "kind": "file",
                    "mode": 493,
                    "size_bytes": 7,
                    "sha256": hashlib.sha256(b"python\n").hexdigest(),
                }
            ],
        }
    ).encode()
    evidence_manifest: list[dict[str, Any]] = []
    platforms: dict[str, Any] = {}

    for platform in PLATFORMS:
        workspace = receipt_dir / platform
        command_results: dict[str, Any] = {}
        for command_id, contract_id in COMMAND_CONTRACTS.items():
            stdout = evidence_dir / platform / "commands" / f"{command_id}.stdout.txt"
            stderr = evidence_dir / platform / "commands" / f"{command_id}.stderr.txt"
            stdout_text = ""
            if command_id == "006-pytest":
                stdout_text = "366 passed in 1.00s\n"
            if command_id in {"014-pdflatex-1", "015-pdflatex-2"}:
                stdout_text = "Output written on framework.pdf (11 pages, 535368 bytes).\n"
            _write(stdout, stdout_text.encode())
            _write(stderr, b"")
            file_name, arguments = _command_record(contract_id, workspace)
            record = {
                "label": command_id,
                "contract_id": contract_id,
                "file": file_name,
                "arguments": arguments,
                "working_directory": str(workspace / "candidate"),
                "started_at": "2026-08-20T00:00:00Z",
                "finished_at": "2026-08-20T00:00:01Z",
                "duration_seconds": 1.0,
                "exit_code": 0,
                "stdout_sha256": _sha256(stdout),
                "stderr_sha256": _sha256(stderr),
            }
            result_path = stdout.with_name(f"{command_id}.result.json")
            _write(result_path, json.dumps(record).encode())
            for path, role, media in (
                (stdout, "command-stdout", "text/plain"),
                (stderr, "command-stderr", "text/plain"),
                (result_path, "command-result", "application/json"),
            ):
                evidence_manifest.append(_entry(receipt_dir, path, platform, media, role))
            command_results[command_id] = {
                "command": f"{file_name} {' '.join(arguments)}",
                "contract_id": contract_id,
                "exit_code": 0,
                "duration_seconds": 1.0,
                "result_path": result_path.relative_to(receipt_dir).as_posix(),
                "result_sha256": _sha256(result_path),
                "stdout_path": stdout.relative_to(receipt_dir).as_posix(),
                "stdout_sha256": _sha256(stdout),
                "stderr_path": stderr.relative_to(receipt_dir).as_posix(),
                "stderr_sha256": _sha256(stderr),
            }

        artifact_results: dict[str, Any] = {}
        for source_path in ARTIFACT_PATHS:
            target = evidence_dir / platform / "artifacts" / source_path
            _write(target, _git_bytes(source_path))
            media = (
                "application/json"
                if source_path.endswith(".json")
                else ("image/gif" if source_path.endswith(".gif") else "image/png")
            )
            role = "validation-json" if media == "application/json" else "visual"
            evidence_manifest.append(
                _entry(receipt_dir, target, platform, media, role, source_path=source_path)
            )
            artifact_results[source_path] = {
                "source_path": source_path,
                "evidence_path": target.relative_to(receipt_dir).as_posix(),
                "sha256": _sha256(target),
                "media_type": media,
            }

        retained: dict[str, str] = {}
        for name, content, media, role in (
            (
                "environment-manifest.json",
                environment_manifest_bytes,
                "application/json",
                "environment-manifest",
            ),
            ("source-manifest.json", source_manifest_bytes, "application/json", "source-manifest"),
            (
                "generated-status-with-ignored.txt",
                b"",
                "text/plain",
                "repository-status",
            ),
            (
                "final-status-with-ignored.txt",
                b"",
                "text/plain",
                "repository-status",
            ),
            (
                "pdf-engine-version.txt",
                (PARAMETERS["pdf_engine_banner"] + "\n").encode(),
                "text/plain",
                "pdf-engine",
            ),
            ("framework.pdf", PDF_SOURCE.read_bytes(), "application/pdf", "pdf"),
        ):
            path = evidence_dir / platform / name
            _write(path, content)
            evidence_manifest.append(_entry(receipt_dir, path, platform, media, role))
            retained[name] = _sha256(path)

        mutation_results = []
        for mutation_id in MUTATION_IDS:
            path = evidence_dir / platform / "mutations" / f"{mutation_id}.json"
            document = {
                "schema_version": 1,
                "mutation_id": mutation_id,
                "platform_family": platform,
                "candidate_commit": CANDIDATE_COMMIT,
                "candidate_tree": CANDIDATE_TREE,
                "rejected": True,
                "attack": f"frozen {mutation_id} adversarial mutation",
                "oracle_id": MUTATION_ORACLES[mutation_id],
                "oracle_errors": [
                    {
                        "error_id": MUTATION_ORACLES[mutation_id],
                        "message": "complete gate rejected mutated evidence",
                    }
                ],
                "execution": {
                    "command": f"pytest frozen::{mutation_id}",
                    "exit_code": 0,
                    "started_at": "2026-08-20T00:00:00Z",
                    "finished_at": "2026-08-20T00:00:01Z",
                    "test_ids": [f"frozen::{mutation_id}"],
                    "passed_test_count": 1,
                    "stdout_sha256": hashlib.sha256(b"1 passed\n").hexdigest(),
                    "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                },
            }
            _write(path, json.dumps(document).encode())
            evidence_manifest.append(
                _entry(receipt_dir, path, platform, "application/json", "mutation-result")
            )
            mutation_results.append(
                {
                    "mutation_id": mutation_id,
                    "rejected": True,
                    "evidence_paths": [path.relative_to(receipt_dir).as_posix()],
                }
            )

        platform_paths = [
            item["path"] for item in evidence_manifest if item["platform_family"] == platform
        ]
        platforms[platform] = {
            "schema_version": 1,
            "campaign_id": "POPGP-VIABILITY-R2-2026-08",
            "packet_id": "VIA-000",
            "platform_family": platform,
            "candidate_commit": CANDIDATE_COMMIT,
            "candidate_tree": CANDIDATE_TREE,
            "uv_version": "uv 0.11.11 (test build metadata)",
            "pdf_engine": "pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)",
            "command_results": command_results,
            "artifact_results": artifact_results,
            "mutation_results": mutation_results,
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
            "environment_manifest_sha256": retained["environment-manifest.json"],
            "source_manifest_sha256": retained["source-manifest.json"],
            "pdf_sha256": retained["framework.pdf"],
            "pdf_page_count": 11,
            "completed_at": "2026-08-20T00:00:01Z",
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
        "preregistration": {"parameters": copy.deepcopy(PARAMETERS)},
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
    return (
        packet,
        receipts,
        {
            "campaign_base": campaign_base,
            "raw_path": raw_path,
            "raw_document": raw_document,
        },
    )


def _validate(
    packet: dict[str, Any], receipts: dict[str, Any], context: dict[str, Any]
) -> list[str]:
    context["raw_path"].write_text(json.dumps(context["raw_document"]), encoding="utf-8")
    return _validate_raw_evidence_contract(
        packet, receipts, "VIA-000", context["campaign_base"], ROOT
    )


def test_raw_evidence_contract_accepts_structural_trusted_runner_fixture(tmp_path: Path) -> None:
    # This fixture exercises typed byte/semantic closure.  Execution honesty belongs
    # to the declared trusted runner/control-plane boundary and is independently
    # audited from the real hosted runner artifacts.
    packet, receipts, context = _fixture(tmp_path)
    assert _validate(packet, receipts, context) == []


def test_raw_evidence_contract_rejects_dummy_or_stale_results(tmp_path: Path) -> None:
    packet, receipts, context = _fixture(tmp_path)
    original = context["raw_document"]
    context["raw_document"] = {
        "capabilities": {
            "evidence-contract": True,
            "cross-platform-reproduction": True,
            "mutation-rejection": True,
        },
        "failed": False,
        "blocked": False,
    }
    assert any("required property" in error for error in _validate(packet, receipts, context))
    context["raw_document"] = copy.deepcopy(original)
    context["raw_document"]["platforms"][PLATFORMS[0]]["artifact_results"] = {}
    assert any("artifact_results" in error for error in _validate(packet, receipts, context))


def test_raw_evidence_contract_rejects_identity_contract_and_blockage(tmp_path: Path) -> None:
    packet, receipts, context = _fixture(tmp_path)
    context["raw_document"]["platforms"][PLATFORMS[0]]["candidate_commit"] = "0" * 40
    assert any(
        "candidate commit differs" in error for error in _validate(packet, receipts, context)
    )

    packet, receipts, context = _fixture(tmp_path / "contract")
    packet["preregistration"]["parameters"]["platform_families"] = [PLATFORMS[0]]
    assert any(
        "malformed or contradictory" in error for error in _validate(packet, receipts, context)
    )

    packet, receipts, context = _fixture(tmp_path / "blocked")
    context["raw_document"]["blocked"] = True
    assert any("False was expected" in error for error in _validate(packet, receipts, context))


def test_raw_evidence_contract_recomputes_commands_artifacts_and_mutations(tmp_path: Path) -> None:
    packet, receipts, context = _fixture(tmp_path)
    platform = context["raw_document"]["platforms"][PLATFORMS[1]]
    platform["command_results"]["006-pytest"]["contract_id"] = "trusted-python-ruff"
    assert any("executable contract" in error for error in _validate(packet, receipts, context))

    packet, receipts, context = _fixture(tmp_path / "mutation")
    platform = context["raw_document"]["platforms"][PLATFORMS[1]]
    platform["mutation_results"][0]["rejected"] = False
    assert any("mutation evidence" in error for error in _validate(packet, receipts, context))

    packet, receipts, context = _fixture(tmp_path / "mutation-detail")
    document = context["raw_document"]
    platform = document["platforms"][PLATFORMS[1]]
    mutation_path = platform["mutation_results"][0]["evidence_paths"][0]
    mutation_file = context["raw_path"].parent / mutation_path
    mutation_id = platform["mutation_results"][0]["mutation_id"]
    mutation_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mutation_id": mutation_id,
                "platform_family": PLATFORMS[1],
                "candidate_commit": CANDIDATE_COMMIT,
                "candidate_tree": CANDIDATE_TREE,
                "rejected": True,
                "attack": "generic mutation assertion",
                "oracle_errors": ["complete gate rejected mutated evidence"],
            }
        ),
        encoding="utf-8",
    )
    manifest_entry = next(
        entry for entry in document["evidence_manifest"] if entry["path"] == mutation_path
    )
    manifest_entry["sha256"] = _sha256(mutation_file)
    manifest_entry["byte_count"] = mutation_file.stat().st_size
    assert any("mutation evidence" in error for error in _validate(packet, receipts, context))

    packet, receipts, context = _fixture(tmp_path / "visual")
    platform = context["raw_document"]["platforms"][PLATFORMS[0]]
    first_visual = next(path for path in ARTIFACT_PATHS if path.endswith(".png"))
    platform["artifact_results"][first_visual]["media_type"] = "application/json"
    assert any("artifact" in error for error in _validate(packet, receipts, context))

    packet, receipts, context = _fixture(tmp_path / "status")
    document = context["raw_document"]
    final_entry = next(
        entry
        for entry in document["evidence_manifest"]
        if entry["platform_family"] == PLATFORMS[0]
        and entry["path"].endswith("final-status-with-ignored.txt")
    )
    final_file = context["raw_path"].parent / final_entry["path"]
    final_file.write_text(" M popgp/simulator.py\n", encoding="utf-8")
    final_entry["sha256"] = _sha256(final_file)
    final_entry["byte_count"] = final_file.stat().st_size
    assert any("repository-status" in error for error in _validate(packet, receipts, context))


def test_raw_evidence_contract_parses_pdf_and_compares_platform_rasters(tmp_path: Path) -> None:
    packet, receipts, context = _fixture(tmp_path / "pdf")
    document = context["raw_document"]
    for platform_name in PLATFORMS:
        pdf_entry = next(
            entry
            for entry in document["evidence_manifest"]
            if entry["platform_family"] == platform_name and entry["role"] == "pdf"
        )
        pdf_path = context["raw_path"].parent / pdf_entry["path"]
        pdf_path.write_bytes(b"%PDF-1.4\n" + b"not-a-pdf-object\n" * 7_000 + b"%%EOF\n")
        digest = _sha256(pdf_path)
        pdf_entry["sha256"] = digest
        pdf_entry["byte_count"] = pdf_path.stat().st_size
        document["platforms"][platform_name]["pdf_sha256"] = digest
    assert any("PDF evidence" in error for error in _validate(packet, receipts, context))

    packet, receipts, context = _fixture(tmp_path / "visual")
    document = context["raw_document"]
    visual_path = next(path for path in ARTIFACT_PATHS if path.endswith(".png"))
    reference = _git_bytes(visual_path)
    with Image.open(io.BytesIO(reference)) as image:
        pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    candidate_indices = np.argwhere((pixels[..., :3] >= 4) & (pixels[..., :3] <= 251))
    row, column, channel = (int(value) for value in candidate_indices[0])
    for platform_name, offset in zip(PLATFORMS, (-4, 4), strict=True):
        changed = pixels.copy()
        changed[row, column, channel] = int(changed[row, column, channel]) + offset
        result = document["platforms"][platform_name]["artifact_results"][visual_path]
        retained_path = context["raw_path"].parent / result["evidence_path"]
        Image.fromarray(changed, mode="RGBA").save(retained_path, format="PNG")
        digest = _sha256(retained_path)
        result["sha256"] = digest
        manifest_entry = next(
            entry
            for entry in document["evidence_manifest"]
            if entry["path"] == result["evidence_path"]
        )
        manifest_entry["sha256"] = digest
        manifest_entry["byte_count"] = retained_path.stat().st_size
    errors = _validate(packet, receipts, context)
    assert any("cross-platform visual" in error for error in errors)
