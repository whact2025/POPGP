"""Fail-closed assembler for the frozen VIA-000 R2 evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: Any) -> None:
    path.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _safe_source(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"unsafe evidence path {relative!r}")
    candidate = (root / relative).resolve()
    candidate.relative_to(root.resolve())
    if not candidate.is_file():
        raise ValueError(f"evidence file is missing: {relative}")
    return candidate


def _parse_bindings(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use PLATFORM=PATH: {value!r}")
        platform, raw_path = value.split("=", 1)
        if platform in result or not platform or not raw_path:
            raise ValueError(f"duplicate or empty {label}: {value!r}")
        result[platform] = Path(raw_path).resolve()
    return result


def _prefixed(platform: str, relative: str) -> str:
    return f"evidence/{platform}/{relative}"


def _copy_manifest_evidence(
    platform: str,
    workspace: Path,
    destination: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence_root = workspace / "evidence"
    summary_path = evidence_root / "platform-summary.json"
    manifest_path = evidence_root / "evidence-manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"{platform}: platform summary or evidence manifest is missing")
    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path)
    if not isinstance(summary, dict) or not isinstance(manifest, list):
        raise ValueError(f"{platform}: malformed platform evidence documents")

    observed_paths: set[str] = set()
    rewritten_manifest: list[dict[str, Any]] = []
    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError(f"{platform}: malformed evidence entry")
        relative = entry.get("path")
        if relative in observed_paths:
            raise ValueError(f"{platform}: duplicate evidence path {relative!r}")
        observed_paths.add(relative)
        source = _safe_source(workspace, relative)
        if source.stat().st_size != entry.get("byte_count") or _sha256(source) != entry.get(
            "sha256"
        ):
            raise ValueError(f"{platform}: evidence bytes disagree with manifest: {relative}")
        target_relative = _prefixed(platform, relative)
        target = destination / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rewritten = dict(entry)
        rewritten["path"] = target_relative
        rewritten_manifest.append(rewritten)

    if set(summary.get("evidence_paths", [])) != observed_paths:
        raise ValueError(f"{platform}: summary paths differ from evidence manifest")

    rewritten_summary = dict(summary)
    rewritten_summary["evidence_paths"] = sorted(
        _prefixed(platform, path) for path in observed_paths
    )
    for command in rewritten_summary.get("command_results", {}).values():
        for field in ("result_path", "stdout_path", "stderr_path"):
            command[field] = _prefixed(platform, command[field])
    for artifact in rewritten_summary.get("artifact_results", {}).values():
        artifact["evidence_path"] = _prefixed(platform, artifact["evidence_path"])
    return rewritten_summary, rewritten_manifest


def _copy_mutations(
    platform: str,
    mutation_path: Path,
    destination: Path,
    candidate_commit: str,
    candidate_tree: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    document = _load_json(mutation_path)
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "platform_family",
        "candidate_commit",
        "candidate_tree",
        "mutations",
    }:
        raise ValueError(f"{platform}: malformed mutation document")
    if (
        document["schema_version"] != 1
        or document["platform_family"] != platform
        or document["candidate_commit"] != candidate_commit
        or document["candidate_tree"] != candidate_tree
    ):
        raise ValueError(f"{platform}: mutation identity mismatch")
    mutations = document["mutations"]
    if not isinstance(mutations, list):
        raise ValueError(f"{platform}: mutations must be an array")
    results: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for mutation in mutations:
        if not isinstance(mutation, dict) or set(mutation) != {
            "mutation_id",
            "rejected",
            "evidence",
        }:
            raise ValueError(f"{platform}: malformed mutation record")
        if mutation["rejected"] is not True:
            raise ValueError(f"{platform}: mutation was not rejected: {mutation['mutation_id']}")
        evidence_paths: list[str] = []
        evidence_items = mutation["evidence"]
        if not isinstance(evidence_items, list) or not evidence_items:
            raise ValueError(f"{platform}: mutation evidence is empty")
        for item in evidence_items:
            if not isinstance(item, dict) or set(item) != {"path", "media_type"}:
                raise ValueError(f"{platform}: malformed mutation evidence")
            source = _safe_source(mutation_path.parent, item["path"])
            target_relative = _prefixed(
                platform, f"mutations/{mutation['mutation_id']}/{item['path']}"
            )
            target = destination / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            evidence_paths.append(target_relative)
            entries.append(
                {
                    "platform_family": platform,
                    "path": target_relative,
                    "sha256": _sha256(target),
                    "byte_count": target.stat().st_size,
                    "media_type": item["media_type"],
                    "role": "mutation-result",
                }
            )
        results.append(
            {
                "mutation_id": mutation["mutation_id"],
                "rejected": True,
                "evidence_paths": evidence_paths,
            }
        )
    return results, entries


def assemble(args: argparse.Namespace) -> None:
    protocol = _load_json(args.protocol)
    schema = _load_json(args.schema)
    contract = protocol["parameters"]["raw_results_contract"]
    required_platforms = contract["required_platforms"]
    platform_roots = _parse_bindings(args.platform_root, "platform root")
    mutation_files = _parse_bindings(args.mutation_file, "mutation file")
    if set(platform_roots) != set(required_platforms) or set(mutation_files) != set(
        required_platforms
    ):
        raise ValueError("platform and mutation inputs must exactly match frozen platforms")
    if args.output_dir.exists():
        raise ValueError(f"output directory must not exist: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent)
    )
    try:
        platforms: dict[str, Any] = {}
        manifest: list[dict[str, Any]] = []
        expected_mutations = set(contract["required_mutation_ids"])
        for platform in required_platforms:
            summary, platform_manifest = _copy_manifest_evidence(
                platform, platform_roots[platform], temporary
            )
            mutation_results, mutation_manifest = _copy_mutations(
                platform,
                mutation_files[platform],
                temporary,
                protocol["parameters"]["candidate_commit"],
                protocol["parameters"]["candidate_tree"],
            )
            if summary.get("candidate_commit") != protocol["parameters"]["candidate_commit"]:
                raise ValueError(f"{platform}: candidate commit mismatch")
            if summary.get("candidate_tree") != protocol["parameters"]["candidate_tree"]:
                raise ValueError(f"{platform}: candidate tree mismatch")
            if set(summary.get("command_results", {})) != set(
                contract["required_command_contracts"]
            ):
                raise ValueError(f"{platform}: incomplete command set")
            if any(item.get("exit_code") != 0 for item in summary["command_results"].values()):
                raise ValueError(f"{platform}: nonzero command result")
            if set(summary.get("artifact_results", {})) != set(contract["required_artifact_paths"]):
                raise ValueError(f"{platform}: incomplete artifact set")
            if {item["mutation_id"] for item in mutation_results} != expected_mutations:
                raise ValueError(f"{platform}: incomplete mutation set")
            summary["mutation_results"] = mutation_results
            summary["mutation_count"] = len(mutation_results)
            summary["mutations_rejected"] = all(item["rejected"] for item in mutation_results)
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
            summary["evidence_paths"] = sorted(
                [*summary["evidence_paths"], *[entry["path"] for entry in mutation_manifest]]
            )
            platforms[platform] = summary
            manifest.extend(platform_manifest)
            manifest.extend(mutation_manifest)

        raw_document = {
            "schema_version": 1,
            "campaign_id": "POPGP-VIABILITY-R2-2026-08",
            "packet_id": "VIA-000",
            "candidate_commit": protocol["parameters"]["candidate_commit"],
            "candidate_tree": protocol["parameters"]["candidate_tree"],
            "platforms": platforms,
            "evidence_manifest": sorted(manifest, key=lambda item: item["path"]),
            "capabilities": {
                "evidence-contract": all(item["commands_passed"] for item in platforms.values()),
                "cross-platform-reproduction": all(
                    all(
                        item[field]
                        for field in (
                            "commands_passed",
                            "semantic_contract_passed",
                            "visual_contract_passed",
                            "source_boundary_passed",
                            "environment_boundary_passed",
                            "pdf_passed",
                        )
                    )
                    for item in platforms.values()
                ),
                "mutation-rejection": all(
                    item["mutations_rejected"] for item in platforms.values()
                ),
            },
            "blocked": False,
        }
        raw_document["failed"] = not all(raw_document["capabilities"].values())
        schema_errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(raw_document),
            key=lambda error: list(error.path),
        )
        if schema_errors:
            raise ValueError("assembled raw results fail schema: " + schema_errors[0].message)
        raw_path = temporary / "raw-results.json"
        _write_json(raw_path, raw_document)
        commitment = {
            "schema_version": 1,
            "campaign_id": raw_document["campaign_id"],
            "packet_id": raw_document["packet_id"],
            "candidate_commit": raw_document["candidate_commit"],
            "candidate_tree": raw_document["candidate_tree"],
            "committed_by": args.committed_by,
            "committed_at": args.committed_at,
            "raw_results_path": "raw-results.json",
            "raw_results_sha256": _sha256(raw_path),
        }
        _write_json(temporary / "output-commitment.json", commitment)
        os.replace(temporary, args.output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--protocol", type=Path, default=here / "VIA-000.json")
    parser.add_argument("--schema", type=Path, default=here / "VIA-000-RAW-RESULTS.schema.json")
    parser.add_argument("--platform-root", action="append", default=[], required=True)
    parser.add_argument("--mutation-file", action="append", default=[], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--committed-by", required=True)
    parser.add_argument("--committed-at", required=True)
    args = parser.parse_args()
    assemble(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
