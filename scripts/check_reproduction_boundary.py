"""Snapshot and verify a clean locked Python reproduction boundary.

The snapshot is taken immediately after a clean ``uv sync --frozen --no-editable``.
It records every Python startup hook installed by the lock on the current platform.
Verification requires byte-for-byte equality with that measured surface, rather than
assuming that every platform installs the same hard-coded ``.pth`` file set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
BLOCKED_ENVIRONMENT = (
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
)
CUSTOMIZE_NAMES = frozenset({"sitecustomize.py", "usercustomize.py"})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json(raw: str, *, source: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        return json.loads(raw, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid strict JSON: {exc}") from exc


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate_environment_directory(repo_root: Path, environment_path: Path) -> Path:
    repo_root = repo_root.resolve()
    unresolved_environment_path = repo_root / environment_path
    if (
        not unresolved_environment_path.is_dir()
        or unresolved_environment_path.is_symlink()
    ):
        raise ValueError(
            f"regular environment directory required: {unresolved_environment_path}"
        )
    environment_path = unresolved_environment_path.resolve()
    try:
        environment_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("environment directory must be within the repository") from exc
    return environment_path


def collect_startup_surface(repo_root: Path, environment_path: Path) -> dict[str, Any]:
    """Return a deterministic manifest of executable Python startup surfaces."""

    environment_path = _validate_environment_directory(repo_root, environment_path)
    entries: list[dict[str, Any]] = []
    for path in sorted(environment_path.rglob("*")):
        if path.is_symlink():
            if path.suffix == ".pth" or path.name in CUSTOMIZE_NAMES:
                raise ValueError(f"startup surface must be a regular file: {path}")
            continue
        if not path.is_file():
            continue
        if path.suffix != ".pth" and path.name not in CUSTOMIZE_NAMES:
            continue
        raw = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(environment_path).as_posix(),
                "size_bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
    return {"manifest_version": MANIFEST_VERSION, "entries": entries}


def write_snapshot(
    repo_root: Path,
    environment_path: Path,
    output_path: Path,
) -> str:
    repo_root = repo_root.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise ValueError("startup manifest must be written outside the candidate repository")
    manifest = collect_startup_surface(repo_root, environment_path)
    raw = _canonical_json(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    return sha256_bytes(raw)


def verify_snapshot(
    repo_root: Path,
    environment_path: Path,
    manifest_path: Path,
    expected_sha256: str,
    *,
    enforce_process_environment: bool = True,
) -> list[str]:
    errors: list[str] = []
    if enforce_process_environment:
        blocked = sorted(name for name in BLOCKED_ENVIRONMENT if os.environ.get(name))
        if blocked:
            errors.append(f"blocked environment variables are set: {blocked}")
        if sys.prefix != sys.base_prefix:
            errors.append("base interpreter required")

    try:
        raw = manifest_path.resolve().read_bytes()
    except OSError as exc:
        return [*errors, f"could not read startup manifest: {exc}"]
    actual_sha256 = sha256_bytes(raw)
    if actual_sha256 != expected_sha256:
        errors.append(
            f"startup manifest hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        expected = _strict_json(raw.decode("utf-8"), source=str(manifest_path))
    except (UnicodeDecodeError, ValueError) as exc:
        return [*errors, str(exc)]
    observed = collect_startup_surface(repo_root, environment_path)
    if expected != observed:
        errors.append(
            "Python startup surface changed after locked sync: "
            f"expected={expected!r}, observed={observed!r}"
        )
    return errors


def _git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def check_repository_residue(repo_root: Path, allowed_paths: set[str]) -> list[str]:
    """Reject untracked files and tracked changes outside ``allowed_paths``."""

    changed = {
        line for line in _git_output(repo_root, "diff", "--name-only", "HEAD").splitlines() if line
    }
    untracked = {
        line
        for line in _git_output(
            repo_root,
            "ls-files",
            "--others",
            "--exclude-standard",
        ).splitlines()
        if line
    }
    errors: list[str] = []
    unexpected = changed - allowed_paths
    if unexpected:
        errors.append(f"unexpected tracked repository changes: {sorted(unexpected)}")
    if untracked:
        errors.append(f"unexpected untracked repository paths: {sorted(untracked)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--environment", type=Path, default=Path(".venv"))
    subparsers = parser.add_subparsers(dest="operation", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    try:
        if args.operation == "snapshot":
            print(write_snapshot(args.repo_root, args.environment, args.output))
            return 0
        errors = verify_snapshot(
            args.repo_root,
            args.environment,
            args.manifest,
            args.expected_sha256,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"reproduction boundary could not be checked: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("Reproduction boundary failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Locked Python startup surface is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
