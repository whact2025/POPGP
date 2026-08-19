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
import tempfile
from collections.abc import Callable
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
IgnoredPathPolicy = Callable[[str], bool]


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


def _process_boundary_errors() -> list[str]:
    errors: list[str] = []
    blocked = sorted(name for name in BLOCKED_ENVIRONMENT if os.environ.get(name))
    if blocked:
        errors.append(f"blocked environment variables are set: {blocked}")
    if sys.prefix != sys.base_prefix:
        errors.append("base interpreter required")
    return errors


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
    process_errors = _process_boundary_errors()
    if process_errors:
        raise ValueError("; ".join(process_errors))
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
        errors.extend(_process_boundary_errors())

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


def _git_output(
    repo_root: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout


def _nul_git_paths(
    repo_root: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> set[str]:
    output = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        env=environment,
    ).stdout
    return {
        os.fsdecode(path)
        for path in output.split(b"\0")
        if path
    }


def _fresh_tree_changes(repo_root: Path, base_ref: str) -> set[str]:
    """Compare actual working bytes to ``base_ref`` using a pristine temporary index."""

    with tempfile.TemporaryDirectory(prefix="popgp-boundary-index-") as directory:
        index_path = Path(directory) / "index"
        environment = os.environ.copy()
        environment.update(
            {
                "GIT_INDEX_FILE": str(index_path),
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
        _git_output(repo_root, "read-tree", base_ref, environment=environment)
        # A freshly populated index has no working-tree stat cache.  Refresh it so
        # diff-files hashes clean files instead of reporting every path as modified;
        # a nonzero return is expected when declared generated artifacts differ.
        subprocess.run(
            ["git", "update-index", "--really-refresh", "--"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            env=environment,
        )
        return _nul_git_paths(
            repo_root,
            "diff-files",
            "--name-only",
            "--no-renames",
            "-z",
            "--",
            environment=environment,
        )


def is_known_runtime_ignored_path(relative_path: str) -> bool:
    """Allow only runtime state that is checked by a separate frozen mechanism."""

    path = Path(relative_path)
    parts = path.parts
    if parts and parts[0] == ".venv":
        return True
    if parts and parts[0] in {".pytest_cache", ".ruff_cache"}:
        return True
    return "__pycache__" in parts and path.suffix == ".pyc"


def check_repository_boundary(
    repo_root: Path,
    allowed_paths: set[str],
    *,
    base_ref: str = "HEAD",
    allowed_ignored_path: IgnoredPathPolicy | None = None,
) -> list[str]:
    """Verify actual source bytes, index state, and untracked/ignored residue.

    A temporary index populated directly from ``base_ref`` forces Git to inspect the
    working tree without trusting mutable assume-unchanged/skip-worktree flags or
    cached stat data in the repository index.
    """

    repo_root = repo_root.resolve()
    errors: list[str] = []

    index_entries = subprocess.run(
        ["git", "ls-files", "-z", "-v"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    flagged = sorted(
        os.fsdecode(entry[2:])
        for entry in index_entries
        if entry and (len(entry) < 2 or entry[:2] != b"H ")
    )
    if flagged:
        errors.append(
            "tracked paths carry nonordinary Git index flags: "
            f"{flagged}"
        )

    staged = _nul_git_paths(
        repo_root,
        "diff",
        "--cached",
        "--name-only",
        "--no-renames",
        "-z",
        base_ref,
        "--",
    )
    if staged:
        errors.append(f"staged repository changes exist: {sorted(staged)}")

    changed = _fresh_tree_changes(repo_root, base_ref)
    unexpected = changed - allowed_paths
    if unexpected:
        errors.append(f"unexpected tracked repository changes: {sorted(unexpected)}")

    untracked = _nul_git_paths(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if untracked:
        errors.append(f"unexpected untracked repository paths: {sorted(untracked)}")

    ignored = _nul_git_paths(
        repo_root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    unexpected_ignored = {
        path
        for path in ignored
        if allowed_ignored_path is None or not allowed_ignored_path(path)
    }
    if unexpected_ignored:
        errors.append(
            "unexpected ignored repository paths: "
            f"{sorted(unexpected_ignored)}"
        )
    return errors


def check_repository_residue(repo_root: Path, allowed_paths: set[str]) -> list[str]:
    """Reject all undeclared repository state outside ``allowed_paths``."""

    return check_repository_boundary(repo_root, allowed_paths)


def run_checked_command(
    repo_root: Path,
    environment_path: Path,
    manifest_path: Path,
    expected_sha256: str,
    command: list[str],
) -> tuple[int | None, list[str]]:
    """Verify the startup boundary before and after running one child command."""

    before = verify_snapshot(
        repo_root,
        environment_path,
        manifest_path,
        expected_sha256,
    )
    if before:
        return None, [f"pre-execution: {error}" for error in before]
    completed = subprocess.run(command, cwd=repo_root, check=False)
    after = verify_snapshot(
        repo_root,
        environment_path,
        manifest_path,
        expected_sha256,
    )
    return completed.returncode, [f"post-execution: {error}" for error in after]


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
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--expected-sha256", required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    try:
        if args.operation == "snapshot":
            print(write_snapshot(args.repo_root, args.environment, args.output))
            return 0
        if args.operation == "run":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command:
                raise ValueError("run requires a child command after --")
            returncode, errors = run_checked_command(
                args.repo_root,
                args.environment,
                args.manifest,
                args.expected_sha256,
                command,
            )
            if errors:
                print("Reproduction boundary failed:", file=sys.stderr)
                for error in errors:
                    print(f"- {error}", file=sys.stderr)
                return 1
            return int(returncode)
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
