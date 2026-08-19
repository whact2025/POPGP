"""Snapshot and verify a clean locked reproduction boundary.

The snapshot is taken immediately after a fresh ``uv sync --frozen --no-editable``
into an external environment.  It records every file and symlink in that environment,
not only Python startup hooks.  Each checked command verifies both those bytes and the
literal Git-tree content/modes before and after execution.  Source comparison reads
the frozen blob with ``git cat-file`` and permits only the platform LF/CRLF checkout
transform, so it does not trust mutable index flags, stat caches, repository clean
filters, or ``.git/info/attributes``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 2
BLOCKED_ENVIRONMENT = (
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
)


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
    unresolved_environment_path = (
        environment_path if environment_path.is_absolute() else repo_root / environment_path
    )
    if (
        not unresolved_environment_path.is_dir()
        or unresolved_environment_path.is_symlink()
    ):
        raise ValueError(
            f"regular environment directory required: {unresolved_environment_path}"
        )
    environment_path = unresolved_environment_path.resolve()
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
    """Return a deterministic byte manifest of the complete locked environment.

    The historical function name is retained for the public API.  Version 2 covers
    every regular file and symlink so imported package code cannot sit outside the
    measured startup-hook subset.
    """

    environment_path = _validate_environment_directory(repo_root, environment_path)
    entries: list[dict[str, Any]] = []
    for path in sorted(environment_path.rglob("*")):
        if path.is_symlink():
            entries.append(
                {
                    "path": path.relative_to(environment_path).as_posix(),
                    "kind": "symlink",
                    "target": os.readlink(path),
                }
            )
            continue
        if not path.is_file():
            continue
        raw = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(environment_path).as_posix(),
                "kind": "file",
                "mode": stat.S_IMODE(path.stat().st_mode),
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
        expected_entries = {
            entry.get("path", f"<entry-{index}>"): entry
            for index, entry in enumerate(expected.get("entries", []))
            if isinstance(entry, dict)
        } if isinstance(expected, dict) else {}
        observed_entries = {
            entry.get("path", f"<entry-{index}>"): entry
            for index, entry in enumerate(observed["entries"])
        }
        differing = sorted(
            path
            for path in expected_entries.keys() | observed_entries.keys()
            if expected_entries.get(path) != observed_entries.get(path)
        )
        errors.append(
            "locked Python environment changed after fresh sync: "
            f"{len(differing)} differing paths; first paths={differing[:20]}"
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


def _literal_tree_entries(repo_root: Path, base_ref: str) -> dict[str, tuple[str, str]]:
    raw = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "--full-tree", base_ref],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    entries: dict[str, tuple[str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, encoded_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type != "blob":
            raise ValueError(
                f"unsupported tracked object type {object_type!r} at {os.fsdecode(encoded_path)!r}"
            )
        entries[os.fsdecode(encoded_path)] = (mode, object_id)
    return entries


def _git_blob_bytes_batch(repo_root: Path, object_ids: set[str]) -> dict[str, bytes]:
    """Read frozen blobs in one filter-independent Git object-database request."""

    if not object_ids:
        return {}
    ordered_ids = sorted(object_ids)
    completed = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repo_root,
        check=True,
        input=("\n".join(ordered_ids) + "\n").encode("ascii"),
        capture_output=True,
    )
    stream = memoryview(completed.stdout)
    offset = 0
    blobs: dict[str, bytes] = {}
    for requested_id in ordered_ids:
        newline = completed.stdout.find(b"\n", offset)
        if newline < 0:
            raise ValueError(f"missing cat-file header for object {requested_id}")
        header = bytes(stream[offset:newline]).decode("ascii").split()
        if len(header) != 3 or header[1] != "blob":
            raise ValueError(f"unexpected cat-file header for {requested_id}: {header!r}")
        resolved_id, _, encoded_size = header
        size = int(encoded_size)
        start = newline + 1
        end = start + size
        if end >= len(stream) or stream[end] != 0x0A:
            raise ValueError(f"truncated cat-file body for object {requested_id}")
        blobs[requested_id] = bytes(stream[start:end])
        blobs[resolved_id] = blobs[requested_id]
        offset = end + 1
    if offset != len(stream):
        raise ValueError("unexpected trailing bytes from git cat-file --batch")
    return blobs


def _portable_literal_bytes_equal(observed: bytes, expected: bytes) -> bool:
    """Compare literal content with only the checkout newline transform allowed."""

    if observed == expected:
        return True
    # Git's platform checkout may materialize CRLF for text.  Canonicalize only
    # newline spelling, never arbitrary configured clean/smudge transformations.
    # NUL-bearing blobs are treated as binary and remain byte-exact.
    if b"\0" in expected[:8000] or b"\0" in observed[:8000]:
        return False
    return observed.replace(b"\r\n", b"\n") == expected.replace(b"\r\n", b"\n")


def _literal_tree_changes(repo_root: Path, base_ref: str) -> set[str]:
    """Compare literal worktree bytes/modes to the frozen tree without Git filters."""

    changes: set[str] = set()
    entries = _literal_tree_entries(repo_root, base_ref)
    blobs = _git_blob_bytes_batch(
        repo_root,
        {object_id for _, object_id in entries.values()},
    )
    for relative_path, (expected_mode, expected_id) in entries.items():
        path = repo_root / relative_path
        expected_bytes = blobs[expected_id]
        try:
            if expected_mode == "120000":
                if not path.is_symlink():
                    changes.add(relative_path)
                    continue
                raw = os.fsencode(os.readlink(path))
            else:
                if not path.is_file() or path.is_symlink():
                    changes.add(relative_path)
                    continue
                if os.name != "nt":
                    observed_executable = bool(path.stat().st_mode & stat.S_IXUSR)
                    expected_executable = expected_mode == "100755"
                    if observed_executable != expected_executable:
                        changes.add(relative_path)
                        continue
                raw = path.read_bytes()
        except OSError:
            changes.add(relative_path)
            continue
        if not _portable_literal_bytes_equal(raw, expected_bytes):
            changes.add(relative_path)
    return changes


def write_repository_snapshot(
    repo_root: Path,
    output_path: Path,
    *,
    base_ref: str = "HEAD",
) -> str:
    """Write a literal worktree/Git-object manifest after proving a clean boundary."""

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
        raise ValueError("repository manifest must be written outside the candidate repository")

    boundary_errors = check_repository_boundary(repo_root, set(), base_ref=base_ref)
    if boundary_errors:
        raise ValueError("; ".join(boundary_errors))
    tree_entries = _literal_tree_entries(repo_root, base_ref)
    manifest_entries: list[dict[str, Any]] = []
    for relative_path, (mode, object_id) in sorted(tree_entries.items()):
        path = repo_root / relative_path
        if mode == "120000":
            observed = os.fsencode(os.readlink(path))
            kind = "symlink"
        else:
            observed = path.read_bytes()
            kind = "file"
        manifest_entries.append(
            {
                "path": relative_path,
                "kind": kind,
                "mode": mode,
                "git_object_id": object_id,
                "size_bytes": len(observed),
                "worktree_sha256": sha256_bytes(observed),
            }
        )
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "base_ref": base_ref,
        "base_commit": _git_output(repo_root, "rev-parse", base_ref).strip(),
        "base_tree": _git_output(repo_root, "rev-parse", f"{base_ref}^{{tree}}").strip(),
        "entries": manifest_entries,
    }
    raw = _canonical_json(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    return sha256_bytes(raw)


def check_repository_boundary(
    repo_root: Path,
    allowed_paths: set[str],
    *,
    base_ref: str = "HEAD",
) -> list[str]:
    """Verify actual source bytes, index state, and untracked/ignored residue.

    Literal blob hashing ignores candidate-controlled clean filters and mutable index
    state.  Ignored state is never exempt: caches and environments must be directed to
    fresh external paths by the trusted runner.
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

    changed = _literal_tree_changes(repo_root, base_ref)
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
    if ignored:
        errors.append(
            "unexpected ignored repository paths: "
            f"{sorted(ignored)}"
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
    *,
    allowed_paths: set[str],
) -> tuple[int | None, list[str]]:
    """Verify environment and literal repository bytes around one child command."""

    before = [
        *verify_snapshot(
        repo_root,
        environment_path,
        manifest_path,
        expected_sha256,
        ),
        *check_repository_boundary(repo_root, allowed_paths),
    ]
    if before:
        return None, [f"pre-execution: {error}" for error in before]
    completed = subprocess.run(command, cwd=repo_root, check=False)
    after = [
        *verify_snapshot(
            repo_root,
            environment_path,
            manifest_path,
            expected_sha256,
        ),
        *check_repository_boundary(repo_root, allowed_paths),
    ]
    return completed.returncode, [f"post-execution: {error}" for error in after]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--environment", type=Path, default=Path(".venv"))
    subparsers = parser.add_subparsers(dest="operation", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--output", type=Path, required=True)
    source_snapshot = subparsers.add_parser("source-snapshot")
    source_snapshot.add_argument("--output", type=Path, required=True)
    source_snapshot.add_argument("--base-ref", default="HEAD")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--expected-sha256", required=True)
    run.add_argument("--allow-path", action="append", default=[])
    run.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    try:
        if args.operation == "snapshot":
            print(write_snapshot(args.repo_root, args.environment, args.output))
            return 0
        if args.operation == "source-snapshot":
            print(
                write_repository_snapshot(
                    args.repo_root,
                    args.output,
                    base_ref=args.base_ref,
                )
            )
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
                allowed_paths=set(args.allow_path),
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
    print("Locked Python environment is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
