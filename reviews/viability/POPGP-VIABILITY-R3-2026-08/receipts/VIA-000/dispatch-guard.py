"""Fail-closed identity guard for a VIA-000 R3 hosted protocol dispatch."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

SHA1_RE = re.compile(r"[0-9a-f]{40}")
TAG_PREFIX = "refs/tags/popgp-via000-r3-protocol-"


def snapshot_ref(commit: str) -> str:
    """Return the sole admissible dispatch ref for ``commit``."""

    if SHA1_RE.fullmatch(commit) is None:
        raise ValueError("protocol snapshot commit must be full lowercase 40-hex")
    return f"{TAG_PREFIX}{commit}"


def _resolve_commit(repo_root: Path, ref: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    resolved = completed.stdout.strip()
    if completed.returncode != 0 or SHA1_RE.fullmatch(resolved) is None:
        raise ValueError(f"dispatch ref does not resolve to one Git commit: {ref}")
    return resolved


def verify_protocol_ref(repo_root: Path, ref: str, expected_commit: str) -> None:
    """Require the content-addressed R3 tag to resolve to ``expected_commit``."""

    expected_ref = snapshot_ref(expected_commit)
    if ref != expected_ref:
        raise ValueError(f"dispatch ref must be {expected_ref}, observed {ref}")
    resolved = _resolve_commit(repo_root.resolve(), ref)
    if resolved != expected_commit:
        raise ValueError(
            "dispatch ref resolves to the wrong protocol snapshot "
            f"({resolved} != {expected_commit})"
        )


def verify_workflow_dispatch(
    repo_root: Path,
    *,
    event_name: str,
    github_ref: str,
    github_sha: str,
    expected_commit: str,
) -> None:
    """Bind the manual event, source ref, event SHA, and checkout HEAD."""

    if event_name != "workflow_dispatch":
        raise ValueError("R3 protocol execution requires workflow_dispatch")
    if github_sha != expected_commit:
        raise ValueError(
            "workflow source SHA differs from the expected protocol snapshot "
            f"({github_sha} != {expected_commit})"
        )
    verify_protocol_ref(repo_root, github_ref, expected_commit)
    head = _resolve_commit(repo_root.resolve(), "HEAD")
    if head != expected_commit:
        raise ValueError(
            "checkout HEAD differs from the expected protocol snapshot "
            f"({head} != {expected_commit})"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--github-ref", required=True)
    parser.add_argument("--github-sha", required=True)
    parser.add_argument("--expected-protocol-snapshot", required=True)
    args = parser.parse_args()
    verify_workflow_dispatch(
        args.repo_root,
        event_name=args.event_name,
        github_ref=args.github_ref,
        github_sha=args.github_sha,
        expected_commit=args.expected_protocol_snapshot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
