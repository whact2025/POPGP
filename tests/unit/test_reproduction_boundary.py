from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.check_reproduction_boundary import (
    check_repository_residue,
    collect_startup_surface,
    verify_snapshot,
    write_snapshot,
)


def _environment(tmp_path: Path) -> Path:
    environment = tmp_path / ".venv"
    site_packages = environment / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "_virtualenv.pth").write_text("import _virtualenv\n", encoding="utf-8")
    return environment


def test_platform_installed_startup_surface_is_measured_not_hardcoded(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    site_packages = environment / "Lib" / "site-packages"
    (site_packages / "_cuda_bindings_redirector.pth").write_text(
        "import _cuda_bindings_redirector\n",
        encoding="utf-8",
    )

    manifest = collect_startup_surface(tmp_path, Path(".venv"))

    assert [entry["path"] for entry in manifest["entries"]] == [
        "Lib/site-packages/_cuda_bindings_redirector.pth",
        "Lib/site-packages/_virtualenv.pth",
    ]


@pytest.mark.negative_control
def test_snapshot_hash_and_post_sync_surface_are_fail_closed(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    manifest_path = tmp_path.parent / f"{tmp_path.name}-startup.json"
    digest = write_snapshot(tmp_path, Path(".venv"), manifest_path)
    assert digest == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert (
        verify_snapshot(
            tmp_path,
            Path(".venv"),
            manifest_path,
            digest,
            enforce_process_environment=False,
        )
        == []
    )

    site_packages = environment / "Lib" / "site-packages"
    injected = site_packages / "attack.pth"
    injected.write_text("import os; os.environ['ATTACK']='1'\n", encoding="utf-8")
    errors = verify_snapshot(
        tmp_path,
        Path(".venv"),
        manifest_path,
        digest,
        enforce_process_environment=False,
    )
    assert any("startup surface changed" in error for error in errors)

    manifest_path.write_text(json.dumps({"manifest_version": 1, "entries": []}), encoding="utf-8")
    errors = verify_snapshot(
        tmp_path,
        Path(".venv"),
        manifest_path,
        digest,
        enforce_process_environment=False,
    )
    assert any("hash mismatch" in error for error in errors)


@pytest.mark.negative_control
def test_repository_residue_allows_only_declared_generated_paths(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "boundary@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Boundary Test"],
        cwd=tmp_path,
        check=True,
    )
    generated = tmp_path / "examples/physics_qg/demo/results/validation.json"
    source = tmp_path / "source.py"
    generated.parent.mkdir(parents=True)
    generated.write_text("{}\n", encoding="utf-8")
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=tmp_path, check=True)

    generated.write_text('{"changed": true}\n', encoding="utf-8")
    allowed = {generated.relative_to(tmp_path).as_posix()}
    assert check_repository_residue(tmp_path, allowed) == []

    source.write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("residue\n", encoding="utf-8")
    errors = check_repository_residue(tmp_path, allowed)
    assert any("unexpected tracked" in error for error in errors)
    assert any("unexpected untracked" in error for error in errors)
