from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_reproduction_boundary import (
    check_repository_residue,
    collect_startup_surface,
    verify_snapshot,
    write_snapshot,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_reproduction_boundary.py"
BOOTSTRAP = Path(__file__).resolve().parents[2] / "scripts/run_without_startup_hooks.py"


def _base_python() -> Path:
    candidate = getattr(sys, "_base_executable", None)
    return Path(candidate) if candidate else Path(sys.base_prefix) / Path(sys.executable).name


def _base_environment(**updates: str) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        environment.pop(name, None)
    environment.update(updates)
    return environment


def _snapshot_with_base(tmp_path: Path, manifest_path: Path) -> str:
    completed = subprocess.run(
        [
            str(_base_python()),
            "-I",
            "-S",
            str(SCRIPT),
            "--repo-root",
            str(tmp_path),
            "--environment",
            ".venv",
            "snapshot",
            "--output",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=_base_environment(),
    )
    return completed.stdout.strip()


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
    digest = _snapshot_with_base(tmp_path, manifest_path)
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
def test_snapshot_creation_requires_base_interpreter_and_clean_environment(
    tmp_path: Path,
) -> None:
    _environment(tmp_path)
    manifest_path = tmp_path.parent / f"{tmp_path.name}-startup.json"
    with pytest.raises(ValueError, match="base interpreter|required|blocked environment"):
        write_snapshot(tmp_path, Path(".venv"), manifest_path)

    completed = subprocess.run(
        [
            str(_base_python()),
            "-I",
            "-S",
            str(SCRIPT),
            "--repo-root",
            str(tmp_path),
            "snapshot",
            "--output",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_base_environment(PYTHONPATH="attacker-controlled"),
    )
    assert completed.returncode == 2
    assert "blocked environment" in completed.stderr


@pytest.mark.negative_control
@pytest.mark.parametrize("carrier_name", ["attack.pth", "sitecustomize.py", "usercustomize.py"])
@pytest.mark.parametrize("behavior", ["persistent", "self_delete", "restore_bytes"])
def test_checked_command_rejects_transient_startup_carriers_before_execution(
    tmp_path: Path,
    carrier_name: str,
    behavior: str,
) -> None:
    environment = _environment(tmp_path)
    manifest_path = tmp_path.parent / f"{tmp_path.name}-{carrier_name}.json"
    carrier = environment / "Lib" / "site-packages" / carrier_name
    measured_content = "# measured startup surface\n"
    if behavior == "restore_bytes":
        carrier.write_text(measured_content, encoding="utf-8")
    digest = _snapshot_with_base(tmp_path, manifest_path)
    marker = tmp_path.parent / f"{tmp_path.name}-{carrier_name}.marker"
    cleanup = {
        "persistent": "",
        "self_delete": "pathlib.Path(__file__).unlink()",
        "restore_bytes": f"pathlib.Path(__file__).write_text({measured_content!r})",
    }[behavior]
    carrier.write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('executed'); "
        f"{cleanup}\n",
        encoding="utf-8",
    )

    runner = tmp_path / "runner.py"
    runner.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('child-ran')\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            str(_base_python()),
            "-I",
            "-S",
            str(SCRIPT),
            "--repo-root",
            str(tmp_path),
            "run",
            "--manifest",
            str(manifest_path),
            "--expected-sha256",
            digest,
            "--",
            str(_base_python()),
            str(runner),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_base_environment(),
    )

    assert completed.returncode == 1
    assert "pre-execution" in completed.stderr
    assert not marker.exists()


@pytest.mark.negative_control
def test_site_disabled_bootstrap_does_not_evaluate_measured_hooks(tmp_path: Path) -> None:
    environment = tmp_path / ".venv"
    subprocess.run(
        [str(_base_python()), "-I", "-S", "-m", "venv", str(environment)],
        check=True,
        env=_base_environment(),
    )
    site_packages = next(environment.glob("Lib/site-packages"), None)
    if site_packages is None:
        site_packages = next(environment.glob("lib/python*/site-packages"))
    marker = tmp_path / "startup.marker"
    (site_packages / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    module = tmp_path / "target_module.py"
    module.write_text("VALUE = 'loaded'\n", encoding="utf-8")
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    completed = subprocess.run(
        [
            str(python),
            "-I",
            "-S",
            str(BOOTSTRAP),
            "--repo-root",
            str(tmp_path),
            "--module",
            "target_module",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_base_environment(),
    )

    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


@pytest.mark.negative_control
def test_repository_residue_allows_only_declared_generated_paths(tmp_path: Path) -> None:
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


@pytest.mark.negative_control
@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_repository_boundary_ignores_no_mutable_index_flags(
    tmp_path: Path,
    index_flag: str,
) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "Boundary Test"], cwd=tmp_path)
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=tmp_path, check=True)
    subprocess.run(["git", "update-index", index_flag, "source.py"], cwd=tmp_path, check=True)
    source.write_text("VALUE = 2\n", encoding="utf-8")

    errors = check_repository_residue(tmp_path, set())

    assert any("index flags" in error for error in errors)
    assert any("unexpected tracked" in error for error in errors)


@pytest.mark.negative_control
def test_repository_boundary_rejects_ignored_staged_deleted_and_renamed_state(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "Boundary Test"], cwd=tmp_path)
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    source = tmp_path / "source.py"
    other = tmp_path / "other.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    other.write_text("OTHER = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=tmp_path, check=True)

    ignored = tmp_path / "ignored" / "attack.py"
    ignored.parent.mkdir()
    ignored.write_text("ATTACK = True\n", encoding="utf-8")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    other.rename(tmp_path / "renamed.py")

    errors = check_repository_residue(tmp_path, set())

    assert any("staged" in error for error in errors)
    assert any("unexpected tracked" in error for error in errors)
    assert any("unexpected untracked" in error for error in errors)
    assert any("unexpected ignored" in error for error in errors)


@pytest.mark.negative_control
def test_repository_boundary_rejects_symlink_substitution_when_supported(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "Boundary Test"], cwd=tmp_path)
    source = tmp_path / "source.py"
    target = tmp_path / "target.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    target.write_text("VALUE = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=tmp_path, check=True)
    source.unlink()
    try:
        source.symlink_to(target.name)
    except OSError:
        # Windows without Developer Mode cannot materialize the link in the worktree.
        # Still exercise the Git mode transition; Linux CI covers the actual link.
        source.write_text(target.name, encoding="utf-8")
        blob = subprocess.run(
            ["git", "hash-object", "-w", "source.py"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "update-index", "--cacheinfo", f"120000,{blob},source.py"],
            cwd=tmp_path,
            check=True,
        )

    errors = check_repository_residue(tmp_path, set())

    assert any(
        "unexpected tracked" in error or "staged" in error
        for error in errors
    )
