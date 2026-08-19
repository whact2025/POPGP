#!/usr/bin/env python3
"""Execute the frozen VIA-000 protocol and mutation plan with raw receipts.

This harness lives on the campaign runner branch.  It never imports candidate code
and writes all evidence outside the exact-candidate checkout supplied by the caller.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CANDIDATE_COMMIT = "9a29e05f803666bf0e3a28417ea399e3e26769fc"
CANDIDATE_TREE = "358fb1af6ca587b6c71ff2ef0fb87e335163eeaf"
CAMPAIGN_ID = "POPGP-VIABILITY-R1-2026-08"
PACKET_ID = "VIA-000"
RUNNER_IDENTITY = "codex-via000-runner"
RUNNER_SESSION = "popgp-viability-r1-2026-08-via000-runner-session"
BLOCKED_ENVIRONMENT = (
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "UV_PROJECT_ENVIRONMENT",
)

PREFLIGHT_CODE = (
    "import os,subprocess,sys; "
    "ps=[subprocess.run(['git','status','--porcelain=v1','--untracked-files=all'],"
    "capture_output=True,text=True),subprocess.run(['git','status','--porcelain=v1',"
    "'--untracked-files=normal','--ignored'],capture_output=True,text=True)]; "
    "[sys.stdout.write(p.stdout) for p in ps]; [sys.stderr.write(p.stderr) for p in ps]; "
    "blocked=sorted(k for k in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV',"
    "'UV_PROJECT_ENVIRONMENT') if os.environ.get(k)); "
    "sys.stderr.write(('blocked environment: '+','.join(blocked)+'\\n') if blocked else ''); "
    "nested=sys.prefix!=sys.base_prefix; sys.stderr.write('base interpreter required\\n' "
    "if nested else ''); raise SystemExit(any(p.returncode for p in ps) or "
    "any(p.stdout for p in ps) or bool(blocked) or nested)"
)

POSTFLIGHT_CODE = (
    "import os,pathlib,subprocess,sys; "
    "p=subprocess.run(['git','status','--porcelain=v1','--untracked-files=all'],"
    "capture_output=True,text=True); sys.stdout.write(p.stdout); sys.stderr.write(p.stderr); "
    "blocked=sorted(k for k in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV',"
    "'UV_PROJECT_ENVIRONMENT') if os.environ.get(k)); "
    "sys.stderr.write(('blocked environment: '+','.join(blocked)+'\\n') if blocked else ''); "
    "nested=sys.prefix!=sys.base_prefix; sys.stderr.write('base interpreter required\\n' "
    "if nested else ''); env_path=pathlib.Path('.venv'); "
    "invalid_env=not env_path.is_dir() or env_path.is_symlink(); "
    "sys.stderr.write('regular .venv directory required\\n' if invalid_env else ''); "
    "hooks=sorted(str(q) for name in ('sitecustomize.py','usercustomize.py') "
    "for q in env_path.rglob(name)) if not invalid_env else []; "
    "pths=sorted(env_path.rglob('*.pth')) if not invalid_env else []; "
    "expected={'_virtualenv.pth':'import _virtualenv'}; "
    "observed={q.name:q.read_text(encoding='utf-8').strip() for q in pths}; "
    "invalid_pth=len(pths)!=len(expected) or observed!=expected; "
    "sys.stdout.write(''.join(h+'\\n' for h in hooks)); "
    "sys.stdout.write(''.join(str(q)+'\\n' for q in pths) if invalid_pth else ''); "
    "raise SystemExit(p.returncode or bool(p.stdout) or bool(blocked) or nested or "
    "invalid_env or bool(hooks) or invalid_pth)"
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


class Harness:
    def __init__(self, candidate: Path, output: Path, platform_family: str, uv_cache: Path):
        self.candidate = candidate
        self.output = output
        self.platform_family = platform_family
        self.uv_cache = uv_cache
        self.commands_dir = output / "commands"
        self.commands_dir.mkdir(parents=True, exist_ok=False)
        self.sequence = 0
        self.commands: list[dict[str, Any]] = []
        self.mutations: list[dict[str, Any]] = []
        self.base_env = os.environ.copy()
        for name in BLOCKED_ENVIRONMENT:
            self.base_env.pop(name, None)
        self.base_env["UV_CACHE_DIR"] = str(uv_cache)

    def run(
        self,
        command_id: str,
        command: str,
        argv: list[str],
        *,
        environment: dict[str, str | None] | None = None,
        category: str = "protocol",
    ) -> dict[str, Any]:
        self.sequence += 1
        stem = f"{self.sequence:03d}-{command_id}"
        env = self.base_env.copy()
        for key, value in (environment or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        started = utc_now()
        start_clock = time.monotonic()
        process = subprocess.run(
            argv,
            cwd=self.candidate,
            env=env,
            capture_output=True,
            check=False,
        )
        duration = time.monotonic() - start_clock
        ended = utc_now()
        stdout_path = self.commands_dir / f"{stem}.stdout.txt"
        stderr_path = self.commands_dir / f"{stem}.stderr.txt"
        stdout_path.write_bytes(process.stdout)
        stderr_path.write_bytes(process.stderr)
        result = {
            "id": command_id,
            "category": category,
            "command": command,
            "argv": argv,
            "started_at": started,
            "ended_at": ended,
            "duration_seconds": round(duration, 6),
            "exit_code": process.returncode,
            "stdout_path": stdout_path.relative_to(self.output).as_posix(),
            "stdout_bytes": len(process.stdout),
            "stdout_sha256": sha256_bytes(process.stdout),
            "stderr_path": stderr_path.relative_to(self.output).as_posix(),
            "stderr_bytes": len(process.stderr),
            "stderr_sha256": sha256_bytes(process.stderr),
        }
        self.commands.append(result)
        return result

    def stdout(self, result: dict[str, Any]) -> str:
        return (self.output / result["stdout_path"]).read_text(encoding="utf-8", errors="replace")

    def stderr(self, result: dict[str, Any]) -> str:
        return (self.output / result["stderr_path"]).read_text(encoding="utf-8", errors="replace")

    def mutation(self, mutation_id: str, description: str, checks: dict[str, Any]) -> None:
        self.mutations.append({"id": mutation_id, "description": description, **checks})

    def git_restore(self, *paths: str) -> None:
        subprocess.run(
            ["git", "restore", "--staged", "--", *paths],
            cwd=self.candidate,
            env=self.base_env,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["git", "restore", "--worktree", "--", *paths],
            cwd=self.candidate,
            env=self.base_env,
            capture_output=True,
            check=True,
        )


def git_value(candidate: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=candidate,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def candidate_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_candidate_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def checker(harness: Harness, command_id: str) -> dict[str, Any]:
    command = (
        "uv run --isolated --frozen --no-editable python scripts/check_validation_artifacts.py"
    )
    return harness.run(
        command_id,
        command,
        [
            "uv",
            "run",
            "--isolated",
            "--frozen",
            "--no-editable",
            "python",
            "scripts/check_validation_artifacts.py",
        ],
        category="mutation",
    )


def postflight(
    harness: Harness,
    command_id: str,
    *,
    environment: dict[str, str | None] | None = None,
    category: str = "mutation",
) -> dict[str, Any]:
    command = f'python -I -c "{POSTFLIGHT_CODE}"'
    return harness.run(
        command_id,
        command,
        ["python", "-I", "-c", POSTFLIGHT_CODE],
        environment=environment,
        category=category,
    )


def preflight(harness: Harness, command_id: str, category: str = "mutation") -> dict[str, Any]:
    command = f'python -I -c "{PREFLIGHT_CODE}"'
    return harness.run(
        command_id,
        command,
        ["python", "-I", "-c", PREFLIGHT_CODE],
        category=category,
    )


def semantic_rejection(result: dict[str, Any], text: str, tokens: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return result["exit_code"] != 0 and all(token.lower() in lowered for token in tokens)


def run_main_protocol(harness: Harness) -> tuple[dict[str, Any], Path]:
    protocol: list[tuple[str, str, list[str]]] = [
        (
            "preflight",
            f'python -I -c "{PREFLIGHT_CODE}"',
            ["python", "-I", "-c", PREFLIGHT_CODE],
        ),
        ("sync", "uv sync --frozen --no-editable", ["uv", "sync", "--frozen", "--no-editable"]),
        (
            "ruff",
            "uv run --isolated --frozen --no-editable ruff check .",
            ["uv", "run", "--isolated", "--frozen", "--no-editable", "ruff", "check", "."],
        ),
        (
            "check-tex",
            "uv run --isolated --frozen --no-editable python scripts/check_tex.py",
            [
                "uv",
                "run",
                "--isolated",
                "--frozen",
                "--no-editable",
                "python",
                "scripts/check_tex.py",
            ],
        ),
        (
            "pytest",
            "uv run --isolated --frozen --no-editable python -m pytest -q",
            [
                "uv",
                "run",
                "--isolated",
                "--frozen",
                "--no-editable",
                "python",
                "-m",
                "pytest",
                "-q",
            ],
        ),
    ]
    for module in (
        "chain_1d",
        "grid_2d",
        "gravity_well",
        "source_law",
        "source_law_many_body",
        "ca_model",
    ):
        protocol.append(
            (
                f"example-{module}",
                f"uv run --isolated --frozen --no-editable python -m examples.physics_qg.{module}",
                [
                    "uv",
                    "run",
                    "--isolated",
                    "--frozen",
                    "--no-editable",
                    "python",
                    "-m",
                    f"examples.physics_qg.{module}",
                ],
            )
        )
    protocol.append(
        (
            "semantic-checker",
            "uv run --isolated --frozen --no-editable python scripts/check_validation_artifacts.py",
            [
                "uv",
                "run",
                "--isolated",
                "--frozen",
                "--no-editable",
                "python",
                "scripts/check_validation_artifacts.py",
            ],
        )
    )

    results: list[dict[str, Any]] = []
    for command_id, command, argv in protocol:
        result = harness.run(command_id, command, argv)
        results.append(result)
        if result["exit_code"] != 0:
            break

    external_pdf = harness.candidate.parent / f"{CAMPAIGN_ID}-{PACKET_ID}-PDF"
    if all(result["exit_code"] == 0 for result in results):
        create_code = (
            "import pathlib; p=(pathlib.Path('..')/"
            f"'{CAMPAIGN_ID}-{PACKET_ID}-PDF').resolve(); "
            "assert not p.exists(), p; p.mkdir()"
        )
        create_result = harness.run(
            "create-external-pdf-dir",
            f'python -I -c "{create_code}"',
            ["python", "-I", "-c", create_code],
        )
        results.append(create_result)
        pdf_command = (
            "pdflatex -interaction=nonstopmode -halt-on-error "
            f"'-output-directory=../{CAMPAIGN_ID}-{PACKET_ID}-PDF' docs/framework.tex"
        )
        pdf_argv = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory=../{CAMPAIGN_ID}-{PACKET_ID}-PDF",
            "docs/framework.tex",
        ]
        if create_result["exit_code"] == 0:
            results.append(harness.run("pdflatex-pass-1", pdf_command, pdf_argv))
            if results[-1]["exit_code"] == 0:
                results.append(harness.run("pdflatex-pass-2", pdf_command, pdf_argv))
        results.append(
            harness.run(
                "git-diff",
                "git diff --exit-code",
                ["git", "diff", "--exit-code"],
            )
        )
        results.append(postflight(harness, "postflight", category="protocol"))

    pytest_result = next((item for item in results if item["id"] == "pytest"), None)
    pytest_text = (
        ""
        if pytest_result is None
        else harness.stdout(pytest_result) + harness.stderr(pytest_result)
    )
    match = re.search(r"(\d+) passed", pytest_text)
    test_count = int(match.group(1)) if match else None
    return {
        "commands": results,
        "all_commands_succeeded": len(results) == 17
        and all(item["exit_code"] == 0 for item in results),
        "test_count": test_count,
        "required_test_count_met": test_count == 187,
        "required_example_count": 6,
        "required_example_count_met": sum(
            1 for item in results if item["id"].startswith("example-") and item["exit_code"] == 0
        )
        == 6,
    }, external_pdf


def run_mutations(harness: Harness) -> None:
    source_rel = "examples/physics_qg/source_law/results/validation.json"
    source_path = harness.candidate / source_rel
    chain_rel = "examples/physics_qg/chain_1d/results/validation.json"
    chain_path = harness.candidate / chain_rel
    many_rel = "examples/physics_qg/source_law_many_body/results/validation.json"
    many_path = harness.candidate / many_rel

    # G01: unregistered noninformational check identity.
    document = candidate_json(source_path)
    document["checks"].append(
        {
            "name": "unregistered_required_probe",
            "criterion": "must be explicitly registered",
            "value": True,
            "passed": True,
        }
    )
    write_candidate_json(source_path, document)
    result = checker(harness, "mutation-g01-checker")
    text = harness.stdout(result) + harness.stderr(result)
    harness.mutation(
        "VIA000-G01",
        "Add an unregistered noninformational check identity; require rejection.",
        {
            "command_ids": [result["id"]],
            "rejected": semantic_rejection(result, text, ("unregistered",)),
            "oracle": "nonzero checker exit and unregistered-identity diagnostic",
        },
    )
    harness.git_restore(source_rel)

    # G02: flip a required pass outcome.
    document = candidate_json(source_path)
    for check in document["checks"]:
        if check["name"] == "relative_entropy_is_quadratic":
            check["passed"] = False
            break
    write_candidate_json(source_path, document)
    result = checker(harness, "mutation-g02-checker")
    text = harness.stdout(result) + harness.stderr(result)
    harness.mutation(
        "VIA000-G02",
        "Flip a required check pass outcome; require rejection.",
        {
            "command_ids": [result["id"]],
            "rejected": semantic_rejection(result, text, ("passed",)),
            "oracle": "nonzero checker exit and pass-outcome diagnostic",
        },
    )
    harness.git_restore(source_rel)

    # G03: measurement-array shape change.
    document = candidate_json(source_path)
    document["measurements"]["relative_entropy"].pop()
    write_candidate_json(source_path, document)
    result = checker(harness, "mutation-g03-checker")
    text = harness.stdout(result) + harness.stderr(result)
    harness.mutation(
        "VIA000-G03",
        "Change a measurement array shape; require rejection.",
        {
            "command_ids": [result["id"]],
            "rejected": semantic_rejection(result, text, ("length",)),
            "oracle": "nonzero checker exit and exact length diagnostic",
        },
    )
    harness.git_restore(source_rel)

    # G04: stable configuration drift beyond tolerance.
    document = candidate_json(chain_path)
    document["config"]["beta"] = 1.01
    write_candidate_json(chain_path, document)
    result = checker(harness, "mutation-g04-checker")
    text = harness.stdout(result) + harness.stderr(result)
    harness.mutation(
        "VIA000-G04",
        "Change a stable configuration beyond tolerance; require rejection.",
        {
            "command_ids": [result["id"]],
            "rejected": semantic_rejection(result, text, ("beta", "rel_tol")),
            "oracle": "nonzero checker exit and stable-input tolerance diagnostic",
        },
    )
    harness.git_restore(chain_rel)

    # G05: empty visual and remove it from the Git index.
    visual_rel = "examples/physics_qg/source_law/results/source_scaling.png"
    visual_path = harness.candidate / visual_rel
    original_visual_sha = sha256_file(visual_path)
    rm_result = harness.run(
        "mutation-g05-index-removal",
        f"git rm --cached -- {visual_rel}",
        ["git", "rm", "--cached", "--", visual_rel],
        category="mutation",
    )
    visual_path.write_bytes(b"")
    result = checker(harness, "mutation-g05-checker")
    text = harness.stdout(result) + harness.stderr(result)
    rejected = (
        rm_result["exit_code"] == 0
        and result["exit_code"] != 0
        and "not tracked" in text.lower()
        and "empty" in text.lower()
    )
    harness.git_restore(visual_rel)
    restored_sha = sha256_file(visual_path)
    harness.mutation(
        "VIA000-G05",
        "Make a required visual empty and untracked; require both defects.",
        {
            "command_ids": [rm_result["id"], result["id"]],
            "rejected": rejected and restored_sha == original_visual_sha,
            "oracle": "checker reports both not-tracked and empty; exact bytes restored",
            "original_sha256": original_visual_sha,
            "restored_sha256": restored_sha,
        },
    )

    # G06: move the frozen sensitive diagnostic beyond its tolerance.
    document = candidate_json(many_path)
    replacements = 0

    def replace_sensitive(value: Any) -> Any:
        nonlocal replacements
        if type(value) is float and value == 3.812167720149195e-08:
            replacements += 1
            return 0.001
        if isinstance(value, list):
            return [replace_sensitive(item) for item in value]
        if isinstance(value, dict):
            return {key: replace_sensitive(item) for key, item in value.items()}
        return value

    document = replace_sensitive(document)
    write_candidate_json(many_path, document)
    result = checker(harness, "mutation-g06-checker")
    text = harness.stdout(result) + harness.stderr(result)
    harness.mutation(
        "VIA000-G06",
        "Move a sensitive scientific diagnostic beyond its frozen threshold; require rejection.",
        {
            "command_ids": [result["id"]],
            "rejected": replacements == 2
            and semantic_rejection(result, text, ("quadratic_coefficient_relative_error",)),
            "oracle": "both canonical copies changed and checker rejects frozen tolerance",
            "replacement_count": replacements,
        },
    )
    harness.git_restore(many_rel)

    # G07: untracked root startup hook plus PYTHONPATH.
    root_hook = harness.candidate / "sitecustomize.py"
    root_hook.write_text(
        "import os\nos.environ['POPGP_G07_ROOT'] = 'active'\n",
        encoding="utf-8",
        newline="\n",
    )
    probe = harness.run(
        "mutation-g07-isolated-probe",
        "uv run --isolated --frozen --no-editable python -c "
        "\"import os; print(os.environ.get('POPGP_G07_ROOT'))\"",
        [
            "uv",
            "run",
            "--isolated",
            "--frozen",
            "--no-editable",
            "python",
            "-c",
            "import os; print(os.environ.get('POPGP_G07_ROOT'))",
        ],
        environment={"PYTHONPATH": "."},
        category="mutation",
    )
    semantic = checker_with_environment(harness, "mutation-g07-checker", {"PYTHONPATH": "."})
    diff = harness.run(
        "mutation-g07-git-diff",
        "git diff --exit-code",
        ["git", "diff", "--exit-code"],
        environment={"PYTHONPATH": "."},
        category="mutation",
    )
    boundary = postflight(
        harness,
        "mutation-g07-postflight",
        environment={"PYTHONPATH": "."},
    )
    boundary_text = harness.stdout(boundary) + harness.stderr(boundary)
    root_hook.unlink()
    harness.mutation(
        "VIA000-G07",
        "Activate an untracked sitecustomize.py through PYTHONPATH; require boundary rejection.",
        {
            "command_ids": [probe["id"], semantic["id"], diff["id"], boundary["id"]],
            "rejected": probe["exit_code"] == 0
            and "active" in harness.stdout(probe)
            and semantic["exit_code"] == 0
            and diff["exit_code"] == 0
            and boundary["exit_code"] != 0
            and "sitecustomize.py" in boundary_text
            and "blocked environment: PYTHONPATH" in boundary_text,
            "oracle": (
                "complete frozen boundary rejects both untracked hook and blocked environment"
            ),
        },
    )

    # G08: ignored customize hooks preplanted before locked sync.
    env_path = harness.candidate / ".venv"
    if env_path.exists():
        shutil.rmtree(env_path)
    site_packages = (
        env_path / "Lib" / "site-packages"
        if os.name == "nt"
        else env_path / "lib" / "python3.11" / "site-packages"
    )
    site_packages.mkdir(parents=True)
    site_hook = site_packages / "sitecustomize.py"
    user_hook = site_packages / "usercustomize.py"
    site_hook.write_text(
        "import os\nos.environ['POPGP_G08_SITE'] = 'active'\n",
        encoding="utf-8",
        newline="\n",
    )
    user_hook.write_text(
        "import os\nos.environ['POPGP_G08_USER'] = 'active'\n",
        encoding="utf-8",
        newline="\n",
    )
    before = preflight(harness, "mutation-g08-preflight")
    sync = harness.run(
        "mutation-g08-sync",
        "uv sync --frozen --no-editable",
        ["uv", "sync", "--frozen", "--no-editable"],
        category="mutation",
    )
    project_python = (
        env_path / "Scripts" / "python.exe" if os.name == "nt" else env_path / "bin" / "python"
    )
    project_probe = harness.run(
        "mutation-g08-project-probe",
        f"{project_python} -c <marker-probe>",
        [
            str(project_python),
            "-c",
            "import os; print(os.environ.get('POPGP_G08_SITE'), os.environ.get('POPGP_G08_USER'))",
        ],
        category="mutation",
    )
    isolated_probe = harness.run(
        "mutation-g08-isolated-probe",
        "uv run --isolated --frozen --no-editable python -c <marker-probe>",
        [
            "uv",
            "run",
            "--isolated",
            "--frozen",
            "--no-editable",
            "python",
            "-c",
            "import os; print(os.environ.get('POPGP_G08_SITE'), os.environ.get('POPGP_G08_USER'))",
        ],
        category="mutation",
    )
    after = postflight(harness, "mutation-g08-postflight")
    before_text = harness.stdout(before) + harness.stderr(before)
    after_text = harness.stdout(after) + harness.stderr(after)
    rejected = (
        before["exit_code"] != 0
        and ".venv" in before_text
        and sync["exit_code"] == 0
        and project_probe["exit_code"] == 0
        and "active" in harness.stdout(project_probe)
        and isolated_probe["exit_code"] == 0
        and "None None" in harness.stdout(isolated_probe)
        and after["exit_code"] != 0
        and "sitecustomize.py" in after_text
        and "usercustomize.py" in after_text
    )
    site_hook.unlink(missing_ok=True)
    user_hook.unlink(missing_ok=True)
    harness.mutation(
        "VIA000-G08",
        "Preplant ignored .venv startup hooks before locked sync; require "
        "preflight and postflight rejection.",
        {
            "command_ids": [
                before["id"],
                sync["id"],
                project_probe["id"],
                isolated_probe["id"],
                after["id"],
            ],
            "rejected": rejected,
            "oracle": "ignored-state preflight and startup-hook postflight both reject",
        },
    )

    # G09: extra executable .pth and modified allowed _virtualenv.pth.
    pth_files = sorted(env_path.rglob("_virtualenv.pth"))
    allowed_pth = pth_files[0] if len(pth_files) == 1 else None
    extra_pth = site_packages / "via000_g09_extra.pth"
    extra_pth.write_text(
        "import os; os.environ['POPGP_G09_EXTRA']='active'\n",
        encoding="utf-8",
        newline="\n",
    )
    project_probe = harness.run(
        "mutation-g09-extra-project-probe",
        f"{project_python} -c <marker-probe>",
        [str(project_python), "-c", "import os; print(os.environ.get('POPGP_G09_EXTRA'))"],
        category="mutation",
    )
    isolated_probe = harness.run(
        "mutation-g09-extra-isolated-probe",
        "uv run --isolated --frozen --no-editable python -c <marker-probe>",
        [
            "uv",
            "run",
            "--isolated",
            "--frozen",
            "--no-editable",
            "python",
            "-c",
            "import os; print(os.environ.get('POPGP_G09_EXTRA'))",
        ],
        category="mutation",
    )
    extra_boundary = postflight(harness, "mutation-g09-extra-postflight")
    extra_text = harness.stdout(extra_boundary) + harness.stderr(extra_boundary)
    extra_rejected = (
        project_probe["exit_code"] == 0
        and "active" in harness.stdout(project_probe)
        and isolated_probe["exit_code"] == 0
        and "None" in harness.stdout(isolated_probe)
        and extra_boundary["exit_code"] != 0
        and "via000_g09_extra.pth" in extra_text
    )
    extra_pth.unlink(missing_ok=True)

    modified_rejected = False
    modified_ids: list[str] = []
    if allowed_pth is not None:
        original_pth = allowed_pth.read_bytes()
        allowed_pth.write_bytes(
            original_pth + b"\nimport os; os.environ['POPGP_G09_ALLOWED']='active'\n"
        )
        project_probe_2 = harness.run(
            "mutation-g09-allowed-project-probe",
            f"{project_python} -c <marker-probe>",
            [str(project_python), "-c", "import os; print(os.environ.get('POPGP_G09_ALLOWED'))"],
            category="mutation",
        )
        isolated_probe_2 = harness.run(
            "mutation-g09-allowed-isolated-probe",
            "uv run --isolated --frozen --no-editable python -c <marker-probe>",
            [
                "uv",
                "run",
                "--isolated",
                "--frozen",
                "--no-editable",
                "python",
                "-c",
                "import os; print(os.environ.get('POPGP_G09_ALLOWED'))",
            ],
            category="mutation",
        )
        modified_boundary = postflight(harness, "mutation-g09-allowed-postflight")
        modified_text = harness.stdout(modified_boundary) + harness.stderr(modified_boundary)
        modified_rejected = (
            project_probe_2["exit_code"] == 0
            and "active" in harness.stdout(project_probe_2)
            and isolated_probe_2["exit_code"] == 0
            and "None" in harness.stdout(isolated_probe_2)
            and modified_boundary["exit_code"] != 0
            and "_virtualenv.pth" in modified_text
        )
        modified_ids = [
            project_probe_2["id"],
            isolated_probe_2["id"],
            modified_boundary["id"],
        ]
        allowed_pth.write_bytes(original_pth)
    harness.mutation(
        "VIA000-G09",
        "Inject executable .pth hooks; isolated commands must ignore them and "
        "postflight reject them.",
        {
            "command_ids": [project_probe["id"], isolated_probe["id"], extra_boundary["id"]]
            + modified_ids,
            "rejected": extra_rejected and modified_rejected,
            "oracle": (
                "extra and modified-allowed .pth variants execute only in project "
                "Python and fail postflight"
            ),
        },
    )

    # G10: historical editable self-cleaning carrier, then amended elimination.
    editable_sync = harness.run(
        "mutation-g10-editable-sync",
        "uv sync --frozen",
        ["uv", "sync", "--frozen"],
        category="mutation",
    )
    editable_pths = [path for path in env_path.rglob("*.pth") if path.name != "_virtualenv.pth"]
    old_exploit = False
    sentinel_exploit = False
    amended_elimination = False
    g10_ids = [editable_sync["id"]]
    carrier_name = None
    if editable_sync["exit_code"] == 0 and len(editable_pths) == 1:
        carrier = editable_pths[0]
        carrier_name = carrier.name
        benign = carrier.read_bytes()
        carrier_literal = repr(str(carrier))
        benign_literal = repr(benign)
        carrier.write_text(
            "import os,pathlib; os.environ['POPGP_G10']='active'; "
            f"pathlib.Path({carrier_literal}).write_bytes({benign_literal})\n",
            encoding="utf-8",
            newline="\n",
        )
        vulnerable_probe = harness.run(
            "mutation-g10-vulnerable-probe",
            "uv run --isolated --frozen python -c <marker-probe>",
            [
                "uv",
                "run",
                "--isolated",
                "--frozen",
                "python",
                "-c",
                "import os; print(os.environ.get('POPGP_G10'))",
            ],
            category="mutation",
        )
        g10_ids.append(vulnerable_probe["id"])
        old_exploit = (
            vulnerable_probe["exit_code"] == 0
            and "active" in harness.stdout(vulnerable_probe)
            and carrier.read_bytes() == benign
        )

        sentinel = harness.output / "g10-sentinel.txt"
        sentinel_literal = repr(str(sentinel))
        carrier.write_text(
            "import pathlib; "
            f"pathlib.Path({sentinel_literal}).write_text('executed',encoding='utf-8'); "
            f"pathlib.Path({carrier_literal}).write_bytes({benign_literal})\n",
            encoding="utf-8",
            newline="\n",
        )
        vulnerable_checker = harness.run(
            "mutation-g10-vulnerable-checker",
            "uv run --isolated --frozen python scripts/check_validation_artifacts.py",
            [
                "uv",
                "run",
                "--isolated",
                "--frozen",
                "python",
                "scripts/check_validation_artifacts.py",
            ],
            category="mutation",
        )
        g10_ids.append(vulnerable_checker["id"])
        sentinel_exploit = (
            vulnerable_checker["exit_code"] == 0
            and sentinel.read_text(encoding="utf-8") == "executed"
            and carrier.read_bytes() == benign
        )
        sentinel.unlink(missing_ok=True)

        carrier.write_text(
            "import os,pathlib; os.environ['POPGP_G10']='active'; "
            f"pathlib.Path({carrier_literal}).write_bytes({benign_literal})\n",
            encoding="utf-8",
            newline="\n",
        )
        noneditable_sync = harness.run(
            "mutation-g10-noneditable-sync",
            "uv sync --frozen --no-editable",
            ["uv", "sync", "--frozen", "--no-editable"],
            category="mutation",
        )
        amended_probe = harness.run(
            "mutation-g10-amended-probe",
            "uv run --isolated --frozen --no-editable python -c <marker-probe>",
            [
                "uv",
                "run",
                "--isolated",
                "--frozen",
                "--no-editable",
                "python",
                "-c",
                "import os; print(os.environ.get('POPGP_G10'))",
            ],
            category="mutation",
        )
        final_boundary = postflight(harness, "mutation-g10-postflight")
        g10_ids.extend([noneditable_sync["id"], amended_probe["id"], final_boundary["id"]])
        amended_elimination = (
            noneditable_sync["exit_code"] == 0
            and not carrier.exists()
            and amended_probe["exit_code"] == 0
            and "None" in harness.stdout(amended_probe)
            and final_boundary["exit_code"] == 0
        )
    harness.mutation(
        "VIA000-G10",
        "Make the editable project .pth self-clean; require non-editable sync and "
        "execution to eliminate it.",
        {
            "command_ids": g10_ids,
            "rejected": old_exploit and sentinel_exploit and amended_elimination,
            "oracle": (
                "historical carrier controls interpreter/checker, self-cleans, then "
                "amended boundary eliminates it"
            ),
            "editable_carrier": carrier_name,
            "historical_exploit_reproduced": old_exploit,
            "semantic_checker_startup_control_reproduced": sentinel_exploit,
            "amended_boundary_eliminated_carrier": amended_elimination,
        },
    )


def checker_with_environment(
    harness: Harness, command_id: str, environment: dict[str, str | None]
) -> dict[str, Any]:
    command = (
        "uv run --isolated --frozen --no-editable python scripts/check_validation_artifacts.py"
    )
    return harness.run(
        command_id,
        command,
        [
            "uv",
            "run",
            "--isolated",
            "--frozen",
            "--no-editable",
            "python",
            "scripts/check_validation_artifacts.py",
        ],
        environment=environment,
        category="mutation",
    )


def tool_output(argv: list[str], cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    process = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, check=False)
    return {
        "argv": argv,
        "exit_code": process.returncode,
        "stdout": process.stdout.decode("utf-8", errors="replace"),
        "stderr": process.stderr.decode("utf-8", errors="replace"),
    }


def copy_pdf_evidence(external_pdf: Path, output: Path) -> dict[str, Any]:
    destination = output / "pdf"
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = []
    if external_pdf.is_dir():
        for source in sorted(external_pdf.iterdir()):
            if not source.is_file():
                continue
            target = destination / source.name
            shutil.copy2(source, target)
            artifacts.append(
                {
                    "name": source.name,
                    "bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                    "retained_path": target.relative_to(output).as_posix(),
                }
            )
    pdf = next((item for item in artifacts if item["name"] == "framework.pdf"), None)
    return {
        "external_directory": str(external_pdf),
        "artifacts": artifacts,
        "framework_pdf_nonempty": pdf is not None and pdf["bytes"] > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-family", required=True)
    parser.add_argument("--uv-cache", type=Path, required=True)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    output = args.output.resolve()
    uv_cache = args.uv_cache.resolve()
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    if uv_cache.exists():
        raise SystemExit(f"uv cache already exists: {uv_cache}")
    output.mkdir(parents=True)
    uv_cache.mkdir(parents=True)

    harness = Harness(candidate, output, args.platform_family, uv_cache)
    started_at = utc_now()
    harness_error: str | None = None
    external_pdf = candidate.parent / f"{CAMPAIGN_ID}-{PACKET_ID}-PDF"
    if external_pdf.exists():
        raise SystemExit(f"external PDF directory already exists: {external_pdf}")

    head = git_value(candidate, "rev-parse", "HEAD")
    tree = git_value(candidate, "rev-parse", "HEAD^{tree}")
    if head != CANDIDATE_COMMIT or tree != CANDIDATE_TREE:
        raise SystemExit(f"candidate identity mismatch: {head} / {tree}")

    environment = {
        "campaign_id": CAMPAIGN_ID,
        "packet_id": PACKET_ID,
        "platform_family": args.platform_family,
        "runner_identity": RUNNER_IDENTITY,
        "runner_session_id": RUNNER_SESSION,
        "model_identity": "unknown",
        "model_version": "unknown",
        "operator": "fuocor",
        "orchestrator_id": "codex-desktop",
        "candidate_commit": head,
        "candidate_tree": tree,
        "candidate_path": str(candidate),
        "uv_cache_path": str(uv_cache),
        "blocked_environment_values": {name: os.environ.get(name) for name in BLOCKED_ENVIRONMENT},
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_runtime": sys.version,
        "selected_process_environment": {
            name: os.environ.get(name)
            for name in (
                "CI",
                "GITHUB_ACTIONS",
                "RUNNER_OS",
                "RUNNER_ARCH",
                "ImageOS",
                "ImageVersion",
            )
        },
        "tool_probes": {
            "git": tool_output(["git", "--version"], candidate, harness.base_env),
            "python": tool_output(["python", "--version"], candidate, harness.base_env),
            "uv": tool_output(["uv", "--version"], candidate, harness.base_env),
            "pdflatex": tool_output(["pdflatex", "--version"], candidate, harness.base_env),
        },
        "hidden_access_declaration": {
            "final_labels_seen": False,
            "secret_seed_seen": False,
            "private_evaluator_seen": False,
            "custody_path_accessed": False,
        },
    }
    strict_write_json(output / "environment.json", environment)

    pdflatex_version = environment["tool_probes"]["pdflatex"]["stdout"]
    engine_exact = (
        environment["tool_probes"]["pdflatex"]["exit_code"] == 0
        and "1.40.29" in pdflatex_version
        and "TeX Live 2026" in pdflatex_version
    )
    protocol_result: dict[str, Any] = {
        "commands": [],
        "all_commands_succeeded": False,
        "test_count": None,
        "required_test_count_met": False,
        "required_example_count": 6,
        "required_example_count_met": False,
    }
    pdf_result: dict[str, Any] = {
        "external_directory": str(external_pdf),
        "artifacts": [],
        "framework_pdf_nonempty": False,
    }
    try:
        if not engine_exact:
            raise RuntimeError("pinned pdfTeX 1.40.29 / TeX Live 2026 is unavailable")
        protocol_result, external_pdf = run_main_protocol(harness)
        pdf_result = copy_pdf_evidence(external_pdf, output)
        if protocol_result["all_commands_succeeded"]:
            run_mutations(harness)
    except Exception as exc:  # evidence must survive unexpected runner failures
        harness_error = f"{type(exc).__name__}: {exc}"
    finally:
        if external_pdf.is_dir():
            shutil.rmtree(external_pdf)

    final_normal = tool_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        candidate,
        harness.base_env,
    )
    final_diff = tool_output(["git", "diff", "--exit-code"], candidate, harness.base_env)
    mutation_count_met = len(harness.mutations) == 10
    mutations_rejected = mutation_count_met and all(item["rejected"] for item in harness.mutations)
    evidence_contract = (
        engine_exact
        and protocol_result["all_commands_succeeded"]
        and protocol_result["required_test_count_met"]
        and protocol_result["required_example_count_met"]
        and pdf_result["framework_pdf_nonempty"]
        and harness_error is None
        and final_normal["exit_code"] == 0
        and final_normal["stdout"] == ""
        and final_diff["exit_code"] == 0
    )
    platform_result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "packet_id": PACKET_ID,
        "runner_identity": RUNNER_IDENTITY,
        "runner_session_id": RUNNER_SESSION,
        "platform_family": args.platform_family,
        "candidate_commit": head,
        "candidate_tree": tree,
        "started_at": started_at,
        "ended_at": utc_now(),
        "engine_exact": engine_exact,
        "protocol": protocol_result,
        "pdf": pdf_result,
        "mutations": harness.mutations,
        "required_mutation_count": 10,
        "required_mutation_count_met": mutation_count_met,
        "mutations_rejected": mutations_rejected,
        "evidence_contract": evidence_contract,
        "harness_error": harness_error,
        "final_candidate_normal_status": final_normal,
        "final_candidate_diff": final_diff,
        "external_pdf_directory_removed": not external_pdf.exists(),
    }
    strict_write_json(output / "platform-results.json", platform_result)
    strict_write_json(output / "command-index.json", harness.commands)
    strict_write_json(output / "mutation-results.json", harness.mutations)
    return 0 if evidence_contract and mutations_rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
