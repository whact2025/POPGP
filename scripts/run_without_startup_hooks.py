"""Run a Python module from a locked environment without evaluating startup hooks.

The caller must invoke this file with Python's ``-I -S`` flags.  ``-S`` prevents
``site`` from evaluating ``.pth``, ``sitecustomize.py``, or ``usercustomize.py``.
This bootstrap then adds only the environment's site-packages directories and the
frozen repository root before executing the requested module.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _environment_root() -> Path:
    executable = Path(sys.executable).absolute()
    if executable.parent.name.lower() not in {"bin", "scripts"}:
        raise RuntimeError(f"cannot locate isolated environment from {executable}")
    return executable.parent.parent


def _site_package_directories(environment_root: Path) -> list[Path]:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = (
        environment_root / "Lib" / "site-packages",
        environment_root / "lib" / version / "site-packages",
        environment_root / "lib64" / version / "site-packages",
        environment_root / "lib" / version / "dist-packages",
    )
    return [path for path in candidates if path.is_dir() and not path.is_symlink()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not sys.flags.isolated or not sys.flags.no_site:
        parser.error("this bootstrap requires Python -I -S")

    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir() or repo_root.is_symlink():
        parser.error(f"regular repository directory required: {repo_root}")

    environment_root = _environment_root()
    site_packages = _site_package_directories(environment_root)
    if not site_packages:
        parser.error(f"no site-packages directory found under {environment_root}")

    if not sys.pycache_prefix:
        parser.error("an external Python bytecode cache is required via -X pycache_prefix")
    pycache_prefix = Path(sys.pycache_prefix)
    if (
        not pycache_prefix.is_absolute()
        or not pycache_prefix.is_dir()
        or pycache_prefix.is_symlink()
    ):
        parser.error(f"regular external bytecode-cache directory required: {pycache_prefix}")
    pycache_prefix = pycache_prefix.resolve()
    for protected_root, label in (
        (repo_root, "repository"),
        (environment_root, "locked environment"),
    ):
        try:
            pycache_prefix.relative_to(protected_root)
        except ValueError:
            continue
        parser.error(f"bytecode cache must be outside the {label}: {pycache_prefix}")

    # No call to site.addsitedir is permitted here: it evaluates executable .pth
    # lines.  Direct path insertion makes the dependency packages importable while
    # leaving every startup hook inert.
    sys.path[:0] = [str(repo_root), *(str(path) for path in site_packages)]
    module_arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [args.module, *module_arguments]
    runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
