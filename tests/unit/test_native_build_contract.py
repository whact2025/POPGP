from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "popgp_engine"


def _cmake_script(script: Path, *definitions: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cmake", *[f"-D{item}" for item in definitions], "-P", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cuda_architecture_argument_contract() -> None:
    script = ENGINE / "cmake" / "ValidateCudaArchitecture.cmake"
    for value in ("native", "all", "all-major", "120", "120-real", "120-virtual"):
        result = _cmake_script(script, f"POPGP_CUDA_ARCHITECTURE={value}")
        assert result.returncode == 0, result.stdout + result.stderr

    for value in ("", "--clean", "not-an-arch", "120;86", "sm_120", "12.0"):
        result = _cmake_script(script, f"POPGP_CUDA_ARCHITECTURE={value}")
        assert result.returncode != 0, value

    for build_script in (ENGINE / "build.bat", ENGINE / "build.sh"):
        text = build_script.read_text(encoding="utf-8")
        assert "ValidateCudaArchitecture.cmake" in text
        assert "--no-tests=error" in text
        assert "VerifyCTestCount.cmake" in text


def test_native_test_gate_rejects_zero_and_partial_discovery(tmp_path: Path) -> None:
    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        """
cmake_minimum_required(VERSION 3.25)
project(native_test_gate NONE)
enable_testing()
add_test(NAME only_test COMMAND ${CMAKE_COMMAND} -E true)
file(WRITE "${CMAKE_BINARY_DIR}/popgp_expected_native_tests.txt" "2\\n")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja"],
        check=True,
        capture_output=True,
        text=True,
    )

    verifier = ENGINE / "cmake" / "VerifyCTestCount.cmake"
    partial = _cmake_script(
        verifier,
        f"POPGP_BUILD_DIR={build}",
        "POPGP_CONFIG=Release",
    )
    assert partial.returncode != 0
    assert "Expected 2 native tests" in partial.stdout + partial.stderr

    (build / "popgp_expected_native_tests.txt").write_text("1\n", encoding="utf-8")
    complete = _cmake_script(
        verifier,
        f"POPGP_BUILD_DIR={build}",
        "POPGP_CONFIG=Release",
    )
    assert complete.returncode == 0, complete.stdout + complete.stderr

    empty_source = tmp_path / "empty-source"
    empty_build = tmp_path / "empty-build"
    empty_source.mkdir()
    (empty_source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(empty_native_tests NONE)\n"
        "enable_testing()\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["cmake", "-S", str(empty_source), "-B", str(empty_build), "-G", "Ninja"],
        check=True,
        capture_output=True,
        text=True,
    )
    empty = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(empty_build),
            "--output-on-failure",
            "--no-tests=error",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert empty.returncode != 0
    assert "No tests were found" in empty.stdout + empty.stderr


def test_cuda_binary_architecture_verifier_rejects_mismatch(tmp_path: Path) -> None:
    binary = tmp_path / "phase_flow.dll"
    binary.write_bytes(b"test fixture; cuobjdump output is injected below")
    script = ENGINE / "cmake" / "VerifyCudaBinary.cmake"

    common = (
        f"POPGP_CUDA_BINARY={binary}",
        "POPGP_CUOBJDUMP_ELF_OUTPUT=ELF file 1: phase_flow.1.sm_120.cubin",
        "POPGP_CUOBJDUMP_PTX_OUTPUT=",
        "POPGP_NATIVE_ARCHITECTURES=120",
        "POPGP_REQUIRE_VISIBLE_CUDA_ARCH=ON",
    )
    matched = _cmake_script(script, "POPGP_CUDA_ARCHITECTURE=120", *common)
    assert matched.returncode == 0, matched.stdout + matched.stderr

    native = _cmake_script(script, "POPGP_CUDA_ARCHITECTURE=native", *common)
    assert native.returncode == 0, native.stdout + native.stderr

    mismatch = _cmake_script(script, "POPGP_CUDA_ARCHITECTURE=86", *common)
    assert mismatch.returncode != 0
    assert "not requested sm_86" in mismatch.stdout + mismatch.stderr
