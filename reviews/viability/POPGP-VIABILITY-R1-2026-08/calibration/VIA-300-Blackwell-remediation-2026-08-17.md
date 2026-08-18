# VIA-300 Blackwell remediation calibration — 2026-08-17

Status: **builder calibration only; not decisive E3 evidence**.

This record supersedes the 2026-08-13 native build and benchmark as verification
evidence. It addresses the executable defects identified by independent review
`POPGP-REVIEW-BLACKWELL-VIABILITY-1`. It was produced before holdout start by the
builder session on the same operator-controlled machine. It is not independent
reproduction, does not satisfy VIA-300, and does not establish Tier R.

## Candidate, hardware, and toolchain

- Scientific candidate: `9a29e05f803666bf0e3a28417ea399e3e26769fc`.
- Candidate tree: `358fb1af6ca587b6c71ff2ef0fb87e335163eeaf`.
- GPU: NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU, compute capability
  `12.0`, driver `595.79`, 12,227 MiB.
- CUDA toolkit: `13.3.73`, isolated at
  `C:\src\POPGP-cuda-toolkit-13.3\local`.
- Official CUDA 13.3.1 installer:
  `https://developer.download.nvidia.com/compute/cuda/13.3.1/local_installers/cuda_13.3.1_windows.exe`.
- Installer SHA-256:
  `d68839fcce644576f0a1c6b066e0c5bc146a62db2fcac9fc6b4e6418e7ec533f`.
- Compiler/build tools: MSVC `19.50.35730`, CMake `4.3.2`, Ninja, and vcpkg
  commit `e5a1490e1409d175932ef6014519e9ae149ddb7c`.

The independent review reported that the CUDA toolkit was absent. That observation
did not hold for the builder environment: `nvcc.exe` and `cuobjdump.exe` both existed
and executed from the isolated path above. The underlying review concern was still
accepted: repository documentation and configure-time checks did not make the
toolkit version or provenance reproducible. The remediation now requires CUDA 12.8
or newer, documents the exact installer URL/hash, and checks the requested binary
architecture after linking.

## Native build and non-vacuous test gate

The public entry point was executed with:

```text
build.bat --clean --test --cuda-arch 120
```

The build gate used `ctest --no-tests=error` and independently compared discovered
tests to a configure-time expected count. It enumerated exactly seven cases and all
seven passed:

1. two-qubit phase evolution;
2. Heisenberg amplitude exchange and norm preservation;
3. aligned-state no-flip-flop control;
4. area-law boundary-cut calculation;
5. pruning transitions;
6. explicit clock-solver not-implemented status;
7. self-validating native benchmark.

`cuobjdump --list-elf` reported three `sm_120` cubins in `phase_flow.dll`. The
post-link verifier rejected a deliberately mismatched `--cuda-arch 86` build on the
visible `sm_120` device. The argument parser also rejected a missing architecture
value and `not-an-arch` before configuration. The final tree was rebuilt for `120`
after those negative controls.

## Hardened benchmark

The benchmark performs a warmup outside the measured interval, checks every launch,
synchronizes before accepting timing, reads the output back, and exits nonzero for
nonfinite values or excessive per-cell norm error. A repeated one-million-cell run
completed 100 red/black steps with:

- elapsed time: `448.57 ms`;
- throughput: `2.23e8` edge updates/s;
- deterministic FNV-1a-64 state checksum: `48e6ef8f40cb137c`;
- maximum per-cell norm error: `1.1435297153639112e-14`.

The benchmark's regression is registered as the seventh native CTest case, so its
validation is part of the non-vacuous build gate rather than an unchecked display
path.

## Python/native integration

A disposable CUDA-enabled PyTorch 2.10 environment exercised `GPUBackend` through
the rebuilt DLL on a 64-cell open Heisenberg chain. It reported 63 edges, finite
output, maximum alpha change `0.01999042898064934`, maximum beta change
`0.010195137584425498`, and maximum per-cell norm error
`5.551115123125783e-16`.

The locked project environment remains CPU-only. The disposable CUDA environment
and generated native build products were not committed.

## Publication and CI

The exact scientific candidate is published at
`https://github.com/whact2025/POPGP/commit/9a29e05f803666bf0e3a28417ea399e3e26769fc`.
GitHub Actions run `32088634733` completed successfully at that exact SHA, including
locked sync, Ruff, TeX source validation, all 187 Python tests, all six documented
examples, and committed validation-contract checks.

## Retained limitations

- This is builder-owned single-machine calibration under a shared operator, not an
  independent campaign receipt.
- The native backend is still a mean-field product-state path. It cannot produce the
  MI/QCMI and reduced-state evidence required by VIA-300.
- The clock solver now fails explicitly as not implemented; it is not a working
  clock reconstruction.
- Native compilation, validated execution, throughput, and norm preservation do not
  establish exact-observable agreement, four-level convergence, independent
  implementation, or mechanism viability.
- `VIA-300` remains `drafted`, the campaign decision remains `pending`, and
  `holdout_started` remains false.
