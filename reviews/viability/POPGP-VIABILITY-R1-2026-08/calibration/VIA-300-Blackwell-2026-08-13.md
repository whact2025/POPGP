# VIA-300 Blackwell pre-holdout calibration — 2026-08-13

Status: **calibration only; not decisive E3 evidence**.

This calibration established that the available Blackwell hardware can execute both
the supported PyTorch CUDA runtime and POPGP's experimental native mean-field engine.
It was performed by the builder/protocol session before holdout start. It therefore
does not replace an independent reproduction runner, falsifier, statistical auditor,
claim auditor, or adjudicator, and it does not satisfy VIA-300's exact-overlap,
refinement, or independent-implementation gates.

## Candidate and hardware

- Source candidate: `3428a24bdd05b5742888bd204482d50a2f613cab`.
- Candidate tree: `42fcba754ed78145054f572bd0beedf8a157562a`.
- GPU: NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU.
- Compute capability: `12.0`.
- Driver: `595.79`; `nvidia-smi` reported CUDA compatibility through `13.2`.
- Memory: 12,227 MiB reported by `nvidia-smi` (12,820,480,000 bytes reported by
  PyTorch).

## CUDA runtime probe

A disposable Python 3.11 environment installed the official PyTorch
`2.10.0+cu130` wheel without changing `uv.lock` or the locked project environment.
The runtime reported CUDA `13.0`, `torch.cuda.is_available() == True`, device
capability `12.0`, and an architecture list containing `sm_120`.

A seeded `4096 × 4096` float32 matrix multiplication completed on the GPU in
approximately `0.0991 s`; every result was finite. Repeating the probe reproduced
the recorded checksum `320.41931545734406` over the selected `8 × 8` output block.

## Native toolchain and build

The official CUDA 13.3.1 Windows installer was downloaded by WinGet and matched the
published SHA-256
`d68839fcce644576f0a1c6b066e0c5bc146a62db2fcac9fc6b4e6418e7ec533f`.
The silent machine-wide installer could not cross the elevation boundary, so its
documented archive payload was extracted into an isolated local toolchain. This
provided `nvcc 13.3.73` without changing the display driver or system PATH.

The corrected public build entry point was then exercised with:

```text
build.bat --clean --test --cuda-arch 120
```

The build used MSVC `19.50.35730`, Ninja, vcpkg commit
`e5a1490e1409d175932ef6014519e9ae149ddb7c`, and the matching manifest baseline.
Configuration, compilation, linkage, and all four CUDA tests passed. `cuobjdump`
reported three native `sm_120` cubins in `phase_flow.dll`.

Native test outcomes:

1. two-qubit phase evolution: passed;
2. Heisenberg amplitude exchange and per-cell norm preservation: passed;
3. aligned-state no-flip-flop control: passed;
4. area-law boundary-cut calculation: passed.

The one-million-cell native benchmark completed 100 red/black phase-flow steps in
`166.45 ms`, approximately `6.01e8` cell-updates/s.

## Python/native integration probe

A 64-cell open Heisenberg chain was prepared by `GPUBackend` on `cuda:0` and evolved
through the compiled `phase_flow.dll`. The output was finite and nontrivial:

- edges: `63`;
- maximum alpha change: `0.019934498535924983`;
- maximum beta change: `0.00972260983729168`;
- maximum per-cell norm error: `5.551115123125783e-16`.

This also reproduced and motivated a Windows loader fix: handles returned by
`os.add_dll_directory()` must remain alive while dependent CUDA DLLs are resolved.

## Retained limitations

- The locked project environment intentionally remains CPU-only; the CUDA PyTorch
  wheel was used only in a disposable calibration environment.
- The GPU backend is a mean-field product-state path. It cannot compute MI/QCMI and
  cannot run the full locality/geometry/clock projection pipeline above the exact
  threshold.
- The native clock solver remains an identity stub and native renderer outputs are
  not integrated into the supported Python pipeline.
- Throughput, finite evolution, and unit tests do not establish VIA-300's required
  four-level convergence, exact-observable agreement, independent implementation,
  mutation resistance, or scientific scalability.
- Formal VIA-300 execution must use the frozen protocol, retain raw outputs and
  environment bytes, commit output before reveal, and pass independent adversarial
  review.
