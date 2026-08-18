# POPGP Engine: C++/CUDA Kernel

This directory contains the high-performance implementation of the **Phase-Ordered Flow** ($\sigma_s$).

## Core Components
1.  **Product-state array:** Stores one two-component spinor for each active cell.
2.  **Adjacency List (Sparse Graph):** Stores the connectivity ($I_{ij}$) and interaction operators ($H_{ij}$).
3.  **Phase Flow Kernel (`phase_flow.cu`):** Parallel CUDA kernel that updates cell states based on local Hamiltonian terms.
4.  **Cut/pruning kernel (`area_law.cu`):** Calculates boundary cut weights and applies the declared entropy/cut threshold rule.

## Implementation Steps (Phase 2)
- [x] Define `Cell` and `Edge` structs for CUDA.
- [x] Implement `apply_gate` device function for unitary evolution.
- [x] Implement the mean-field `phase_flow_kernel`.
- [x] Implement caller-side node-disjoint batching in `GPUBackend` and the benchmark.
- [ ] Implement an entangling state representation capable of MI/QCMI.
- [ ] Implement the native graph-Laplacian clock solver. The exported placeholder
      currently returns `POPGP_STATUS_NOT_IMPLEMENTED` without writing output.

The phase-flow kernel does **not** color an arbitrary edge list. Every call must contain
a node-disjoint batch. `GPUBackend._edge_color_batches` supplies this contract for the
Python path; `popgp_sim` uses a red/black partition for its one-dimensional chain.

## Build and test

The top-level engine build defaults to the GPU detected at configure time. For a
frozen build, pass the numeric CUDA architecture explicitly:

```text
cd popgp_engine
build.bat --clean --test --cuda-arch 120
```

The Windows script discovers the installed Visual Studio C++ environment, uses
Ninja, bootstraps the pinned vcpkg commit, and fails if configure, compile, or any
native test fails. It also rejects malformed architecture values, requires every
expected native test, and uses `cuobjdump` to verify that the produced library contains
device code for the requested and locally visible architecture.

CUDA Toolkit 12.8 is the minimum supported version because it is the first toolkit
that can emit Blackwell `sm_120` code. The independently exercised Windows toolchain is
CUDA 13.3.1 (`nvcc 13.3.73`), available from:

```text
https://developer.download.nvidia.com/compute/cuda/13.3.1/local_installers/cuda_13.3.1_windows.exe
SHA-256: d68839fcce644576f0a1c6b066e0c5bc146a62db2fcac9fc6b4e6418e7ec533f
```

Verify the installer before running it:

```powershell
$installer = "cuda_13.3.1_windows.exe"
$expected = "d68839fcce644576f0a1c6b066e0c5bc146a62db2fcac9fc6b4e6418e7ec533f"
if ((Get-FileHash -Algorithm SHA256 $installer).Hash.ToLowerInvariant() -ne $expected) {
    throw "CUDA installer hash mismatch"
}
```

Install the official toolkit normally, or set `CUDA_PATH` to an isolated extraction
that contains `bin/nvcc`, `bin/cuobjdump`, headers, import libraries, and the required
runtime DLLs. For the recorded local extraction this is:

```powershell
$env:CUDA_PATH = "C:\src\POPGP-cuda-toolkit-13.3\local"
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\bin\x64;$env:PATH"
```

The path is an example, not a hidden requirement: configuration discovers the toolkit
through `CUDA_PATH`, enforces version 12.8 or newer, and fails if `cuobjdump` or a
matching visible GPU is absent. On Linux, use
`build.sh --clean --test --cuda-arch <value>` with `VCPKG_ROOT` set.
