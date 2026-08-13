# POPGP Engine: C++/CUDA Kernel

This directory contains the high-performance implementation of the **Phase-Ordered Flow** ($\sigma_s$).

## Core Components
1.  **State Vector Array:** Stores the quantum amplitudes/density matrices for all active Cells.
2.  **Adjacency List (Sparse Graph):** Stores the connectivity ($I_{ij}$) and interaction operators ($H_{ij}$).
3.  **Phase Flow Kernel (`phase_flow.cu`):** Parallel CUDA kernel that updates cell states based on local Hamiltonian terms.
4.  **Area Law Pruner (`area_law.cu`):** Kernel that calculates the boundary cut size and freezes "bulk" nodes to optimize computation.

## Implementation Steps (Phase 2)
- [x] Define `Cell` and `Edge` structs for CUDA.
- [x] Implement `apply_gate` device function for unitary evolution.
- [x] Implement `phase_flow_kernel` with graph coloring for parallel safety.
- [x] Implement `host_step` function to launch the kernel from Python.

## Build and test

The top-level engine build defaults to the GPU detected at configure time. For a
frozen build, pass the numeric CUDA architecture explicitly:

```text
cd popgp_engine
build.bat --clean --test --cuda-arch 120
```

The Windows script discovers the installed Visual Studio C++ environment, uses
Ninja, bootstraps the pinned vcpkg commit, and fails if configure, compile, or any
native test fails. Set `CUDA_PATH` when the toolkit is not installed in its default
location. On Linux or macOS, use `build.sh --clean --test --cuda-arch <value>` with
`VCPKG_ROOT` set.
