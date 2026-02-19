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
