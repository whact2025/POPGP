# Phase-Ordered Pre-Geometric Projection (POPGP) Framework

**A Constructive Approach to Emergent Quantum Gravity**

This repository contains the conceptual framework and computational simulator for the POPGP theory. The framework proposes that spacetime, time, and matter emerge from a more fundamental, atemporal algebraic substrate via a stability-selection principle.

## Repository Structure

*   `docs/`: Core theoretical documents.
    *   [`framework.md`](docs/framework.md): The principal definition of the framework (v0.10).
    *   [`simulator_analysis_and_plan.md`](docs/simulator_analysis_and_plan.md): Codebase analysis and implementation roadmap.
*   `popgp/`: Python package — the unified simulator.
    *   `config.py`: Canonical parameter configuration (every parameter labeled).
    *   `backend.py`: Backend abstraction (`ExactBackend` / `GPUBackend`).
    *   `simulator.py`: Unified projection pipeline `Π = Π_time ∘ Π_geom ∘ Π_loc ∘ Π_res`.
    *   `engine.py`: Lazy-loaded Python bindings for the C++/CUDA kernel.
*   `popgp_engine/`: High-performance C++/CUDA engine.
    *   `kernel/`: Phase-flow kernel, area-law cut, clock solver (stub).
    *   `renderer/`: 3D visualization pipeline (planned).
*   `examples/`: Runnable demonstrations (each is a Python package).
    *   `chain_1d/`: 1D stability selection and geometry recovery.
    *   `grid_2d/`: 2D emergent geometry from scrambled algebra.
    *   `ca_model/`: Cellular automata stability selection and cooling.

## Key Concepts

1.  **Substrate:** An algebraic object (Operator Algebra) with no intrinsic spacetime.
2.  **Phase Flow:** A fundamental ordering generator that drives evolution.
3.  **Projection:** A map that extracts stable subsystems ("Cells") that minimize information leakage.
4.  **Emergence:**
    *   **Space:** Arises from mutual information correlations between stable cells.
    *   **Time:** Arises from the phase-order flow, scaled by local information density.
    *   **Matter:** Arises as the persistent, low-entropy structures selected by the projection.

## Quick Start

This project uses `uv` for dependency management.

Prerequisites: [Install uv](https://github.com/astral-sh/uv).

```bash
# Install dependencies
uv sync

# Run any example as a module
uv run python -m examples.grid_2d
uv run python -m examples.chain_1d
uv run python -m examples.ca_model
```

### Programmatic Usage

```python
from popgp import Simulator, SimulatorConfig

cfg = SimulatorConfig.for_grid(width=3, height=3, beta=2.0)
sim = Simulator(cfg)
result = sim.run()

print(f"Emergent dimension: D* = {result.pi_geom.D_star}")
print(f"Clock potential range: [{result.pi_time.phi.min():.3f}, {result.pi_time.phi.max():.3f}]")
```

## Status

*   **Conceptual Framework:** v0.10 (Complete Design Spec)
*   **Simulator Architecture:** Implemented (config, backends, unified pipeline)
*   **Validation:**
    *   [x] 1D Geometry Emergence
    *   [x] 2D Grid Emergence from Scrambled Algebra
    *   [x] Stability Selection via Thermodynamics (CA Model)
    *   [x] Spectral dimension and complexity-stress dimension selection
    *   [x] Clock potential via graph Laplacian
*   **Next Steps:**
    *   [ ] Strict Π_res: variational cell selection with SU(2) equivariance
    *   [ ] Local metric `h_ab` and Delaunay / Regge curvature
    *   [ ] Araki source term with KMS vacuum baseline
    *   [ ] Observables: lensing, redshift, Shapiro delay
    *   [ ] 3D Gravity Simulation
