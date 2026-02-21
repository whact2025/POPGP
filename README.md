# Phase-Ordered Pre-Geometric Projection (POPGP) Framework

**A Constructive Approach to Emergent Quantum Gravity**

## The Problem

Physics has two spectacularly successful theories that refuse to work together. **General Relativity** describes gravity as the curvature of spacetime -- planets orbit the Sun because the Sun warps the fabric of space and time around it. **Quantum Mechanics** describes the subatomic world as a probabilistic dance of particles and fields. Both theories have been tested to extraordinary precision, yet every attempt to combine them into a single "quantum gravity" breaks down: the math produces infinities, paradoxes, or both.

Both theories share a hidden assumption: **spacetime already exists**. Relativity curves it; quantum mechanics puts particles on it. But what if spacetime isn't fundamental at all? What if space, time, and matter all *emerge* from something deeper -- something that has no geometry, no clocks, and no distances built in?

## The POPGP Idea

POPGP starts from an unusual premise: **there is no space. There is no time. There is only a mathematical structure -- an algebra of relationships -- and a single ordering parameter (called "phase order") that provides a notion of sequence without being a clock.**

From this bare starting point, the framework constructs everything we observe through a four-stage **projection**, which you can think of as a lens that brings familiar physics into focus:

### Stage 1 -- Stability Selection ("What survives?")

The substrate is vast and mostly chaotic. The projection's first job is to identify **cells** -- small subsystems that hold onto their information instead of leaking it into the surroundings. Think of it as natural selection for quantum states: only the stable patterns persist. These cells become the "atoms of space."

*What the simulator shows:* Given an 8-qubit spin chain, the optimizer exhaustively tests all 105 possible ways to group the qubits into pairs. It finds that **contiguous pairs** (neighbors on the chain) leak the least information -- locality is not assumed, it **emerges** from the stability principle.

### Stage 2 -- Locality from Correlations ("Who is near whom?")

With the stable cells identified, the framework asks: how much does each cell "know" about every other cell? This is measured by **mutual information** -- a precise quantum quantity borrowed from information theory. Cells that share a lot of mutual information are defined as "close"; cells that share little are "far apart."

No ruler is needed. Distance is defined purely by how much two cells are correlated.

*What the simulator shows:* On a 3x3 grid of qubits, the mutual information between nearest neighbors is roughly 10x stronger than between diagonal qubits -- the correlation pattern faithfully encodes the grid topology, even though the simulator never tells the framework what the grid looks like.

### Stage 3 -- Emergent Geometry ("What shape is the world?")

The correlation-derived distances are fed into a standard technique from data science (multidimensional scaling) to find the best-fit geometry. The framework doesn't assume three dimensions -- it tests all possibilities and selects the dimension that best fits the correlation data with the least complexity.

*What the simulator shows:* A 1D chain of qubits recovers a line. A 2D grid of qubits recovers a plane. The correct number of dimensions is **selected**, not imposed.

### Stage 4 -- Emergent Time ("How fast does each clock tick?")

The final stage builds a "clock potential" from the entropy landscape of the cells. The framework solves a Poisson-like equation on the correlation graph -- structurally identical to the equation that governs the Newtonian gravitational potential. Higher entropy contrast at a cell means its local clock ticks at a different rate.

This is where gravity enters: **the clock potential IS the gravitational potential**, encoded on the emergent geometry. Time dilation near a mass is not added by hand; it falls out of the math.

*What the simulator shows:* When a point source of entropy is placed at the center of a 3x3 grid, the clock potential peaks at the source and decays monotonically outward -- exactly as a gravitational field should. The framework even predicts a measurable "gravitational redshift" between cells at different potentials.

## What Makes This Different

| Feature | POPGP | Other Approaches |
|---------|-------|-----------------|
| **Starting point** | Abstract algebra with no geometry | Strings on a background, spin networks, etc. |
| **Space** | Emerges from correlations | Often assumed or discretized |
| **Time** | Emerges from phase ordering | Often assumed or quantized |
| **Gravity** | Falls out of the clock equation | Typically added via action principle |
| **Predictions** | Constructive -- each stage is computable | Often perturbative or non-constructive |
| **Falsifiable** | Yes -- area-law violations, Lorentz violation bounds, singularity saturation | Varies |

## Why a Simulator?

Unlike many approaches to quantum gravity that live entirely on paper, POPGP is designed to be **computed**. Every stage of the projection is an algorithm that takes quantum states as input and produces numbers as output. The simulator in this repository implements the full pipeline and lets you:

- Watch locality emerge from an abstract spin system.
- See the correct number of spatial dimensions get selected automatically.
- Observe a gravitational potential form from entropy contrast.
- Measure redshift between cells at different clock rates.

Each example produces a `results/validation.json` file with structured, machine-readable output of every metric and PASS/FAIL check -- suitable for automated verification by AI or CI systems.

## Repository Structure

*   `docs/`: Core theoretical documents.
    *   [`framework.md`](docs/framework.md): The principal definition of the framework (v0.10).
    *   [`simulator_analysis_and_plan.md`](docs/simulator_analysis_and_plan.md): Codebase analysis and implementation roadmap.
*   `popgp/`: Python package -- the unified simulator.
    *   `config.py`: Canonical parameter configuration (every parameter labeled).
    *   `backend.py`: Backend abstraction (`ExactBackend` / `GPUBackend`).
    *   `simulator.py`: Unified projection pipeline.
    *   `coarse_grain.py`: Variational cell selection (Phase 1).
    *   `capacity.py`: Cut-capacity functional and Araki bounds.
    *   `engine.py`: Lazy-loaded Python bindings for the C++/CUDA kernel.
*   `popgp_engine/`: High-performance C++/CUDA engine.
    *   `kernel/`: Phase-flow kernel, area-law cut, clock solver (stub).
    *   `renderer/`: 3D visualization pipeline (planned).
*   `examples/`: Runnable demonstrations (each is a Python package with its own README).
    *   `chain_1d/`: 1D stability selection and geometry recovery.
    *   `grid_2d/`: 2D emergent geometry from scrambled algebra.
    *   `ca_model/`: Cellular automata stability selection and cooling.
    *   `gravity_well/`: Localized source test -- first gravitational observable.

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
uv run python -m examples.gravity_well
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
    *   [x] Strict cell selection with SU(2) equivariance (Phase 1)
    *   [x] Gravity well: localized source with monotonic potential falloff + redshift
*   **Next Steps:**
    *   [ ] Local metric reconstruction and Delaunay / Regge curvature
    *   [ ] Araki source term with KMS vacuum baseline
    *   [ ] Observables: gravitational lensing, Shapiro delay
    *   [ ] 3D gravity simulation and GR matching
