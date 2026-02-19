# Phase-Ordered Pre-Geometric Projection (POPGP) Framework

**A Constructive Approach to Emergent Quantum Gravity**

This repository contains the conceptual framework and computational toy models for the POPGP theory. The framework proposes that spacetime, time, and matter emerge from a more fundamental, atemporal algebraic substrate via a stability-selection principle.

## Repository Structure

*   `docs/`: Contains the core theoretical documents.
    *   `framework.md`: The principal definition of the framework (v0.8).
*   `src/`: Computational toy models validating the postulates.
    *   `main.py`: 1D geometry recovery.
    *   `grid_2d.py`: 2D geometry recovery.
    *   `ca_model.py`: Cellular Automata demonstrating stability selection and "cooling".
*   `results/`: (To be created) Simulation outputs.

## Key Concepts

1.  **Substrate:** An algebraic object (Operator Algebra) with no intrinsic spacetime.
2.  **Phase Flow:** A fundamental ordering generator that drives evolution.
3.  **Projection:** A map that extracts stable subsystems ("Cells") that minimize information leakage.
4.  **Emergence:**
    *   **Space:** Arises from mutual information correlations between stable cells.
    *   **Time:** Arises from the phase-order flow, scaled by local information density.
    *   **Matter:** Arises as the persistent, low-entropy structures selected by the projection.

## Running the Models

Prerequisites: Python 3.8+, NumPy, PyTorch, Matplotlib.

```bash
cd src
python grid_2d.py  # Run the 2D emergent geometry simulation
python ca_model.py # Run the stability/cooling simulation
```

## Status

*   **Conceptual Framework:** v0.8 (Complete Design Spec)
*   **Validation:**
    *   [x] 1D Geometry Emergence
    *   [x] 2D Grid Emergence from Scrambled Algebra
    *   [x] Stability Selection via Thermodynamics (CA Model)
*   **Next Steps:**
    *   [ ] 3D Gravity Simulation (Lensing tests)
    *   [ ] Standard Model Embedding
