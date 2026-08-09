# POPGP Engine Development Questions

This document tracks open questions regarding the implementation of the `popgp_engine` to ensure alignment with the framework.

## 1. Scale & Performance
*   **Target Node Count ($N$)**: What is the target simulation scale?
    *   *Implication*: $N < 5000$ allows dense linear algebra ($O(N^3)$). $N > 10^5$ requires sparse/iterative methods and spatial hashing.
    *   *Current Assumption*: Start with $N \approx 1024$ (Dense/Direct methods) for rigorous validation, then scale up.
*   **Precision Requirements**: The framework has not derived a numerical Lorentz-violation prediction or a universal tolerance.
    *   *Question*: Which observable-specific error budget is required before comparing a derived coefficient with current experimental bounds?
    *   *Recommendation*: Keep scientific geometry, stress, and Regge diagnostics in FP64; rendering may use lower precision after the measured quantities are finalized.

## 2. Dependencies & Reuse
*   **Host-Side Geometry**: Is `CGAL` (via vcpkg) acceptable for Delaunay Triangulation?
    *   *Context*: "Rigorous reuse" suggests using a robust library rather than writing a fragile GPU Delaunay kernel.
    *   *Constraint*: Data transfer latency (Device $\to$ Host $\to$ Device) per phase step.
*   **Linear Algebra**: Should we use `AmgX` (NVIDIA Algebraic Multigrid) for the Laplacian solver, or standard `cuSOLVER`?
    *   *Context*: `AmgX` is better for large sparse systems but adds dependency complexity.

## 3. Theoretical Implementation Details
*   **Vacuum Reference**: Section 4.4.5 defines the vacuum as the "KMS state of the modular flow".
    *   *Question*: How do we construct this algebraically in the engine?
    *   *Proposal*: For finite toy models, compute the thermal state $\rho_{kms} \propto e^{-\beta H_{mod}}$ explicitly.
*   **Spectral Dimension**: Calculating $D_S$ requires the trace of the heat kernel over continuous time $t$.
    *   *Question*: What range of $t$ should be integrated/fit to determine $D_S$?

## 4. Visualization
*   **Headless vs. Interactive**: Should the engine support headless mode (rendering to video file) for server-side simulation?
    *   *Current Build*: Seems set up for local execution.
