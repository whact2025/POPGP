# Native Model: 2D Heisenberg Grid (CUDA)

GPU-accelerated mean-field simulation of a 30×30 = 900-cell grid using the
compiled CUDA phase-flow kernel. Validates the same geometry recovery as the
toy model (`src/toy/grid_2d/`) but at 100× scale.

Requires a CUDA GPU and the compiled `phase_flow` kernel (`popgp_engine/build.bat`).

## Framework Sections Validated

| Principle | Framework Reference | What this script tests |
|---|---|---|
| Correlation-Based Locality | Section 4.4.3 | Time-averaged Sz Pearson correlations encode 2D grid distance. |
| Geometry Recovery (MDS) | Section 4.4.4 | MDS embedding into 2D recovers grid topology: neighbours are closer than non-neighbours. |
| Emergent 3D Space | Postulate P2 | Spatial dimensionality emerges from dynamics on an abstract graph. |

## Key Difference: Toy vs Native

| Aspect | Toy (full quantum) | Native (mean-field) |
|---|---|---|
| Physics | Exact Heisenberg, full 2^9 Hilbert space | Mean-field Heisenberg (each cell = independent qubit) |
| Scale | 3×3 = 9 qubits | 30×30 = 900 cells |
| Correlation metric | Mutual information I(i,j) from thermal state | Time-averaged Sz Pearson correlation from dynamics |
| Engine | PyTorch CPU matrix exponentiation | CUDA kernel (GPU) |

## Algorithm

1. **Initialise** 900 cells near |0⟩ with small random perturbations.
2. **Build graph**: 2D grid with checkerboard (two-colour) edge colouring
   for conflict-free parallel GPU execution.
3. **Evolve** for `WARMUP_STEPS` (500) steps to let dynamics settle.
4. **Measure** for `MEASURE_STEPS` (2000) steps, accumulating time-averaged
   Sz values and Sz⊗Sz products for 150 sampled cells.
5. **Build correlation matrix**: compute Pearson correlation from time-averaged
   moments: Cov(i,j) = ⟨Sz_i·Sz_j⟩_t − ⟨Sz_i⟩_t·⟨Sz_j⟩_t.
6. **Build distance matrix**: d(i,j) = -log(|Corr(i,j)|).
7. **MDS embedding**: extract the top-2 eigenvectors to embed into 2D.
8. **Validate**: compare average embedded distance for grid-neighbours vs
   non-neighbours. Neighbours should be closer (typically ~15% gap).

## Parameters

| Parameter | Value | Role |
|---|---|---|
| WIDTH × HEIGHT | 30 × 30 | Grid dimensions (900 cells) |
| dt | 0.2 | Phase-order step size (tunable hyperparameter) |
| PERTURBATION | 0.15 | Initial state perturbation amplitude (tunable hyperparameter) |
| WARMUP_STEPS | 500 | Steps before measurement begins |
| MEASURE_STEPS | 2000 | Steps for time-averaged correlation accumulation |
| SAMPLE_CELLS | 150 | Cells sampled for correlation matrix and MDS |

## How to Run

```bash
uv run src/native/grid_2d/grid_2d_native.py
```

## Results and How to Interpret

### 2D Embedding

![2D Embedding](results/embedding.png)

**What you see**: a scatter plot of 150 sampled cells in a two-dimensional
coordinate space recovered by MDS. Thin black lines connect points that are
nearest neighbours on the original 30×30 grid.

**How to read it**:
- The x and y axes are the two leading MDS dimensions, derived entirely from
  time-averaged Sz correlations -- no spatial information was provided to the
  algorithm.
- **Success criterion**: the points should show spatial clustering consistent
  with a 2D lattice. Grid-neighbours (connected by black lines) should be
  visibly closer together than non-neighbours. The script prints the average
  embedded distance for neighbours vs non-neighbours; a gap of ~15% or more
  indicates success.
- The embedding may appear rotated, reflected, or non-uniformly scaled relative
  to the physical grid -- these are all symmetries of MDS and do not indicate
  failure.
- Because this uses mean-field dynamics rather than exact quantum mechanics,
  the embedding is noisier than the toy model's. The correlation signal is
  weaker per pair, but the larger system provides more data points.
