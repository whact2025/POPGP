# Native Model: 1D Heisenberg Chain (CUDA)

GPU-accelerated mean-field simulation of a 100-cell 1D chain using the compiled
CUDA phase-flow kernel. Validates the same framework principles as the toy model
(`src/toy/chain_1d/`) but at a scale impossible with full quantum mechanics.

Requires a CUDA GPU and the compiled `phase_flow` kernel (`popgp_engine/build.bat`).

## Framework Sections Validated

| Principle | Framework Reference | What this script tests |
|---|---|---|
| Stability Selection | Section 4.4.2a | Time-averaged ⟨Sz_i·Sz_j⟩ is higher for contiguous (local) pairs than for scattered (non-local) pairs. |
| Correlation-Based Locality | Section 4.4.3 | Pearson correlation of Sz time series encodes graph distance. |
| Geometry Recovery (MDS) | Section 4.4.4 | Classical MDS recovers 1D chain ordering with rank correlation ~0.97. |

## Key Difference: Toy vs Native

| Aspect | Toy (full quantum) | Native (mean-field) |
|---|---|---|
| Physics | Exact Heisenberg, full 2^N Hilbert space | Mean-field Heisenberg (each cell = independent qubit) |
| Scale | 8 qubits (4 cells) | 100 cells |
| Entanglement | Yes (partial-trace entropy) | No (product states always) |
| Correlation metric | Mutual information I(i,j) | Time-averaged Sz Pearson correlation |
| Engine | PyTorch CPU matrix exponentiation | CUDA kernel (GPU) |

## CUDA Kernel Physics

The kernel implements the **full Heisenberg mean-field interaction**:

```
U = exp(-i · J · dt · h⃗ · σ⃗)
```

where h⃗ = (2Re(α*β), 2Im(α*β), |α|²-|β|²) is the Bloch vector of the
neighbour. This is a full SU(2) rotation that:
- **Exchanges amplitude** between |0⟩ and |1⟩ (the "flip-flop" Sx+Sy terms).
- **Rotates phase** (the Sz term, which was the only term in the old Ising kernel).

The amplitude exchange drives dynamic Sz evolution, creating distance-dependent
correlations that MDS can exploit for geometry recovery.

## Algorithm

1. **Initialise** 100 cells near |0⟩ with small random perturbations (amplitude
   `PERTURBATION = 0.15`). The near-uniform starting state allows spin-wave
   correlations to develop coherently.
2. **Build graph**: 1D chain with red/black edge colouring for conflict-free
   parallel GPU execution.
3. **Evolve** for `WARMUP_STEPS` (1000) steps to let dynamics settle.
4. **Measure** for `MEASURE_STEPS` (5000) steps, accumulating:
   - Per-step ⟨Sz_i·Sz_j⟩ for local vs non-local pairs (stability metric).
   - Running sums for time-averaged Pearson correlation of Sz over all sampled cells.
5. **Build distance matrix**: d(i,j) = -log(|Corr(Sz_i, Sz_j)|).
6. **MDS embedding**: extract the leading eigenvector of the double-centred
   distance matrix to recover a 1D coordinate for each cell.
7. **Validate**: compare MDS ordering to true chain position via rank correlation.

## Parameters

| Parameter | Value | Role |
|---|---|---|
| N | 100 | Number of cells |
| dt | 0.3 | Phase-order step size (tunable hyperparameter) |
| PERTURBATION | 0.15 | Initial state perturbation amplitude (tunable hyperparameter) |
| WARMUP_STEPS | 1000 | Steps before measurement begins |
| MEASURE_STEPS | 5000 | Steps over which time-averaged correlations accumulate |
| SAMPLE_CELLS | 100 | Cells included in correlation matrix (all cells at this scale) |

## How to Run

```bash
uv run src/native/chain_1d/chain_1d_native.py
```

## Results and How to Interpret

### Stability Selection

![Stability](results/stability.png)

**What you see**: two time-series curves plotted over phase order (x-axis).
The y-axis is the instantaneous ⟨Sz_i · Sz_j⟩ averaged over each pair set.

**How to read it**:
- **Blue solid line** = local (contiguous) pairs -- cells that are direct
  neighbours on the chain.
- **Red dashed line** = non-local (scattered) pairs -- cells separated by
  half the chain length.
- **Success criterion**: the blue curve should be consistently higher than (or
  above) the red curve. This means neighbouring cells have more correlated
  magnetisation than distant cells -- the local neighbourhood is dynamically
  preferred, confirming the stability selection principle.
- The curves may oscillate because Sz values evolve dynamically under the
  Heisenberg interaction. What matters is the **average gap**, not individual
  time points.

### Geometry Recovery (MDS Embedding)

![1D Embedding](results/embedding.png)

**What you see**: a scatter plot where each dot is one cell. The x-axis is the
MDS coordinate (recovered from correlations alone) and the y-axis is the true
chain position (ground truth).

**How to read it**:
- **Success criterion**: the points should lie along a straight line (positive
  or negative slope). A tight linear relationship means MDS successfully
  recovered the 1D ordering from time-averaged Sz correlations.
- The title displays the Pearson rank correlation `r`. Values of |r| > 0.9
  indicate excellent geometry recovery. A negative `r` simply means the MDS
  axis is flipped (a trivial symmetry).
- Scatter or curvature in the plot indicates regions where the correlation
  signal is weaker (e.g. chain endpoints, which have only one neighbour).
