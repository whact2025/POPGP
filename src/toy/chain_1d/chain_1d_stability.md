# Toy Model: 1D Heisenberg Chain

Pure-Python, full-quantum simulation of an 8-qubit Heisenberg spin chain.
This is the foundational validation of the POPGP framework -- it proves that
stability selection, correlation-based locality, and geometry recovery all
work when the underlying quantum mechanics is exact.

## Framework Sections Validated

| Principle | Framework Reference | What this script tests |
|---|---|---|
| Stability Selection | Section 4.4.2a | "Valid" (contiguous) 2-qubit cells accumulate less entropy than "invalid" (scattered) cells under time evolution. Locality **emerges** from stability. |
| Correlation-Based Locality | Section 4.4.3 | Mutual information between cells in a thermal state defines a distance metric d(i,j) = -log(I_ij). Neighbours have small d; distant cells have large d. |
| Geometry Recovery (MDS) | Section 4.4.4 | Classical MDS applied to the distance matrix recovers the correct 1D ordering of the 4 cells. |

## Algorithm

### Phase 1 -- Stability Selection (dynamic)

1. Construct the 8-qubit Heisenberg Hamiltonian:
   H = Σ_i (Sx_i·Sx_{i+1} + Sy_i·Sy_{i+1} + Sz_i·Sz_{i+1})
2. Initialise the system in the Neel state |01010101⟩.
3. Evolve the full 256×256 density matrix under U(dt) = exp(-iHdt) for 20 steps.
4. At each step, compute the Von Neumann entropy S = -Tr(ρ log ρ) of:
   - **Valid cells**: contiguous 2-qubit blocks [0,1], [2,3], [4,5], [6,7].
   - **Invalid cells**: scattered pairs [0,4], [1,5], [2,6], [3,7].
5. **Result**: invalid cells reach higher entropy faster -- they are less stable.

### Phase 2 -- Locality & Geometry (static)

1. Prepare a thermal state ρ = exp(-βH)/Z at inverse temperature β = 1.0.
2. Compute mutual information I(i,j) = S_i + S_j - S_{ij} for all pairs of valid cells.
3. Convert to a distance metric: d(i,j) = -log(I_ij).
4. Apply classical MDS to the 4×4 distance matrix.
5. **Result**: MDS perfectly recovers the 1D chain order [0, 1, 2, 3].

## Parameters

| Parameter | Value | Role |
|---|---|---|
| N (qubits) | 8 | System size (full Hilbert space 2^8 = 256) |
| k (block size) | 2 | Qubits per cell → 4 cells |
| β (temperature) | 1.0 | Inverse temperature for thermal state (tunable hyperparameter) |
| dt | 0.1 | Phase-order step size (tunable hyperparameter) |
| steps | 20 | Evolution steps for stability measurement |

## How to Run

```bash
uv run src/toy/chain_1d/chain_1d_stability.py
```

## Results and How to Interpret

### Entropy Growth

![Entropy Growth](results/entropy_growth.png)

**What you see**: two curves over phase-order time (x-axis). The y-axis is the
average Von Neumann entropy of the reduced density matrices.

**How to read it**:
- **Blue solid line** = "Valid" cells (contiguous 2-qubit blocks). These are the
  **local** subsystems that the framework predicts should be stable.
- **Red dashed line** = "Invalid" cells (scattered pairs like qubits [0,4]).
  These are **non-local** subsystems that should be unstable.
- **Success criterion**: the red curve should rise faster and plateau higher than
  the blue curve. This confirms that contiguous (local) subsystems leak less
  information to their environment than scattered (non-local) ones. Locality is
  not assumed -- it **emerges** from the stability principle.

### 1D Embedding

![1D Embedding](results/embedding.png)

**What you see**: points labelled 0–3 (the four valid cells) plotted along one
emergent dimension.

**How to read it**:
- Each point is a cell's MDS coordinate derived purely from mutual-information
  distances -- no knowledge of the physical chain was used.
- **Success criterion**: the points should appear in the correct sequential order
  (0, 1, 2, 3), confirming that MDS recovered the 1D topology from correlations
  alone. Any monotonic ordering (including reversed) counts as success, since
  reflection is a symmetry of MDS.

## Limitations

- Full Hilbert space scales as 2^N, limiting this approach to ~12 qubits.
- This is the theoretical reference implementation; the native CUDA version
  (`src/native/chain_1d/`) scales to hundreds of cells via mean-field.
