# Toy Model: 2D Heisenberg Grid

Pure-Python, full-quantum simulation of a 3×3 = 9-qubit Heisenberg lattice.
Demonstrates that the framework's geometry recovery pipeline (correlation →
distance → MDS) reconstructs **two-dimensional** structure from a scrambled
algebraic substrate, extending the 1D chain result to higher dimensions.

## Framework Sections Validated

| Principle | Framework Reference | What this script tests |
|---|---|---|
| Correlation-Based Locality | Section 4.4.3 | Mutual information between single-qubit cells in a thermal state encodes the 2D grid topology. |
| Geometry Recovery (MDS) | Section 4.4.4 | Classical MDS on the MI-derived distance matrix recovers a faithful 2D embedding. |
| Emergent 3D Space | Postulate P2 | Proof-of-concept that spatial dimensionality emerges from correlations alone. |

## Algorithm

1. Construct the 9-qubit Heisenberg Hamiltonian on a 3×3 grid with nearest-neighbour
   couplings (12 edges: 6 horizontal + 6 vertical).
2. Diagonalise the 512×512 Hamiltonian and prepare the thermal state ρ = exp(-βH)/Z
   with β = 2.0.
3. Compute single-site Von Neumann entropies S_i and pair entropies S_{ij} via
   partial traces for all 36 pairs.
4. Build the mutual information matrix: I(i,j) = S_i + S_j - S_{ij}.
5. Convert to distance: d(i,j) = -log(I_ij / I_max).
6. Apply classical MDS, keeping the top-2 eigenvalues to embed into 2D.
7. **Validate**: grid-neighbours should be closer than non-neighbours in the embedding.

## Parameters

| Parameter | Value | Role |
|---|---|---|
| WIDTH × HEIGHT | 3 × 3 | Grid dimensions (9 qubits, 512-dim Hilbert space) |
| β | 2.0 | Inverse temperature (tunable hyperparameter; lower β enhances correlations) |

## How to Run

```bash
uv run src/toy/grid_2d/grid_2d.py
```

## Results and How to Interpret

### 2D Embedding

![2D Embedding](results/embedding.png)

**What you see**: a scatter plot of 9 numbered points (one per qubit) in a
two-dimensional coordinate space recovered by MDS. Thin black lines connect
points that are nearest neighbours on the original 3×3 grid.

**How to read it**:
- The x and y axes are the two leading MDS dimensions, derived entirely from
  mutual information -- no spatial information was provided to the algorithm.
- **Success criterion**: the 9 points should form a recognisable 3×3 grid
  pattern. Nearest neighbours on the Hamiltonian graph (connected by black
  lines) should be close together; diagonal or distant pairs should be farther
  apart. Rotation, reflection, or uniform scaling of the grid are all valid
  (they are symmetries of MDS).
- The script also prints a quantitative check: the average embedded distance
  between the centre cell (qubit 4) and its 4 neighbours vs its 4 non-neighbours.
  Neighbours should be significantly closer.

## Limitations

- Hilbert space is 2^9 = 512 dimensions; the approach cannot scale beyond ~12 qubits.
- The native CUDA version (`src/native/grid_2d/`) scales to 30×30 = 900+ cells
  via mean-field dynamics.
