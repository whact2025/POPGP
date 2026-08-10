# 8-qubit chain stability and embedding benchmark

This exact finite benchmark exercises partition selection, mutual-information
topology, MDS dimension selection, and the placeholder clock pipeline. It supports
those named toy computations; it does not prove a causal factorization attractor or
large-system emergence.

## Run

```text
uv run python -m examples.physics_qg.chain_1d
```

The exact backend uses a 256-dimensional Hilbert space, exact diagonalization,
partial traces, entropy, and mutual information.

## Partition-selection benchmark

The pipeline enumerates all 105 partitions of eight qubits into four two-qubit cells.
For each admissible partition it estimates a channel-leakage objective using the same
seeded Haar probes, avoiding candidate-dependent Monte Carlo noise. It checks sampled
global SU(2) equivariance and a permissive retention bound, then uses drift only for a
declared leakage tie.

At the published parameters it selects:

`[[0,1], [2,3], [4,5], [6,7]]`

A separate Néel-state evolution compares these selected contiguous cells with one
chosen scattered partition. The final mean entropy increases are approximately 1.11
and 1.30 respectively. This is a finite illustrative comparison, not a proof that all
non-local decompositions are dynamically unstable.

The selected partition was also observed for common-probe seeds 0, 1, 2, 42, and 99.
Leakage estimates varied from roughly 0.0051 to 0.0061, so the ordering is stable in
this limited sweep but the magnitude is sampling-dependent.

![Selected versus comparison-cell entropy](results/entropy_growth.png)

## Correlation embedding

The thermal-state cell MI matrix is converted to positive edge lengths and blind
adaptive-gap connectivity. Graph geodesics are embedded with classical MDS. Current
output:

- selected embedding dimension D*=1;
- finite-graph heat-kernel dimension peak approximately 0.84;
- near-zero MDS stress at displayed precision; and
- recovered cell order equal to the chain order up to reflection.

![One-dimensional embedding](results/embedding.png)

The diffusion value is a finite-graph peak on four nodes, not a continuum spectral
dimension. The Hamiltonian already encodes nearest-neighbor chain interactions; the
test asks whether the correlation pipeline recovers that encoded locality without
receiving coordinates or reference edges.

The inferred three-edge cell graph matches the held-out coarse Hamiltonian graph, but
this four-cell case is non-discriminating: the mandatory minimum spanning tree alone
produces the same edge set. It is an ordering/consistency benchmark, not evidence that
the adaptive-gap rule independently discovered the chain.

## Clock panel

![Placeholder clock potential](results/clock_potential.png)

The plotted source is local von Neumann entropy from the standard pipeline. It is
explicitly non-physical and must not be interpreted as a gravitational field. End
versus interior variation is a boundary/entropy diagnostic only.

## Published parameters

| Parameter | Value | Note |
|---|---:|---|
| Qubits | 8 | Exact finite benchmark |
| Cell size | 2 | 105 equal-size partitions |
| β | 1.0 | Thermal state for projection pipeline |
| Stability state | Néel product state | Separate dynamics comparison |
| Evolution `dt`, steps | 0.1, 20 | Comparison curve |
| Leakage window samples | 5 | Numerical control |
| Leakage probes | 8 | Common seeded probes |
| Retention ε | 10.0 | Deliberately permissive, not a derived capacity scale |

## Backend limitation

The mean-field CUDA backend used above the exact threshold evolves product states and
cannot represent entanglement or compute MI. It therefore fails explicitly in the
locality stage. No large-N claim follows from this exact example.

## Artifacts

- `results/entropy_growth.png`
- `results/embedding.png`
- `results/clock_potential.png`
- `results/validation.json`
