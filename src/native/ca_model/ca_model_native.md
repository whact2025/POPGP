# Native Model: Cellular Automata (CUDA)

GPU-accelerated version of the cellular automata stability selection model.
The coherent qubit-qubit interaction phase runs on the CUDA kernel, while
selection, cooling, and reproduction remain in Python. Validates the same
emergent persistence as the toy model (`src/toy/ca_model/`) but on a 50×50 grid
(2500 sites vs 100).

Requires a CUDA GPU and the compiled `phase_flow` kernel (`popgp_engine/build.bat`).

## Framework Sections Validated

| Principle | Framework Reference | What this script tests |
|---|---|---|
| Stability Selection | Section 4.4.2a | Cells exceeding the entropy threshold are culled each step. |
| Emergent Persistence | Section 4.4.2a | Populations of stable cells self-organise and grow. |
| Radiative Cooling | Section 4.4.2a | Entropy export (cooling) sustains dense populations. |

## Key Difference: Toy vs Native

| Aspect | Toy (pure Python) | Native (CUDA hybrid) |
|---|---|---|
| Grid | 10×10 (100 sites) | 50×50 (2500 sites) |
| Interaction | Python `interact()` directly shrinks Bloch vectors | CUDA kernel (coherent SU(2) rotation) **+ CPU decoherence channel** |
| Cell state | Bloch vector (rx, ry, rz) | Qubit amplitudes (α, β) ↔ Bloch vector (converted each step) |
| Selection/Cooling | Python | Python (same logic) |

## Hybrid Architecture

Each time step involves two phases:

### GPU Phase -- Coherent Interaction
The CUDA kernel applies the full Heisenberg mean-field rotation:
```
U = exp(-i · J · dt · h⃗ · σ⃗)
```
This is a **unitary** (norm-preserving) operation. It exchanges amplitude
between |0⟩ and |1⟩ and evolves phases, but does not change single-cell
purity by itself.

### CPU Phase -- Decoherence Channel
After downloading the kernel's output, the CPU applies a **non-unitary** channel
that mirrors the toy model's `interact()` function:
1. **Purity decay**: Bloch vector shrinks proportional to misalignment with
   neighbours: decay ∝ (1 - dot²) × DECAY_J × dt.
2. **Alignment force**: small pull toward neighbour average (strength ALIGN_J × dt).

This two-phase approach separates the coherent quantum dynamics (GPU) from the
dissipative entropy-producing dynamics (CPU), while matching the toy model's
overall physics.

### CPU Phase -- Selection / Cooling / Reproduction
Same as the toy model:
- **Selection**: kill cells with entropy > `LEAKAGE_THRESHOLD`.
- **Cooling**: with probability `COOLING_PROB`, reset cell to a pure state
  (direction preserved, magnitude restored).
- **Reproduction**: stable cells (entropy < 0.1) can replicate into empty
  neighbours with small mutation.

## Parameters

| Parameter | Value | Role |
|---|---|---|
| WIDTH × HEIGHT | 50 × 50 | Grid size (2500 sites) |
| INITIAL_DENSITY | 0.4 | Fraction of sites seeded at start |
| dt | 0.05 | Phase-flow step size (tunable hyperparameter) |
| DECAY_J | 0.3 | Decoherence purity-decay coupling (tunable hyperparameter) |
| ALIGN_J | 0.1 | Decoherence alignment coupling (tunable hyperparameter) |
| LEAKAGE_THRESHOLD | 0.4 | Entropy death threshold (universal constant) |
| REPLICATION_PROB | 0.05 | Per-step replication probability (tunable) |
| MUTATION_RATE | 0.02 | Bloch-vector perturbation on replication (tunable) |
| COOLING_PROB | 0.02 | Per-step entropy reset probability (tunable) |
| STEPS | 80 | Total simulation steps |

## How to Run

```bash
uv run src/native/ca_model/ca_model_native.py
```

## Results and How to Interpret

### Population Dynamics

![Population Dynamics](results/dynamics_cooling.png)

**What you see**: a dual-axis line chart over time steps (x-axis).

**How to read it**:
- **Red line (left y-axis)** = live cell count at each step. A successful run
  shows the population growing from the initial seed (~1000 cells) and
  stabilising or continuing to grow (typical final population ~2300). A crash
  to zero means the cooling/interaction balance is wrong.
- **Blue line (right y-axis)** = average entropy of surviving cells. This should
  remain well below the `LEAKAGE_THRESHOLD` (0.4). Values near 0.1–0.2
  indicate a healthy, moderately pure population.
- **What to look for**: steady or rising population with low average entropy
  demonstrates that the stability selection + cooling mechanism produces
  self-sustaining populations. The population curve may show an initial dip
  (fragile cells dying) followed by recovery as cooling and replication take hold.

### Grid Evolution Animation

![Grid Evolution](results/evolution_cooling.gif)

**What you see**: an animated heatmap of the 50×50 grid, one frame per time step,
using the 'inferno' colour map.

**How to read it**:
- **Bright yellow/white cells** = high purity (Bloch radius near 1), low entropy,
  very stable. These are the "winners" of stability selection.
- **Dark red/orange cells** = moderate purity, approaching the death threshold.
  These are at risk of being culled.
- **Black squares** = empty sites (no cell present).
- Over time you should see:
  1. **Early frames**: sparse distribution of bright cells with many dark gaps.
  2. **Middle frames**: clusters of bright cells expanding as stable cells
     replicate into neighbouring empty sites.
  3. **Late frames**: large connected regions of stable (bright) cells filling
     most of the grid, with occasional dark patches where interactions are
     causing entropy to rise.
- The spatial clustering is emergent -- it arises from the combination of
  local interaction (neighbours influence each other), selection (high-entropy
  cells die), and replication (stable cells spread locally). No clustering
  rule was programmed; it is a consequence of the framework's principles.
