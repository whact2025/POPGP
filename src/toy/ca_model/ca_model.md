# Toy Model: Cellular Automata with Stability Selection

Pure-Python simulation of a population of qubit-like "cells" on a 10×10 grid.
Demonstrates the framework's **stability selection principle** in a dynamical,
biological-like context: cells that leak too much information (entropy exceeds
a threshold) die, while stable cells persist and replicate.

## Framework Sections Validated

| Principle | Framework Reference | What this script tests |
|---|---|---|
| Stability Selection | Section 4.4.2a | Cells with entropy above `LEAKAGE_THRESHOLD` are eliminated. |
| Emergent Persistence | Section 4.4.2a | Populations of stable cells self-organise and persist over time. |
| Radiative Cooling | Section 4.4.2a | Entropy export ("cooling") is necessary for dense, stable populations. |

## Cell Representation

Each cell is a single effective qubit represented by a Bloch vector (rx, ry, rz):
- **Radius** |r| = 1 → pure state (zero entropy, maximum stability).
- **Radius** |r| < 1 → mixed state (non-zero entropy, less stable).
- **Entropy**: S = -p₁ log(p₁) - p₂ log(p₂), where p₁,₂ = (1 ± |r|)/2.
- **Purity**: (1 + |r|²)/2.

## Algorithm (per time step)

### 1. Interaction Phase
For each cell with neighbours (Von Neumann neighbourhood):
- **Purity decay**: Bloch vector shrinks proportional to misalignment with neighbours.
  Decay rate ∝ (1 - dot²) × J × dt. Aligned cells decay less; orthogonal cells decay most.
- **Alignment force**: small pull toward neighbour average (self-organisation).

### 2. Selection Phase
- Cells with entropy > `LEAKAGE_THRESHOLD` (0.4) are removed from the grid.

### 3. Cooling Phase
- Each surviving cell has probability `COOLING_PROB` (0.02) of resetting to a pure
  state (direction preserved, magnitude restored to 1). Models radiative cooling /
  entropy export to an environment.

### 4. Reproduction Phase
- Cells with very low entropy (< 0.1) have probability `REPLICATION_PROB` (0.05) to
  replicate into an empty neighbouring site with small Bloch-vector mutation.

## Parameters

| Parameter | Value | Role |
|---|---|---|
| WIDTH × HEIGHT | 10 × 10 | Grid size |
| Initial density | 40% | Fraction of grid seeded with random pure-state cells |
| LEAKAGE_THRESHOLD | 0.4 | Entropy above which cells die (universal constant) |
| REPLICATION_PROB | 0.05 | Per-step replication probability for stable cells (tunable) |
| MUTATION_RATE | 0.02 | Bloch-vector perturbation on replication (tunable) |
| COOLING_PROB | 0.02 | Per-step probability of entropy reset (tunable) |
| Interaction J | 0.3 | Purity-decay coupling strength (tunable) |
| Alignment J | 0.1 | Alignment coupling strength (tunable) |
| STEPS | 50 | Total simulation steps |

## How to Run

```bash
uv run src/toy/ca_model/ca_model.py
```

## Results and How to Interpret

### Population Dynamics

![Population Dynamics](results/dynamics_cooling.png)

**What you see**: a dual-axis line chart over time steps (x-axis).

**How to read it**:
- **Red line (left y-axis)** = live cell count at each step. A healthy simulation
  shows the population rising from the initial seed, stabilising, or fluctuating
  around an equilibrium. A population crash to zero means cooling is insufficient.
- **Blue line (right y-axis)** = average entropy of surviving cells. This should
  remain below the `LEAKAGE_THRESHOLD` (0.4) since cells above it are culled.
  Lower average entropy means the population is collectively purer and more stable.
- **Key insight**: if you disable cooling (set `COOLING_PROB = 0`), the population
  collapses because interactions steadily increase entropy until every cell exceeds
  the death threshold. Cooling (entropy export) is what allows dense, persistent
  populations -- matching the framework's prediction that open systems with entropy
  export are necessary for stable structures.

### Grid Evolution Animation

![Grid Evolution](results/evolution_cooling.gif)

**What you see**: an animated heatmap of the 10×10 grid, one frame per time step,
using the 'inferno' colour map.

**How to read it**:
- **Bright yellow/white cells** = high purity (Bloch radius near 1), low entropy,
  very stable.
- **Dark red/orange cells** = moderate purity, approaching the death threshold.
- **Black squares** = empty sites (no cell present).
- Over time you should see clusters of bright (stable) cells growing and filling
  the grid, while isolated or misaligned cells darken and die. The spatial
  clustering is an emergent property of the interaction + selection dynamics.

## Key Insight

Without cooling, the population collapses: interactions always increase entropy,
eventually pushing every cell above the death threshold. With cooling (entropy
export), stable populations emerge and persist. This mirrors the framework's
requirement that open systems with entropy export are necessary for complex,
stable structures.
