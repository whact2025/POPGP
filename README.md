# POPGP: finite quantum correlation and clock-constraint experiments

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18728240-blue)](https://doi.org/10.5281/zenodo.18728240)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

POPGP is a research framework and simulator for testing whether finite quantum
correlations can support relational topology, low-dimensional embeddings, and
clock-rate constraints. The current code provides controlled toy-model evidence;
it does not derive General Relativity, establish a physical gravitational source
law, or demonstrate singularity resolution.

> The theoretical proposal is [*A Phase-Ordered Pre-Geometric Projection
> Framework*](https://doi.org/10.5281/zenodo.18728240) (R. Fuoco, 2026). The
> manuscript contains conjectures and proposed falsification criteria beyond the
> implemented simulator.

## Toy-model validation results

All figures below are reproducible with fixed seeds and machine-readable reports.

<p align="center">
  <img src="examples/physics_qg/grid_2d/results/embedding.png" width="48%" alt="MDS embedding from an inferred MI graph"/>
  &nbsp;
  <img src="examples/physics_qg/gravity_well/results/gravity_well.png" width="48%" alt="Negative-source graph clock constraint"/>
</p>

- **Left:** A label-blind correlation-gap rule applied to the mutual-information
  matrix of a 3×3 Heisenberg thermal state recovers all 12 held-out interaction
  edges in this benchmark (precision and recall 1.0), then selects a two-dimensional
  MDS embedding. The Hamiltonian contains a grid interaction graph, so this is
  coordinate-free reconstruction of encoded locality—not topology-free emergence.
- **Right:** A small negative diagnostic source on the inferred weighted graph
  produces a negative potential well, slower source clocks, and positive
  emitter-to-boundary redshift under `dτ ∝ exp(Φ)`. This validates the numerical
  Green-function solve and sign convention only; the source is manually injected.

Regenerate them with:

```text
uv run python -m examples.physics_qg.grid_2d
uv run python -m examples.physics_qg.gravity_well
```

## Quick start

Install [uv](https://docs.astral.sh/uv/), then run:

```text
uv sync
uv run pytest -q
uv run ruff check popgp tests examples

uv run python -m examples.physics_qg.chain_1d
uv run python -m examples.physics_qg.grid_2d
uv run python -m examples.physics_qg.gravity_well
uv run python -m examples.physics_qg.source_law
uv run python -m examples.physics_qg.ca_model
```

Programmatic usage:

```python
from popgp import Simulator, SimulatorConfig

config = SimulatorConfig.for_grid(width=3, height=3, beta=2.0)
result = Simulator(config).run()

print(f"Selected embedding dimension: {result.pi_geom.D_star}")
print(f"Blindly inferred edges: {result.pi_loc.edges}")
print(f"Clock source status: {result.pi_time.source_status}")
```

The advertised grid constructor uses one qubit per cell and runs as written.
Exact density-matrix simulation is the supported end-to-end path. Above the exact
threshold, the mean-field CUDA backend cannot compute mutual information and the
locality stage fails explicitly instead of substituting a false MI proxy.

## What is implemented

### Resolution selection (`Π_res`)

For small exact systems, the code enumerates equal-size partitions, applies a
retention constraint and sampled SU(2)-equivariance check, and ranks admissible
partitions using common-random-number leakage probes with a drift tie-breaker. In
the 8-qubit chain benchmark it selects contiguous two-qubit blocks. Exhaustive
search is a finite toy-model surrogate; the proposed causal gradient-flow
attractor and its equivalence to this minimizer have not been established.

### Correlation topology (`Π_loc`)

The exact backend computes pairwise quantum mutual information. The default blind
connectivity rule identifies the largest multiplicative gap in positive MI values,
reports whether that gap clears a configured separability threshold, and adds a
minimum spanning tree only if needed for connectivity. A fixed-k k-NN baseline is
still available. Ground-truth Hamiltonian edges are used only after inference for
validation metrics and plots.

The controlled 8-site chain and 3×3 grid are exactly recovered at the published
parameters. This result is not yet evidence of robustness to long-range models,
non-geometric controls, larger lattices, or topological entanglement. The proposed
QCMI/Markov filtering stage is not implemented.

### Embedding diagnostics (`Π_geom`)

Classical MDS embeds graph-geodesic distances. The selected dimension minimizes

`stress(D) + λ_dim (D - D_spectral)^2`

over dimensions supported by the number of graph nodes. Reports keep unpenalized
stress separate from the selection objective. The heat-kernel quantity is reported
as a scale-dependent **finite-graph peak**, because finite graphs have zero spectral
dimension in both asymptotic limits. The current benchmarks select D*=1 for the
chain and D*=2 for the grid, but parameter and refinement studies remain required.

The pipeline now fits regularized SPD metrics in the selected MDS coordinates and
reports rank, condition number, and residual at every node. For 2D geometric
candidates it can also build an explicitly labeled embedding-space Delaunay/angle-
deficit proxy. These are diagnostic prototypes: intrinsic complex construction,
dual-volume Regge curvature, refinement convergence, Einstein closure, and
Bianchi/conservation tests are not implemented.

### Clock constraint (`Π_time`)

The numerical solver handles the graph-Laplacian zero mode without pinning an
arbitrary node. For an unscreened finite graph it either subtracts the constant
source mode or rejects an incompatible source; screened solves can be normalized
to report only clock-rate contrasts. The redshift convention is

`1 + z = exp(Φ_observer - Φ_emitter)`.

The standard pipeline still defaults to local von Neumann entropy as an explicitly
non-physical placeholder. Explicit negative relative-entropy and modular-energy
candidate modes now accept a caller-supplied reference state, but temporal averaging,
conservation, localization, and a validated physical KMS source remain open.
Scientific regression tests establish a negative
result important to the program: near a faithful reference, relative entropy and
the resulting potential begin at second order in perturbation amplitude, whereas
modular-energy variation begins at first order. Equal-energy KMS controls also show
that raw relative entropy changes with entropy at fixed energy. Raw relative entropy
is therefore falsified as a standalone linear mass source in this regime; modular
energy is retained as a candidate, not declared a final law.

## Repository map

- `popgp/`: exact simulator, information primitives, geometry diagnostics, and CUDA binding.
- `tests/unit/`: mathematical and implementation invariants.
- `tests/scientific/`: source-law, sign, scaling, topology, and permutation tests.
- `examples/physics_qg/`: reproducible finite toy models and validation JSON.
- `docs/scientific_hardening/`: claim classification, theory/code gaps, and staged
  falsification plan.
- `popgp_engine/`: experimental CUDA components. The native clock solver is still
  an identity stub, and renderer curvature code is not integrated with the Python
  projection pipeline.

## Current evidence and open claims

| Topic | Current status |
|---|---|
| Exact finite-dimensional entropy and relative entropy | Implemented with support checks |
| Chain/grid MI edge recovery | Passes selected finite benchmarks |
| Label-permutation invariance | Covered by scientific regression test |
| Dimension selection | Passes selected chain/grid parameters; not a continuum result |
| Finite graph clock constraint and redshift sign | Numerically validated |
| Physical Araki/KMS source | Candidate APIs exist; raw RE linear source falsified |
| QCMI filtering and non-geometric controls | Not implemented / incomplete |
| Local metric / angle deficits | Embedding-space diagnostics; intrinsic Regge geometry open |
| Newtonian/GR closure | Not demonstrated |
| Lorentz recovery | Conjectural |
| Singularity resolution or causal completeness | Conjectural |

## Reproducibility and review

Each example writes `results/validation.json`; a passing JSON check supports only
the criterion named in that check. Negative results are retained. See
[`docs/scientific_hardening`](docs/scientific_hardening/) for claim-by-claim scope,
acceptance criteria, and unresolved risks.

Contributions should include unit tests for implementation changes and scientific
regression or falsification tests for physical claims. Please report contradictions
between the manuscript, examples, and code as first-class issues.

## Citation

```bibtex
@misc{fuoco2026popgp,
  author       = {Fuoco, Richard},
  title        = {A Phase-Ordered Pre-Geometric Projection Framework},
  year         = {2026},
  howpublished = {\url{https://github.com/rfuoco/POPGP}},
  note         = {v1.0, Submission Draft}
}
```

POPGP is developed by Richard Fuoco and released under the MIT License.
