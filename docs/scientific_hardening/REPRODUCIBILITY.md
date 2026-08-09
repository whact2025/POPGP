# Reproducibility record

Audit date: 2026-08-09 (America/New_York).

## Environment

- Windows / PowerShell
- Python 3.11.15
- uv 0.11.11
- CMake 4.3.2
- Locked environment from `uv.lock`
- `nvcc`: unavailable
- `pdflatex`: unavailable
- GitHub CLI 2.97.0; authenticated as `fuocor`

The absent CUDA and TeX toolchains prevent native compilation/tests and a fresh PDF
build in this environment. GitHub CLI access was verified against pull request #1.

## Commands

```text
uv sync
uv run ruff check popgp tests examples
uv run pytest -q
uv run python -m examples.physics_qg.chain_1d
uv run python -m examples.physics_qg.grid_2d
uv run python -m examples.physics_qg.gravity_well
uv run python -m examples.physics_qg.source_law
uv run python -m examples.physics_qg.ca_model
```

Additional controlled checks exercised five common-probe seeds for chain partition
selection and swept k-NN values before adopting the blind adaptive-gap inference.

## Hardened results

| Command | Approx. runtime | Result |
|---|---:|---|
| `pytest -q` | 4 s | 50 passed |
| chain example | 15 s | contiguous blocks, D*=1, finite spectral peak≈0.84 |
| grid example | 6 s | 12/12 edges, P=R=1, D*=2; singleton Pi_res inadmissible |
| gravity diagnostic | 7 s | Green-function checks pass; singleton Pi_res inadmissible |
| source-law example | 7 s | RE slope≈1.9997; modular slope=1.0; negative result retained |
| CA analogy | 9 s | population declined 33 to 27; overall check fails; PNG/GIF/JSON regenerated |

All examples regenerated their committed `validation.json` artifacts. Wall-clock
timestamps have been removed so repeated runs are deterministic under the locked
toolchain. PNG metadata remains environment-dependent, so cross-environment artifacts
should be compared by structured metrics rather than binary hashes.

## Negative and sensitivity results

- Relative entropy and its induced potential have fitted perturbative slope ≈2.
- Modular-energy variation and its induced potential have slope ≈1 under the affine
  mixture by exact linearity in the state; these are identity regressions, not evidence
  for a source law.
- Equal-energy states relative to a KMS reference have different entropy and raw
  relative entropy, exposing an entropy confound for a mass-source interpretation.
- The old fixed 3-nearest-neighbor grid inference had precision 0.75 and recall 1.0
  (16 inferred edges versus 12 reference edges).
- The new MI-gap rule recovers the selected grid exactly and passes label permutation
  equivariance. The four-cell chain result is MST-degenerate. Uniform and disjoint-Bell
  correlations are marked non-separable, but the Bell control still receives a false
  D*=1 `geometric_candidate` declaration.
- A Petersen expander selects D*=4 at the weak default penalty, but λ=1 can force
  D*=2 with more than twice the stress; the embedding gate reports that result as
  `poor_fit` rather than a geometric candidate.
- Local metric fits use a Frobenius-orthonormal symmetric basis, report an explicit
  underdetermined flag with fixed rank tolerance, and are invariant under rotations
  of the MDS frame. The canonical MDS frame removes arbitrary orientation changes
  from serialized tensors. Earlier regularized scalar diagnostics could shift by
  roughly 30–60% under a rotation of the degenerate 2D eigenspace. The 3×3 grid still
  has underdetermined boundary fits; its 2D Delaunay deficits are proxy-only.
- Chain partition selection is unchanged for probe seeds 0, 1, 2, 42, and 99, while
  leakage estimates vary by roughly 19% across that small sweep.
- The 3×3 grid provides only two nonzero radial shells; its log-distance fit cannot
  validate a continuum potential law.

## Baseline mismatches corrected

Before hardening, the public grid constructor used an incompatible two-qubit default,
the gravity example used a positive source and reversed redshift interpretation,
the grid visualization overlaid reference edges without edge-recovery metrics, and
the geometry report called a penalized objective “stress.” Those outputs should not
be compared as evidence-equivalent to the regenerated artifacts.
