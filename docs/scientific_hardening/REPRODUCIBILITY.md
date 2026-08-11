# Reproducibility record

Audit date: 2026-08-09 (America/New_York).

Review-protocol remediation verified: 2026-08-10 (America/New_York).

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
`docs/framework.tex` is authoritative for the revised manuscript; the committed PDF
predates the claims audit and must be rebuilt when a TeX toolchain is available.

## Commands

```text
uv sync
uv run ruff check popgp tests examples
uv run pytest -q
uv run python -m examples.physics_qg.chain_1d
uv run python -m examples.physics_qg.grid_2d
uv run python -m examples.physics_qg.gravity_well
uv run python -m examples.physics_qg.source_law
uv run python -m examples.physics_qg.source_law_many_body
uv run python -m examples.physics_qg.ca_model
uv run python scripts/check_validation_artifacts.py
```

Additional controlled checks exercised five common-probe seeds for chain partition
selection and swept k-NN values before adopting the blind adaptive-gap inference.

## Hardened results

| Command | Approx. runtime | Result |
|---|---:|---|
| `pytest -q` | ~1054 s | 179 passed |
| chain example | 15 s | contiguous blocks, D*=1, finite spectral peak≈0.84 |
| grid example | 6 s | 12/12 edges, P=R=1, D*=2; singleton Pi_res inadmissible |
| gravity diagnostic | 7 s | Green-function checks pass; singleton Pi_res inadmissible |
| source-law example | 7 s | RE slope≈1.9997; modular slope=1.0; negative result retained |
| many-body source-law example | 9 s | direct quadratic/floor gates and Kubo--Mori/Richardson susceptibility pass through β=3, including β=2.5 |
| CA analogy | 9 s | population declined 33 to 27; overall check fails; PNG/GIF/JSON regenerated |

All examples regenerated their committed `validation.json` artifacts. Wall-clock
timestamps have been removed, and every result artifact was byte-identical across two
consecutive runs in the locked Windows environment. Linux CI exposed bounded LAPACK
and floating-point drift despite unchanged scientific gates. CI therefore compares a
declared validation contract rather than serialized bytes: JSON schema, types, array
shapes, metadata, stable configuration, check identities, and pass/fail outcomes stay
strict; all numbers must be finite; ordinary diagnostics use narrow tolerances; and
the small set of sensitive fit/error fields has an explicit bounded allowlist in
`scripts/check_validation_artifacts.py`. Required visual outputs must be tracked,
present, and nonempty. CI does not verify that every checked-out visual was rewritten
during the current run, and PNG pixels and metadata are not hashed across environments.

## Negative and sensitivity results

- Relative entropy and its induced potential have fitted perturbative slope ≈2.
- Under an affine mixture, modular energy, physical energy, and the linear clock solve
  are exactly proportional to ε. Their unit slopes are analytic identities and are
  excluded from the robustness claim.
- In an exact five-site non-affine KMS family `ρ(ε)∝exp[-β(H+εV)]`, relative
  coefficient, absolute slope, normalized residual, and precision-floor gates verify
  `D=O(ε²)` and reject a synthetic first-order control. Signed Richardson estimates
  give a nonzero modular susceptibility and match exact Kubo--Mori values across the
  declared Heisenberg/Ising, β≤3 (including 2.5), and three-size finite sweep. Energy
  response is not independently gated because it is fixed by the KMS identity. An
  isospectral unitary
  control instead has `ΔS=0` and quadratic `D=Δ⟨K⟩=βΔ⟨H⟩`. A separate quench
  conserves the global Hamiltonian expectation while the audited profile spreads
  under Heisenberg dynamics; the commuting
  Ising profile remains stationary. This is a family-dependent feasibility result.
- The one-site reduced modular-energy mode is numerically blind in the symmetric KMS
  chain. The separately named exact-backend `−βΔ⟨h_i⟩` candidate reproduces the
  audited local profile. Its runtime requires the supplied reference to match the
  backend Gibbs state at the same β within trace distance `1e-10`; under that premise
  it sums to minus the global modular-energy change. Its microscopic decomposition
  dependence is retained as a limitation.
- Equal-energy states relative to a KMS reference have different entropy and raw
  relative entropy, exposing an entropy confound for a mass-source interpretation.
- The old fixed 3-nearest-neighbor grid inference had precision 0.75 and recall 1.0
  (16 inferred edges versus 12 reference edges).
- The new MI-gap rule recovers the selected chain/grid and passes label permutation
  equivariance. The four-cell chain is MST-degenerate. Uniform and disjoint-Bell
  correlations are marked non-separable, but the Bell control still receives a false
  D*=1 `geometric_candidate` declaration.
- A Petersen expander selects D*=4 at the weak default penalty, but λ=1 can force
  D*=2 with more than twice the stress; the embedding gate reports that result as
  `poor_fit` rather than a geometric candidate.
- Local metric fits use a Frobenius-orthonormal symmetric basis, report an explicit
  underdetermined flag with fixed rank tolerance, and are invariant under rotations
  of the MDS frame. The canonical MDS frame fixes the full represented subspace,
  including rank-deficient degenerate embeddings, and removes arbitrary orientation
  changes from serialized tensors. Earlier regularized scalar diagnostics could shift by
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
