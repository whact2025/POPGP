# Reproducibility record

Audit date: 2026-08-09 (America/New_York).

Review-protocol remediation verified: 2026-08-10 (America/New_York).

Blackwell native-build calibration verified: 2026-08-13 (America/New_York).

Blackwell review remediation verified: 2026-08-17 (America/New_York).

## Environment

- Windows / PowerShell
- Python 3.11.15
- uv 0.11.11
- CMake 4.3.2
- Locked environment from `uv.lock`
- NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU, compute capability 12.0,
  driver 595.79, 12,227 MiB
- CUDA 13.3.73 development tools in the isolated local extraction
  `C:\src\POPGP-cuda-toolkit-13.3\local`; the installer URL and SHA-256 are pinned
  in `popgp_engine/kernel/README.md`; the locked Python environment remains CPU-only
  (`torch 2.10.0+cpu`)
- pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)
- GitHub CLI 2.97.0; authenticated as `fuocor`

GitHub CLI access was verified against pull request #1. `docs/framework.tex` is
authoritative for the revised manuscript. A two-pass fresh PDF build produced an
11-page, 535,368-byte artifact with SHA-256
`8fd4be5c2003dde42f1ff390a0d3e4a955167e58524c16f5c7114944c9759323`;
layout warnings and the missing bibliography remain recorded limitations.

## Commands

```text
uv sync --frozen --no-editable
uv run --frozen --no-editable ruff check .
uv run --frozen --no-editable python scripts/check_tex.py
uv run --frozen --no-editable python -m pytest -q -p no:cacheprovider
uv run --frozen --no-editable python -m examples.physics_qg.chain_1d
uv run --frozen --no-editable python -m examples.physics_qg.grid_2d
uv run --frozen --no-editable python -m examples.physics_qg.gravity_well
uv run --frozen --no-editable python -m examples.physics_qg.source_law
uv run --frozen --no-editable python -m examples.physics_qg.source_law_many_body
uv run --frozen --no-editable python -m examples.physics_qg.ca_model
uv run --frozen --no-editable python -m scripts.check_validation_artifacts --enforce-change-boundary
```

Additional controlled checks exercised five common-probe seeds for chain partition
selection and swept k-NN values before adopting the blind adaptive-gap inference.

## Hardened results

| Command | Approx. runtime | Result |
|---|---:|---|
| `pytest -q` | about 1200 s | 379 passed |
| chain example | 15 s | contiguous blocks, D*=1, finite spectral peak≈0.84 |
| grid example | 6 s | 12/12 edges, P=R=1, D*=2; singleton Pi_res inadmissible |
| gravity diagnostic | 7 s | Green-function checks pass; singleton Pi_res inadmissible |
| source-law example | 7 s | RE slope≈1.9997; modular slope=1.0; negative result retained |
| many-body source-law example | 9 s | direct quadratic/floor gates and Kubo--Mori/Richardson susceptibility pass through β=3, including β=2.5 |
| CA analogy | 9 s | population declined 33 to 27; overall check fails; PNG/GIF/JSON regenerated |

The original 2026-08-13 native receipt is superseded as verification evidence because
its CTest invocation could succeed with zero discovered tests and its benchmark did
not validate kernel execution or output. The 2026-08-17 remediation build used MSVC
19.50, CUDA 13.3.73, the pinned vcpkg commit, and explicit architecture 120. The build
gate independently enumerated exactly seven CTest cases, all of which passed: three
phase-flow controls, boundary-cut calculation, pruning transitions, explicit clock
not-implemented behavior, and the self-validating benchmark. `cuobjdump --list-elf`
reported three `sm_120` cubins.

The hardened one-million-cell benchmark performed a warmup outside the timed region,
checked every launch and synchronization, read the final state back, and rejected
nonfinite or non-unit-norm output. A repeat completed 100 two-color steps in 448.57 ms
(`2.23e8` edge-updates/s), emitted FNV-1a-64 state checksum
`48e6ef8f40cb137c`, and reported maximum norm error
`1.1435297153639112e-14`. A separate CUDA-enabled PyTorch 2.10 probe exercised the
Python `GPUBackend` through the rebuilt DLL on 64 cells; the step was finite and
nontrivial, with maximum per-cell norm error `5.55e-16`. These remain implementation
and throughput calibrations, not evidence that the mean-field backend reproduces exact
MI/QCMI or the full projection pipeline.

All examples regenerated their committed `validation.json` artifacts. Wall-clock
timestamps have been removed, and every result artifact was byte-identical across two
consecutive runs in the locked Windows environment. Linux CI exposed bounded LAPACK
and floating-point drift despite unchanged scientific gates. CI therefore compares a
declared validation contract rather than serialized bytes: JSON schema, types, array
shapes, metadata, stable configuration, check identities, and pass/fail outcomes stay
strict; all numbers must be finite; ordinary diagnostics use narrow tolerances; and
the small set of sensitive fit/error fields has an explicit bounded allowlist in
`scripts/check_validation_artifacts.py`. Every registered check Boolean is also
recomputed from its retained typed operands. Required raster outputs must preserve
format/geometry. After canonicalizing the one declared near-zero chain legend value,
retained Windows/Linux evidence shows at most four intensity levels of per-channel
drift, so every raster channel is bounded by 4. This calibrated maximum rejects
compact features, one-pixel and dashed curves, rendered text, and actual plot-
annotation removal; encoded PNG bytes and metadata are not compared across
environments.

The prospective VIA-000 R2 protocol adds stricter evidence-integrity execution beyond
ordinary CI. The workflow creates a fresh locked non-editable environment and every
runtime cache outside the checkout. A base interpreter invoked with `-I -S` extracts
the boundary checker from the frozen Git object database and creates complete external
environment and literal-source manifests. `scripts/check_reproduction_boundary.py
run` hashes every environment file/symlink and compares worktree bytes/modes directly
to batched frozen Git blobs before and after each child; it does not trust index stat
flags or clean filters, and it permits no ignored checkout state. Python children use
`scripts/run_without_startup_hooks.py` under `-I -S` with an external bytecode cache;
the bootstrap inserts dependency directories directly and never evaluates `.pth`,
`sitecustomize.py`, or `usercustomize.py`. Every registered decision also binds
duplicated pipeline/check fields and recomputes its fits, quadratic/Richardson
assessments, first-law and static/evolved local-global identities, global-energy and
endpoint statistics, controls, sensitivity cases, and derived precision ratios from
the lowest-level retained raw values. Retained clock solves additionally carry their
complete mutual-information matrix, inferred-edge support, unmodified source,
removed constant mode, zero-mode policy, gauge choice, and configured mass. The
checker reconstructs the sparse symmetric nonnegative zero-diagonal weight matrix
from MI plus edges, reconstructs the effective source from the raw source and policy,
and only then recomputes the finite-graph equation. The localized gravity diagnostic
also derives its raw source from the declared center and point strength. The
checker rejects malformed, non-finite, mis-shaped, unordered, or zero-denominator
operands rather than accepting a stale serialized summary.
These mechanisms await a fresh preregistered two-platform R2 run; they are not a new
viability result.

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
