# Scientific hardening project plan

Campaign-level viability tiers, dependency-ordered work packets, adversarial roles,
portable orchestration fields, and fail-closed decision rules are defined in the
[adversarial viability demonstration plan](VIABILITY_DEMONSTRATION_PLAN.md).

## PR 2 — source law and linear response

**Objective:** determine which vacuum-relative quantities can source a weak clock
constraint, separating analytic affine-family identities from discriminating
non-affine response tests.

- Equations: finite `D(ρ||σ)`, `Δ⟨Kσ⟩`, `ΔS`, and `(L+μ²I)Φ=s`.
- Implementation: information primitives, precision-floor-aware asymptotic fitting,
  direct Richardson/Kubo--Mori susceptibility, KMS equal-energy controls,
  negative-source/redshift tests, and parameter sweeps.
- Acceptance: analytic identities are labeled as such; quadratic response is verified
  by relative nested-window agreement, an absolute slope band, normalized residual,
  an absolute precision floor, and a mandatory first-order negative control. Signed
  first-order response must clear a stated Richardson error margin and match the exact
  Kubo--Mori susceptibility. Signs agree across code/docs; manual Green-function tests
  are labeled non-physical.
- Falsification: raw relative entropy is rejected as a standalone linear source if
  its Φ slope is quadratic or it changes the far-field source at fixed energy.
- Artifact: JSON/CSV scaling data plus plots and a source-law decision record.
- Out of scope: claiming Einstein closure or selecting a final law from one qubit.

Status: raw relative entropy fails the linear-source criteria. Under affine mixing,
modular energy, physical energy, and the linear clock solve are exactly proportional
to the mixture amplitude; their unit slopes are identity regressions. The many-body
experiment now uses `ρ(ε)∝exp[-β(H+εV)]`, verifies `D=O(ε²)` with direct asymptotic
and precision-floor gates, and finds a nonzero Richardson-extrapolated modular
susceptibility matching the exact Kubo--Mori value through β=3 in the declared finite
sweep (including β=2.5). Physical-energy response follows from the exact KMS identity
rather than a duplicate slope gate. An isospectral unitary control instead gives
`ΔS=0` and `D=Δ⟨K⟩=βΔ⟨H⟩=O(ε²)`, making the family qualifier explicit. A separate
quench provides the audited conservation/spreading result, while a commuting Ising
control shows that spreading is not universal. Scalable refinement, temporal
averaging, covariant conservation, and an independently measured clock observable
are still required.
The pipeline comparison also falsifies naive one-site reduced modular energy as a
local source in the symmetric KMS control. A separately named exact-backend
`−βΔ⟨h_i⟩` candidate repairs that blindness while keeping its microscopic
Hamiltonian dependence explicit.

## PR 3 — blind geometry, Regge, and closure foundations

**Objective:** separate inference from validation and construct intrinsic discrete
geometry with convergence diagnostics.

- Add `popgp/geometry/` modules for local SPD metric fits, neighborhood condition
  numbers, intrinsic complexes, Regge deficits/dual volumes, and closure residuals.
- Test label permutations, chains, square/triangular grids, feasible 3D lattices,
  expanders, random regular graphs, Bell-pair controls, and long-range Hamiltonians.
- Acceptance: inference never reads reference edges; metrics include P/R/F1 and
  distance distortion; local fits report uncertainty; flat and constant-curvature
  fixtures converge under refinement.
- Falsification: persistent false low-D geometry in controls, non-convergent metric
  fits, or curvature/closure residuals that do not decrease.
- Out of scope: interpreting a Delaunay visualization proxy as intrinsic geometry.

## PR 4 — evidence-scoped paper revision

**Objective:** produce a manuscript in which every empirical statement maps to a
reproducible artifact and every untested statement is labeled assumption/conjecture.

- Split implemented results, negative results, hypotheses, and roadmap.
- Add prior work on modular energy, relative entropy, tensor-network geometry,
  thermodynamic gravity, and Regge calculus.
- Acceptance: clean TeX build; claim matrix cross-references; figures regenerate;
  no claim of GR, Lorentz recovery, or singularity resolution without its test.
- Falsification: any headline conclusion depends on a placeholder or tuned example.

## PR 5 — controlled high-curvature completeness model

**Objective:** test a symmetry-reduced or spherical model through a regime where the
corresponding classical solution becomes incomplete.

- First recover its low-curvature GR limit with fixed calibration.
- Evolve with a universal finite-capacity modification; measure bounded invariants,
  constraint conservation, and causal-trajectory extendibility.
- Acceptance: convergent low-curvature recovery and deterministic continuation with
  no solution-specific free function; identify which theorem assumption changes.
- Falsification: tuning per solution, loss of constraint control, or inability to
  continue causal trajectories.
- Dependencies: PR 3 geometry/closure and a source decision from PR 2.

## Cross-cutting quality gates

- Python 3.11+, locked dependencies, lint, unit/scientific tests, fixed seeds.
- Store raw metrics and configuration, not only rendered figures.
- Treat exact, approximation, and native backend results as separate evidence tiers.
- Native CUDA checks are required before claiming scalable equivalence.
- Keep PRs reviewable; pilot repairs may be split before upstream submission.

## Open questions retained for follow-up

- Run a preregistered CA cooling study with matched cooling/no-cooling arms and at
  least 50 shared seeds; report survival and population effects with intervals.
- Extend blind topology recovery beyond MST-degenerate chains to larger lattices,
  long-range models, and non-geometric controls.
- Replace the fixed dimension penalty with a stress-driven, preregistered selection
  rule and test whether false geometry persists for Bell-pair controls.
- Derive the retention budget from an independently specified capacity law rather
  than selecting it as a toy-model tolerance.
- Test local-metric identifiability under neighborhood perturbations and refinement.
- Repeat the weakest KMS response cases in independently implemented extended
  precision; current complex128 signals clear the declared floor but are not an
  arbitrary-precision cross-check.
- Treat Lorentz recovery, covariant conservation, and continuum closure as open until
  direct numerical tests exist.
