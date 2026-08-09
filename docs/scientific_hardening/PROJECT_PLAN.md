# Scientific hardening project plan

## PR 2 — source law and linear response

**Objective:** determine which vacuum-relative quantities can source a weak clock
constraint with the required first-order response.

- Equations: finite `D(ρ||σ)`, `Δ⟨Kσ⟩`, `ΔS`, and `(L+μ²I)Φ=s`.
- Implementation: information primitives, uncertainty-aware power-law fitting,
  KMS equal-energy controls, negative-source/redshift tests, and parameter sweeps.
- Acceptance: analytic cases pass; slope estimates include standard errors; signs
  agree across code/docs; manual Green-function tests are labeled non-physical.
- Falsification: raw relative entropy is rejected as a standalone linear source if
  its Φ slope is quadratic or it changes the far-field source at fixed energy.
- Artifact: JSON/CSV scaling data plus plots and a source-law decision record.
- Out of scope: claiming Einstein closure or selecting a final law from one qubit.

Pilot status: the core analytic tests are implemented and raw relative entropy fails
the linear-source criteria. Modular energy passes first-order scaling and remains a
candidate. A spatially localized many-body KMS experiment is still required.

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
