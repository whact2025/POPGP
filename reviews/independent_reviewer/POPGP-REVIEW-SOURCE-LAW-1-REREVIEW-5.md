# Independent re-review: POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-5

```yaml
artifact_schema_version: 1
review_id: "POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-5"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "claude-opus-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
review_date: "2026-08-10"
commit_reviewed: "3974bbfab89d97bac79ce298285a7ce8f36f69fd"
baseline_commit: "c03800e47ba4988bd125b81acd3a1b6ae07728f7"
prior_review_ref: "676e03d6de2405414df417a4d47c444733514256:reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-4.md"
builder_response_ref: "reviews/codex/POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-4.md at 3974bbfab89d97bac79ce298285a7ce8f36f69fd"
context_hash: "672e171a6532159c388aba804188fba28295c835"
context_hash_method: "git rev-parse \"3974bbfab89d97bac79ce298285a7ce8f36f69fd^{tree}\""
files_reviewed:
  - "C:/src/POPGP-review-source-law-rereview-5/.github/workflows/ci.yml"
  - "C:/src/POPGP-review-source-law-rereview-5/.gitignore"
  - "C:/src/POPGP-review-source-law-rereview-5/README.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/framework.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/framework.tex"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/governance/REVIEWER_IDENTITY.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/scientific_hardening/DECISIONS.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/scientific_hardening/REPRODUCIBILITY.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "C:/src/POPGP-review-source-law-rereview-5/docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "C:/src/POPGP-review-source-law-rereview-5/examples/physics_qg/ca_model/__main__.py"
  - "C:/src/POPGP-review-source-law-rereview-5/examples/physics_qg/chain_1d/__main__.py"
  - "C:/src/POPGP-review-source-law-rereview-5/examples/physics_qg/gravity_well/__main__.py"
  - "C:/src/POPGP-review-source-law-rereview-5/examples/physics_qg/grid_2d/__main__.py"
  - "C:/src/POPGP-review-source-law-rereview-5/examples/physics_qg/source_law/__main__.py"
  - "C:/src/POPGP-review-source-law-rereview-5/examples/physics_qg/source_law_many_body/__main__.py"
  - "C:/src/POPGP-review-source-law-rereview-5/examples/physics_qg/*/results/validation.json (all six)"
  - "C:/src/POPGP-review-source-law-rereview-5/popgp/backend.py"
  - "C:/src/POPGP-review-source-law-rereview-5/popgp/coarse_grain.py"
  - "C:/src/POPGP-review-source-law-rereview-5/popgp/config.py"
  - "C:/src/POPGP-review-source-law-rereview-5/popgp/diagnostics.py"
  - "C:/src/POPGP-review-source-law-rereview-5/popgp/simulator.py"
  - "C:/src/POPGP-review-source-law-rereview-5/pyproject.toml"
  - "C:/src/POPGP-review-source-law-rereview-5/reviews/codex/POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-4.md"
  - "C:/src/POPGP-review-source-law-rereview-5/reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-4.md"
  - "C:/src/POPGP-review-source-law-rereview-5/scripts/check_tex.py"
  - "C:/src/POPGP-review-source-law-rereview-5/scripts/check_validation_artifacts.py"
  - "C:/src/POPGP-review-source-law-rereview-5/tests/scientific/test_many_body_source_law.py"
  - "C:/src/POPGP-review-source-law-rereview-5/tests/scientific/test_source_law_controls.py"
  - "C:/src/POPGP-review-source-law-rereview-5/tests/scientific/test_topology_recovery.py"
  - "C:/src/POPGP-review-source-law-rereview-5/tests/unit/test_backend.py"
  - "C:/src/POPGP-review-source-law-rereview-5/tests/unit/test_claim_wording.py"
  - "C:/src/POPGP-review-source-law-rereview-5/tests/unit/test_diagnostics.py"
  - "C:/src/POPGP-review-source-law-rereview-5/tests/unit/test_review_guidance.py"
  - "C:/src/POPGP-review-source-law-rereview-5/tests/unit/test_simulator.py"
  - "C:/src/POPGP-review-source-law-rereview-5/tests/unit/test_validation_artifact_contract.py"
  - "git diff 4caa4f16b407a47a9531442751ccde1e446b8271..3974bbfab89d97bac79ce298285a7ce8f36f69fd (complete, all 23 files)"
  - "git archive 3974bbfa / 4caa4f16 / c03800e4 (disposable byte copies; all mutation and A/B execution)"
  - "hybrid tree: popgp/ from 4caa4f16 + tests/ from 3974bbfa (negative-control verification of the new tests)"
access_level: public-repository-only
independence_statement: |-
  This is process separation, not external scientific validation. Stated plainly, with the
  weaknesses first.

  1. MODEL. The reviewing seat is claude-opus-5, per this runtime's own identity report; the
     version/snapshot string is not exposed to me and is recorded as `unknown` rather than
     guessed. The builder is gpt-5.6-sol, so the model DOES differ from the builder. It does
     NOT differ from the seat that produced rounds 3 and 4 of this chain, which was also
     claude-opus-5. This round therefore verifies findings written by the same model for the
     third consecutive round. A systematic error shared by rounds 3, 4 and 5 is invisible to
     this process. That is the single largest independence weakness here, and it has now
     compounded across three rounds rather than being corrected.
  2. SESSION. This IS a fresh session. It began with the frozen tree, the handoff metadata and
     the reading list, and had no access to the working context, intermediate reasoning or
     unrecorded conclusions of rounds 3 or 4. What it did receive is what the workflow requires
     a re-review to receive: the committed REREVIEW-4 artifact and RESPONSE-4. Round 4 recorded
     `shared_session: true`; this round records false, and that is a genuine improvement.
  3. ORCHESTRATOR. No external orchestrator relayed information between builder and reviewer.
     Within this session I did spawn subagents; they are not independent of me or of each other,
     they inherit my operator and access boundary, and I treat their output as my own work
     rather than as corroboration. I record `shared_orchestrator: false` because no orchestrator
     sits between the two ROLES, which is what the field asks.
  4. OPERATOR. Richard Fuoco operates both roles and has operated every round of this chain.
     There is no operator separation anywhere.
  5. DIRECTED ATTENTION. The task named the five findings, the seven tests and the two
     unresolved clauses to check. Those items were therefore found by direction. Where I report
     confirming a round-4 claim, discount accordingly. The fifteen NEW findings below were not
     directed; the task asked for new issues without naming any.
  6. WHAT IS ACTUALLY EVIDENCE. Every prior finding was re-verified by replaying its ORIGINAL
     defect condition against the prior candidate and then against this one, not by reading the
     builder response or the round-4 artifact. Every requested test was checked by running it
     against pre-fix source. Every governance claim was checked by mutation. Fourteen candidate
     findings were refuted under adversarial verification and are excluded, including two I had
     drafted myself. That is real evidence about specific propositions. It is not independent
     experimental confirmation of any physical claim in this repository.
  7. CONCLUSION. A review by a different model, a different operator and an unaffiliated group
     remains warranted before any external claim is made. This artifact does not satisfy an
     external-validation requirement and must not be cited as one.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: false
  builder_model_identity: "gpt-5.6-sol"
  reviewer_model_differs_from_builder: true
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  All five round-4 findings are verified-resolved and all seven round-4 requested tests are
  verified-satisfied. The two clauses left unresolved since round 3 - TST-GOV-003 clause 2 and
  TST-GOV-004 clause (c) - are now satisfied as well. This is the first round of this chain with
  zero unresolved blocking findings.

  The remediation is real and, on the two items that mattered, better than a minimal fix.
  REG-001's unbounded mutual recursion is not patched with a depth guard: the rank test was moved
  to global singular values and the label-ordered greedy scan was replaced by a maximum-volume
  scan, so the recursion now provably terminates because the represented column count strictly
  decreases. I reproduced RecursionError at the prior candidate on all 19 specified inputs and
  finite coordinates on all 19 here. MDS-005 took the correct branch as well: the ill-conditioned
  Gram inverse-square-root was replaced by an SVD polar factor, and measured isometry improved
  from 1.185e-08 to 2.5e-15 worst case across a scale sweep, while right-orthogonal equivariance
  - the LIB-004 property - improved from 7.945e-07 to 3.857e-15 over 1500 randomized cases rather
  than regressing. The published consequence is visible and in the honest direction: chain_1d
  stress_by_dimension["2"] moved 6.639527346370152e-10 -> 5.034432205558175e-16, removing the
  canonicalization noise round 4 identified as 1.3e6x the true value.

  No gate outcome moved anywhere. grid_2d's coordinates changed only by a proper rotation
  (orthogonal Procrustes residual 1.776e-15 against a coordinate scale of 2.5487, det +1.000000,
  max relative pairwise-distance change 4.382e-16). All 18 committed artifacts regenerate
  byte-identically and deterministically across two full runs.

  Fifteen new findings are recorded, all low severity, none blocking, none affecting a published
  number, a gate outcome or a scientific claim. The most consequential group is a family of
  absolute-versus-relative tolerance mismatches that the REG-001 fix introduced into the same
  function (MDS-006, MDS-007): _canonicalize_embedding now raises RuntimeError or a bare
  StopIteration on inputs the prior candidate embedded, purely because their absolute coordinate
  scale is small. I record these as non-blocking after measuring what round 4 could not: through
  the public Simulator.run() API, Pi_geom hard failures went from 5 in 420 configurations at the
  prior candidate to 0 in 420 here. Round 4's stated reachability bound for REG-001 ("not
  reachable through any supported SubstrateConfig today") was too generous - it was reachable -
  which makes the fix more valuable than round 4 credited and the new corner strictly narrower.

  The governance layer genuinely hardened. All four mutations round 4 demanded now fail, plus
  five more I added. The un-closed half is filed as GOV-013: an acceptance gate added directly to
  example code, committed with its regenerated artifact, still passes the entire authoritative CI
  green, while GATE_TEST_REGISTRY.md:3-4 asserts every active gate has an executable control.
  I reproduced that end to end.

  Recommendation: approve, 0 blocking findings.

findings:
  - id: "MDS-006"
    severity: low
    category: code
    location: "popgp/simulator.py:847 (`if maximum <= epsilon * scale: raise RuntimeError`) with `scale = max(1.0, float(torch.linalg.matrix_norm(centered).item()))` at :835, against the relative rank test `rank_tolerance = 1e-8 * float(singular_values[0].item())` at :809; introduced by 21baacd2c98f86533c51a38080294ad1a05ea73d"
    evidence: |-
      The rank test that decides whether to enter _select_embedding_anchors is RELATIVE
      (1e-8 * sigma_max, :809-812). The degeneracy guard inside that scan is ABSOLUTE: because
      `scale` is clamped by `max(1.0, ...)`, the threshold never falls below
      eps = 2.220446e-16 no matter how small the configuration is. Whenever the smallest
      retained singular value lies between those two tolerances the two disagree and the code
      raises rather than reconciling them - the message
      "embedding rank and anchor selection disagree" names the disagreement exactly.

      EXECUTED by me in disposable `git archive` copies, every run asserting
      os.path.abspath(popgp.__file__) resolves under the intended tree (the venv contains
      `_editable_impl_popgp.pth` pointing at the frozen worktree, which silently shadows a naive
      run; my first attempt hit exactly that and was discarded):
        - Uniform rescaling of a supported topology. `d` = 6-node OPEN CHAIN hop metric
          (torch.cdist of arange(6) reshaped, p=1). `Simulator._classical_mds(d * s, 2)` at the
          candidate: OK for s = 1e0..1e-9, 1e-12, 1e-14, 1e-15; RuntimeError for
          s = 1e-10, 1e-11, 1e-13. At the prior candidate 4caa4f16 every one of the 16 scales
          returns finite coordinates. The failure is non-monotone in s, which is the signature of
          a tolerance artifact rather than a genuine degeneracy.
        - Isometry over the same sweep, for contrast: candidate max relative pairwise-distance
          error <= 2.498e-15 at every scale that succeeds; prior candidate up to 1.185e-08.
        - Randomized sweep of 4000 matrices (varied n, D, scale, conditioning, duplicated rows,
          collapsed rows): candidate 3818 ok / 182 RuntimeError; prior candidate 4000 ok / 0.
        - Minimal 2-column family with prescribed singular values: sv = [1e-8, 2e-16] raises
          (rank test says 2 because 2e-16 > 1e-16; anchor guard says degenerate because
          2e-16 <= 2.22e-16). sv = [1.0, 1e-8] and [1e-3, 1e-10] both succeed.
      REACHABILITY, MEASURED RATHER THAN ASSERTED. I swept the PUBLIC entry point
      `Simulator(cfg).run()` over 420 configurations (chains n=4..9 x {open, periodic} x
      beta in {0.5, 1.0, 2.0} x I_0_multiplier in {1+1e-6 ... 1+1e-14, e}, plus 60 grid
      configurations). Candidate: 0 RuntimeError and 0 RecursionError. Prior candidate: 5
      RecursionError (REG-001, e.g. n=7 open beta=0.5 I_0_multiplier=1+1e-8). So public-API
      availability strictly IMPROVED. The new corner is reachable only by calling the private
      `_classical_mds` / `_canonicalize_embedding` directly, or by a distance matrix whose
      absolute scale is below ~1e-8, which the default kernel
      (`f(u)=max(0,-log u)` at popgp/config.py:33-38 with `I_0 = e * max(I_ij)` at :169) cannot
      produce because every off-diagonal graph distance is then >= 1.
    finding: |-
      The REG-001 fix reintroduces, in a new form, the relative-versus-absolute tolerance
      mismatch that caused REG-001. `_canonicalize_embedding` accepts a matrix as full rank
      using a scale-free relative tolerance and then rejects it inside `_select_embedding_anchors`
      against an absolute floating-point floor, so `_classical_mds` - a mathematically
      scale-covariant operation - hard-fails purely because the input distances are small.
      Inputs the prior candidate embedded correctly now raise.
    failure_scenario: |-
      `Simulator._classical_mds(d, D)` where `d` is the 6-node open-chain hop metric scaled by
      1e-10 raises `RuntimeError: embedding rank and anchor selection disagree` for every
      D in 2..5. The identical matrix at scale 1e0 or 1e-9 returns finite, distance-preserving
      coordinates, as does the 1e-10 matrix at 4caa4f16. Nothing in `run_pi_geom` catches
      RuntimeError, so the whole Pi_geom stage aborts.
    consequence: |-
      Bounded, and I state the bound rather than implying reach it does not have. No shipped
      example, gate, published number or test is affected; all 142 tests pass and all 18
      artifacts regenerate byte-identically. Zero of 420 public-API configurations reach it,
      against 5 that reached REG-001 at the prior candidate, so this is a strictly narrower
      corner than the defect it replaced and the failure is loud and immediate rather than a
      1000-frame recursion. The substantive point is that a geometry routine's success now
      depends on the arbitrary length unit of its input, which is a property no reader would
      expect and no docstring discloses.
    required_action: |-
      Make the guard relative to the configuration being processed. Replace
      `scale = max(1.0, float(torch.linalg.matrix_norm(centered).item()))` at :835 with the
      largest singular value already computed at :808, so the guard reads
      `maximum <= epsilon * n_scale * sigma_max` and cannot disagree with the rank test at :809.
      Add the regression specified in TST-MDS-006.
    verification: confirmed-by-execution
    blocking: false

  - id: "MDS-007"
    severity: low
    category: code
    location: "popgp/simulator.py:849 (`tie_tolerance = 256.0 * epsilon * max(1.0, maximum)`), consumed at :850-852 and :854; introduced by 21baacd2c98f86533c51a38080294ad1a05ea73d"
    evidence: |-
      The same `max(1.0, ...)` clamp appears in the tie tolerance, giving it an absolute floor of
      256*eps = 5.684e-14. Whenever the largest residual norm is below that value, the selection
      threshold `maximum - tie_tolerance` at :851 is NEGATIVE, every remaining row satisfies the
      predicate, and `next(...)` returns the lowest-labelled row regardless of its residual. Two
      consequences, both executed:

      (a) NON-HOMOGENEOUS CANONICAL FRAME. Relative deviation of
      `_canonicalize_embedding(alpha*C)` from `alpha*_canonicalize_embedding(C)`:
      candidate, deterministic 3x3-grid fixture: alpha=1e-13 -> 1.785e-16, 1e-14 -> 3.162e-01,
      1e-15 -> 3.162e-01, 1e-16 -> RuntimeError. Candidate, random 8x3: 1e-13/1e-14/1e-15 ->
      1.433e+00. Prior candidate 4caa4f16, both fixtures, alpha from 1e20 to 1e-17: all
      <= 4.744e-15. The result is not a sign flip or permutation but a different orthogonal
      frame chosen as canonical, as a function of the units the coordinates are expressed in.
      Right-orthogonal equivariance is NOT affected (<= 1.6e-15 at every scale in both trees) -
      the output remains an isometry and remains basis independent.

      (b) ZERO-RESIDUAL ANCHOR -> NaN -> BARE StopIteration. When a tied row has exactly zero
      residual, `residual / torch.linalg.vector_norm(residual)` at :854 yields a NaN basis
      vector; on the next step every residual norm is NaN, no row satisfies the predicate, and
      the generator in `next(...)` at :850-852 is exhausted, raising `StopIteration` with an
      empty message. Minimal deterministic reproducer, executed read-only against the FROZEN
      tree (popgp.__file__ = C:\src\POPGP-review-source-law-rereview-5\popgp\__init__.py):
      `Simulator._canonicalize_embedding(1e-15 * torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,1.,0.],
      [0.,0.,1.],[-2.,-1.,-1.]], dtype=torch.float64))` -> `StopIteration ''`. The same matrix at
      scale 1.0 and 1e-6 returns correctly; the prior candidate returns exact,
      distance-preserving output at all four scales. Tiny-scale sweep: candidate 10
      StopIteration in 4000 trials, prior candidate 0.
      Under PEP 479 an enclosing generator converts this to
      `RuntimeError: generator raised StopIteration` - loud, but still undiagnostic.
    finding: |-
      The absolute floor in `tie_tolerance` silently converts the maximum-volume anchor scan into
      an unconditional "take the lowest-labelled remaining row" rule for any embedding whose
      residual norms fall below 5.684e-14, which (a) breaks positive homogeneity of the canonical
      frame - canon(alpha*C) is not alpha*canon(C) below alpha ~ 1e-13 - and (b) allows a row
      with exactly zero residual to be selected, producing a NaN basis vector and an unlabelled
      StopIteration escaping from library internals.
    failure_scenario: |-
      `Simulator._canonicalize_embedding(M)` with
      M = 1e-15 * [[1,0,0],[1,0,0],[0,1,0],[0,0,1],[-2,-1,-1]] (float64). The rank test at
      :809-812 reports rank 3 of 3, the guard at :847 passes because 4.4e-15 > 2.22e-16, and the
      scan then selects the duplicate row whose residual is exactly zero, raising `StopIteration`
      with args=() and an empty string. The identical matrix scaled by 1e9 succeeds.
    consequence: |-
      No shipped artifact, gate or scientific claim changes: the examples run at coordinate scale
      O(1) where the tolerance is relative, all 18 artifacts regenerate byte-identically, and
      pairwise geometry and right-orthogonal equivariance hold at every scale tested. The defect
      is that the routine's determinism becomes unit-dependent in a corner, and that the failure
      surfaces as an exception carrying no diagnostic text at all.
    required_action: |-
      Drop the clamp: use `tie_tolerance = 256.0 * epsilon * maximum` at :849, which is safe
      because `maximum` is already known nonzero after the guard at :847, so the tie window scales
      with the residual it compares. Additionally assert at :854 that the selected residual norm
      is strictly positive and raise a descriptive error rather than dividing by zero. Add the
      regression specified in TST-MDS-007.
    verification: confirmed-by-execution
    blocking: false

  - id: "MDS-008"
    severity: low
    category: claim
    location: "popgp/simulator.py:796 (docstring summary line)"
    evidence: |-
      Read verbatim at the frozen candidate, :796 is
      "Fix the arbitrary orthogonal MDS frame using label-ordered anchors." That described the
      implementation at 4caa4f16, which took "the first linearly independent, label-ordered
      coordinate rows". It does not describe the implementation here: :818 calls
      `_select_embedding_anchors`, whose own docstring at :830 says "Select a stable
      maximum-volume row basis with label-ordered ties" and whose body at :837-855 performs a
      greedy maximum-volume scan in which label order is only the tie-break.
      `git show 21baacd -- popgp/simulator.py` shows the commit rewrote :799-801 and the entire
      body but left :796 untouched, so this staleness was introduced by this remediation.
      Mitigating and stated for balance: the corrected description appears two lines below at
      :799, so a reader of the whole docstring is not deceived, and the neighbouring claim at
      :800 that "the SVD polar factor is exactly orthogonal to floating-point precision" is
      accurate - measured max|Q^T Q - I| over 2055 constructed configurations spanning n in 4..12,
      D in 2..6 and condition numbers 1e0..1e15 is 2.220446e-15, i.e. 10 machine epsilon.
    finding: |-
      The one-line docstring summary of `_canonicalize_embedding` still describes the algorithm
      this commit replaced.
    failure_scenario: |-
      A maintainer reading only the summary line of `_canonicalize_embedding` believes anchors are
      selected in label order and reasons about determinism, tie behaviour or the REG-001 fix on
      that basis; the actual selection is by maximum incremental volume with label order used only
      to break ties within `tie_tolerance`.
    consequence: |-
      Documentation only. No number, gate, test or artifact is affected. Recorded because the
      round-4 required action for MDS-005 was explicitly "Either fix the docstring at :796-801 or
      make the code meet it", and the code was changed under a summary line that no longer matches.
    required_action: |-
      Change :796 to say "maximum-volume anchors with label-ordered ties". Add the assertion
      specified in TST-MDS-008.
    verification: confirmed-by-execution
    blocking: false

  - id: "CACHE-002"
    severity: low
    category: code
    location: "popgp/backend.py:151-154 (cache fields), :194-195 and :207-208 (interaction-term cache), :240-242 and :253-255 (cell-Hamiltonian cache), against the plain mutable `@dataclass SubstrateConfig` at popgp/config.py:50"
    evidence: |-
      `_interaction_terms` is keyed on NOTHING and `_cell_hamiltonians` only on the cell indices,
      but both depend on `config.substrate.hamiltonian`, `coupling_J` and, through
      `build_edges()`, on `boundary`, `topology`, the grid dimensions and `n_qubits`.
      `SubstrateConfig` is a plain `@dataclass`, so every field is assignable, and the repository
      does mutate config objects elsewhere (for example `cfg.pi_res.cell_dim = 1`).

      EXECUTED A/B, N=4 open Heisenberg chain, import path asserted on both trees. After
      `cfg.substrate.coupling_J = 5.0` following a first `build_interaction_terms()`:
        candidate  ||term0|| = 1.732051, cell[0,1] = 0.866025, cell[2,3] = 0.866025,
                   build_hamiltonian norm = 3.000000
        prior      8.660254 / 4.330127 / 4.330127 / 15.000000  (all correct, matching a freshly
                   constructed reference backend)
      After `hamiltonian = 'ising'`: candidate ||term0|| stays 1.732051, prior returns 1.000000.
      After `boundary = 'periodic'`: candidate n_terms stays 3 while `build_edges()` returns 4.
      Note `cell[2,3]` had NEVER been cached before the mutation and is still stale, because it
      is rebuilt from the stale interaction terms - so the widening is real and is not merely the
      pre-existing `_H` cache resurfacing.

      COUNTERWEIGHT, established by adversarial verification and adopted here rather than buried.
      (i) The hazard CLASS is pre-existing: `build_hamiltonian` has returned the cached `self._H`
      by reference since the baseline c03800e4, and executed on all three trees a
      post-construction `coupling_J` mutation leaves `build_hamiltonian` and `prepare_state`
      stale identically. This remediation extends an existing convention to two more accessors;
      it does not create a new class of hazard. (ii) For the ordering every real pipeline uses -
      `prepare_state()` first, then a config change, then `build_local_energy_operators()` - the
      PRIOR behaviour was worse: prior `max|sum(terms) - H_state| = 2.0`, candidate `0.0`. Pre-fix,
      a mutated-config sweep silently produced a local-energy decomposition that did not sum to
      the Hamiltonian that generated the state, breaking the exact invariant the source-law
      diagnostics rest on. Post-fix everything is coherently frozen at first construction. The
      delta is not "correct -> silently wrong"; it is "silently incoherent -> internally
      consistent but stale". (iii) No caller in popgp/, examples/, scripts/ or tests/ mutates a
      substrate field after a backend exists; I re-read all 22 non-backend call sites.
    finding: |-
      The two new caches carry no invalidation key for the substrate fields they depend on, so
      mutating a `SubstrateConfig` field after the first call silently returns operators for the
      old configuration - including for cells never previously built. The staleness class already
      existed for `build_hamiltonian`; this remediation widens it to the interaction-term
      decomposition that the LIB-001 fix made every other construction derive from.
    failure_scenario: |-
      `backend = ExactBackend(cfg); terms = backend.build_interaction_terms();
      cfg.substrate.coupling_J = 5.0; backend.build_interaction_terms()` returns
      ||term0|| = 1.732051 at the candidate where 8.660254 is correct, and a subsequent
      `backend.build_cell_hamiltonian([2,3])` - a cell never built before - returns 0.866025 where
      4.330127 is correct. The prior candidate returns the correct value in both cases. Note this
      also defeats the `coupling_J != backend.config.substrate.coupling_J` precondition at
      popgp/coarse_grain.py:139, which reads the MUTATED config and therefore passes while the
      cached generator still encodes the old coupling.
    consequence: |-
      Latent, with no active reach: nothing in the repository mutates a substrate field after
      constructing a backend, every committed number is unaffected, and all 18 artifacts
      regenerate byte-identically. It matters because sweeps over `coupling_J` or Hamiltonian
      family are exactly the code the source-law experiments invite, and the failure is silent.
    required_action: |-
      Key the caches on a signature tuple of the substrate fields they depend on (hamiltonian,
      coupling_J, boundary, topology, grid dims, n_qubits) compared at :194 and :240; or declare
      `SubstrateConfig` frozen so the staleness is structurally unreachable; or document at
      popgp/backend.py:145-159 that the backend snapshots the substrate at construction and configs
      must not be mutated afterwards. Do NOT "fix" this by returning defensive copies: the
      by-reference contract is deliberate and pinned by tests/unit/test_backend.py:114-121, and
      changing :195/:208 to `return list(self._interaction_terms)` fails that test at :120.
      Add the regression specified in TST-CACHE-002.
    verification: confirmed-by-execution
    blocking: false

  - id: "CACHE-004"
    severity: low
    category: code
    location: "popgp/coarse_grain.py:111 (new public keyword), :141-148 (cache layer), :443 and :469 (plumbing through optimize_cells)"
    evidence: |-
      `ExactBackend._cell_hamiltonians` (popgp/backend.py:240-242) already memoises on the
      identical key, `tuple(cell_indices)`, so the `cell_hamiltonian_cache` layer threaded through
      `compute_leakage`'s public signature is redundant for the only backend that implements the
      method. Measured single-threaded CPU time, N=8 / cell_dim=2 / phase_window_samples=5,
      min of 2 reps: prior (no caches) 51.984 s; candidate (both caches) 16.938 s; candidate with
      ONLY the coarse_grain layer removed 16.984 s; candidate with ONLY the backend caches removed
      19.047 s. The backend caches alone recover 3.06x; the extra layer buys 0.046 s, i.e. 0.3%,
      inside the run-to-run spread. At N=6 the layer is not measurably positive at all.
      The parameter is public and is not documented in the docstring at
      popgp/coarse_grain.py:126-133, and the cache key carries no backend identity.
      For balance: the backend cell-Hamiltonian cache IS independently pinned - removing
      popgp/backend.py:241-242 fails tests/unit/test_backend.py:121 - so there is no coverage gap
      here, only redundant surface.
    finding: |-
      `compute_leakage` gained a second cache layer, exposed on its public signature and plumbed
      through `optimize_cells`, that duplicates the backend-level memoisation for a measured 0.3%
      and adds an undocumented, backend-unkeyed correctness surface.
    failure_scenario: |-
      A caller invokes the public `compute_leakage(..., cell_hamiltonian_cache=shared)` twice with
      backends built from different `SimulatorConfig`s - for example a Heisenberg and an Ising
      chain over the same sites - and the second call silently reuses the first backend's
      generators. Nothing in the repository does this: `optimize_cells` at
      popgp/coarse_grain.py:443 creates a fresh dict per call.
    consequence: |-
      Unnecessary public API surface and a second staleness path for no measurable gain. No
      current caller is affected and no number changes.
    required_action: |-
      Drop `cell_hamiltonian_cache` from `compute_leakage` and `optimize_cells` and rely on
      `ExactBackend._cell_hamiltonians`; or, if it is kept, document the parameter's
      backend-scoping requirement in the docstring at popgp/coarse_grain.py:126-133 and validate
      it. Add the regression specified in TST-CACHE-004.
    verification: confirmed-by-execution
    blocking: false

  - id: "BACKEND-501"
    severity: low
    category: code
    location: "popgp/backend.py:207-208 (`self._interaction_terms = terms`), with the class docstring at popgp/backend.py:139"
    evidence: |-
      Before this round the interaction-term list was transient and freed as soon as
      `build_hamiltonian` / `build_local_energy_operators` / `build_cell_hamiltonian` finished
      with it; the steady-state footprint was one dense Hamiltonian. It is now |E|+1 dense
      2^N x 2^N complex128 tensors retained for the lifetime of the backend. At the documented
      supported size N=12 (`ExactBackend` docstring: "Full density-matrix backend for N <= 12
      qubits") one operator is 4096^2 * 16 B = 268.4 MiB, so an 11-edge open chain retains about
      2.95 GiB where it previously retained about 268 MiB.
      Measured and stated for balance: PEAK usage is unchanged - the prior code already held all
      |E| terms live inside `build_hamiltonian`. K32GetProcessMemoryInfo on one ExactBackend at
      N=11 with `prepare_state()`: candidate working-set delta 1050.8 MiB / peak 1266.4 MiB
      versus prior 1058.3 MiB / peak 1273.3 MiB. Only the RETENTION is new, and the OOM scenario
      as originally drafted was refuted by that measurement.
    finding: |-
      `ExactBackend` now retains the full interaction-term decomposition for its lifetime rather
      than transiently, raising the steady-state footprint from one dense Hamiltonian to |E|+1 at
      a class whose docstring advertises N <= 12.
    failure_scenario: |-
      `ExactBackend(SimulatorConfig.for_chain(n=12))` followed by `build_hamiltonian()` retains
      about 3.2 GiB after the call returns, where the same sequence at 4caa4f16 retained about
      268 MiB. A caller holding two such backends alive - a Heisenberg/Ising comparison, which is
      exactly what tests/unit/test_backend.py parametrizes - doubles that.
    consequence: |-
      Resource regression at the documented supported system size, with no in-repo impact: the
      largest `ExactBackend` actually used is N=9 (3x3 grid, 12 edges, ~50 MiB), no committed run
      is affected and no number changes. It is a property of the public class rather than of any
      committed configuration.
    required_action: |-
      Drop the interaction-term list once `_H` is cached and rebuild on demand for the rarer
      decomposition consumers, or bound/opt-in the cache, or document the |E|-fold retention on
      `ExactBackend` and adjust the advertised N limit. Any fix must not reintroduce the
      per-partition rebuild cost REG-003 removed - the per-cell cache at popgp/backend.py:240-254
      already delivers that benefit independently. Add the regression specified in TST-BACKEND-501.
    verification: confirmed-by-execution
    blocking: false

  - id: "GEOM-501"
    severity: low
    category: science
    location: "popgp/simulator.py:891 (the `_mds_stress` degenerate-denominator short circuit)"
    evidence: |-
      Kruskal stress-1 is dimensionless and scale invariant, but the implementation returns
      exactly 0.0 - the value meaning "perfect embedding" - whenever the sum of squared target
      distances falls below an ABSOLUTE 1e-15. Stress feeds both the dimension objective
      `F_D = stress + lambda_dim * (D - D_S)^2` and `embedding_status` at
      popgp/simulator.py:651-656, so a purely conventional rescaling of the distance kernel can
      turn a poor embedding into a zero-stress "geometric_candidate".
      PRE-EXISTING, NOT A REGRESSION: this line is not in the round-4-to-round-5 diff and behaves
      identically at 4caa4f16 and at the baseline. Recorded as a fresh-eyes finding.
      No committed configuration is close: the committed grid_2d configuration
      (`cfg.pi_loc.I_0 = 1.0` at examples/physics_qg/grid_2d/__main__.py:37) gives
      sum d^2 = 309.289526 and stress 1.681725e-01, matching the committed
      `"stress": 0.1681...` - i.e. 3.1e17 above the trip point.
    finding: |-
      A scale-invariant scientific goodness-of-fit statistic has a scale-dependent silent
      "perfect fit" fallback that propagates into the headline `embedding_status` declaration.
      The degenerate branch signals perfection where it should signal degeneracy.
    failure_scenario: |-
      A distance kernel whose outputs are uniformly small enough that
      sum over pairs of d_target^2 < 1e-15 makes `_mds_stress` return 0.0 for every candidate
      dimension. The objective is then minimised by the spectral penalty alone, and
      `embedding_status` at :655 is set to "geometric_candidate" with stress 0.0 - a maximally
      confident geometric declaration produced by a numerical guard rather than by any fit.
    consequence: |-
      No current number is wrong and no committed configuration approaches the threshold. The
      defect is that the one statistic which decides whether the pipeline declares an emergent
      geometry can silently report its most favourable possible value on a degenerate input, in a
      codebase whose stated purpose is falsification.
    required_action: |-
      Make the guard relative or make it loud: trip on
      `denom <= n_pairs * eps * max(d_target)**2` rather than on an absolute 1e-15, or return NaN
      with a diagnostic on numerically degenerate targets and propagate that into
      `embedding_status` as inadmissible rather than as `geometric_candidate`. Add the regression
      specified in TST-GEOM-501.
    verification: confirmed-by-execution
    blocking: false

  - id: "GOV-013"
    severity: low
    category: governance
    location: "docs/scientific_hardening/GATE_TEST_REGISTRY.md:3-4 (the completeness claim) against tests/unit/test_review_guidance.py:128-176 (which reads only the two markdown documents)"
    evidence: |-
      GATE_TEST_REGISTRY.md:3-4 asserts "Every active acceptance gate and every gate retaining a
      deliberate negative result has a stable ID and at least one executable negative control."
      The strengthened meta-test binds the registry table to the falsification-matrix table and
      to the cited test nodes, but nothing binds either document to the acceptance gates actually
      executing in example code.

      EXECUTED END TO END BY ME, reproducing round 4's probe (e) at this candidate. I added a new
      numeric acceptance check `relative_entropy_fit_quality_floor`
      (`"passed": fits["relative_entropy"].r_squared > 0.999`) to
      examples/physics_qg/source_law/__main__.py, feeding `overall_pass` through the unchanged
      `all(check["passed"] for check in report["checks"])` at :244. It has no GATE ID, no matrix
      row, no registry entry and no negative control. I regenerated validation.json (which now
      lists five checks including the new one, `"overall_pass": true`), committed BOTH files in my
      disposable copy, and ran the entire authoritative command set:
        ruff "All checks passed!" exit 0; check_tex.py exit 0; pytest -q "142 passed";
        all six examples exit 0; check_validation_artifacts.py
        "Validation artifact contracts and required visual outputs are valid." exit 0;
        pytest -q tests/unit/test_review_guidance.py "4 passed".
      Nothing objects. Scale of the gap: 35 non-informational checks across the six committed
      examples feed `overall_pass`, against 6 registered gate IDs, with no mechanical link.
    finding: |-
      This is the un-closed half of round-4 finding GOV-006, whose text named both routes: "A new
      acceptance gate can be added to a falsification-matrix row OR DIRECTLY TO EXAMPLE CODE with
      no registry entry and no control, and nothing objects." The matrix-row route is now closed
      and mechanically enforced. The example-code route is not, so the registry's repository-wide
      completeness claim at :3-4 can silently become false while CI stays green.
    failure_scenario: |-
      A contributor adds an acceptance check to any `examples/physics_qg/*/__main__.py` checks
      list, regenerates the artifact, and commits both. The gate now gates `overall_pass` and is
      reported in a published validation document, while GATE_TEST_REGISTRY.md continues to
      assert that every active gate has an executable negative control, and
      `test_every_declared_gate_has_an_executable_negative_control` passes. Executed above:
      the full CI is green.
    consequence: |-
      Enforcement of the project's central anti-overclaiming rule reaches the two governance
      documents but not the code that computes the gates, so it still rests on contributor
      diligence plus the independent-review step at AGENT_REVIEW_WORKFLOW.md:80-83. No gate is
      currently mis-registered and no present claim is false; this is a hardening gap against
      future drift. It is narrower than at round 4, where the matrix-row route was open too.
      Note that the NEW sentence added this round at GATE_TEST_REGISTRY.md:6-8 is accurate - it
      describes only what CI does to the table - so the overclaim is confined to the older
      sentence at :3-4.
    required_action: |-
      Either (i) bind the registry to code: assert that every check name appearing in
      examples/physics_qg/*/results/validation.json with severity != "informational" maps to a
      registered gate ID or to an explicit allow-list, ideally inside
      `check_validation_semantics` so the rule is content-intrinsic and survives commit; or
      (ii) narrow GATE_TEST_REGISTRY.md:3-4 to say that the completeness property covers gates
      DECLARED in this registry and is human-reviewed for gates implemented in example code.
      Add the regression specified in TST-GOV-013.
    verification: confirmed-by-execution
    blocking: false

  - id: "GOV-009"
    severity: low
    category: governance
    location: "tests/unit/test_review_guidance.py:98-101 (the authority loop) and :66-81 (`_quality_authority_sentences`)"
    evidence: |-
      The loop asserts `".github/workflows/ci.yml" in sentence` for every sentence containing
      "authoritative" together with "quality suite" or "pre-freeze". That verifies the ci.yml
      token APPEARS in the sentence, not that ci.yml is the thing being designated.
      EXECUTED by me on a disposable copy, appending each sentence to
      docs/governance/REVIEWER_IDENTITY.md and running the file:
        - Round-4's exact mutation, "The README Quick start is the authoritative pre-freeze
          quality suite; run `uv sync` and `uv run pytest -q` only."
          -> 1 failed. The demonstrated bypass is closed.
        - "Although .github/workflows/ci.yml exists, the authoritative quality suite is the
          README Quick start." -> 4 passed. A concessive clause naming ci.yml defeats the check
          while designating a competing suite.
        - "The README Quick start is the authoritative test suite for this project."
          -> 4 passed (misses the "quality suite" / "pre-freeze" token pair).
        - "The canonical quality suite is the README Quick start." -> 4 passed (misses
          "authoritative").
      Also read-only: of the repository's two authority declarations, only
      docs/governance/AGENT_REVIEW_WORKFLOW.md is positively pinned, by the substring assertion at
      tests/unit/test_review_guidance.py:96; the runbook's own declaration at
      docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md:40 has no positive pin.
      For balance: clause 1 of TST-GOV-003 is intact and sensitive - `_uv_commands` equality across
      ci.yml, README.md and the runbook fails when I add `--fix` to the ruff command in ci.yml,
      and README.md really does carry all 11 commands including check_validation_artifacts.py and
      the six regenerations, so there is no "shorter README subset" to be captured by.
    finding: |-
      The new authority assertion is polarity-blind and lexically narrow: it is satisfied by any
      sentence that merely mentions the ci.yml path, so a one-clause rewrite that explicitly
      demotes CI passes, and competing designations phrased with near-synonyms are invisible to
      the extractor entirely.
    failure_scenario: |-
      A future contributor appends "Although .github/workflows/ci.yml exists, the authoritative
      quality suite is the README Quick start." to any document under docs/governance/. The
      repository now designates a different suite as authoritative, and
      `test_documented_quality_commands_match_authoritative_ci` passes (executed: 4 passed).
    consequence: |-
      The governance-capture scenario the assertion was added to prevent remains reachable at the
      cost of one extra clause. No live defect: no competing designation exists in the current
      tree, the assertion does close the specific bypass round 4 demonstrated, and review
      artifacts under reviews/ are outside the glob so they cannot trip or evade it.
    required_action: |-
      Replace the substring check with a designation check: require the ci.yml token to occur
      before the trigger word in the sentence, or reject any matched sentence that also names a
      competing suite after the trigger word. Separately, add a positive substring pin for
      docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md:40 mirroring the one at line 96. Add the
      regression specified in TST-GOV-003C.
    verification: confirmed-by-execution
    blocking: false

  - id: "GOV-007"
    severity: low
    category: governance
    location: "tests/unit/test_review_guidance.py:42-50 (`_markdown_table`)"
    evidence: |-
      `start = next(index for index, line in enumerate(lines) if line.startswith(header))` takes
      only the FIRST header match, and the row loop `for line in lines[start + 2:]` breaks on the
      first line that does not start with "|". Neither condition is asserted, so a single blank
      line inside the falsification-matrix table, or a second table appended to either governance
      document, removes rows from governance coverage with no failure and no warning.
      Read-only confirmation that nothing is currently hidden: each document contains exactly one
      table (FALSIFICATION_MATRIX.md header at :7 with contiguous pipe rows :7-19,
      GATE_TEST_REGISTRY.md header at :10 with contiguous pipe rows :10-17), and my own row-wise
      parse recovers all 11 matrix rows and all 6 registry rows.
      Corrected after adversarial verification and stated here rather than in the original,
      stronger form: appending a new gated row at the END of the table does NOT bypass the guard -
      that mutation fails at :144. The bypass requires a blank line to be inserted so that
      subsequent rows fall outside the parse.
    finding: |-
      The new markdown table parser truncates silently instead of erroring, so table rows can be
      removed from governance coverage by whitespace alone, without editing the test file where
      such a change would be visible in review.
    failure_scenario: |-
      A contributor inserts a blank line into docs/scientific_hardening/FALSIFICATION_MATRIX.md
      immediately before a new ACTIVE acceptance-gate row. `_markdown_table` stops at the blank
      line, the new row is never parsed, the ungated-row assertion at :143-144 never sees it, and
      `pytest -q tests/unit/test_review_guidance.py` reports 4 passed.
    consequence: |-
      The tamper-evidence the round-4 remedy relies on - that bypassing it requires an edit to
      `UNGATED_MATRIX_ROWS`, which is visible in a test-file diff - is weakened to a whitespace
      edit in a documentation file. No live defect at this commit.
    required_action: |-
      Make the parser strict: assert exactly one line in each document starts with the header, and
      assert that every line beginning with "|" other than the header and the "|---|" separator is
      accounted for by the parsed row list. Add the regression specified in TST-GOV-007.
    verification: confirmed-by-execution
    blocking: false

  - id: "GOV-008"
    severity: low
    category: governance
    location: "tests/unit/test_review_guidance.py:53-62 (`_has_negative_control_marker`) and pyproject.toml:32"
    evidence: |-
      The check is `ast.unparse(decorator).startswith("pytest.mark.negative_control")`, a PREFIX
      match, so `@pytest.mark.negative_control_pending` or any longer name beginning with that
      string satisfies it. The marker registration at pyproject.toml:32 is unenforced: I deleted
      the `"negative_control: ..."` line from `[tool.pytest.ini_options] markers` in a disposable
      copy and ran the full suite - `142 passed, 7 warnings`, the warnings being
      PytestUnknownMarkWarning. No `--strict-markers` is configured. The guard also does not
      reject a `@pytest.mark.skip`/`skipif` decorator on a cited control.
      For balance, and after adversarial correction: node EXISTENCE is genuinely enforced -
      renaming a cited control gives StopIteration at :55 (`1 failed, 3 passed`), and moving one
      to another file does the same - so GATE_TEST_REGISTRY.md:6-8 ("requires every control node
      to exist") is literally true. At this commit all seven markers are exact, the marker set
      equals the citation set byte for byte (7 cited, 7 marked, zero spurious, zero missing, which
      I verified by parsing both), and `pytest -m negative_control` collects 7.
    finding: |-
      The machine-readable negative-control marker is checked by prefix rather than exact match,
      its registration is not enforced by `--strict-markers`, and a cited control could be skipped
      without the guard noticing - while GATE_TEST_REGISTRY.md:6-8 now advertises that CI verifies
      the marker.
    failure_scenario: |-
      A contributor decorates a cited control with `@pytest.mark.negative_control_pending`. The
      AST prefix match accepts it, the mark is unregistered so pytest emits only a
      PytestUnknownMarkWarning, `pytest -m negative_control` no longer collects that test, and
      `test_every_declared_gate_has_an_executable_negative_control` passes. Executed for the
      registration half: removing pyproject.toml:32 leaves the suite at 142 passed with 7 warnings.
    consequence: |-
      A reader of GATE_TEST_REGISTRY.md:6-8 over-trusts the mechanical guarantee. Practical
      exposure is small: nothing is currently mis-marked or skipped, and both failure modes
      surface as an "s" or a warning in ordinary pytest output.
    required_action: |-
      Compare `ast.unparse(decorator)` for exact equality with "pytest.mark.negative_control";
      add `--strict-markers` to the pytest addopts so an unregistered mark is an error; and reject
      a skip/skipif decorator on a cited control. Add the regression specified in TST-GOV-008.
    verification: confirmed-by-execution
    blocking: false

  - id: "FRAMEWORK-003"
    severity: low
    category: code
    location: "tests/unit/test_claim_wording.py:75-86 (`test_framework_locality_status_discloses_qcmi_gap_in_both_sources`)"
    evidence: |-
      The guard selects the row with
      `next(line for line in source.splitlines() if "Emergent locality" in line)` and then asserts
      four loose substrings: "Definition / partial prototype", "Pairwise", "QCMI screening is not"
      and "non-geometric controls". Two evasions follow directly and were executed:
        (a) INVERTED MEANING PASSES. A row that contains all four substrings while asserting the
            opposite - for example text stating that QCMI screening is not OPTIONAL and that
            non-geometric controls are fully handled - satisfies every assertion.
        (b) DECOY SHADOWING. `next(...)` returns the FIRST matching line, so a LaTeX comment line
            beginning with "%" that contains "Emergent locality" and the four substrings is
            selected instead of the real row, which can then be rewritten freely. Comment lines do
            not render and `scripts/check_tex.py` does not reject them.
      For balance, verified by mutation: the guard IS a real control against deletion - removing
      any one of the four substrings from either document fails it - and the underlying
      correction it guards is sound (see the prior-finding entry for the framework parity item).
    finding: |-
      The new manuscript-parity guard pins four tokens rather than a claim. Text carrying the
      opposite meaning satisfies it, and a non-rendering LaTeX comment can shadow the row it is
      supposed to check.
    failure_scenario: |-
      A contributor inserts `% Emergent locality: Definition / partial prototype. Pairwise MI ...
      QCMI screening is not required. ... non-geometric controls ...` above the real item in
      docs/framework.tex and then rewrites the real row to reassert QCMI screening as implemented.
      `next(...)` picks the comment, all four assertions pass, and the authoritative manuscript
      source now overstates implementation status again - the exact round-4 defect.
    consequence: |-
      The regression the builder added for the round-4 manuscript observation does not pin the
      claim it names. The claim is correct in the tree today; what is missing is a check that can
      keep it correct. docs/scientific_hardening/REPRODUCIBILITY.md:20 designates
      docs/framework.tex the authoritative manuscript source, so this is the document that matters.
    required_action: |-
      In tests/unit/test_claim_wording.py, skip LaTeX comment lines
      (`lstrip().startswith("%")`), assert that exactly one candidate row matches rather than
      taking `next(...)`, and replace the four loose substrings with a normalized whole-clause
      comparison so inverted phrasing cannot satisfy it. Add the regression specified in
      TST-FRAMEWORK-003.
    verification: confirmed-by-execution
    blocking: false

  - id: "FRAMEWORK-004"
    severity: low
    category: science
    location: "docs/scientific_hardening/FALSIFICATION_MATRIX.md:13 (Non-geometric controls readiness cell) and tests/scientific/test_topology_recovery.py::test_disjoint_bell_pairs_are_marked_nonseparable"
    evidence: |-
      The row's failure threshold is "False low-D selection or 'separable' edge claim across
      perturbations", and its readiness cell mitigates the admitted false D*=1 by stating
      "Uniform and disjoint-Bell controls are non-separable". Executed: the disjoint-Bell half of
      that mitigation holds only at exact algebraic MI degeneracy. Evolving the 4-qubit
      disjoint-Bell state under the framework's own Hamiltonian for dt = 0.05 breaks the
      degeneracy, and the pipeline then emits BOTH halves of the stated failure threshold at once:
      `connectivity_separable` becomes True with `connectivity_gap_ratio` about 1.03e5 (against
      `minimum_gap_ratio` 1.5), while `D_star` remains 1 with `embedding_status`
      "geometric_candidate".
      Corrected after adversarial verification and stated in the narrower, accurate form: the
      UNIFORM half of the readiness clause is robust - perturbing the 6-node uniform weight
      fixture by eps in {1e-6, 1e-3, 1e-2} across seeds {0,1,2} keeps separable False with
      gap_ratio 1.0000-1.0426, far below 1.5. Only the disjoint-Bell half is knife-edge. The
      committed clause is literally true of both committed control STATES; what is not true is
      that it survives the perturbation the row's own threshold demands.
    finding: |-
      The mitigation recorded against the failing non-geometric-controls gate is a knife-edge
      artifact of exact mutual-information degeneracy for the disjoint-Bell control. Under a
      generic perturbation the control trips the row's stated failure threshold in full - both a
      "separable" edge claim and a false low-dimensional geometric declaration - and the test name
      `test_disjoint_bell_pairs_are_marked_nonseparable` advertises a robustness property that
      survives only at that degeneracy.
    failure_scenario: |-
      A reader of FALSIFICATION_MATRIX.md:13 concludes that although the disjoint-Bell control
      receives a false D*=1, the pipeline still correctly refuses to call it separable, so the
      failure is partial. Build the 4-qubit disjoint-Bell state, evolve for dt = 0.05, and the
      pipeline reports separable True at gap_ratio 1.03e5 AND D_star 1 with embedding_status
      "geometric_candidate" - the complete stated failure condition.
    consequence: |-
      The gate row understates how badly GATE-NONGEOMETRIC-CONTROLS currently fails. This is not a
      newly discovered pipeline defect - the false D*=1 is already openly admitted and pinned as a
      permanent regression, which remains to the project's credit - and no committed number
      changes. What is inaccurate is the recorded readiness, on a row that is one of the six
      registered falsification gates.
    required_action: |-
      Amend docs/scientific_hardening/FALSIFICATION_MATRIX.md:13 to state that non-separability of
      the disjoint-Bell control holds only for the exact degenerate state and fails under
      perturbation, citing the measured gap_ratio at dt = 0.05, and extend
      tests/scientific/test_topology_recovery.py with a perturbed variant so the recorded
      readiness matches the row's own "across perturbations" threshold. Add the regression
      specified in TST-FRAMEWORK-004.
    verification: confirmed-by-execution
    blocking: false

  - id: "TSTGUARD-001"
    severity: low
    category: code
    location: "tests/unit/test_review_guidance.py:185-191 (`subprocess.run([...], check=True, capture_output=True)`)"
    evidence: |-
      The nested collection runs with `check=True` and `capture_output=True`, so a non-zero exit
      raises `CalledProcessError` whose message is an exit status; the captured stdout and stderr
      that contain the actual collection error are discarded from the failure report. Executed by
      adding a module that raises on import to a disposable copy: running the node in isolation
      reports a `CalledProcessError` attributed to subprocess.py rather than naming the offending
      module. There is also no `timeout=` argument and no pytest-timeout plugin installed, so a
      hung nested collection has no independent deadline.
    finding: |-
      The new reproducibility guard discards the child process's diagnostic output on failure,
      reporting an exit status instead of the collection error that caused it.
    failure_scenario: |-
      A contributor introduces an import error in any test module and runs
      `pytest tests/unit/test_review_guidance.py` (or just the guard's node id). The guard fails
      with a bare `CalledProcessError` and no indication of which module is broken.
    consequence: |-
      Diagnostic noise only. The guard still fails - it never falsely passes - and in a full-suite
      run the real collection error surfaces first from the parent process. No gate is bypassable.
    required_action: |-
      Drop `check=True` and assert `collected.returncode == 0` with `collected.stdout` and
      `collected.stderr` interpolated into the assertion message; optionally add a `timeout=`.
      Add the regression specified in TST-TSTGUARD-001.
    verification: confirmed-by-execution
    blocking: false

  - id: "TSTGUARD-002"
    severity: low
    category: claim
    location: "tests/unit/test_review_guidance.py:179-194 against docs/scientific_hardening/REPRODUCIBILITY.md:45"
    evidence: |-
      REPRODUCIBILITY.md:45 records "| `pytest -q` | 5 s | 142 passed |" - a PASSED count - while
      the guard compares it against the COLLECTED count from `pytest --collect-only -q`. The two
      coincide only while nothing is skipped, xfailed, xpassed or errored. Executed on a
      disposable copy: adding a `@pytest.mark.skip` test and setting the row to 143 leaves the
      guard passing while the document's "143 passed" is false.
      For balance, the guard IS a real control on the quantity it does measure: changing 142 to
      141 fails it (`1 failed, 3 passed`), and adding a test fails it. Today collected == passed
      == 142, so the committed figure is correct. Note also that the naive remedy of rewriting the
      row to "142 collected" BREAKS the guard, because the regex at :180 requires the literal
      " passed |".
    finding: |-
      The guard protects a documented PASS count using a COLLECTED count, so any future skip or
      xfail makes the reproducibility record stale without failing the test whose entire purpose
      is to stop that figure drifting.
    failure_scenario: |-
      A contributor adds `@pytest.mark.skip` to one test and updates REPRODUCIBILITY.md:45 to
      "143 passed". Collection reports 143, the guard passes, and the audit record states 143
      passed where 142 passed and 1 skipped is the truth.
    consequence: |-
      Latent only: the shipped suite has zero skips and zero xfails, so the committed 142 is
      correct and REPRO-002 is genuinely fixed. The hole is in the guard's semantics.
    required_action: |-
      Additionally assert that no collected item carries a skip/xfail marker, so that collected
      equals passed by construction; or parse the terminal summary of a real run. Do not simply
      reword the document to "142 collected" - that breaks the regex at :180. Add the regression
      specified in TST-TSTGUARD-002.
    verification: confirmed-by-execution
    blocking: false

requested_tests:
  - id: "TST-MDS-006"
    description: |-
      Add a scale-covariance regression for `Simulator._canonicalize_embedding` and
      `Simulator._classical_mds`. (a) For the 6-node open-chain hop metric `d`, assert
      `Simulator._classical_mds(d * s, D)` returns finite coordinates for every
      s in {1, 1e-4, 1e-8, 1e-10, 1e-11, 1e-13, 1e-15} and every D in 1..5. (b) For an 8x3
      configuration with singular values (s0, s0*sqrt(r), s0*r) with s0 in {1e-8, 1e-9, 1e-10} and
      r in {1e-8, 1.5e-8, 1e-7}, assert the same and assert the maximum relative pairwise-distance
      change is below 1e-14. The test must FAIL at 3974bbfa, where s = 1e-10, 1e-11 and 1e-13
      raise `RuntimeError: embedding rank and anchor selection disagree`, and PASS at
      4caa4f16b407a47a9531442751ccde1e446b8271.
    rationale: |-
      MDS-006. The rank test at popgp/simulator.py:809 is relative while the anchor guard at :847
      is absolute below unit scale, so the two disagree and the code raises on uniformly rescaled
      inputs. 182 of 4000 randomized matrices and 3 of 16 uniform rescalings of a supported
      topology's hop metric raise at the candidate; the prior candidate raises on none of them.
    blocking: false
  - id: "TST-MDS-007"
    description: |-
      Add two assertions. (a) POSITIVE HOMOGENEITY: for a seeded random 8x3 centred fixture and
      the 3x3 integer grid fixture, and for alpha in {1e20, 1e10, 1e-5, 1e-10, 1e-13, 1e-14,
      1e-15}, assert
      `max|_canonicalize_embedding(alpha*C) - alpha*_canonicalize_embedding(C)|
      <= 1e-12 * alpha * max|_canonicalize_embedding(C)|`. (b) NO ZERO-RESIDUAL ANCHOR: assert
      `Simulator._canonicalize_embedding(s * torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,1.,0.],
      [0.,0.,1.],[-2.,-1.,-1.]], dtype=torch.float64))` returns finite, distance-preserving
      coordinates for s in {1.0, 1e-6, 1e-15}. Both must FAIL at 3974bbfa - (a) at 3.162e-01 for
      alpha = 1e-14 on the grid fixture and 1.433e+00 on the random fixture, (b) with a bare
      `StopIteration` at s = 1e-15 - and PASS at 4caa4f16, where every homogeneity deviation is
      <= 4.744e-15.
    rationale: |-
      MDS-007. `tie_tolerance = 256.0 * epsilon * max(1.0, maximum)` at popgp/simulator.py:849 has
      an absolute floor of 5.684e-14, below which the selection threshold is negative, every row
      ties, the max-volume rule degenerates to lowest-label, and a zero-residual row can be chosen.
    blocking: false
  - id: "TST-MDS-008"
    description: |-
      Correct the docstring summary at popgp/simulator.py:796 to describe maximum-volume anchor
      selection with label-ordered ties. Add an assertion in tests/unit/test_simulator.py that the
      first line of `Simulator._canonicalize_embedding.__doc__` does not contain the substring
      "label-ordered anchors", so the summary cannot silently drift out of step with the algorithm
      again. The assertion must FAIL at 3974bbfa.
    rationale: |-
      MDS-008. The fix commit rewrote :799-801 and the whole body but left the :796 summary
      describing the algorithm it replaced.
    blocking: false
  - id: "TST-CACHE-002"
    description: |-
      Add a unit test that constructs an `ExactBackend`, calls `build_interaction_terms()` and
      `build_cell_hamiltonian([0,1])`, then mutates `config.substrate.coupling_J` (and separately
      `.hamiltonian` and `.boundary`) and asserts that the recomputed values either match a freshly
      constructed backend or raise. Include `build_cell_hamiltonian([2,3])` - a cell never built
      before the mutation - which is the case that shows the widening is not merely the
      pre-existing `_H` cache. Under the current code the coupling_J case returns ||term0|| =
      1.732051 where 8.660254 is correct and cell[2,3] = 0.866025 where 4.330127 is correct. Do
      NOT implement this by returning defensive copies: that breaks the deliberate identity
      contract at tests/unit/test_backend.py:114-121.
    rationale: |-
      CACHE-002. `_interaction_terms` is keyed on nothing and `_cell_hamiltonians` only on cell
      indices, while both depend on substrate fields of a plain mutable dataclass.
    blocking: false
  - id: "TST-CACHE-004"
    description: |-
      Either remove `cell_hamiltonian_cache` from `compute_leakage` and `optimize_cells` and add a
      call-count assertion showing the backend cache alone still bounds constructions to one per
      distinct cell; or, if the parameter is kept, document its backend-scoping requirement in the
      docstring at popgp/coarse_grain.py:126-133 and add an assertion that passing a cache
      populated by a different backend raises rather than silently reusing foreign generators.
    rationale: |-
      CACHE-004. The layer duplicates `ExactBackend._cell_hamiltonians` on the identical key for a
      measured 0.3% at the shipped configuration, while adding an undocumented, backend-unkeyed
      public parameter.
    blocking: false
  - id: "TST-BACKEND-501"
    description: |-
      Add a unit test that constructs an `ExactBackend` at a size where the difference is
      unambiguous (N=10), calls `build_hamiltonian()`, and asserts that the total retained bytes
      across tensors reachable from the backend instance is bounded by a small multiple of
      2^N * 2^N * 16 B (for example <= 3x), so that whole-decomposition retention cannot be
      reintroduced silently. The assertion must FAIL at 3974bbfa, where |E|+1 dense operators are
      retained.
    rationale: |-
      BACKEND-501. `self._interaction_terms = terms` at popgp/backend.py:207 turns a transient
      allocation into lifetime retention of |E| dense 2^N x 2^N complex128 operators, about
      2.95 GiB at the documented supported size N=12.
    blocking: false
  - id: "TST-GEOM-501"
    description: |-
      Add a scientific regression asserting that `Simulator._mds_stress(d_target * c, coords * c)`
      is independent of c for c in {1, 1e-6, 1e-9, 1e-12} on a fixture with known nonzero stress,
      and a pipeline-level assertion that a globally rescaled distance kernel does not change
      `D_star` or `embedding_status` for the committed grid_2d configuration. Both must FAIL on
      the current absolute 1e-15 short circuit at popgp/simulator.py:891, which returns exactly
      0.0 - "perfect embedding" - on a degenerate target.
    rationale: |-
      GEOM-501. A dimensionless, scale-invariant goodness-of-fit statistic that decides
      `embedding_status` has a scale-dependent silent "perfect fit" fallback. Pre-existing, not a
      regression of this remediation.
    blocking: false
  - id: "TST-GOV-013"
    description: |-
      Bind the gate registry to the gates that actually execute. Add an assertion - preferably
      inside `check_validation_semantics` in scripts/check_validation_artifacts.py so the rule is
      content-intrinsic and survives commit rather than relying on a diff against HEAD - that every
      check name in examples/physics_qg/*/results/validation.json whose severity is not
      "informational" maps either to a registered GATE-* ID or to an explicit, per-path allow-list.
      Demonstrate it FAILING on the executed mutation that currently passes the entire CI green:
      add `relative_entropy_fit_quality_floor`
      (`"passed": fits["relative_entropy"].r_squared > 0.999`) to
      examples/physics_qg/source_law/__main__.py, regenerate validation.json, and commit both.
      If binding to code is judged out of scope, instead narrow GATE_TEST_REGISTRY.md:3-4 to the
      gates declared in that registry and add an assertion pinning the narrowed wording.
    rationale: |-
      GOV-013, the un-closed half of round-4 GOV-006. 35 non-informational checks feed
      `overall_pass` across the six examples against 6 registered gate IDs, with no mechanical
      link; the round-4 probe (e) reproduces exactly at this candidate.
    blocking: false
  - id: "TST-GOV-003C"
    description: |-
      Extend the authority loop in tests/unit/test_review_guidance.py so it fails on a sentence
      that names .github/workflows/ci.yml while designating a different suite as authoritative -
      for example by requiring the ci.yml token to precede the trigger word, or by rejecting a
      matched sentence that also names a competing suite after it. Demonstrate it FAILING on this
      exact executed mutation, which currently passes: append "Although .github/workflows/ci.yml
      exists, the authoritative quality suite is the README Quick start." to
      docs/governance/REVIEWER_IDENTITY.md. Separately, add a positive substring pin for the
      runbook's own authority declaration at docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md:40,
      mirroring the existing pin at tests/unit/test_review_guidance.py:96.
    rationale: |-
      GOV-009. The assertion checks that the ci.yml token appears in an authority sentence, not
      that ci.yml is what is being designated, and only one of the repository's two authority
      declarations is positively pinned.
    blocking: false
  - id: "TST-GOV-007"
    description: |-
      Add assertions to tests/unit/test_review_guidance.py that (i) each of
      docs/scientific_hardening/FALSIFICATION_MATRIX.md and GATE_TEST_REGISTRY.md contains exactly
      one line starting with its table header, and (ii) every line in those documents beginning
      with "|" other than the header and the "|---|" separator is accounted for by the parsed row
      list. Demonstrate FAILING on the executed mutation that currently passes: insert a blank line
      into the falsification-matrix table immediately before a newly added ACTIVE gate row.
    rationale: |-
      GOV-007. `_markdown_table` takes the first header match and breaks on the first non-pipe
      line, both silently, so rows can be removed from governance coverage by whitespace alone.
    blocking: false
  - id: "TST-GOV-008"
    description: |-
      Change `_has_negative_control_marker` to compare `ast.unparse(decorator)` for exact equality
      with "pytest.mark.negative_control", add `--strict-markers` to the pytest addopts, and assert
      that no cited control carries a skip/skipif decorator. Demonstrate FAILING on all three
      currently-passing mutations: (a) rename the decorator on a cited control to
      `@pytest.mark.negative_control_pending`; (b) delete the marker registration at
      pyproject.toml:32 (currently 142 passed with 7 PytestUnknownMarkWarnings); (c) add
      `@pytest.mark.skip` to a cited control.
    rationale: |-
      GOV-008. The marker check is a prefix match, its registration is unenforced, and skipping a
      cited control is undetected, while GATE_TEST_REGISTRY.md:6-8 advertises CI verification.
    blocking: false
  - id: "TST-FRAMEWORK-003"
    description: |-
      Harden
      tests/unit/test_claim_wording.py::test_framework_locality_status_discloses_qcmi_gap_in_both_sources
      to skip LaTeX comment lines (`lstrip().startswith("%")`), assert exactly one candidate row
      matches rather than taking `next(...)`, and replace the four loose substrings with a
      normalized whole-clause comparison. Re-run both evasions as mutation controls: (1) a row
      satisfying all four substrings while asserting the opposite meaning must FAIL; (2) a
      `% Emergent locality ...` decoy comment carrying the four substrings, combined with a
      rewritten real row, must FAIL.
    rationale: |-
      FRAMEWORK-003. The guard added for the round-4 manuscript observation pins four tokens
      rather than the claim, and its first-match row selection can be shadowed by non-rendering
      content in the document REPRODUCIBILITY.md:20 designates authoritative.
    blocking: false
  - id: "TST-FRAMEWORK-004"
    description: |-
      Add
      tests/scientific/test_topology_recovery.py::test_perturbed_disjoint_bell_control_becomes_separable_and_still_declares_geometry,
      marked `negative_control`: build the 4-qubit disjoint-Bell state, evolve for dt = 0.05,
      assert `connectivity_separable` is True with `connectivity_gap_ratio` above 1e4, and assert
      `D_star == 1` with `embedding_status == "geometric_candidate"`. Amend
      docs/scientific_hardening/FALSIFICATION_MATRIX.md:13 so the recorded readiness states that
      disjoint-Bell non-separability holds only at exact MI degeneracy, and register the new
      control against GATE-NONGEOMETRIC-CONTROLS.
    rationale: |-
      FRAMEWORK-004. The row's failure threshold is stated "across perturbations", and under a
      generic perturbation the disjoint-Bell control trips both halves of it while the recorded
      readiness cell says otherwise.
    blocking: false
  - id: "TST-TSTGUARD-001"
    description: |-
      Change tests/unit/test_review_guidance.py:185-191 to drop `check=True`, assert
      `collected.returncode == 0` with the child's stdout and stderr interpolated into the
      assertion message, and add a `timeout=`. Add a unit test that points the helper at a
      temporary root containing a deliberately broken test module and asserts the resulting
      failure message contains the offending module name. It must FAIL on the current
      `check=True` implementation, whose message is an exit status.
    rationale: |-
      TSTGUARD-001. The guard discards the captured collection error, so an unrelated import
      failure surfaces as an opaque `CalledProcessError` attributed to subprocess.py.
    blocking: false
  - id: "TST-TSTGUARD-002"
    description: |-
      Extend the reproducibility guard so it also fails when collected != passed - for example by
      asserting that no collected item carries a skip or xfail marker. The added assertion must
      FAIL when `@pytest.mark.skip` is applied to any one existing test and the
      REPRODUCIBILITY.md row is set to 143, a state the current guard accepts. Do not implement
      this by rewriting the row to "142 collected": the regex at
      tests/unit/test_review_guidance.py:180 requires the literal " passed |".
    rationale: |-
      TSTGUARD-002. The document records a passed count and the guard compares a collected count;
      they coincide only while nothing is skipped or xfailed.
    blocking: false

prior_finding_results:
  - finding_id: "REG-001"
    outcome: verified-resolved
    evidence: |-
      Verified by replaying the original defect, not by reading the response. IMPORT HYGIENE
      FIRST, because it invalidated my own first attempt exactly as it did round 4's: the venv
      contains `_editable_impl_popgp.pth` pointing at the frozen worktree, so a script run from
      outside a copy silently imports the FROZEN tree. My first A/B produced identical output for
      both trees and was discarded; every result below comes from a run that printed and asserted
      `os.path.abspath(popgp.__file__)` under the intended tree.

      PRIOR CANDIDATE 4caa4f16: `Simulator._classical_mds` raises RecursionError on the 5-node
      star hop metric at D=1..4, the 6-node star at D=1..5, and the 3x3 grid hop metric with the
      centre relabelled to index 0 at D=1..8; `Simulator._canonicalize_embedding` raises on
      [[1e-9,0],[1,1],[-1,0.5],[-1e-9,-1.5]] and on [[1e-12,0],[1,0],[0,1],[-1,-1]]. 19 of 19
      inputs raise. CANDIDATE 3974bbfa: all 19 return finite coordinates.

      THE FIX IS STRUCTURAL, NOT A DEPTH GUARD. The rank test moved from
      `torch.linalg.matrix_rank(centered, rtol=1e-8)` to `torch.linalg.svdvals` plus an explicit
      count (:808-812), and the label-ordered greedy scan that could return fewer than `dimension`
      anchors was replaced by `_select_embedding_anchors` (:825-857), which always returns exactly
      `dimension` rows or raises. Recursion now occurs only via
      `_canonicalize_rank_deficient_embedding` when `represented_rank < dimension` (:813), and
      that helper recurses on a matrix with exactly `represented_rank` columns, so the column
      count strictly decreases to the `represented_rank == 0` base case at :866. Termination is
      therefore provable, not empirical. Empirically confirmed anyway: 36,058 calls (2,600 random
      matrices plus 33,458 structured graph fixtures covering every relabelling of stars, cycles,
      paths, complete and bipartite graphs for n <= 6, plus six grids) under
      `sys.setrecursionlimit(61)` produced zero RecursionError and a maximum helper nesting depth
      of 3; the same inputs at the prior tree produced 1,803 RecursionErrors at depth 19.

      PUBLIC-API REACHABILITY, WHICH ROUND 4 GOT WRONG IN THE BUILDER'S FAVOUR. Round 4 recorded
      that REG-001 was "not reachable through any supported SubstrateConfig today". I swept
      `Simulator(cfg).run()` over 420 chain configurations (n=4..9 x {open, periodic} x
      beta in {0.5,1.0,2.0} x I_0_multiplier in {1+1e-6 ... 1+1e-14, e}) plus 60 grid
      configurations. The prior candidate raises RecursionError in 5 of them - for example
      n=7 open, beta=0.5, I_0_multiplier=1+1e-8. The candidate raises nothing in any of the 480.
      So the defect WAS publicly reachable and is now not, which makes the fix more valuable than
      round 4 credited.

      NO EQUIVARIANCE REGRESSION - AN IMPROVEMENT. With `torch.linalg.eigh` monkeypatched to
      rotate every exactly degenerate eigenblock by a random orthogonal matrix (6 seeds), worst
      max-coordinate deviation of `_classical_mds` across all D: 5-cycle 1.110e-15 -> 8.882e-16,
      6-cycle 1.887e-15 -> 3.275e-15, 3x3 grid 3.553e-15 -> 4.052e-15, 4x4 grid
      2.838e-12 -> 5.607e-15 (prior -> candidate). Direct right-orthogonal invariance of
      `_canonicalize_embedding` over 1500 randomized cases: worst relative coordinate deviation
      7.945e-07 -> 3.857e-15.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The fix introduces two new low-severity failure modes in the same function, filed as MDS-006
      (absolute-versus-relative tolerance mismatch raising RuntimeError) and MDS-007 (absolute tie
      tolerance breaking scale homogeneity and permitting a zero-residual anchor). Neither is
      blocking: both are strictly narrower than the defect they replace, both fail loudly and
      immediately rather than after a 1000-frame recursion, and public-API availability improved
      from 5 failures in 420 configurations to 0. A candidate finding claiming the new
      `_select_embedding_anchors` is unpinned by any test was REFUTED under adversarial
      verification: reverting the scan to the pre-fix label-order algorithm fails 2 committed tests
      with RecursionError, and degrading max-residual to min-residual selection fails 3 including
      both parametrizations of
      test_classical_mds_is_invariant_to_degenerate_eigenbasis_for_every_dimension.
  - finding_id: "REG-003"
    outcome: verified-resolved
    evidence: |-
      Verified by call counting and by mutation, not from the response. INSTRUMENTED
      `optimize_cells` at N=6, cell_dim=2, phase_window_samples=5, wrapping the bound
      `build_cell_hamiltonian` to count invocations, import path asserted on both trees:
      prior candidate 45 calls over 15 distinct cells; candidate 15 calls over 15 distinct cells -
      exactly one construction per distinct cell. Wall time 15.18 s -> 7.33 s on a loaded machine;
      on a quiet single-threaded measurement at N=8 the same comparison is 51.984 s -> 16.938 s
      (3.06x). Round 4's headline number (420 constructions over 28 distinct cells at the shipped
      chain_1d settings) is the same phenomenon at N=8.

      OUTPUT UNCHANGED. All 18 committed artifacts regenerate byte-identically at the candidate
      (blob SHA comparison of `git hash-object` against `git ls-files -s`, 18/18), and identically
      across two independent full regeneration runs, so the memoisation changed no number.

      THE COMMITTED TESTS ARE GENUINE CONTROLS. Against a hybrid tree (pre-fix `popgp/` plus
      candidate `tests/`), both
      `test_backend_reuses_interaction_and_cell_hamiltonian_caches` and
      `test_optimize_cells_constructs_each_distinct_cell_generator_once` FAIL ("2 failed, 2
      passed"). The call-count bound `len(calls) <= 15` is not vacuous: at N=6, cell_dim=2 there
      are 15 partitions of 3 cells each, so the uncached count is 45 with duplicates, and the
      companion assertion `len(calls) == len(set(calls))` fails first. Removing only
      popgp/backend.py:241-242 fails tests/unit/test_backend.py:121, so the backend cache is
      independently pinned as well.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Three low-severity consequences of the caching are filed separately: CACHE-002 (no
      invalidation key for substrate fields), CACHE-004 (the redundant compute_leakage layer), and
      BACKEND-501 (lifetime retention of |E| dense operators). Two candidate findings were REFUTED
      and are excluded: an aliasing finding I drafted myself (returning cached objects by
      reference) was refuted because `build_hamiltonian` has returned `self._H` by reference since
      the baseline c03800e4 - verified by executing the identical poison against all three trees -
      and because the by-reference contract is deliberately pinned by
      tests/unit/test_backend.py:114-121; and a claim that the round-4 runtime figures are now
      stale was refuted by direct re-measurement.
  - finding_id: "MDS-005"
    outcome: verified-resolved
    evidence: |-
      Verified by measuring the property, not by reading the diff. The ill-conditioned construction
      `anchor_matrix.T @ (A A^T)^{-1/2}` is replaced at popgp/simulator.py:819-822 by the SVD
      polar factor `U @ V^H` of `anchor_matrix.T`, which is orthogonal to machine precision
      independently of conditioning.

      ISOMETRY, EXECUTED. Uniform rescaling of the 6-node open-chain hop metric, maximum relative
      pairwise-distance error against the target, prior -> candidate: the prior candidate ranges up
      to 1.185e-08 (at scale 1e-8) with values of 3.959e-09, 3.318e-09, 2.679e-09, 7.623e-10 and
      1.082e-09 at other scales; the candidate is <= 2.498e-15 at every scale it accepts.
      Independently, worst right-orthogonal coordinate deviation over 1500 randomized matrices:
      7.945e-07 -> 3.857e-15. Orthogonality of the returned factor, measured over 2055 constructed
      configurations spanning n in 4..12, D in 2..6 and condition numbers 1e0..1e15:
      max|Q^T Q - I| = 2.220446e-15, so the docstring claim at :800 is accurate.

      THE COMMITTED TEST IS A GENUINE CONTROL. Against the hybrid tree (pre-fix `popgp/`, candidate
      `tests/`), `test_mds_canonical_frame_is_isometric_near_rank_threshold` fails in 12 of its 16
      parametrizations, including all four padded rank-deficient cases at delta in
      {1e-7, 3e-8, 1.5e-8, 1.1e-8}; the 4 that pass are the large-delta cases where the polar
      factor was well conditioned anyway. It passes at the candidate.

      PUBLISHED CONSEQUENCE, IN THE HONEST DIRECTION. chain_1d
      stress_by_dimension["2"] 6.639527346370152e-10 -> 5.034432205558175e-16 and ["3"]
      7.121160675246565e-11 -> 5.41134677012244e-16, i.e. the canonicalization noise round 4
      identified as 1.3e6x the true value is gone and the reported stress is now at machine
      precision. stress_span 6.639522486020637e-10 -> 5.5099725543532064e-17 against an unchanged
      `selection_driver_tolerance` of 1.32e-08, so the chain_1d selection_driver gate went from
      about 20x to about 2.4e8x headroom while remaining "spectral_penalty".
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Independently verified that the geometry change is a pure rigid motion and moves no gate.
      chain_1d coords are bit-identical (orthogonal Procrustes residual 0.0). grid_2d coords are
      related by a PROPER rotation: Procrustes residual 1.776e-15 against a coordinate scale of
      2.5487, det(R) = +1.000000, maximum relative pairwise-distance change 4.382e-16. Across all
      three changed artifacts, zero `passed` booleans, zero `overall_pass` values, zero `criterion`
      strings, zero `D_star`, `embedding_status` or check names changed; gravity_well moved only
      metric-diagnostic residuals and condition numbers at <= 7e-13 relative. The residual
      docstring staleness at :796 is filed as MDS-008; the two new tolerance defects introduced by
      the companion REG-001 fix are MDS-006 and MDS-007.
  - finding_id: "GOV-006"
    outcome: verified-resolved
    evidence: |-
      Round 4's required action offered two branches and enumerated branch (i) precisely: parse the
      registry table by column; require every row to cite at least one node; mark negative-control
      tests machine-readably ("for example a required @pytest.mark.negative_control") and assert
      cited nodes carry it; put the gate ID inside each active or explicitly-negative
      FALSIFICATION_MATRIX row and derive matrix_ids from rows rather than from the prose
      paragraph. Every one of those items is implemented at tests/unit/test_review_guidance.py:42-50,
      :53-62, :128-176 and pyproject.toml:32, and the IDs now live in FALSIFICATION_MATRIX rows.

      ALL FOUR MUTATIONS ROUND 4 DEMANDED NOW FAIL. Executed by me on a disposable copy, each
      applied and reverted, baseline 4 passed before and after:
        (m1) GATE-BLIND-TOPOLOGY control cell -> "(to be added)"    -> 1 failed, 3 passed
        (m2) every citation -> "control pending" (8 replacements)   -> 1 failed, 3 passed
        (m3) all rows repointed at the unmarked identity regression
             test_adaptive_gap_is_permutation_equivariant           -> 1 failed, 3 passed
        (m4) new ACTIVE gate row with no GATE ID                    -> 1 failed, 3 passed
      Five further mutations I added also fail: (m8) delete a registry row; (m11) rename a cited
      control; (m12) move a cited control to another file; (m13) corrupt a registry "Matrix row"
      cell; (m14) malformed row with the wrong column count.

      THE BINDING IS COMPLETE AND CLEAN, verified by parsing both documents myself: 11 matrix rows,
      6 carrying exactly one in-row GATE-* ID, 5 ungated and all 5 present in `UNGATED_MATRIX_ROWS`
      with no dead allow-list entries and none unlisted; 6 registry rows, all with existing cited
      nodes; 7 distinct cited controls and exactly 7 marked functions, an exact bijection with zero
      spurious and zero missing markers; `pytest -m negative_control` collects 7. I read all seven
      controls and they are genuine falsifiers.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      DISAGREEMENT DISCLOSED RATHER THAN BURIED. One reviewing pass in this session assigned
      `unresolved`, on two grounds that I independently reproduced and accept as facts: (1) adding
      one `@pytest.mark.negative_control` line to the very identity regression round 4 named makes
      all six gates cite it and the guard still passes (executed: 4 passed); (2) round-4 probe (e)
      reproduces - an unregistered acceptance gate in example code ships through the whole CI green.
      I nonetheless record verified-resolved, and the reason is scoping rather than charity. Ground
      (1) is a limitation of the mechanism round 4 itself prescribed by name; the builder
      implemented exactly what was asked, and the same reviewing pass concedes this in its own
      TST-GOV-006 assessment. Ground (2) is real and is not closed - so rather than leave the
      finding open indefinitely over a scope question, I carry that half forward as a new linked
      finding, GOV-013, with its own requested test and its own executed evidence, so a maintainer
      sees the lineage and the open work rather than a single ambiguous label. Two further
      low-severity defects in the newly added code are filed as GOV-007 and GOV-008.
  - finding_id: "REPRO-002"
    outcome: verified-resolved
    evidence: |-
      docs/scientific_hardening/REPRODUCIBILITY.md:45 now reads
      "| `pytest -q` | 5 s | 142 passed |". The count is correct: `uv run pytest -q` at the frozen
      worktree reports "142 passed in 17.03s" and `pytest --collect-only -q` collects 142.
      The new guard is a real control, verified by mutation on a disposable copy: changing 142 to
      141 gives "1 failed, 3 passed" on
      `test_reproducibility_record_matches_collected_test_count`; adding a test function fails it
      as well; the unmutated copy gives 4 passed.
      The runtime column was checked rather than assumed: measured `pytest -q` on this Windows host
      is 11.6-23.7 s across runs against a documented approximate 5 s. I do not raise this as a
      finding - the column is explicitly "Approx. runtime", it is strongly machine- and
      load-dependent, the prior "4 s" figure was recorded on the maintainer's environment rather
      than mine, and a candidate finding asserting the runtime figures are now stale was refuted
      under adversarial verification by direct re-measurement. It is recorded here so the number is
      on the record.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Two low-severity defects in the new guard itself are filed as TSTGUARD-001 (a bare
      CalledProcessError discards the child's collection output) and TSTGUARD-002 (the document
      records a PASSED count while the guard compares a COLLECTED count). Neither affects the
      correctness of the committed 142. I also verified that the nested `pytest --collect-only`
      subprocess does not dirty the working tree: pytest writes `.pytest_cache/.gitignore`
      containing `*`, so `git status --porcelain` is empty before and after.

prior_requested_test_results:
  - requested_test_id: "TST-REG-001"
    outcome: verified-satisfied
    evidence: |-
      Implemented as tests/unit/test_simulator.py::test_mds_canonicalization_terminates_when_first_label_is_symmetry_center
      and ::test_mds_direct_anchor_scan_uses_global_rank_tolerance. Coverage matches the request
      literally: the 5-node and 6-node star hop metrics with the hub at index 0, the 3x3 grid hop
      metric with the centre relabelled to index 0 (via the explicit permutation
      [4,0,1,2,3,5,6,7,8]), every candidate dimension D in 1..n-1 asserted to return finite
      coordinates, and the direct rank-two fixture [[1e-9,0],[1,1],[-1,0.5],[-1e-9,-1.5]] with an
      added cdist invariance assertion.
      NEGATIVE-CONTROL STATUS CONFIRMED, which is what the request turned on: against a hybrid tree
      (pre-fix `popgp/` from 4caa4f16 plus candidate `tests/`) both nodes FAIL with
      "RecursionError: maximum recursion depth exceeded" raised at popgp/simulator.py:809, and both
      PASS at the candidate. That is exactly the "must FAIL on the current code and PASS on
      763fd18" bar the request set, one candidate later.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      The tests pin termination and the global rank tolerance. They do not pin the scale
      covariance of the new guard, which is the gap MDS-006 and MDS-007 record and TST-MDS-006 and
      TST-MDS-007 request.
  - requested_test_id: "TST-REG-003"
    outcome: verified-satisfied
    evidence: |-
      Implemented as tests/unit/test_backend.py::test_backend_reuses_interaction_and_cell_hamiltonian_caches
      and ::test_optimize_cells_constructs_each_distinct_cell_generator_once. The latter is the
      call-count assertion the request specified: it monkeypatches the bound
      `build_cell_hamiltonian`, runs a full `optimize_cells`, and asserts both
      `len(calls) == len(set(calls))` and `len(calls) <= 15`.
      NOT VACUOUS, verified two ways. The bound equals the number of distinct 2-site cells at
      n_qubits=6 (C(6,2)=15) while the uncached count is 45 across 15 partitions of 3 cells, so the
      duplicate-free assertion fails first on unmemoised code - I measured exactly 45 calls at the
      prior candidate and 15 at this one. Against the hybrid tree both nodes FAIL. Removing only
      the backend cell-Hamiltonian cache fails the sibling test at tests/unit/test_backend.py:121,
      so both cache layers are pinned.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      The request also asked for a `build_interaction_terms` cache alongside the generator memo;
      both were implemented. The equality anchor the request noted as already existing
      (tests/unit/test_backend.py, both families) is unchanged and still passes.
  - requested_test_id: "TST-MDS-005"
    outcome: verified-satisfied
    evidence: |-
      Implemented as tests/unit/test_simulator.py::test_mds_canonical_frame_is_isometric_near_rank_threshold,
      parametrized over exactly the requested delta set {1e-3, 1e-5, 1e-6, 1e-7, 3e-8, 1.5e-8,
      1.1e-8, 1e-9} crossed with `padded` in {False, True}, asserting maximum relative pairwise
      distance change below 1e-14 - the threshold the request named.
      NEGATIVE-CONTROL STATUS CONFIRMED: against the hybrid tree it fails in 12 of 16
      parametrizations, including all four padded rank-deficient cases at delta in
      {1e-7, 3e-8, 1.5e-8, 1.1e-8} that the request singled out. Headroom is real, not marginal:
      measured worst relative change at the candidate over 2055 constructed configurations is
      2.801364e-11 against the truncated reference and <= 2.5e-15 on the isometric families, with
      max|Q^T Q - I| = 2.220446e-15.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      The request also asked to extend rather than replace the existing cdist invariance assertion
      and to regenerate chain_1d; both were done - the earlier assertion is untouched and chain_1d
      regenerates byte-identically with the improved stress values recorded under MDS-005.
  - requested_test_id: "TST-GOV-006"
    outcome: verified-satisfied
    evidence: |-
      The request's hard bar was "Demonstrate the strengthened test FAILING on all four executed
      mutations". I reproduced all four independently on my own copy and all four fail: control
      cell -> "(to be added)"; every citation -> "control pending"; all rows repointed at
      test_adaptive_gap_is_permutation_equivariant; a new ACTIVE row with no ID. Each gives
      "1 failed, 3 passed" with a specific assertion message
      ("GATE-BLIND-TOPOLOGY has no executable negative control"; the marker assertion; the
      ungated-row membership assertion).
      The three structural sub-requests are implemented as asked: the registry is parsed row-wise
      with a per-row citation requirement, the GATE-* IDs were moved into the FALSIFICATION_MATRIX
      rows with `matrix_gates` derived from rows rather than the prose paragraph, and a
      machine-readable marker is registered at pyproject.toml:32 and AST-verified.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      One honest caveat, which belongs to GOV-013 rather than to this request: the mechanism the
      request itself prescribed - a required marker - is inherently an author assertion, so the
      sub-goal "a cited node can be asserted to be a control rather than an identity regression" is
      not actually achieved. Quantified by execution: adding one marker line to the very identity
      regression the request names makes all six gates cite it and the guard passes. The declared
      fallback (rename the test and state the limitation) was not taken, so the test still asserts
      less than its name. I do not downgrade the outcome, because the builder implemented precisely
      the mechanism the reviewer prescribed. Residual defects in the new code are GOV-007 and
      GOV-008.
  - requested_test_id: "TST-GOV-003B"
    outcome: verified-satisfied
    evidence: |-
      Implemented at tests/unit/test_review_guidance.py:66-81 and :98-101 as
      `_quality_authority_sentences` plus an assertion loop over README.md, docs/governance/*.md
      and docs/reviews/*.md. The request's exact mutation now FAILS: inserting "The README Quick
      start is the authoritative pre-freeze quality suite; run `uv sync` and `uv run pytest -q`
      only." into docs/governance/REVIEWER_IDENTITY.md gives "1 failed, 3 passed" on
      `test_documented_quality_commands_match_authoritative_ci` with the offending path and
      sentence in the message. Executed by me on a disposable copy; baseline 4 passed before and
      after revert.
      The request's explicit prohibition was honoured: the command extractor was NOT widened to
      every `run:` step body, so the uv bootstrap step and the block-scalar marker do not break an
      otherwise-correct tree. Clause 1 of the parent request remains intact and sensitive -
      mutating the ruff command in ci.yml fails the same node.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Residual lexical and polarity holes in the new extractor are filed as GOV-009 with requested
      test TST-GOV-003C; they are outside the literal scope of this request, which specified one
      mutation and got it.
  - requested_test_id: "TST-REPRO-002"
    outcome: verified-satisfied
    evidence: |-
      Both halves of the request are implemented. The document was updated to the collected count
      at the committed tree (REPRODUCIBILITY.md:45, "142 passed"), and
      tests/unit/test_review_guidance.py::test_reproducibility_record_matches_collected_test_count
      asserts the recorded number against a live `pytest --collect-only -q` at the current tree.
      Verified as a real control by mutation: 142 -> 141 fails it; adding a test function fails it;
      the count is correct at 142 by two independent measurements (`pytest -q` "142 passed" and
      `--collect-only -q` "142 collected").
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Two defects in the guard itself are filed as TSTGUARD-001 and TSTGUARD-002. Neither makes the
      committed figure wrong today.
  - requested_test_id: "TST-LEAKGUARD-006"
    outcome: verified-satisfied
    evidence: |-
      Implemented as tests/unit/test_backend.py::test_cell_hamiltonian_validates_indices_and_backend_capability
      and ::test_compute_leakage_requires_backend_edges_and_coupling, covering exactly the five
      preconditions the request named: `compute_leakage` raising when `edges !=
      backend.build_edges()` and when `coupling_J` differs from the backend config;
      `build_cell_hamiltonian` raising on empty, duplicated and out-of-range (both negative and
      too-large) cell indices; and `Backend.build_cell_hamiltonian`'s NotImplementedError.
      LOAD-BEARING, verified by mutating each guard away individually in a disposable copy and
      re-running both nodes (baseline 2 passed):
        `if not cell_indices:`                                  -> 1 failed, 1 passed
        `if len(set(cell_indices)) != len(cell_indices):`        -> 1 failed, 1 passed
        `if any(site < 0 or site >= self._N ...)`                -> 1 failed, 1 passed
        `if edges != backend.build_edges():`                     -> 1 failed, 1 passed
        `if coupling_J != backend.config.substrate.coupling_J:`  -> 1 failed, 1 passed
      All five guards are pinned; none is unreached.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      These two nodes PASS against pre-fix `popgp/` on the hybrid tree, which is correct and
      expected: the guards themselves were added in the previous round (0e8d7be) and this request
      asked for coverage of existing preconditions, not for new behaviour. The guard-by-guard
      mutation above is therefore the appropriate control, and it is satisfied. One caveat worth
      recording: the `coupling_J` precondition can be defeated by the new cache, because it reads
      the mutated config while the cached generator encodes the old coupling - see CACHE-002.
  - requested_test_id: "TST-GOV-003"
    outcome: verified-satisfied
    evidence: |-
      Carried unresolved since round 3 on clause 2 only ("and that no other governance document
      designates a different suite as authoritative"). Clause 1 was already satisfied and remains
      sensitive: `_uv_commands(workflow) == _uv_commands(readme) == _uv_commands(runbook) ==
      QUALITY_COMMANDS` at tests/unit/test_review_guidance.py:93-95 fails when I change the ruff
      command in .github/workflows/ci.yml. Clause 2 is now implemented by the negative assertion
      loop at :98-101, and the specific live bypass round 4 demonstrated by executed mutation -
      inserting a competing authority sentence into docs/governance/REVIEWER_IDENTITY.md - now
      fails. I re-ran that exact mutation myself rather than trusting the prior record.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      I record this satisfied rather than carrying it a fourth round. The requested remedy
      (TST-GOV-003B) was implemented, and the bypass that justified the unresolved label is closed.
      The clause is not perfectly enforced - a sentence that names ci.yml while designating a
      different suite still passes, and near-synonym phrasings are outside the filter - but those
      are new residuals of the new mechanism rather than the unaddressed original request, and they
      are filed as GOV-009 with requested test TST-GOV-003C so they remain tracked.
  - requested_test_id: "TST-GOV-004"
    outcome: verified-satisfied
    evidence: |-
      Carried unresolved since round 3 on clause (c) only ("add a meta-test asserting that every
      gate row names an existing mutation-test id"). Round 4's evidence for non-satisfaction was
      that the node regex scanned the whole file and that no table row named a GATE ID. Both
      conditions are gone, verified by reading and by parsing the documents myself: the six IDs now
      sit in FALSIFICATION_MATRIX rows and `matrix_gates` is built row-wise, while the registry is
      parsed row-wise with a per-row `assert test_nodes` and `assert test_path.is_file()`. Every
      round-4 counterexample for clause (c) now fails: "(to be added)" in one cell; all citations
      replaced by "control pending"; a new ACTIVE row with no ID. Node liveness is enforced too -
      renaming or relocating a cited control both fail, as does deleting a registry row.
      Clauses (a) and (b) remain satisfied: all seven cited controls exist, carry the marker, and I
      read each one - they are genuine falsifiers (0.4*eps + 0.7*eps**2 rejected by
      `assess_quadratic_response`; the commuting-Ising quench profile allclose to 1e-13; the
      reduced modular source below 1e-12 while the KMS density is not; the equal-energy
      different-entropy analytic control; uniform 0.2 correlations giving separable False; the
      disjoint-Bell control pinning the retained false D_star == 1; a sub-floor response raising
      "absolute precision floor"). Clause (d) remains declined and disclosed at
      GATE_TEST_REGISTRY.md:19-23, DECISIONS.md D012 and in the response's risks, unchanged and
      unweakened by this commit.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Clause (c) as written asks that every gate row name an existing mutation-test id, and that is
      now mechanically enforced row-wise and demonstrably fails on all of round 4's
      counterexamples. The deeper question it gestures at - whether the named id is genuinely a
      control that can fail, and whether gates outside the registry are covered at all - is not
      clause (c); it stays open under the new finding GOV-013 with requested test TST-GOV-013.

predictions:
  experiment_id: "MDS-006-scale-covariance-regression"
  predicted_outcome: |-
    If TST-MDS-006 is committed as specified and run against
    3974bbfab89d97bac79ce298285a7ce8f36f69fd, `Simulator._classical_mds(d * s, D)` on the 6-node
    open-chain hop metric will raise `RuntimeError: embedding rank and anchor selection disagree`
    for s in {1e-10, 1e-11, 1e-13} at every D in 2..5, and will succeed for
    s in {1, 1e-4, 1e-8, 1e-9, 1e-12, 1e-14, 1e-15}. The same test at
    4caa4f16b407a47a9531442751ccde1e446b8271 will return finite coordinates at every one of those
    16 scales, with maximum relative pairwise-distance error against the target ranging up to
    1.185e-08 (that being MDS-005, which this candidate fixes). After the required action is
    applied - deriving `scale` at popgp/simulator.py:835 from the largest singular value already
    computed at :808 rather than from `max(1.0, ||centered||_F)` - the test will pass at every
    scale, and tests/unit/test_simulator.py's existing MDS nodes plus
    test_mds_canonicalization_terminates_when_first_label_is_symmetry_center will continue to pass
    unchanged.
  predicted_failure_mode: |-
    The prediction is falsified if the rescaled chain fixtures return finite coordinates at this
    candidate. The most likely benign cause of that outcome is NOT a defect in the finding but an
    import-path error: the venv at C:/src/POPGP-review-source-law-rereview-5/.venv contains
    `_editable_impl_popgp.pth` pointing at the frozen worktree, so any script executed from outside
    a copy imports the FROZEN tree regardless of cwd, and a naive A/B then shows both trees
    behaving identically. This bit round 4 and it bit my own first attempt in this round. Any
    replication must print `os.path.abspath(popgp.__file__)`, assert it resolves under the intended
    tree, and additionally assert `hasattr(Simulator, "_select_embedding_anchors")` matches the
    tree under test, before concluding anything. A genuine refutation would require
    `_select_embedding_anchors` to reach `dimension` anchors with a maximum residual norm above
    `eps * max(1.0, ||centered||_F)` on a matrix whose smallest singular value is below
    2.22e-16 while exceeding `1e-8 * sigma_max`.
  confidence_statement: |-
    High confidence in the resolution of all five prior findings and the satisfaction of all nine
    requested-test items: each was verified by replaying its original defect condition against both
    trees, and each new test was run against pre-fix source on a hybrid tree rather than merely
    read. High confidence in MDS-006 and MDS-007, which are reproducible from a five-line script
    against the frozen tree and were confirmed by two independent passes. High confidence that no
    gate outcome or published number moved, which rests on a complete leaf-by-leaf diff of all
    three changed artifacts plus an orthogonal Procrustes check, not on sampling. Moderate
    confidence on cross-platform behaviour: every command in this review ran on Windows 11 /
    CPython 3.11.15 / torch 2.10.0+cpu, and the 18/18 byte-identity result is a Windows result -
    PNG and GIF regeneration on Linux is not verifiable within this access boundary. Moderate
    confidence in the severity calibration of MDS-006: I record it non-blocking on the strength of
    a 480-configuration public-API sweep finding zero reachable failures, and a wider or
    differently-shaped sweep could change that judgment. Low confidence that this review would
    detect an error shared with rounds 3 and 4, because the same model has now occupied the
    reviewing seat for three consecutive rounds.

recommendation:
  approve: true
  blocking_findings: 0
  rationale: |-
    blocking_findings = 0 unresolved prior findings + 0 blocking new findings = 0.

    All five round-4 findings are verified-resolved by executed replay of their original defect
    conditions, and all seven round-4 requested tests are verified-satisfied with their
    negative-control status confirmed against pre-fix source. The two clauses carried unresolved
    since round 3 - TST-GOV-003 clause 2 and TST-GOV-004 clause (c) - are satisfied. Round 4's one
    blocking finding, REG-001, is fixed structurally rather than patched: termination is now
    provable because the represented column count strictly decreases, and public-API Pi_geom
    failures went from 5 in 420 configurations at the prior candidate to 0 in 480 here.

    Fifteen new findings are recorded and every one is non-blocking. None affects a published
    number, a gate outcome, an acceptance threshold or a scientific claim; the full quality suite
    passes and all 18 committed artifacts regenerate byte-identically and deterministically.

    The judgment most likely to be disputed is disclosed rather than buried. MDS-006 and MDS-007
    are availability defects in the same function whose availability defect was round 4's blocker,
    and a reviewer applying round 4's reasoning mechanically would mark them blocking too. I do not,
    for three measured reasons: the new corner is strictly narrower than the one it replaces
    (0 of 480 public-API configurations versus 5 that reached REG-001 at the prior candidate);
    the failure is a loud immediate exception rather than an unbounded recursion; and reaching it
    requires either a direct private-API call or a distance matrix at absolute scale below ~1e-8,
    which the default kernel cannot produce. A maintainer who weighs "a regression in the function
    just remediated" more heavily than reachability may reasonably require MDS-006 before merge;
    the evidence to make that call is in the finding.

    Two further judgments are disclosed. GOV-006 is recorded verified-resolved although one
    reviewing pass in this session argued unresolved; the branch of its required action that the
    builder chose was implemented item-for-item, and the genuinely un-closed half - an
    unregistered acceptance gate in example code shipping green through the whole CI, which I
    reproduced end to end - is carried forward as the new linked finding GOV-013 rather than left
    as an ambiguous label. And approval here is approval of a re-review, not of merge: this round
    shares an operator with the builder and reuses the reviewing model of rounds 3 and 4, so the
    external re-review recommended by round 4 is still owed.
```

## 1. Method

**Frozen worktree.** All review work was conducted from
`C:/src/POPGP-review-source-law-rereview-5`, a worktree created at the exact candidate on the
branch `review/source-law-linear-response-rereview-5`.

**HEAD and tree verified before, during and after.** `git rev-parse HEAD` returned
`3974bbfab89d97bac79ce298285a7ce8f36f69fd` and `git rev-parse "HEAD^{tree}"` returned
`672e171a6532159c388aba804188fba28295c835`, matching the declared `context_hash` exactly.
`git status --short` was empty before any work began, after the full quality suite including all
six example regenerations, and again at the end. The builder worktree
`C:/src/POPGP-source-law-remediation-1` was confirmed clean at the same SHA and was never written
to, and `C:/src/POPGP` was not touched.

**The frozen tree stayed clean.** Every mutation, counterexample and A/B comparison ran in a
**separate disposable copy** produced with `git archive <commit> | tar -x` into the session
scratchpad — copies of the candidate `3974bbfa`, the prior candidate `4caa4f16`, the baseline
`c03800e4`, a **hybrid tree** pairing pre-fix `popgp/` with candidate `tests/`, and per-probe
copies for the governance mutations. No `git add`, `git commit` or `git checkout` was run in the
frozen worktree; the only writes there were the quality-suite regenerations, which produced zero
drift, and this artifact.

**Import shadowing — the methodological hazard of this repository.** The venv at
`.venv/Lib/site-packages/_editable_impl_popgp.pth` points at the frozen worktree, so *any* script
run from outside a copy imports the frozen tree regardless of `cwd`. My first A/B produced
**identical output for both trees** and was discarded. Every subsequent execution injected the
target tree at `sys.path[0]`, printed `os.path.abspath(popgp.__file__)`, asserted it resolved under
the intended tree, and additionally asserted `hasattr(Simulator, "_select_embedding_anchors")`
matched the tree under test. Round 4 recorded this hazard; it is real, it recurs, and it produces
false negatives that look like refutations.

**Quality suite — authoritative command set, in CI order, at the frozen worktree:**

| Command | Exit | Observed result |
|---|---:|---|
| `uv sync --frozen` | 0 | CPython 3.11.15; numpy 2.4.2, torch 2.10.0+cpu, scipy 1.17.0, pytest 9.1.1, ruff 0.16.2 |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | 652 lines; brace balance 0; all 9 environments matched; no markdown remnants; no non-ASCII |
| `uv run pytest -q` | 0 | **142 passed in 17.03s** |
| `uv run pytest --collect-only -q` | 0 | 142 collected |
| `uv run python -m examples.physics_qg.chain_1d` | 0 | MDS stress 0.0000; coords `[1.50151012, 0.5, -0.5, -1.50151012]`; Φ range ±0.0221 |
| `uv run python -m examples.physics_qg.grid_2d` | 0 | separation 36.3%, `SUCCESS: Local structure preserved` |
| `uv run python -m examples.physics_qg.gravity_well` | 0 | `GREEN-FUNCTION DIAGNOSTIC: PASS`; monotonic recovery PASS; grid symmetry PASS |
| `uv run python -m examples.physics_qg.source_law` | 0 | relative_entropy slope 1.999684; modular_energy slope 1.000000 |
| `uv run python -m examples.physics_qg.source_law_many_body` | 0 | quadratic coeff 7.784273e-02; drift 0.000e+00; t=1 endpoint fraction 0.120782 |
| `uv run python -m examples.physics_qg.ca_model` | 0 | 32 cells, artifacts written |
| `uv run python scripts/check_validation_artifacts.py` | 0 | `Validation artifact contracts and required visual outputs are valid.` — **no "accepted N bounded numeric drifts" line**, i.e. zero tolerance consumed |
| `git status --short` | 0 | **empty** |

**Artifact bit-identity: 18 of 18, and deterministic.** In a clean disposable copy I regenerated all
six examples **twice** and compared `sha256` across runs — all 18 files under
`examples/physics_qg/*/results/` identical between runs, including the PNG and GIF binaries. I then
compared `git hash-object` of each regenerated file against the committed blob SHA from
`git ls-files -s`: **18/18 matched**.

**CI.** `.github/workflows/ci.yml` is byte-unchanged from the prior candidate; every step was
reproduced locally in order and returned 0. The exact-SHA CI evidence supplied in the handoff was
recorded but **not** used as a substitute for local execution, and no authenticated `gh` access was
used at any point.

## 2. The remediation diff

`git diff 4caa4f16..3974bbfa` touches 23 files: 2,693 insertions, 128 deletions. Excluding the two
review artifacts (2,241 lines) the substantive change is small and I read all of it —
`popgp/simulator.py` (+64/−?), `popgp/backend.py` (+16), `popgp/coarse_grain.py` (+14),
`pyproject.toml` (+1), four documentation files, three changed `validation.json` files, two PNGs,
and 306 lines of tests across eight test files.

**No acceptance threshold was widened anywhere.** I enumerated every changed numeric literal in
`popgp/`, `scripts/`, `examples/*/__main__.py` and `tests/`: the only new constants are in the new
MDS code paths and the new test fixtures. No constant in `popgp/diagnostics.py` changed.

## 3. Mutation probes

The builder marked every item implemented. Mutation is the only way to separate "implemented" from
"implemented and load-bearing". Every probe below was executed by me in a disposable copy, applied
then reverted, with the baseline re-confirmed after each revert.

| Probe | Mutation | Caught? | Exact result |
|---|---|:--:|---|
| **REG-001** | Run the 19 round-4 fixtures against pre-fix `popgp/` | **YES** | 19/19 `RecursionError`; 19/19 finite at the candidate |
| **REG-001 tests** | Both new nodes on the hybrid tree | **YES** | `2 failed`, `RecursionError` at `simulator.py:809` |
| **MDS-005 test** | `test_mds_canonical_frame_is_isometric_near_rank_threshold` on the hybrid tree | **YES** | `12 failed, 4 passed` of 16 parametrizations |
| **REG-003 tests** | Both cache tests on the hybrid tree | **YES** | `2 failed, 2 passed` |
| **LEAKGUARD ×5** | Remove each of the 5 preconditions individually | **YES ×5** | `1 failed, 1 passed` each time |
| **GOV m1** | Control cell → `(to be added)` | **YES** | `1 failed` — "GATE-BLIND-TOPOLOGY has no executable negative control" |
| **GOV m2** | All 8 citations → `control pending` | **YES** | `1 failed, 3 passed` |
| **GOV m3** | Repoint all rows at the unmarked identity regression | **YES** | `1 failed, 3 passed` |
| **GOV m4** | New ACTIVE matrix row with no GATE ID | **YES** | `1 failed, 3 passed` |
| **GOV m8/m11/m12/m13/m14** | Delete a registry row / rename / relocate a control / corrupt the "Matrix row" cell / malformed row | **YES ×5** | `1 failed, 3 passed` each |
| **GOV m5** | m3 **plus** one `@pytest.mark.negative_control` line on the identity regression | **NO** | `4 passed` — the marker is an author assertion |
| **GOV m7** | New ungated row **plus** one line added to `UNGATED_MATRIX_ROWS` | **NO** | `4 passed` — an allow-list is a one-line bypass by construction |
| **GOV m9** | Delete the marker registration at `pyproject.toml:32` | **NO** | `142 passed, 7 warnings` — no `--strict-markers` (**GOV-008**) |
| **GOV m10 / probe (e)** | New acceptance gate in `source_law/__main__.py`, no registry entry, **committed with its regenerated artifact** | **NO** | ruff 0, check_tex 0, **142 passed**, 6 examples 0, `check_validation_artifacts` 0 (**GOV-013**) |
| **TST-GOV-003B** | Round-4's competing-authority sentence into `REVIEWER_IDENTITY.md` | **YES** | `1 failed, 3 passed` |
| **Authority evasion** | "**Although** ci.yml exists, the authoritative quality suite is the README Quick start." | **NO** | `4 passed` — polarity blindness (**GOV-009**) |
| **TST-GOV-003 clause 1** | Add `--fix` to the ruff command in `ci.yml` | **YES** | `1 failed, 3 passed` |
| **TST-REPRO-002** | `142` → `141` in `REPRODUCIBILITY.md` | **YES** | `1 failed, 3 passed` |

## 4. The geometry change is a rigid motion, and no gate moved

I diffed every JSON leaf of the three changed artifacts, classified each change, and checked the
transform rather than trusting the description.

| Artifact | Changed leaves | Gate booleans / `overall_pass` / `criterion` / `D_star` / `embedding_status` | Coordinate transform |
|---|---:|---|---|
| `chain_1d` | 7 | **none changed** | coords **bit-identical** (Procrustes residual 0.0) |
| `grid_2d` | 48 | **none changed** | **proper rotation**: residual 1.776e-15 vs coord scale 2.5487, det(R) = **+1.000000**, max relative pairwise-distance change **4.382e-16** |
| `gravity_well` | 19 | **none changed** | no coords block; metric diagnostics moved ≤ **7e-13** relative |

Zero leaves added, zero removed, check names identical in all three. The chain_1d
`selection_driver` gate compares `stress_span` against `max(1e-12, 1e-6 × selection_margin)` =
1.32e-8; it held before at 6.64e-10 and holds now at 5.51e-17 — the fix **increased** headroom from
about 20× to about 2.4e8× without changing the verdict.

**Negative results survived.** Three of six examples still ship `overall_pass: false`; `grid_2d` and
`gravity_well` still fail `pi_res_admissibility`; `chain_1d` still carries `mst_degenerate: true`;
raw relative entropy is still falsified as a linear source; the reduced-state modular candidate is
still recorded as blind; the disjoint-Bell false `D*=1` is still disclosed. A remediation that
rotated an entire embedding un-failed nothing.

## 5. The framework.tex / framework.md locality disagreement

Round 4 recorded this as an **unverified** observation (section 8, item 6). It was real, and it is
corrected.

- **Prior `framework.tex:511`:** *(Definition)* — "Locality is defined from QCMI-screened
  correlations and multi-hop routing on weighted graphs. **Failure mode:** Definition is checkable;
  failure mode is pathological graphs…"
- **Prior and current `framework.md:887`:** "Definition / partial prototype … Pairwise MI and blind
  graph routing are implemented; QCMI screening is not… Fails if non-geometric controls produce
  stable geometric declarations…"
- **Current `framework.tex:511`:** *(Definition / partial prototype)* — "Pairwise mutual
  information and blind multi-hop graph routing are implemented; QCMI screening is not… **Failure
  mode:** Falsified if non-geometric controls produce stable geometric declarations…"

The `.md` row was already correct; the `.tex` row — which `REPRODUCIBILITY.md:20` designates the
**authoritative** manuscript source — was the one overstating implementation status on all three
axes round 4 named. It is now substantively, not merely keyword-, equivalent.

**The new text is true**, verified against the code rather than accepted: pairwise MI is implemented
(`popgp/simulator.py` Π_loc), blind multi-hop routing is implemented (adaptive-gap/kNN + MST +
Floyd–Warshall), and QCMI screening genuinely has **no** implementation anywhere — a tree-wide grep
finds only disclaimers (`README.md:106` and `:197`, `CLAIMS_MATRIX.md` C07 "Unimplemented",
`THEORY_CODE_GAP.md:9`, `popgp/simulator.py:396`, and the abstracts of both manuscripts).

Two residuals are recorded rather than glossed. The guard that pins this parity is substring-only
and shadowable by a LaTeX comment (**FRAMEWORK-003**). And §4.4.3 of both manuscripts still
describes the QCMI filter in the present tense as part of the construction — defensible as a
*specification* section given the abstract, the status list and three other documents all disclaim
implementation, which is why I do not raise it as a finding, but worth a maintainer's eye when the
manuscript is next touched.

## 6. What holds up

Five rounds is a lot of de-escalation, and the substance of this one is good. Being specific about
where credit is earned:

- **REG-001 was fixed the right way.** The cheap fix was a recursion-depth guard or a
  `represented_rank`-passing flag. Instead the rank test was moved to global singular values and
  the failing greedy scan was *replaced*, so termination follows from the column count strictly
  decreasing rather than from a bound. It is provable, not empirical — and the empirical check
  agrees: 36,058 calls at `recursionlimit(61)`, zero failures, maximum nesting depth 3.
- **Round 4 understated the value of this fix.** It recorded REG-001 as unreachable through any
  supported `SubstrateConfig`. It was reachable: 5 of 420 public `Simulator.run()` configurations
  raise `RecursionError` at the prior candidate. This candidate raises nothing in 480.
- **MDS-005 took the branch that costs something.** Documentation-only was permitted. Instead the
  estimator was replaced with an SVD polar factor, isometry improved by seven orders of magnitude,
  and — the part that matters scientifically — the fix *changed a published diagnostic*, moving
  `chain_1d stress_by_dimension["2"]` from 6.64e-10 to 5.03e-16 and thereby retiring a number round
  4 identified as 1.3 million times its true value.
- **Equivariance improved rather than being traded away.** The LIB-004 property was the thing most
  at risk from rewriting the anchor scan. Measured over 1500 randomized cases the worst
  right-orthogonal deviation went from 7.945e-07 to 3.857e-15, and on the 4×4 grid under adversarial
  degenerate-eigenbasis mixing from 2.838e-12 to 5.607e-15.
- **The governance layer grew real teeth.** Nine of my fourteen governance mutations now fail,
  including all four round 4 demanded. The registry-to-matrix binding is exact: 6 gated rows, 6
  registry rows, 7 cited controls, 7 markers, a perfect bijection with no dead allow-list entries.
- **The negative-control markers went on the right functions.** I checked all seven individually
  rather than trusting the registry cells. They are genuine falsifiers, including the one that pins
  a *known false positive* — the disjoint-Bell `D*=1` — as a permanent regression rather than
  deleting it.
- **The two long-running unresolved clauses were actually closed**, not argued away. TST-GOV-003
  clause 2 and TST-GOV-004 clause (c) have been open since round 3; both now fail under the exact
  mutations that justified keeping them open.

## 7. Open scientific questions

Unchanged from round 4 except where noted; the builder claimed as much and I spot-checked the claim
by auditing every changed documentation line for silent strengthening and found none.

1. **The source law is still an identity plus a convention, not a derived law.** No discrete
   continuity current, no local conservation law, no covariant closure is implemented.
2. **One gate carries the falsification pressure.** `minimum_signal_to_floor` is the only swept
   statistic that tracks the physics monotonically and approaches its threshold.
3. **Emergent dimension remains parameter-dependent.** This round makes the *frame* canonical and
   now genuinely isometric; it does not make the *dimension* canonical.
4. **Cross-platform reproducibility is attested, not demonstrated to me.** The 18/18 byte-identity
   result is a Windows result.
5. **The KMS reference gate does not scale.**
6. **New this round:** the recorded mitigation for the failing non-geometric-controls gate is a
   knife-edge artifact of exact MI degeneracy (**FRAMEWORK-004**). Under a generic perturbation the
   disjoint-Bell control trips *both* halves of that row's own stated failure threshold. The false
   `D*=1` was already admitted; what is inaccurate is the recorded readiness that softens it.
7. **New this round:** a scale-invariant goodness-of-fit statistic has a scale-dependent silent
   "perfect fit" fallback that feeds the headline `embedding_status` (**GEOM-501**, pre-existing).

## 8. What a maintainer should do with this artifact

**Zero blocking findings.** The fifteen new findings are housekeeping and can be scheduled. If any
one is pulled forward, make it **MDS-006** — not because it is reachable today (it is not, in 480
public-API configurations) but because it is the *third* appearance in this chain of the same root
cause: an absolute tolerance mixed into a scale-free algorithm. REG-001 was that bug. MDS-006 and
MDS-007 are that bug again, in the code written to fix it. A single pass making every tolerance in
`_canonicalize_embedding` relative to `singular_values[0]` would close the pattern rather than the
instance. **GOV-013** is the other one worth doing early, because it is the only finding here that
lets a *future* overclaim ship green.

And treat the independence statement as binding rather than boilerplate. This round is a genuine
improvement on round 4 — a fresh session, no orchestrator between the roles, evidence reproduced
rather than inherited — but it still shares an operator with the builder, and the same model has now
occupied the reviewing seat for three consecutive rounds. Approval here means the re-review found no
blocking defect. It is not independent experimental confirmation of any physical claim in this
repository, and it does not discharge the external re-review round 4 recommended.
