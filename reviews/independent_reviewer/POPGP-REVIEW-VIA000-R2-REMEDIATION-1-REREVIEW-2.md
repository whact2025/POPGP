# VIA-000 R2 reproducibility-remediation independent re-review 2

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-2"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "unknown"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-via000-r2-remediation-independent-rereview-session-2"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-19"
commit_reviewed: "3ceb92c37aa0d88706add080d458ca42c153524d"
baseline_commit: "d0381f0e8562c5f70c4e315dc3df4afd0d6bfbf4"
prior_review_ref: "9012780d93df9ffa7e531135dd0e7f150cc5aa62:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-1.md"
builder_response_ref: "3ceb92c37aa0d88706add080d458ca42c153524d:reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-1-RESPONSE-1.md"
context_hash: "c500f18799b62c9b245498223dda118b88721d1d"
context_hash_method: "git rev-parse \"3ceb92c37aa0d88706add080d458ca42c153524d^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - ".github/workflows/via000-r2-boundary.yml"
  - ".gitignore"
  - "README.md"
  - "pyproject.toml"
  - "uv.lock"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
  - "docs/scientific_hardening/CLAIMS.md"
  - "docs/scientific_hardening/DECISIONS.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/PROJECT_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-VIA000-R2-REMEDIATION-PLAN.md"
  - "scripts/check_reproduction_boundary.py"
  - "scripts/check_validation_artifacts.py"
  - "scripts/run_without_startup_hooks.py"
  - "scripts/check_tex.py"
  - "popgp/diagnostics.py"
  - "examples/physics_qg/ca_model/__main__.py"
  - "examples/physics_qg/ca_model/results/validation.json"
  - "examples/physics_qg/chain_1d/__main__.py"
  - "examples/physics_qg/chain_1d/results/clock_potential.png"
  - "examples/physics_qg/chain_1d/results/validation.json"
  - "examples/physics_qg/gravity_well/__main__.py"
  - "examples/physics_qg/gravity_well/results/source_comparison.png"
  - "examples/physics_qg/gravity_well/results/validation.json"
  - "examples/physics_qg/grid_2d/__main__.py"
  - "examples/physics_qg/grid_2d/results/clock_potential.png"
  - "examples/physics_qg/grid_2d/results/validation.json"
  - "examples/physics_qg/source_law/__main__.py"
  - "examples/physics_qg/source_law/results/validation.json"
  - "examples/physics_qg/source_law_many_body/__main__.py"
  - "examples/physics_qg/source_law_many_body/results/many_body_source.png"
  - "examples/physics_qg/source_law_many_body/results/validation.json"
  - "tests/scientific/test_many_body_source_law.py"
  - "tests/unit/test_reproduction_boundary.py"
  - "tests/unit/test_validation_artifact_contract.py"
  - "tests/unit/test_review_guidance.py"
access_level: "public-repository-only plus public GitHub Actions metadata, logs, and downloadable artifacts"
independence_statement: |-
  This was a fresh independent-reviewer task in a new isolated worktree and branch
  created directly from exact handoff commit 3ceb92c37aa0d88706add080d458ca42c153524d.
  Before conclusions, HEAD, tree c500f18799b62c9b245498223dda118b88721d1d,
  branch, source remote, and clean status were frozen and recorded. The complete
  baseline-to-handoff diff and the full original review/response/re-review/response
  chain were read. Candidate source, tests, workflows, scientific documents, and
  retained results were not modified. Only this review artifact was added.

  Prior attacks and broader visual, semantic, startup, environment, repository,
  index, path, and hosted-platform variants were independently replayed. Builder
  statements, response evidence, tests, CI, and the current checker were treated as
  hypotheses. No untracked handoff file, custody material, sealed material, hidden
  label, secret seed, private evaluator, or private hardware profile was read. The
  same human operator and Codex Desktop orchestrator are shared with the builder,
  while task/session and worktree are distinct. Exact served-model identities and
  snapshots were not exposed, so both remain unknown and model separation is false.
  This is internal adversarial process separation, not external scientific validation.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "unknown"
  builder_session_id: "popgp-via000-r2-remediation-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  Changes requested with one unresolved blocking finding. The calibrated visual
  maximum-channel rule, Python startup boundary, and literal repository/environment
  boundary survive independent replay and broader variants. Focused visual/startup/
  residue/review-guidance testing passes, and exact-candidate ordinary CI plus the
  trusted Windows/Ubuntu boundary workflow are green. Downloaded retained bundles
  bind the exact commit, tree, trusted checker blob, complete environment manifest,
  literal source manifest, regenerated JSON and raster artifacts, and final clean
  status. Honest hosted raster drift is two counts on Ubuntu and three on Windows,
  below the frozen limit of four.

  The existing semantic finding does not survive re-review. The portability
  comparator accepts 4e-9 changes to retained raw diagnostic operands, while the
  semantic checker often binds only stale serialized summaries to stale check copies.
  Independently executed mutations invalidate all four simple source-law fits, the
  affine identity, and both solver-ratio relationships without an error. In the
  many-body report, accepted raw mutations invalidate the top-level quadratic gate,
  KMS/first-law/local-decomposition identities, global-energy conservation, initial
  locality, isospectral identities, and the commuting-control identity. Mirroring one
  accepted raw mutation across the two retained order-sensitivity aliases makes the
  raw quadratic assessment fail in every one of the twelve declared sweep cases while
  comparison and semantic validation both report zero errors. These are direct
  continuations of VIA000-R2-SEMANTIC-001 and
  TST-VIA000-R2-SEMANTIC-MARGIN-001, so no duplicate IDs are introduced.

  The focused 89-test suite passed in 125.85s. Ruff and TeX source validation passed.
  A two-pass local pdfLaTeX build produced an 11-page PDF (535368 bytes,
  SHA-256 cdaf84102bc4e1085a24337c5cde2d1eca17c6f5927d786f403b79a7e15cd8dd)
  with only documented layout/bibliography warnings. The authoritative local suite
  passed 252 tests in 1292.96s under the base-owned `-I -S` wrapper with external
  bytecode/cache paths; after adding this artifact, the review-guidance suite passed
  9 tests in 25.47s. Exact-SHA hosted runs execute all six examples and the full suite
  on both supported hosted operating systems. Review-chain artifacts are
  duplicate-key clean and Draft 2020-12 schema-valid; immutable refs and fix commits
  resolve, and `git diff --check` is clean. No tracked AGENTS.md is present. The task's
  named docs/reviews/INDEPENDENT_REVIEW_PROTOCOL.md is also absent at the exact
  handoff; the tracked agent-review governance, launch guide, templates, and schemas
  were used instead. That documentation-name mismatch is recorded as a limitation,
  not a surviving implementation-boundary blocker. The local machine exposes an
  NVIDIA RTX PRO 3000 Blackwell GPU (compute capability 12.0, driver 595.79), but
  `nvcc` is not on PATH. CUDA execution is not required by the frozen R2 evidence or
  its declared gate, so this is a limitation rather than a blocker.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VIA000-R2-VISUAL-001"
    outcome: verified-resolved
    evidence: |-
      The current comparator imposes a calibrated maximum per-channel difference of
      four after canonicalizing the near-zero chain legend display text without
      changing the raw plotted phi_mean. Independent focused execution passed 89
      tests, including compact 18x18 features, one-pixel continuous curves, sparse
      dashed curves, rendered text, real retained numeric annotations, and <=4/>4
      controls. Retained exact-SHA artifacts independently compare at maximum channel
      drift two on Ubuntu and three on Windows. The current checker reports zero
      semantic and visual errors for both complete bundles.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The resolution is an implementation-boundary result under the retained raster
      renderer and declared maximum-channel contract, not a scientific image-quality
      endorsement.

  - finding_id: "VIA000-R2-SEMANTIC-001"
    outcome: unresolved
    evidence: |-
      Every counterexample below was executed against the exact frozen candidate.
      Unless a paired alias is explicitly listed, only the named raw JSON number was
      increased by 4e-9; the stored fit, metric, Boolean, and check value were left
      stale. Each named case produced the stated accepted numeric drift count, zero
      portability errors, and zero semantic errors.

      In examples/physics_qg/source_law/results/validation.json, each of these four
      one-drift mutations is accepted: $.measurements.relative_entropy[0] changes the
      raw fit slope from stored 1.999683771505978 to 1.7974278943324882;
      $.measurements.modular_energy[0] changes it from 0.9999999999995276 to
      0.9999547950910586; $.measurements.relative_entropy_phi_amplitude[0] changes it
      from 1.9996837715059763 to 1.7451161434755034; and
      $.measurements.modular_energy_phi_amplitude[0] changes it from
      0.9999999999995274 to 0.9998640620742912. Thus all four retained fit objects must
      be recomputed as `fit_power_law(config.epsilons, abs(raw_response))`, with all
      slope/intercept/residual/R-squared fields bound to both serialized aliases.

      The same source-law mutations cross decision relationships. For
      $.measurements.modular_energy[0], the raw affine residual
      `max_i |modular_energy[i] - eps[i]*(modular_energy[-1]/eps[-1])|` becomes
      4.000000027750794e-9 against the `<1e-12` criterion in
      affine_modular_linearity_identity_regression. For the four raw paths above,
      `max(ptp(relative_entropy_phi_amplitude/relative_entropy),
      ptp(modular_energy_phi_amplitude/abs(modular_energy)))` respectively becomes
      0.3273545910961746, 0.00031338450821460917, 67.20035255523553, and
      0.0009441780009089573 against the `<1e-10` criterion in
      linear_solver_homogeneity_identity_regression. The relative-entropy mutation
      also makes `abs(fit_power_law(eps, relative_entropy).slope-2)` equal
      0.2025721056675118 against `<0.02` in relative_entropy_is_quadratic.

      The following one-drift mutations in
      examples/physics_qg/source_law_many_body/results/validation.json are also
      accepted with zero semantic errors, while the named raw formula crosses its
      check gate:

      * $.measurements.relative_entropy[1] makes
        `assess_quadratic_response(config.epsilons, relative_entropy, floor)` fail:
        raw slope 1.55733432666701, slope deviation 0.4426656733329899,
        full/lower coefficient difference 0.609221799129978, and maximum normalized
        RMSE 54.12887444792176. Stored passed remains true. Affected check:
        nonaffine_kms_response_orders.
      * $.measurements.modular_energy[3] makes
        `max_i |modular_energy[i]-beta*total_energy_change[i]|`
        3.9999988293262285e-9 against `<5e-13`. Affected metric/check:
        kms_identity_error / kms_and_local_decomposition_identities.
      * $.measurements.entropy_change[4] makes
        `max_i |relative_entropy[i]-(modular_energy[i]-entropy_change[i])|`
        3.999999616750079e-9 against `<5e-13`. Affected metric/check:
        first_law_identity_error / kms_and_local_decomposition_identities.
      * $.measurements.local_energy_profiles[4][0] makes
        `max_i |sum_j(local_energy_profiles[i][j])-total_energy_change[i]|`
        4.0000000000019946e-9 against `<5e-13`. Affected metric/check:
        local_decomposition_error / kms_and_local_decomposition_identities.
      * $.measurements.evolved_total_energy[1] makes
        `ptp(evolved_total_energy)` 3.999999997894577e-9 against `<1e-12`.
        Affected metric/check: generator_observable_consistency_drift /
        local_energy_decomposition_consistency_and_spreading.
      * $.measurements.evolved_local_energy_profiles[0][0] makes
        `sum(abs(profile[0][[0,4]]))/sum(abs(profile[0]))`
        5.882790846465623e-9 against `<1e-12`. Affected metric/check:
        initial_endpoint_fraction /
        local_energy_decomposition_consistency_and_spreading.
      * $.measurements.isospectral_unitary_control.relative_entropy[0] makes
        `max_i |relative_entropy[i]-modular_energy[i]|`
        4.000000084797154e-9; independently,
        $.measurements.isospectral_unitary_control.entropy_change[0] makes
        `max_i |entropy_change[i]|` 4e-9. Both exceed the dimension-scaled tolerance
        7.105427357601002e-15. Affected metrics/check: max_D_minus_modular_energy and
        max_entropy_change / isospectral_unitary_identity_regression.
      * $.measurements.commuting_ising_control.t1_profile[0] makes
        `max_j |t1_profile[j]-initial_profile[j]|`
        4.000000006938894e-9 against `<1e-12`. Affected metric/check:
        maximum_profile_change_at_t1 / spreading_requires_noncommuting_dynamics.

      The four many-body descriptive fit sources are independently mutable while
      their retained fits stay stale. Accepted raw paths and stored-to-recomputed
      slope changes are $.measurements.relative_entropy[1],
      2.000037751119905 to 1.55733432666701;
      $.measurements.modular_energy[3], 1.0000333334890334 to
      1.0000253181436145; $.measurements.total_energy_change[0],
      1.0000333335872473 to 0.9997991533614549; and
      $.measurements.potential_amplitudes[0], 1.000019825843085 to
      0.9996222681868456. The first is decision-bearing through the quadratic gate;
      the other three are declared descriptive but still require raw binding to avoid
      internally contradictory evidence.

      The parameter-sensitivity check stores each raw case twice. For every case
      index j=0..11, increasing both exact paths
      $.measurements.order_sensitivity[j].relative_entropy[4] and
      $.checks[4].value[j].relative_entropy[4] by 4e-9 produces two accepted drifts,
      zero comparison errors, and zero semantic errors, while recomputing
      `assess_quadratic_response(case.epsilons, case.relative_entropy,
      absolute_precision_floor=case.absolute_precision_floor,
      lower_window_size=5)` returns false. $.checks[4] is the check named
      nonaffine_kms_parameter_sensitivity in the frozen JSON. Exact cases and raw
      slope/coefficient-difference/max-RMSE triples are:

      * j=0, n=5/heisenberg/beta=0.3: 2.1393727986233655,
        3.4265103696272243, 0.9095461246132468.
      * j=1, n=5/heisenberg/beta=1.0: 2.03555258152986,
        0.8723037888823999, 2.7143725050553242.
      * j=2, n=5/heisenberg/beta=2.0: 2.0174200007497944,
        0.4056097808810738, 0.2711450141698039.
      * j=3, n=5/heisenberg/beta=2.5: 2.015280801965751,
        0.35244942866801054, 0.21626447658505363.
      * j=4, n=5/heisenberg/beta=3.0: 2.0139012160408334,
        0.318580139850288, 0.1857723309913247.
      * j=5, n=5/ising/beta=0.3: 2.210289430693337,
        4.326837408413858, 1.1485233694806816.
      * j=6, n=5/ising/beta=1.0: 2.0815678906007546,
        2.1149387228583056, 0.7537246694779863.
      * j=7, n=5/ising/beta=2.0: 2.036979799614857,
        0.9093461788217551, 3.9857106663606574.
      * j=8, n=5/ising/beta=2.5: 2.0288925939467126,
        0.6965942637725109, 0.9122462802419744.
      * j=9, n=5/ising/beta=3.0: 2.024270134537068,
        0.577505162635687, 0.5431147872813216.
      * j=10, n=3/heisenberg/beta=1.0: 2.034573370225662,
        0.8461620574136219, 2.1855323080402544.
      * j=11, n=7/heisenberg/beta=1.0: 2.035502106794163,
        0.8707252177312027, 2.6762755090407286.

      scripts/check_validation_artifacts.py recomputes only the precision-floor ratio
      in `_quadratic_assessment_outcome`; it does not recompute coefficients, power-law
      slope, residuals, or passed from the raw arrays. It recomputes Richardson's first
      three raw points correctly, so tests must cover all later signed-response points
      and all all-window fits separately. It binds identity metrics and fit objects to
      stored measurement summaries rather than recomputing them. The diagnostic final
      local-energy profile used by `kms_density_match_error` is not retained, so that
      raw relationship cannot be independently recomputed from the JSON at all.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Required remediation remains under the existing ID. Recompute every fit and
      decision-bearing metric from the lowest-level retained operands, bind every
      duplicated alias, and fail closed on malformed/non-finite/shape/order/zero-
      denominator inputs. Retain the diagnostic final local-energy profile and
      recompute the KMS-density match. Add exhaustive independent mutations for every
      raw element and every alias combination, including all twelve sensitivity cases,
      all four fits, both sides of each identity, threshold crossings, and later
      Richardson-array points. Stale summaries and Booleans must never be authorities.

  - finding_id: "VIA000-R2-STARTUP-001"
    outcome: verified-resolved
    evidence: |-
      Independent focused execution covers persistent and self-cleaning .pth,
      sitecustomize.py, and usercustomize.py carriers; all are rejected before marker
      or child execution. Complete environment bytes are manifested and verified
      before and after each child, and Python runs under the base-owned `-I -S`
      bootstrap. Base-interpreter discovery covers Windows and POSIX symlinked virtual
      environments in the unit suite and passed in exact-SHA hosted Windows/Ubuntu
      execution. Malformed manifests, blocked environments, virtual bases, and child
      mutations fail closed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Host runner, base interpreter, operating-system loader, and GitHub-hosted runner
      integrity remain declared trust boundaries rather than measured candidate state.

  - finding_id: "VIA000-R2-RESIDUE-001"
    outcome: verified-resolved
    evidence: |-
      The frozen boundary uses a base-owned literal Git tree/blob manifest and hashes
      the complete executable environment. It rejects installed-package mutation,
      ignored executable bytecode, transient/restored tracked source, clean-filter and
      attribute attacks, assume-unchanged, skip-worktree, staged/index-only changes,
      deletions, renames, symlinks, mode changes, and ordinary/ignored path residue.
      Verification occurs before and after every child, and caches are outside the
      repository and measured environment. During re-review, an accidental local
      `.ruff_cache` plus ignored `__pycache__` residue was independently enumerated and
      rejected by `--enforce-change-boundary`; after explicit cleanup, the same exact
      command passed. Exact hosted artifacts bind 552 unique literal source entries,
      their modes/object IDs, the base commit/tree, and complete environment manifests.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      External operating-system caches and the declared trusted base are outside the
      candidate mutation boundary. This resolution does not extend scientific claims.

prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R2-VISUAL-LOCAL-001"
    outcome: verified-satisfied
    evidence: |-
      The focused suite replays compact, thin, dashed, rendered-text, real-annotation,
      and calibrated <=4/>4 raster variants. Retained exact-SHA Windows/Ubuntu rasters
      pass with observed maximum channel differences three and two respectively.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The requested meaningful-feature matrix and honest platform calibration are both present."

  - requested_test_id: "TST-VIA000-R2-SEMANTIC-MARGIN-001"
    outcome: unresolved
    evidence: |-
      The exact raw mutations and formulas listed under VIA000-R2-SEMANTIC-001 cross
      source-law and many-body gates while the current comparison/semantic oracles
      accept. The twelve mirrored sensitivity attacks demonstrate that check-copy
      binding alone is not raw recomputation. All cases belong to this existing test
      ID; creating a second ID would duplicate its operand/alias/margin scope.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Implement the listed paths as exhaustive negative controls, then extend the
      matrix mechanically over every raw array element, paired alias, summary field,
      and both sides of every threshold and identity.

  - requested_test_id: "TST-VIA000-R2-STARTUP-TRANSIENT-001"
    outcome: verified-satisfied
    evidence: |-
      All persistent, self-deleting, and byte-restoring variants of .pth,
      sitecustomize.py, and usercustomize.py are exercised under the actual trusted
      wrapper sequence and rejected pre-execution. Exact Windows/Ubuntu runs repeat
      the wrapper, site-disabled bootstrap, and final verification successfully.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Complete environment hashing and per-child checks cover package-level mutations separately."

  - requested_test_id: "TST-VIA000-R2-RESIDUE-INDEX-001"
    outcome: verified-satisfied
    evidence: |-
      The focused suite covers installed packages, ignored pyc, transient restored
      source, filters/attributes, assume-unchanged, skip-worktree, index/staged state,
      deletion, rename, symlink, mode, ignored/untracked paths, and per-child pre/post
      enforcement. The re-review's own ignored cache residue was rejected in practice.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The literal tree/blob and complete environment manifests close the prior control-plane gaps."

  - requested_test_id: "TST-VIA000-R2-CROSS-PLATFORM-001"
    outcome: unresolved
    evidence: |-
      Execution and retention are independently verified: ordinary CI run 32308061359
      and trusted boundary run 32308061345 both completed success at exact SHA
      3ceb92c37aa0d88706add080d458ca42c153524d. Boundary jobs passed 252 tests in
      393.77s on Windows and 107.64s on Ubuntu, regenerated all six examples, and
      completed pre/post boundary checks. Artifact 9385681991 has API digest
      sha256:67e18cae6a4b3c89f9ffed205549de5aca77ae4ba036f8999848f2472485dc7d;
      artifact 9385575556 has API digest
      sha256:57142561024ea8f0f1d89f6fde13d528e9c6d82dc1f22293eedea9e3aa001348.
      Downloaded bytes, sidecars, manifests, checker blob, candidate SHA/tree, and final
      status reconcile. Canonical environment-manifest SHA-256 values are
      722619045d50b876869762d91f265c631bf47283ca2e14b97ecea5a6400f399b on
      Windows and b93666d8db3e0c2a2562b6fd6de827e27fd6d966c4eff1c8e6d3f8c9c5aa60eb
      on Ubuntu, covering 21319 Windows and 22219 Ubuntu unique entries. Literal
      source-manifest values are
      197e9545c1b7c5e78a6cd3516b699f0ffeda7286ca87b9e6124fc9f451e3badf and
      d238a488997c0f75d0bf268ec779bd4bc7a3e3621ad160ba5a0b51408981b43c,
      each covering 552 unique Git-tree entries with modes and object IDs.

      The test remains unresolved only because the retained semantic negative-control
      matrix is incomplete and the same incomplete oracle accepts the raw attacks
      above. Green hosted execution cannot validate a missing predicate. The frozen
      governance also states that a formal R2 campaign/preregistered holdout has not
      yet been executed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      After semantic remediation, repeat both exact-SHA workflows, download and verify
      both bundles again, and run the exhaustive negative-control matrix before
      declaring this requested test satisfied.

predictions:
  experiment_id: "TST-VIA000-R2-BOUNDARY-ADVERSARIAL-REREVIEW-002"
  predicted_outcome: |-
    On commit 3ceb92c37aa0d88706add080d458ca42c153524d, every listed raw source-law
    and many-body mutation will continue to be accepted by the portability comparator
    and semantic checker when stored summary fields and Booleans remain stale; paired
    order-sensitivity raw aliases will remain accepted in all twelve cases. A complete
    remediation will reject each mutation while retaining the current exact-SHA
    Windows/Ubuntu evidence under unchanged calibrated visual/startup/residue rules.
  predicted_failure_mode: |-
    Without remediation, an R2 evidence bundle can carry lowest-level raw operands
    that falsify its fit, identity, locality, conservation, or control gate while the
    duplicated serialized metric and Boolean still report success.
  confidence_statement: |-
    High confidence for the implementation-boundary prediction because every listed
    bypass was executed against the exact frozen candidate and independently
    recomputed from the repository's own scientific functions or literal formulas.
    No inference is made about POPGP's physical mechanism or scientific viability.

recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    Changes requested. VIA000-R2-VISUAL-001, VIA000-R2-STARTUP-001, and
    VIA000-R2-RESIDUE-001 and their requested tests are independently resolved. The
    execution/retention portion of the cross-platform test is strong. Existing finding
    VIA000-R2-SEMANTIC-001 and requested test
    TST-VIA000-R2-SEMANTIC-MARGIN-001 remain blocking under direct raw-operand
    counterexamples, so the cross-platform test also cannot close. Recompute and bind
    every listed raw relationship, add exhaustive negative controls, rerun both exact-
    SHA workflows, and obtain another fresh independent re-review before merging or
    preregistering the R2 holdout. Approval here would concern only pre-holdout R2
    readiness; it would not be a scientific viability result or permission to reveal
    hidden data.
```
