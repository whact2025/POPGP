# VIA-000 R2 reproducibility-remediation independent re-review 4

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-4"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "unknown"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-via000-r2-remediation-independent-rereview-session-4"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-19"
commit_reviewed: "41571bbaba6b7f63f4cf234078dd4b351f8b6e8e"
baseline_commit: "d0381f0e8562c5f70c4e315dc3df4afd0d6bfbf4"
prior_review_ref: "072d7046e9802ca7db6695e64cb65a27c3e044b2:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-3.md"
builder_response_ref: "41571bbaba6b7f63f4cf234078dd4b351f8b6e8e:reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-3-RESPONSE-1.md"
context_hash: "f8b2341e14e9b988d5fa6553d006b92b40e49003"
context_hash_method: "git rev-parse \"41571bbaba6b7f63f4cf234078dd4b351f8b6e8e^{tree}\""
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
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-2.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-3.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-2-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-3-RESPONSE-1.md"
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
  - "tests/scientific/test_source_law_controls.py"
  - "tests/scientific/test_source_law_scaling.py"
  - "tests/scientific/test_topology_recovery.py"
  - "tests/unit/test_diagnostics.py"
  - "tests/unit/test_reproduction_boundary.py"
  - "tests/unit/test_validation_artifact_contract.py"
  - "tests/unit/test_review_guidance.py"
access_level: "public-repository-only plus public GitHub Actions metadata, logs, and downloadable artifacts"
independence_statement: |-
  This was a fresh independent-reviewer task in a new isolated worktree and branch
  created directly from exact handoff commit 41571bbaba6b7f63f4cf234078dd4b351f8b6e8e.
  Before conclusions, HEAD, tree f8b2341e14e9b988d5fa6553d006b92b40e49003,
  parent f274fee82a0dca6483974fb5f9f6a68d729fed1b, branch, source remote,
  ancestry, immutable review/response refs, and clean status were independently
  verified. The complete relevant baseline-to-handoff and remediation diffs,
  governance, launch documents, review chain, candidate source, tests, workflows,
  scientific documents, and retained results were inspected. Candidate code, docs,
  tests, workflows, and existing artifacts were not modified; only this review
  artifact was added.

  Builder statements, tests, workflow success, and downloaded artifacts were treated
  as hypotheses. Every REREVIEW-2/REREVIEW-3 semantic attack was independently
  replayed, followed by exhaustive simple and many-body raw-operand/alias sweeps,
  all twelve sensitivity cases, static/evolved decompositions, energy and KMS
  identities, fit families, local/global correlations, summary/Boolean staleness,
  malformed/nonfinite/shape/order/zero-denominator/type-confusion cases, and broader
  sign, scale, permutation, uniform-shift, and row/column transformations. No
  untracked handoff memo, custody material, sealed material, hidden label, secret
  seed, private evaluator, or restricted R2 material was read. R2 was unfrozen.

  The same human operator and Codex Desktop orchestrator are shared with the builder,
  while task/session, branch, and worktree are distinct. Exact served-model identities
  and snapshots were not exposed, so both remain unknown and model separation is
  false. This is internal adversarial process separation, not external scientific
  validation.

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
  Changes requested with one unresolved blocking finding. The latest remediation
  correctly closes the three exact REREVIEW-3 bypass families: every simple-source
  entropy-change element and the D=DeltaK-DeltaS identity, every evolved many-body
  local-profile cell versus its paired global energy, and the correlated uniform
  global-energy shift now reject. The prior four-fit, affine, solver-ratio,
  many-body raw/alias, static/evolved decomposition, KMS/first-law, endpoint,
  global-energy, quadratic/Richardson/all-window, twelve-case sensitivity,
  isospectral, commuting, density, clock/source, redshift, malformed, nonfinite,
  shape, order, denominator, stale-summary, stale-Boolean, and type controls were
  replayed.

  The semantic oracle remains incomplete for other retained lowest-level operands.
  In the grid report, adding 4e-9 to one raw pi_time.phi value is accepted as one
  bounded portability drift and produces zero semantic errors. Recomputed phi range
  becomes 4.00000016051907e-9, above the check's explicit 1e-12 bound, while stale
  phi_min, phi_max, phi_range, phi_mean, check value, and `passed: true` survive.
  All nine grid phi cells, all four chain phi cells, and all nine natural-gravity phi
  cells independently accept the same one-element mutation. Scaling all grid phi
  cells by 1e7 likewise passes comparison and semantics while recomputed range is
  2.7514450339362564e-9. Uniform shift, sign flip, reversal, and row/column roll
  transformations also demonstrate that raw arrays can diverge from their retained
  summaries without semantic rejection. This is a direct continuation of
  VIA000-R2-SEMANTIC-001 and its requested negative-control scope, not a distinct
  finding, so no new finding ID is introduced.

  Visual, startup, and residue resolutions remain verified. A clean external locked,
  no-editable authority run passed Ruff, TeX source validation, 278 tests in 1406.20s,
  a focused 115-test semantic/boundary/guidance suite in 125.95s, all six generators,
  the strengthened artifact/change-boundary checker, final environment verification,
  and clean normal/ignored status. Exact-SHA ordinary CI and Windows/Ubuntu boundary
  workflows are successful. Both retained bundles independently bind candidate
  SHA/tree/checker bytes, matching sidecars, unique manifests, allowed final status,
  semantic documents, rasters, and Git objects. Cross-platform execution and
  retention are verified, but the requested test cannot close while the same checker
  accepts a decision-falsifying raw report. R2 remains unfrozen. Local CUDA/nvcc
  absence is not a blocker because the frozen boundary does not require CUDA. No
  conclusion about scientific viability is made.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VIA000-R2-VISUAL-001"
    outcome: verified-resolved
    evidence: |-
      The maximum-per-channel comparator and targeted compact, thin, dashed, text,
      retained-annotation, <=4, and >4 controls passed independently. Both exact-SHA
      retained bundles pass all twelve raster comparisons; maximum channel difference
      is two on Ubuntu and three on Windows, below the enforced limit of four. No
      executed in-contract visual counterexample was found.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      This verifies the declared retained-raster boundary, not scientific image
      quality or invariance outside the frozen renderer contract.

  - finding_id: "VIA000-R2-SEMANTIC-001"
    outcome: unresolved
    evidence: |-
      Exact candidate 41571bbaba6b7f63f4cf234078dd4b351f8b6e8e rejects the named
      REREVIEW-2/REREVIEW-3 attacks over all simple-source arrays, every entropy-change
      element and D=DeltaK-DeltaS, all four fits, affine identity, both solver ratios,
      the named many-body raw-element/alias attacks, all evolved local-profile cells
      versus paired global energies, the correlated uniform global-energy shift, static/evolved
      decompositions, KMS/first-law identities, endpoint/global-energy statistics,
      quadratic/Richardson/all-window fits, all twelve sensitivity cases and aliases,
      isospectral, commuting, density, clock/source, redshift, and focused fail-closed
      malformed/nonfinite/shape/order/denominator/type cases.

      A fresh mechanically broadened replay found a surviving decision-bearing raw
      operand. Starting from the committed grid validation document, deep-copying it,
      applying `candidate["pipeline"]["pi_time"]["phi"][0] += 4e-9`, then running
      `compare_validation_documents(reference, candidate)` and
      `check_validation_semantics(candidate, relative_path)` yields comparison passed,
      zero comparison errors, one accepted numeric drift, and zero semantic errors.
      The serialized range remains 2.7514450339362562e-16 and the placeholder check
      remains true, but direct `numpy.ptp` over the mutated raw array is
      4.00000016051907e-9, falsifying its declared `phi_range < 1e-12` condition. The
      recomputed max and mean are 4.00000002231619e-9 and 4.444444444444445e-10 while
      their retained summaries remain stale.

      Repeating the one-element mutation accepts indices 0-8 for grid phi, indices
      0-3 for chain phi, and indices 0-8 for gravity phi_point. A scale-by-1e7 grid
      transformation accepts nine drifts with zero semantic errors and recomputed
      range 2.7514450339362564e-9. A uniform +4e-9 grid shift accepts nine drifts and
      zero semantic errors while the recomputed mean is 4e-9; sign flip, reversal,
      and type-preserving row and column rolls likewise pass with stale summaries.
      Thus generic numerical comparison is still being used as a semantic predicate
      for a retained raw array capable of contradicting a decision.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Bind every retained decision/descriptive raw array to all serialized summaries
      and checks derived from it, beginning with grid/chain/gravity potential arrays.
      Add exhaustive one-element plus scale/sign/permutation/uniform-shift and
      row/column correlation negative controls, then mechanically re-enumerate every
      retained raw operand. Generic portability tolerance must not substitute for a
      recomputed identity or threshold.

  - finding_id: "VIA000-R2-STARTUP-001"
    outcome: verified-resolved
    evidence: |-
      Persistent and self-cleaning .pth, sitecustomize.py, and usercustomize.py
      carriers remain rejected before marker or child execution. The base-owned
      `-I -S` bootstrap, complete environment manifest, Windows/POSIX interpreter
      discovery, per-child verification, and malformed/blocked/mutated-environment
      controls pass locally and on exact-SHA hosted Windows and Ubuntu. No executed
      in-contract counterexample was found.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The host runner, operating-system loader, base interpreter, and hosted runner
      remain declared trust boundaries rather than candidate-measured state.

  - finding_id: "VIA000-R2-RESIDUE-001"
    outcome: verified-resolved
    evidence: |-
      The literal Git tree/blob and complete environment boundaries reject installed-
      package mutation, ignored bytecode, transient/restored source, filters,
      attributes, assume-unchanged, skip-worktree, staged/index-only state, deletion,
      rename, symlink, mode changes, and ignored/untracked residue before and after
      children. The independent authority run used external caches and ended clean at
      exact SHA in normal and ignored status. Downloaded bundle source manifests each
      contain 556 unique entries with zero Git object/mode mismatches. No executed
      in-contract counterexample was found.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      External operating-system caches and the declared trusted base remain outside
      this mutation boundary. This result does not extend scientific claims.

prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R2-VISUAL-LOCAL-001"
    outcome: verified-satisfied
    evidence: |-
      Targeted compact, one-pixel, dashed, text, annotation, <=4, and >4 controls pass.
      Exact-SHA Ubuntu and Windows retained rasters pass, with maximum observed
      per-channel differences two and three respectively.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The meaningful-feature matrix and honest platform calibration are both retained."

  - requested_test_id: "TST-VIA000-R2-SEMANTIC-MARGIN-001"
    outcome: unresolved
    evidence: |-
      The handoff remediation rejects all named prior simple-source and many-body raw
      attacks, including the exact three REREVIEW-3 families. The fresh grid raw-phi
      one-element and scale attacks listed under VIA000-R2-SEMANTIC-001 are accepted
      by comparison and semantics while directly falsifying the serialized
      placeholder threshold. All grid, chain, and gravity potential-array elements
      also accept bounded one-element mutations with stale summaries.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Expand the exhaustive semantic matrix beyond the most recent remediation fields
      to every retained raw operand and every derived decision/descriptive alias.

  - requested_test_id: "TST-VIA000-R2-STARTUP-TRANSIENT-001"
    outcome: verified-satisfied
    evidence: |-
      Persistent, self-deleting, and byte-restoring startup carriers and package/
      environment mutations are rejected under the actual wrapper sequence locally
      and on exact-SHA hosted Windows/Ubuntu execution.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Pre/post child verification and complete environment hashing remain in force."

  - requested_test_id: "TST-VIA000-R2-RESIDUE-INDEX-001"
    outcome: verified-satisfied
    evidence: |-
      The focused suite covers installed packages, ignored pyc, transient restoration,
      filters/attributes, assume-unchanged, skip-worktree, index/staged state,
      deletion, rename, symlink, mode, ignored/untracked paths, and per-child pre/post
      enforcement. The disposable authoritative worktree ended clean at exact SHA.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The literal tree/blob and complete environment manifests close the prior control-plane gaps."

  - requested_test_id: "TST-VIA000-R2-CROSS-PLATFORM-001"
    outcome: unresolved
    evidence: |-
      Execution and retention are independently verified at exact candidate SHA.
      Ordinary CI run 32322949858 succeeded with 278 tests in 119.92s. Boundary run
      32322949812 succeeded with 278 tests in 111.91s on Ubuntu and 403.84s on Windows,
      regenerated all six examples, and passed the committed boundary checker.
      Downloaded Windows artifact 9390565380 has API digest
      sha256:e7d9c140c522ea8f51bf4a0c327e8ada3f6e5a6f655365b58fd63ae156cf49ee;
      Ubuntu artifact 9390483505 has API digest
      sha256:7eb3d9d4607c72073b7833fcaaeddf6086be6bde56632486b65472584dc677dc.

      Both bundles bind commit 41571bbaba6b7f63f4cf234078dd4b351f8b6e8e,
      tree f8b2341e14e9b988d5fa6553d006b92b40e49003, and checker SHA-256
      69e56aba44f77d766a6faa27ca6a8adfaba7825c4b8429ab1b0d9b07408b671c.
      Ubuntu/Windows environment manifests contain 22219/21319 unique entries with
      canonical SHA-256 values
      355b606fe1d8d673b5dcf2daf2ae9158711c65fb39a43bb1535df26342750e0a and
      eac98822049cbb91c466eb4248f215f1b5dacab91273df742f202e8bff2c0fdd.
      Their 556-entry source manifests have canonical SHA-256 values
      670c9d23edb6126b39401d0a3a63e861d011f086c0f2aa686127a52ecd42facc and
      265e6db8aadd6d74718d015908fc91552840750f42be125bee90dbc8bd9a25df.
      Sidecars match, paths are unique, Git object/mode mismatches are zero, and final
      status contains only declared generated artifacts (16 Ubuntu, 7 Windows). All
      semantic documents and twelve rasters pass; raw replay records 1240 Ubuntu and
      1153 Windows accepted numeric drifts with maximum raster difference two/three.

      This test remains unresolved only because the same exact checker accepts the
      decision-falsifying raw grid transformations documented above. Successful
      cross-platform execution cannot validate an absent semantic predicate. R2 is
      not frozen.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      After semantic remediation, repeat both exact-SHA workflows, independently
      reconcile both bundles, and rerun the exhaustive negative-control matrix.

predictions:
  experiment_id: "TST-VIA000-R2-BOUNDARY-ADVERSARIAL-REREVIEW-004"
  predicted_outcome: |-
    On commit 41571bbaba6b7f63f4cf234078dd4b351f8b6e8e, each listed grid,
    chain, and gravity potential-element mutation and the grid scale/uniform-shift/
    sign/permutation transformations will remain accepted when stale serialized
    summaries and checks are retained. A complete remediation will reject each
    contradiction while preserving the current visual, startup, residue, source-law,
    many-body, and hosted-platform results.
  predicted_failure_mode: |-
    Without remediation, an R2 bundle can retain a raw potential array whose directly
    recomputed range violates its decision criterion while stale summaries and a true
    Boolean cause both comparison and semantic validation to report success.
  confidence_statement: |-
    High confidence for the implementation-boundary prediction because the decisive
    one-element and scale bypasses were executed against the exact handoff and their
    threshold was independently recomputed from retained operands. No inference is
    made about POPGP's physical mechanism or viability.

recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    Changes requested. VIA000-R2-VISUAL-001, VIA000-R2-STARTUP-001, and
    VIA000-R2-RESIDUE-001 remain independently resolved, and the latest semantic
    remediation closes all exact REREVIEW-3 bypasses. VIA000-R2-SEMANTIC-001 and
    TST-VIA000-R2-SEMANTIC-MARGIN-001 remain blocking under a fresh lowest-level
    retained-operand counterexample; therefore the cross-platform requested test also
    cannot close. Bind every retained potential array to its summaries, checks, and
    solver relationships, broaden the exhaustive negative-control matrix to every
    retained raw operand, repeat both exact-SHA workflows, and obtain another fresh
    independent re-review before preregistering or freezing R2. This verdict concerns
    only pre-holdout R2 readiness. It is not a scientific viability result, merge
    permission by itself, or permission to reveal restricted data.
```
