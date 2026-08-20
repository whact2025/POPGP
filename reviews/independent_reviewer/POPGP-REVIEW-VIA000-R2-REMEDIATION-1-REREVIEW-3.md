# VIA-000 R2 reproducibility-remediation independent re-review 3

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-3"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "unknown"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-via000-r2-remediation-independent-rereview-session-3"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-19"
commit_reviewed: "a44493610052cdb7513e5a80c37a27648cb1f45d"
baseline_commit: "d0381f0e8562c5f70c4e315dc3df4afd0d6bfbf4"
prior_review_ref: "8c3447d7f5d23e030e48097b12a318c88ee40f53:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-2.md"
builder_response_ref: "a44493610052cdb7513e5a80c37a27648cb1f45d:reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-2-RESPONSE-1.md"
context_hash: "adfbe0321d2124831ab3fabcd53eaf9491aa8653"
context_hash_method: "git rev-parse \"a44493610052cdb7513e5a80c37a27648cb1f45d^{tree}\""
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
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-2-RESPONSE-1.md"
  - "reviews/codex/POPGP-VIA000-R2-REMEDIATION-PLAN.md"
  - "scripts/check_reproduction_boundary.py"
  - "scripts/check_validation_artifacts.py"
  - "scripts/run_without_startup_hooks.py"
  - "scripts/check_tex.py"
  - "popgp/diagnostics.py"
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
  created directly from exact handoff commit a44493610052cdb7513e5a80c37a27648cb1f45d.
  Before conclusions, HEAD, tree adfbe0321d2124831ab3fabcd53eaf9491aa8653,
  branch, source remote, ancestry, immutable review/response refs, and clean status
  were independently verified. The complete relevant baseline-to-handoff and
  semantic-remediation diffs, governance, review chain, candidate source, tests,
  workflows, scientific documents, and retained results were inspected. Candidate
  code, docs, tests, workflows, and existing artifacts were not modified; only this
  review artifact was added.

  Builder statements, tests, workflow success, and downloaded artifacts were treated
  as hypotheses. Named counterexamples and broader operand, alias, threshold,
  nonfinite, malformed, shape, order, zero-denominator, stale-summary, stale-Boolean,
  and type-confusion cases were independently exercised. No untracked handoff memo,
  custody material, sealed material, hidden label, secret seed, private evaluator, or
  restricted R2 material was read. The same human operator and Codex Desktop
  orchestrator are shared with the builder, while task/session, branch, and worktree
  are distinct. Exact served-model identities and snapshots were not exposed, so both
  remain unknown and model separation is false. This is internal adversarial process
  separation, not external scientific validation.

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
  materially improves semantic raw binding: all 64 single-element perturbations over
  the four simple-source fit arrays, 154 of 172 many-body raw-element perturbations,
  and all 168 order-sensitivity perturbations across twelve cases, both raw aliases,
  and seven response elements are rejected after the portability comparator accepts
  the bounded drift. Named prior affine, solver-ratio, quadratic-assessment, signed-
  response, Richardson, identity, energy, endpoint, isospectral, commuting, and KMS-
  density attacks and the focused malformed/nonfinite/boundary matrix were replayed.

  The semantic oracle is nevertheless incomplete at a still-retained lower level.
  Changing one evolved local-energy-profile interior value by 4e-9 is accepted by
  comparison and semantic validation even though its sum no longer equals the paired
  evolved global energy. The same bypass applies to 18 of 172 enumerated many-body
  elements. Correlating five +4e-9 global-energy mutations keeps the serialized
  conservation metric true while breaking every evolved profile/global-energy
  relationship. Separately, changing simple-source entropy_change[0] by 4e-9 leaves
  the raw identity D=DeltaK-DeltaS false while both validators accept. These are direct
  continuations of VIA000-R2-SEMANTIC-001 and its requested negative-control test, so
  no duplicate ID is introduced.

  Visual, startup, and residue resolutions remain verified. The clean local authority
  run passed Ruff, TeX source validation, all six documented generators, the
  strengthened artifact checker, review-guidance checks, and 276 tests in 1339.05s
  (0:22:19). The focused semantic/boundary suite passed 113 tests in 136.59s. Exact-
  SHA ordinary CI and Windows/Ubuntu boundary workflows are successful, and both
  retained bundles independently bind the candidate SHA/tree, checker bytes,
  sidecars, unique manifests, allowed final status, semantic/visual results, and Git
  objects. Cross-platform execution and retention are verified, but that requested
  test cannot close while a decision-bearing semantic predicate remains absent. R2
  remains unfrozen. Lack of local CUDA/nvcc is not a blocker because this boundary
  does not require CUDA. No conclusion about scientific viability is made.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VIA000-R2-VISUAL-001"
    outcome: verified-resolved
    evidence: |-
      The calibrated maximum-per-channel comparator and targeted compact, thin,
      dashed, text, retained-annotation, and <=4/>4 controls passed independently.
      Both exact-SHA retained bundles pass all raster comparisons; Ubuntu matches the
      committed raster bytes at maximum pixel delta zero, while the largest Windows
      delta is three in gravity-well source_comparison.png, below the limit of four.
      No executed in-contract counterexample was found.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      This verifies the declared retained-raster boundary, not scientific image
      quality or invariance outside the frozen renderer contract.

  - finding_id: "VIA000-R2-SEMANTIC-001"
    outcome: unresolved
    evidence: |-
      The original REREVIEW-2 raw attacks were replayed against exact candidate
      a44493610052cdb7513e5a80c37a27648cb1f45d. The new implementation rejects every
      one-element +4e-9 mutation in the four simple-source fit operands (64/64), all
      168 order-sensitivity element/alias cases (twelve cases, two aliases, seven
      elements), and 154/172 enumerated many-body operands. It also rejects the named
      prior affine identity, both solver-ratio, global-energy conservation, initial
      endpoint, isospectral, commuting, KMS/first-law/local-decomposition, signed-
      response, Richardson, all-window, and quadratic-assessment counterexamples.
      Focused fail-closed controls for threshold crossings, nonfinite numbers,
      malformed structures, shapes, ordering, zero denominators, stale summaries,
      stale Booleans, and type confusion pass.

      Three independently executed in-contract attacks still survive. First, adding
      4e-9 to
      $.measurements.evolved_local_energy_profiles[1][0] in the many-body report
      produces one accepted bounded drift and zero semantic errors. The stored
      local_energy_decomposition_consistency_and_spreading result and generator drift
      remain passing/zero, while recomputation gives
      abs(sum(evolved_local_energy_profiles[1])-evolved_total_energy[1]) =
      4.000000330961484e-9, above the declared 1e-12 consistency boundary. Exhaustive
      enumeration finds 18 accepted elements: row 0 interior columns 1, 2, and 3, and
      all five columns of rows 1, 2, and 4. Only row 0 endpoints and all of row 3 are
      covered by the committed negative controls.

      Second, adding 4e-9 to all five evolved_total_energy values produces five
      accepted drifts and zero semantic errors. Their peak-to-peak value remains zero,
      so the serialized conservation Boolean stays true, but all five independently
      retained evolved profile/global-total identities are false; the maximum mismatch
      is 3.999999997894577e-9.

      Third, adding 4e-9 to
      examples/physics_qg/source_law/results/validation.json
      $.measurements.entropy_change[0] produces one accepted drift and zero semantic
      errors. Recomputing D = DeltaK - DeltaS gives a maximum violation of
      3.999999959335096e-9, versus 1.079594313252441e-16 in the clean report. Thus a
      lowest-level retained operand can still contradict a scientific identity while
      comparison and semantic validation accept.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Bind every evolved local profile sum to its corresponding evolved global total,
      bind the simple-source D=DeltaK-DeltaS identity directly to all raw operands,
      and add exhaustive one-element plus correlated/global-shift negative controls.
      A check that only recomputes global peak-to-peak conservation cannot establish
      per-time local/global decomposition consistency.

  - finding_id: "VIA000-R2-STARTUP-001"
    outcome: verified-resolved
    evidence: |-
      Focused replay verifies rejection of persistent and self-cleaning .pth,
      sitecustomize.py, and usercustomize.py carriers before marker or child execution.
      The base-owned `-I -S` bootstrap, complete environment manifest, Windows/POSIX
      base-interpreter discovery, per-child verification, and malformed/blocked/
      mutated-environment fail-closed cases pass locally and on exact-SHA hosted
      Windows and Ubuntu. No executed in-contract counterexample was found.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The host runner, operating-system loader, base interpreter, and hosted runner
      remain declared trust boundaries rather than candidate-measured state.

  - finding_id: "VIA000-R2-RESIDUE-001"
    outcome: verified-resolved
    evidence: |-
      The literal Git tree/blob and complete executable-environment boundaries reject
      installed-package mutation, ignored bytecode, transient/restored source,
      filters/attributes, assume-unchanged, skip-worktree, staged/index-only state,
      deletion, rename, symlink, mode changes, and ignored/untracked residue before and
      after every child. Review-created .venv and __pycache__ residue were quarantined
      outside the candidate and the authoritative disposable candidate completed with
      empty exact-SHA status. Downloaded bundle source manifests contain 554 unique
      entries each with no object/mode mismatch. No executed in-contract
      counterexample was found.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      External operating-system caches and the declared trusted base remain outside
      this mutation boundary. This result does not extend scientific claims.

prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R2-VISUAL-LOCAL-001"
    outcome: verified-satisfied
    evidence: |-
      Targeted compact, one-pixel, dashed, text, annotation, and <=4/>4 controls pass.
      Exact-SHA Ubuntu and Windows retained rasters pass, with maximum observed
      per-channel deltas zero and three respectively.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The meaningful-feature matrix and honest platform calibration are both retained."

  - requested_test_id: "TST-VIA000-R2-SEMANTIC-MARGIN-001"
    outcome: unresolved
    evidence: |-
      The remediation's exhaustive matrix now rejects 64/64 simple fit elements,
      154/172 many-body elements, and 168/168 order-sensitivity element/alias cases,
      plus the named malformed and threshold controls. The 18 evolved-profile bypasses,
      correlated five-element global shift, and simple entropy-change identity attack
      listed under VIA000-R2-SEMANTIC-001 are accepted by both current validators.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Add the missing one-element, correlated/global-shift, and lowest-level identity
      controls, then mechanically re-enumerate all retained raw operands and aliases.

  - requested_test_id: "TST-VIA000-R2-STARTUP-TRANSIENT-001"
    outcome: verified-satisfied
    evidence: |-
      Persistent, self-deleting, and byte-restoring .pth/sitecustomize/usercustomize
      variants and package/environment mutations are rejected under the actual
      wrapper sequence locally and on exact-SHA hosted Windows/Ubuntu execution.
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
      Ordinary CI run 32317083415 succeeded with 276 tests in 120.56s. Boundary run
      32317083423 succeeded with 276 tests in 84.75s on Ubuntu and 436.76s on Windows,
      regenerated all six examples, and passed the boundary checker. Downloaded
      artifact 9388593509 has API digest
      sha256:e9001e33f3e9b54f30ed3a3b13269ae71d042b8269b97efea9198a33ea95302b;
      artifact 9388698878 has API digest
      sha256:676c7e398567dade4ac5d4bad300ec6c3cdc9275060056961a7f716b0ef0d7db.

      Both bundles bind commit a44493610052cdb7513e5a80c37a27648cb1f45d,
      tree adfbe0321d2124831ab3fabcd53eaf9491aa8653, and checker SHA-256
      69e56aba44f77d766a6faa27ca6a8adfaba7825c4b8429ab1b0d9b07408b671c.
      The Ubuntu environment manifest has 22219 unique entries and canonical SHA-256
      f48e90683b46702816d35f6da15ce08c8c28506e90d909aa66d7f81944a3d6d3;
      Windows has 21319 and
      f05afa5e1fb6bd0048c128386e53f60a95c44e71da1c7a5a50734db115980986.
      The two literal source manifests contain 554 unique entries with SHA-256 values
      7f8a08c37b75378a0aa1b1e65c40fb61707edd0519bcf497573b17afc4d51dd3
      and 1854ee73cf4e9aeb7ed6436486c6e1ceca9ab90d0f88dc866d28df1f76f86a0c.
      There are zero Git object/mode mismatches; final status contains only the allowed
      declared artifacts (16 Ubuntu, 7 Windows). All six reports and rasters pass;
      raw bundle replay reports 1324 Ubuntu and 1152 Windows accepted bounded numeric
      drifts and maximum raster difference three.

      This test remains unresolved only because the same retained semantic oracle
      accepts the raw contradictions documented above. Successful cross-platform
      execution cannot validate an absent semantic predicate. R2 is not frozen.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      After semantic remediation, repeat both exact-SHA workflows, independently
      reconcile both bundles, and rerun the exhaustive negative-control matrix.

predictions:
  experiment_id: "TST-VIA000-R2-BOUNDARY-ADVERSARIAL-REREVIEW-003"
  predicted_outcome: |-
    On commit a44493610052cdb7513e5a80c37a27648cb1f45d, each of the 18 listed
    evolved-profile element changes, the correlated five-element global-energy shift,
    and the simple entropy_change[0] mutation will remain accepted when stale
    serialized checks are retained. A complete remediation will reject each mutation
    while preserving the current visual, startup, residue, and hosted-platform results.
  predicted_failure_mode: |-
    Without remediation, an R2 bundle can retain local profiles and global totals that
    contradict one another, or raw D/DeltaK/DeltaS operands that violate their identity,
    while comparison and semantic validation both report success.
  confidence_statement: |-
    High confidence for the implementation-boundary prediction because every bypass
    was executed against the exact handoff and independently recomputed from retained
    operands. No inference is made about POPGP's physical mechanism or viability.

recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    Changes requested. VIA000-R2-VISUAL-001, VIA000-R2-STARTUP-001, and
    VIA000-R2-RESIDUE-001 remain independently resolved, and the latest semantic
    remediation closes most prior bypasses. VIA000-R2-SEMANTIC-001 and
    TST-VIA000-R2-SEMANTIC-MARGIN-001 remain blocking under direct lowest-level
    retained-operand counterexamples; therefore the cross-platform requested test
    also cannot close. Bind every evolved local profile to its paired global total and
    the simple D=DeltaK-DeltaS identity to raw operands, add exhaustive negative
    controls, repeat both exact-SHA workflows, and obtain another fresh independent
    re-review before merging or preregistering a holdout. This verdict concerns only
    pre-holdout R2 readiness. It is not a scientific viability result, merging
    permission by itself, or permission to reveal restricted data.
```
