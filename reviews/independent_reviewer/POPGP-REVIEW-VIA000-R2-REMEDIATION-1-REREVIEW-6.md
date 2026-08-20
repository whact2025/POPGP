# VIA-000 R2 reproducibility-remediation independent re-review 6

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-6"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "unknown"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-via000-r2-remediation-independent-rereview-session-6"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-20"
commit_reviewed: "5be3c38a0822d49953d0933f14ccab32ca12c896"
baseline_commit: "d0381f0e8562c5f70c4e315dc3df4afd0d6bfbf4"
prior_review_ref: "186a013b02a37ab13100b52431f876b8ea79188d:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-5.md"
builder_response_ref: "5be3c38a0822d49953d0933f14ccab32ca12c896:reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-5-RESPONSE-1.md"
context_hash: "6ad387f9f4e0bab7f97df1bb54a03177887f0707"
context_hash_method: "git rev-parse \"5be3c38a0822d49953d0933f14ccab32ca12c896^{tree}\""
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
  - "docs/scientific_hardening/CLAIMS_MATRIX.md"
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
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-4.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-5.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-2-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-3-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-4-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-5-RESPONSE-1.md"
  - "reviews/codex/POPGP-VIA000-R2-REMEDIATION-PLAN.md"
  - "scripts/check_reproduction_boundary.py"
  - "scripts/check_validation_artifacts.py"
  - "scripts/run_without_startup_hooks.py"
  - "scripts/check_tex.py"
  - "popgp/config.py"
  - "popgp/diagnostics.py"
  - "popgp/information.py"
  - "popgp/simulator.py"
  - "examples/physics_qg/ca_model/__main__.py"
  - "examples/physics_qg/ca_model/results/validation.json"
  - "examples/physics_qg/chain_1d/__main__.py"
  - "examples/physics_qg/chain_1d/results/validation.json"
  - "examples/physics_qg/gravity_well/__main__.py"
  - "examples/physics_qg/gravity_well/results/validation.json"
  - "examples/physics_qg/grid_2d/__main__.py"
  - "examples/physics_qg/grid_2d/results/validation.json"
  - "examples/physics_qg/source_law/__main__.py"
  - "examples/physics_qg/source_law/results/validation.json"
  - "examples/physics_qg/source_law_many_body/__main__.py"
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
  created directly from exact response-containing handoff commit
  5be3c38a0822d49953d0933f14ccab32ca12c896. Before review, HEAD, tree
  6ad387f9f4e0bab7f97df1bb54a03177887f0707, parent
  7046f48553f21e1474b7d2065b229c9b090e6a0d, source remote, branch, ancestry,
  immutable REREVIEW-5/RESPONSE-1 refs and blobs, and clean normal/ignored state were
  independently verified. The complete baseline-to-handoff and remediation/response
  diffs, governance, schemas, scientific plan and matrices, full initial review
  through REREVIEW-5 and RESPONSE-1 history, candidate checker, generators, tests,
  workflows, retained results, exact hosted logs, and retained bundles were inspected.
  Candidate code, docs, tests, workflows, and prior artifacts were not modified; only
  this review artifact was added to the reviewer branch.

  Builder statements, green tests, workflow success, and retained evidence were
  treated as hypotheses. Every exact REREVIEW-5 counterexample was replayed on all
  four retained clock records and then broadened across graph, source, solver,
  configuration, summaries, decisions, malformed values, and correlated transforms.
  A complete finite numeric-leaf perturbation sweep and the prior visual, startup,
  residue, and cross-platform attacks were also repeated. No untracked handoff memo,
  custody material, R2 packet, sealed material, hidden label, secret seed, private
  evaluator, or restricted R2 evidence was accessed. R2 remained unfrozen and
  unrevealed.

  The same human operator and Codex Desktop orchestrator are shared with the builder,
  while reviewer task/session, branch, and worktree are distinct. Exact served-model
  identities and snapshots were not exposed, so both remain unknown and model
  separation is false. This is internal adversarial process separation, not external
  scientific validation.

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
  Approve with zero blocking findings. The response closes the remaining graph,
  source, and solver-provenance continuation of VIA000-R2-SEMANTIC-001. On each of
  chain pi_time.phi, grid pi_time.phi, gravity pi_time_natural.phi, and gravity
  gravity_test.phi_point, the exact REREVIEW-5 diagonal/self-loop, asymmetric,
  symmetric, unsupported/new-edge, correlated weight/source, mu branch-crossing,
  stale-config-mu, source-background, and non-zero-sum transformations now produce
  semantic errors even when the generic numeric comparator accepts their in-tolerance
  leaves and their cached residuals are updated self-consistently.

  The checker now retains a finite, nonnegative, symmetric, zero-diagonal mutual-
  information matrix; canonical unique in-range inferred edges; raw source,
  background, zero-mode and normalization policy; and relevant configuration. It
  reconstructs the sparse weight matrix from retained MI and edge support, reconstructs
  the effective source, enforces applicable zero-sum and gauge invariants, binds mu,
  normalization, policy, and source model to authoritative configuration, and then
  recomputes the finite-graph equation. Diagnostic gravity additionally derives the
  point source from retained center/strength and derives graph distance from the
  inferred graph. The retained equation is therefore no longer merely a consistent
  relation among mutually mutable cached operands.

  Fresh broad attacks covered diagonal, negative, asymmetric, symmetric, new-edge,
  complete row and column, and correlated matrix/source transforms; duplicate,
  out-of-range, reversed, and disconnected edge structures; MI diagonal, negative,
  nonfinite, asymmetric, type, shape, support, extrema, and summary contradictions;
  raw/effective/background transforms; subtract_mean and require_zero_sum boundaries;
  the mu 0-to-positive solver branch; stale diagnostic configuration; gauge,
  normalization, source-model, and configuration mismatches; gravity center,
  strength, graph-distance and raw-point-source changes; and malformed/nonfinite/type/
  shape operands. A focused 40-case graph/MI/row/column/edge matrix rejected all 40
  composite attacks and all 40 semantically. Correlated transformations preserving
  residuals and cached summaries did not preserve a contradictory decision.

  A mechanical +4e-9 sweep covered all 2,652 finite float leaves across the six
  validation documents: 268 mutations were rejected by exact/composite comparison,
  2,163 additional comparator-accepted mutations were rejected semantically, and 221
  in-envelope descriptive or non-decision mutations were accepted. None of the 221
  changed a retained predicate. In particular, +4e-9 changes to connectivity gap/
  threshold summaries and a symmetric non-edge MI pair can remain within the declared
  numeric envelope while leaving inferred support and the separability decision
  unchanged. These are self-consistent in-tolerance alternatives, not stale or
  contradictory retained decisions. Exact unsupported-edge and threshold/predicate
  contradictions reject.

  In the isolated handoff worktree, a fresh external `uv sync --frozen --no-editable`
  succeeded. The exact documented commands `uv run --frozen --no-editable ruff check
  .`, `uv run --frozen --no-editable python scripts/check_tex.py`, and `uv run
  --frozen --no-editable python -m pytest -q -p no:cacheprovider` passed; pytest
  reported 366 passed in 865.45s. All six documented generators completed in order,
  and `uv run --frozen --no-editable python -m
  scripts.check_validation_artifacts --enforce-change-boundary` reported valid
  contracts and required visuals. Regeneration was byte-clean against the tracked
  handoff, and both normal and ignored Git status remained empty.

  Exact-SHA hosted evidence also reconciles. Fix CI 32393846039 and boundary
  32393846081 are successful at 7046f48553f21e1474b7d2065b229c9b090e6a0d;
  response-bound CI 32395141975 and boundary 32395142030 are successful at exact
  handoff 5be3c38a0822d49953d0933f14ccab32ca12c896. All four downloaded boundary
  bundles bind the expected commit/tree and literal checker blob; their source and
  environment sidecars hash correctly; all paths are unique; all 559 fix and 560
  response source entries match exact Git object IDs and modes; status contains only
  the 16 Ubuntu or seven Windows declared generated paths; all six strict JSON
  documents pass semantic and composite checks; and all twelve visuals pass.
  Independently measured Ubuntu/Windows maximum raster delta is three channels in
  gravity_well/source_comparison.png. Across the four retained clock records the
  maximum raw-field delta is 8.881784197001252e-16; a derived phi_range subtraction
  reaches 1.0408340855860843e-15, also within the registered composite envelope.

  The historical visual, semantic, startup, residue, semantic-margin, and hosted
  cross-platform obligations are therefore independently satisfied for this public
  pre-holdout contract. R2 remains unfrozen and unrevealed. This review does not
  establish POPGP scientific viability, supply merge permission by itself, validate
  private evidence, or authorize a holdout reveal.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VIA000-R2-VISUAL-001"
    outcome: verified-resolved
    evidence: |-
      The compact/thin/dashed/text/annotation/structured-corruption controls remain
      fail-closed while calibrated in-bound raster drift remains accepted. All twelve
      visuals in both fix and response-bound Ubuntu/Windows bundles satisfy the exact
      commit's semantic envelope. The independently recomputed maximum cross-platform
      per-channel difference is three in gravity_well/source_comparison.png.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      This verifies the declared retained-raster contract, not scientific image
      quality or renderer behavior outside the locked boundary.

  - finding_id: "VIA000-R2-SEMANTIC-001"
    outcome: verified-resolved
    evidence: |-
      Every exact REREVIEW-5 graph/source/config counterexample now rejects on all four
      retained clock records. Diagonal self-loops, negative weights, asymmetric and
      symmetric changes, unsupported/new edges, row/column transforms, and correlated
      W/source updates fail because weight_matrix is reconstructed from finite,
      symmetric, nonnegative, zero-diagonal MI plus canonical inferred-edge support.
      Duplicate, out-of-range, reversed, disconnected, malformed, negative,
      nonfinite, asymmetric, and shape/type graph evidence also rejects.

      Raw density/source, background, zero-mode policy, effective source, normalize
      flag, source model, and mu are retained and cross-bound to authoritative config.
      Effective source, applicable zero-sum behavior, equation residual, and gauge are
      independently recomputed. Mu 0-to-positive branch changes, stale gravity mu,
      raw/effective/background changes, non-zero-sum correlated sources, stale policy,
      gauge, normalize and source-model changes reject. Gravity's center, strength,
      raw point source, graph distances, radial profile, log fit, and redshift are
      derived or rebound. A fresh 40-case broad matrix and the exact response tests
      rejected every attack semantically; no residual-preserving contradictory
      decision survived.

      The 2,652-leaf sweep accepted only 221 descriptive/non-decision mutations that
      stayed in the registered numeric envelope and did not change a predicate. Small
      self-consistent non-edge MI and connectivity-summary drift with unchanged edge/
      separability decisions was explicitly distinguished from a stale decision.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Optional future defense-in-depth is to retain the exact adaptive-gap selection
      inputs and recompute connectivity threshold/gap summaries directly from MI. It
      is not blocking here: the composite contract binds stable inputs exactly,
      contradictory support/predicates fail, and observed accepted changes preserve
      the retained decision within its declared tolerance.

  - finding_id: "VIA000-R2-STARTUP-001"
    outcome: verified-resolved
    evidence: |-
      The complete installed environment and literal candidate tree are checked before
      and after each trusted child. Persistent, self-deleting, byte-restoring, package,
      startup-hook, blocked-environment, malformed-manifest, and interpreter-boundary
      controls pass in the 366-test local suite and both exact-SHA hosted platforms.
      The literal checker copied from the candidate Git blob is identical across all
      four downloaded bundles.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Host runner, base interpreter, operating-system loader, and declared public
      GitHub Actions infrastructure remain explicit trust boundaries.

  - finding_id: "VIA000-R2-RESIDUE-001"
    outcome: verified-resolved
    evidence: |-
      Filters/attributes, assume-unchanged, skip-worktree, staged/index state,
      deletion, rename, symlink/mode substitution, ignored bytecode, untracked state,
      installed-package mutation, and transient/restored bytes remain fail-closed.
      Downloaded manifests contain 559 unique source entries for the fix commit and
      560 for the response commit on each platform, with zero Git object/mode
      mismatches. Local regeneration remained byte-clean; final normal and ignored
      status were empty.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Platform CRLF worktree materialization is kept distinct from authoritative Git
      blob and mode binding.

prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R2-VISUAL-LOCAL-001"
    outcome: verified-satisfied
    evidence: |-
      Targeted compact, one-pixel, dashed, thin-feature, text, annotation, calibrated
      in-bound, and over-bound controls pass. Exact-SHA fix and response bundles each
      retain twelve accepted visuals on Ubuntu and Windows; maximum cross-platform
      per-channel delta is three.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The registered meaningful-feature and hosted-platform matrix remains retained."

  - requested_test_id: "TST-VIA000-R2-SEMANTIC-MARGIN-001"
    outcome: verified-satisfied
    evidence: |-
      The calibrated potential-moment and raw-potential transformations remain
      fail-closed. The expanded provenance controls cover all four clock records and
      reject diagonal, negative, asymmetric, symmetric, unsupported/new edge,
      row/column, correlated W/source, graph-evidence, raw/effective/background,
      zero-mode, normalize/gauge, source-model, mu/config, gravity source/distance,
      malformed/type/shape/nonfinite, and cached-summary transforms. The independent
      2,652-leaf sweep found no accepted decision-changing mutation.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      In-envelope numeric alternatives with unchanged predicates were not mislabeled
      as stale decisions.

  - requested_test_id: "TST-VIA000-R2-STARTUP-TRANSIENT-001"
    outcome: verified-satisfied
    evidence: |-
      Persistent, self-deleting, byte-restoring, installed-package, and startup-hook
      carriers are rejected under the actual pre/post wrapper sequence. The fresh
      locked environment, local 366-test execution, and both hosted operating systems
      independently preserve the complete-environment manifest.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The isolated no-site bootstrap and per-child environment verification remain active."

  - requested_test_id: "TST-VIA000-R2-RESIDUE-INDEX-001"
    outcome: verified-satisfied
    evidence: |-
      The authoritative suite exercises mutable-index flags, filters/attributes,
      tracked worktree bytes and modes, staged/index-only state, symlink, deletion,
      rename, ignored/untracked residue, executable bytecode, and transient
      restoration. All four source manifests have unique paths and exact Git object/
      mode bindings; local normal and ignored status are clean after regeneration.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The literal Git-object and complete-environment boundary remains closed."

  - requested_test_id: "TST-VIA000-R2-CROSS-PLATFORM-001"
    outcome: verified-satisfied
    evidence: |-
      Fix CI 32393846039 and boundary 32393846081 succeeded at 7046f48553f21e1474b7d2065b229c9b090e6a0d.
      Fix artifacts are Ubuntu 9416081855 with API digest
      sha256:3cffc0c3c66bd1a05fbd9e3714cca337a18fececbfa4776895a47e2a8ad48d05
      and Windows 9416187269 with API digest
      sha256:bda5ef90f03c93527861dad1c5ebb6cd2e981b52b01a3331f093d44af12a615d.
      Response CI 32395141975 and boundary 32395142030 succeeded at exact handoff
      5be3c38a0822d49953d0933f14ccab32ca12c896. Response artifacts are Ubuntu
      9416537999 with API digest
      sha256:484f52390a6f8c36b8fe9b06537c792e2175625105f539fbb394b5c779ad862d
      and Windows 9416685053 with API digest
      sha256:37da899b2bf77f29c18cc0dfa545782e9c7a181be3338df2134872071a6a63c5.

      All four bundles bind their exact commit/tree and checker blob SHA-256
      69e56aba44f77d766a6faa27ca6a8adfaba7825c4b8429ab1b0d9b07408b671c.
      Fix source-manifest SHA-256 values are Ubuntu
      24488f1fc930ecfe8b5515fc6eb34fcededc3f59cf79b7b9cfd0606bf093a10f
      and Windows e0100bc1e4dca3a16d28deec03b9ca404158c5ad3752e40e61ef14e8d9497593;
      response values are Ubuntu
      613f90c727d8a7af988bfa66bdd3b049fb67c33db79abc5869fe0441d4575aa1
      and Windows 3393b5ae97f4ac5f6c54803a374fbf9e9bb0e570894cf995cf9613c51ebf2657.
      Environment-manifest values are respectively
      55ed0e8cdbee2528b6babfdd18c68e8dc17e096d053538af0e1d6de40befa0f8,
      9233fadfead5d0d85b2d72e7b64777d1d217b73d54393597e8a4d4bd408a063c,
      e9dd856c060a08e00d39c94b37584783e4d7ec08cf1116c7a3e756adf2e67a89,
      and afa6ced194ce64d32399d704094302f80678c338dc0a367ae966f690117e140a.
      Sidecars, 559/560 unique Git-bound entries, six JSON semantics, twelve visuals,
      and generated-only status reconcile. Raw clock delta is at most
      8.881784197001252e-16 and raster delta at most three.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      This satisfies the public pre-holdout cross-platform execution and retention
      obligation only; it does not validate a private holdout or scientific claim.

predictions:
  experiment_id: "TST-VIA000-R2-BOUNDARY-ADVERSARIAL-REREVIEW-006"
  predicted_outcome: |-
    At exact commit 5be3c38a0822d49953d0933f14ccab32ca12c896, repeated graph,
    source, solver, configuration, raw-potential, visual, startup, residue, and hosted-
    platform negative controls will continue to reject contradictory decisions while
    accepting only registered in-envelope platform drift. The public R2 evidence
    boundary can proceed to its next governed pre-freeze step if all other plan
    prerequisites remain satisfied.
  predicted_failure_mode: |-
    A future change that stops deriving weight/source/configuration identities or
    weakens exact input, zero-sum, gauge, manifest, or raster binding would reopen a
    historical finding even if cached residuals and CI stayed green.
  confidence_statement: |-
    High confidence for this public implementation-boundary prediction because exact
    prior attacks, a broad 40-case matrix, all finite numeric leaves, the full 366-test
    suite, clean regeneration, and both exact hosted operating systems were executed.
    No inference is made about POPGP's physical mechanism or scientific viability.

recommendation:
  approve: true
  blocking_findings: 0
  rationale: |-
    Approve this VIA-000 R2 reproducibility remediation for the public pre-holdout
    contract. All four historical findings are independently verified resolved and
    all five historical requested tests are independently verified satisfied. The
    remaining REREVIEW-5 graph/source/configuration attacks reject across all four
    retained clock records; no new failure class was found; the authoritative local
    sequence and exact-SHA Ubuntu/Windows evidence reconcile with clean boundaries.
    R2 remains unfrozen and unrevealed, so any freeze, preregistration, holdout
    execution, or reveal must still follow the governed viability plan. This verdict
    is not a scientific viability result, external validation, merge permission by
    itself, or authorization to access restricted material.
```
