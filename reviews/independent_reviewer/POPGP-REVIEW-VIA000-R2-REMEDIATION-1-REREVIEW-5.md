# VIA-000 R2 reproducibility-remediation independent re-review 5

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-5"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "unknown"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-via000-r2-remediation-independent-rereview-session-5"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-20"
commit_reviewed: "7b05ea3b4bd1ba8923a4477ea64852512b375799"
baseline_commit: "d0381f0e8562c5f70c4e315dc3df4afd0d6bfbf4"
prior_review_ref: "baf4401ddc6f7d691554913aeb4622f1390dd66c:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-4.md"
builder_response_ref: "7b05ea3b4bd1ba8923a4477ea64852512b375799:reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-4-RESPONSE-1.md"
context_hash: "8ed3672179b4766986872e1125611835dfdf4e18"
context_hash_method: "git rev-parse \"7b05ea3b4bd1ba8923a4477ea64852512b375799^{tree}\""
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
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-4.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-2-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-3-RESPONSE-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-4-RESPONSE-1.md"
  - "reviews/codex/POPGP-VIA000-R2-REMEDIATION-PLAN.md"
  - "scripts/check_reproduction_boundary.py"
  - "scripts/check_validation_artifacts.py"
  - "scripts/run_without_startup_hooks.py"
  - "scripts/check_tex.py"
  - "popgp/diagnostics.py"
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
  created directly from exact handoff commit 7b05ea3b4bd1ba8923a4477ea64852512b375799.
  Before review, HEAD, tree 8ed3672179b4766986872e1125611835dfdf4e18,
  parent 09bce1e7f705133a6b2f02a85b47db84332bd65f, source remote, branch,
  ancestry, immutable review/response refs, and clean normal/ignored state were
  independently verified. The complete baseline-to-handoff and REREVIEW-4-to-response
  diffs, governance, review chain, candidate source, tests, workflows, scientific
  documents, retained results, exact hosted logs, and retained bundles were inspected.
  Candidate code, docs, tests, workflows, and prior artifacts were not modified; only
  this review artifact was added to the reviewer branch.

  Builder statements, tests, workflow success, and downloaded artifacts were treated
  as hypotheses. All exact REREVIEW-4 raw-potential transformations were replayed and
  broadened across retained numeric operands, solver aliases, graph matrices, sources,
  mass, gauge policy, summaries, checks, shapes, ordering, finiteness, and types. The
  finite-graph equation and index moment were independently recomputed. Visual,
  startup, residue, hosted-platform, manifest, Git-object/mode, and environment
  boundaries were also replayed. No untracked handoff memo, custody material, R2
  packet, sealed material, hidden label, secret seed, private evaluator, or restricted
  R2 evidence was accessed. R2 remained unfrozen and unrevealed.

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
  Changes requested with one unresolved blocking historical finding. The remediation
  correctly closes the exact REREVIEW-4 raw-potential attacks: one-element +4e-9,
  1e7 scale, sign, reversal, roll/permutation, uniform shift, shaped row/column
  transforms, stale summaries/check aliases, and correlated summary updates reject
  across chain, grid, natural-gravity, and diagnostic-gravity potentials. The
  dedicated normalized index-moment policy independently accepts the previously
  observed 5.49e-19 Linux recomputation delta and rejects the registered 4.46e-18
  transformation; serialized probes at and just around 1e-18 fail closed, and the
  largest alternate-order recomputation difference observed was 8.47e-20.

  The new retained finite-graph equation is nevertheless self-consistent without
  being bound to the graph, source, and solver policy that produced it. On the exact
  handoff, changing any retained solver matrix diagonal by +4e-9 passes comparison
  and semantics for chain pi_time.phi, grid pi_time.phi, gravity pi_time_natural.phi,
  and gravity gravity_test.phi_point. A diagonal self-loop cancels identically from
  diag(row_sum)-W, but contradicts the graph's required zero diagonal. Asymmetric and
  symmetric off-diagonal changes, including addition of a previously absent edge,
  also pass when effective_source is updated by the exact induced L*phi change.
  These documents preserve a small recomputed residual while contradicting symmetry,
  edge support, and the retained/inferred graph meaning.

  Solver-policy contradictions pass independently as well. Changing an unscreened
  retained mu from 0 to 4e-9 passes although the implementation changes from the
  augmented zero-mean gauge solve to a screened solve. Gravity diagnostic mu can be
  changed with a correlated effective-source update while stable config.mu remains
  0.1; source_background can be changed alone. Non-zero-sum effective sources for
  zero-mode-subtracted solves pass when residual/check aliases are recomputed within
  their retained margins. The checker validates only the retained equation and does
  not bind weight_matrix to inferred graph structure, effective_source to raw source
  plus background/zero-mode policy, or solver mu to retained configuration. This is
  a direct continuation of VIA000-R2-SEMANTIC-001, not a separate failure class, so
  no new finding ID is created.

  A mechanical +4e-9 sweep covered 2,432 finite numeric leaves across all six reports.
  The semantic oracle rejected the decision-bearing source-law and many-body attack
  families already registered, while the solver sweep exposed the provenance gap
  above. Malformed/nonfinite/shape/negative-mu/numeric-Boolean attacks fail closed.
  In a disposable detached worktree at the exact handoff, the authoritative command
  sequence `uv sync --frozen --no-editable`, `uv run --frozen --no-editable ruff
  check .`, `uv run --frozen --no-editable python scripts/check_tex.py`, and `uv run
  --frozen --no-editable python -m pytest -q -p no:cacheprovider` passed; pytest
  reported 295 passed in 919.40s. The exact CI generator commands `uv run --frozen
  --no-editable python -m examples.physics_qg.chain_1d`, `uv run --frozen
  --no-editable python -m examples.physics_qg.grid_2d`, `uv run --frozen
  --no-editable python -m examples.physics_qg.gravity_well`, `uv run --frozen
  --no-editable python -m examples.physics_qg.source_law`, `uv run --frozen
  --no-editable python -m examples.physics_qg.source_law_many_body`, and `uv run
  --frozen --no-editable python -m examples.physics_qg.ca_model` completed. The
  command `uv run --frozen --no-editable python -m
  scripts.check_validation_artifacts --enforce-change-boundary` reported the
  contracts and required visual outputs valid.
  The focused command `uv run --frozen --no-editable python -m pytest -q -p
  no:cacheprovider tests/unit/test_validation_artifact_contract.py
  tests/unit/test_reproduction_boundary.py tests/unit/test_review_guidance.py`
  independently reported 132 passed in 108.10s.

  Visual, startup, and residue resolutions remain verified. Exact-SHA CI and Ubuntu/
  Windows boundary runs and their retained bundles are authentic and internally
  consistent, but successful execution of the same incomplete semantic oracle cannot
  close the gate. R2 remains unfrozen. No conclusion about scientific viability is
  made.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VIA000-R2-VISUAL-001"
    outcome: verified-resolved
    evidence: |-
      The committed comparator still rejects compact, thin, dashed, text, annotation,
      and over-bound raster changes while accepting registered in-bound drift. All
      twelve retained visual files from exact-SHA Ubuntu and Windows bundles pass;
      independently recomputed maximum cross-platform per-channel difference is three
      in gravity_well/source_comparison.png. No in-contract visual bypass was found.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      This verifies the declared retained-raster boundary only, not scientific image
      quality or renderer behavior outside the frozen contract.

  - finding_id: "VIA000-R2-SEMANTIC-001"
    outcome: unresolved
    evidence: |-
      Exact handoff 7b05ea3b4bd1ba8923a4477ea64852512b375799 rejects every exact
      REREVIEW-4 potential counterexample across chain pi_time.phi, grid pi_time.phi,
      gravity pi_time_natural.phi, and gravity gravity_test.phi_point: all one-element
      +4e-9 changes, 1e7 scale, sign, reversal, roll/permutation, uniform +4e-9 shift,
      shaped row/column transforms, stale min/max/range/mean/max-absolute/index-moment
      aliases, stale checks/radial/log-fit/redshift aliases, range-preserving absolute-
      magnitude changes, and correlated cached-summary updates. The prior source-law,
      many-body, malformed, nonfinite, shape, order, denominator, Boolean, startup,
      residue, and visual negative-control families also reject.

      The solver binding at scripts/check_validation_artifacts.py:_bind_clock_solver
      decodes finite shapes, nonnegative mu, a strict Boolean normalize flag, and
      recomputes ||(diag(row_sum)-W+mu^2 I)phi-source||. It does not require W to be
      symmetric, nonnegative, zero-diagonal, or supported by the retained inferred
      edges; bind W to the upstream pi_loc graph; derive effective_source from retained
      raw density/source, background, and zero-mode policy; or bind mu to configuration.

      The following fresh exact-handoff mutations each passed
      compare_validation_documents(reference,candidate) and produced zero errors from
      check_validation_semantics(candidate,path):

      * `weight_matrix[0][0] += 4e-9` for chain, grid, natural gravity, and diagnostic
        gravity (one accepted drift each). This creates a forbidden self-loop that
        cancels exactly from diag(row_sum)-W.
      * Asymmetric `W[0,1] += 4e-9` plus
        `effective_source[0] += 4e-9*(phi[0]-phi[1])` passed all four solver records.
        The symmetric paired form and its two paired source updates also passed.
      * Adding a new symmetric non-edge at (0,2) with the exact paired source update
        passed all four records while retained inferred-edge/config structure stayed
        unchanged. The chain raw phi difference was -0.04425624186514357; grid and
        natural-gravity were approximately -1.146e-16; diagnostic gravity was
        approximately 4.337e-19.
      * `pipeline.gravity_test.source_background += 4e-9` passed alone. Changing
        diagnostic gravity mu by +4e-9 and adding
        `((mu_new)^2-(mu_old)^2)*phi` to effective_source passed while config.mu stayed
        0.1.
      * Changing mu from 0 to 4e-9 alone passed chain, grid, and natural gravity. The
        implementation in popgp/simulator.py takes distinct mu==0 augmented-gauge and
        mu>0 screened-solve branches, so this is not a portability-only change.
      * Adding 4e-12 to every chain effective-source entry, then recomputing residual
        and its check alias, passed with source sum 1.5999111757647455e-11 and residual
        7.99955631250482e-12. Adding 1e-9 to every natural-gravity source similarly
        passed with source sum 8.999999777955396e-09 and residual
        2.999999925985132e-09. Neither is a valid post-subtraction zero-mode source.

      The equation recomputation therefore proves only internal consistency among
      mutually mutable retained operands. Correlated mutation can preserve its
      residual while contradicting the graph, source construction, configuration,
      and zero-mode decisions the report claims to retain.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Required action matrix: (1) retain and bind raw source/delta_rho, background,
      zero_mode_policy, normalize_potential, and configured mu; reconstruct the exact
      effective source and enforce its applicable zero-sum invariant; (2) bind W to
      retained upstream graph evidence and enforce square finite symmetric nonnegative
      zero-diagonal structure plus exact inferred-edge support; (3) bind diagnostic
      gravity source construction to point strength/center/background and require its
      solver mu to equal config.mu; (4) add fail-closed tests for diagonal, negative,
      asymmetric, new-edge, row/column, correlated W/source, mu branch-crossing,
      stale-config-mu, background, and non-zero-sum source mutations on all four solver
      records; (5) rerun the complete numeric/raw enumeration and both hosted boundary
      platforms. Do not replace these semantic identities with a generic tolerance.

  - finding_id: "VIA000-R2-STARTUP-001"
    outcome: verified-resolved
    evidence: |-
      Persistent, self-cleaning, and restoring startup/environment carriers remain
      rejected before trusted child execution. The base-owned isolated bootstrap,
      interpreter discovery, complete environment manifest, per-child verification,
      and malformed/blocked/mutated-environment controls pass locally and on exact-SHA
      Ubuntu and Windows runs. No in-contract startup bypass was found.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Host runner, operating-system loader, and declared trusted base remain explicit
      trust boundaries.

  - finding_id: "VIA000-R2-RESIDUE-001"
    outcome: verified-resolved
    evidence: |-
      Git object/mode and complete-environment boundaries still reject installed-
      package mutation, ignored bytecode, transient/restored source, filters,
      attributes, assume-unchanged, skip-worktree, staged/index-only state, deletion,
      rename, symlink, mode, ignored, and untracked residue. Downloaded Ubuntu and
      Windows source manifests each contain 558 unique entries with zero Git object/
      mode mismatches. The reviewer worktree remained clean in normal and ignored
      status; regeneration was confined to a disposable exact-SHA worktree.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Windows CRLF-normalized worktree bytes were not falsely equated with the
      authoritative Git blob/object and mode checks.

prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R2-VISUAL-LOCAL-001"
    outcome: verified-satisfied
    evidence: |-
      Targeted compact, one-pixel, dashed, text, annotation, within-bound, and
      over-bound controls pass. Exact-SHA Ubuntu and Windows retained visuals pass,
      with maximum observed cross-platform per-channel difference three.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The meaningful-feature matrix and declared platform calibration remain retained."

  - requested_test_id: "TST-VIA000-R2-SEMANTIC-MARGIN-001"
    outcome: unresolved
    evidence: |-
      The dedicated 1e-18 normalized-index-moment policy is independently supported:
      the 5.49e-19 honest recomputation delta is accepted, the 4.46e-18 registered
      transformation rejects, probes serialized at approximately 1e-18 reject, and
      alternate reduction-order delta is at most 8.47e-20. The raw potential and
      previous source-law/many-body controls pass. However, every exact solver
      counterexample listed under VIA000-R2-SEMANTIC-001 is accepted because correlated
      retained operands are not bound to upstream graph/source/configuration meaning.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Extend the semantic negative-control matrix with the required action matrix in
      VIA000-R2-SEMANTIC-001; retain the calibrated moment policy.

  - requested_test_id: "TST-VIA000-R2-STARTUP-TRANSIENT-001"
    outcome: verified-satisfied
    evidence: |-
      Persistent, self-deleting, and byte-restoring startup carriers and package/
      environment mutations are rejected under the actual wrapper sequence locally
      and on exact-SHA hosted Windows and Ubuntu execution.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Pre/post child verification and complete environment hashing remain in force."

  - requested_test_id: "TST-VIA000-R2-RESIDUE-INDEX-001"
    outcome: verified-satisfied
    evidence: |-
      Focused execution covers installed packages, ignored bytecode, transient
      restoration, filters/attributes, assume-unchanged, skip-worktree, index/staged
      state, deletion, rename, symlink, mode, ignored/untracked paths, and pre/post
      enforcement. Both exact hosted manifests have unique paths and zero Git object/
      mode mismatches; the reviewer branch remains clean.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The literal tree/blob and complete-environment boundaries remain closed."

  - requested_test_id: "TST-VIA000-R2-CROSS-PLATFORM-001"
    outcome: unresolved
    evidence: |-
      Execution and retention are authentic at exact handoff SHA. Ordinary CI run
      32386405595 succeeded with 295 tests in 81.59s. Boundary run 32386405500
      succeeded in 5m13s on Ubuntu and 9m44s on Windows; its artifacts are Ubuntu
      9413267497 (API digest
      sha256:00528b8de4224340774c299207b8080209fa24d707dcf4416d0d9e10fb8c8b10)
      and Windows 9413439632 (API digest
      sha256:eb3d4f8d3a839aa6531f726344ab97dc65f96ad38141c6d6575c58b87988edbe).

      Both bundles bind commit 7b05ea3b4bd1ba8923a4477ea64852512b375799, tree
      8ed3672179b4766986872e1125611835dfdf4e18, and checker SHA-256
      69e56aba44f77d766a6faa27ca6a8adfaba7825c4b8429ab1b0d9b07408b671c.
      Ubuntu/Windows source manifests each have 558 unique paths, zero Git object/mode
      mismatches, matching sidecars, and canonical SHA-256 values
      832fefbb4c96662d363054220b2ae984bef36a80cd04c6f8f747ef7bb551609f and
      85a34e72e19d0d53a654fd20b6de0c00b45f67d4cc760fdd13afc146f1340637.
      Their 22219/21319-entry environment manifests have canonical SHA-256 values
      8df4c8fa1c13f015f81163fde3e0ac915b958e11c9efa3b273c1c0e5136e90ac and
      8d7f135cacd255d0280329dffe78c37f4441b9114597860571bbebc1bfc886a1.
      Final status has only 16 Ubuntu and seven Windows declared generated paths,
      zero ignored paths, all six documents and twelve visuals pass, and maximum
      cross-platform channel delta is three.

      This test remains unresolved only because the exact retained checker accepts
      the implementation-boundary counterexamples documented above. A green run of
      an incomplete semantic predicate cannot establish zero-blocker R2 readiness.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      After semantic remediation, repeat both exact-SHA workflows, independently
      reconcile both retained bundles, and obtain a fresh zero-blocker re-review.

predictions:
  experiment_id: "TST-VIA000-R2-BOUNDARY-ADVERSARIAL-REREVIEW-005"
  predicted_outcome: |-
    On commit 7b05ea3b4bd1ba8923a4477ea64852512b375799, the listed diagonal,
    asymmetric, symmetric, new-edge, correlated-source, mu branch-crossing,
    stale-config-mu, background, and non-zero-sum source mutations will remain
    accepted. A complete remediation will reject each while preserving the current
    raw-potential, calibrated-index-moment, visual, startup, residue, and hosted-
    platform results.
  predicted_failure_mode: |-
    Without remediation, a retained R2 document can present an internally consistent
    finite-graph residual for a graph, source, or solver policy that contradicts its
    upstream retained structure and decisions.
  confidence_statement: |-
    High confidence for this implementation-boundary prediction because each decisive
    correlated bypass was executed against the exact frozen handoff and the residual
    was independently recomputed. No inference is made about POPGP's physical
    mechanism or scientific viability.

recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    Changes requested. VIA000-R2-VISUAL-001, VIA000-R2-STARTUP-001, and
    VIA000-R2-RESIDUE-001 remain independently resolved, and the remediation closes
    the exact REREVIEW-4 raw-potential attacks plus its calibrated moment control.
    VIA000-R2-SEMANTIC-001 and TST-VIA000-R2-SEMANTIC-MARGIN-001 remain blocking
    because the retained equation is not bound to graph structure/provenance, source
    derivation/zero-mode policy, or configured mu; TST-VIA000-R2-CROSS-PLATFORM-001
    therefore also cannot close. Implement the five-part required action matrix,
    rerun both exact-SHA hosted workflows, reconcile their retained bundles, and
    obtain another fresh independent re-review before preregistering or freezing R2.
    This verdict concerns only pre-holdout R2 evidence readiness. It is not a
    scientific viability result, merge permission by itself, or permission to reveal
    restricted material.
```
