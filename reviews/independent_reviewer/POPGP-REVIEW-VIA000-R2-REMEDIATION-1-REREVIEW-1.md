# VIA-000 R2 reproducibility-remediation independent re-review 1

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R2-REMEDIATION-1-REREVIEW-1"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "unknown"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-via000-r2-remediation-independent-rereview-session-1"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-19"
commit_reviewed: "bb822b378169fb2b529b1b983a9bf53190c0e770"
baseline_commit: "d0381f0e8562c5f70c4e315dc3df4afd0d6bfbf4"
prior_review_ref: "90c9c3c8d6c70146baf772cc3d33ea11a736b619:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1.md"
builder_response_ref: "bb822b378169fb2b529b1b983a9bf53190c0e770:reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-RESPONSE-1.md"
context_hash: "6358d6c881039713026f4579156495554956265e"
context_hash_method: "git rev-parse \"bb822b378169fb2b529b1b983a9bf53190c0e770^{tree}\""
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
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-REMEDIATION-1.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R2-REMEDIATION-1-RESPONSE-1.md"
  - "reviews/codex/POPGP-VIA000-R2-REMEDIATION-PLAN.md"
  - "scripts/check_reproduction_boundary.py"
  - "scripts/check_validation_artifacts.py"
  - "scripts/run_without_startup_hooks.py"
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
  - "tests/unit/test_reproduction_boundary.py"
  - "tests/unit/test_validation_artifact_contract.py"
  - "tests/unit/test_review_guidance.py"
access_level: "public-repository-only plus public GitHub Actions metadata, logs, and downloadable artifacts"
independence_statement: |-
  This was a fresh independent-reviewer task in a new isolated worktree at the exact
  response-containing candidate. It independently replayed the prior attacks and
  broader locality, decision-alias, startup, repository, environment-package, bytecode,
  and Git-control-plane mutations. It did not edit candidate implementation, tests,
  thresholds, scientific documents, or retained candidate artifacts. Only this review
  artifact was added. The same human operator and Codex Desktop orchestrator are shared
  with the builder, while the task/session is distinct. Exact served-model identities
  and snapshots were not exposed for either role, so both remain unknown and model
  separation is false. This is internal adversarial process separation, not external
  scientific validation. No R2 final label, secret seed, private evaluator, credential
  material, custody material, or unrestricted private hardware profile was accessed.

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
  Changes requested with three unresolved blocking findings. The remediation makes
  real progress: the exact 180x71 red overwrite is rejected; the exact duplicated
  grid phi_range=4e-9 mutation is rejected by an executable predicate; injected
  persistent, self-deleting, and byte-restoring .pth/sitecustomize/usercustomize
  carriers are rejected before execution; measured carriers remain inert under
  Python -I -S; and persistent assume-unchanged, skip-worktree, staged, deletion,
  rename, symlink, ordinary-untracked, and ignored-executable fixtures are rejected.
  The exact response candidate passed ordinary CI run 32296438580 and the Windows/
  Ubuntu boundary run 32296438583 with 237 tests per job. Downloaded startup manifests
  hash-match their retained digests and both regenerated evidence sets pass the current
  semantic and raster checker.

  The stronger claims do not survive independent counterexamples. The raster oracle
  accepts removal of an 18x18 compact feature, a 472-pixel one-pixel curve, a sparse
  dashed curve, ordinary rendered text, and an actual numeric annotation from the
  committed source-comparison plot. Decision validation accepts threshold-crossing
  values when only an unbound pipeline alias moves, and it accepts a nested precision
  floor that makes the retained signal/floor ratio 0.07784 against the stated >=1000
  criterion. The repository/runtime boundary permits modified installed dependency
  code and timestamp-valid ignored bytecode to execute, checks tracked source only at
  the end so byte-restoring source executes undetected, and delegates actual-byte
  comparison to Git conversion filters controlled by unmeasured .git state. These are
  within the prior findings' scopes, so no duplicate finding IDs are introduced.

  Authoritative positive controls were independently rerun through the base-owned
  snapshot and checked site-disabled bootstrap. The affected 65-test suite passed in
  69.00s; Ruff, TeX validation, all six generators, semantic/visual validation, the
  current source boundary, and final startup verification passed in 167s. Both the
  prior review and builder response validate against their v2 schemas with duplicate
  keys rejected. The original review blob at 90c9c3 is byte-identical to the copy in
  the response candidate. Baseline/original-candidate ancestry, tree binding, diff
  checks, ordinary index state, and the clean current-boundary check also pass.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VIA000-R2-VISUAL-001"
    outcome: unresolved
    evidence: |-
      Positive replay confirmed that the original centered 180x71 red rectangle is
      rejected by the new local/component checks, and exact-candidate Windows/Ubuntu
      artifacts pass. Broader executed attacks still return no errors from
      compare_visual_artifact: removing a black 18x18 feature from a 512x512 raster
      (324 changed pixels, max local-32 error 0.237305, largest component 324), a
      one-pixel 472-pixel curve (max local 0.0234375, component 472), a disconnected
      two-pixel dashed curve (480 changed pixels, largest component 16), rendered
      PASS=TRUE text (218 changed pixels, largest component 53), and one actual `0.0`
      annotation from gravity_well/source_comparison.png (280 changed pixels, max
      local 0.0856924, largest component 114). All are below the fixed local 0.25 and
      768-pixel component limits. Retained exact-candidate cross-platform comparison
      shows that, apart from the chain near-zero legend region, every pixel/channel
      differs by at most 3; source_comparison reaches 3 and many_body_source reaches 2.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The original exact attack is fixed, but the finding required compact-region,
      thin-curve, annotation, and near-limit structured rejection, not rejection of
      one dense surrogate. Canonicalize the chain near-zero legend and replace the
      permissive aggregate/local budget with a calibrated per-channel maximum <=4,
      or an equally strict demonstrated feature oracle. Re-run every attack above and
      both retained platform bundles before closure.

  - finding_id: "VIA000-R2-SEMANTIC-001"
    outcome: unresolved
    evidence: |-
      Positive replay confirmed that changing both grid_2d copies of phi_range to
      4e-9 leaves the portability comparator green but now produces a semantic
      recomputation error, and the registered 25-case decision suite passes. Two
      executed cross-field mutations remain accepted by both comparison and semantic
      validation: changing only gravity_well
      pipeline.gravity_test.relative_constraint_residual from about 1.99e-16 to 4e-9
      while its `<1e-12` check copy remains stale, and changing only chain_1d
      pipeline.pi_time.constraint_residual from about 4.44e-16 to 4e-9 while its
      `<1e-10` check copy remains stale. Each reports one accepted drift and zero
      errors. A nested many-body mutation changing both serialized assessment copies'
      full/lower absolute_precision_floor to 1e-10 reports four accepted drifts and
      zero semantic errors; the raw minimum signal is 7.783995670251898e-12, so the
      implied signal/floor ratio is 0.0778399567 despite the criterion requiring
      >=1000. _quadratic_assessment_outcome does not recompute that floor predicate or
      bind it to the raw responses.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The exact first counterexample is fixed, but the finite predicate registry is
      not a complete binding graph. Bind every duplicated decision operand across
      pipeline/measurements/checks, recompute derived fields such as signal-to-floor
      from raw values, and mutate every alias and nested predicate independently on
      both sides of its threshold.

  - finding_id: "VIA000-R2-STARTUP-001"
    outcome: verified-resolved
    evidence: |-
      The independently executed 65-test focused suite includes all nine combinations
      of persistent, self-deleting, and byte-restoring attack.pth, sitecustomize.py,
      and usercustomize.py carriers. Each post-snapshot carrier is rejected by the
      pre-execution check before its marker or child can run. Snapshot creation rejects
      a virtual interpreter and blocked environment; manifest tampering fails the
      runner-held digest; and the site-disabled bootstrap executes while a measured
      sitecustomize hook remains inert. Exact-candidate run 32296438583 repeats the
      actual base snapshot, checked wrapper, Python -I -S bootstrap, and final verify
      successfully on Windows and Ubuntu. Package/source byte-integrity attacks found
      in this re-review are assigned to the separate prior residue finding rather than
      used to duplicate this startup-hook result.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The resolution is limited to the declared .pth/customize startup-hook surface and does not attest all executable dependency or repository bytes."

  - finding_id: "VIA000-R2-RESIDUE-001"
    outcome: unresolved
    evidence: |-
      The registered persistent index/path fixtures pass, but four independent attacks
      bypass the public boundaries. First, after a clean snapshot, appending marker
      code to .venv/Lib/site-packages/numpy/__init__.py changes its SHA-256 from
      b16a4f347c6583c878e1973a208564e01686b79cc70911ad0157e05d8eecda37 to
      87f2f90fde8405aa2c08525689c42533913717c91e5dcc911a3f5f83096eb9d1;
      the actual checked uv/Python -I -S bootstrap returns 0, the marker executes,
      semantic/artifact validation reports valid, and startup postflight stays clean.
      Second, a timestamp/size-valid malicious
      scripts/__pycache__/check_tex.cpython-311.pyc executes under the same bootstrap;
      the final `--enforce-change-boundary` command returns 0 because all such pyc
      paths are allowed. Third, a tracked target module changed after snapshot executes
      under the actual checked bootstrap, restores its original bytes, and the final
      common repository boundary returns `[]`. Fourth, in a temporary Git repository,
      an unmeasured .git/info/attributes plus repository clean-filter configuration
      maps changed source bytes back to the committed blob; the actual source contains
      `VALUE = 999` and `ATTACK = True` while check_repository_residue returns `[]`.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      `.venv` and all __pycache__/*.pyc are unconditional ignored exceptions, but only
      .pth/customize files in the environment are hashed. The repository check runs at
      the end rather than around every child and uses Git conversion semantics rather
      than a base-owned literal file/mode manifest. Hash the complete locked executable
      environment, forbid executable ignored bytecode or hash it, verify literal frozen
      source bytes/modes before and after every child with a base-owned manifest that
      does not honor candidate Git filters/configuration, and add byte-restoring
      variants to the actual workflow.

prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R2-VISUAL-LOCAL-001"
    outcome: unresolved
    evidence: |-
      The dense original rectangle and registered width-three line/filled-rectangle
      surrogates reject, but the executed 18x18 feature, one-pixel continuous curve,
      sparse dashed curve, actual font text, and retained-plot annotation removal all
      pass. Honest exact-candidate platform output supports a materially tighter
      contract after canonicalizing the one chain legend discrepancy.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The requested smallest meaningful spatial/annotation matrix is not satisfied."

  - requested_test_id: "TST-VIA000-R2-SEMANTIC-MARGIN-001"
    outcome: unresolved
    evidence: |-
      The exact grid check/pipeline pair and 25 selected check-value crossings reject,
      but independent mutations of gravity/chain pipeline aliases and the nested
      many-body precision-floor operand cross declared criteria with stale Booleans
      while both public semantic oracles report zero errors.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The current parameterized matrix is check-name coverage, not exhaustive decision-operand and alias coverage."

  - requested_test_id: "TST-VIA000-R2-STARTUP-TRANSIENT-001"
    outcome: verified-satisfied
    evidence: |-
      All nine requested .pth/sitecustomize/usercustomize behavior combinations ran in
      the focused suite and were rejected before marker execution. Base-interpreter and
      blocked-environment snapshot controls pass. The actual trusted wrapper/site-
      disabled target path and final snapshot check are green at the exact candidate
      on Windows and Linux in run 32296438583.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "This result does not convert environment package or repository bytecode into startup-hook findings."

  - requested_test_id: "TST-VIA000-R2-RESIDUE-INDEX-001"
    outcome: unresolved
    evidence: |-
      Persistent assume-unchanged, skip-worktree, staged, deleted, renamed, symlink,
      ordinary-untracked, and ignored attack.py fixtures reject. Ignored executable
      package code in .venv, executable timestamp-valid pyc cache state, transient
      tracked byte restoration, and Git clean-filter control-plane state nevertheless
      execute or conceal changed bytes while both public final boundaries accept.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Exercise ignored package/cache and byte-restoring attacks against the actual wrapper sequence, not only a persistent temporary-repository end state."

  - requested_test_id: "TST-VIA000-R2-CROSS-PLATFORM-001"
    outcome: unresolved
    evidence: |-
      The execution portion is strong: exact-candidate ordinary CI run 32296438580 is
      green, and run 32296438583 passed 237 tests, all six examples, the current
      semantic/visual/source checker, and startup pre/post checks on Ubuntu and Windows.
      Artifacts 9381450986 and 9381678716 retain 23 files each and bind commit bb822b3,
      tree 6358d6c, startup manifests/digests, final status, raw JSON, and rasters.
      Canonical extracted-manifest SHA-256 values (sorted POSIX relative path, tab,
      byte length, tab, file SHA-256, newline) are
      db2c34b6273dbb473c5549d49cbbba7741d6a95b7d30fca03fcadc34d8c2db2a
      for Ubuntu and
      bdaa0cffaed210e1702d23af6d0d9cafd4100857e0ef7441d6f691bcc451b391
      for Windows. However, the bundles contain no literal source-byte manifest, the
      blocking negative controls above still accept, and no R2 campaign/preregistered
      protocol exists in the frozen tree. Earlier de3a64f artifacts 9381141997 and
      9381287208 were also inspected; their equivalent extracted-manifest hashes are
      8f116fc0b63acfe3516015def1c46237abd5d0c84193117a7dd0ea35814c5ede
      and 06596ee5381171b1c9f40ad0669d6c463ee6d9f8c384da3f9e7cd90ddf62d3a9.
      The paired earlier ordinary-CI run 32295482351 was also inspected and completed
      success with 237 tests and six generators at exact de3a64f.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Green two-platform calibration is necessary but does not satisfy the requested
      source-manifest, complete negative-control, and preregistered-protocol conditions.

predictions:
  experiment_id: "TST-VIA000-R2-BOUNDARY-ADVERSARIAL-REREVIEW-001"
  predicted_outcome: |-
    On the frozen candidate, replaying the compact/curve/text raster removals, unbound
    pipeline and nested-floor mutations, environment-package or ignored-pyc execution,
    transient restored source, and Git clean-filter attack will continue to produce
    accepted results. A complete remediation will reject each while accepting the
    hashed retained Windows/Ubuntu evidence under a materially tighter calibrated
    visual rule.
  predicted_failure_mode: |-
    Without remediation, an R2 workflow can report equivalent visuals and scientific
    decisions and clean frozen execution even when a compact plotted feature is gone,
    a raw retained operand contradicts its check, or code outside the frozen source/
    locked environment bytes executes and erases or filters its evidence.
  confidence_statement: |-
    High confidence for the implementation-boundary prediction because every listed
    bypass was executed against the frozen code or actual trusted wrapper. No inference
    is made about POPGP's physical mechanism or scientific viability.

recommendation:
  approve: false
  blocking_findings: 3
  rationale: |-
    Changes requested. VIA000-R2-STARTUP-001 and its requested transient-hook test are
    independently resolved, and the exact response candidate is green on both hosted
    platforms. VIA000-R2-VISUAL-001, VIA000-R2-SEMANTIC-001, and
    VIA000-R2-RESIDUE-001 remain blocking under executed counterexamples; four of five
    prior requested tests therefore remain unresolved. Do not preregister or execute a
    new R2 viability holdout until a new remediation candidate rejects these attacks,
    retains the complete two-platform evidence requested, and receives another fresh
    independent re-review.
```
