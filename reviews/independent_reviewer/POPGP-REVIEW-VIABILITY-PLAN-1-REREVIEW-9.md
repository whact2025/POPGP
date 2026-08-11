# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-9

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-9"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_9"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "07130dd7c9ad92d85ead1cf3fd90e72d691db93d"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "d84ee7589e1c65873a80311fd405896aea3d0fcb:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-8.md"
builder_response_ref: "07130dd7c9ad92d85ead1cf3fd90e72d691db93d:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-9.md"
context_hash: "a69b4ea0ce8822ea9bededca2b48f4646db3abb1"
context_hash_method: "git rev-parse \"07130dd7c9ad92d85ead1cf3fd90e72d691db93d^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - "README.md"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "docs/templates/DISAGREEMENT_LOG_TEMPLATE.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
  - "docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml"
  - "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
  - "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
  - "docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "docs/scientific_hardening/DECISIONS.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/PROJECT_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "schemas/viability/campaign-v2.schema.json"
  - "schemas/viability/packet-v2.schema.json"
  - "schemas/viability/protocol-manifest-v2.schema.json"
  - "schemas/viability/primary-protocol-v1.schema.json"
  - "schemas/viability/requirements-v2.json"
  - "schemas/viability/independent-review-v1.schema.json"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/independent-rereview-v1.schema.json"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/review-response-v1.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_viability_campaign_contract.py"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-1.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-2.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-2.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-3.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-3.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-4.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-4.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-5.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-5.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-6.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-6.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-7.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-7.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-8.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-8.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-9.md"
  - "pyproject.toml"
  - "uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..07130dd7c9ad92d85ead1cf3fd90e72d691db93d (complete 45-file diff)"
  - "git diff 565568030c42dfaf24af52aa08a3a4b31c5c415a..07130dd7c9ad92d85ead1cf3fd90e72d691db93d (exact 5-file remediation diff)"
access_level: local/public-repository-only
independence_statement: |-
  This was a fresh adversarial re-review in the required dedicated worktree and
  reviewer session. It remained under the builder's human operator and root Codex
  orchestrator, and both seats identify as OpenAI Codex GPT-5; exact model snapshots
  are unavailable, so model separation is false. The immutable history was visible.
  Builder claims were treated only as hypotheses: I independently bound the candidate
  commit and tree, audited the complete baseline and exact remediation diffs, executed
  the authoritative suite, and constructed disposable real-Git campaigns and hostile
  inputs. No final labels, secret seeds, private evaluator logic, credentials,
  unaffiliated implementation, private hardware result, or empirical campaign result
  was available. This is process-separated local contract review, not external
  scientific validation.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "OpenAI Codex GPT-5"
  builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
  builder_orchestrator_id: "codex-multi-agent-root"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  Changes requested with two blocking contract findings. The response's exact
  constant, nesting, hostile-path, recursive-type, and residual-percent repairs are
  independently verified. Strict JSON constants reject NaN and Infinity, excessive
  JSON/YAML nesting produces controlled errors, hostile public path arguments return
  deterministic error lists, nested/list Boolean-integer-float protocol substitutions
  are rejected, and double/triple encoded or malformed repository hosts are rejected
  consistently with Git.

  One historical scope remains unresolved. A complete hash-reconciled Tier-R campaign
  placed the syntactically valid exact JSON number `1e999` in an unused field of its
  decisive raw-results receipt. Both public validation calls returned `[]`: Python
  decoded the number to positive infinity, and finite-value enforcement covered only
  bound values or selected structured-receipt call sites. This contradicts the plan's
  invariant that every structured receipt contains finite JSON numbers.

  One genuinely new bounded-resource defect was reproduced. A complete pre-frozen,
  hash-reconciled Tier-E campaign used a 16 MiB sparse-zero external Git bundle. The
  validator reached bundle cloning but did not return within a 79-second outer bound
  despite its declared 30-second subprocess timeout, and an orphan `git clone` process
  remained. Two direct helper attempts likewise exceeded 75/90-second outer bounds;
  a supervised direct Git run remained live after 40 seconds. Spawned processes were
  identified and terminated after each probe. Captured child handles on Windows make
  the nominal parent-process timeout non-total.

  The separate contract suite passed 19 tests, the full suite passed 178 tests, all
  six README examples ran, validation artifacts passed, and regeneration was clean.
  The other nine historical findings are resolved and the other ten historical tests
  are satisfied. These results establish only local executable-contract behavior; no
  Tier R, G, or E result or POPGP scientific validation was produced.

findings:
  - id: "VPLAN-RESOURCE-001"
    severity: high
    category: governance
    location: "scripts/check_viability_campaign.py:1279-1338"
    evidence: |-
      The bundle validator runs `git clone` with `capture_output=True` and
      `timeout=30`. On this Windows host, a 16 MiB sparse-zero bundle caused the public
      validator for an otherwise complete, hash-reconciled Tier-E campaign to exceed
      an independent 79-second supervisor; a READY marker proved fixture completion,
      while no RETURNED marker was written. Process inspection found the still-live
      clone child. Two direct helper runs exceeded 75 and 90 seconds, and direct Git
      remained live after 40 seconds. Every lingering process was then killed.
    finding: |-
      The declared Git-bundle timeout does not bound the spawned process tree or
      captured-pipe drain on Windows.
    failure_scenario: |-
      An untrusted external-repository bundle starts a Git child that remains alive
      after Python times out the parent process. The child retains captured handles,
      so validation never returns and leaves an orphan process.
    consequence: |-
      A submitted evidence package can indefinitely stall the mechanical campaign
      gate or CI worker and consume processes, so fail-closed adjudication is not
      resource-bounded.
    required_action: |-
      Enforce one wall-clock bound over the whole Git process tree and pipe drain on
      every supported platform, terminate descendants on timeout, and verify cleanup.
      Add cheap bundle size/header controls if desired, but do not rely on them as the
      process bound. Persist the requested full-campaign regression.
    verification: confirmed-by-execution
    blocking: true

requested_tests:
  - id: "TST-VPLAN-RESOURCE-001"
    description: |-
      Build a complete honestly pre-frozen Tier-E campaign whose external bundle is a
      hash-reconciled 16 MiB sparse-zero file. Under a supervisor stricter than the
      campaign worker's resource budget, the public validator must return one stable
      controlled error within the documented clone bound, leave no Git descendants,
      and remove its temporary checkout. Repeat on Windows and the primary CI host;
      include a valid-bundle positive control.
    rationale: |-
      Demonstrates that untrusted provenance cannot hang the portable mechanical gate
      even when a spawned Git child retains inherited output handles.
    blocking: true

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The exact nonstandard constants, 1,500-level receipt, and NUL public-entry cases
      now return deterministic errors, and duplicate keys and malformed types remain
      rejected. However, `_load_json_text` converts the valid exact token `1e999` to a
      Python infinity. `_binding_values` checks finiteness only for referenced binding
      values rather than traversing the whole raw-results document. After receipt and
      output-commitment hashes were reconciled, a complete Tier-R campaign containing
      an unused overflow field validated twice with `[]`.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Constants and nesting are repaired, but the promised all-structured-receipt
      finite-number invariant is still not enforced.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The 19-test campaign suite independently retained blind exposure, prohibited
      identity/session reuse, custodian-only reveal, chronology, output commitments,
      manifest substitution, resolved path/byte distinction, retention, and structured
      commitment/reveal reconciliation. All passed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for local custody consistency; off-system custody remains external."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      The complete Tier-G positive and raw-false controls remained green. Missing or
      false 3D, acceleration/geodesic, same-source lensing and Shapiro, two-potential,
      laboratory, Lorentz, and no-signaling capabilities; alternate pointers/rules;
      missing bindings; and Boolean/number substitutions remained rejected.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the executable gate contract only; no Tier-G result exists."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      Scientific, capability, resource, access, and invalid cause families; complete
      pass/fail/block truth vectors; missing evidence; valid/pending states;
      precedence; and honest external disagreement remained deterministic.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for locally computed adjudication."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      The lower-wave, acyclic, tier-transitively-closed DAG remained enforced. Unknown
      edges, cycles, same-wave prerequisites, malformed dependency values, and
      premature dependent holdout start remained rejected.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for dependency topology and execution order."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      Campaign-owned E4 floors and VIA-900's E5 floor remained immutable. Declared and
      achieved downgrades, unknown levels, and missing cumulative receipt kinds were
      rejected, while a deliberately stricter declaration remained valid.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Receipt-class semantic truth still requires substantive review."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      Real-Git tests retained nonexistent commit, candidate-tree, packet-rule,
      requirements, executing-contract, manifest/path/hash, schema migration, bundle
      commit/tree/content, and fix-ancestry controls. The remediation commit is an
      ancestor of the exact reviewed candidate.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for frozen object identity; resource-bounded bundle execution is a new scope."

  - finding_id: "VPLAN-FREEZE-003"
    outcome: verified-resolved
    evidence: |-
      Same-path changes to parameters, procedures, budgets, commands, mutations,
      protocol paths, and bytes remained bound to protocol-commit blobs and the current
      packet-freeze version after mutable hashes were recomputed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for preregistration content freeze."

  - finding_id: "VPLAN-PROTOCOL-001"
    outcome: verified-resolved
    evidence: |-
      The closed primary-protocol schema retained all extra-authority negatives and the
      persisted Boolean/integer substitution now fails. A fresh full campaign used
      nested lists and mappings with integer/float and Boolean/integer substitutions;
      recursive exact-envelope comparison rejected it. Direct probes also rejected
      list and nested mapping substitutions and unequal key sets.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolved for exact recursive JSON envelope identity; this does not establish the
      scientific adequacy of preregistered parameters.

  - finding_id: "VPLAN-INDEPENDENCE-001"
    outcome: verified-resolved
    evidence: |-
      The persisted complete matrix rejected repository aliases, output reuse, bundle
      commit/tree defects, orchestrator reuse, chronology defects, invalid disagreement,
      and strict comparison substitutions while retaining its positive clean-room case.
      Direct comparison with Git showed one-decode percent hosts normalized
      conservatively, while double/triple encoded percent hosts, residual-percent hosts,
      and malformed IPv6 hosts were invalid locally and rejected by Git.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolved for local URI/provenance consistency only. Organization, authorship,
      exposure, and control remain externally verified facts.

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The enlarged matrix now covers strict constants, excessive JSON/YAML nesting,
      hostile entry paths, duplicate keys, malformed field types, immutable refs, and
      governance provenance. It omits finite exponent-overflow traversal of unused
      fields: two complete validations accepted a hash-consistent `1e999` raw-results
      field, so the portable fail-closed property remains incomplete.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Add exact exponent-overflow/underflow boundary cases at every structured-receipt
      call site, including fields not selected by JSON pointers.

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Exposure, session/identity reuse, custodian authority, reveal order, output and
      manifest hashes, resolved path/byte distinction, structured receipts, retention,
      and missing-field mutations all remained covered and passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for locally enforceable custody facts."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: verified-satisfied
    evidence: |-
      The Tier-G positive, raw-false, missing/false capability, alternate rule/pointer,
      missing-binding, and strict Boolean/number cases remained persisted and green.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    outcome: verified-satisfied
    evidence: |-
      All requested cause families, truth vectors, missing-evidence, ambiguity,
      valid/pending, campaign-precedence, and external-disagreement rows remained green.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-DEP-001"
    outcome: verified-satisfied
    evidence: |-
      Canonical DAG, wave, tier closure, cycle, missing-edge, malformed-value, and
      premature-holdout cases remained covered and passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      E4/E5 declared and achieved downgrades, unknown levels, cumulative receipt kinds,
      and stricter declarations remained covered and passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-FREEZE-002"
    outcome: verified-satisfied
    evidence: |-
      Real Git object/tree, rule, requirements, executing-contract, manifest/path/hash,
      escape, schema-version, bundle content, bundle commit/tree, and candidate-reuse
      mutations remained covered. Identity/provenance behavior is satisfied; the new
      process-tree resource test is recorded separately.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for frozen Git and bundle provenance."

  - requested_test_id: "TST-VPLAN-FREEZE-003"
    outcome: verified-satisfied
    evidence: |-
      Canonical preregistration fields, paths, bytes, budgets, commands, and mutations
      remained bound to the protocol commit and rejected.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-PROTOCOL-001"
    outcome: verified-satisfied
    evidence: |-
      Historical extra-authority controls and the ordinary exact-envelope positive
      remain green. Persisted scalar and fresh nested/list Boolean-integer-float
      substitutions were rejected through strict recursive comparison.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for exact primary-protocol envelope equality."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      Distinct typed organization/operator/implementation, exposure, prediction,
      commitment, and generic/malformed receipt negatives remained green; a positive
      typed clean-room fixture still validates.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the typed clean-room identity and receipt scope."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    outcome: verified-satisfied
    evidence: |-
      Candidate custody output, path/byte distinction, bundle provenance, external
      orchestrator, typed comparison, pointer/tolerance/order, all accumulated aliases,
      strict receipt substitutions, residual percent, multi-encoding, and invalid-host
      cases failed closed; the valid positive retained acceptance.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for locally enforceable repository and comparison integrity."

predictions:
  experiment_id: "finite-receipt-and-bounded-bundle-remediation"
  predicted_outcome: |-
    A complete remediation will reject exponent-overflow values anywhere in every
    structured receipt and will return a deterministic bundle error within one declared
    process-tree deadline without leaving descendants or temporary checkouts.
  predicted_failure_mode: |-
    Checking only bound numeric values will continue to admit unused infinities, while
    timing only the direct Git parent with captured pipes will continue to hang when a
    descendant retains the handles.
  confidence_statement: |-
    High for local executable behavior: the numeric acceptance repeated twice in a
    complete campaign, and the bundle hang repeated at helper, direct-Git, and complete
    campaign levels under independent supervisors. Confidence does not extend to
    off-system independence truth or POPGP physics.

recommendation:
  approve: false
  blocking_findings: 2
  rationale: |-
    Approval remains fail closed. Nine historical findings are resolved and ten
    historical requested tests are satisfied, including the exact protocol and URI
    remediations. One historical parser/validator scope remains unresolved because an
    unused exponent-overflow value is accepted in decisive structured evidence. One new
    blocking resource finding shows that hostile bundle validation can exceed its
    stated deadline and orphan a process. No Tier R, G, or E result, external scientific
    validation, or merge authority follows from this review.
```

## Frozen identity and history

The required reviewer branch began clean at `07130dd7c9ad92d85ead1cf3fd90e72d691db93d`
with tree `a69b4ea0ce8822ea9bededca2b48f4646db3abb1`; the baseline, prior review carrier,
and remediation commit are ancestors. The immutable prior-review ref resolves, and its
blob is byte-identical to the candidate copy. The complete 45-file,
14,449-insertion/147-deletion baseline diff and exact five-file,
354-insertion/11-deletion remediation diff were audited. No candidate file was changed.

## Independent execution

| Command or probe | Observed result |
|---|---|
| `uv sync --frozen` | exit 0; CPython 3.11.15 environment created from the lock; 60 packages installed |
| `uv run ruff check .` | exit 0; all checks passed |
| `uv run python scripts/check_tex.py` | exit 0; 652 lines, balanced braces/environments, no Markdown remnants |
| `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | exit 0; 19 passed in 683.57 s |
| `uv run pytest -q` | exit 0; 178 passed in 721.13 s |
| six documented `python -m examples.physics_qg.*` commands | all exit 0; expected finite diagnostics and artifacts regenerated |
| `uv run python scripts/check_validation_artifacts.py` | exit 0; contracts and required visuals valid |
| strict loader/type/URI/hostile-path probes | constants and depth failed closed; overflow acceptance reproduced; recursive types and residual/multi-encoding rejected |
| complete Tier-R overflow campaign, repeated public validation | both calls returned `[]` |
| complete Tier-E sparse-zero bundle campaign | READY marker written; validator exceeded 79 s; no RETURNED marker; orphan Git process verified and terminated |
| `git diff --exit-code` after regeneration | exit 0 before this sole review artifact was created |

`pdflatex` and `nvcc` were unavailable, so no PDF or native-CUDA build success is
claimed. Those toolchain absences are limitations, not scientific failures. The
campaign test fixtures are synthetic, and the resource countermodel was reproduced on
Windows; the requested cross-platform regression remains necessary.

## Enforceability boundary

The blockers concern facts the local executable contract can decide: finite structured
evidence and termination of hostile provenance processing. A local validator cannot
prove that an organization is unaffiliated, code is independently authored, exposure
declarations are truthful, or custody occurred off-system. Green tests and examples do
not close the source, geometry, continuum, closure, Lorentz, hardware, or empirical gaps
already named by the scientific plan.
