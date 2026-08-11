# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-8

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-8"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_8"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "0675367fdcdea86b0710dd2bad2e1a6f53ee92e7"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "cdb9df65a7bc28b04702871b01eb65bdc104aad0:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-7.md"
builder_response_ref: "0675367fdcdea86b0710dd2bad2e1a6f53ee92e7:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-8.md"
context_hash: "2b09d4cfced4ffe33375eebdb2cf1aec9584b8e4"
context_hash_method: "git rev-parse \"0675367fdcdea86b0710dd2bad2e1a6f53ee92e7^{tree}\""
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
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/independent-rereview-v2.schema.json"
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
  - "pyproject.toml"
  - "uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..0675367fdcdea86b0710dd2bad2e1a6f53ee92e7 (complete 43-file diff)"
access_level: local/public-repository-only
independence_statement: |-
  This was a fresh adversarial re-review in the required dedicated worktree and
  session. It remained under the builder's human operator and root Codex
  orchestrator, and both seats are identified as OpenAI Codex GPT-5; exact model
  snapshots are unavailable, so model separation is false. The immutable review and
  response history was necessarily visible. Builder assertions were hypotheses only:
  the exact frozen tree, complete baseline diff, full quality suite, and fresh
  disposable real-Git Tier-E campaigns were independently checked. No final labels,
  secret seeds, private evaluator logic, credentials, unaffiliated implementation,
  private hardware result, or empirical campaign result was available. This is
  internal process-separated contract review, not external scientific validation.

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
  Changes requested with three unresolved blocking findings. RESPONSE-8's exact fixes
  are verified. Complete honestly pre-frozen Tier-E campaigns now return stable errors
  for the 5,000-digit dotted host and NUL receipt path, and reject native Windows drive
  reuse through opaque `file:C:/...` plus IPv4 reuse through IPv4-mapped IPv6. The
  persisted nineteen-campaign matrix retains `%00`, `%ZZ`, default port, DNS dot, dot
  segments, terminal `.git`, percent host, IPv6 spelling, hierarchical file URI,
  leading-zero IPv4, ssh/git+ssh, and all four strict comparison substitutions.

  Broader complete-campaign probes found three locally enforceable defects within
  existing historical scopes. First, a hash-consistent blinded-prediction JSON receipt
  containing the non-JSON token `NaN` was accepted twice with `[]`, while a 1,500-level
  nested JSON receipt raised an uncaught `RecursionError` twice. Second, a frozen packet
  parameter with JSON number `1` and its primary-protocol counterpart with JSON Boolean
  `true` were accepted twice because ordinary Python mapping equality conflates those
  values. Third, an external repository URI with double-encoded host text
  `https://%2567ithub.com/...` was accepted twice with `[]`; Git rejected the same URI
  as `Bad hostname`. These reopen VPLAN-SCHEMA-001, VPLAN-PROTOCOL-001, and
  VPLAN-INDEPENDENCE-001 without new IDs. Their matching historical requested-test
  scopes are unresolved.

  All other historical findings and requested tests remain independently closed. The
  separate contract suite passed 18 tests, the full suite passed 177 tests, all six
  examples regenerated, artifact validation passed, and regeneration was clean.
  A 16 MiB malformed, hash-reconciled Git bundle returned the same controlled clone
  error twice. Symlink creation was unavailable on this Windows account; a hard-linked
  candidate/external output was rejected through byte-identity checks. Direct NUL in
  the API's `campaign_path` or `repo_root` argument still raises before document load,
  but it was not used as a decisive finding because such a string cannot designate a
  complete campaign. None of this demonstrates Tier R, G, E, or POPGP physics.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      RESPONSE-8's exact totality fixes are verified at the public boundary. Fresh
      complete Tier-E campaigns returned deterministic error lists for a 5,000-digit
      dotted host and a NUL output-commitment receipt path. NUL/control/5,000-character
      packet, receipt, bundle, review-ref, protocol, and manifest paths also returned
      stable error lists. However, a pre-frozen blinded-prediction receipt containing
      `NaN` passed twice with `[]`, although `NaN` is not valid JSON. Replacing its
      predictions array with 1,500 nested arrays raised uncaught `RecursionError` on
      both calls. `_load_json_text` at scripts/check_viability_campaign.py:180 uses the
      permissive standard decoder and structured receipt callers do not catch recursion.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The exact REREVIEW-7 cases are resolved, but deterministic fail-closed handling
      remains incomplete for hash-consistent external structured receipts. Direct NUL
      API path arguments also raise before loading but were not decisive here.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The complete 18-test campaign suite retained blind exposure, prohibited identity
      and session reuse, custodian-only reveal, chronology, output commitment, manifest
      substitution, resolved path/byte distinction, retention, and structured
      commitment/reveal reconciliation; all passed. Hard-linked output bytes were also
      rejected independently.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for local custody consistency; off-system custody remains an external fact."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      The dedicated suite retained the complete Tier-G positive fixture and raw-false
      countermodel. Missing or false 3D, acceleration/geodesic, same-source lensing and
      Shapiro, two-potential, laboratory, Lorentz, and no-signaling capabilities;
      alternate pointers/rules; missing bindings; and Boolean/number gate substitutions
      were rejected.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "This resolves the executable gate contract only; no Tier-G result exists."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      Scientific, capability, resource, access, and invalid cause families; complete
      pass/fail/block truth vectors; missing evidence; valid/pending; precedence; and
      honest external disagreement remained deterministic in the green contract suite.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for locally computed adjudication."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      Requirements and the dedicated suite retained the lower-wave, acyclic,
      tier-transitively-closed DAG and rejected unknown edges, cycles, same-wave
      prerequisites, malformed dependency values, and premature dependent holdout start.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for dependency topology and execution order."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      Campaign-owned E4 floors and VIA-900's E5 floor remained immutable. Declared and
      achieved downgrades, unknown levels, and missing cumulative receipt kinds were
      rejected, while the deliberately stricter VIA-000 declaration remained valid.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Receipt-class semantic truth still requires substantive review."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      Real-Git tests retained nonexistent commit, candidate-tree, packet-rule,
      requirements, executing-contract, manifest/path/hash, v1 migration, external
      bundle commit/tree/content, and fix-ancestry controls. Fix commit
      1cd76fa57855f3ce94c40bb20b0a552440075083 is the reviewed candidate's parent.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for frozen Git/protocol objects."

  - finding_id: "VPLAN-FREEZE-003"
    outcome: verified-resolved
    evidence: |-
      Same-path changes to canonical preregistration parameters, procedures, budgets,
      commands, mutations, paths, and bytes remained bound to protocol-commit blobs and
      packet-freeze-v4 after mutable hashes were recomputed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for preregistration content freeze."

  - finding_id: "VPLAN-PROTOCOL-001"
    outcome: unresolved
    evidence: |-
      The closed primary-protocol schema still rejects every historical extra threshold,
      exclusion, measurement, command, and resource authority. Nevertheless, one fully
      hash-consistent pre-frozen Tier-E campaign declared packet parameter
      `replicates: 1` and primary-protocol parameter `replicates: true`. Both complete
      validations returned `[]`. The schema admits arbitrary parameter values, and the
      ordinary `document != expected_document` comparison at
      scripts/check_viability_campaign.py:1047 uses Python's `True == 1` coercion rather
      than strict JSON type equality.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Extra-key ambiguity is fixed, but the promised exact canonical envelope is not
      exact for JSON Boolean/number substitutions.

  - finding_id: "VPLAN-INDEPENDENCE-001"
    outcome: unresolved
    evidence: |-
      RESPONSE-8's opaque Windows file-URI and IPv4-mapped IPv6 repairs are verified,
      and the complete matrix still rejects all historical repository aliases, strict
      comparison substitutions, output reuse, bundle commit/tree defects,
      orchestrator reuse, chronology defects, and invalid disagreement claims. A new
      honestly pre-frozen Tier-E campaign used external repository
      `https://%2567ithub.com/independent/repo`; two public validations returned `[]`.
      Direct normalization leaves residual `%67ithub.com`, while Git rejects that same
      repository URI with `URL rejected: Bad hostname`. Thus locally invalid repository
      provenance can still be promoted as a distinct external repository.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      This is local URI/provenance validation, not an inference about organization,
      authorship, exposure, or control of a valid off-system repository.

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The exact huge-host and NUL receipt cases and the broader in-document path matrix
      now return deterministic errors. The suite also retains duplicate keys, malformed
      field types, immutable refs, and governance provenance. It does not reject JSON
      non-finite constants or bound structured-receipt nesting, so one complete receipt
      passes malformed JSON and another raises `RecursionError`.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Extend the existing public-boundary matrix with strict RFC JSON constants and bounded nesting."

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Blind exposure, session/identity reuse, custodian authority, reveal order,
      output/manifest hashes, resolved path and byte distinction, structured receipts,
      retention, missing fields, and a hard-linked output control all failed closed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for locally enforceable custody facts."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: verified-satisfied
    evidence: |-
      The Tier-G positive, raw-false, missing/false capability, alternate rule/pointer,
      missing binding, and strict Boolean/number gate cases remain persisted and green.
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
      Canonical DAG, wave, tier closure, cycles, missing edges, malformed values, and
      premature holdout cases remained covered and passed.
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
      escape, schema-version, bundle content, commit/tree, and candidate-reuse mutations
      remained covered. A fresh 16 MiB malformed bundle returned an identical controlled
      clone error twice without raising.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for frozen Git and bundle provenance."

  - requested_test_id: "TST-VPLAN-FREEZE-003"
    outcome: verified-satisfied
    evidence: |-
      Canonical preregistration field, path, byte, budget, command, and mutation
      substitutions remained bound to the protocol commit and rejected.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-PROTOCOL-001"
    outcome: unresolved
    evidence: |-
      Historical extra-authority negatives and the ordinary exact-envelope positive
      remain green. The new complete `1` versus `true` parameter substitution passes,
      so the requested exact canonical-envelope property is incomplete despite the
      absence of extra keys.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Retain extra-key controls and add recursive strict JSON equality mutations for arbitrary parameters."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      Distinct typed organization/operator/implementation, exposure, blinded prediction,
      output commitment, and generic/malformed receipt negatives remained green; a
      positive typed clean-room fixture still validates.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the typed clean-room identity and receipt scope."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    outcome: unresolved
    evidence: |-
      Candidate custody output, distinct paths/bytes, bundle commit/tree, external
      orchestrator, typed comparison, pointer/tolerance/order, every historical alias,
      and strict external-receipt Boolean/number substitutions remain green. The matrix
      does not reject a residual percent escape in the decoded host, and the resulting
      Git-invalid URI still passes complete Tier-E validation.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Retain all current cases and add percent-decode-depth plus post-decode host-validity negatives."

predictions:
  experiment_id: "TST-VPLAN-SCHEMA-001, TST-VPLAN-PROTOCOL-001, and TST-VPLAN-INDEPENDENCE-002"
  predicted_outcome: |-
    A complete remediation will preserve every current positive and historical
    negative while rejecting non-finite JSON constants, returning deterministic errors
    for bounded excessive structured-receipt nesting, distinguishing Boolean from
    number recursively in the primary protocol, and rejecting residual percent escapes
    or otherwise invalid post-decode repository hosts.
  predicted_failure_mode: |-
    The permissive JSON decoder will keep accepting `NaN`; uncaught decoder recursion
    will keep escaping; ordinary mapping equality will keep equating `true` and `1`;
    and one-pass host decoding followed by permissive IDNA encoding will keep accepting
    a Git-invalid residual-percent host.
  confidence_statement: |-
    High for executable local contract behavior. Every decisive mutation was a complete
    Tier-E campaign with real candidate/protocol commits, real external Git-bundle
    provenance, reconciled hashes, and mutation before protocol-rule freeze; repeated
    public calls produced the same acceptance or exception. This confidence does not
    extend to POPGP physics or off-system independence truth.

recommendation:
  approve: false
  blocking_findings: 3
  rationale: |-
    Approval is fail closed. RESPONSE-8 repairs all four exact REREVIEW-7 defects and
    the complete historical contract suite remains green. Broader complete campaigns
    nevertheless accept malformed external JSON, raise on structured-receipt depth,
    conflate primary-protocol Boolean and numeric parameters, and accept an external
    repository URI that Git rejects as a bad hostname. VPLAN-SCHEMA-001,
    VPLAN-PROTOCOL-001, and VPLAN-INDEPENDENCE-001 are therefore unresolved. The other
    seven historical findings are resolved; eight historical requested tests are
    satisfied and three are unresolved. No Tier R, G, or E result, external scientific
    validation, or merge authority follows from this review.
```

## Frozen identity and history

The required reviewer branch was clean at `0675367fdcdea86b0710dd2bad2e1a6f53ee92e7`,
with tree `2b09d4cfced4ffe33375eebdb2cf1aec9584b8e4`; the baseline and RESPONSE-8 fix commit
are ancestors. The immutable REREVIEW-7 ref resolves and its blob equals the candidate
copy. The reviewer commit is an equivalent review-branch commit rather than a literal
candidate ancestor, consistent with the preserved workflow. The complete 43-file,
13,662-insertion/147-deletion baseline diff and all surrounding contract code were
audited; `git diff --check` was clean.

## Independent execution

| Command | Observed result |
|---|---|
| `uv sync --frozen` | exit 0; 60 locked packages checked |
| `uv run ruff check .` | exit 0; all checks passed |
| `uv run python scripts/check_tex.py` | exit 0; 652 lines, balanced braces/environments, no Markdown remnants |
| `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | exit 0; 18 passed in 618.53 s |
| `uv run pytest -q` | exit 0; 177 passed in 638.65 s |
| six documented `python -m examples.physics_qg.*` commands | all exit 0; expected finite diagnostics and artifacts regenerated |
| `uv run python scripts/check_validation_artifacts.py` | exit 0; contracts and required visuals valid |
| `git diff --exit-code` and complete status after regeneration | exit 0; clean |

`pdflatex` and `nvcc` were unavailable. No PDF/native build success is claimed. Those
toolchain absences are limitations, not local contract blockers and not scientific
failures.

## Enforceability boundary

The three blockers concern only facts the local executable contract can decide:
strict parsing/totality, exact JSON envelope equality, and validity of the repository
identity it compares. A local validator cannot prove that a real organization is
unaffiliated, code is independently authored, exposure declarations are truthful, or
custody occurred off-system; those remain external maintainer obligations. Likewise,
green CI and regenerated examples do not close the source, geometry, continuum,
closure, Lorentz, hardware, or empirical gaps named by the scientific plan.
