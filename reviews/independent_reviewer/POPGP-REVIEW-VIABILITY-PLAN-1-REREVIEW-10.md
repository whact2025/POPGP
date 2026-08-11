# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-10

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-10"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_10"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "12517855a9f0b24f0e01c2790098733702412ee9"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "b674a5522a19613e1d36c96f34f32ee253feb37f:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-9.md"
builder_response_ref: "12517855a9f0b24f0e01c2790098733702412ee9:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-10.md"
context_hash: "360362303dc118c42e02cbc4714f2587f7c9c163"
context_hash_method: "git rev-parse \"12517855a9f0b24f0e01c2790098733702412ee9^{tree}\""
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
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-9.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-10.md"
  - "pyproject.toml"
  - "uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..12517855a9f0b24f0e01c2790098733702412ee9 (complete 47-file diff)"
  - "git diff 07130dd7c9ad92d85ead1cf3fd90e72d691db93d..12517855a9f0b24f0e01c2790098733702412ee9 (exact six-file review-and-remediation diff)"
access_level: local/public-repository-only
independence_statement: |-
  This was a fresh adversarial re-review in the required dedicated worktree and
  reviewer session. It remained under the builder's human operator and root Codex
  orchestrator, and both seats identify as OpenAI Codex GPT-5; exact model snapshots
  are unavailable, so model separation is false. The immutable history was visible.
  Builder statements were treated only as hypotheses. I independently bound the exact
  candidate commit and tree, audited the complete baseline and remediation diffs,
  executed the locked suites, and created disposable real-Git campaigns and supervised
  hostile-input processes. No final labels, secret seed, private evaluator, credentials,
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
  Changes requested with one blocking historical contract finding. RESPONSE-10's exact
  numeric and Git-process repairs are independently verified. A temporary validator
  snapshot at the REREVIEW-9 candidate accepted the complete hash-reconciled unused
  `1e999` campaign twice; the reviewed candidate rejected the same countermodel twice.
  JSON exponent overflow and nonzero underflow-to-zero fail closed, finite/subnormal/
  exact-zero and 256-character positive boundaries remain accepted, 257-character
  numeric tokens fail, and selected and unselected non-finite values were rejected in
  all 126 kind/media cases.

  The exact historical 16 MiB sparse-zero Tier-E campaign remained live after 45.1
  seconds under the old validator and left a verified Git clone process. After explicit
  cleanup, the candidate returned one deterministic timeout error in 36.61 seconds,
  left neither checkout nor Git process, and accepted a valid-bundle full-campaign
  control in 6.56 seconds. Three direct Windows timeout repetitions killed both parent
  and heartbeat child, froze output, and left no process; positive, fast-nonzero, and
  spawn-error paths were also total. The POSIX process-group branch was not executable
  on this Windows host.

  Broader attack found a small accepted resource countermodel inside the historical
  structured-parser totality scope. A complete hash-reconciled Tier-R campaign put a
  42-level YAML alias DAG in an unused field of a structured receipt. Its maximum
  logical depth is below 128, but traversal revisits shared alias nodes exponentially.
  A READY marker was written and neither public validation nor a RETURNED marker
  completed within the seven-second supervisor; the process tree was then killed.
  This prevents approval despite 20/20 contract tests, 179/179 repository tests, six
  green examples, valid generated artifacts, and clean regeneration. Those are local
  executable-contract results only; no Tier R, G, or E scientific result was produced.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The exact old unused-overflow acceptance was reproduced twice, while the candidate
      rejected it twice. Direct boundary probes accepted depth 128 JSON, finite maximum,
      subnormal, exact-zero, and 256-character numbers; rejected depth 129, duplicate
      keys, constants, overflow, underflow-to-zero, and 257-character tokens; and
      rejected 126 selected/unselected JSON, YAML, and Markdown non-finite cases across
      21 structured receipt kinds. However, a complete Tier-R campaign with a roughly
      one-kilobyte, 42-level YAML alias DAG in an unused output-commitment field wrote
      READY but did not return inside seven seconds. `_structured_receipt_document`
      expands the shared object graph without an identity/visit or work-item budget.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Numeric completeness is repaired. Parser totality remains unresolved for
      bounded-depth YAML alias graphs, so this is the same historical scope rather than
      a duplicate resource finding.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The complete contract suite retained blind exposure, prohibited identity/session
      reuse, custodian-only reveal, chronology, output commitments, manifest
      substitution, resolved path/byte distinction, retention, and structured
      commitment/reveal reconciliation. All 20 tests passed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for local custody consistency; off-system custody remains external."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      Tier-G positive and raw-false controls remained green. Missing or false 3D,
      acceleration/geodesic, same-source lensing and Shapiro, two-potential, laboratory,
      Lorentz, and no-signaling capabilities; alternate pointers/rules; missing
      bindings; and Boolean/number substitutions remained rejected.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the executable gate contract only; no Tier-G result exists."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      Scientific, capability, resource, access, and invalid cause families; complete
      pass/fail/block truth vectors; missing evidence; valid/pending states; precedence;
      and honest external disagreement remained deterministic in the green suite.
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
      Campaign-owned E4 floors and the external packet's E5 floor remained immutable.
      Declared and achieved downgrades, unknown levels, and missing cumulative receipt
      kinds were rejected, while a deliberately stricter declaration remained valid.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Receipt-class semantic truth still requires substantive review."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      Real-Git tests retained nonexistent commit, candidate-tree, packet-rule,
      requirements, executing-contract, manifest/path/hash, schema migration, bundle
      commit/tree/content, and fix-ancestry controls. The response fix and artifact
      commits are ancestors or immutable equivalent-tree carriers of the candidate.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for frozen object identity."

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
      The closed primary-protocol schema retained all extra-authority negatives, while
      scalar, list, and nested mapping Boolean/integer/float substitutions and unequal
      key sets remained rejected by recursive exact-type envelope comparison.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolved for exact recursive JSON envelope identity; scientific adequacy of the
      preregistered parameter choice is outside this local result.

  - finding_id: "VPLAN-INDEPENDENCE-001"
    outcome: verified-resolved
    evidence: |-
      The persisted matrix rejected repository aliases, output reuse, bundle commit/tree
      defects, orchestrator reuse, chronology defects, invalid disagreement, strict
      comparison substitutions, residual percent forms, multi-encoding, and invalid
      hosts while retaining the positive typed clean-room case.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolved for local URI/provenance consistency only. Organization, authorship,
      exposure, affiliation, and control remain externally verified facts.

  - finding_id: "VPLAN-RESOURCE-001"
    outcome: verified-resolved
    evidence: |-
      The old exact 16 MiB Tier-E campaign was still live after 45.1 seconds with a Git
      clone and no RETURNED marker. The candidate returned one controlled timeout error
      in 36.61 seconds, removed the checkout, and left no Git process; a valid-bundle
      full-campaign control returned no errors in 6.56 seconds. Three direct Windows
      helper repetitions timed out in 1.89-2.13 seconds under a 0.8-second execution
      deadline, killed parent and child PIDs, and stopped heartbeat writes. Exit-zero,
      exit-seven, and missing-executable paths returned or raised promptly and retained
      no output handles or residue.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolved for the exact Git process-tree scope on Windows. POSIX process-group
      execution remains a CI/external-platform limitation; the alias-DAG failure is
      counted under the pre-existing structured-parser totality scope.

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The numeric parser and all-receipt traversal close the exact exponent and
      non-finite omissions, with positive boundaries retained. The requested portable
      fail-closed matrix is still incomplete because the supervised full-campaign YAML
      alias DAG did not return. Persist a bounded-work/alias regression at the public
      entry point and prove prompt deterministic failure without relying on outer kill.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Unresolved only for bounded-depth shared YAML object graphs."

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Exposure, session/identity reuse, custodian authority, reveal order, output and
      manifest hashes, path/byte distinction, structured receipts, retention, and
      missing-field mutations all remained covered and passed.
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
      mutations remained covered and green.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for frozen Git and bundle provenance."

  - requested_test_id: "TST-VPLAN-FREEZE-003"
    outcome: verified-satisfied
    evidence: |-
      Canonical preregistration fields, paths, bytes, budgets, commands, and mutations
      remained bound to the protocol commit and rejected when altered.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-PROTOCOL-001"
    outcome: verified-satisfied
    evidence: |-
      Historical extra-authority controls and the exact-envelope positive remain green;
      scalar and nested/list Boolean-integer-float substitutions remain rejected.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for exact primary-protocol envelope equality."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      Distinct typed organization/operator/implementation, exposure, prediction,
      commitment, and generic/malformed receipt negatives remained green; the positive
      clean-room fixture still validates.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for typed clean-room identity and receipt scope."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    outcome: verified-satisfied
    evidence: |-
      Candidate custody output, path/byte distinction, bundle provenance, external
      orchestrator, typed comparison, pointer/tolerance/order, accumulated aliases,
      residual percent, multi-encoding, and invalid-host cases failed closed; the valid
      positive retained acceptance.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for locally enforceable repository and comparison integrity."

  - requested_test_id: "TST-VPLAN-RESOURCE-001"
    outcome: verified-satisfied
    evidence: |-
      The exact old hang was reproduced under an outer supervisor. On the candidate the
      complete malformed-bundle campaign returned one stable timeout error, left no Git
      descendant or checkout, and the valid bundle passed. Repeated direct helper runs
      verified descendant termination and no heartbeat/output retention; prompt normal,
      nonzero, and spawn-error paths were positive controls.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied on Windows; the POSIX branch remains unexecuted locally."

predictions:
  experiment_id: "bounded-yaml-alias-traversal-remediation"
  predicted_outcome: |-
    A complete remediation will make traversal identity-aware or impose a deterministic
    node/work/alias budget, reject the exact small DAG promptly through both public
    validation calls, retain ordinary aliases and all numeric positive boundaries, and
    leave no process or temporary output.
  predicted_failure_mode: |-
    A depth-only patch will continue to revisit a shallow shared object graph
    exponentially even though neither parser recursion nor logical depth exceeds 128.
  confidence_statement: |-
    High for local executable behavior: the complete campaign wrote READY and exceeded
    its supervisor, while direct cyclic aliases terminate through the existing depth
    check and all ordinary numeric controls were deterministic. Confidence does not
    extend to off-system independence truth, POSIX cleanup execution, or POPGP physics.

recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    Approval remains fail closed. Ten historical findings are independently resolved,
    including the exact numeric remediation's finite-value behavior and the exact Git
    process-tree resource repair. Eleven historical requested tests are satisfied.
    The remaining historical parser-totality blocker admits a sub-kilobyte bounded-depth
    YAML alias graph that stalls public campaign validation, so one executable contract
    blocker remains. No Tier R, G, or E result, external scientific validation, or merge
    authority follows from this review.
```

## Frozen identity and immutable history

The required reviewer branch began clean at
`12517855a9f0b24f0e01c2790098733702412ee9` with tree
`360362303dc118c42e02cbc4714f2587f7c9c163`; the baseline is an ancestor. The immutable
prior-review ref recorded by RESPONSE-10 resolves at `b674a5522a19613e1d36c96f34f32ee253feb37f`.
The candidate's own ancestry carries a byte-identical review artifact at
`3b04571bb50b612880be70da60c6b5857155564a`; both carrier commits have tree
`467abbc60451506c83aad65b1ac1f9cdae90bf00`. The response ref resolves at the reviewed
candidate. The complete 47-file, 15,327-insertion/147-deletion baseline diff and exact
six-file, 890-insertion/12-deletion review-and-remediation diff were audited, and both
passed `git diff --check`. No candidate file was changed.

## Independent execution

| Command or probe | Observed result |
|---|---|
| `uv sync --frozen` | exit 0; CPython 3.11.15 environment created from the lock; 60 packages installed |
| `uv run ruff check .` | exit 0; all checks passed |
| `uv run python scripts/check_tex.py` | exit 0; 652 lines, balanced braces/environments, no Markdown remnants |
| separate `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | exit 0; 20 passed in 666.87 s |
| `uv run pytest -q` | exit 0; 179 passed in 745.55 s |
| all six documented `python -m examples.physics_qg.*` commands | all exit 0; expected finite diagnostics and artifacts regenerated |
| `uv run python scripts/check_validation_artifacts.py` | exit 0; contracts and required visuals valid |
| regeneration `git diff --exit-code` | exit 0 before this sole review artifact was created |
| old/current complete Tier-R overflow campaign, two calls each | old returned `[]` twice; candidate returned stable parse/binding errors twice |
| 126-case structured non-finite matrix | every selected/unselected JSON, YAML, and Markdown case rejected across 21 kinds |
| exponent, integer, token, nesting, duplicate, and malformed controls | overflow/underflow/257/depth-129/duplicates failed; finite/subnormal/zero/256/depth-128 positives passed |
| old/current exact 16 MiB sparse-bundle full campaigns | old live at 45.1 s with orphan Git; candidate one error in 36.61 s, no checkout/process |
| valid-bundle Tier-E full campaign | returned `[]` in 6.56 s |
| Windows bounded-process helper matrix | three descendant kills stable; zero/nonzero/spawn-error paths prompt; no retained output or residue |
| complete Tier-R YAML alias-DAG campaign | READY written; no return inside 7 s; supervised process tree terminated |

The deliberately reproduced old orphan was identified by PID and command line, killed,
and its temporary checkout removed before candidate probes continued. Final inspection
found no matching Git process and no `popgp-external-bundle-*` checkout. `pdflatex` and
`nvcc` were unavailable, so no PDF or native-CUDA build success is claimed. The test
fixtures are synthetic, and process-tree execution was Windows-only; Linux/POSIX CI is
an external action, not evidence obtained here.

## Enforceability boundary

The remaining blocker is a fact the local executable contract can decide: termination
of hash-valid structured input processing. A local validator cannot prove unaffiliated
organization, independent authorship, truthful exposure declarations, or off-system
custody. Green suites and examples do not close the source, geometry, continuum,
closure, Lorentz, hardware, or empirical gaps already named by the scientific plan.
