# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-3

```yaml
artifact_schema_version: 1
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-3"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
review_date: "2026-08-10"
commit_reviewed: "e8f7862910edcd7a949ffff88b28446ef2334f31"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "a1068262fbb9df902a11e73c46f3d09d92f881f7:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-2.md"
builder_response_ref: "e8f7862910edcd7a949ffff88b28446ef2334f31:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-3.md"
context_hash: "a742115d77a6d104703459cdee92ef459f3c0597"
context_hash_method: "git rev-parse \"e8f7862910edcd7a949ffff88b28446ef2334f31^{tree}\""
files_reviewed:
  - "C:/src/POPGP-review-viability-plan-rereview-3/.github/workflows/ci.yml"
  - "C:/src/POPGP-review-viability-plan-rereview-3/README.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/framework.md (empirical-compatibility, falsification, and claim-scope sections)"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/governance/REVIEWER_IDENTITY.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/scientific_hardening/PROJECT_PLAN.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/scientific_hardening/DECISIONS.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/docs/scientific_hardening/REPRODUCIBILITY.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-1.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-1.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-2.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-2.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-3.md"
  - "C:/src/POPGP-review-viability-plan-rereview-3/schemas/viability/campaign-v2.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-3/schemas/viability/packet-v2.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-3/schemas/viability/protocol-manifest-v2.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-3/schemas/viability/requirements-v2.json"
  - "C:/src/POPGP-review-viability-plan-rereview-3/schemas/viability/independent-review-v1.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-3/schemas/viability/review-response-v1.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-3/schemas/viability/independent-rereview-v1.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-3/scripts/check_viability_campaign.py"
  - "C:/src/POPGP-review-viability-plan-rereview-3/tests/unit/test_viability_campaign_contract.py"
  - "C:/src/POPGP-review-viability-plan-rereview-3/examples/physics_qg/* (all six documented example entry points and regenerated validation artifacts)"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..e8f7862910edcd7a949ffff88b28446ef2334f31 (complete, all 28 changed files)"
access_level: local/public-repository-only
independence_statement: |-
  This is a fresh reviewer task under the same human operator and root orchestrator as
  the builder. It is process separation, not external scientific independence. The
  builder and reviewer identities exposed to this run are in the same OpenAI Codex
  GPT-5 model family; no exact snapshot/version is exposed, so the reviewer version is
  `unknown` and model separation is false. Re-review necessarily received all earlier
  reviews and builder responses. Builder claims were not accepted as proof: the exact
  frozen tree, full history diff, schemas, validator, tests, surrounding scientific
  contracts, and prior mutation families were read and independently exercised. No
  final labels, secret seeds, private evaluator logic, credentials, private hardware,
  or empirical campaign data were available. This review is not an unaffiliated
  replication, empirical validation, or external scientific confirmation of POPGP.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "OpenAI Codex GPT-5"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  Changes requested with three unresolved blocking findings. The third remediation
  closes the concrete minimal-artifact, malformed-outcome, duplicate-key, immutable-ref,
  and post-protocol substitution counterexamples from re-review 2. A schema-complete
  positive review chain validates, all seven canonical preregistration field families
  and a path substitution are rejected after current receipt hashes and the packet
  self-hash are recomputed, and the prior custody, Tier G, outcome, dependency,
  evidence-floor, and Git-freeze mutations remain closed. VPLAN-CUSTODY-001,
  VPLAN-SCI-001, VPLAN-OUTCOME-001, VPLAN-DEP-001, VPLAN-EVIDENCE-001,
  VPLAN-FREEZE-002, and VPLAN-FREEZE-003 are verified-resolved; their requested tests
  are verified-satisfied.

  VPLAN-SCHEMA-001 remains unresolved because malformed external input still escapes
  the advertised fail-closed API and schema-valid governance receipts can contain
  internally contradictory independence/provenance claims. An integer nested
  `dependencies` field raises `TypeError`; a hash-valid malformed YAML output-commitment
  receipt raises `yaml.parser.ParserError`; identical reviewer/builder model identities
  can declare model separation true; and a response can cite an unrelated builder
  identity and nonexistent fix commit while the complete chain validates.

  Two new blockers were found. VPLAN-PROTOCOL-001 shows that the primary protocol is
  compared only on seven selected fields; extra competing experiment-defining keys are
  accepted even though the document is advertised as the canonical protocol.
  VPLAN-INDEPENDENCE-001 shows that a complete Tier E campaign validates when every
  seat across every packet has the same operator and model and the VIA-900 external
  capability/receipt is self-declared. This contradicts the runbook's unaffiliated
  clean-room pass condition. No POPGP viability tier is demonstrated here.

findings:
  - id: "VPLAN-PROTOCOL-001"
    severity: high
    category: governance
    location: "scripts/check_viability_campaign.py:888-926; schemas/viability/packet-v2.schema.json:340-411; docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md:537-547"
    evidence: |-
      `_validate_preregistration` constructs `expected_document_fields` for parameters,
      measurement, uncertainty, statistics, budget, commands, and mutation plan, then
      compares only `document.get(field)` for those names plus `packet_id`. There is no
      schema for the primary protocol document, no `additionalProperties: false`
      envelope, and no exact object comparison.

      I modified the fixture constructor before the protocol commit so every current
      receipt hash, preregistration artifact hash, Git blob, and manifest rule hash was
      honest and immutable. The primary JSON additionally contained
      `execution_threshold_override: 0.10` and
      `unregistered_exclusion_rule: discard all nonpassing replicates`. The canonical
      packet block still declared threshold 0.95 and no such exclusion. The complete
      Tier R campaign returned `[]` and printed
      `PRIMARY_PROTOCOL_EXTRA_EXPERIMENT_FIELDS_ACCEPTED True []`.
    finding: |-
      The purported primary protocol is not required to equal one canonical protocol
      envelope. It may carry experiment-defining fields outside the canonical packet
      block while all immutable Git and receipt checks pass.
    failure_scenario: |-
      A preregistered command reads an extra top-level threshold override or exclusion
      rule from the primary JSON instead of the canonical `parameters` and
      `statistical_analysis` fields. Both definitions are frozen before holdout, but
      they conflict. The validator reports a valid campaign because it ignores the
      competing keys, leaving the executor or adjudicator to choose which definition
      governed the raw Boolean result.
    consequence: |-
      Threshold, exclusion, measurement, or resource semantics can be ambiguous at the
      exact boundary intended to eliminate post-selection. Immutable bytes do not make
      two competing protocol definitions canonical.
    required_action: |-
      Define and freeze a versioned primary-protocol JSON Schema with an explicit
      envelope and `additionalProperties: false`, then compare the complete parsed
      document to the canonical packet preregistration representation. Reject extra or
      competing experiment-defining keys and add TST-VPLAN-PROTOCOL-001.
    verification: confirmed-by-execution
    blocking: true

  - id: "VPLAN-INDEPENDENCE-001"
    severity: high
    category: claim
    location: "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md:19-22, 29-33, 480-502; schemas/viability/requirements-v2.json:218-240; scripts/check_viability_campaign.py:930-1096, 1682-1715"
    evidence: |-
      The runbook defines Tier E as unaffiliated replication with independently written
      code and says the same operator/orchestrator remains internal evidence. The
      validator enforces selected session and evaluator identity separation, but it
      never requires VIA-900's operator or organization to differ from the internal
      campaign, never binds `independent-core-code` to independent code provenance, and
      treats `unaffiliated-operator` as another self-declared raw Boolean.

      The repository's own complete Tier E fixture uses `operator: test-operator` and
      `model_identity: test-model` for every seat of every packet. Its
      `external-replication`, `independent-implementation`, and `blinded-prediction`
      receipt kinds are satisfied by generic local fixture bytes, while VIA-900 raw
      capability Booleans are true. Independent execution printed
      `TIER_E_SAME_OPERATOR_ACCEPTED True operators=['test-operator'] models=['test-model']`
      and `validate_campaign(...)` returned `[]`.
    finding: |-
      Tier E can mechanically pass with no unaffiliated operator or independently
      written implementation. The campaign contract converts internal, same-operator
      assertions into the externally supported claim that its prose explicitly forbids.
    failure_scenario: |-
      One operator runs all internal and purported external seats in separate sessions,
      copies the original core implementation, emits generic files labeled as external
      receipts, and sets the VIA-900 capability Booleans true. All Tier G packets and
      VIA-900 validate, so the campaign decision becomes Tier E even though the only
      evidence is internal process separation.
    consequence: |-
      An external-support claim can be promoted without the independence condition that
      defines it, overstating both evidence level E5 and POPGP's scientific status.
    required_action: |-
      Add a typed external-replication contract that binds an unaffiliated organization
      and operator, independent repository/implementation provenance, prior exposure,
      blinded predictions, output commitments, and cross-implementation results to the
      VIA-900 receipts. Enforce operator/affiliation and core-code independence against
      the internal campaign identities, or keep Tier E externally gated and impossible
      for this local validator to mark passed. Add TST-VPLAN-INDEPENDENCE-001.
    verification: confirmed-by-execution
    blocking: true

requested_tests:
  - id: "TST-VPLAN-PROTOCOL-001"
    description: |-
      Build a passing campaign whose primary protocol is a correct immutable Git blob
      and matches every named canonical field, but adds a competing threshold,
      exclusion rule, measurement procedure, command, or resource override outside the
      canonical envelope. The campaign command must reject every extra key. Retain a
      positive exact-envelope protocol and all post-protocol substitution mutations.
    rationale: |-
      Hash equality proves byte identity, not that the primary document has exactly one
      unambiguous experiment definition.
    blocking: true

  - id: "TST-VPLAN-INDEPENDENCE-001"
    description: |-
      Start from a complete Tier E campaign and reuse the same operator, organization,
      model/session family, or core implementation provenance for the purported
      VIA-900 replication. Also replace typed external receipts with generic bytes or
      omit blinded prediction/output bindings. Every case must prevent Tier E pass,
      while a genuinely distinct typed clean-room fixture remains positive.
    rationale: |-
      A self-declared `unaffiliated-operator: true` Boolean cannot establish the
      external independence that defines Tier E.
    blocking: true

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The remediation is effective for the re-review-2 counterexamples it persisted: a
      complete positive review/response/re-review chain validated; minimal response and
      re-review artifacts were rejected by their versioned schemas; a malformed outcome
      returned schema errors; wrong immutable refs, duplicate campaign YAML, and
      duplicate raw-result JSON were rejected. Review item coverage, blocker counts,
      candidate trees, baseline refs, and receipt/ref pairing are checked against exact
      Git blobs.

      Totality and semantic provenance remain open. Directly calling
      `validate_requirements` with `packets.VIA-010.dependencies: 7` raised uncaught
      `TypeError: 'int' object is not iterable`: lines 462-469 sanitize a local value,
      but the DFS at lines 511-518 iterates the original malformed field. A hash-valid
      `application/yaml` output-commitment receipt containing
      `packet_id: [unterminated` raised uncaught `yaml.parser.ParserError`; the catches
      at lines 1033-1036 and 1063-1066 omit `yaml.YAMLError`.

      A schema-valid immutable initial review with reviewer and builder model identities
      equal, `reviewer_model_differs_from_builder: true`, the same operator but
      `shared_operator: false`, and a nonreproducible context-hash method validated. A
      complete response/re-review chain also validated when the response claimed an
      unrelated builder identity and a nonexistent 40-hex fix commit. The exact outputs
      were `CONTRADICTORY_INDEPENDENCE_AND_CONTEXT_METHOD_ACCEPTED True []` and
      `UNBOUND_RESPONSE_IDENTITY_AND_NONEXISTENT_FIX_COMMIT_ACCEPTED True []`.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Presence and immutable bytes are now enforced, but malformed nested input can
      still escape and typed identity/commit/independence fields can contradict their
      own cross-document facts. Make every public validation path total, validate
      independence facts against identities/operators, and resolve nonempty fix commits.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The dedicated suite passed, and an independent seven-case matrix rejected blind
      runner exposure, shared builder/falsifier session, evaluator/builder identity
      reuse, builder-authorized reveal, builder output commitment, output hash mismatch,
      and changed post-reveal seed bytes. Earlier role/exposure/canonicalization/
      retention cases remain persisted. The malformed-YAML exception is assigned to
      VPLAN-SCHEMA-001 because the custody invariants are correct once input is parsed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the local receipt-consistency contract; declarations remain assertions, not external custody proof."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      A complete Tier G fixture validated. Independent mutations rejected a binding-free
      pass rule, alternate capability pointer, alternate capability rule, raw false 3D
      recovery/lensing gates, and integer-for-Boolean substitution: 5/5. All nine VIA-700
      capabilities remain canonical. The separate Tier E independence bypass is recorded
      as VPLAN-INDEPENDENCE-001 and does not reopen the Tier G countermodel finding.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for Tier G gate consumption; this is not evidence that POPGP passes any gate."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      The independent eight-vector pass/fail/block truth table returned the intended
      result for all 8 cases: exactly-one-true rows validated with matching outcomes and
      zero/multiple-true rows returned nonexclusive/incomplete errors. All five invalid
      causes validated only as invalid-round/pending cases (5/5), and the committed full
      cause matrix passed. No outcome mutation raised an exception.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Malformed structured-receipt parsing remains under VPLAN-SCHEMA-001."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      `validate_requirements` accepted the canonical registry; its 41 edges are all
      strictly lower-wave and every tier contains each packet's transitive closure.
      Independent campaigns rejected a pending prerequisite for every dependent target
      VIA-010, VIA-100, VIA-150, VIA-200, VIA-400, VIA-500, VIA-600, VIA-700, VIA-800,
      and VIA-900 (10/10).
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for well-typed canonical requirements; malformed nested-type totality remains under VPLAN-SCHEMA-001."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      Independent achieved-evidence downgrades were rejected for every E4 packet
      VIA-300, VIA-400, VIA-500, VIA-600, VIA-700, VIA-800 and E5 packet VIA-900 (7/7).
      The dedicated tests retained declared-floor, unknown-level, achieved-below-declared,
      and stricter-declaration coverage.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Evidence-label truth still needs substantive review; Tier E identity binding is the separate new blocker."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      The dedicated tests using real temporary Git commits passed. Independent
      representative mutations rejected a nonexistent candidate, wrong candidate tree,
      changed packet rule with recomputed self-hash, supplied requirements downgrade,
      and post-protocol executing-validator change (5/5). Static and committed cases
      retain baseline/protocol object, manifest/path/hash, requirements, checkout,
      path-escape, and v1 migration coverage.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Git is immutable local provenance, not an external timestamp or custody service."

  - finding_id: "VPLAN-FREEZE-003"
    outcome: verified-resolved
    evidence: |-
      A complete preregistered fixture validated. After the protocol commit, I changed
      each of parameters, measurement procedure, uncertainty procedure, statistical
      analysis, resource budget, commands, and mutation plan at the same campaign path;
      for every mutation I updated the current protocol receipt SHA-256, the packet
      preregistration copy/artifact SHA-256, and `protocol_rule_sha256`. All seven were
      rejected by both the frozen manifest rule and original Git-blob hash. A campaign
      path substitution with recomputed receipt and packet hashes was also rejected.
      The targeted v3 matrix was 14/14 including the complete review-chain cases.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The requested post-protocol substitution defect is closed. The distinct ambiguity
      from extra fields already present in the frozen primary protocol is recorded as
      VPLAN-PROTOCOL-001 rather than counted twice here.

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The 13 committed contract tests pass, and independent positive/minimal/malformed/
      immutable-ref/duplicate-key cases reproduced the intended results. The regression
      does not exercise malformed nested requirement types or malformed YAML custody
      documents, both of which raise. It also lacks contradictory identity/independence
      and nonexistent-fix-commit cases. The requested fail-closed totality and complete
      governance binding are therefore not satisfied.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Add the accepted exception and semantic-provenance counterexamples to the same public API regression."

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Committed and independent cases cover blind exposure, role/session reuse,
      evaluator-only authority, reproduced-phase reveal, runner identity/output hash,
      structured receipt reconciliation, manifest substitution, canonicalization,
      retention, and required custody records. The independent representative matrix
      was 7/7.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for well-formed local custody documents."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: verified-satisfied
    evidence: |-
      The complete Tier G positive case and five independent raw-binding/gate mutations
      behaved as required. The raw-false 3D/lensing countermodel cannot pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the requested Tier G countermodel."

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    outcome: verified-satisfied
    evidence: |-
      All eight Boolean predicate vectors, all five invalid cause codes, and the
      committed scientific/capability/resource/access/missing-receipt/precedence rows
      returned deterministic outcomes or validation errors without raising.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-DEP-001"
    outcome: verified-satisfied
    evidence: |-
      The canonical 41-edge lower-wave DAG and tier closure passed, and pending
      prerequisites were rejected for all ten dependent targets.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the canonical registry and holdout-order rule."

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      Every E4/E5 achieved-floor downgrade was independently rejected (7/7), and the
      committed test covers declared downgrades, unknown levels, and a stricter packet
      declaration.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-FREEZE-002"
    outcome: verified-satisfied
    evidence: |-
      Real-Git committed and independent cases reject nonexistent objects, wrong tree,
      packet-rule change, requirements downgrade, executing-contract change,
      manifest/path/hash substitutions, escapes, and version migration.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the requested Git/rule/requirements freeze scope."

  - requested_test_id: "TST-VPLAN-FREEZE-003"
    outcome: verified-satisfied
    evidence: |-
      Same-path changes to all seven canonical preregistration field families and a
      campaign-path substitution were independently rejected after recomputing current
      receipt hashes and the packet self-hash. Original Git blob identity remained the
      decisive immutable boundary.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for post-protocol substitutions; exact-envelope coverage is requested separately."

predictions:
  experiment_id: "TST-VPLAN-SCHEMA-001, TST-VPLAN-PROTOCOL-001, and TST-VPLAN-INDEPENDENCE-001"
  predicted_outcome: |-
    A complete remediation will return validation errors for every malformed nested
    input and structured receipt, reject contradictory identity/independence and false
    fix-commit provenance, reject any extra experiment-defining key in the primary
    protocol envelope, and make a same-operator or copied-core Tier E campaign
    non-passing while preserving the positive internal Tier R/G chains.
  predicted_failure_mode: |-
    Catching only the two reproduced exceptions will leave other public-parser paths
    partial. Comparing only seven selected protocol fields will continue to permit a
    competing frozen override. Requiring different session IDs or a raw Boolean named
    `unaffiliated-operator` will continue to relabel internal process separation as
    external replication unless typed actors and code provenance are reconciled.
  confidence_statement: |-
    High. Each blocker is an accepted campaign or uncaught exception against the exact
    frozen public validator. This is confidence in contract behavior, not validation of
    any POPGP physical claim.

recommendation:
  approve: false
  blocking_findings: 3
  rationale: |-
    One prior blocker remains open because advertised fail-closed totality and semantic
    governance provenance are incomplete. Two new blockers permit ambiguous primary
    protocol authority and same-operator promotion to the externally supported Tier E.
    The requested post-protocol freeze, all six previously resolved finding/test pairs,
    and their regression matrices remain closed, but approval requires all three
    blockers and the two new requested tests to be independently remediated.
```

## Frozen-tree, receipt, history, and diff audit

The worktree was clean and on the requested reviewer branch before review. The named
prior-review commit `a1068262...` and the line's ancestor `d0cf72cd...` have the same
parent, tree, and re-review blob `043f75ce420ebac572a33749d4a3e61b10bec950`.
The response-3 blob in the frozen candidate is
`f77cc6e7c8d13f46fc5ba005f31f3f67586664b5`; both immutable refs used above therefore
identify the exact locally read bytes.

| Command | Exit | Observed result |
|---|---:|---|
| `git branch --show-current` | 0 | `review/adversarial-viability-runbook-rereview-3` |
| `git rev-parse HEAD` | 0 | `e8f7862910edcd7a949ffff88b28446ef2334f31` |
| `git rev-parse "HEAD^{tree}"` | 0 | `a742115d77a6d104703459cdee92ef459f3c0597` |
| `git status --porcelain=v1 --untracked-files=all` | 0 | empty before review execution |
| `git merge-base --is-ancestor 0bdff136c3c5fba8d8868fdd6355f3f824245a8e e8f7862910edcd7a949ffff88b28446ef2334f31` | 0 | original candidate is an ancestor |
| `git diff --stat 0bdff136..e8f7862` | 0 | 28 files; 7,823 insertions; 144 deletions |
| `git diff --name-status 0bdff136..e8f7862` | 0 | implementation, schemas, tests, docs, and preserved review/response history enumerated |
| `git diff --check 0bdff136..e8f7862` | 0 | no output |
| local relative-Markdown-link scan over changed Markdown | 0 | `LOCAL_MARKDOWN_LINK_CHECK: PASS` |

The complete history diff was read, including the exact validator, every schema and
requirements object, the complete 1,879-line contract test, templates, protocol prose,
lockfile/dependency changes, and all prior review artifacts. Builder summaries were used
to identify claimed remediation only, never as resolution evidence.

## Authoritative suite and repository checks

Every command from `.github/workflows/ci.yml` was run at the frozen handoff.

| Command | Exit | Observed result |
|---|---:|---|
| `uv sync --frozen` | 0 | CPython 3.11.15; fresh `.venv`; 60 locked packages installed |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | 652 lines; balanced braces/environments; no Markdown remnants |
| `uv run pytest -q` | 0 | `172 passed in 101.56s` |
| `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | 0 | `13 passed in 78.24s` |
| `uv run python -m examples.physics_qg.chain_1d` | 0 | contiguous blocks; D*=1; artifacts regenerated |
| `uv run python -m examples.physics_qg.grid_2d` | 0 | precision/recall 1.0; D*=2; artifacts regenerated |
| `uv run python -m examples.physics_qg.gravity_well` | 0 | Green-function diagnostic pass |
| `uv run python -m examples.physics_qg.source_law` | 0 | relative-entropy slope 1.999684; modular slope 1.0 |
| `uv run python -m examples.physics_qg.source_law_many_body` | 0 | quadratic, Kubo--Mori, Richardson, and spreading diagnostics reproduced |
| `uv run python -m examples.physics_qg.ca_model` | 0 | PNG/GIF/JSON regenerated; documented analogy limitations retained |
| `uv run python scripts/check_validation_artifacts.py` | 0 | validation contracts and required visual outputs valid |
| `git status --porcelain=v1 --untracked-files=all` after regeneration | 0 | empty |

`pdflatex` and `nvcc` were unavailable. They are not commands in the authoritative CI
suite, and no remediation result claims a native/PDF campaign pass. Their absence limits
native/PDF coverage but does not cause any validator counterexample.

## Independent adversarial evidence

All campaign mutations used disposable `TemporaryDirectory` repositories, real Git
commits/blobs, the candidate fixture constructor, and the public
`validate_campaign(..., repo_root=...)` or `validate_requirements(...)` entry point. The
shell form was a PowerShell inline here-string piped to `uv run python -`; no candidate
file was changed. Representative exact operations were:

```python
# Complete review-chain positive and totality negatives.
_attach_review_round(packet_path, frozen_repo, "independent-complete")
assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []
requirements["packets"]["VIA-010"]["dependencies"] = 7
validate_requirements(requirements)  # raised TypeError: 'int' object is not iterable
receipt_path.write_text("packet_id: [unterminated\n", encoding="utf-8")
receipt.update(media_type="application/yaml", sha256=sha256(receipt_path))
validate_campaign(campaign_path, repo_root=frozen_repo["root"])  # raised ParserError

# Rehash every mutable/current protocol record after a post-protocol field change.
protocol[field] = changed_value
receipt["sha256"] = sha256(protocol_path)
packet["preregistration"][field] = changed_value
packet["preregistration"]["protocol_artifacts"][0]["sha256"] = receipt["sha256"]
packet["protocol_rule_sha256"] = packet_rule_sha256(packet)
assert validate_campaign(campaign_path, repo_root=frozen_repo["root"])

# Exact-envelope and external-independence counterexamples are frozen before validation.
primary_protocol["execution_threshold_override"] = 0.10
primary_protocol["unregistered_exclusion_rule"] = "discard all nonpassing replicates"
assert validate_campaign(campaign_path, repo_root=frozen_repo["root"]) == []
operators = {
    seat["operator"]
    for packet_path in tier_e_packets.values()
    for seat in _load(packet_path)["seats"].values()
}
assert operators == {"test-operator"}
assert validate_campaign(tier_e_campaign, repo_root=frozen_repo["root"]) == []
```

| Mutation/check | Observed result |
|---|---|
| Complete schema-valid review/response/re-review chain | accepted |
| Minimal response/re-review; malformed outcome; wrong immutable ref | all rejected without raising |
| Duplicate campaign YAML and raw-results JSON keys | both rejected |
| Nested integer `dependencies` in `validate_requirements` | **uncaught `TypeError`** |
| Malformed YAML output-commitment receipt with current hash | **uncaught `yaml.parser.ParserError`** |
| Contradictory immutable independence facts/context method | **accepted** |
| Unrelated response identity plus nonexistent fix commit | **accepted** |
| Same-path post-protocol changes to seven canonical field families after rehash | all 7 rejected by manifest and Git blob |
| Post-protocol campaign-path substitution after receipt/self rehash | rejected |
| Primary protocol with extra threshold/exclusion authority, frozen in Git | **accepted** |
| Complete Tier E with one operator/model for every seat | **accepted** |
| Tier G raw-binding/capability/type mutation matrix | 5/5 rejected |
| Pass/fail/block Boolean truth table and invalid causes | 8/8 and 5/5 deterministic |
| Custody regression matrix | 7/7 rejected |
| Canonical DAG/lower-wave/tier closure | pass; 41 edges |
| Pending prerequisite for every dependent target | 10/10 rejected |
| Every E4 packet and VIA-900 E5 achieved downgrade | 7/7 rejected |
| Representative Git/tree/rule/requirements/contract regressions | 5/5 rejected |

## Limitations and recommendation

These synthetic mutations test executable contract behavior, not the truth of off-system
identity, affiliation, exposure, or scientific-quality declarations. Git provides local
immutability but not independent timestamping or institutional custody. No hidden
campaign, external group, private evaluator, native CUDA environment, TeX build, or
empirical dataset was available. The shared operator/orchestrator/model-family review is
useful process separation only and must not be described as external scientific
validation.

Changes requested: **three unresolved blocking findings**. VPLAN-SCHEMA-001 remains open;
VPLAN-PROTOCOL-001 and VPLAN-INDEPENDENCE-001 are new. VPLAN-FREEZE-003 and its requested
test are verified closed, as are every previously resolved finding/test pair. A later
fresh re-review must verify total input safety and governance provenance, one exact
primary protocol authority, and a typed externally independent VIA-900 gate. Green CI
and same-family agent agreement do not close those blockers or demonstrate POPGP
viability.
