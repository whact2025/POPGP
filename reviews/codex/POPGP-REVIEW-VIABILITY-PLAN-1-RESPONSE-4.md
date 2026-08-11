# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-4

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-4"
response_round: 4
response_date: "2026-08-10"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-3"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-3.md"
review_commit: "7f24eb727d4be69ddb145e2080664fc591936dfd"
candidate_commit_reviewed: "e8f7862910edcd7a949ffff88b28446ef2334f31"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  All three blockers and requested tests from re-review 3 were accepted and
  implemented in 4a069d58e95d828701a09129a677d987c04b5e87. Public validation now
  returns errors for the reproduced malformed nested requirement and malformed YAML
  receipt inputs. Versioned v2 review and response schemas bind model, operator,
  session, orchestrator, and builder organization identities to packet-seat facts;
  response fix commits must exist and be ancestors of the re-reviewed candidate.

  A closed primary-protocol schema and exact object equality remove competing frozen
  threshold, exclusion, command, and resource authorities. Tier E now requires a
  frozen typed external-replication contract with distinct organization, operator,
  agent, model, session, and repository provenance; blinded prediction and reveal;
  external output commitment and bytes; and an explicit candidate/external agreement
  comparison. These controls make a future campaign fail closed. They do not supply an
  unaffiliated replication or demonstrate POPGP scientific viability.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Nested requirements are type-checked before graph traversal, and structured
      custody YAML parse failures are converted into validation errors. Canonical
      context-hash methods and reviewer/builder model, operator, session, and
      orchestrator comparisons are reconciled to the packet builder seat. Builder
      responses use a new v2 schema with model, operator, session, orchestrator, and
      organization fields; all are reconciled to the packet, and each nonempty fix
      commit must resolve and precede the candidate audited by the paired re-review.
      The published v1 schemas remain unchanged; stronger requirements are versioned
      as v2 and frozen by the protocol manifest.
    changed_files:
      - "schemas/viability/independent-review-v2.schema.json"
      - "schemas/viability/independent-rereview-v2.schema.json"
      - "schemas/viability/review-response-v2.schema.json"
      - "schemas/viability/packet-v2.schema.json"
      - "schemas/viability/protocol-manifest-v2.schema.json"
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/governance/REVIEWER_IDENTITY.md"
      - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
      - "docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
    fix_commits: ["4a069d58e95d828701a09129a677d987c04b5e87"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 16 passed in 141.83 s"
      - command: "uv run pytest -q"
        result: "exit 0; 175 passed in 164.81 s"
      - command: "uv run ruff check ."
        result: "exit 0; all checks passed"
    residual_risk: "Repository checks can reconcile declared identities and Git provenance but cannot independently prove off-system identity, affiliation, exposure, or authorship."
    disagreement_ref: ""

  - finding_id: "VPLAN-PROTOCOL-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The new primary-protocol-v1 schema is a closed envelope with
      additionalProperties false. The validator loads it with duplicate-key rejection
      and compares the complete parsed object to the one canonical packet-derived
      protocol object, rather than selecting seven fields. Extra threshold overrides,
      exclusion rules, commands, or resource authorities therefore invalidate the
      packet even when preregistered honestly and frozen in Git.
    changed_files:
      - "schemas/viability/primary-protocol-v1.schema.json"
      - "schemas/viability/protocol-manifest-v2.schema.json"
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
    fix_commits: ["4a069d58e95d828701a09129a677d987c04b5e87"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_primary_protocol_rejects_competing_experiment_fields"
        result: "exit 0; honestly frozen extra threshold and exclusion fields were rejected"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_protocol_content_and_budget_are_frozen_before_holdout"
        result: "exit 0; all post-freeze canonical field/path/hash mutations remained rejected"
    residual_risk: "A unique frozen protocol may still be scientifically inadequate; reviewers must assess its power, measurements, exclusions, and uncertainty model before holdout execution."
    disagreement_ref: ""

  - finding_id: "VPLAN-INDEPENDENCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      VIA-900 now requires a typed external_replication object frozen into packet-rule
      v3. Its organization, operator, agent, model, session, and implementation
      repository must be distinct from all internal campaign seats and the candidate
      repository. Independent-code and zero-prior-exposure declarations, commit/tree
      provenance, blinded prediction/reveal chronology, output commitment/bytes, and a
      strict true agreement comparison are reconciled to content-addressed receipts.
      Same-identity, internal-repository, copied-core, generic-receipt, and malformed
      clean-room packages fail validation.
    changed_files:
      - "schemas/viability/packet-v2.schema.json"
      - "schemas/viability/protocol-manifest-v2.schema.json"
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
      - "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
    fix_commits: ["4a069d58e95d828701a09129a677d987c04b5e87"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_requires_typed_unaffiliated_clean_room"
        result: "exit 0; positive typed contract accepted; same-identity, internal-repository, copied-core, and generic-receipt mutations rejected"
      - command: "uv run pytest -q"
        result: "exit 0; 175 passed, including all prior custody, Tier G, outcome, dependency, evidence, and freeze regressions"
    residual_risk: "The validator checks evidence consistency, not the real-world truth of affiliation, exposure, or independent authorship; an external custodian and unaffiliated reproducer remain required."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed"
      - "tests/unit/test_viability_campaign_contract.py::test_review_artifacts_are_schema_complete_immutable_and_total"
      - "tests/unit/test_viability_campaign_contract.py::test_requirements_reject_missing_cycles_and_invalid_waves"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 16 passed, including malformed nested dependencies/YAML and contradictory identity/fix-commit provenance"
    rationale: "The public API now returns deterministic errors for the reproduced malformed inputs and rejects the semantic provenance counterexamples while preserving a complete positive review chain."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-PROTOCOL-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_primary_protocol_rejects_competing_experiment_fields"
      - "tests/unit/test_viability_campaign_contract.py::test_protocol_content_and_budget_are_frozen_before_holdout"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k 'primary_protocol or protocol_content_and_budget'"
        result: "exit 0; exact-envelope and post-freeze substitution regressions passed"
    rationale: "The regression constructs an honestly frozen protocol with extra experiment-defining authority and verifies rejection, while retaining a positive exact-envelope campaign."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_requires_typed_unaffiliated_clean_room"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_requires_typed_unaffiliated_clean_room"
        result: "exit 0; positive external package accepted and same-operator/model/session/organization/repository plus copied-core and generic-receipt mutations rejected"
    rationale: "The Tier E test now requires typed, content-addressed clean-room provenance and rejects relabeling the internal builder/orchestrator as an external reproduction."
    disagreement_ref: ""

new_or_changed_risks:
  - "Packet freeze popgp-packet-freeze-v3 intentionally changes the packet-rule digest; older draft packets must be re-preregistered before holdout execution."
  - "Review/response artifacts used by new campaigns must use v2; published v1 artifacts remain readable history but are not accepted by the v2 campaign validator."
  - "A cryptographically consistent external-replication package is still not independent proof that its real-world declarations are truthful."
  - "No POPGP scientific viability tier, hidden campaign, native CUDA campaign, or external empirical confirmation was produced by this remediation."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""
  - action: "For any Tier E claim, appoint an unaffiliated reproducer and custodian to create the required off-system evidence package."
    owner: "future campaign operator"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "VPLAN-SCHEMA-001, VPLAN-PROTOCOL-001, VPLAN-INDEPENDENCE-001, all three requested tests, all prior resolved findings/tests for regression, the complete history diff, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder implementation status is not independent resolution. Re-review the exact response-containing commit and preserve every prior review and response artifact."
```

The builder does not assign final resolution status. That determination belongs to an
independent re-review artifact.
