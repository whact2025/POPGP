# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-11

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-11"
response_round: 11
response_date: "2026-08-11"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-10"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-10.md"
review_commit: "3813260cca3bef7bd902569026f52bce37327fe2"
candidate_commit_reviewed: "12517855a9f0b24f0e01c2790098733702412ee9"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  The sole unresolved REREVIEW-10 schema/parser finding and requested test were accepted
  and implemented in e4d43b0a507bba770968e51f2d71ae6f8c481431. Structured receipt
  traversal now computes logical expanded size and maximum height with object-identity
  memoization. Shared aliases are not re-expanded computationally, but their logical
  subtree sizes still count at every reference. Active-path identity detects cycles.
  Receipts exceeding 100,000 logically expanded nodes, 128 levels, or a cycle fail
  deterministically. A four-level ordinary shared-alias graph remains a positive.

  The exact 42-level binary alias DAG is persisted in a complete, hash-consistent
  Tier-R output-commitment receipt and returns a controlled expanded-node error under a
  15-second full-campaign supervisor. A self-referential YAML alias returns a controlled
  cycle error. As a complementary parser-resource boundary, campaign, packet, schema,
  and structured-receipt text is read through a 16 MiB ceiling; a 16 MiB+1 sparse
  campaign fails before parsing. The exact final repository suite passed all 179 tests,
  the separate contract suite passed all 20 tests, and lint, TeX, examples, artifact,
  regeneration, and process-cleanup checks passed. These are local contract results,
  not evidence of POPGP scientific viability or external independence.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The validator no longer traverses a YAML representation as an exponentially
      expanded Python tree. It memoizes each container's bounded logical subtree size
      and height, reuses those metrics for aliases, detects active-path cycles, and
      rejects expansion beyond a named 100,000-node limit. The exact reviewer DAG is
      rejected after bounded unique-node work while ordinary alias sharing remains
      valid. All structured file reads are additionally limited to 16 MiB before decode
      or parse, closing a direct oversized-input substitute for the alias attack.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["e4d43b0a507bba770968e51f2d71ae6f8c481431"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed"
        result: "exit 0; ordinary aliases passed and the exact 42-level DAG plus cyclic alias returned controlled errors"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_public_validator_fails_closed_for_invalid_entry_paths"
        result: "exit 0; NUL entry paths and a sparse 16 MiB+1 campaign returned prompt deterministic validation errors"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; all 20 campaign-contract tests passed in 717.19 s before the bounded-reader addition; the final full-suite run re-executed the complete file"
      - command: "uv run pytest -q"
        result: "exit 0; all 179 repository tests passed on the final implementation in 1054.29 s"
      - command: "uv run ruff check . && uv run python scripts/check_tex.py && uv run python scripts/check_validation_artifacts.py"
        result: "exit 0; repository lint, 652-line TeX source validation, and structured/visual artifact checks passed"
    residual_risk: "The 16 MiB input, 100,000-node logical expansion, and 128-level depth ceilings are deliberate contract limits; larger legitimate evidence requires a versioned format change or external binary artifact reference."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed"
      - "tests/unit/test_viability_campaign_contract.py::test_public_validator_fails_closed_for_invalid_entry_paths"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_public_validator_fails_closed_for_invalid_entry_paths tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed"
        result: "exit 0; 2 passed in 89.00 s with ordinary alias, alias-DAG, cycle, overflow/underflow, and oversized-input boundaries"
    rationale: "The exact small alias DAG is persisted at the public full-campaign boundary, ordinary alias sharing is retained, cycles and over-budget logical expansion fail closed, and structured input bytes are independently bounded."
    disagreement_ref: ""

new_or_changed_risks:
  - "Structured text inputs larger than 16 MiB are rejected and must be represented by a bounded summary plus separately governed binary artifact."
  - "YAML alias graphs are allowed only when acyclic, depth-bounded, and at or below 100,000 logically expanded nodes."
  - "The 15-second regression supervisor covers complete Tier-R validation under host load; the graph traversal itself is deterministically node-bounded rather than time-polled."
  - "No POPGP Tier R, G, or E campaign or external empirical validation was produced by this remediation."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA, including the POSIX process-group and YAML alias regressions."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""
  - action: "Before any Tier E claim, have an unaffiliated custodian verify organization, exposure, authorship, and retained Git-bundle provenance off-system."
    owner: "future campaign operator"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "VPLAN-SCHEMA-001, TST-VPLAN-SCHEMA-001, every prior finding/test for regression, the complete history diff, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder implementation status is not independent resolution. Re-review the exact response-containing commit and preserve every prior review and response artifact."
```

The builder does not assign final resolution status. That determination belongs to an
independent re-review artifact.
