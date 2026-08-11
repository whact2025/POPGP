# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-12

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-12"
response_round: 12
response_date: "2026-08-11"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-11"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-11.md"
review_commit: "1839eca1ebcbe4ee40d25044bb5c8b5d30fdab04"
candidate_commit_reviewed: "06561b3e83fedb52707b42e89b278d411311154c"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  The sole unresolved REREVIEW-11 finding and requested test were accepted and
  implemented in 70039a2733dd7ae2a3884004dcc4f5eb13587ba5. The existing
  identity-aware logical expansion, height, cycle, and finite-number budget is now a
  single structured-input invariant applied immediately after every JSON or YAML parse,
  before JSON Schema evaluation, schema-error rendering, canonical hashing, or recursive
  comparison. Direct in-memory packet hashing and requirements-document API entry points
  apply the same invariant before consuming caller-supplied object graphs.

  The exact 42-level shared binary alias DAG now returns a controlled expanded-node
  error at packet, campaign, requirements-validation, and requirements-override
  boundaries. A complete frozen campaign with ordinary four-level packet and protocol
  aliases remains valid. The final repository suite passed all 179 tests in 949.98
  seconds; focused boundary tests, lint, TeX validation, review-guidance tests, all six
  examples, artifact validation, regeneration cleanliness, and process-residue checks
  also passed. These are local executable-contract results, not scientific-viability or
  external-independence evidence.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Graph validation was generalized from receipt payloads to every parsed structured
      document. Each container identity is traversed once, while its logical subtree
      size is counted at every alias reference. Active-path identity rejects cycles;
      cached height preserves the 128-level limit at every reference; logical expansion
      above 100,000 nodes and non-finite floats fail before any unbounded schema or
      canonicalization consumer runs. The same check guards direct packet-rule hashing,
      validate_requirements, and validate_campaign requirements overrides.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["70039a2733dd7ae2a3884004dcc4f5eb13587ba5"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed tests/unit/test_viability_campaign_contract.py::test_public_validator_fails_closed_for_invalid_entry_paths"
        result: "exit 0; 2 passed in 100.91 s with controlled packet, campaign, requirements, receipt, cycle, and oversized-input failures plus ordinary-alias positives"
      - command: "uv run pytest -q"
        result: "exit 0; all 179 repository tests passed on the final implementation in 949.98 s"
      - command: "uv run ruff check . && uv run python scripts/check_tex.py && uv run pytest -q tests/unit/test_review_guidance.py && uv run python scripts/check_validation_artifacts.py"
        result: "exit 0; lint, 652-line TeX validation, 9 review-guidance tests, and structured/visual artifact checks passed"
      - command: "all six documented python -m examples.physics_qg.* commands, followed by git diff --check and process-residue inspection"
        result: "all examples exited 0; artifacts remained valid; no generated diff, temporary external checkout, or matching Git process remained"
    residual_risk: "The 16 MiB text, 100,000 logical-node, 128-level depth, 256-character JSON-number, and bounded subprocess ceilings are explicit contract limits. Larger legitimate evidence requires a versioned format change or an externally governed binary reference."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed"
      - "tests/unit/test_viability_campaign_contract.py::test_public_validator_fails_closed_for_invalid_entry_paths"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed tests/unit/test_viability_campaign_contract.py::test_public_validator_fails_closed_for_invalid_entry_paths"
        result: "exit 0; exact hostile graphs returned promptly at every YAML-bearing public boundary and ordinary aliases remained valid"
      - command: "uv run pytest -q"
        result: "exit 0; all 179 tests passed, including every historical campaign-contract regression"
    rationale: |-
      Persisted tests now cover the exact hostile graph at receipt, packet, campaign,
      requirements-validation, and requirements-override boundaries. The positive uses
      matching protocol and packet aliases in a complete frozen campaign, proving that
      alias sharing itself is supported while over-budget logical expansion fails closed.
    disagreement_ref: ""

new_or_changed_risks:
  - "Every parsed structured document is now subject to the same declared graph and numeric limits before schema evaluation or canonicalization."
  - "Direct callers that supply over-budget in-memory packet or requirements graphs receive a controlled validation error or ValueError rather than an unbounded traversal."
  - "No POPGP Tier R, G, or E campaign, external empirical validation, or unaffiliated clean-room result was produced by this remediation."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA, including the generalized YAML graph regressions."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""
  - action: "Before any Tier E claim, have an unaffiliated custodian verify organization, exposure, authorship, and retained Git-bundle provenance off-system."
    owner: "future campaign operator"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "VPLAN-SCHEMA-001, TST-VPLAN-SCHEMA-001, every prior finding/test for regression, the complete history diff, and new findings within the declared executable contract"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder status is not independent resolution. Re-review the exact response-containing commit. Approval should require the declared limits to fail closed, not unbounded acceptance of inputs outside those limits."
```

The builder does not assign final resolution status. That determination belongs to an
independent re-review artifact.
