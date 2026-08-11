# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-13

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-13"
response_round: 13
response_date: "2026-08-11"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-12"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-12.md"
review_commit: "940284c93e77c2044d027b4d91024d71f9e50fc0"
candidate_commit_reviewed: "9ae2e83f3d54a1e5adabf24da5aa21f222cfe55e"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  VPLAN-SCHEMA-002 and TST-VPLAN-SCHEMA-002 were accepted and implemented in
  90f749762c57b31668da4688d622cd5da0d5f80d. The authoritative YAML constructor
  now detects repeated keys among the mapping's explicit entries before applying merge
  defaults. Standard YAML merge defaults may therefore be overridden explicitly, as
  the shipped packet template requires, while genuine repeated explicit keys and
  unhashable keys still fail deterministically.

  The production packet-hash CLI now hashes the exact shipped template. The regression
  also parses it through both the authoritative loader and PyYAML reference semantics,
  proves the complete documents match, checks the builder-specific identity override
  and inherited operator, and proves an appended duplicate packet_id still raises the
  controlled duplicate-key error. The exact full repository suite passed 179 tests in
  924.68 seconds; focused tests, lint, TeX, review guidance, all six examples, artifact
  validation, regeneration, and residue checks also passed. No scientific viability or
  external independence is claimed.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-002"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Duplicate detection now operates on the original explicit key nodes, before
      flatten_mapping inserts inherited merge pairs. After that check, standard merge
      flattening applies deterministic YAML precedence, so explicit seat identity and
      session values override the shared anchor without being mistaken for repeated
      source keys. The post-flatten construction still rejects unhashable keys. This
      fixes the documented template path without weakening genuine duplicate rejection.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["90f749762c57b31668da4688d622cd5da0d5f80d"]
    verification:
      - command: "uv run python scripts/check_viability_campaign.py --packet-rule-sha256 docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
        result: "exit 0; emitted deterministic SHA-256 a549585ebff920eba000f529d6d157c1031924750edb0c157e9f61f51ac009bb"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_shipped_templates_conform_to_versioned_schemas tests/unit/test_viability_campaign_contract.py::test_review_artifacts_are_schema_complete_immutable_and_total"
        result: "exit 0; 2 passed in 26.73 s; legal overrides matched reference semantics and a genuine duplicate remained rejected"
      - command: "uv run pytest -q"
        result: "exit 0; all 179 repository tests passed in 924.68 s"
      - command: "uv run ruff check . && uv run python scripts/check_tex.py && uv run pytest -q tests/unit/test_review_guidance.py && uv run python scripts/check_validation_artifacts.py"
        result: "exit 0; lint, 652-line TeX validation, 9 review-guidance tests, and structured/visual artifact checks passed"
      - command: "all six documented python -m examples.physics_qg.* commands, followed by regeneration and process-residue checks"
        result: "all examples exited 0; no generated diff, temporary external checkout, or matching Git process remained"
    residual_risk: "Only standard deterministic YAML merge precedence is accepted; repeated explicit data keys remain invalid. The declared byte, depth, logical-node, numeric, and subprocess limits are unchanged."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-002"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_shipped_templates_conform_to_versioned_schemas"
      - "tests/unit/test_viability_campaign_contract.py::test_review_artifacts_are_schema_complete_immutable_and_total"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_shipped_templates_conform_to_versioned_schemas tests/unit/test_viability_campaign_contract.py::test_review_artifacts_are_schema_complete_immutable_and_total"
        result: "exit 0; production loader, CLI hash, override values, inherited values, and duplicate explicit key behavior all passed"
    rationale: |-
      The shipped template is now tested through the exact production loader and CLI,
      not only yaml.safe_load and JSON Schema. Equality to reference merge semantics
      verifies every resulting field, with named override/inheritance assertions making
      the intended behavior explicit; the existing public duplicate test plus the new
      local duplicate case preserve fail-closed ambiguity handling.
    disagreement_ref: ""

new_or_changed_risks:
  - "Legal YAML merge defaults are accepted and may be overridden explicitly; repeated explicit mapping keys remain rejected."
  - "No POPGP Tier R, G, or E campaign, external empirical validation, or unaffiliated clean-room result was produced."

external_actions:
  - action: "Run GitHub Actions on the exact response and approval handoff SHA."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""
  - action: "Before any Tier E claim, independently verify off-system organization, authorship, exposure, and retained bundle provenance."
    owner: "future campaign operator"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "VPLAN-SCHEMA-002, TST-VPLAN-SCHEMA-002, the exact template/loader delta, every historical regression, and new findings within the declared executable contract"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Reproduce the exact shipped-template CLI command, compare authoritative merge results, and retain genuine duplicate rejection. Approve only if no in-contract blocker remains."
```

The builder does not assign final resolution status. That determination belongs to an
independent re-review artifact.
