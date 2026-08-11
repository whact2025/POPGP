# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-14

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-14"
response_round: 14
response_date: "2026-08-11"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-13"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-13.md"
review_commit: "564765088edf901aaf427d23bb4b55cccb97bc54"
candidate_commit_reviewed: "ef96a6481a91a31f87cd58b46b1057ef4f009584"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  The remaining VPLAN-SCHEMA-002 / TST-VPLAN-SCHEMA-002 merge-source bypass was
  accepted and implemented in cf1b356e2bdcc73da97f2b64e36a25ffdd85f914.
  Duplicate explicit keys are now checked across the parsed YAML node graph before any
  mapping constructor calls flatten_mapping. This includes mappings reachable only as
  inline or aliased merge sources, which PyYAML otherwise copies without invoking their
  mapping constructor.

  The exact inline source containing two threshold keys now raises a controlled
  duplicate-key error directly and when embedded in a complete public campaign. Legal
  shipped-template overrides still match safe YAML semantics and hash deterministically;
  ordinary aliases and graph limits are unchanged. The exact repository suite passed
  179 tests in 820.53 seconds; focused tests, lint, TeX, review guidance, all six
  examples, artifact validation, regeneration, and residue checks passed. This remains
  executable-contract evidence only, not scientific viability or external validation.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-002"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The loader now traverses MappingNode and SequenceNode identities before merge
      flattening, memoizing checked node identities and breaking alias cycles. Every
      mapping's original non-merge key nodes are checked for hashability and repetition.
      Thus an inline merge source cannot bypass duplicate detection, while flattening
      still applies only after the entire reachable node graph is proven unambiguous.
      The existing post-parse graph budget remains responsible for logical expansion,
      depth, cycle, and finite-number enforcement.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["cf1b356e2bdcc73da97f2b64e36a25ffdd85f914"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_shipped_templates_conform_to_versioned_schemas tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed"
        result: "exit 0; 2 passed in 81.43 s; exact inline-source duplicate rejected directly and through public campaign validation"
      - command: "uv run python scripts/check_viability_campaign.py --packet-rule-sha256 docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
        result: "exit 0; legal template merge overrides retained deterministic SHA-256 a549585ebff920eba000f529d6d157c1031924750edb0c157e9f61f51ac009bb"
      - command: "uv run pytest -q"
        result: "exit 0; all 179 repository tests passed in 820.53 s"
      - command: "uv run ruff check . && uv run python scripts/check_tex.py && uv run pytest -q tests/unit/test_review_guidance.py && uv run python scripts/check_validation_artifacts.py"
        result: "exit 0; lint, 652-line TeX validation, 9 review-guidance tests, and artifact checks passed"
      - command: "all six documented python -m examples.physics_qg.* commands plus regeneration and process-residue checks"
        result: "all examples exited 0; no generated diff, temporary external checkout, or matching Git process remained"
    residual_risk: "YAML node-graph duplicate scanning is identity-memoized and cycle-safe. Standard deterministic merge precedence remains accepted; repeated explicit data keys at any reachable source mapping are invalid."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-002"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_shipped_templates_conform_to_versioned_schemas"
      - "tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed"
      - "tests/unit/test_viability_campaign_contract.py::test_review_artifacts_are_schema_complete_immutable_and_total"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_shipped_templates_conform_to_versioned_schemas tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed tests/unit/test_viability_campaign_contract.py::test_review_artifacts_are_schema_complete_immutable_and_total"
        result: "exit 0 in focused executions; shipped CLI/merge positives, inline-source/direct/public duplicates, graph protections, and historical duplicate paths passed"
    rationale: |-
      The regression now covers the reviewer’s exact constructor-bypass shape both at
      loader level and inside a complete public campaign. It also retains the legal
      shipped-template hash, all-field reference equality, named inherited/overridden
      values, ordinary aliases, and the pre-existing direct/public duplicate controls.
    disagreement_ref: ""

new_or_changed_risks:
  - "Every reachable YAML source mapping is inspected before merge flattening; work is memoized by YAML node identity and remains subject to parser and post-parse graph limits."
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
  scope: "VPLAN-SCHEMA-002, TST-VPLAN-SCHEMA-002, exact inline merge-source duplicate rejection, legal merge positives, all historical regressions, and new findings within the declared contract"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Reproduce the exact inline-only duplicate source through the authoritative loader and a complete public campaign. Approve only if it fails closed without weakening legal merges or graph limits."
```

The builder does not assign final resolution status. That determination belongs to an
independent re-review artifact.
