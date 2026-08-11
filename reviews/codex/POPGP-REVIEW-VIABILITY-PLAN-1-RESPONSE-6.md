# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-6

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-6"
response_round: 6
response_date: "2026-08-11"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-5"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-5.md"
review_commit: "e264b03babd09a428d1a1b236e6dbc6567cf55fa"
candidate_commit_reviewed: "c5f2528bb72bd267d79448ad17c9a2d9c3b4d650"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  Both unresolved blockers and all three active requested tests were accepted and
  implemented in 5f31d810c1bbbfacc319f85f4ee67429373a2159. Repository
  canonicalization now rejects decoded control characters before URL parsing or
  filesystem access and returns a validation error instead of raising. Network
  identities normalize credentials, transport spelling, default ports, DNS trailing
  dots, repeated slashes, dot segments, percent encoding, case, and terminal `.git`
  suffix or path-segment forms before candidate/external comparison.

  Structured external receipts now use recursive strict JSON equality. Boolean,
  integer, and floating-point values are distinct even where ordinary Python equality
  would conflate them, so numeric agreement and Boolean measured-value substitutions
  fail. One persisted end-to-end test constructs nine honestly pre-frozen Tier-E
  campaigns covering every accepted REREVIEW-5 counterexample. All 18 campaign-contract
  tests, all 177 repository tests, the six examples, lint, TeX source validation, and
  structured-artifact validation pass. This remediation establishes local validator
  behavior only; it does not establish real unaffiliated authorship or POPGP viability.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Repository text is percent-decoded and checked for C0/DEL control characters
      before URL or Path operations. Invalid encodings, invalid host/port forms, unsafe
      decoded text, and filesystem normalization failures return `None`; the public
      campaign validator converts that result into a deterministic repository-identity
      error. The exact schema-valid `file:///%00bad` campaign is now persisted as an
      honestly pre-frozen negative case and returns an error list without escaping.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["5f31d810c1bbbfacc319f85f4ee67429373a2159"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; 1 persisted matrix test passed in 154.97 s, including controlled embedded-NUL rejection"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; all 18 campaign-contract tests passed in 438.01 s"
      - command: "uv run pytest -q"
        result: "exit 0; all 177 repository tests passed in 461.33 s"
    residual_risk: "Public validation is designed to be total for schema-valid campaign inputs, but independent mutation remains necessary because no finite malformed-input suite proves totality for every platform, URL parser, or filesystem edge case."
    disagreement_ref: ""

  - finding_id: "VPLAN-INDEPENDENCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Network repository identity now uses parsed host/port fields, removes default
      transport ports and DNS terminal dots, normalizes percent-decoded POSIX paths and
      dot segments, and removes terminal `.git` suffix and `/.git` forms. The four
      reviewer-demonstrated aliases therefore equal the candidate identity and fail.
      Structured contract, provenance, reveal, commitment, and comparison receipts are
      compared recursively with exact JSON scalar types. Numeric `1`/`1.0` cannot stand
      in for Boolean agreement, and Boolean `true` cannot stand in for measured numeric
      values.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["5f31d810c1bbbfacc319f85f4ee67429373a2159"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; four canonical repository aliases and four Boolean/number substitutions were rejected while the embedded-NUL case failed closed"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 18 passed, retaining the positive Tier-E and every prior clean-room mutation"
      - command: "uv run ruff check . && uv run python scripts/check_tex.py"
        result: "exit 0; repository lint and the 652-line manuscript source check passed"
      - command: "six documented python -m examples.physics_qg.* commands && uv run python scripts/check_validation_artifacts.py"
        result: "all six examples exited 0; validation contracts and required visual outputs passed; regeneration added no diff"
    residual_risk: "Canonical identifiers, strict receipts, and a valid bundle establish internal evidence consistency, not unaffiliated real-world organization, authorship, exposure history, or institutional custody. An external maintainer must verify those facts."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; schema-valid file:///%00bad returned a deterministic invalid-repository error rather than raising"
    rationale: "The exact escaping input is persisted at the public validate_campaign boundary after honest packet/protocol freezing."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_requires_typed_unaffiliated_clean_room"
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_binds_outputs_git_bundle_orchestrator_and_comparison"
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 18 tests retain the clean-room positive and reject the expanded identity, provenance, receipt, comparison, custody, and bundle countermodels"
    rationale: "The original clean-room matrix is retained and now includes all strict JSON-type and canonical-alias negatives requested by REREVIEW-5."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_binds_outputs_git_bundle_orchestrator_and_comparison"
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 18 passed, including all narrowed output, bundle, orchestrator, comparison, alias, and type-substitution cases"
    rationale: "The reviewer-demonstrated default-port, DNS-dot, dot-segment, nested `/.git`, numeric-agreement, and Boolean-measured-value cases are now persisted alongside the earlier narrowed mutations."
    disagreement_ref: ""

new_or_changed_risks:
  - "Repository canonicalization is deliberately conservative and rejects unsafe or unparseable identities; operators needing a new transport or repository naming form must version and test it rather than bypass the gate."
  - "Git-bundle cloning remains bounded by timeout and avoids checkout, but evidence-package byte limits remain an operator-side resource control."
  - "No finite local mutation suite proves all platform-specific URL/filesystem behavior; clean independent re-review remains required."
  - "No POPGP Tier R, G, or E campaign or external empirical validation was produced by this remediation."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""
  - action: "Before any Tier E claim, have an unaffiliated custodian verify organization, exposure, authorship, and retained Git-bundle provenance off-system."
    owner: "future campaign operator"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "VPLAN-SCHEMA-001, VPLAN-INDEPENDENCE-001, TST-VPLAN-SCHEMA-001, TST-VPLAN-INDEPENDENCE-001, TST-VPLAN-INDEPENDENCE-002, every prior finding/test for regression, the complete history diff, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder implementation status is not independent resolution. Re-review the exact response-containing commit and preserve every prior review and response artifact."
```

The builder does not assign final resolution status. That determination belongs to an
independent re-review artifact.
