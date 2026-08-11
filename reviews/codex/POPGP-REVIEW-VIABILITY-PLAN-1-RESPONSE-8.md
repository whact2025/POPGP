# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-8

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-8"
response_round: 8
response_date: "2026-08-11"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-7"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-7.md"
review_commit: "cdb9df65a7bc28b04702871b01eb65bdc104aad0"
candidate_commit_reviewed: "1e378d9dc30be2aa070997e6d04e48f2d808c3eb"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  Both unresolved REREVIEW-7 findings and all three active requested tests were accepted
  and implemented in 1cd76fa57855f3ce94c40bb20b0a552440075083. Opaque
  `file:C:/...` URIs are recognized before SCP rewriting and normalize with native
  Windows drive paths. IPv4-mapped IPv6 addresses collapse to their mapped IPv4
  endpoint. Dotted-decimal conversion is length-bounded before integer parsing, and
  repository-contained path resolution catches filesystem, recursion, and value errors.

  The persisted public-boundary matrix now constructs nineteen honestly pre-frozen
  Tier-E campaigns. It retains every prior malformed identity, repository alias, and
  strict JSON case while adding opaque file URI, IPv4-mapped IPv6, a 5,000-digit dotted
  host, and a NUL receipt path. The aliases are rejected as candidate reuse and hostile
  inputs return deterministic error lists without raising. All 18 campaign-contract
  tests, all 177 repository tests, lint, TeX source validation, six examples, and
  structured-artifact validation pass. These are local contract results, not evidence
  of unaffiliated authorship or POPGP scientific viability.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Host text longer than 253 characters is rejected before dotted-decimal or IDNA
      conversion, and dotted-decimal octets longer than three characters are rejected
      before `int`. The 5,000-digit host therefore returns an invalid repository error
      without invoking Python's integer digit limit. `_resolve_inside` now guards both
      path and root resolution and containment under OSError, RuntimeError, and
      ValueError. A schema-valid NUL receipt path therefore becomes the existing
      deterministic path-containment error instead of escaping from Path/stat.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["1cd76fa57855f3ce94c40bb20b0a552440075083"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; the nineteen-campaign matrix passed in 329.22 s, including controlled huge-host and NUL receipt-path rejection"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; all 18 campaign-contract tests passed in 600.97 s"
      - command: "uv run pytest -q"
        result: "exit 0; all 177 repository tests passed in 629.83 s"
    residual_risk: "No finite malformed-input suite proves totality across every parser and filesystem edge case; continued independent public-boundary mutation is required."
    disagreement_ref: ""

  - finding_id: "VPLAN-INDEPENDENCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      `file:` is recognized as a URI scheme before SCP-like rewriting, so both opaque
      and hierarchical Windows file URIs flow through the local-path normalizer and
      equal native drive syntax. Parsed IPv6Address values with `ipv4_mapped` are
      replaced by their canonical IPv4Address before authority construction, making
      IPv4 and `::ffff:` spellings one endpoint. Both exact reviewer campaigns now fail
      as candidate repository reuse while every earlier alias remains rejected.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["1cd76fa57855f3ce94c40bb20b0a552440075083"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; opaque file URI and IPv4-mapped IPv6 aliases were rejected alongside every prior identity and strict-type case"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 18 passed, preserving the valid Tier-E clean-room path and all historical controls"
      - command: "uv run ruff check . && uv run python scripts/check_tex.py"
        result: "exit 0; repository lint and the 652-line manuscript source check passed"
      - command: "six documented python -m examples.physics_qg.* commands && uv run python scripts/check_validation_artifacts.py"
        result: "all examples exited 0; artifact contracts and required visuals passed; regeneration introduced no additional diff"
    residual_risk: "Local canonical identities and evidence consistency cannot prove real unaffiliated organization, authorship, exposure history, or custody. An external maintainer must verify those facts."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; the 5,000-digit dotted host and NUL receipt path returned deterministic errors without raising"
    rationale: "Both exact REREVIEW-7 totality failures are persisted at the public validate_campaign boundary while every earlier malformed-input case remains covered."
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
        result: "exit 0; 18 tests retain the positive clean-room campaign and reject all accumulated identity, provenance, receipt, comparison, custody, and bundle countermodels"
    rationale: "The clean-room matrix now contains every exact REREVIEW-5 through REREVIEW-7 repository and strict-receipt counterexample."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_binds_outputs_git_bundle_orchestrator_and_comparison"
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 18 passed, including every output, bundle, orchestrator, comparison, alias, strict-type, and valid-disagreement case"
    rationale: "Canonical identity negatives now include both opaque/hierarchical file-URI equivalence and IPv4-mapped IPv6 equivalence in addition to the complete prior matrix."
    disagreement_ref: ""

new_or_changed_risks:
  - "Repository/path normalization intentionally rejects unsafe or overlong identities; future platform-specific forms require a versioned, tested extension."
  - "The 253-character host and three-character dotted-octet limits bound hostile conversion while matching declared DNS/canonical dotted-decimal forms."
  - "Git-bundle cloning remains timeout-bounded and no-checkout, but evidence-package byte limits remain an operator-side resource control."
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
