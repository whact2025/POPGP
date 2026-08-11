# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-7

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-7"
response_round: 7
response_date: "2026-08-11"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-6"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-6.md"
review_commit: "ec7bafa83c87fcbb4c61dd52c3d8a71f39913f40"
candidate_commit_reviewed: "50df04b05e6991598a9ca0d0ffac45f8b0e0643e"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  Both unresolved REREVIEW-6 findings and all three active requested tests were accepted
  and implemented in 2bcbb2047fdecde358ca67ccb9199326ca5d9457. Repository
  identities now reject malformed percent triplets before decoding. Reg-name hosts are
  percent-decoded and IDNA-normalized, IPv4 and IPv6 literals are canonicalized, Windows
  drive paths and equivalent file URIs share one local identity, and `git+ssh` port 22
  is treated as the SSH default.

  The persisted public-boundary regression constructs fifteen honestly pre-frozen
  Tier-E campaigns. It retains all REREVIEW-5 cases and adds malformed `%ZZ`,
  percent-encoded host, compressed/expanded IPv6, Windows/file URI, leading-zero IPv4,
  and `ssh`/`git+ssh` aliases from REREVIEW-6. All invalid identities return controlled
  errors and all aliases are rejected as candidate reuse; strict JSON substitutions
  remain rejected. All 18 campaign-contract tests, all 177 repository tests, lint, TeX
  source validation, six examples, and structured-artifact validation pass. These are
  local contract results, not proof of unaffiliated authorship or POPGP viability.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Repository text now fails validation when any percent sign is not followed by
      exactly two hexadecimal digits. This check occurs before URL parsing, percent
      decoding, host canonicalization, or filesystem operations. The exact
      `https://example.com/%ZZ/repo` Tier-E campaign is persisted and now returns a
      deterministic invalid-repository error list. Existing NUL/control, bad host/port,
      bracket, encoding, and structured-receipt totality cases remain green.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["2bcbb2047fdecde358ca67ccb9199326ca5d9457"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; the fifteen-campaign matrix passed in 242.89 s, including controlled `%ZZ` rejection"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; all 18 campaign-contract tests passed in 560.63 s"
      - command: "uv run pytest -q"
        result: "exit 0; all 177 repository tests passed in 574.19 s"
    residual_risk: "No finite malformed-input suite proves totality across every URL parser and filesystem spelling; continued independent mutation remains required."
    disagreement_ref: ""

  - finding_id: "VPLAN-INDEPENDENCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Host canonicalization now decodes valid percent escapes, rejects decoded host
      delimiters and controls, normalizes trailing dots/case/IDNA, converts dotted
      decimal IPv4 octets to canonical decimal, and uses `ipaddress` compressed output
      for IP literals. IPv6 authorities retain unambiguous brackets. Windows file-URI
      drive paths lose the URI-only leading slash and resolve to the same identity as
      native drive syntax. The `git+ssh` transport now shares SSH's default port 22.
      All five accepted alias classes from REREVIEW-6 therefore reconcile to candidate
      reuse, including the two additional portability-risk variants.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["2bcbb2047fdecde358ca67ccb9199326ca5d9457"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; encoded-host, IPv6, Windows/file, leading-zero IPv4, and git+ssh aliases were rejected as candidate reuse alongside all prior alias/type cases"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 18 passed, retaining the valid Tier-E clean-room path and every prior negative control"
      - command: "uv run ruff check . && uv run python scripts/check_tex.py"
        result: "exit 0; repository lint and the 652-line manuscript source check passed"
      - command: "six documented python -m examples.physics_qg.* commands && uv run python scripts/check_validation_artifacts.py"
        result: "all six examples exited 0; validation contracts and required visuals passed; regeneration introduced no additional diff"
    residual_risk: "Canonical identifiers and internally consistent evidence cannot prove unaffiliated organization, authorship, exposure history, or custody. Those off-system facts still require an external maintainer."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; `%ZZ`, embedded-NUL, and the complete repository-identity matrix returned errors without raising"
    rationale: "The exact newly accepted malformed-percent campaign is persisted at the public validate_campaign boundary while all prior totality cases are retained."
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
        result: "exit 0; 18 tests retain the clean-room positive and reject all accumulated identity, provenance, receipt, comparison, custody, and bundle countermodels"
    rationale: "The clean-room matrix now includes every exact REREVIEW-5 and REREVIEW-6 strict-type and repository-alias counterexample."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_binds_outputs_git_bundle_orchestrator_and_comparison"
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 18 passed, including all output, bundle, orchestrator, comparison, alias, and JSON-type cases"
    rationale: "Canonical identity negatives now include encoded reg-name hosts, IP literal spellings, local/file equivalence, leading-zero IPv4, and git+ssh in addition to every earlier narrowed mutation."
    disagreement_ref: ""

new_or_changed_risks:
  - "Repository identity normalization intentionally rejects malformed or unsafe names and may require a versioned extension for future transports or platform-specific naming forms."
  - "Dotted-decimal IPv4 normalization treats leading-zero octets as decimal to eliminate the demonstrated alias; other historical non-dotted IPv4 notations remain outside the declared contract."
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
