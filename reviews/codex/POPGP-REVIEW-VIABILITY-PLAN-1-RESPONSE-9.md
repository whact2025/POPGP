# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-9

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-9"
response_round: 9
response_date: "2026-08-11"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-8"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-8.md"
review_commit: "d84ee7589e1c65873a80311fd405896aea3d0fcb"
candidate_commit_reviewed: "0675367fdcdea86b0710dd2bad2e1a6f53ee92e7"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  All three unresolved REREVIEW-8 findings and their three active requested tests were
  accepted and implemented in 870e2788f154706aa4c7a2b4d71c21caa10f8f1d.
  JSON receipt loading now rejects non-finite constants and scans nesting iteratively
  before decoding; every structured receipt is also traversed iteratively to enforce a
  128-level mapping/list ceiling and finite floating-point values. YAML parser recursion
  is converted to a validation error, and the public validator has a final hostile-input
  boundary so filesystem or library exceptions return a deterministic error list.

  Primary-protocol envelope equality now uses recursive exact JSON type equality, so a
  Boolean cannot substitute for an integer or float. Repository host and path
  canonicalization reject residual percent escapes after one strict decode, closing the
  double-encoded-host acceptance path. The persisted Tier-E matrix adds hash-consistent
  NaN and 1,500-level prediction receipts and the exact residual-percent repository;
  the protocol test adds the exact `true` versus `1` substitution. All 178 repository
  tests, lint, TeX source validation, six examples, review-guidance checks, and
  structured-artifact validation pass. These are local contract results, not evidence
  of unaffiliated authorship or POPGP scientific viability.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Strict JSON loading rejects `NaN` and infinities, and a string-aware iterative
      bracket scan rejects more than 128 nested arrays/objects before the recursive
      standard decoder runs. An iterative post-parse traversal applies the same bound
      to JSON or YAML structured receipts and rejects non-finite floats. YAML recursion
      is converted to ValueError. Finally, validate_campaign wraps its complete public
      boundary so hostile paths, parser recursion, and unforeseen library exceptions
      become deterministic fail-closed errors rather than escaping to the caller.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["870e2788f154706aa4c7a2b4d71c21caa10f8f1d"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; the complete frozen-campaign matrix rejected hash-consistent NaN and 1,500-level blinded-prediction receipts"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_public_validator_fails_closed_for_invalid_entry_paths"
        result: "exit 0; NUL campaign and repository entry paths returned one deterministic validation error each"
      - command: "uv run pytest -q"
        result: "exit 0; all 178 repository tests passed in 715.40 s"
    residual_risk: "A finite malformed-input matrix cannot enumerate every parser, URI, filesystem, or dependency failure; the public boundary is now total for ordinary Exception failures and continued independent mutation remains required."
    disagreement_ref: ""

  - finding_id: "VPLAN-PROTOCOL-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The primary-protocol document is compared with the exact frozen packet envelope
      using the existing recursive strict JSON comparator. Mapping keys and sequence
      lengths must match, and scalar values must have identical Python/JSON types before
      equality is considered, so `true`, `1`, and `1.0` are distinct.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["870e2788f154706aa4c7a2b4d71c21caa10f8f1d"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_primary_protocol_rejects_competing_experiment_fields"
        result: "exit 0; the exact protocol Boolean versus packet integer substitution was rejected alongside all extra-authority controls"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "covered by the final 178-test green repository run and the focused final-state campaign checks"
    residual_risk: "Exact envelope equality establishes contract identity, not the scientific adequacy or truth of preregistered parameters."
    disagreement_ref: ""

  - finding_id: "VPLAN-INDEPENDENCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      After the one permitted strict URI decode, any residual percent character in a
      repository host or canonical Git path makes the identity invalid. The exact
      `https://%2567ithub.com/...` counterexample therefore fails locally before it can
      be represented as distinct external provenance, matching Git's rejection while
      retaining every historical valid canonicalization case.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["870e2788f154706aa4c7a2b4d71c21caa10f8f1d"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; the residual-percent host was rejected with all accumulated repository aliases and strict external-receipt substitutions"
      - command: "six documented python -m examples.physics_qg.* commands && uv run python scripts/check_validation_artifacts.py"
        result: "all examples exited 0; artifact contracts and required visuals passed; regeneration introduced no additional diff"
    residual_risk: "Canonical local identities and evidence consistency cannot prove actual unaffiliated organization, authorship, exposure history, or control. An external maintainer must verify those facts."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
      - "tests/unit/test_viability_campaign_contract.py::test_public_validator_fails_closed_for_invalid_entry_paths"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_public_validator_fails_closed_for_invalid_entry_paths tests/unit/test_viability_campaign_contract.py::test_primary_protocol_rejects_competing_experiment_fields tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; 3 passed in 334.16 s"
    rationale: "The persisted public-boundary matrix now covers strict RFC JSON constants, bounded receipt nesting, and direct hostile entry paths without exceptions."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-PROTOCOL-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_primary_protocol_rejects_competing_experiment_fields"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_primary_protocol_rejects_competing_experiment_fields"
        result: "exit 0; retained exact-envelope positive/extra-field negatives and rejected recursive JSON type substitution"
    rationale: "The exact reviewer Boolean/integer parameter countermodel is persisted against strict recursive envelope equality."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_rejects_repository_aliases_and_json_type_substitution"
        result: "exit 0; the exact residual-percent repository and all prior aliases, output, bundle, comparison, and strict-type cases failed closed"
    rationale: "Post-decode repository identity validation now rejects residual encoding instead of promoting a Git-invalid endpoint as independent provenance."
    disagreement_ref: ""

new_or_changed_risks:
  - "Structured receipts are intentionally limited to 128 nested mapping/array levels and finite JSON numbers; legitimate deeper evidence would require a versioned contract change."
  - "Repository identities with a residual percent after one decode are intentionally invalid; new encoding forms require an explicit, tested canonicalization rule."
  - "The public validator converts ordinary Exception failures into errors because campaign artifacts are untrusted; KeyboardInterrupt, SystemExit, and process termination remain outside the contract."
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
  scope: "VPLAN-SCHEMA-001, VPLAN-PROTOCOL-001, VPLAN-INDEPENDENCE-001, TST-VPLAN-SCHEMA-001, TST-VPLAN-PROTOCOL-001, TST-VPLAN-INDEPENDENCE-002, every prior finding/test for regression, the complete history diff, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder implementation status is not independent resolution. Re-review the exact response-containing commit and preserve every prior review and response artifact."
```

The builder does not assign final resolution status. That determination belongs to an
independent re-review artifact.
