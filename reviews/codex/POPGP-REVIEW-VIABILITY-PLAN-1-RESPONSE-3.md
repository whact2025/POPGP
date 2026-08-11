# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-3

```yaml
artifact_schema_version: 1
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-3"
response_round: 3
response_date: "2026-08-10"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-2"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-2.md"
review_commit: "a1068262fbb9df902a11e73c46f3d09d92f881f7"
candidate_commit_reviewed: "5cd803a190da704bf7295974a8e481e46562211a"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs or private evaluator were available. The two accepted blockers were remediated from the immutable review's executed counterexamples."

summary: |-
  Both blockers and both requested tests from re-review 2 were accepted and implemented
  in 1d6f08022075a32b0daed96f37f87415f9843a97. Three versioned governance-artifact
  schemas now require complete reviewer identity, independence, provenance, evidence,
  builder disposition, verification, and recommendation fields. Every review-chain
  receipt is bound to an immutable Git commit:path blob; duplicate YAML/JSON keys and
  malformed external types fail closed without escaping the public validation API.

  Packet freeze v2 adds canonical parameters, measurement and uncertainty procedures,
  statistical analysis, resource budgets, commands, mutation plans, seat assignments,
  and protocol-artifact references to the packet-rule hash. Exactly one primary protocol
  JSON must match the canonical preregistration block, its receipt path/hash/media type,
  and the raw Git blob stored at the named protocol commit. The authoritative suite
  passes with 172 tests, all six documented examples, and the artifact checker. These
  controls make a future campaign auditable; they do not demonstrate a POPGP viability
  tier or constitute external scientific validation.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Initial reviews, builder responses, and re-reviews now validate against separate
      versioned JSON Schemas before any result is consumed. The schemas require every
      field in the repository governance templates, including exact reviewed/baseline
      commits, context tree, immutable prior refs, identity and independence declarations,
      evidence, verification level, disposition, implementation status, and recommendation.
      Cross-document logic verifies commit trees, response round/candidate bindings,
      prior-review and builder-response refs, byte identity at each Git commit:path ref,
      complete item coverage, blocker counts, and successors. Duplicate-rejecting YAML
      and JSON loaders remove last-key-wins ambiguity. Malformed outcomes are rejected by
      schema/type checks, and validate_requirements handles mistyped packet entries without
      raising.
    changed_files:
      - "schemas/viability/independent-review-v1.schema.json"
      - "schemas/viability/review-response-v1.schema.json"
      - "schemas/viability/independent-rereview-v1.schema.json"
      - "schemas/viability/packet-v2.schema.json"
      - "schemas/viability/protocol-manifest-v2.schema.json"
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
      - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
      - "docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
    fix_commits: ["1d6f08022075a32b0daed96f37f87415f9843a97"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 13 passed, including a complete positive chain plus minimal-artifact, malformed-outcome, immutable-ref, duplicate YAML/JSON, and malformed-requirements mutations"
      - command: "uv run pytest -q"
        result: "exit 0; 172 passed in 89.65 s"
      - command: "uv run ruff check ."
        result: "exit 0; all checks passed"
    residual_risk: "Schema-valid identity and access declarations remain assertions by their signers. External campaigns still need unaffiliated operators and custody to establish independence beyond repository provenance."
    disagreement_ref: ""

  - finding_id: "VPLAN-FREEZE-003"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Packet freeze v2 includes an explicit preregistration block with canonical parameters,
      measurement and uncertainty procedures, statistical analysis, resource budget,
      commands, mutation plan, and content-addressed protocol artifacts. Each artifact
      freezes its role, campaign receipt path, protocol-commit path, SHA-256, and media
      type. The validator requires the receipt fields and current bytes to match, resolves
      the original blob from protocol_commit, and compares its raw SHA-256. Exactly one
      primary protocol JSON is required and its experiment-defining fields must equal the
      canonical packet block. The packet-rule hash also freezes preregistered seat identities.
    changed_files:
      - "schemas/viability/packet-v2.schema.json"
      - "schemas/viability/protocol-manifest-v2.schema.json"
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
      - "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
    fix_commits: ["1d6f08022075a32b0daed96f37f87415f9843a97"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k protocol_content_and_budget"
        result: "exit 0; same-path threshold change, path/content substitution, changed resource ceiling, recomputed mutable receipt hashes, and recomputed packet self-hash are rejected"
      - command: "uv run python scripts/check_validation_artifacts.py"
        result: "exit 0; validation artifact contracts and required visual outputs are valid"
      - command: "six documented python -m examples.physics_qg.* commands"
        result: "all six exited 0; regeneration left no structured-artifact diff"
    residual_risk: "A frozen protocol may still be scientifically inadequate; the contract prevents substitution and ambiguity but independent reviewers must judge thresholds, power, uncertainty models, and resource ceilings before holdout execution."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_review_artifacts_are_schema_complete_immutable_and_total"
      - "tests/unit/test_viability_campaign_contract.py::test_review_chain_is_reconciled_to_hashed_artifact_bytes"
      - "tests/unit/test_viability_campaign_contract.py::test_schema_contract_rejects_cross_field_and_receipt_mutations"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; 13 passed"
    rationale: "The requested complete-chain positive case and nonconforming response/re-review, missing-field, ref-pairing, duplicate-key, malformed-outcome, and public-helper totality negatives are now persisted."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-FREEZE-003"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_protocol_content_and_budget_are_frozen_before_holdout"
      - "tests/unit/test_viability_campaign_contract.py::test_campaign_is_bound_to_frozen_git_and_protocol_content"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k 'protocol_content_and_budget or frozen_git'"
        result: "exit 0; protocol byte/path/hash/threshold/resource substitutions and the existing Git/rule/requirements/contract mutations are rejected"
    rationale: "The regression starts from real passing temporary-Git campaigns and crosses both the mutable receipt boundary and the immutable protocol snapshot boundary requested by the reviewer."
    disagreement_ref: ""

new_or_changed_risks:
  - "Packet freeze version popgp-packet-freeze-v2 is intentionally incompatible with the earlier v1 freeze digest; preexisting draft packets must be re-preregistered rather than silently upgraded."
  - "Review artifacts used in campaign adjudication must exist as raw-identical blobs in locally available Git commits."
  - "Cryptographic provenance and agent review still do not demonstrate POPGP physical viability or external empirical confirmation."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "VPLAN-SCHEMA-001, VPLAN-FREEZE-003, both requested tests, all prior resolved findings/tests for regression, full diff, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder implementation status is not independent resolution. Re-review the exact response-containing commit and preserve the prior artifacts."
```

The builder does not assign final resolution status. That determination belongs to an
independent re-review artifact.
