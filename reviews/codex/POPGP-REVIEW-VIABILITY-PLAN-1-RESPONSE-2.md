# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-2

```yaml
artifact_schema_version: 1
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-2"
response_round: 2
response_date: "2026-08-10"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-1"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-1.md"
review_commit: "3c84fc6c0d5ef31d314da5fd848b9f54f66e65dd"
candidate_commit_reviewed: "fba32f03ba92998e587560793f6118cf618a41f7"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign data were available. The five accepted blockers were remediated from the review's frozen mutation evidence; dependency and evidence-floor controls verified by the reviewer were retained."

summary: |-
  All five blockers and all five unresolved/requested tests from re-review 1 were accepted
  and implemented in 0de2154de5ea5dc936cfb695ed09368e449a5231. Contract v1 was withdrawn
  and replaced by breaking v2 schemas, requirements, and rule language. V2 requires canonical
  raw Boolean capability gates and strict JSON types, derives review state from hashed artifact
  bytes, binds reveal to the evaluator/custodian and runner output bytes, makes malformed outcome
  combinations return errors, and binds candidate/protocol/rules/requirements/contract files to
  real Git objects. The authoritative suite passes with 170 tests and all six examples. Builder
  dispositions are not resolution labels and POPGP viability itself remains unproven.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Rule language v2 requires every pass/fail/block expression to consume a SHA-256-verified raw-results binding with an explicit JSON type. Each capability must use its same-named Boolean binding at /capabilities/<name> and the canonical equality-to-true rule. Equality rejects Boolean/integer substitution. Review summaries are recomputed from parsed JSON/YAML/Markdown review, response, and re-review receipt bytes and must match exactly; coverage, blocker counts, round pairing, and supersession successors are validated."
    changed_files: ["scripts/check_viability_campaign.py", "schemas/viability/packet-v2.schema.json", "schemas/viability/campaign-v2.schema.json", "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md", "tests/unit/test_viability_campaign_contract.py"]
    fix_commits: ["0de2154de5ea5dc936cfb695ed09368e449a5231"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py", result: "exit 0; 11 passed, including raw-false, integer/Boolean, omitted-review, and dangling-supersession mutations"}, {command: "uv run pytest -q", result: "exit 0; 170 passed"}]
    residual_risk: "The contract proves that decisions consume named raw fields and frozen expressions; reviewers must still assess whether preregistered metrics and thresholds are scientifically adequate."
    disagreement_ref: ""

  - finding_id: "VPLAN-CUSTODY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Only the evaluator/custodian identity may authorize reveal, and reveal requires holdout_started plus lifecycle reproduced or later. The reproduction runner identity must create a structured output-commitment receipt that identifies and hashes the raw-results receipt. A separate structured reveal receipt must match authorization, time, output commitment, and both post-reveal manifest hashes."
    changed_files: ["scripts/check_viability_campaign.py", "schemas/viability/packet-v2.schema.json", "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml", "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md", "tests/unit/test_viability_campaign_contract.py"]
    fix_commits: ["0de2154de5ea5dc936cfb695ed09368e449a5231"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k custody", result: "exit 0; builder-authorized reveal, preregistration reveal, wrong committer, and raw-output hash mismatch rejected with the prior exposure/session/hash cases"}]
    residual_risk: "Cryptographic receipt consistency cannot prove that a human or agent truthfully declared every off-system exposure; unaffiliated custody remains required for external validation."
    disagreement_ref: ""

  - finding_id: "VPLAN-SCI-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Tier G capability names now correspond to mandatory raw Boolean fields. The validator rejects binding-free or alternate-pointer capability rules, then evaluates every canonical capability gate. A countermodel with false raw 3D recovery and lensing is rejected even when all other Tier G fields pass."
    changed_files: ["scripts/check_viability_campaign.py", "schemas/viability/packet-v2.schema.json", "schemas/viability/requirements-v2.json", "tests/unit/test_viability_campaign_contract.py", "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"]
    fix_commits: ["0de2154de5ea5dc936cfb695ed09368e449a5231"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k tier_g", result: "exit 0; capability omissions, literal-rule substitution, raw-false countermodel, and integer-as-Boolean mutation rejected"}]
    residual_risk: "No current POPGP evidence satisfies these gates; the change prevents false promotion but does not demonstrate Tier G."
    disagreement_ref: ""

  - finding_id: "VPLAN-OUTCOME-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Cause validation now uses the computed exclusive outcome, so a schema-valid pending declaration cannot index a nonexistent cause class or crash. The persisted truth table covers scientific negative, implementation-capability failure, tested resource exhaustion, unavailable TeX/toolchain, hardware, external access, missing receipt, simultaneous fail/block, invalid protocol, invalid receipt, and valid-round/pending contradiction."
    changed_files: ["scripts/check_viability_campaign.py", "tests/unit/test_viability_campaign_contract.py"]
    fix_commits: ["0de2154de5ea5dc936cfb695ed09368e449a5231"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k outcome_truth_table", result: "exit 0; all requested rows pass and valid/pending returns a deterministic error"}]
    residual_risk: "Cause codes remain preregistered classifications whose scientific interpretation requires independent adjudication."
    disagreement_ref: ""

  - finding_id: "VPLAN-FREEZE-002"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Campaign validation now resolves candidate, baseline, and protocol Git commits; recomputes the candidate tree; loads a hash-verified protocol manifest from the protocol commit; verifies the exact requirements and contract-file blobs; rejects execution with post-protocol contract changes; and compares canonical packet-rule hashes to the manifest. Requirements semantics moved to v2 with a pinned canonical content hash, so same-version downgrades fail. Cross-platform helpers print canonical packet and Git-blob SHA-256 values."
    changed_files: ["scripts/check_viability_campaign.py", "schemas/viability/campaign-v2.schema.json", "schemas/viability/packet-v2.schema.json", "schemas/viability/protocol-manifest-v2.schema.json", "schemas/viability/requirements-v2.json", "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json", "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md", "tests/unit/test_viability_campaign_contract.py"]
    fix_commits: ["0de2154de5ea5dc936cfb695ed09368e449a5231"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k frozen_git", result: "exit 0; nonexistent commits, wrong tree, post-protocol rules, same-version requirements downgrade, and changed executing validator rejected"}, {command: "uv run python scripts/check_viability_campaign.py --git-blob-sha256 HEAD README.md", result: "exit 0; canonical blob hash printed"}]
    residual_risk: "Git establishes immutable local provenance, not independent timestamping or institutional custody; external campaigns should additionally anchor the protocol commit in an external append-only service."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_schema_contract_rejects_cross_field_and_receipt_mutations", "tests/unit/test_viability_campaign_contract.py::test_review_chain_is_reconciled_to_hashed_artifact_bytes", "tests/unit/test_viability_campaign_contract.py::test_tier_g_countermodel_cannot_omit_minimum_physics_comparisons"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py", result: "exit 0; 11 passed"}]
    rationale: "Adds the missing raw-binding/type, artifact-derived review closure, and dangling-supersession mutations while retaining the original schema, receipt, identifier, evidence, and DAG cases."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_blind_custody_rejects_leaks_role_reuse_and_manifest_mutations"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k custody", result: "exit 0; expanded custody matrix passed"}]
    rationale: "Persists custodian-only authorization, reproduced-phase reveal, runner identity, and runner-output-byte binding in addition to exposure/session/manifest/retention mutations."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-SCI-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_tier_g_countermodel_cannot_omit_minimum_physics_comparisons"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k tier_g", result: "exit 0; raw-false Tier G countermodel rejected"}]
    rationale: "The requested countermodel now records failed 3D and same-source lensing observations in the hashed raw receipt; canonical capability expressions consume them and prevent pass."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_outcome_truth_table_is_deterministic"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k outcome_truth_table", result: "exit 0; full requested classification matrix passed"}]
    rationale: "All named failure, blockage, invalidity, missing-evidence, ambiguity, and malformed cross-field rows are now persisted; the public API returns errors instead of raising."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-FREEZE-002"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_campaign_is_bound_to_frozen_git_and_protocol_content"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k frozen_git", result: "exit 0; every requested Git, tree, packet-rule, requirements, and contract-file mutation rejected"}]
    rationale: "The test uses real temporary Git commits and blobs; placeholder hashes are no longer a positive fixture."
    disagreement_ref: ""

new_or_changed_risks:
  - "V2 is intentionally incompatible with the withdrawn v1 draft; v1 campaigns must be preregistered again under v2 and cannot be silently translated."
  - "The validator now invokes local Git and requires the protocol commit and referenced blobs to be available in the repository object database."
  - "Git provenance and cryptographic receipts do not constitute external scientific validation or demonstrate any POPGP viability tier."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "all five unresolved findings, all five requested tests, the two previously resolved findings/tests for regression, full v2 diff, validator mutation resistance, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder dispositions are not resolution labels. Approval still requires an independent re-review of the frozen v2 candidate."
```

The builder does not assign final resolution status. That determination belongs to
an independent re-review artifact.
