# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-1

```yaml
artifact_schema_version: 1
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-1"
response_round: 1
response_date: "2026-08-10"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1.md"
review_commit: "064a233c426f1c620184809bad3a06d8a531a7d3"
candidate_commit_reviewed: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign data were available. Remediation used only the frozen public/local review, repository contents, and generated test fixtures."

summary: |-
  All six blocking findings and all six requested tests were accepted and implemented in
  4930f95c85f64f2067df0fe99db8472d79a95eff. The plan now delegates campaign authority to
  versioned packet, campaign, and requirements contracts plus a fail-closed validator. The
  validator enforces typed executable rules, receipt hashes, custody and reveal ordering,
  dependency closure, campaign-owned evidence floors, complete Tier G capabilities, review-chain
  closure, and deterministic round/campaign outcomes. The complete authoritative suite passes
  with 168 tests, all six documented examples regenerate, and committed validation artifacts
  remain conformant. This response does not assign resolution; that belongs to re-review.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Added versioned campaign/packet JSON Schemas, canonical requirements, and a repository validator. The contract now has one lifecycle/outcome model, a typed Boolean expression AST with hash-verified JSON Pointer bindings, referential integrity, DAG checks, receipt existence/hashing, evidence ordering, decisive-receipt checks, review-chain closure, and computed campaign adjudication."
    changed_files: ["schemas/viability/campaign-v1.schema.json", "schemas/viability/packet-v1.schema.json", "schemas/viability/requirements-v1.json", "scripts/check_viability_campaign.py", "docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml", "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml", "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md", "tests/unit/test_viability_campaign_contract.py"]
    fix_commits: ["4930f95c85f64f2067df0fe99db8472d79a95eff"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py", result: "exit 0; 9 passed"}, {command: "uv run pytest -q", result: "exit 0; 168 passed"}]
    residual_risk: "Version 1 validates local file receipts; integrations that import receipts from external orchestrators still need adapters that preserve the same bytes and hashes."
    disagreement_ref: ""

  - finding_id: "VPLAN-CUSTODY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Added a required evaluator/custodian seat, per-seat identity/model/operator/session/access/exposure records, prohibited role/session reuse, SHA-256 raw-byte manifest commitments, pre-reveal output commitments, reveal authorization/timestamps, post-reveal hash verification, and retention/archive records."
    changed_files: ["schemas/viability/packet-v1.schema.json", "scripts/check_viability_campaign.py", "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml", "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md", "tests/unit/test_viability_campaign_contract.py"]
    fix_commits: ["4930f95c85f64f2067df0fe99db8472d79a95eff"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k custody", result: "exit 0; custody mutation coverage passed"}]
    residual_risk: "The validator proves consistency of retained receipts, not the real-world honesty of identity or exposure declarations; unaffiliated custody remains necessary for external validation."
    disagreement_ref: ""

  - finding_id: "VPLAN-SCI-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Tier G now requires 3D recovery, acceleration/geodesic response, same-source lensing and Shapiro delay, two-potential consistency, laboratory interference and entanglement statistics, Lorentz bounds, and operational no-signaling. Each required capability must have an executable passing rule, so listing a capability name cannot satisfy the tier."
    changed_files: ["schemas/viability/requirements-v1.json", "schemas/viability/packet-v1.schema.json", "scripts/check_viability_campaign.py", "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md", "tests/unit/test_viability_campaign_contract.py"]
    fix_commits: ["4930f95c85f64f2067df0fe99db8472d79a95eff"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k tier_g", result: "exit 0; Tier G omission and false-rule countermodels rejected"}]
    residual_risk: "These are executable campaign gates, not evidence that POPGP currently passes any Tier G physics comparison."
    disagreement_ref: ""

  - finding_id: "VPLAN-OUTCOME-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Separated round status from packet adjudication. Invalid protocols produce an invalid round and pending packet; valid rounds require exactly one pass/fail/blocked expression; disjoint cause-code classes distinguish scientific failure, infrastructure blockage, and invalid evidence. Campaign precedence is failed, then blocked, then all-passed, otherwise pending."
    changed_files: ["schemas/viability/packet-v1.schema.json", "schemas/viability/campaign-v1.schema.json", "scripts/check_viability_campaign.py", "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md", "tests/unit/test_viability_campaign_contract.py"]
    fix_commits: ["4930f95c85f64f2067df0fe99db8472d79a95eff"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k outcome_truth_table", result: "exit 0; deterministic truth-table cases passed"}]
    residual_risk: "Campaign authors must preregister expressions and cause codes correctly; the validator prevents contradictory execution but cannot establish scientific adequacy of a chosen threshold."
    disagreement_ref: ""

  - finding_id: "VPLAN-DEP-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Replaced conceptual dependencies with canonical packet IDs, corrected all execution waves, validates an acyclic lower-wave DAG, and rejects holdout execution until every prerequisite packet is adjudicated passed."
    changed_files: ["schemas/viability/requirements-v1.json", "scripts/check_viability_campaign.py", "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md", "tests/unit/test_viability_campaign_contract.py"]
    fix_commits: ["4930f95c85f64f2067df0fe99db8472d79a95eff"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k 'requirements or dependencies'", result: "exit 0; missing, cycle, wave, and premature-holdout cases rejected"}]
    residual_risk: "The DAG enforces prerequisite order but does not schedule or provision external agent infrastructure."
    disagreement_ref: ""

  - finding_id: "VPLAN-EVIDENCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Canonical campaign requirements now own immutable per-packet evidence floors. VIA-300/400/500/600/700/800 require at least E4 and VIA-900 requires E5; a packet may raise but cannot lower its floor, and cumulative typed evidence receipts are required."
    changed_files: ["schemas/viability/requirements-v1.json", "scripts/check_viability_campaign.py", "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md", "tests/unit/test_viability_campaign_contract.py"]
    fix_commits: ["4930f95c85f64f2067df0fe99db8472d79a95eff"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k evidence_floors", result: "exit 0; E4/E5 downgrades and unknown levels rejected; stricter packet floor accepted"}]
    residual_risk: "Evidence labels remain governance classifications; receipt review must still assess whether the underlying evidence genuinely meets its declared class."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_schema_contract_rejects_cross_field_and_receipt_mutations", "tests/unit/test_viability_campaign_contract.py::test_schema_contract_rejects_unresolved_or_incomplete_review_chain", "tests/unit/test_viability_campaign_contract.py::test_requirements_reject_missing_cycles_and_invalid_waves"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py", result: "exit 0; 9 passed"}]
    rationale: "Rejects lifecycle/evidence contradictions, unknown claims/gates, missing and hash-mismatched receipts, empty decisive evidence, malformed bindings, incomplete review chains, missing dependencies, cycles, and invalid waves through the same validator."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_blind_custody_rejects_leaks_role_reuse_and_manifest_mutations"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k custody", result: "exit 0; requested custody mutations rejected"}]
    rationale: "Covers per-seat exposure, prohibited session/role reuse, manifest bytes and canonicalization, reveal timing, missing custodian/retention data, and post-reveal substitution."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-SCI-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_tier_g_countermodel_cannot_omit_minimum_physics_comparisons"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k tier_g", result: "exit 0; omitted and false capabilities prevent pass"}]
    rationale: "The lower-dimensional countermodel cannot pass without every canonical Tier G comparison and an executable true rule for each."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_outcome_truth_table_is_deterministic"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k outcome_truth_table", result: "exit 0; truth table passed"}]
    rationale: "Covers scientific negative, capability code defect, inaccessible infrastructure, tested resource exhaustion, invalid protocol, missing receipt, and simultaneous fail/block predicates."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-DEP-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_requirements_reject_missing_cycles_and_invalid_waves", "tests/unit/test_viability_campaign_contract.py::test_holdout_cannot_start_until_dependencies_pass"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k 'requirements or dependencies'", result: "exit 0; requested dependency mutations rejected"}]
    rationale: "Parses the canonical packet register as a DAG and rejects missing/conceptual dependencies, cycles, same/later-wave prerequisites, and premature holdout execution."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_viability_campaign_contract.py::test_campaign_owned_evidence_floors_cannot_be_downgraded"]
    verification: [{command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py -k evidence_floors", result: "exit 0; requested evidence-floor cases passed"}]
    rationale: "VIA-300 and VIA-400 cannot pass below E4, VIA-900 cannot pass below E5, unknown levels fail, and stricter packet requirements are allowed."
    disagreement_ref: ""

new_or_changed_risks:
  - "The validator establishes a portable local contract but does not itself provide an external orchestration or receipt-storage service."
  - "Tier G gates are now explicit and stricter; no current result is promoted, and actual viability remains unproven until a complete campaign passes."
  - "Identity, exposure, and custody receipts are auditable assertions; external scientific independence still requires unaffiliated operators and infrastructure."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "all six findings, all six requested tests, regressions, exact artifact identity, new findings, and executable validator mutation resistance"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder dispositions are not resolution labels. The campaign plan remains a viability protocol, not evidence that POPGP has passed Tier R, G, or E."
```

The builder does not assign final resolution status. That determination belongs to
an independent re-review artifact.
