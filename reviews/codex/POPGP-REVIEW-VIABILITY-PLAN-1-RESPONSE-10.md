# Builder response: POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-10

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-10"
response_round: 10
response_date: "2026-08-11"

builder_seat: builder
builder_model_identity: "OpenAI Codex GPT-5"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"
builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
builder_orchestrator_id: "codex-multi-agent-root"
builder_organization: "user-operated local workspace"

review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-9"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-9.md"
review_commit: "b674a5522a19613e1d36c96f34f32ee253feb37f"
candidate_commit_reviewed: "07130dd7c9ad92d85ead1cf3fd90e72d691db93d"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden campaign inputs, private evaluator, or unaffiliated replication results were available."

summary: |-
  The unresolved historical schema finding and new bounded-resource finding from
  REREVIEW-9 were accepted and implemented in
  8e491036bc56e216ad5a8dea42ceecb3c7b4add5. JSON float parsing now preserves an
  exact Decimal check before conversion and rejects exponent overflow, nonzero
  underflow to zero, non-finite values, and numeric tokens longer than 256 characters.
  Integer tokens have the same length bound. Every hash-valid JSON, YAML, or Markdown
  structured receipt is parsed and traversed even if no outcome binding selects it.

  External bundle cloning no longer uses captured pipes. It runs in a new process
  group/session with a 30-second execution deadline and five-second cleanup budget.
  Timeout kills the Windows process tree with taskkill or the POSIX process group with
  SIGKILL, reaps the parent within the cleanup deadline, and lets TemporaryDirectory
  remove the checkout. The persisted full Tier-E regression reconciles every bundle,
  provenance, contract, and receipt hash around the exact 16 MiB sparse-zero bundle;
  it returns one controlled error within the 35-second total bound and retains a valid
  bundle positive control. A separate parent/heartbeat-child test proves descendant
  termination. The final repository suite passed all 179 tests, and repeated OS checks
  found no matching Git process or bundle checkout. These are local contract results,
  not POPGP scientific or external-independence evidence.

finding_responses:
  - finding_id: "VPLAN-SCHEMA-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      `_load_json_text` now supplies bounded integer and float parsers. The float parser
      checks the exact Decimal token and its binary float conversion, rejecting
      non-finite conversion and nonzero values that underflow to zero. Receipt mapping
      parses every hash-valid structured receipt before downstream binding selection,
      so an unused `1e999` field cannot bypass the invariant. Full Tier-R regressions
      reconcile the raw-result and output-commitment hashes for exact `1e999` and
      `1e-999` tokens and require deterministic parse errors.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["8e491036bc56e216ad5a8dea42ceecb3c7b4add5"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed"
        result: "exit 0; hash-reconciled unused exponent overflow and underflow fields were rejected with the other structured-receipt controls"
      - command: "uv run pytest -q"
        result: "exit 0; all 179 repository tests passed on the final implementation in 760.94 s"
      - command: "uv run ruff check . && uv run python scripts/check_tex.py"
        result: "exit 0; repository lint and the 652-line manuscript source check passed"
    residual_risk: "The 256-character numeric-token and binary-float representability rules are deliberate contract limits; broader numeric domains require a versioned parser and schema change."
    disagreement_ref: ""

  - finding_id: "VPLAN-RESOURCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Bundle validation uses a dedicated bounded-process helper with DEVNULL output,
      eliminating captured-pipe drain. The clone owns a process group/session. On
      timeout the helper terminates descendants and reaps the parent under one bounded
      cleanup deadline. The exact reviewer sparse bundle is persisted in a complete
      Tier-E campaign, while a separate spawned-child heartbeat proves that descendants
      stop after timeout on the current Windows host.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_viability_campaign_contract.py"
      - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits: ["8e491036bc56e216ad5a8dea42ceecb3c7b4add5"]
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_bounded_process_terminates_descendants"
        result: "exit 0; timed-out parent and heartbeat child stopped within the asserted bound"
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_tier_e_binds_outputs_git_bundle_orchestrator_and_comparison"
        result: "exit 0; exact 16 MiB sparse-bundle campaign returned one error inside the total bound, removed its checkout, and the valid-bundle control passed (152.90 s for the complete test)"
      - command: "post-test Win32 process and TEMP popgp-external-bundle-* inspection"
        result: "zero matching Git processes and zero temporary bundle directories after focused, contract, and full-suite runs"
    residual_risk: "Windows cleanup depends on the operating-system taskkill tree primitive and POSIX cleanup on process-group SIGKILL; both remain subject to host scheduler delay within the explicit five-second test tolerance."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py::test_structured_receipts_and_governance_provenance_fail_closed"
        result: "exit 0; exact unused overflow and underflow tokens failed closed after complete receipt/hash reconciliation"
    rationale: "Global numeric parsers plus unconditional hash-valid receipt parsing cover selected and unselected structured fields at every receipt call site."
    disagreement_ref: ""

  - requested_test_id: "TST-VPLAN-RESOURCE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_campaign_contract.py::test_bounded_process_terminates_descendants"
      - "tests/unit/test_viability_campaign_contract.py::test_tier_e_binds_outputs_git_bundle_orchestrator_and_comparison"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_campaign_contract.py"
        result: "exit 0; all 20 contract tests passed in 659.45 s, including full malformed/valid bundle paths; the final 179-test run re-executed the complete file after the cleanup-deadline refinement"
      - command: "Get-CimInstance Win32_Process plus TEMP popgp-external-bundle-* inspection"
        result: "zero matching Git processes and zero temporary bundle directories after each final-state bundle/full-suite run"
    rationale: "The full-campaign regression exercises the exact hostile bundle and checkout cleanup; the direct process-tree regression makes descendant termination observable and portable."
    disagreement_ref: ""

new_or_changed_risks:
  - "JSON numeric tokens are intentionally limited to 256 characters and finite binary-float range; underflow to zero is invalid rather than silently rounded."
  - "Bundle clone diagnostics intentionally omit child stdout/stderr so hostile descendants cannot retain captured pipes or fill an unbounded diagnostic buffer."
  - "The clone execution deadline is 30 seconds and cleanup-inclusive total bound is 35 seconds, with tests allowing five seconds of host scheduling tolerance."
  - "Git-bundle content size is not separately capped; the process-tree deadline remains the controlling portable resource bound."
  - "No POPGP Tier R, G, or E campaign or external empirical validation was produced by this remediation."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA, including the POSIX process-group regression."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""
  - action: "Before any Tier E claim, have an unaffiliated custodian verify organization, exposure, authorship, and retained Git-bundle provenance off-system."
    owner: "future campaign operator"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "VPLAN-SCHEMA-001, VPLAN-RESOURCE-001, TST-VPLAN-SCHEMA-001, TST-VPLAN-RESOURCE-001, every prior finding/test for regression, the complete history diff, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Builder implementation status is not independent resolution. Re-review the exact response-containing commit and preserve every prior review and response artifact."
```

The builder does not assign final resolution status. That determination belongs to an
independent re-review artifact.
