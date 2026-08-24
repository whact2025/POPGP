# VIA-000 R3 recovery-protocol independent re-review 15

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-15"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-15"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "3a3ed71885d5c8e457644eeeffa63334464d7059"
baseline_commit: "eea317e78b47c8903e144ab64588b6a9309203b8"
prior_review_ref: "eea317e78b47c8903e144ab64588b6a9309203b8:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-14.md"
builder_response_ref: "3a3ed71885d5c8e457644eeeffa63334464d7059:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-14-RESPONSE-1.md"
context_hash: "c0882c515845800271cc544d3714dc995dc7ce81"
context_hash_method: 'git rev-parse "3a3ed71885d5c8e457644eeeffa63334464d7059^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-14-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-14.md"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and GitHub run 32742067802 logs/job/cache/artifact metadata; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-15. Exact handoff/content ancestry,
  origin, tree, prior-review/response bytes, workflow, all hosted job logs, affected
  production and receipt scripts, tests, validator, and local PowerShell parser results
  were independently inspected. No campaign implementation or external state was
  changed. Operator/orchestrator are shared; session/worktree/branch differ. Builder
  model is shared and external scientific validation is not claimed.
independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "OpenAI Codex (GPT-5)"
  builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false
hidden_access_declaration: {final_labels_seen: false, secret_seed_seen: false, private_evaluator_seen: false}
summary: |-
  CHANGES REQUESTED with one high-severity syntax-validation blocker. The correction
  is narrow and the architecture remains viable, but the complete hosted proof remains
  absent and the production runner is also syntactically invalid.

  Run 32742067802 is exact campaign head 3a3ed71885d5c8e457644eeeffa63334464d7059,
  push event, attempt 1. All six Ubuntu/Windows candidate/pdf/mutation jobs reached the
  proof-runner invocation and failed before digest or cache work at
  VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:332. PowerShell reported `ParserError` and
  `Unexpected token '-cjoin' in expression or statement.` Aggregate job 97478771044
  observed six failures and rejected before checkout/restore/output. Verifier
  97478810107 was skipped. Artifact count and run-time cache count are zero.

  Local System.Management.Automation.Language.Parser.ParseFile independently produces
  the same primary error. The defect occurs in four frozen files, not one: production
  proof runner line 332, production campaign runner line 761, and their two receipt
  copies. Each comparison contains two invalid `-cjoin` tokens. PowerShell provides
  `-join`; it does not provide a case-sensitive `-cjoin` variant.

  A one-line `-join` repair is syntactically valid but not an exact array comparison.
  Join maps distinct arrays to the same string when an element contains the delimiter;
  it also maps an empty array and a one-element empty-string array to the same value.
  Both collisions were reproduced locally. Use count plus ordinal element-by-element
  string comparison, including explicit rejection of null/non-string elements.

  The existing contract missed this because its RR14 test asserts source substrings and
  executes only VIA-000-CONTAINMENT.ps1; it never parses either runner. Campaign
  validation checks schemas, hashes, receipts, and data invariants, not PowerShell
  grammar. JSON/YAML parsing and actionlint cannot parse a referenced standalone ps1.
  Source/receipt equality faithfully duplicated the same invalid bytes. The focused
  RR14 identity test passed 1/1 and campaign validation returned valid while ParseFile
  failed four files, directly demonstrating the gap.
findings:
  - id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:332; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1:761; reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1:332; reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1:761"
    evidence: |-
      Exact Git-normalized SHA-256 values are
      ce66cfc61bf4c41fe99ce9277ee3130e0a25d320528be178d994701930a02579
      for both proof-runner copies and
      0e2129ae3724629d3acdf18e501a183fe8df5c16f6a39b8e174b968d3e8bd179
      for both production-runner copies. Thus receipt equality is exact but does not
      provide independent syntax validity.

      Jobs 97478637656, 97478637728, 97478637737, 97478637971, 97478637980,
      and 97478638001 all log the identical proof-runner line-332 parser error on the
      respective hosted operating system. Their digest, archive-tool, cache preflight,
      cache save, and envelope stages were skipped.

      ParseFile over every standalone ps1 in the R3 protocol directory and VIA-000
      receipt directory found eight files total: four valid and the four listed files
      invalid. The first errors are exactly line 332 column 30 or line 761 column 32;
      later missing-brace errors are parser cascades. A direct ParseInput probe rejects
      `-cjoin` and accepts both ordinary `-join` and explicit ordinal elementwise code.
      Collision probes proved `@('a','b\nc')` and `@('a\nb','c')` compare equal after
      joining with literal `\n`, as do `@()` and `@('')`, although each pair is a
      different array.

      `test_r3_rr14_no_lua_low_il_production_identity_is_bound` passed in 0.49 seconds
      and the campaign validator reported valid at the exact head. Repository search
      found no ParseFile, ParseInput, or Language.Parser gate in tests/scripts. The
      RR14 response reports JSON/YAML parsers and actionlint, but neither validates
      referenced standalone PowerShell grammar.
    finding: "Two production R3 PowerShell runners and their receipt copies contain a nonexistent operator, and sealing has no fail-closed PowerShell syntax gate."
    failure_scenario: "The hosted proof or production workflow invokes a hash-valid, receipt-equal script whose parser aborts before any contained stage, digest, cache, evidence, or commitment can be created."
    consequence: "No retained 2x3 proof exists, the production runner cannot start, and merge, signer/custodian gates, refreeze, activation, and holdout remain unauthorized."
    required_action: |-
      Replace every affected comparison coherently in both production scripts and both
      byte-identical receipt copies. Do not serialize arrays with `-join`. Implement one
      reviewed exact-array predicate that first requires equal counts, then for each
      index requires both values to be strings and
      `[string]::Equals(left, right, [StringComparison]::Ordinal)`. It must distinguish
      empty from one empty element, preserve order and case, reject null/non-string
      entries, and return exactly one Boolean. Apply it to token_restriction_flags at
      proof-runner line 332 and campaign-runner line 761. Retain existing schema and
      privilege checks.

      Add a mandatory pre-seal regression that enumerates every `.ps1` beneath the R3
      protocol directory and the public VIA-000 receipt directory, sorts and records
      the exact path set, calls
      `[System.Management.Automation.Language.Parser]::ParseFile` on each, and fails on
      any parser error. It must run with supported PowerShell 7 on Windows and Ubuntu,
      cannot skip when pwsh is absent, and must include a negative fixture proving an
      injected invalid operator is rejected. Also test exact-array equality against
      order, case, count, null/type, delimiter-collision, and empty-array mutations.
      Keep source/receipt hash equality as a separate gate.

      After all eight frozen scripts parse with zero errors, rerun the focused R3 tests,
      campaign validator, and exact proof-only hosted workflow at a new immutable head.
      Require six green cells, six distinct exact-key cache transports, retained
      seven-file aggregate, green redownload verifier, and independent artifact review.
      The prior failed run cannot be reused.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001"
    description: "Parse every R3 protocol and public VIA-000 receipt ps1 with PowerShell 7 ParseFile and zero errors; negatively inject invalid grammar; separately prove ordinal elementwise token-flag equality rejects case/order/count/type/null/delimiter/empty-array substitutions; then rerun and retain the exact hosted 2x3 proof."
    rationale: "Hash equality and source-token assertions do not prove executable syntax, while join-based comparison is lossy even after changing to the valid `-join` operator."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved Git-normalized behavior; standalone runner syntax is the current distinct scope.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Fixed workflow input reaches the frozen runner and fails closed on parse.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "No successful retained hosted proof exists.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "All cells fail before contained execution.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "No hosted subject or descendant evidence is produced.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Safe proof path fails before execution and retention.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "No Windows envelope is produced.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No envelope reaches transport.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "No cache stage executes.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Built-in control planes execute and report the standalone-script parser error.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "No custom no-op shell recurred.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "RR14 no-LUA implementation remains present; current failure occurs before it is invoked.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: unresolved, evidence: "The implementation and local containment test exist, but its required hosted proof fails on invalid downstream PowerShell syntax.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "The fixed proof dispatch reached the frozen runner.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "No retained hosted proof.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "No contained stage executes.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "No hosted subject exists.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Six cells fail before evidence.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "No envelope is produced.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No transport executes.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "No cache executes.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: unresolved, evidence: "Built-in execution works; retained six-cell proof is still absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved diagnostic result.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: unresolved, evidence: "Local/static portions pass, but required hosted 2x3 retention fails before execution.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", notes: ""}
predictions: {experiment_id: "", predicted_outcome: "", predicted_failure_mode: "", confidence_statement: "No scientific execution occurred."}
recommendation: {approve: false, blocking_findings: 1, rationale: "Bounded CHANGES REQUESTED. Replace all invalid runner comparisons with exact ordinal elementwise array equality, add the mandatory all-R3-ps1 parse gate, and rerun/retain the exact hosted 2x3 proof. Architecture remains viable."}
```

## Verification ledger

- Exact handoff `3a3ed71885d5c8e457644eeeffa63334464d7059`, tree `c0882c515845800271cc544d3714dc995dc7ce81`, sole parent/content `cefe020104ba1919cbcb0355981e0b34eba46c5b`, tree `bacbfe920fa2c412fd603ab86fb943efa2a3ebf4`, and origin campaign head matched. RR14 review Git-normalized SHA-256 was `74fe3edac45eb1e9bff1e2ff35a51f9c6ac41ea273037ed22a8d94fe85030168`; RR14 response SHA-256 was `99d7add55eb5fa6b14488d959856d75c4318965bacf1c785c2399a2b742e0561`.
- Run `32742067802` exact head/event/attempt, all eight jobs, every step, and all six producer logs were inspected. Six identical parser failures occurred before digest/cache; aggregate failed closed; verifier skipped; artifacts and matching run-time caches were zero.
- Direct ParseFile closure: four passes and four failures among eight R3 protocol/receipt PowerShell files. Direct ParseInput: invalid `-cjoin` failed, ordinary `-join` parsed, explicit ordinal elementwise comparison parsed. Both join-collision probes and the focused-test/validator false-negative controls reproduced.
- Focused test: `1 passed in 0.49s`. Campaign validator: valid. These are negative evidence for the missing syntax gate, not evidence that either runner executes.
- R3 remains drafted, `holdout_started: false`, unrevealed, pending, and has no R3 tag. No signer, key, custody, lifecycle, holdout, scientific execution, commitment, or reveal was accessed, performed, or authorized.
