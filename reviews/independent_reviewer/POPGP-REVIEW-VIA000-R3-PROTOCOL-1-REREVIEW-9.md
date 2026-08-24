# VIA-000 R3 recovery-protocol independent re-review 9

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-9"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-9"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "6be7f53e403dadc928d0aa6d652dfc3aac7a8a9a"
baseline_commit: "12158dd52288a08642d7d29376fbb94cd7809b25"
prior_review_ref: "12158dd52288a08642d7d29376fbb94cd7809b25:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-8.md"
builder_response_ref: "6be7f53e403dadc928d0aa6d652dfc3aac7a8a9a:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-8-RESPONSE-1.md"
context_hash: "53028476a2f69f1398339d3ca4df0a79a9743700"
context_hash_method: 'git rev-parse "6be7f53e403dadc928d0aa6d652dfc3aac7a8a9a^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-ENVELOPE.schema.json"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-8-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-8.md"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and GitHub run 32710277023 logs/metadata; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-9. Exact commit/tree/parent/origin and
  response/prior-review hashes were verified. All seven hosted job logs and artifact
  API results were independently inspected. No implementation or external state was
  changed. Operator/orchestrator are shared; session/worktree/branch differ. Builder
  model is unknown and external validation is not claimed.
independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "unknown"
  builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false
hidden_access_declaration: {final_labels_seen: false, secret_seed_seen: false, private_evaluator_seen: false}
summary: |-
  CHANGES REQUESTED with one high-severity export/transport blocker; the architecture
  remains viable. Run 32710277023 was a non-scientific push at the exact reviewed
  handoff. Windows PDF/candidate/mutation jobs 97379982873, 97379982937, and
  97379982960 all passed production containment. Each trusted step then required one
  output directory entry named envelope.json, regular/non-reparse identity, and
  successful canonical envelope validation before returning success. The following
  pinned artifact step nevertheless reported that exact envelope path absent in all
  three jobs. Canonicalization did not solve Windows artifact discovery.

  Ubuntu jobs 97379982708, 97379983023, and 97379983143 failed before envelope export
  with `Cannot bind argument to parameter 'Bytes' because it is an empty array`.
  Source identifies Get-ProofBytesSha256's mandatory byte-array parameter; legitimate
  zero-byte stdout/stderr reaches it. The narrow correction is to permit an empty
  collection (`[AllowEmptyCollection()]`) or call SHA256.HashData directly without
  parameter binding, with regression assertions for the standard empty SHA-256.

  Aggregate 97380096956 correctly failed at the all-cells requirement and skipped
  checkout, download, validation, and upload. Artifact API total_count was zero. No
  campaign/lifecycle/signing/custody/holdout/scientific/commitment/reveal action ran.

  Replacing the matrix outputs with six explicit proof jobs and one Ubuntu aggregate
  is a bounded viable closure. Matrix job outputs are unsuitable because same-named
  outputs from matrix children collide with unspecified last-writer behavior. Explicit
  jobs provide six statically unique `needs` channels and identities.
findings:
  - id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:123-128,394-504"
    evidence: |-
      All three Windows containment/envelope construction steps passed their exact-one
      regular envelope checks, then upload-artifact reported the explicit RUNNER_TEMP
      envelope absent. All three Ubuntu cells failed at the same empty-array Bytes
      binder before export. No artifact exists and aggregate failed closed. Thus RR8
      remains open; the failure is transport/binding, not containment execution.
    finding: "The canonical envelope is valid in trusted step memory but cannot yet cross all six hosted job boundaries into one retained aggregate."
    failure_scenario: "Matrix outputs overwrite one another, GitHub skips a secret-like or oversized output, a newline/control value corrupts transport, or one missing/truncated/cross-cell envelope is accepted as another cell."
    consequence: "Exact 2x3 retained evidence remains absent and RR4/RR6 cannot be approved."
    required_action: |-
      Use six explicit jobs with unique job output names. After teardown and canonical
      validation, read bounded envelope bytes, require a conservative decoded cap and
      base64 expansion well below GitHub's 1 MiB per-job UTF-16-counted output limit,
      encode single-line standard base64, reject CR/LF/control characters, and append
      exactly one value through trusted GITHUB_OUTPUT without logging it. Untrusted
      processes must never inherit/read GITHUB_OUTPUT or RUNNER_TEMP and must be dead
      before this step. Use minimal read-only permissions and no secrets.

      The Ubuntu aggregate must require six distinct nonempty `needs` outputs, enforce
      encoded/decoded caps before allocation, strict-decode in memory, validate UTF-8
      canonical JSON and envelope schema, recompute envelope/inner hashes, and require
      exact run/attempt/repository/workflow/source/platform/stage 2x3 uniqueness. Reject
      missing, duplicate, equal, overwritten, truncated, masked, newline, corrupt, or
      cross-cell values. Only then write six canonical envelopes plus one canonical
      aggregate into a fresh Ubuntu output directory and upload that single directory.
      Re-download and revalidate the artifact in a subsequent job. Consolidated Ubuntu
      retention is sufficient because it preserves the exact six validated envelope
      byte strings and their identities; per-cell artifact transport is not required.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001"
    description: "Prove empty stdout/stderr SHA-256; six explicit unique job outputs; caps below GitHub accounting limits; strict base64/canonical validation; missing/empty/duplicate/collision/truncated/masked/newline/corrupt/cross-cell rejection; one Ubuntu consolidated artifact; subsequent download and exact seven-file/hash revalidation; no envelope log disclosure or post-job untrusted access."
    rationale: "Only end-to-end needs transport plus retained aggregate evidence closes the platform artifact boundary."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Windows containment passed but no retained 2x3 artifact exists.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Blocked by RR9."}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Containment ran, but retained 2x3 proof is absent.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Blocked by RR9."}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: verified-resolved, evidence: "Safe non-scientific hosted workflow exists and ran.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Canonical envelope remains undiscoverable by Windows upload and Ubuntu binder regressed.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Superseded in remedy by RR9 transport finding."}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "2x3 evidence absent.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "2x3 evidence absent.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted path exists but no complete artifact.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Attempt retained zero artifacts.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: "Superseded by RR9 test."}
predictions: {experiment_id: "", predicted_outcome: "", predicted_failure_mode: "", confidence_statement: "No scientific execution occurred."}
recommendation: {approve: false, blocking_findings: 1, rationale: "Fixable CHANGES REQUESTED. Implement six explicit bounded outputs, strict Ubuntu consolidation, retained artifact revalidation, and the zero-byte hash regression; architecture remains viable."}
```

## Verification ledger

- Exact handoff/tree/content/tree/origin, response SHA-256, and prior-review SHA-256 matched.
- Run 32710277023 and jobs 97379982708/83023/83143, 97379982873/82937/82960, and 97380096956 inspected; artifact count zero.
- Windows: containment plus exact-one-envelope validation passed, artifact discovery failed. Ubuntu: all cells failed on empty Bytes binding. Aggregate failed closed.
- No signer, tag, custody, lifecycle, holdout, scientific execution, commitment, or reveal occurred.
