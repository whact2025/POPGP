# VIA-000 R3 recovery-protocol independent re-review 8

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-8"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-8"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "611b39cc2d2bb68ee4a8705f32f8d36b2d53d5dd"
baseline_commit: "5c588f66e43bc9a100b91ebb9a6cf48b197369ce"
prior_review_ref: "5c588f66e43bc9a100b91ebb9a6cf48b197369ce:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-7.md"
builder_response_ref: "611b39cc2d2bb68ee4a8705f32f8d36b2d53d5dd:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-7-RESPONSE-1.md"
context_hash: "dbcc7450c0c4dcde0f88e72d539c262fd32c5f81"
context_hash_method: 'git rev-parse "611b39cc2d2bb68ee4a8705f32f8d36b2d53d5dd^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-7-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-7.md"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository, GitHub run 32705009673 logs/metadata, and downloaded non-scientific Ubuntu proof artifacts; no custody, signer, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-8. Exact three-layer commit/tree
  ancestry, origin, RR7 response/artifact hashes, hosted run/job/artifact metadata,
  logs, and downloadable subjects were independently inspected. No implementation or
  lifecycle state was changed. Operator/orchestrator are shared; session/worktree/
  branch differ. Builder model is unknown and external validation is not claimed.
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
  CHANGES REQUESTED with one high-severity, fixable packaging blocker. Run
  32705009673 was a non-scientific push workflow at source
  689ba359f229dea5bead7e90e6dc84dde2e2fc4a. All three Ubuntu cells completed and
  retained artifacts 9511903681 (candidate), 9511903097 (PDF), and 9511902419
  (mutation). Downloaded artifacts contain exactly containment-result.json,
  proof.json, stdout.txt, and stderr.txt. They bind run/attempt/repository/workflow/
  source/platform/stage and report a child-of-child observed, delayed write absent,
  protected read/write/tool/replace/hardlink attacks denied, control environment
  scrubbed, zero active descendants, two zero-UID-process checks, and ephemeral
  account removal. Inner hashes reconcile and logs show successful uploads.

  Windows jobs 97364106435, 97364106550, and 97364106303 each passed the production
  containment step. Their scripts then enumerated exactly four files and recomputed
  result/stdout/stderr hashes without throwing. The pinned upload action immediately
  calculated the correct common output root but reported all four explicit files as
  absent. The identical boundary on all cells, after successful medium-integrity
  PowerShell reads, is a packaging/discovery problem involving the security-labelled
  output tree, not a demonstrated Job Object/low-integrity failure. No Windows artifact
  was retained, so the containment claim is still unreviewable end to end.

  Aggregate job 97364218625 correctly failed at `Require every containment cell` and
  skipped checkout/download/validation/upload. The run created no production tag,
  authorization, campaign execution, custody access, holdout, commitment, or reveal.
  The architecture remains viable; a bounded export layer can fix the evidence gap.
findings:
  - id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:101-204"
    evidence: |-
      Each Windows proof step succeeded and its source requires an existing output
      directory, exact sorted four-name set, readable JSON, and matching SHA-256 fields.
      Upload logs then say the common root is the same via000-proof-output directory
      but `No files were found` for all four explicit paths. No Windows artifacts are
      listed by the run API; only the three Ubuntu artifacts exist. Thus containment
      execution passed its retained preconditions but evidence transport failed.
    finding: "Windows proof subjects remain trapped in a security-labelled tree that the artifact action cannot discover, leaving only logs rather than independently downloadable evidence."
    failure_scenario: "The same discovery behavior occurs in the production proof rerun; aggregate fails closed forever or a future workaround weakens ACL/integrity containment by uploading directly from the protected tree."
    consequence: "The required retained 2x3 containment evidence is incomplete, so RR4/RR6 and merge readiness cannot be approved."
    required_action: |-
      After containment teardown and exact four-file/hash validation, create a fresh
      ordinary medium-integrity export directory with explicit inherited medium-user
      DACL. Copy bytes without following links or preserving labels/ACLs/ADS, verify
      regular non-reparse single-link files, then create one deterministic archive
      with exactly four canonical member names, fixed timestamps/modes/order, plus a
      canonical external manifest containing cell/run/source/member size+SHA-256 and
      archive SHA-256. Upload only archive and manifest. Aggregate must reject absolute
      or traversal names, duplicate/case-fold collisions, links/devices/ADS, extra or
      missing members, noncanonical metadata/order, decompression limits, manifest/
      archive mismatch, and cross-cell identity, then extract to a fresh directory and
      revalidate every inner hash. A simpler safe equivalent is a canonical JSON
      envelope containing four base64 byte strings and their hashes, plus envelope
      SHA-256; this avoids archive traversal/link/metadata ambiguity entirely.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001"
    description: "Rerun all six cells with deterministic medium-integrity export; retain three Windows artifacts; aggregate exact 2x3 successfully. Mutate traversal, duplicate/case collision, link/reparse/hardlink, ADS, extra/missing member, metadata, archive/inner hash, DACL/integrity label, zip-bomb, and cross-cell identity inputs and require fail-closed zero aggregate output."
    rationale: "Containment success is not independently reviewable until its exact subjects cross the platform artifact boundary without weakening containment."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Unchanged in final ancestry.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Unchanged in final ancestry.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Unchanged in final ancestry.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Unchanged in final ancestry.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Unchanged in final ancestry.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Unchanged in final ancestry.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Hosted Windows containment ran successfully but its evidence was not retained.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Blocked by RR8 export finding."}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Six-job architecture remains intact.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Ubuntu retained proof is positive; Windows evidence was not retained.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Blocked by RR8 export finding."}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: verified-resolved, evidence: "Separate non-scientific push workflow executed without campaign lifecycle operations.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Evidence retention now has the narrower RR8 blocker."}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Windows artifact absent.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Only Ubuntu retained proof is independently downloadable.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Safe hosted path exists and ran, but exact 2x3 artifact set is incomplete.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: "Superseded in scope by RR8 export test."}
predictions: {experiment_id: "", predicted_outcome: "", predicted_failure_mode: "", confidence_statement: "No scientific campaign execution was performed."}
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Fixable CHANGES REQUESTED. Preserve the containment architecture, add a safe deterministic Windows export envelope/archive, rerun the non-scientific 2x3 gate, and independently review all six retained artifacts."
```

## Verification ledger

- Exact handoff/tree, sole parent/tree, underlying proof-code/tree, RR7 response SHA-256, and prior artifact SHA-256 matched.
- Run 32705009673 and all seven job logs inspected; artifacts 9511903681, 9511903097, 9511902419 downloaded and checked as exact four-file Ubuntu cells.
- Windows jobs 97364106435/97364106550/97364106303: containment/pre-upload validation success; identical upload discovery failure. Aggregate 97364218625 failed closed.
- No scientific workflow, tag, signer, custody, lifecycle, holdout, assembly, commitment, or reveal operation was performed.
