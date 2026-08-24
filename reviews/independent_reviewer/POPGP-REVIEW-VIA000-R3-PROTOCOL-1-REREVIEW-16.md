# VIA-000 R3 recovery-protocol independent re-review 16

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-16"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-16"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "f3c98b283473bad0c8801bda1bf1ffbaf35000ea"
baseline_commit: "1c296f3b19c81f44ff718de6046346614578be0d"
prior_review_ref: "1c296f3b19c81f44ff718de6046346614578be0d:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-15.md"
builder_response_ref: "f3c98b283473bad0c8801bda1bf1ffbaf35000ea:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-15-RESPONSE-1.md"
context_hash: "03a44b2afe3151a43a441bf9c66a0fde39c13f74"
context_hash_method: 'git rev-parse "f3c98b283473bad0c8801bda1bf1ffbaf35000ea^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-15-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-15.md"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository, GitHub run 32745872694 logs/job/cache/artifact metadata, and public Microsoft Windows security documentation; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-16. Exact handoff/content ancestry,
  origin, trees, prior review/response bytes, workflow, all hosted job logs, affected
  source/receipt scripts, frozen predicate behavior, parser closure, and validators were
  independently inspected. No campaign implementation or external state was changed.
  Operator/orchestrator are shared; session/worktree/branch differ. Builder model is
  shared and external scientific validation is not claimed.
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
  CHANGES REQUESTED with two high-severity, bounded implementation blockers. Neither
  failure shows an architectural containment defect. The exact six-cell retained proof
  is still absent.

  Run 32745872694 is exact head f3c98b283473bad0c8801bda1bf1ffbaf35000ea,
  push event, attempt 1. Ubuntu candidate 97491127445, mutation 97491127454,
  and pdf 97491127500 completed production containment, then failed line 364 with
  `production containment result has invalid token or protected-label evidence`.
  Windows pdf 97491127153, candidate 97491127534, and mutation 97491127559
  passed production containment, hostile-marker/hash checks, token/protected-label
  validation, and in-memory four-subject validation, then failed line 508 with
  `Windows proof envelope export root is not an inherited medium-user DACL`.
  Aggregate 97491347537 observed six failures and rejected before checkout/restore;
  verifier 97491403910 was skipped. No digest, cache save, artifact, commitment, or
  lifecycle step occurred.

  Ubuntu's actual containment record is deterministic and canonical:
  token_restriction_flags `[]`, token_integrity_sid `""`,
  enabled_privilege_count `0`, enabled_privileges `[]`, and
  protected_label_policy `owner-only-protected-root`. Intended expected values are
  identical. But `$expectedTokenFlags = if (...) { ... } else { @() }` emits no objects
  on the Ubuntu branch, so assignment collapses the expected array to `$null`.
  `$observedTokenFlags = @(...)` remains a non-null empty Object[]. The RR15 ordinal
  predicate correctly rejects null. Frozen-AST replay proves the array match alone is
  false while every other Ubuntu comparison is true. The same caller bug is present in
  the production runner and both receipt copies. The predicate itself is correct.

  Windows necessarily observed/expected exact flags `[DISABLE_MAX_PRIVILEGE]`, SID
  `S-1-16-4096`, and label `medium-integrity-no-write-up-no-read-up`; otherwise it could
  not reach line 508. Its enabled privilege evidence was one of the two exact admitted
  states: count 0/list `[]`, or count 1/list `[SeChangeNotifyPrivilege]`. The failed jobs
  neither logged nor retained which admitted state, so a more precise claim would be
  invented.

  The Windows output root is under `GITHUB_WORKSPACE` at
  `.via000-r3-proof-cache/windows-x86_64/<stage>`, not under `RUNNER_TEMP`. Trusted
  PowerShell creates it only after whole-tree teardown and protected subject/hash
  checks, then successfully calls SetNamedSecurityInfo for a medium-integrity
  no-write-up label. The combined check reveals only that at least one of inheritance
  protection, explicit-rule count, or parent/child owner equality differed; no ACL facts
  were logged or retained. It therefore does not prove the directory was unsafe. It
  proves that relying on the hosted workspace's inherited descriptor is nondeterministic
  and that the check tests ancestry shape rather than the required low-IL write denial.

  Windows Mandatory Integrity Control evaluates before DACL access and denies a
  low-integrity principal writes to a medium-integrity object even when its DACL grants
  that principal write. A fresh explicit current-runner-SID DACL plus re-queried medium
  no-write-up label, created by trusted PowerShell after teardown, is a bounded and
  stronger closure. Do not move export into a low-integrity/mutable root.
findings:
  - id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:345-364; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1:774-793; corresponding receipt copies"
    evidence: |-
      The frozen proof-runner function rejects null, as RR15 required. Direct replay of
      its exact AST with the deterministic Ubuntu record produced:
      observed flags non-null/count 0; expected flags null; token-array match false;
      observed/expected integrity both empty and equal; privilege count/list both empty
      and canonical; observed/expected label both owner-only-protected-root and equal.
      PowerShell conditional output unrolls collections, so the `@()` branch yields no
      assignment value. Wrapping the whole conditional in `@(...)`, or initializing
      `$expectedTokenFlags = @()` before an explicit Windows-only assignment, preserves
      the required non-null empty array.

      All three Ubuntu jobs failed at the same comparison only after the preceding
      primitive, privilege-separation, exit, timeout, descendant-quiescence, and
      zero-active checks passed. Exact proof-runner SHA-256 is
      5b40e8cc9084a69c1e0b14ffc06339b61ff6885d8679ef05142422eeb1ba14f7;
      the receipt copy matches. Production runner SHA-256 is
      693b6d40ceeb7d04636f22f9b249090e098e74c552592a5bb6a39d9ec7f3ec5b;
      its receipt copy matches.
    finding: "The exact ordinal array predicate is correct, but both callers construct Ubuntu's canonical expected empty array as null."
    failure_scenario: "Every Ubuntu proof or production stage supplies valid empty token evidence and is falsely rejected before proof/export."
    consequence: "Ubuntu remains 0/3 and a complete 2x3 retained proof cannot exist."
    required_action: |-
      In both production callers and both receipt copies, construct the expected flags
      as an actual non-null Object[]: initialize `@()` and assign the one Windows string
      only in the Windows branch, or wrap the entire conditional output in `@(...)`.
      Do not weaken Test-ExactOrdinalStringArray or admit null. Add a caller-level test
      that passes the exact ConvertFrom-Json Ubuntu record through the complete
      comparison, not only direct predicate literals. Require `[]` to pass and null,
      scalar empty string, `[""]`, wrong type/count/order/case to fail. Exercise both
      proof and production callers and exact receipts on PowerShell 7 Windows/Ubuntu.
    verification: confirmed-by-execution
    blocking: true
  - id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:492-515; .github/workflows/via000-r3-containment-proof.yml:399-438 and corresponding Windows cells"
    evidence: |-
      Each Windows job reached only the combined condition after its low-IL hostile
      process and descendants were terminated, zero-active was proven, protected
      evidence/tool hashes were rechecked, four subject bytes and inner hashes were
      validated in trusted memory, the workspace-relative cache hierarchy was freshly
      created, and Set-Via000RootIntegrity successfully applied the medium no-write-up
      label. Failure cleanup removed the cache hierarchy; no envelope/digest/cache or
      artifact survived.

      The condition requires AreAccessRulesProtected false, zero explicit DACL rules,
      and output owner text equal to cachePlatform owner text. Logs retain only the
      combined exception, not these three values or normalized ACE/SID/mandatory-label
      facts. Thus the exact hosted ACL shape and failed subpredicate are unknowable from
      this run. `RUNNER_TEMP` was D:\a\_temp, but OutputRoot was under
      D:\a\POPGP\POPGP; attributing this error to RUNNER_TEMP would be factually wrong.

      Exact workflow SHA-256 is
      028a957f03aeb76822559f271e79a0e6c428701170634256723deeec88458cee;
      containment helper SHA-256 is
      0e6d63c08d148f1589b129820191df6b987bdf45a67a4ad08ce2ea184847b051.
      Microsoft documents that MIC precedes DACL evaluation and denies low-to-medium
      writes. Inherited-versus-explicit ACE provenance is not that security property.
    finding: "Windows export depends on an unrecorded hosted-workspace inherited DACL shape instead of creating and verifying a deterministic medium-integrity descriptor."
    failure_scenario: "A safely medium-labeled, trusted-post-teardown directory has a hosted-platform owner or ACE provenance different from the assumed shape and is rejected; alternatively, a shape-matching descriptor could pass without proving its mandatory label was re-read or low-IL writes are denied."
    consequence: "Windows remains 0/3 and no envelope reaches cache transport, although the containment boundary itself completed successfully."
    required_action: |-
      Keep export in a fresh ordinary cell-specific root created by trusted built-in
      PowerShell after containment teardown and all subject checks. Do not trust parent
      inheritance. On Windows, capture the current trusted runner user SID, create a
      protected non-inheriting DACL with one reviewed allow rule granting that SID the
      rights needed by the next trusted digest/cache steps, set owner to that SID, and
      apply an exact medium-integrity `NO_WRITE_UP` mandatory label. Apply/verify the
      corresponding inherited file descriptor on `envelope.json`. Do not grant a low
      integrity SID, broad mutable group, Everyone, or Authenticated Users write access.

      Re-read with SID-based native/.NET APIs and fail unless root/file are non-reparse,
      owner SID is exact, DACL protection and normalized explicit/inherited ACE sets are
      exact, mandatory-label SID is S-1-16-8192 with NO_WRITE_UP, and path/parent/cell
      identities remain exact. Recheck the same normalized facts in the following
      trusted digest step immediately before reading and again before cache save. Record
      bounded owner-SID hash or exact public runner SID, DACL policy identifier, medium
      label/policy, creation-after-teardown, and subject hash in proof evidence; never
      log envelope content.

      Add a live Windows negative regression in which the production low-IL token tries
      create/write/rename/delete/reparse/replace-restore against an identically secured
      export root and all attempts fail while a trusted medium process can write/read.
      Also mutate owner, inheritance protection, each ACE SID/type/rights/flags, medium
      label, policy, reparse state, and post-check replacement; every mutation must
      reject and clean all output. A small non-scientific hosted ACL diagnostic may
      first log normalized facts, but final acceptance still requires all three exact
      Windows cells, cache transport, retained aggregate, and verifier.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001"
    description: "Exercise the complete proof and production caller comparison with canonical deserialized Ubuntu empty arrays; require non-null empty expected/observed arrays to pass and null/scalar/empty-string/type/count/order/case mutations to fail on both PowerShell platforms and receipts."
    rationale: "Direct predicate tests did not cover PowerShell conditional-output collapse at the caller."
    blocking: true
  - id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001"
    description: "Create a fresh post-teardown Windows export root with explicit current-runner-SID DACL and re-queried medium NO_WRITE_UP label; prove trusted access, low-IL create/write/rename/delete/reparse denial, exact root/file descriptor evidence, cross-step rechecks, mutation rejection, cleanup, and final retained 2x3 transport/verifier success."
    rationale: "The required boundary is deterministic low-IL write denial, not conformity to a hosted parent's inherited ACL provenance."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved Git-normalized behavior; hosted export is distinct.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Fixed proof dispatch reaches and executes parsed runners.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "No retained exact hosted proof.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", notes: "Ubuntu caller also blocks."}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Containment runs, but no cell exports evidence.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", notes: "Ubuntu caller also blocks."}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "No retained hosted subject exists.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "All six proof cells fail before transport.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", notes: "Ubuntu caller also blocks."}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Windows export root is rejected before envelope creation.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No envelope reaches transport.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "No digest or cache save executes.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Built-in control plane and trusted runner execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "No no-op recurrence.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "Windows hostile containment now executes through teardown.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: unresolved, evidence: "No-LUA containment succeeds in all Windows cells, but required retained hosted proof is blocked at export.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "All eight standalone scripts parse; hosted jobs progressed beyond the former parser line on both platforms.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Parsed fixed runners execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "No retained hosted proof.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "No retained six-cell context.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "No retained hosted subject.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "All six fail before transport.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", notes: "Ubuntu caller also blocks."}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Windows envelope is not written.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No transport.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "No cache save.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: unresolved, evidence: "Built-in execution works; retained proof remains absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved diagnostic result.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: unresolved, evidence: "Windows containment passes; retained hosted acceptance remains absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Eight files and workflow blocks parse; focused regression passed 1/1 and hosted jobs passed parsing.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
predictions: {experiment_id: "", predicted_outcome: "", predicted_failure_mode: "", confidence_statement: "No scientific execution occurred."}
recommendation: {approve: false, blocking_findings: 2, rationale: "Bounded CHANGES REQUESTED. Preserve the exact predicate but construct Ubuntu empty expectations as arrays; replace Windows inherited-DACL assumptions with a verified explicit current-runner-SID DACL plus medium NO_WRITE_UP label. Then rerun and retain the exact 2x3 proof. Architecture remains viable."}
```

## Verification ledger

- Exact handoff `f3c98b283473bad0c8801bda1bf1ffbaf35000ea`, tree `03a44b2afe3151a43a441bf9c66a0fde39c13f74`, sole parent/content `cb38a02ed7da525cd1305326f81abe72362903eb`, tree `ea1ef5cc98a4944cacc85d4d29a19e3f09e41dc6`, and origin campaign head matched. RR15 review Git-normalized SHA-256 was `55fae7b2c27fa5af17ecc31b2ad04426c45ceb52bf081a1c9fdebee319ac9b12`; response SHA-256 was `e572c41b1e2681b0ebdbed5079a281af0df01ed44b002c6a2b42fb449823fb6e`.
- Run `32745872694` exact head/event/attempt, all eight jobs, every step, and all producer logs were inspected. Ubuntu failed 3/3 at the caller comparison; Windows failed 3/3 at export ACL validation; aggregate failed closed; verifier skipped; artifacts and matching run-time caches were zero.
- Frozen-AST Ubuntu replay isolated expected-array null collapse with all other exact values equal. All eight R3 protocol/receipt PowerShell files independently parsed with zero errors. Focused RR15 parser/array test: `1 passed in 7.06s`. Campaign validator: valid.
- Hosted Windows logs do not contain normalized ACL facts or the optional privilege-list value. This review states the maximum supported inference and requires those facts in the next run. Microsoft references reviewed: Mandatory Integrity Control, SYSTEM_MANDATORY_LABEL_ACE, and PowerShell Set-Acl/SetAccessRuleProtection.
- R3 remains drafted, `holdout_started: false`, unrevealed, pending, and has no R3 tag. No signer, key, custody, lifecycle, holdout, scientific execution, commitment, or reveal was accessed, performed, or authorized.

Primary references: [Mandatory Integrity Control](https://learn.microsoft.com/en-us/windows/win32/secauthz/mandatory-integrity-control), [SYSTEM_MANDATORY_LABEL_ACE](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_mandatory_label_ace), and [Set-Acl](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/set-acl?view=powershell-7.5).
