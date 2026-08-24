# VIA-000 R3 recovery-protocol independent re-review 14

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-14"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-14"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "ebf31ce25049d6a8f5a5641fc36cbdd6e25dc068"
baseline_commit: "f018ef40dbc03f3373b05585cd61314748d47abb"
prior_review_ref: "f018ef40dbc03f3373b05585cd61314748d47abb:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-13.md"
builder_response_ref: "ebf31ce25049d6a8f5a5641fc36cbdd6e25dc068:protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RR13-CHILD-FACTOR-RUNNER.ps1"
context_hash: "76c0399e79738852f82ca410a2111ee1945f7a45"
context_hash_method: 'git rev-parse "ebf31ce25049d6a8f5a5641fc36cbdd6e25dc068^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-windows-transport-diagnostic.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RR13-CHILD-FACTOR-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RR13-NATIVE-DIAGNOSTIC.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-HOSTILE.ps1"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-13.md"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and GitHub run 32732141667 logs/job/cache/artifact metadata plus public Microsoft Win32 API documentation; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-14. Exact campaign, experiment,
  RR13 review, origin, parent, tree, file-hash, hosted-run, job, cache, and artifact
  identities were independently checked. The native diagnostic, production primitive,
  factor runner, workflow, five job logs, and both trusted rechecks were inspected. No
  campaign or experiment implementation and no external state was changed.
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
  CHANGES REQUESTED with one high-severity, bounded production-integration blocker.
  The RR13 experiment resolves the diagnosis and supports removing only LUA_TOKEN from
  the Windows production restricted-token flags. It does not approve the unchanged
  campaign or replace the mandatory exact six-cell retained proof.

  Run 32732141667 is a successful non-authoritative push at exact experiment commit
  ebf31ce25049d6a8f5a5641fc36cbdd6e25dc068 and attempt 1. All five explicit
  windows-2025 jobs executed GitHub's built-in pwsh control body and passed an
  independent next-step summary/hash recheck. A, B, and C retained the production
  DISABLE_MAX_PRIVILEGE|LUA_TOKEN flags and independently terminated PowerShell,
  minimal PowerShell, and minimal cmd with -1073741502 / 3221225794 / 0xC0000142,
  empty stdout/stderr, and no sentinel. D changed the environment source to a scrubbed
  CreateEnvironmentBlock allowlist while retaining those flags and failed identically.
  Environment construction and command complexity are therefore falsified causes.

  E restored the same manual filtered environment and minimal PowerShell command used
  by B, but passed DISABLE_MAX_PRIVILEGE without LUA_TOKEN at the CreateRestrictedToken
  call. It exited zero and wrote the exact `rr13` sentinel. Before child creation, the
  diagnostic queried the token and required exact integrity SID S-1-16-4096; it also
  rejected every enabled privilege except an optional single SeChangeNotifyPrivilege.
  The trusted recheck required both facts. The native sequence otherwise matches the
  production sequence: CreateProcessAsUser suspended, atomic Job Object assignment,
  resume, wait, explicit tree termination, and zero-active-process proof. The
  diagnostic additions are phase recording and read-only token queries, not alternate
  child creation. Thus the operative child-boundary factor is LUA_TOKEN.

  Removing LUA_TOKEN does not remove the no-write-up boundary. DISABLE_MAX_PRIVILEGE
  remains, the token is explicitly relabeled low, and protected roots remain medium
  with mandatory no-write-up/no-read-up labels. Microsoft documents mandatory
  integrity enforcement before DACL evaluation and specifically denies low-integrity
  writes to medium-integrity objects. E rechecked both fixed protected subjects
  unchanged and ended with zero active descendants. However, E used only the minimal
  sentinel; the production patch must still pass the exact hostile replace/restore and
  descendant test before the security claim is accepted.

  The five jobs created no artifact or matching cache. The workflow had contents:read,
  no secrets, no campaign/lifecycle path, and no scientific or custody checkout. No
  tag, signer, refreeze, activation, holdout, commitment, or reveal occurred.
findings:
  - id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1:131-133,274-300; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RR13-NATIVE-DIAGNOSTIC.ps1:218-258,300-343; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RR13-CHILD-FACTOR-RUNNER.ps1:1-271"
    evidence: |-
      The exact experiment source hashes are: workflow
      cdcc9c0455b4874cebc571b2dd7db658457d4f5e0f545d2bddb9c3ec23e45fd9,
      factor runner
      6b67b410a40c02d6e43d4f11c9d5fdad2153f0b38ab712e2af8b5685ffd5bc2d,
      native diagnostic
      c5c11dcfee1268a6a10ad5563d18e10713b65565e2fa0a78ac86e8439224f2a1,
      and reviewed production containment
      44908f2529876865a88ac0ae82c5da7d38089ccd0159def4e2dbf2f55f43500d.
      Every log bound source_sha and workflow_sha to ebf31ce25049d6a8f5a5641fc36cbdd6e25dc068.

      Jobs A 97446525811, B 97446525806, C 97446525523, and D 97446525956
      all produced 0xC0000142. Job E 97446525911 produced exit 0 and the expected
      sentinel. B and E share the same absolute PowerShell executable and SHA-256
      362a356ce7f0940ec74f73a8fc2c990a2cc24a38a11c90bbd8eca947110ad139,
      the same minimal arguments, manual environment construction, standard handles,
      working root, process flags, Job settings, and teardown. At the native API
      boundary E changes only CreateRestrictedToken flags from
      DISABLE_MAX_PRIVILEGE|LUA_TOKEN to DISABLE_MAX_PRIVILEGE; its added
      GetTokenInformation calls are read-only assertions. D independently shows that
      using a token-derived environment does not correct the failure.

      E's trusted summary and consumer require low_integrity_sid S-1-16-4096 and
      privileges_disabled true. Source inspection proves that the latter rejects every
      enabled privilege except at most one SeChangeNotifyPrivilege; the boolean does
      not claim whether the optional privilege was present. Fixed protected evidence
      and tool subjects have SHA-256
      ff29f1d7588caa999d35cd3d0d34ca470fd142c09df06f93f34c6850b74239f1
      and d1e80a1fc3fbdceecaa3521032c8031b3e06fa95db3e8d87dd5489f3227bb5ca;
      their before/after hashes matched. Both output hashes were the canonical empty
      hash e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855,
      and active_processes_after_teardown was zero.
    finding: "The hosted failure is caused by the production LUA_TOKEN combination; the campaign still contains that flag and therefore still cannot produce Windows containment evidence."
    failure_scenario: "If the campaign runs unchanged, each Windows contained process is created and assigned safely but terminates with STATUS_DLL_INIT_FAILED before its payload; aggregation remains unable to retain a complete 2x3 proof."
    consequence: "Campaign merge, signer/custodian gates, refreeze, activation, and holdout remain unauthorized even though the Windows containment architecture is recoverable."
    required_action: |-
      Make the smallest campaign change: in the production NativeContainment call,
      replace `DISABLE_MAX_PRIVILEGE | LUA_TOKEN` with `DISABLE_MAX_PRIVILEGE` only.
      Preserve explicit SetTokenInformation S-1-16-4096, the reviewed manual environment,
      CreateProcessAsUser flags and arguments, protected-root mandatory labels, suspended
      creation, Job assignment before resume, no-breakaway policy, explicit termination,
      zero-active check, closure rechecks, and all cleanup. Remove the unused LUA_TOKEN
      constant so a stale path cannot silently retain it.

      Port the RR13 fail-closed token queries into production: after relabeling and
      before process creation, require the queried TokenIntegrityLevel to equal
      S-1-16-4096, require no enabled privilege other than at most one
      SeChangeNotifyPrivilege, and bind token_flags, exact integrity SID, enabled
      privilege count/name list, protected-label policy, and teardown facts into every
      Windows containment result/envelope/manifest and its independent validator.
      Update the protocol, packet, receipt copies, design/README, schemas, hashes, and
      identity tests coherently; no other factor should change.

      Add source/static tests that reject LUA_TOKEN anywhere in the production or
      receipt primitive and reject drift between copies. Add parser/validator mutations
      for missing, forged, high-integrity, unexpected-privilege, mismatched token-flags,
      noncanonical privilege-list, nonzero-active, or changed-protected-root facts.
      Replay the local live hostile test using the exact patched primitive and require
      the attack payload actually starts, its direct and detached replace/restore/read/
      write attempts fail, protected hashes remain exact, the post-sign subjects match,
      and teardown proves zero active processes.

      Finally run the safe hosted proof from scratch at the exact reviewed source. All
      Windows candidate/pdf/mutation cells must execute the exact hostile payload under
      DISABLE_MAX_PRIVILEGE-only plus verified low IL; all Ubuntu cells must also pass.
      Require six distinct exact-key transports, aggregate exact 2x3 identity and inner
      hash validation, one retained seven-file artifact, and an independent verifier
      success. Reject missing/extra/duplicate/cross-cell evidence. Independently review
      those retained bytes before any signer or custodian amendment.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001"
    description: "Patch only the production LUA_TOKEN flag, port exact token-state assertions and evidence binding, replay the local hostile descendant/replace-restore test, and retain/revalidate the complete hosted six-cell artifact with all Windows cells proving DISABLE_MAX_PRIVILEGE-only, exact S-1-16-4096, the privilege whitelist, protected-root hashes, and zero active descendants."
    rationale: "The experiment isolates a compatible token construction, but the minimal E sentinel neither executes the hostile workload nor creates authoritative retained 2x3 evidence."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Git-normalized source behavior is preserved; hosted low-IL compatibility is the distinct current scope.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "The diagnostic ran through built-in pwsh with fixed inputs.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "A successful exact production Windows proof remains pending.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "The minimal E cell succeeds, but exact hostile production closure remains pending.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "No successful exact hosted Windows hostile subject is retained yet.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "The diagnostic is non-authoritative and has no retained 2x3 artifact.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "The successful E cell intentionally exported no envelope.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No canonical transport was in this experiment.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "No caches were created by this experiment.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Every cell and trusted consumer executed through built-in pwsh.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "No custom dot-source shell was used.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "The exact factor experiment isolates LUA_TOKEN: A-D fail identically and no-LUA E succeeds while low IL and privilege restrictions remain.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: "Diagnosis resolved; production integration remains blocked."}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Built-in pwsh executed fixed inputs.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Awaiting successful exact Windows hosted replay.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Awaiting exact hostile production replay.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Awaiting retained exact Windows hostile subject.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Non-authoritative diagnostic has no retained 2x3 artifact.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "No Windows proof envelope was exported.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No transport was exercised.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "No cache was exercised.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: unresolved, evidence: "Built-in execution is proved, but its requested authoritative six-cell retention still awaits the patch.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Five explicit cells ran; A-D reproduced the failure and E isolated a low-IL no-LUA success with complete teardown.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
predictions: {experiment_id: "", predicted_outcome: "", predicted_failure_mode: "", confidence_statement: "No scientific execution occurred."}
recommendation: {approve: false, blocking_findings: 1, rationale: "Bounded CHANGES REQUESTED. The architecture is viable. Apply only the no-LUA production correction and token-state evidence assertions, then pass the local exact hostile regression and retain/revalidate the complete hosted 2x3 proof before any signer/custodian or lifecycle step."}
```

## Verification ledger

- Exact RR13 review `f018ef40dbc03f3373b05585cd61314748d47abb`, tree `7af22db6210ddb849d1dd767f81cbc81bc5364c2`, blob `9f05c712870958c40cf5bbef3a989911f0009e3b`, and Git-normalized SHA-256 `354fe3e058020d4059eb46e4ba67ff7e89fc24745b1abef302d01831a13cd276` matched. Exact experiment commit `ebf31ce25049d6a8f5a5641fc36cbdd6e25dc068`, tree `76c0399e79738852f82ca410a2111ee1945f7a45`, sole parent `9994c6d77ae8a7a3ebdc84fbb132bf72564530f8`, and origin experiment head matched. Campaign handoff/origin remained `a24fa811679c9bbec9357ca58fde3bb2b320ce60`, tree `2d8bb4c468a134ab16fe081b7b5d0c8d95d4c7ed`.
- Run `32732141667` was exact head, push event, attempt 1, and completed success. Jobs A `97446525811`, B `97446525806`, C `97446525523`, D `97446525956`, and E `97446525911` plus every trusted recheck passed their declared expected result. A-D were exact negative controls; E was the positive no-LUA/low-IL control. Artifact count and matching cache count were zero.
- The experiment preserves low-integrity mandatory enforcement independent of LUA filtering. Microsoft documents that `DISABLE_MAX_PRIVILEGE` disables every privilege except `SeChangeNotifyPrivilege`, while `LUA_TOKEN` separately requests a LUA token. Microsoft also documents that Mandatory Integrity Control is evaluated before DACLs and prevents a low-integrity principal from writing a medium-integrity object. Source inspection confirms exact token-query fail-closed checks rather than relying on declarations.
- No local scientific or campaign execution was run. R3 remains drafted, holdout false, unrevealed, and pending. This review authorizes only the bounded campaign patch and its tests; it does not authorize merge, signer/key amendment, custodian verification, refreeze, tag, activation, holdout, commitment, or reveal.

Primary API references: [CreateRestrictedToken](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-createrestrictedtoken) and [Mandatory Integrity Control](https://learn.microsoft.com/en-us/windows/win32/secauthz/mandatory-integrity-control).
