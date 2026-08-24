# VIA-000 R3 recovery-protocol independent re-review 13

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-13"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-13"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "a24fa811679c9bbec9357ca58fde3bb2b320ce60"
baseline_commit: "80e6005542033843628639cce1a3a30fc05e723d"
prior_review_ref: "80e6005542033843628639cce1a3a30fc05e723d:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-12.md"
builder_response_ref: "a24fa811679c9bbec9357ca58fde3bb2b320ce60:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-12-RESPONSE-1.md"
context_hash: "2d8bb4c468a134ab16fe081b7b5d0c8d95d4c7ed"
context_hash_method: 'git rev-parse "a24fa811679c9bbec9357ca58fde3bb2b320ce60^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - ".github/workflows/via000-r3-protocol.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-HOSTILE.ps1"
  - "tests/unit/test_via000_r3_identity.py"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-12-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-12.md"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and GitHub run 32728995796 logs/job/cache/artifact metadata plus public Microsoft Win32 API documentation; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-13. Exact content/handoff/tree/parent/
  origin identities, RR12 response/review bytes, all eight hosted job records and
  logs, cache/artifact metadata, the complete Windows native primitive, proof caller,
  focused local Windows containment test, and primary Win32 API contracts were
  independently inspected. No implementation or external state was changed.
  Operator/orchestrator are shared; session/worktree/branch differ. Builder model is
  shared and external validation is not claimed.
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
  CHANGES REQUESTED with one high-severity Windows child-initialization blocker. The
  security architecture appears recoverable without weakening untrusted write
  separation, but the retained 2x3 proof remains absent.

  Run 32728995796 is an exact-handoff, non-scientific push and is the first valid
  hosted Windows execution of this proof boundary. Windows jobs 97436685825,
  97436685838, and 97436685698 ran through GitHub built-in pwsh at
  `C:\Program Files\PowerShell\7\pwsh.EXE`; each passed the canonical process,
  PSHOME, PS7, no-profile, and sanitized-PATH entry checks. Each entered the frozen
  proof runner and NativeContainment primitive, then its contained PowerShell child
  exited `-1073741502`, unsigned `3221225794`, hexadecimal `0xC0000142` /
  `STATUS_DLL_INIT_FAILED`. The outer same-step check converted this into failure and
  all later Windows digest/cache steps were skipped. This is fail-closed and is not
  the RR12 no-op shell defect.

  The code path proves more than the generic status alone: CreateRestrictedToken,
  low-integrity SID assignment, CreateProcessAsUser suspended creation, Job Object
  assignment before resume, resume, wait/exit capture, explicit tree termination, and
  the zero-active-process query all completed without throwing. Invoke then accepted
  quiescence, rechecked the trusted closure, copied bounded output, wrote its result,
  cleaned staging, and only afterward rejected the nonzero child exit at line 641.
  Therefore the observed failure is child initialization after process creation and
  before the intended PowerShell payload, not shell dispatch, Job assignment, or
  teardown failure. The status does not identify which DLL or why initialization
  failed.

  Ubuntu jobs 97436685860, 97436686022, and 97436685789 succeeded end to end, emitted
  three distinct digests, and retained three exact caches. Aggregate 97436858212
  rejected the three failed Windows dependencies before restore/output; verifier
  97436921959 was skipped and artifact count is zero. No campaign/lifecycle action ran.

  The most useful hypotheses are hosted PowerShell-specific initialization under the
  low-integrity restricted context, and the hand-built inherited environment versus a
  token-derived essential environment. The current local exact-source negative control
  successfully launched contained Python, killed its detached descendant tree, kept
  protected roots unchanged, and returned zero active processes. Thus the primitive
  is not universally unable to launch a low-integrity child. LUA_TOKEN interaction is
  plausible but unproven. Desktop/window-station access is a lower-priority conditional
  hypothesis; CreateProcessAsUser succeeded, CREATE_NO_WINDOW is used, and lpDesktop
  null inherits the parent boundary. CreateProcessWithTokenW is not yet justified.
findings:
  - id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001"
    severity: high
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1:72-103,106-313,451-641; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:216-344"
    evidence: |-
      All three Windows jobs executed the built-in pwsh body and reached the frozen
      NativeContainment error at line 641 with identical child exit -1073741502.
      `0xC0000142` is the generic DLL-initialization-failed termination status.
      CreateProcessAsUser documentation explicitly states that the call returns before
      process initialization completes and a required DLL that cannot be located or
      initialized terminates the child; this matches successful creation followed by
      the observed exit, but does not identify a DLL.

      NativeContainment currently creates a primary restricted token with
      DISABLE_MAX_PRIVILEGE | LUA_TOKEN, changes its mandatory label to low integrity
      SID S-1-16-4096, manually serializes a filtered copy of the parent environment,
      leaves STARTUPINFO.lpDesktop null, creates the child suspended with
      CreateProcessAsUser and inherited standard handles, assigns a kill-on-close Job
      Object, then resumes. Microsoft documents CreateEnvironmentBlock as the standard
      token-specific environment source and documents that a null lpDesktop inherits
      the parent's window station/desktop. No log evidence currently distinguishes
      environment, token flag, executable/runtime, or desktop initialization.

      Exact local command
      `uv run --frozen --no-editable python -m pytest -q tests/unit/test_via000_r3_identity.py::test_r3_windows_production_containment_kills_detached_replace_restore_tree`
      passed in 37.09 seconds. It uses the same primitive to launch isolated Python,
      proves direct/detached replacement attempts cannot alter protected state, and
      proves zero active descendants. This narrows but does not resolve the hosted
      PowerShell failure.
    finding: "The valid hosted Windows boundary cannot initialize its contained PowerShell child under the current low-integrity restricted-token environment, so no Windows proof envelope can be produced."
    failure_scenario: "CreateProcessAsUser returns a suspended child and the parent atomically contains/resumes it, but the child terminates with STATUS_DLL_INIT_FAILED before executing the fixed payload; every Windows proof cell fails and aggregation has only Ubuntu evidence."
    consequence: "The mandatory retained Windows candidate/pdf/mutation evidence, exact 2x3 aggregate, verifier, and production Windows viability proof remain unavailable."
    required_action: |-
      Run one disposable, Windows-only, non-authoritative diagnostic workflow at the
      exact source. Use independent explicit jobs (not a matrix output), synthetic fixed
      commands only, no secrets, no repository writes, and no scientific/lifecycle
      path. Preserve in every cell: the low-integrity SID S-1-16-4096, protected-root
      no-write-up/no-read-up separation, a fresh mutable low-integrity root, exact
      executable paths/hashes, CREATE_SUSPENDED, atomic Job Object assignment before
      resume, no breakaway, kill-on-close plus explicit termination, zero-active proof,
      fixed bounded standard handles, and post-run closure/absence checks. Record only
      API phase, unsigned/hex exit, sentinel presence, stdout/stderr hashes, token flags,
      environment-construction ID, executable ID, and teardown facts; do not log content.

      Use a common minimal-pwsh reference and vary exactly one factor per paired cell:

      1. `exact-proof-baseline`: the current code and exact current synthetic hostile
         proof invocation. This must reproduce 0xC0000142 and proves experiment fidelity.
      2. `minimal-pwsh-baseline`: same current token, manual environment, native API,
         flags, null desktop, and Job Object, changing only the PowerShell arguments to
         a fixed `-NoLogo -NoProfile -NonInteractive -Command` that writes one known
         sentinel under the mutable root and exits zero.
      3. `minimal-cmd-baseline`: identical to cell 2 except the executable/arguments are
         canonical `C:\Windows\System32\cmd.exe /d /s /c` writing the same fixed
         sentinel. A cmd success with pwsh failure localizes the issue to PowerShell/
         .NET initialization; both failing favors shared token/environment/desktop state.
      4. `token-environment-pwsh`: identical to cell 2 except environment construction.
         Call userenv!CreateEnvironmentBlock on the restricted primary token with
         inheritance false, parse it strictly, select a fixed reviewed essential Windows
         allowlist, reject duplicates/control bytes/forbidden names, and override PATH,
         PATHEXT, TEMP/TMP/HOME/USERPROFILE/APPDATA/LOCALAPPDATA with canonical trusted
         or fresh low-integrity mutable values. Destroy the native block correctly.
         No GITHUB/ACTIONS/RUNNER, package, Git, Python, TeX, credential, or secret
         variable may cross. Keep CREATE_UNICODE_ENVIRONMENT. This cell varies only the
         environment source/normalization.
      5. `no-lua-low-il-pwsh`: identical to cell 2 except CreateRestrictedToken uses
         DISABLE_MAX_PRIVILEGE without LUA_TOKEN; still set low integrity explicitly,
         verify the resulting token integrity/privilege state before creation, and keep
         every mandatory-label protected root. This isolates LUA filtering while
         retaining no-write-up separation and disabled privileges.

      Each job must include same-step sentinels around native phases so a no-op cannot
      pass, plus a separate trusted consumer recheck. Require baseline reproduction and
      interpret only paired differences. Do not add a medium-integrity untrusted cell.

      Add an lpDesktop experiment only if both minimal cmd and pwsh still fail and the
      environment/no-LUA cells do not isolate the cause. It must vary only lpDesktop,
      first audit the inherited/target window-station and desktop access read-only, and
      must not relax their DACLs. Microsoft documents desktop security as a possible
      CreateProcessAsUser initialization failure, but current evidence does not make it
      the first factor. Consider CreateProcessWithTokenW only after the desktop cell and
      only as an exact same-token/environment/flags comparison with no profile load or
      new credentials; its session/desktop behavior differs, so it is not a current fix.

      After a factor restores both minimal pwsh and the exact synthetic hostile proof,
      integrate only that factor, replay the local live hostile test, and rerun the
      complete exact six-cell hosted proof with six cache restores, retained seven-file
      artifact, and green verifier. Independently review those bytes before production.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001"
    description: "Five-cell explicit Windows diagnostic: exact proof reproduction; minimal pwsh; executable-only cmd comparison; CreateEnvironmentBlock-derived scrubbed essential environment comparison; no-LUA but low-integrity comparison. Every cell preserves atomic Job assignment/teardown and protected-root separation, records phase/exit/hash facts only, and has same-step plus cross-step sentinel checks. Desktop and CreateProcessWithTokenW are conditional follow-ups only."
    rationale: "STATUS_DLL_INIT_FAILED is nonspecific. Paired one-factor cells can identify whether the hosted failure is PowerShell-specific, environment-derived, or token-filter-derived without weakening low-integrity write separation or touching scientific state."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Git-normalized source behavior preserved; hosted native child initialization is a distinct scope.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Built-in pwsh executed the fixed proof body; input grammar remains frozen.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "The hosted Windows child cannot initialize, so production tool identity remains unproved there.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Closure was rechecked, but the hosted Windows child did not execute its payload.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Job teardown succeeded, but no successful hosted Windows payload/subject exists.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Safe hosted path executes but Windows 0/3 prevents retained 2x3 evidence.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "No successful Windows envelope reaches export.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "Ubuntu canonical transport passed; Windows produces no envelope.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Ubuntu digest/cache path passed; Windows producer fails before transport.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Control-plane failure was isolated and the campaign now runs built-in pwsh.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "All three Windows jobs demonstrably executed the built-in pwsh body and failed inside NativeContainment rather than no-oping.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Full retained replay remains blocked by RR13."}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Built-in pwsh executed and rejected on native child failure.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Hosted Windows payload did not initialize.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Hosted Windows child did not execute.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "No successful hosted Windows subject exists.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Windows 0/3; retained artifact absent.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "No Windows envelope exists.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "Windows producer fails before transport.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Only Ubuntu 3/3 caches exist.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: unresolved, evidence: "Built-in Windows execution is satisfied, but the requested six-cell retained proof fails at child initialization.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", notes: ""}
predictions: {experiment_id: "", predicted_outcome: "", predicted_failure_mode: "", confidence_statement: "No scientific execution occurred."}
recommendation: {approve: false, blocking_findings: 1, rationale: "Bounded CHANGES REQUESTED. Run the five-cell Windows native-child factor diagnostic, apply only the isolated low-integrity-compatible correction, replay the local hostile test, and rerun/retain/revalidate the exact six-cell proof. Security architecture remains viable."}
```

## Verification ledger

- Exact content `b114b04c9b445fe7da9b7e6ef6e635736ba11eda`, tree `397601d8b68d9cb48dc7d4c1c013f1f022b89219`, handoff `a24fa811679c9bbec9357ca58fde3bb2b320ce60`, tree `2d8bb4c468a134ab16fe081b7b5d0c8d95d4c7ed`, sole-parent relationship, and origin campaign head matched. RR12 review Git-normalized SHA-256 was `22a1d902c7732e105398bf29e359f13e5ed9052e80a6a512f5be3b7178d7ef3b`; response SHA-256 was `e94bf0d9e46caa27da9dd4788ad6fad44a85bafc9f7f9746205f0c1a18c5a0cc`.
- Run `32728995796` exact head/event/attempt and every job/step inspected. Windows candidate/pdf/mutation failed identically inside the actual native child; Ubuntu candidate/pdf/mutation passed and caches `6934887831`, `6934886674`, and `6934884933` exist. Aggregate failed closed; verifier skipped; artifact count zero.
- Decimal/unsigned/hex status equality checked: `-1073741502` = `3221225794` = `0xC0000142`. The status name/meaning and CreateProcessAsUser initialization timing were checked against Microsoft Win32 documentation. CreateRestrictedToken flags, CreateEnvironmentBlock contract, lpDesktop inheritance, and CreateProcessWithTokenW differences were reviewed from the same primary documentation.
- Focused exact-source local Windows hostile containment test: `1 passed in 37.09s`. This is supporting diagnostic evidence only and does not override hosted failure.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending. No signer, key, tag, refreeze, custody, lifecycle, holdout, scientific execution, commitment, or reveal was performed or authorized.

Primary API references: [CreateProcessAsUser](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessasusera), [CreateEnvironmentBlock](https://learn.microsoft.com/en-us/windows/win32/api/userenv/nf-userenv-createenvironmentblock), [CreateRestrictedToken](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-createrestrictedtoken), and [CreateProcessWithTokenW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-createprocesswithtokenw).
