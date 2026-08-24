# VIA-000 R3 recovery-protocol independent re-review 12

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-12"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-12"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "d2ce792a71002310f9ef760962bcb1141ec99437"
baseline_commit: "185defd140fb9a60fab00d05cf1aee7fb188bcea"
prior_review_ref: "185defd140fb9a60fab00d05cf1aee7fb188bcea:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-11.md"
builder_response_ref: "d2ce792a71002310f9ef760962bcb1141ec99437:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10-RESPONSE-1.md"
context_hash: "8973a54cf0bdc7f33becedbbb14f3af50066075e"
context_hash_method: 'git rev-parse "d2ce792a71002310f9ef760962bcb1141ec99437^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - ".github/workflows/via000-r3-protocol.yml"
  - ".github/workflows/via000-r3-windows-transport-diagnostic.yml at d16a90d73837c2e7737525c8bf40d1c70f64754e"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-11.md at 185defd140fb9a60fab00d05cf1aee7fb188bcea"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and GitHub diagnostic runs 32722238480, 32722416771, 32722567809, 32722822767, 32723138041 plus prior public hosted proof metadata/logs; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-12. The unchanged campaign handoff,
  RR11 artifact, experiment commit/tree/workflow hash/history, all five experiment
  runs and both decisive jobs, affected campaign workflow source, prior hosted Windows
  job identities, and cache/artifact metadata were independently inspected. The
  experiment branch adds only its disposable workflow over the campaign handoff and
  is not campaign evidence. No implementation or external state was changed.
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
  CHANGES REQUESTED with one critical Windows execution-boundary blocker. The
  architecture remains viable, but all prior hosted Windows containment/check/cache
  evidence produced through the custom shell is invalidated.

  The five-run experiment is decisive for the GitHub-hosted Windows runner behavior.
  Run 32722238480 showed that Windows PowerShell `-File {0}` receives an extensionless
  runner script and rejects it explicitly. After changing that command to
  `-Command ". '{0}'"`, runs 32722416771, 32722567809, and 32722822767 reported the
  generator step green, while the following built-in cmd step could not find its
  required file and exited 91. At final commit d16a90d73837c2e7737525c8bf40d1c70f64754e,
  run 32723138041 changed only the generator boundary to built-in `shell: pwsh` plus
  observability. Windows job 97418627482 then ran as
  `C:\Program Files\PowerShell\7\pwsh.EXE`, passed assertions for that canonical path,
  PS major 7, the sanitized PATH, exact 54/65-byte same-step files, single-link/no-
  reparse/no-stream metadata, and both fixed hashes. Built-in `shell: cmd` saw the
  file, emitted digest 02332abe36219e1d511d97cbcaf93df11d7104732f0fc8a9eb7b4267b1dc545a,
  and runner finalization propagated it. Exact digest-bound cache preflight missed.

  The remaining custom dot-source staging and revalidation steps then each reported
  green but created no path. Cache/save warned the path did not exist and returned
  success; Ubuntu job 97418736127 received the exact digest/key and failed the exact
  restore. No cache or artifact exists. Therefore, for this hosted runner and its
  extensionless custom-shell script, `powershell.exe -Command ". '{0}'"` is a
  demonstrated fail-open no-op, not merely a suspected output or Node visibility
  issue. It invalidates green metadata for every Windows scripted check using it.

  The production workflow has a separate but equally blocking unsupported form:
  quoted custom `C:\Program Files\PowerShell\7\pwsh.exe ... -File {0}` matrix values.
  Earlier hosted diagnostics already rejected the spaced custom-shell expression,
  and the campaign workflow has never established a Windows production execution
  with it. Both custom forms must be removed. Ubuntu's no-space absolute pwsh shell
  is not inferred defective: prior hosted Ubuntu steps executed, emitted outputs,
  created caches, and failed when their scripted assertions deliberately rejected.
findings:
  - id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001"
    severity: critical
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:309-430; .github/workflows/via000-r3-protocol.yml:30-50,68-70,214-216,413-414,517-519,658-660,793-795,824-825,851-853"
    evidence: |-
      Experiment commits/runs form a controlled causal chain. Run 32722238480 logged
      `Processing -File 'D:\a\_temp\...' failed because the file does not have a
      '.ps1' extension`. With the dot-source workaround, runs 32722567809 and
      32722822767 reported generator success but built-in cmd exited 91 because the
      required file did not exist. In run 32723138041, built-in pwsh created and
      same-step validated the fixed files; built-in cmd observed them and runner
      finalization logged `Set output 'diagnostic_digest'`. Preflight recorded exact
      miss key
      `via000-r3-win-transport-v1-1161864810-d16a90d73837c2e7737525c8bf40d1c70f64754e-d16a90d73837c2e7737525c8bf40d1c70f64754e-32723138041-1-windows-x86_64-diagnostic-02332abe36219e1d511d97cbcaf93df11d7104732f0fc8a9eb7b4267b1dc545a`.
      The subsequent dot-source staging and revalidation steps remained green, yet
      cache/save warned the path did not exist and Ubuntu exact restore failed.
      Artifact and matching-cache counts are zero.

      At the campaign handoff the proof workflow uses the demonstrated no-op shell
      for four steps in each Windows cell: Prove contained Windows stage and stage
      envelope; Validate staged envelope and emit digest; Assert exact cross-OS
      archive tools; Require fresh exact cache key. Thus all twelve scripted checks
      are affected. The save actions execute but their green status is non-evidence
      because missing-path warnings are swallowed and only aggregate restore can
      establish transport.

      The production workflow assigns all three Windows matrix cells a quoted,
      spaced-path custom pwsh `-File {0}` shell. It affects Reject non-snapshot
      lifecycle refs; Validate complete trusted toolchain; Prove production
      containment; Execute frozen clean platform protocol; Execute frozen mutation-
      test matrix (Windows mutation only); Capture exact attestation subjects; Retain
      producer attestation bundle; and failure-only Remove rejected platform workspace.
      No Windows production run validates this form.
    finding: "Unsupported custom Windows PowerShell shells either reject GitHub's extensionless runner script or return success without executing it, so the proof workflow is fail-open and the production workflow cannot establish its Windows gates."
    failure_scenario: "GitHub marks a security-critical Windows run step successful although none of its containment, validation, staging, digest, or cache-preflight-check script ran; downstream actions see missing files/outputs and may themselves return warning-success."
    consequence: "All Windows hosted conclusions from the affected steps are invalid; exact retained 2x3 evidence is absent; production authorization, tool identity, containment, execution, mutation, subject capture, attestation retention, and cleanup have no valid Windows workflow boundary."
    required_action: |-
      Replace every affected Windows dot-source and spaced-path custom PowerShell
      shell in both workflows with supported built-in `shell: pwsh`. Do not retain a
      custom wrapper around `{0}`. Before shell resolution, constrain the step/job PATH
      to `C:\Program Files\PowerShell\7;C:\Windows\System32;C:\Windows` (plus only
      separately verified immutable directories when materially necessary), with no
      workspace, RUNNER_TEMP, candidate, package-cache, or user-writable directory.
      At entry to every trusted pwsh step, require the current process MainModule path
      and `$PSHOME\pwsh.exe` to equal `C:\Program Files\PowerShell\7\pwsh.exe`
      case-insensitively, require the frozen PS7 version policy, and require PATH to
      equal the declared sanitized value. Candidate-controlled state must never be
      able to add or replace the resolved executable.

      Retain native built-in `shell: cmd` only for the tiny fixed digest-output bridge
      if a built-in-pwsh output regression is not yet independently demonstrated. It
      may read only a trusted, post-teardown, exactly hashed 65-byte digest file and
      append one fixed-name lowercase 64-hex value to GITHUB_OUTPUT; the next trusted
      step/job must reject missing or malformed output without logging it. Do not use
      cmd for validation, containment, staging, or other material work.

      Every material side effect must be asserted in the same executing pwsh step and
      independently rechecked across the next boundary: contained-tree teardown and
      zero descendants; exact closure hashes; newly created ordinary/single-link/non-
      reparse files; exact sizes/hashes; output presence/shape; subject hashes before
      and after signing; cache path after preflight and immediately before save; and
      cleanup absence. Action success alone remains non-evidence.

      First rerun the local live Windows hostile replacement test. A single disposable
      Windows-built-in-pwsh diagnostic may then prove post-preflight staging, next-step
      revalidation, cache save, and exact Ubuntu restore. It remains non-authoritative.
      Final approval requires a fresh exact-source six-cell hosted proof from scratch,
      six distinct exact cache restores, canonical 2x3 aggregation, one retained
      seven-file artifact, and green dependent redownload verification. Then rerun and
      independently review the complete production workflow without creating a signer,
      tag, lifecycle transition, or scientific result.

      Audit Ubuntu absolute shells separately and keep their observed working boundary
      or replace them consistently only after equivalent identity assertions. Do not
      classify Ubuntu evidence invalid merely because Windows custom shells failed.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001"
    description: "Regression and hosted replay proving all Windows trusted steps use built-in pwsh resolved to canonical non-writable PowerShell 7 under exact sanitized PATH; sentinel side effects and deliberate throws execute; every material output/file/process/hash has same-step postconditions and cross-step rechecks; custom dot-source/spaced-path forms are absent; optional cmd bridge is digest-only; local hostile test, six-cell exact hosted proof, retained seven-file artifact, and dependent verifier pass."
    rationale: "Green step metadata is meaningless when the shell can no-op; execution observability plus downstream state rechecks is required before containment or transport evidence is admissible."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Git-normalized cross-platform blob result preserved; this does not validate hosted shell execution.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: unresolved, evidence: "Input grammar remains frozen, but the Windows shell execution boundary is fail-open.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Prior hosted Windows tool checks using the no-op shell are invalid.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Source closure remains designed, but the Windows workflow did not execute its checks.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Prior hosted Windows containment/subject checks using the no-op shell are invalid.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Safe proof path exists, but its Windows green scripted steps did not execute.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "The apparent Windows containment/canonical checks before export were no-ops.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "Windows envelope-generation and validation steps were no-ops.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Cache design remains viable, but prior Windows producers did not execute.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "The bounded five-run experiment isolated the custom-shell no-op and ruled out cmd output propagation and generic cache/Node visibility as the primary cause.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: unresolved, evidence: "Windows command execution boundary failed open.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Prior hosted Windows checks did not execute.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: unresolved, evidence: "Windows workflow execution was not established.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Windows containment/subject checks did not execute.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Exact retained 2x3 proof remains absent and Windows green checks are invalid.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Windows producer never created validated evidence.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "Windows canonical envelope code did not execute.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Windows cache producer code did not execute.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Run 32723138041 proved built-in pwsh and cmd execution/output, exact preflight miss, dot-source no-op, missing save path, and exact Ubuntu restore miss without scientific work.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", notes: ""}
predictions: {experiment_id: "", predicted_outcome: "", predicted_failure_mode: "", confidence_statement: "No scientific execution occurred."}
recommendation: {approve: false, blocking_findings: 1, rationale: "Critical but bounded CHANGES REQUESTED. Replace every affected Windows custom shell, prove actual execution and postconditions, rerun local hostile checks and the complete retained six-cell hosted proof; architecture remains viable."}
```

## Verification ledger

- Campaign handoff `d2ce792a71002310f9ef760962bcb1141ec99437`, tree `8973a54cf0bdc7f33becedbbb14f3af50066075e`, sole parent `4c8fbeee32e45848e307d4c2b54c058c9072ff22`, and origin branch head matched. RR11 commit `185defd140fb9a60fab00d05cf1aee7fb188bcea` had Git-normalized SHA-256 `9274ce391c6047cb076e4b043b7ed0e20c98004cf8956ac2a3f4532821dfd745`.
- Experiment commit `d16a90d73837c2e7737525c8bf40d1c70f64754e`, tree `93112c07e47064e6d50ef70d32550e485e39f0d2`, branch head, five-commit ancestry from the campaign handoff, artifact-only experiment scope, and workflow SHA-256 `a5d0595c23f01195557dda442d447d75e3970892f3acc892241e7ce91de00311` matched.
- Runs `32722238480`, `32722416771`, `32722567809`, `32722822767`, and `32723138041` plus all producer/consumer step conclusions and logs were inspected. The final Windows job was `97418627482`; Ubuntu was `97418736127`. Exact key/digest propagated; preflight missed; dot-source steps green/no path; save warning; exact restore failure; zero cache/artifact.
- Invalidated prior hosted Windows scripted evidence includes runs/jobs: `32705009673` (`97364106435`, `97364106550`, `97364106303`); `32710277023` (`97379982937`, `97379982873`, `97379982960`); `32714269336` (`97392081274`, `97392081512`, `97392081475`); `32718921969` (`97405985825`, `97405986037`, `97405985752`); and `32720581939` (`97410945232`, `97410945352`, `97410945403`). Their action-level failures remain factual, but their green custom-shell containment, validation, digest, archive, or preflight-check steps are not execution evidence.
- Prior Ubuntu absolute-shell evidence remains admissible within its declared hosted control-plane boundary: those scripts produced bound files/outputs/caches and their deliberate failures surfaced as failed steps.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending; no matching local or remote VIA-000 R3 tag exists. No signer, key, tag, refreeze, custody, lifecycle, holdout, scientific execution, commitment, or reveal was performed or authorized.
