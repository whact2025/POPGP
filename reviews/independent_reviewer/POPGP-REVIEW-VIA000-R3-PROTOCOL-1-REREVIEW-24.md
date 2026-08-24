# VIA-000 R3 recovery-protocol independent re-review 24

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-24"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "768167269def1d6689ee38b0051127f51d670579"
baseline_commit: "aec96bda933951f32b536d4410fe480427bec2d7"
prior_review_ref: "aec96bda933951f32b536d4410fe480427bec2d7:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-23.md"
builder_response_ref: "79f282a6dc8a5604a91f18e90c620e1dd71aa195:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22-RESPONSE-1.md"
context_hash: "e6ce5b646bee311a71d79ee4b7c2dd2d8b595160"
context_hash_method: 'git rev-parse "768167269def1d6689ee38b0051127f51d670579^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-ubuntu-pwsh-identity-diagnostic.yml"
  - ".github/workflows/via000-r3-containment-proof.yml at 79f282a6dc8a5604a91f18e90c620e1dd71aa195"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml at 79f282a6dc8a5604a91f18e90c620e1dd71aa195"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-23.md at aec96bda933951f32b536d4410fe480427bec2d7"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22-RESPONSE-1.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and exact GitHub diagnostic run 32783816053/job 97611378261 plus production run 32782295879 logs/API; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-24 at exact experiment commit
  768167269def1d6689ee38b0051127f51d670579. Exact experiment identity, tree,
  sole-parent campaign ancestry, remote head, workflow bytes, complete diagnostic
  source/log/API, decisive production log, prior review, focused regressions,
  parser closure, guidance, and campaign state were inspected. No implementation,
  campaign, experiment, lifecycle, or scientific state was changed. Operator and
  orchestrator are shared; session, worktree, and branch differ. Builder model is
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
  CHANGES REQUESTED with one high-severity, fail-closed Ubuntu in-process PATH
  contract blocker. The RR23 experiment is decisive for the production compound
  mismatch and supports a direct correction without another diagnostic. Under the
  exact literal launcher, the job log proves MainModule
  `/opt/microsoft/powershell/7/pwsh`, PSHOME `/opt/microsoft/powershell/7`, PowerShell
  `7.6.5`, and the exact six-element argv
  `/opt/.../pwsh -NoLogo -NoProfile -NonInteractive -File <runner-script>` all pass.
  Although the GitHub environment preamble records PATH `/usr/bin:/bin`, the live
  PowerShell process observes exactly
  `/opt/microsoft/powershell/7:/usr/bin:/bin`. The one trusted runtime-directory
  prefix fully explains the unchanged production gate's rejection.

  Experiment commit 768167269def1d6689ee38b0051127f51d670579 has exact tree
  e6ce5b646bee311a71d79ee4b7c2dd2d8b595160, sole parent campaign handoff
  79f282a6dc8a5604a91f18e90c620e1dd71aa195, and adds only the diagnostic workflow.
  Origin's experiment and campaign refs matched. The diagnostic workflow
  Git-normalized SHA-256 is
  8f0e0cebfdaedf974358a14fbdab5b9a33e30ff3ac4e805b764e7d32f79c5980.
  Prior RR23 commit/tree are aec96bda933951f32b536d4410fe480427bec2d7/
  c853d02d40afbe981be18fb8ec6a8996b365db87, and its Git-normalized artifact
  SHA-256 is 244d944f9f6b17bc2afab67f34f6baceb569d8ea23d9768772c0ff35f758846b.

  Run 32783816053 attempt 1 and sole job 97611378261 are exact at the experiment
  commit. The first step used the exact literal launcher and exposed the values
  above. It also reported `/usr/bin/readlink` for the attempted `/proc/self/exe`
  probe: because external `readlink` resolves `/proc/self` as its own process, that
  is a diagnostic misprobe, not PowerShell identity evidence. MainModule already
  directly identifies the live PowerShell executable. The workflow then stopped at
  `trusted path owner or mode differs for /opt`; its overstrict helper requires every
  ancestor to have uid/gid zero and no group/world write but does not log `/opt`'s
  rejected stat. That auxiliary condition is absent from the production compound
  gate. The stop occurred before profile metadata, synthetic `-NoProfile`, synthetic
  GITHUB_OUTPUT, or the second step, so none may be claimed as executed. The run
  created zero artifacts and contains no checkout, candidate, campaign, lifecycle,
  custody, holdout, commitment, reveal, or scientific path.

  Profile-file nonexistence is not the security invariant. The exact trusted
  executable plus exact `-NoProfile` argv is the control that suppresses profile
  execution; merely finding a vendor/user profile path says neither that it loaded
  nor that it influenced the process. Production should not read profile contents
  or require trusted hosted-image profile files to be absent. It should require the
  exact initial in-process PATH, sanitize it explicitly before any artifact/output
  operation, and retain exact executable, PSHOME, version, and argv controls.
findings:
  - id: "VIA000-R3-RR24-UBUNTU-PWSH-INPROCESS-PATH-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:807-860 at 79f282a6dc8a5604a91f18e90c620e1dd71aa195 and mirrored receipt"
    evidence: |-
      Production run 32782295879 logs the exact literal launcher and job environment
      PATH `/usr/bin:/bin`, then rejects its combined identity gate. Under the same
      launcher and job PATH, experiment run 32783816053 logs exact passing
      MainModule, PSHOME, 7.6.5 version, and argv, but live PATH is exactly
      `/opt/microsoft/powershell/7:/usr/bin:/bin`. This single observed mismatch is
      in the production predicate and deterministically explains the rejection.
      Later experiment failures are outside that production predicate and occurred
      after the decisive observations.
    finding: "The production normalizer compares the live PowerShell PATH to the pre-launch job PATH even though the exact trusted PowerShell runtime prepends its own directory at process startup."
    failure_scenario: |-
      Every proof cell, cache, aggregate, and retained artifact can validate, but the
      normalizer always rejects before canonical output emission. Simply accepting
      arbitrary prefixes or leaving the expanded PATH in place would weaken the
      trusted-command boundary.
    consequence: "The exact hosted verifier remains red, so merge and every signer, custodian, refreeze, activation, or holdout gate remain unauthorized."
    required_action: |-
      Patch the production normalizer and exact receipt only through the bound
      campaign amendment. Keep the literal shell declaration. At script entry,
      require MainModule exactly `/opt/microsoft/powershell/7/pwsh`, PSHOME exactly
      `/opt/microsoft/powershell/7`, PSVersion exactly the reviewed `7.6.5` (or an
      equally explicit bound version value that fails closed on image drift), and
      live PATH exactly `/opt/microsoft/powershell/7:/usr/bin:/bin`.

      Read `/proc/self/cmdline` in-process and require exactly six arguments: exact
      executable, `-NoLogo`, `-NoProfile`, `-NonInteractive`, `-File`, and one
      runner-script operand in the expected trusted runner temporary form. Do not
      use external `readlink /proc/self/exe`; if a proc executable check is retained,
      address `/proc/$PID/exe` and review it separately. MainModule is already proven
      exact here.

      Immediately after those checks, assign `$env:PATH = '/usr/bin:/bin'` and
      reassert exact equality before reading VIA000_RAW_ARTIFACT_* values, before the
      absolute `/usr/bin/stat` call, immediately before GITHUB_OUTPUT append, and
      after the append. Never accept arbitrary/reordered/duplicate prefixes or extra
      path entries. Preserve raw ID/lowercase digest/URL/repository/run grammar,
      single `sha256:` canonicalization, GITHUB_OUTPUT ordinary-file/single-link and
      exact byte-growth checks, and the downstream artifact ID/name/run/attempt/
      source verifier bindings.

      Remove only the `$PROFILE` file-nonexistence predicate. Do not read profile
      contents. Exact literal executable plus exact argv `-NoProfile` is the relevant
      no-load invariant. Add source/receipt regressions that require the exact initial
      and sanitized PATH values in the correct order; exact argv and flags; exact
      executable/PSHOME/version; no profile-existence condition; and rejection of
      missing, extra, moved, reordered, duplicated, case-varied, or attacker-writable
      PATH/argv values. Mirror all hashes and receipts.

      No further diagnostic run is necessary. Rerun the exact proof-only 2x3 workflow
      from the new campaign handoff. Acceptance requires six green producers/caches,
      green aggregate and normalizer with exact canonical outputs, one exact seven-
      file retained artifact, and a green fresh redownload verifier. Prior experiment
      and production runs are not lifecycle proof.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001"
    description: "Require exact literal launcher/MainModule/PSHOME/version/six-argument `-NoProfile` argv and exact initial trusted-runtime-prefixed PATH, sanitize to `/usr/bin:/bin` before all artifact/output operations with repeated checks, reject all variants, and pass a new exact-handoff 2x3 artifact plus redownload verifier."
    rationale: "The hosted diagnostic identifies the only production compound mismatch; ordered sanitization closes it without accepting a mutable or caller-controlled PATH."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Hosted producer command boundaries pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-resolved, evidence: "Six producer identities and retained subjects pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Six contexts and retained bytes pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-resolved, evidence: "Six post-quiescence subjects pass retained verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Retained proof validates, but hosted normalizer/verifier is red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR24-UBUNTU-PWSH-INPROCESS-PATH-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 export, cache, aggregate, and retain exact envelopes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six canonical envelopes survive transport and retained verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six digest-bound cache saves/restores pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "Supported Windows and Ubuntu literal shells execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "No-LUA low-integrity children execute.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: verified-resolved, evidence: "Windows 3/3 low-IL containment and retention pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "Eight scripts parse and hosted proof reaches normalizer.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu 3/3 subjects aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: verified-resolved, evidence: "Windows 3/3 export boundaries retain exact evidence.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", outcome: verified-resolved, evidence: "Native descriptors pass all Windows cells and retention.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-resolved, evidence: "Windows 3/3 exact zstd/cache identity passes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", outcome: verified-resolved, evidence: "All live inner JSON subjects aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Production normalization/verifier remain blocked by the now-isolated PATH contract.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR24-UBUNTU-PWSH-INPROCESS-PATH-001", notes: ""}
  - {finding_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: unresolved, evidence: "Literal launcher passes; live PATH expectation is wrong.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR24-UBUNTU-PWSH-INPROCESS-PATH-001", notes: ""}
  - {finding_id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Experiment logs each decisive production identity term and isolates live PATH.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Auxiliary /proc and /opt probes are invalid/overstrict but do not obscure the production cause."}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Hosted producer boundaries pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-satisfied, evidence: "Six producer identities pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Six exact contexts pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-satisfied, evidence: "Six post-quiescence attestations validate.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted normalizer/verifier remains red.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-satisfied, evidence: "Windows 3/3 export and retain exact envelopes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six canonical envelopes pass retained verification.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six exact caches and retained bytes pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: verified-satisfied, evidence: "Supported Windows and Ubuntu literal shells execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: verified-satisfied, evidence: "Windows 3/3 low-IL subjects retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Eight source/receipt scripts parse.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu 3/3 subjects aggregate and retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: verified-satisfied, evidence: "Windows 3/3 export boundaries retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", outcome: verified-satisfied, evidence: "Native descriptors pass all Windows cells.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-satisfied, evidence: "Exact zstd/cache identity passes all Windows cells.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", outcome: verified-satisfied, evidence: "Live writers and six-cell retained bytes pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Production output normalization/verifier remain blocked by PATH.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: unresolved, evidence: "Literal launcher passes but exact live PATH contract remains wrong.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC-001", outcome: superseded, evidence: "Decisive identity values isolate PATH; auxiliary profile/output portions did not execute and are replaced by direct production closure requirements.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001", notes: "No second diagnostic is required."}
predictions:
  experiment_id: "VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION"
  predicted_outcome: "The exact trusted-runtime-prefixed initial PATH will pass, explicit sanitization will leave `/usr/bin:/bin`, canonical artifact outputs will propagate, and the fresh redownload verifier will accept the exact seven retained files."
  predicted_failure_mode: "Any executable, PSHOME, version, argv, initial/sanitized PATH, raw artifact, output-control, archive, or retained-byte mismatch rejects before acceptance; no lifecycle path runs."
  confidence_statement: "High confidence that the hosted evidence identifies a bounded deterministic PATH-contract defect and that architecture remains viable. The later experiment failures are understood diagnostic limitations and do not require another diagnostic. No scientific prediction or execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. Apply the exact initial-PATH and ordered-sanitization correction, replace profile-file absence with exact `-NoProfile` argv enforcement, preserve every artifact/output binding, add negative controls, and rerun the exact proof-only 2x3 gate through a green redownload verifier. No additional experiment is needed. Neither run authorizes merge, signer/custodian work, refreeze, activation, holdout, or scientific execution. Architecture remains viable."
```

## Verification ledger

- Exact experiment commit/tree/sole-parent/remote identities matched. Its sole change is the diagnostic workflow; Git-normalized SHA-256 is `8f0e0cebfdaedf974358a14fbdab5b9a33e30ff3ac4e805b764e7d32f79c5980`.
- Exact run `32783816053`/job `97611378261` source, API, and full log were inspected against production run `32782295879`. The exact passing/failing values, diagnostic stop, skipped second step, and zero artifacts match this review.
- Focused RR21/RR22 regressions passed 2/2. Review-guidance passed 9/9. Eight R3 source/receipt PowerShell scripts parsed with zero errors. Campaign validator reported valid and experiment diff check passed. Full suites, TeX, Ruff, and actionlint were not rerun because the experiment adds only YAML and live hosted observations bind the narrow verdict.
- Campaign state remains pending; the packet remains drafted with `holdout_started: false` and unrevealed. No production tag, signer/key, lifecycle/refreeze, custody access, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
