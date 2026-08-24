# VIA-000 R3 recovery-protocol independent re-review 22

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-22"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "77de37b64fbb5030f90da0b9207371e59d440f44"
baseline_commit: "bd3bcaf79c7b1e67dd8a290ce497583253308c62"
prior_review_ref: "bd3bcaf79c7b1e67dd8a290ce497583253308c62:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-21.md"
builder_response_ref: "77de37b64fbb5030f90da0b9207371e59d440f44:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-21-RESPONSE-1.md"
context_hash: "de785145985a272443778f260c8a0359227681da"
context_hash_method: 'git rev-parse "77de37b64fbb5030f90da0b9207371e59d440f44^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-aggregator.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-21.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-21-RESPONSE-1.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository, exact GitHub run 32777599858 jobs/logs/API, exact runner-image source/release metadata, and retained non-scientific artifact 9538560907; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-22 at exact handoff
  77de37b64fbb5030f90da0b9207371e59d440f44. Exact origin, remote head,
  sole-parent content ancestry, trees, review/response history, workflow/receipt
  sources, every hosted job timeline/log, artifact API metadata, all seven retained
  files, frozen retained verification, exact runner-image release source, focused
  regressions, parser closure, guidance, and campaign state were inspected. No
  implementation, campaign, lifecycle, or scientific state was changed. Operator
  and orchestrator are shared; session, worktree, and branch differ. Builder model
  is shared and external scientific validation is not claimed.
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
  CHANGES REQUESTED with one high-severity, fail-closed Ubuntu PowerShell launcher
  identity blocker. RR21's digest grammar and source normalization are bounded and
  remain viable, but the new normalizer did not execute them. All six producers and
  caches passed, aggregate job 97592385056 restored and validated all six envelopes,
  retained artifact 9538560907 contains exactly seven valid ordinary files, and the
  frozen aggregator independently accepts the downloaded bytes. The adjacent
  normalizer failed only its first compound process/profile/version/PATH gate after
  GitHub resolved built-in `shell: pwsh` as `/usr/bin/pwsh -command ". '{0}'"`; the
  verifier consequently skipped.

  Handoff 77de37b64fbb5030f90da0b9207371e59d440f44 has exact tree
  de785145985a272443778f260c8a0359227681da and sole parent/content
  914ca1ae2e9cb3b916ae2231ef288d8f3a597ed8 with tree
  5b060b1fa08a468854f4fa0eca214f81a936c1ec. Origin's campaign head matched.
  The handoff adds only the RR21 response and campaign/packet review metadata over
  the remediation content. Imported RR21 review Git-normalized SHA-256 is
  494124e190e0a7f037af5f19435e35ca89b075a47496adac6bf2db16344c1dee;
  its response is fa72c3e205d67a5a6e41b453abddee4d01e82c1b5632c77f867ec7af3295642b.

  Hosted push run 32777599858 attempt 1 was exact at the handoff. Ubuntu producer
  jobs 97592144271/97592144546/97592144576 and Windows producer jobs
  97592144509/97592144705/97592144834 passed every containment, digest, cache-save,
  and post-save check. Aggregate job 97592385056 required six green/distinct jobs,
  restored six exact hit/key-equal caches, passed canonical aggregation, wrote seven
  retained files, and uploaded ID 9538560907, name
  via000-r3-containment-proof-retained, size 48022, raw digest
  45e25356f4a81415f2ccd01d5c33befa55748f00f011ea9018e1324e6ff0a4dc,
  and exact URL
  https://github.com/whact2025/POPGP/actions/runs/32777599858/artifacts/9538560907.
  The API reports exactly that artifact and canonical digest
  sha256:45e25356f4a81415f2ccd01d5c33befa55748f00f011ea9018e1324e6ff0a4dc.

  Independent download found exactly aggregate.json plus the six expected platform/
  stage envelopes, all ordinary non-link files. Their SHA-256 values are
  31e2fc72628649849cddfe5b7894176453ab4f42978ce95f3211afa694e3062f,
  97ef7703b451166bcdb432e78bfdeda2e2519d4cedfb8e921ec99a9ab7a7d067,
  9717927633feb73943ab8034edf3fc302341815058dd0fa195e0b522f97a4fdf,
  c283d1398dfab1432704fb47e77727689b9b864ddb1a5a1967e745411be8f6fc,
  97204db1f4a7b5259f524c91dd25e52106e3ceb14e8cb45619a6325081cac293,
  90f5a50c0bf725ce41a53b190f87b12c7d5faf144ec9d9c76c2c834c0e5fb091,
  and c3afca939cba4722e77da2cde3428780b0d75a89530f07e4787f95b33097dffd.
  The exact frozen `--verify-retained` invocation returned zero for source/ref/
  workflow/run/attempt 77de37b.../campaign branch/exact proof workflow/
  32777599858/1.

  The normalizer log exposes exact raw ID, digest, URL, repository, run ID, and PATH,
  then throws from the first gate before assigning or testing those raw values. It
  directly records the runner-selected shell as `/usr/bin/pwsh -command ". '{0}'"`,
  whereas the script requires process path `/opt/microsoft/powershell/7/pwsh` and
  PSHOME `/opt/microsoft/powershell/7`. The compound exception does not print
  MainModule, PSHOME, version, or profile paths individually, so the log alone must
  not be described as a direct value dump of each predicate. However, the exact
  runner launcher is the only new boundary; the same job's successful trusted Ubuntu
  steps use `/opt/microsoft/powershell/7/pwsh -NoLogo -NoProfile -NonInteractive
  -File {0}`, the exact image release documents PowerShell 7.6.5, the Microsoft
  install layout binds stable PSHOME to `/opt/microsoft/powershell/7`, and logged PATH
  is exact. Reusing the already-green literal shell closes the evidence-supported
  discrepancy without accepting a second executable pathname or weakening any gate.
findings:
  - id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:811-828 and corresponding containment-proof-workflow.yml receipt"
    evidence: |-
      The normalizer uniquely declares `shell: pwsh`; GitHub logged its expansion as
      `/usr/bin/pwsh -command ". '{0}'"`. Its source requires observed process path
      `/opt/microsoft/powershell/7/pwsh`, then failed that compound identity gate
      before touching the valid raw artifact values. In the same job, five prior
      PowerShell steps invoked the exact `/opt/.../pwsh -NoLogo -NoProfile
      -NonInteractive -File {0}` form and passed. Ubuntu producer digest steps using
      that literal form also wrote and propagated GITHUB_OUTPUT values successfully.
      Thus no built-in-shell or symlink acceptance is needed for this trusted output
      step.
    finding: "The post-upload normalizer asks GitHub's built-in Ubuntu shell resolver to launch `/usr/bin/pwsh` while simultaneously requiring the process identity to be the different literal `/opt/microsoft/powershell/7/pwsh` path."
    failure_scenario: |-
      All six containment cells and artifact bytes pass, then the normalizer rejects
      before emitting job outputs. The dependent verifier has no artifact identity
      values and skips. Broadly accepting arbitrary PATH-resolved launchers would
      weaken the trusted-tool boundary.
    consequence: "The exact end-to-end hosted gate remains red, so merge and every signer, custodian, refreeze, activation, or holdout gate remain unauthorized."
    required_action: |-
      Change only the normalizer shell declaration from built-in `pwsh` to the exact
      already-proven Ubuntu shell
      `/opt/microsoft/powershell/7/pwsh -NoLogo -NoProfile -NonInteractive -File {0}`.
      Retain the current exact MainModule `/opt/microsoft/powershell/7/pwsh`, PSHOME,
      PS7 semantic version, absent profile files, sanitized `/usr/bin:/bin` PATH, raw
      ID/digest/URL grammar, single-prefix normalization, GITHUB_OUTPUT ordinary-file/
      single-link/write-length, and verifier predicates. Mirror the receipt and all
      bound hashes. Do not add `/usr/bin/pwsh` to an allowlist: the exact target shell
      is already used successfully in this job and has proven GITHUB_OUTPUT behavior.

      If a future design has a real need for built-in `shell: pwsh`, it must instead
      make `/usr/bin/pwsh` an explicit reviewed link object: require lstat to show one
      root-owned non-group/world-writable symbolic link, exact non-relative direct
      target and canonical realpath `/opt/microsoft/powershell/7/pwsh`, root-owned
      non-group/world-writable ancestors, an exact regular trusted target, and
      link/target identity and hash unchanged immediately before and after use. It
      must also retain exact PSHOME, PS7, no loaded/existing profiles, sanitized PATH,
      and reject alternate, chained, relative, missing, replaced, or retargeted links.
      That larger mechanism is unnecessary for RR22 and should not replace the one-line
      literal-shell correction.

      Update the RR21 regression to require the exact literal shell and explicitly
      reject built-in `pwsh`, `/usr/bin/pwsh`, alternate target paths, omitted
      NoProfile/NonInteractive/File flags, and source/receipt divergence. Parse the
      resulting workflow block. Rerun the exact proof-only 2x3 workflow from the new
      handoff. Acceptance requires six green producers/caches, green aggregate/
      normalizer, exact canonical outputs, one seven-file artifact, and a green fresh
      redownload verifier. Do not reuse run 32777599858 as lifecycle proof.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001"
    description: "Require the normalizer's exact literal `/opt/.../pwsh -NoLogo -NoProfile -NonInteractive -File {0}` shell, reject built-in/symlink/alternate/flag/profile/path substitutions, preserve strict digest canonicalization, and pass the full exact-handoff six-cell artifact plus redownload verifier."
    rationale: "Source-only digest tests missed that the new trusted step selected a launcher incompatible with its own production identity predicate."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "All hosted command boundaries before the new normalizer execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-resolved, evidence: "Six producer tool identities and retained subjects pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Six contexts and retained bytes pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-resolved, evidence: "Six post-quiescence subjects pass retained verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Exact retained proof validates, but hosted normalizer/verifier chain is red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 export, cache, aggregate, and retain exact envelopes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six canonical envelopes survive transport and retained verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six digest-bound cache saves/restores pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "Supported Windows built-in pwsh executes all Windows steps; Ubuntu exact -File shell also executes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "No-LUA low-integrity children execute.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: verified-resolved, evidence: "Windows 3/3 production low-IL containment and retention pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "Scripts and workflow blocks parse; hosted proof reaches the new normalizer.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu 3/3 exact subjects aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: verified-resolved, evidence: "Windows 3/3 exact descriptors pass through retention.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", outcome: verified-resolved, evidence: "Native descriptors pass all Windows cells and retained aggregation.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-resolved, evidence: "Windows 3/3 exact zstd/cache identity passes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", outcome: verified-resolved, evidence: "All twelve live inner JSON subjects aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Raw values and source grammar are exact, but production normalization and verifier never execute past the incompatible launcher gate.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Prior and producer command boundaries pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-satisfied, evidence: "Six producer tool manifests and retained subjects pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Six exact contexts and retained subjects pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-satisfied, evidence: "Six post-quiescence attestations validate.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted normalizer/verifier chain remains red.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-satisfied, evidence: "Windows 3/3 export and retain exact envelopes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six canonical envelopes pass independent retained verification.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six exact caches and retained bytes pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: verified-satisfied, evidence: "Windows built-in pwsh and Ubuntu literal -File shells execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: verified-satisfied, evidence: "Windows 3/3 low-IL subjects aggregate and retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Eight R3 scripts and current workflow blocks parse.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu 3/3 subjects aggregate and retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: verified-satisfied, evidence: "Windows 3/3 export boundaries retain exact evidence.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", outcome: verified-satisfied, evidence: "Native descriptors pass all Windows cells and retained aggregation.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-satisfied, evidence: "Exact zstd/cache identity passes all Windows cells.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", outcome: verified-satisfied, evidence: "Live canonical writers and retained six-cell bytes pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Source mutation corpus passes, but production normalizer and required green verifier are blocked by its launcher identity.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY"
  predicted_outcome: "Using the already-green literal `/opt/.../pwsh ... -File {0}` shell will satisfy the unchanged identity gate, emit one canonical digest, and allow the frozen verifier to pass."
  predicted_failure_mode: "Any launcher, target, PSHOME, version, profile, PATH, raw artifact, output-control, archive, or extracted-byte mismatch rejects before acceptance; no lifecycle path runs."
  confidence_statement: "High confidence that this is a bounded deterministic launcher-contract defect and that architecture remains viable. The compound gate did not log every internal value, so the exact hosted rerun remains the final proof. No scientific prediction or execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. Make the one-line exact Ubuntu shell substitution, preserve all normalizer/verifier predicates and receipt bindings, add launcher negative controls, and rerun the exact proof-only 2x3 gate through a green redownload verifier. Run 32777599858 and artifact 9538560907 are synthetic, scientifically/lifecycle inert, and cannot authorize merge, signer/custodian work, refreeze, activation, or holdout."
```

## Verification ledger

- Exact handoff/content/tree/sole-parent/origin identities matched. Handoff scope is the RR21 response plus campaign/packet review metadata. Workflow/receipt and aggregator/receipt are byte-identical with Git-normalized SHA-256 `4b74a1ca23c8aa24978aa524eff2245f8ff32544f6ec9017e5a23993804a2270` and `03a77df68c86477d6fb74d86b8aa540f6f96c2d8dce8ff26a7326e35d2f439c3`.
- All eight job records and aggregate logs for run `32777599858` were inspected. Six producers passed. Aggregate `97592385056` passed every step through upload, then only the new normalizer failed its first gate; verifier `97592549418` skipped. The API reports exactly artifact `9538560907` and the exact canonical digest above.
- The artifact was independently downloaded; it contained exactly seven ordinary files and the frozen `--verify-retained` invocation returned zero under the exact source/ref/workflow/run/attempt identity. Its aggregate asserts `non_scientific`, `no_campaign_execution`, `no_candidate_checkout`, `no_lifecycle_mutation`, `no_custody_access`, and `no_commitment_or_reveal`.
- Focused RR10/RR19/RR20/RR21 regressions passed 4/4 in 29.32 seconds. Review-guidance tests passed 9/9. Eight R3 source/receipt PowerShell scripts parsed with zero errors. Ruff passed on changed Python, campaign validator reported valid, and diff check passed. Full suites, TeX, and actionlint were not rerun because this narrow artifact-only review changes no implementation and the decisive hosted launcher failure plus independently valid retained bytes bind the verdict.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending. No R3 production tag, signer/key, lifecycle/refreeze, custody access, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
