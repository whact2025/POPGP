# VIA-000 R3 recovery-protocol independent re-review 23

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-23"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-23"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "79f282a6dc8a5604a91f18e90c620e1dd71aa195"
baseline_commit: "f20a7e48ddbf51b7e0e59bce14647517fbdec441"
prior_review_ref: "f20a7e48ddbf51b7e0e59bce14647517fbdec441:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22.md"
builder_response_ref: "79f282a6dc8a5604a91f18e90c620e1dd71aa195:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22-RESPONSE-1.md"
context_hash: "87f8785740e04f462266ed068553e003b6f1a03f"
context_hash_method: 'git rev-parse "79f282a6dc8a5604a91f18e90c620e1dd71aa195^{tree}"'
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
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22-RESPONSE-1.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository, exact GitHub run 32782295879 jobs/logs/API, and retained non-scientific artifact 9540212981; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-23 at exact handoff
  79f282a6dc8a5604a91f18e90c620e1dd71aa195. Exact origin, remote head,
  sole-parent content ancestry, trees, review/response history, workflow/receipt
  sources, every hosted job timeline/log, artifact API metadata, all seven retained
  files, frozen retained verification, focused regressions, parser closure,
  guidance, and campaign state were inspected. No implementation, campaign,
  lifecycle, or scientific state was changed. Operator and orchestrator are
  shared; session, worktree, and branch differ. Builder model is shared and
  external scientific validation is not claimed.
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
  CHANGES REQUESTED with one high-severity, fail-closed Ubuntu PowerShell identity
  observability blocker. RR22's literal launcher remediation executed exactly, but
  the unchanged compound process/PSHOME/version/profile/PATH predicate still failed
  without logging its individual nonsecret observations. The failed component is
  therefore not inferable from the source and hosted log. PATH and the literal
  launcher are directly proven correct; MainModule.FileName, /proc/self/exe,
  PSHOME, version, and profile-file state are not individually recorded. No direct
  correction is evidence-supported until a bounded Ubuntu-only diagnostic separates
  those values.

  Handoff 79f282a6dc8a5604a91f18e90c620e1dd71aa195 has exact tree
  87f8785740e04f462266ed068553e003b6f1a03f and sole parent/content
  48e9986a3d9e83965584a1eddee32a303cc39ad3 with tree
  32374c00f177d076482f6d5faa4baa5305b8c691. Origin's campaign head matched.
  The handoff adds only the RR22 response and campaign/packet review metadata over
  the remediation content. Imported RR22 review Git-normalized SHA-256 is
  7f92ea0c6a0e7066fbc93114304470f1a5abb139ba8b7e84cbf1644c6e33ebaf;
  its response is 6b82cd53b86898de934dcd3ee8905770f7b02c76816d446d73af68d2305babe5.

  Hosted push run 32782295879 attempt 1 was exact at the handoff. Ubuntu producer
  jobs 97606772940/97606772967/97606772968 and Windows producer jobs
  97606772730/97606773002/97606773059 passed every containment, digest, cache-save,
  and post-save check. Aggregate job 97606964119 required six green/distinct jobs,
  restored six exact hit/key-equal caches, passed canonical aggregation, wrote the
  exact seven retained files, uploaded artifact 9540212981, then failed only the
  identity gate in `Normalize exact retained artifact identity`. Its log shows the
  exact literal launcher `/opt/microsoft/powershell/7/pwsh -NoLogo -NoProfile
  -NonInteractive -File {0}` and exact PATH `/usr/bin:/bin`. The step emitted no
  artifact outputs; verifier 97607103166 skipped.

  The API reports exactly one artifact: ID 9540212981, name
  via000-r3-containment-proof-retained, size 48018, and canonical digest
  sha256:1f03bd61c7634c3da4fbad0b14be8ca85d8d55953a9f684a01ecd188ea818941.
  Independent download found exactly aggregate.json plus the six expected platform/
  stage envelopes, all ordinary non-link files. Their SHA-256 values are
  2af40e62916fe1ca19ee42761f89a089b749cfa3e27cdb015c205a1d4542d26d,
  38987ff7132563a382a67598c21aac37454fa5c052e8a3e5462fa5c394680aa0,
  2a39ae8a2d37a867a7440f7f21f045eff48ed17cef6db61cf8f4e8f4f79f4756,
  882d94f895604de8e614391d3ec2fe0274f2a9f5ed13364edddbaa7d1f4677c3,
  8b12d03f980bb51fab46d43066ad88f1fe1b015b0f80eba852c695ffa4b683b2,
  eb03151f5e82e861b04a1581ee2ac0e05454b2555ffab9f7a18a0a66dd879c9d,
  and 2dcfaba37ff24845ed06cf2a8770e18275a0ac8aecbc7e754a81d18f2938d270.
  The frozen `--verify-retained` invocation returned zero for the exact source/ref/
  workflow/run/attempt. aggregate.json asserts non_scientific,
  no_campaign_execution, no_candidate_checkout, no_lifecycle_mutation,
  no_custody_access, and no_commitment_or_reveal. The artifact is inert evidence;
  its existence does not authorize lifecycle use.
findings:
  - id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-OBSERVABILITY-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:807-860 and corresponding containment-proof-workflow.yml receipt"
    evidence: |-
      Run 32782295879 proves that GitHub invoked the exact reviewed literal shell and
      that PATH was `/usr/bin:/bin`. The script then calculates MainModule.FileName,
      four `$PROFILE` paths, PSHOME, and PSVersion and combines all five identity
      classes plus PATH into one conditional. It throws only `trusted Ubuntu built-in
      pwsh identity, profile, version, or PATH differs`; no individual observed value
      is printed. The throw occurs before raw artifact assignment and GITHUB_OUTPUT
      append. Consequently the log cannot distinguish MainModule, PSHOME, version,
      or any existing profile file. Source inspection cannot establish a hosted
      runtime observation that the run did not retain.
    finding: "The RR22 literal launcher is fixed, but its production identity gate still rejects and provides insufficient nonsecret evidence to identify the mismatched predicate safely."
    failure_scenario: |-
      Guessing a correction could either retain the real incompatibility or broadly
      weaken executable/profile identity. In particular, `-NoProfile` is the launch
      control that prevents profile loading; the mere existence of a profile file is
      a separate condition and must not be assumed to prove profile execution.
    consequence: "The exact end-to-end hosted verifier remains red, so merge and every signer, custodian, refreeze, activation, or holdout gate remain unauthorized."
    required_action: |-
      Run one fresh Ubuntu-24.04, proof-only diagnostic job on an experiment branch,
      with minimal read-only permissions, no secrets, caches, artifacts, repository
      writes, candidate checkout, lifecycle path, or scientific execution. Invoke
      only the literal `/opt/microsoft/powershell/7/pwsh -NoLogo -NoProfile
      -NonInteractive -File {0}` shell with PATH `/usr/bin:/bin`.

      Record each nonsecret observation separately: normalized
      Process.MainModule.FileName; canonical `/proc/self/exe` from exact
      `/usr/bin/readlink -f`; PSHOME; full PSVersion; PATH; and the process argument
      vector proving `-NoLogo`, `-NoProfile`, `-NonInteractive`, and `-File` (normalize
      the temporary script pathname). For every unique `$PROFILE` path, record only
      normalized path, existence, object/link type, canonical target, uid/gid,
      owner, octal mode, link count, length, and SHA-256 when present--never contents.
      Also record the literal executable and ancestor type/owner/mode/hash facts.
      Each predicate must report pass/fail independently rather than through a
      compound exception.

      In the same synthetic job, feed public fixed artifact ID, lowercase 64-hex
      digest, URL, repository, and run values through the unchanged normalizer,
      append to the actual protected GITHUB_OUTPUT, and have a following trusted
      step check exact values while logging pass/fail only. The experiment is
      diagnostic and non-authoritative.

      Correct production only after the observation identifies the mismatch. If
      profile-file existence alone differs, retain literal `-NoProfile` and exact
      argv proof, and remove only the unrelated nonexistence predicate; do not infer
      loading or delete a trusted hosted file. If MainModule differs while canonical
      `/proc/self/exe` proves the exact trusted target, replace only MainModule with
      the reviewed canonical proc identity plus regular-file/owner/mode/hash checks.
      If PSHOME, PSVersion, PATH, launcher, or target differs, retain fail-closed
      behavior and investigate rather than widening acceptance. Mirror the receipt
      and hashes, add positive and negative diagnostic/identity regressions, and then
      rerun the exact six-cell proof from the new handoff. Acceptance requires six
      producers, aggregate/normalizer, one exact seven-file artifact, and a fresh
      green redownload verifier. Do not reuse run 32782295879 for lifecycle proof.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC-001"
    description: "Under the exact literal Ubuntu launcher, separately retain MainModule, `/proc/self/exe`, PSHOME, version, argv, profile metadata, PATH, target identity, and synthetic GITHUB_OUTPUT results; reject each alternate independently and then pass a new exact-handoff 2x3 proof plus redownload verifier."
    rationale: "The compound gate proves rejection but cannot support a narrow evidence-based correction or distinguish profile existence from profile loading."
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
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Retained proof validates, but hosted normalizer/verifier is red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 export, cache, aggregate, and retain exact envelopes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six canonical envelopes survive transport and retained verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six digest-bound cache saves/restores pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "Supported Windows pwsh and Ubuntu literal -File shells execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "No-LUA low-integrity children execute.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: verified-resolved, evidence: "Windows 3/3 low-IL containment and retention pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "Eight scripts parse and hosted proof reaches the normalizer.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu 3/3 subjects aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: verified-resolved, evidence: "Windows 3/3 export boundaries retain exact evidence.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", outcome: verified-resolved, evidence: "Native descriptors pass all Windows cells and retention.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-resolved, evidence: "Windows 3/3 exact zstd/cache identity passes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", outcome: verified-resolved, evidence: "All live inner JSON subjects aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Raw values/source grammar are exact, but production normalization and verifier remain blocked.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: unresolved, evidence: "Literal launcher executed exactly, but the unchanged compound identity gate still failed.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-OBSERVABILITY-001", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Hosted producer boundaries pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-satisfied, evidence: "Six producer tool manifests and retained subjects pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Six exact contexts pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-satisfied, evidence: "Six post-quiescence attestations validate.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted normalizer/verifier remains red.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-satisfied, evidence: "Windows 3/3 export and retain exact envelopes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six canonical envelopes pass retained verification.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six exact caches and retained bytes pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: verified-satisfied, evidence: "Windows built-in and Ubuntu literal shells execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: verified-satisfied, evidence: "Windows 3/3 low-IL subjects retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Eight source/receipt scripts parse.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu 3/3 subjects aggregate and retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: verified-satisfied, evidence: "Windows 3/3 export boundaries retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", outcome: verified-satisfied, evidence: "Native descriptors pass all Windows cells.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-satisfied, evidence: "Exact zstd/cache identity passes all Windows cells.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", outcome: verified-satisfied, evidence: "Live canonical writers and six-cell retained bytes pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Source corpus passes, but production normalizer/verifier remain blocked.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: unresolved, evidence: "Exact launcher passes selection, but compound identity and verifier remain red.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC-001", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC"
  predicted_outcome: "The bounded diagnostic will identify exactly one mismatched identity term while confirming the literal launcher, `-NoProfile` argv, sanitized PATH, and public GITHUB_OUTPUT grammar."
  predicted_failure_mode: "Any launcher, executable, PSHOME, version, argv, profile metadata, PATH, output-control, raw artifact, or retained-byte mismatch rejects independently; no lifecycle path exists."
  confidence_statement: "High confidence that this remains a bounded trusted-control-plane diagnostic and that the architecture is viable. The current evidence does not identify which compound term failed, so no direct production relaxation is approved. No scientific prediction or execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. Run the one-job Ubuntu diagnostic, make only its evidence-supported narrow correction while preserving literal `-NoProfile` execution and all artifact grammar/binding checks, add negative controls, and rerun the exact proof-only 2x3 gate through a green redownload verifier. Run 32782295879 and artifact 9540212981 are synthetic, scientifically/lifecycle inert, and cannot authorize merge, signer/custodian work, refreeze, activation, or holdout. Architecture remains viable."
```

## Verification ledger

- Exact handoff/content/tree/sole-parent/origin identities matched. Handoff scope is the RR22 response plus campaign/packet review metadata. Workflow/receipt and aggregator/receipt are byte-identical with Git-normalized SHA-256 `dbfc1e81f4665be987f8fdc8d7d34e4be5d2fed22816e884f67e0ddac6e910be` and `03a77df68c86477d6fb74d86b8aa540f6f96c2d8dce8ff26a7326e35d2f439c3`.
- All eight job records and aggregate logs for run `32782295879` were inspected. Six producers passed. Aggregate `97606964119` passed through upload, then only the compound normalizer identity gate failed; verifier `97607103166` skipped. The API reports exactly artifact `9540212981` and digest `sha256:1f03bd61c7634c3da4fbad0b14be8ca85d8d55953a9f684a01ecd188ea818941`.
- Independent artifact download contained exactly seven ordinary non-link files. Frozen `--verify-retained` returned zero under exact source/ref/workflow/run/attempt. The aggregate explicitly records every non-scientific/no-lifecycle assertion listed above.
- Focused RR10/RR19/RR20/RR21/RR22 regressions passed 5/5 in 42.22 seconds. Review-guidance passed 9/9. Eight R3 source/receipt PowerShell scripts parsed with zero errors. Ruff passed, campaign validator reported valid, and source/receipt diff checks passed. Full suites, TeX, and actionlint were not rerun because this narrow artifact-only review changes no implementation and the decisive hosted gate failure plus independently valid retained bytes bind the verdict.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending. No production tag, signer/key, lifecycle/refreeze, custody access, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
