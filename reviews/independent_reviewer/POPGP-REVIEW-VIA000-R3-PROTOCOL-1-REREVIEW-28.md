# VIA-000 R3 recovery-protocol independent re-review 28

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-28"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-28"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "0acec59f60ca1fdf60a66b25b60459e3c044bde9"
baseline_commit: "b4a890777633ac6791f4e59ffffeec11cc5bfaf3"
prior_review_ref: "b4a890777633ac6791f4e59ffffeec11cc5bfaf3:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-27.md"
builder_response_ref: "fe83901ca3a78c99de1a199d21e6a157ba5c6308:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-26-RESPONSE-1.md"
context_hash: "a69fbbe2440f2cfe58aee7d366203046135a1f98"
context_hash_method: 'git rev-parse "0acec59f60ca1fdf60a66b25b60459e3c044bde9^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-rr27-setup-python-context-diagnostic.yml"
  - ".github/workflows/via000-r3-containment-proof.yml at fe83901ca3a78c99de1a199d21e6a157ba5c6308"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml at fe83901ca3a78c99de1a199d21e6a157ba5c6308"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-27.md"
  - "actions/setup-python action.yml and src/find-python.ts at ece7cb06caefa5fff74198d8649806c4678c61a1 via GitHub API"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository, exact GitHub diagnostic run 32798221946/job 97653800101, and exact-SHA generic CI run 32798221877; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-28 at exact experiment commit
  0acec59f60ca1fdf60a66b25b60459e3c044bde9. Exact origin, experiment
  branch/commit/tree, parent chain through sealed RR27 and campaign handoff, sole
  workflow diff, action pin/source identity, full workflow/log/API evidence, every
  observation, zero-artifact result, production/receipt predicates, prior review,
  schema/parser/guidance checks, and campaign state were inspected. No campaign,
  implementation, lifecycle, or scientific state was changed. Operator and
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
  CHANGES REQUESTED with one high-severity, exactly isolated production PATH-binding
  defect. The RR27 exact-context diagnostic is decisive and satisfies its requested
  observation: on runner 2.336.0, Ubuntu 24.04 image 20260816.277.1, after exact
  pinned setup-python 3.11.15 with update-environment true, literal pwsh 7.6.5 sees
  exactly five PATH components in this order:
  `/opt/microsoft/powershell/7`,
  `/opt/hostedtoolcache/Python/3.11.15/x64/bin`,
  `/opt/hostedtoolcache/Python/3.11.15/x64`, `/usr/bin`, `/bin`.
  The exact joined string is
  `/opt/microsoft/powershell/7:/opt/hostedtoolcache/Python/3.11.15/x64/bin:/opt/hostedtoolcache/Python/3.11.15/x64:/usr/bin:/bin`.

  Of all current production predicates, only `entry_path` and its equivalent
  component observation fail the old expectation. MainModule, PSHOME, version,
  argc/argv0-5, exact RUNNER_TEMP parent, extensionless lowercase UUID, ordinary
  non-reparse file, regular/single-link/runner UID+GID/mode-0644 metadata, and reset
  to `/usr/bin:/bin` all pass. One supplemental `image_version` observation reports
  false because the experiment's non-production regex permits only one dotted
  suffix while the real value is `20260816.277.1`; it is explicitly marked
  current_predicate false and is not part of the production compound or the final
  failure vector. No further diagnostic is required.

  Experiment commit 0acec59f60ca1fdf60a66b25b60459e3c044bde9 has exact tree
  a69fbbe2440f2cfe58aee7d366203046135a1f98 and sole parent sealed RR27
  b4a890777633ac6791f4e59ffffeec11cc5bfaf3, tree
  9d54f375e9962230c52e3fd11d50047be4e5029a, whose sole parent is campaign
  handoff fe83901ca3a78c99de1a199d21e6a157ba5c6308. All remote refs match.
  Diagnostic workflow normalized SHA-256 is
  6c9a2e00d0bacc1800b0546eb01787ac828fdd822ad600b410071c9c3ebfac11;
  sealed RR27 normalized SHA-256 is
  9a98456ef37b0e96b68da338f804ba150e3956df605277efe2f81d39a3157040.
  Setup-python exact commit/tree are ece7cb06caefa5fff74198d8649806c4678c61a1 /
  39217d5f784fb44e42ddb8e5702f05256b29829e and its action.yml SHA-256 is
  a95cf5ade72699eb8a044e77a74b1e127a223759f13c42a1068bb9cf918b8830.

  The experiment adds exactly one workflow, has empty permissions, and defines no
  checkout, cache action, artifact action, job outputs, or custom GITHUB_OUTPUT
  writes. It consumes only setup-python's declared path/version outputs, writes
  observations to the public log, then intentionally exits one because the two
  old PATH expectations fail. API reports zero artifacts. There is no candidate,
  campaign, custody, lifecycle, holdout, commitment, reveal, or scientific path.
findings:
  - id: "VIA000-R3-RR28-SETUP-PYTHON-PATH-BINDING-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:723-727,807-842 at fe83901ca3a78c99de1a199d21e6a157ba5c6308 and mirrored receipt"
    evidence: |-
      Run 32798221946/job 97653800101 logs the exact five-component live PATH
      and indices 0/1/2 for PSHOME, Python bin, and Python root. Exact setup action,
      action outputs, pythonLocation/root variables, literal shell, version, argv,
      runner-script identity/metadata, and sanitized PATH pass. The final vector has
      observation_count 44 and exactly
      failed_current_predicates:[entry_path,entry_path_components]. Production
      still requires only PSHOME:/usr/bin:/bin, so that one comparison is wrong.
    finding: "Production omits the exact pinned setup-python bin and root entries from the trusted initial PATH it compares before artifact normalization."
    failure_scenario: |-
      Six valid containment cells and an exact retained upload reach the normalizer.
      The trusted setup action has added its fixed bin and root to PATH, pwsh has
      prepended PSHOME, and the three-entry expectation rejects the legitimate
      five-entry context before artifact outputs and the redownload verifier.
    consequence: "The hosted proof remains unavailable until the exact observed setup-python-aware PATH binding is amended and replayed."
    required_action: |-
      Make one bounded campaign amendment to production workflow and exact receipt.
      Keep literal trust anchors rather than deriving acceptance from arbitrary
      action-controlled text: expected PSHOME
      `/opt/microsoft/powershell/7`, expected Python root
      `/opt/hostedtoolcache/Python/3.11.15/x64`, derived/fixed bin `$root/bin`, and
      expected executable `$bin/python`. Pass setup-python's `python-path` and
      `python-version` outputs into the normalizer and require exact equality to the
      fixed executable and `3.11.15`; independently require pythonLocation and all
      Python root environment variables equal the fixed root. Preserve the exact
      pinned action revision and existing Python version/executable checks.

      Build a fixed five-element expected array
      `[PSHOME, PythonBin, PythonRoot, /usr/bin, /bin]`. Split live PATH on `:` and
      require count exactly five plus ordinal case-sensitive equality of every
      element, then also require the raw string equal the fixed array joined by `:`.
      Do not accept prefixes, contains/subsequence checks, wildcards, alternate
      toolcache roots, or a value merely because it came from an action output.
      Immediately set PATH to `/usr/bin:/bin` and reassert it before runner-script
      metadata, artifact ID/digest/URL reads, GITHUB_OUTPUT inspection/append, and
      after every trusted child operation exactly as already ordered.

      Add source/receipt tests for the exact positive vector and reject missing,
      extra, duplicated, reordered, empty, relative, alternate-version/architecture/
      root, case-varied, trailing-colon, `/usr/local/bin`, swapped root/bin, repeated
      PSHOME, manipulated action output/version/pythonLocation/root variables, and
      unpinned action/source/receipt divergence. Remove the diagnostic workflow,
      update protocol/receipt/manifest hashes, and rerun the exact fresh proof-only
      2x3 workflow. Require six green producers, exact caches, aggregate/normalizer,
      one exact seven-file artifact, canonical outputs, and green redownload
      verifier. No more diagnostic is needed before this correction.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR28-EXACT-FIVE-COMPONENT-PATH-001"
    description: "Require exact ordinal five-component PSHOME/setup-python-bin/setup-python-root/usr/bin/bin PATH plus exact pinned action outputs/root variables, reject every mutation, sanitize immediately, and pass a fresh exact-handoff 2x3 retained artifact and verifier."
    rationale: "The exact-context run isolates the omitted two setup-python components as the sole current mismatch and provides the complete positive vector."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Hosted producer boundaries pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-resolved, evidence: "Six producer identities pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Six contexts and retained bytes pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-resolved, evidence: "Six post-quiescence subjects pass retained verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Production normalizer/verifier remains red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR28-SETUP-PYTHON-PATH-BINDING-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 retain exact envelopes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six canonical envelopes survive transport.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six exact cache saves/restores pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "Supported literal shells execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "No-LUA low-integrity children execute.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: verified-resolved, evidence: "Windows 3/3 low-IL containment passes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "Eight scripts parse; hosted workflow reaches normalizer.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu 3/3 subjects aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: verified-resolved, evidence: "Windows export boundaries retain exact evidence.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", outcome: verified-resolved, evidence: "Native descriptors pass all Windows cells.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-resolved, evidence: "Windows zstd/cache identity passes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", outcome: verified-resolved, evidence: "All live inner JSON subjects retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Normalizer outputs/verifier remain blocked.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR28-SETUP-PYTHON-PATH-BINDING-001", notes: ""}
  - {finding_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: verified-resolved, evidence: "Exact literal launcher, module, PSHOME, version, and argv all pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Original PATH mismatch isolated.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR24-UBUNTU-PWSH-INPROCESS-PATH-001", outcome: unresolved, evidence: "Exact setup-python-aware PATH now observed but production is not amended.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR28-SETUP-PYTHON-PATH-BINDING-001", notes: ""}
  - {finding_id: "VIA000-R3-RR25-NORMALIZER-ARGUMENT5-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Exact argv and runner script identity pass in production context.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR26-RUNNER-SCRIPT-SUFFIX-001", outcome: verified-resolved, evidence: "Extensionless UUID and complete metadata pass in exact setup-python context.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR27-SETUP-PYTHON-PATH-CONTEXT-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Exact five-component live PATH and sole current mismatches are logged.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Hosted boundaries pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-satisfied, evidence: "Six producer identities pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Six contexts pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-satisfied, evidence: "Six attestations validate.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted normalizer/verifier remains red.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR28-EXACT-FIVE-COMPONENT-PATH-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-satisfied, evidence: "Windows 3/3 retain exact envelopes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six canonical envelopes verify.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six exact caches pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: verified-satisfied, evidence: "Supported literal shells execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: verified-satisfied, evidence: "Windows low-IL subjects retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Eight scripts parse.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu subjects retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: verified-satisfied, evidence: "Windows export boundaries retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", outcome: verified-satisfied, evidence: "Native descriptors pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-satisfied, evidence: "Exact zstd/cache identity passes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", outcome: verified-satisfied, evidence: "Live writers and retained bytes pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Normalizer outputs/verifier remain blocked.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR28-EXACT-FIVE-COMPONENT-PATH-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: verified-satisfied, evidence: "Exact literal launcher and argv pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC-001", outcome: superseded, evidence: "Earlier diagnostic was incomplete.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001", outcome: unresolved, evidence: "Exact vector known; production amendment/replay pending.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR28-EXACT-FIVE-COMPONENT-PATH-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Exact argv5 and metadata pass in setup-python context.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR26-EXTENSIONLESS-RUNNER-SCRIPT-IDENTITY-001", outcome: verified-satisfied, evidence: "Extensionless UUID and complete metadata pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR27-EXACT-SETUP-PYTHON-CONTEXT-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Exact context logs all terms and isolates only the two equivalent PATH observations.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR28-EXACT-FIVE-COMPONENT-PATH"
  predicted_outcome: "The exact ordinal five-component binding passes; the normalizer emits canonical artifact identity outputs and the fresh redownload verifier accepts the exact seven retained files."
  predicted_failure_mode: "Any PATH/action-output/root/version/runner-script/artifact drift rejects before acceptance; no lifecycle path runs."
  confidence_statement: "High confidence the exact-context diagnostic isolates the sole current production mismatch and supports a direct bounded correction. No further diagnostic or scientific prediction/execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. The next authorized action is only the exact five-component production/receipt amendment, strict output/root/version and negative fixtures, diagnostic removal, hash updates, and fresh proof-only 2x3 replay through green normalizer and redownload verifier. No more diagnostic is needed. Merge and every signer/custodian/refreeze/activation/holdout/scientific gate remain unauthorized. Architecture remains viable."
```

## Verification ledger

- Exact experiment tree/sole-parent chain and all remote experiment/RR27/campaign refs matched. Diff is exactly one new diagnostic workflow; campaign bytes are unchanged.
- Run/job/API and full log were inspected. Exact pin/inputs, runner/image, literal shell, all 44 observations, exact failure vector, intentional exit, and zero artifacts match source. The image-version auxiliary false result is documented rather than silently treated as a production failure.
- Generic CI run 32798221877 completed success at the exact experiment SHA; it is independent of the intentionally failing diagnostic result.
- Source contains empty permissions and no checkout, cache action, artifact action, job output, custom GITHUB_OUTPUT, candidate, campaign, custody, lifecycle, holdout, commitment, reveal, or scientific route. Setup-python's declared outputs are only read as observations.
- Diagnostic YAML, review YAML/schema, exact 31/31 prior ID reconciliation, guidance, PowerShell parse closure, campaign validator, diff check, and normal/ignored cleanliness were checked. Broad suites/TeX/Ruff were not repeated because this branch adds only the diagnostic YAML and its hosted result is decisive.
- Campaign remains drafted, `holdout_started: false`, unrevealed, and pending. No tag, signer/key, custody access, lifecycle/refreeze, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
