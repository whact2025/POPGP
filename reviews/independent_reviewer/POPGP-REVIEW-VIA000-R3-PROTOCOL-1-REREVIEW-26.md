# VIA-000 R3 recovery-protocol independent re-review 26

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-26"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-26"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "a0eb47d2405c17d8f593823f5c9bb7566b59936a"
baseline_commit: "216f31bf75fb5b16e492a73546e3e1632b5590d0"
prior_review_ref: "216f31bf75fb5b16e492a73546e3e1632b5590d0:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-25.md"
builder_response_ref: "c519e4f5b361d1a5ec01056ac302fe832714ef0c:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24-RESPONSE-1.md"
context_hash: "05c2434aee5e882ad51d86879fd550a502d6238b"
context_hash_method: 'git rev-parse "a0eb47d2405c17d8f593823f5c9bb7566b59936a^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-ubuntu-pwsh-compound-diagnostic.yml"
  - ".github/workflows/via000-r3-containment-proof.yml at c519e4f5b361d1a5ec01056ac302fe832714ef0c"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml at c519e4f5b361d1a5ec01056ac302fe832714ef0c"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-25.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24-RESPONSE-1.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and exact GitHub diagnostic run 32791645412/job 97634217441 plus prior public proof and review evidence; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-26 at exact experiment commit
  a0eb47d2405c17d8f593823f5c9bb7566b59936a. Exact origin, experiment
  branch/commit/tree, parent chain through sealed RR25 and campaign handoff, sole
  workflow diff, full workflow/log/API evidence, every retained observation,
  zero-artifact result, production predicate, review history, YAML/schema/parser
  checks, guidance, and campaign state were inspected. No implementation, campaign,
  lifecycle, or scientific state was changed. Operator and orchestrator are shared;
  session, worktree, and branch differ. Builder model is shared and external
  scientific validation is not claimed.
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
  CHANGES REQUESTED with one high-severity, exactly isolated runner-script suffix
  blocker. The RR25 diagnostic is decisive: of nineteen pre-summary observations,
  exactly one fails--the invented lowercase UUID-plus-`.ps1` predicate. The real
  argv[5] is the extensionless
  `/home/runner/work/_temp/e55a009e-9f76-4952-89e2-d236f894e8c8`.
  Its parent equals exact RUNNER_TEMP; its basename is a lowercase canonical UUID;
  and it is an ordinary non-reparse, single-link, runner-UID/GID-owned mode-0644
  regular file. Every other observation passes. A direct production correction is
  now evidence-backed and no further diagnostic is necessary.

  Experiment commit a0eb47d2405c17d8f593823f5c9bb7566b59936a has exact tree
  05c2434aee5e882ad51d86879fd550a502d6238b and sole parent RR25 review
  216f31bf75fb5b16e492a73546e3e1632b5590d0, whose sole parent is campaign handoff
  c519e4f5b361d1a5ec01056ac302fe832714ef0c. All three remote refs matched.
  The experiment adds only `.github/workflows/via000-r3-ubuntu-pwsh-compound-
  diagnostic.yml`; Git-normalized SHA-256 is
  71ab14420daa53a805979c577949add0be7171180a1222b60635542cc1c3a406.
  RR25 normalized artifact SHA-256 is
  c3ec46019411b97cb0438d33fb6516950234749a9d7bcec4fc10bbf308de9f03.

  Run 32791645412 attempt 1 and sole job 97634217441 are exact at the experiment
  commit. Runner 2.336.0 on Ubuntu 24.04 image 20260816.277.1 invoked the exact
  literal `/opt/microsoft/powershell/7/pwsh -NoLogo -NoProfile -NonInteractive
  -File {0}` shell. MainModule, PSHOME, exact 7.6.5, entry PATH, argc, argv0-4,
  argv5 nonempty/full-path, RUNNER_TEMP full-path, argv5 parent, basename nonempty,
  file metadata, hosted runner identity, and ordered sanitized PATH all logged
  `pass:true`. The exact file metadata are System.IO.FileInfo, reparse false, stat
  regular file, link count 1, owner UID/GID 1001 matching runner UID/GID 1001, and
  mode 644. Only `argv5_lowercase_uuid_ps1_regex` logs `pass:false`. The final vector
  reports `failed_current_predicates:["argv5_lowercase_uuid_ps1"]`, metadata true,
  sanitization true, and observation_count 19, then intentionally exits one. The API
  reports zero artifacts. The workflow has empty permissions and no checkout,
  cache, artifact, secret, candidate, campaign, custody, lifecycle, holdout,
  commitment, reveal, or scientific path.

  Requiring an extensionless lowercase UUID basename, exact RUNNER_TEMP parent,
  ordinary/non-reparse/single-link file identity, runner UID/GID ownership, and
  exact mode 0644 is evidence-backed for this reviewed runner/image. It is stricter
  and safer than merely deleting `.ps1` from an otherwise unverified operand. Future
  runner drift will fail closed and require review rather than broaden acceptance.
findings:
  - id: "VIA000-R3-RR26-RUNNER-SCRIPT-SUFFIX-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:823-836 at c519e4f5b361d1a5ec01056ac302fe832714ef0c and mirrored receipt"
    evidence: |-
      The diagnostic logs all current production terms independently. Its final
      failed predicate list contains only `argv5_lowercase_uuid_ps1`; actual basename
      is extensionless UUID `e55a009e-9f76-4952-89e2-d236f894e8c8`. Parent, full
      path, file type, link state, owner UID/GID, mode, and all other runtime/argv/PATH
      terms pass. No competing failed predicate remains.
    finding: "Production requires a `.ps1` suffix that GitHub runner 2.336.0 does not place on the literal `-File {0}` temporary script."
    failure_scenario: |-
      Six valid proof cells and an exact retained aggregate reach the normalizer, but
      the invented suffix rejects before artifact identity outputs and the redownload
      verifier. Removing all script identity checks would instead weaken the trusted
      runner-control boundary.
    consequence: "The exact hosted proof gate remains red until the evidence-backed extensionless script identity is implemented and replayed."
    required_action: |-
      Make one bounded campaign amendment to production workflow and exact receipt.
      Replace only the basename regex with the observed extensionless lowercase UUID
      grammar `^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`.
      Retain exact literal shell, MainModule, PSHOME, version 7.6.5, initial PATH,
      argc, argv0-4, absolute argv5, and exact parent-equals-RUNNER_TEMP checks.

      Before reading raw artifact outputs, keep the immediate PATH reset to
      `/usr/bin:/bin` and reassert it. Under that sanitized PATH, add the diagnostic's
      evidence-backed argv5 identity checks using .NET plus literal `/usr/bin/stat`
      and `/usr/bin/id`: System.IO.FileInfo, no reparse point, stat regular file,
      link count exactly one, owner UID and GID exactly equal current runner UID/GID,
      and mode exactly 644. Reassert sanitized PATH immediately before and after the
      absolute tools. Recheck the full path, parent, extensionless UUID basename, and
      file metadata immediately before GITHUB_OUTPUT append and after the append;
      reject any change. Do not read or hash the temporary script contents.

      Preserve every raw ID/lowercase digest/URL/repository/run check, single-prefix
      normalization, GITHUB_OUTPUT ordinary/non-reparse/single-link/exact-growth
      check, source/receipt/hash binding, and redownload verifier. Delete the
      diagnostic workflow when importing the amendment so it cannot become an
      alternate authority.

      Update the RR24/RR25 regression with the exact extensionless positive fixture
      and reject `.ps1` or any suffix, uppercase/non-UUID basename, alternate or
      nested parent, relative/path-escape operand, missing/nonregular/reparse file,
      hardlink count above one, wrong UID/GID, non-0644 mode, extra/reordered argv,
      and source/receipt divergence. No further diagnostic is required. Rerun the
      exact proof-only 2x3 workflow from the new handoff; require six green producers,
      green aggregate/normalizer, canonical outputs, one exact seven-file artifact,
      and a fresh green redownload verifier.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR26-EXTENSIONLESS-RUNNER-SCRIPT-IDENTITY-001"
    description: "Accept only the observed extensionless lowercase UUID runner script in exact RUNNER_TEMP with ordinary/non-reparse/single-link/runner-owned/0644 metadata, reject every path/name/link/owner/mode/argv variant, and pass a fresh exact-handoff 2x3 artifact plus redownload verifier."
    rationale: "The live diagnostic isolates `.ps1` as the sole false assumption and supplies a stronger complete script-object identity to retain."
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
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted normalizer/verifier remains red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR26-RUNNER-SCRIPT-SUFFIX-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 export/cache/retain exact envelopes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six canonical envelopes survive transport and verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six digest-bound cache saves/restores pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
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
  - {finding_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Production normalizer/verifier still red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR26-RUNNER-SCRIPT-SUFFIX-001", notes: ""}
  - {finding_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: unresolved, evidence: "Literal launcher passes; suffix blocks completion.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR26-RUNNER-SCRIPT-SUFFIX-001", notes: ""}
  - {finding_id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Original PATH mismatch isolated.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR24-UBUNTU-PWSH-INPROCESS-PATH-001", outcome: unresolved, evidence: "PATH terms pass; invented suffix remains.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR26-RUNNER-SCRIPT-SUFFIX-001", notes: ""}
  - {finding_id: "VIA000-R3-RR25-NORMALIZER-ARGUMENT5-OBSERVABILITY-001", outcome: verified-resolved, evidence: "All terms and unredacted argv5 are logged; suffix is sole mismatch.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
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
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted verifier remains red.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR26-EXTENSIONLESS-RUNNER-SCRIPT-IDENTITY-001", notes: ""}
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
  - {requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Normalizer/verifier remain blocked.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR26-EXTENSIONLESS-RUNNER-SCRIPT-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: unresolved, evidence: "Literal launcher passes; suffix blocks completion.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR26-EXTENSIONLESS-RUNNER-SCRIPT-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC-001", outcome: superseded, evidence: "Original diagnostic isolated PATH but redacted argv5.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001", outcome: unresolved, evidence: "PATH terms pass; suffix predicate rejects.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR26-EXTENSIONLESS-RUNNER-SCRIPT-IDENTITY-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", outcome: superseded, evidence: "All diagnostic terms executed and isolate suffix; final production proof remains pending under RR26 test.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR26-EXTENSIONLESS-RUNNER-SCRIPT-IDENTITY-001", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR26-EXTENSIONLESS-RUNNER-SCRIPT-IDENTITY"
  predicted_outcome: "The extensionless UUID and exact metadata checks will pass, artifact outputs will normalize, and the fresh redownload verifier will accept the exact seven retained files."
  predicted_failure_mode: "Any runner-script path/name/object/owner/mode drift or any preserved runtime/artifact mismatch rejects before acceptance; no lifecycle path runs."
  confidence_statement: "High confidence that the diagnostic identifies the sole bounded defect and that architecture remains viable. No further diagnostic is needed. No scientific prediction or execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. The next authorized action is only the exact extensionless-script production amendment, evidence-derived negative test, receipt/hash update, diagnostic removal, and fresh proof-only 2x3 replay through a green redownload verifier. Merge and every signer/custodian/refreeze/activation/holdout gate remain unauthorized. Architecture remains viable."
```

## Verification ledger

- Exact experiment commit/tree/sole-parent chain and all three remote refs matched. The experiment adds exactly one diagnostic workflow; no campaign file is changed.
- Exact run/job/API and full hosted log were inspected. Nineteen pre-summary observations executed: eighteen passed, only the `.ps1` regex failed, and the final vector named exactly that predicate. The expected intentional job failure and zero artifacts match source.
- The extensionless argv5 path and all file metadata above are directly logged. Exact PATH sanitization also passed. No inference is substituted for the retained values.
- Diagnostic YAML parsed, review-guidance passed 9/9, eight R3 source/receipt PowerShell scripts parsed with zero errors, campaign validator reported valid, and experiment diff check passed. Broad suites, TeX, Ruff, and actionlint were not rerun because the experiment adds only YAML and the live diagnostic is decisive.
- Campaign remains drafted, `holdout_started: false`, unrevealed, and pending. No tag, signer/key, custody access, lifecycle/refreeze, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
