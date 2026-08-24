# VIA-000 R3 recovery-protocol independent re-review 25

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-25"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-25"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "c519e4f5b361d1a5ec01056ac302fe832714ef0c"
baseline_commit: "b036813954c842afd945420e847423b75e9e9632"
prior_review_ref: "b036813954c842afd945420e847423b75e9e9632:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24.md"
builder_response_ref: "c519e4f5b361d1a5ec01056ac302fe832714ef0c:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24-RESPONSE-1.md"
context_hash: "783265daa602da24790a65ba9e01c16007edf012"
context_hash_method: 'git rev-parse "c519e4f5b361d1a5ec01056ac302fe832714ef0c^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - ".github/workflows/via000-r3-ubuntu-pwsh-identity-diagnostic.yml at 768167269def1d6689ee38b0051127f51d670579"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24-RESPONSE-1.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "tests/unit/test_viability_raw_evidence_contract.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository, exact GitHub proof run 32790475904 jobs/logs/API/artifact 9542964254, RR23 diagnostic run 32783816053, and generic CI run 32790475972; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-25 at exact handoff
  c519e4f5b361d1a5ec01056ac302fe832714ef0c. Exact origin, remote head,
  sole-parent content ancestry, trees, imported review/response bytes, workflow and
  receipt sources, every proof job/API/log, retained artifact API and seven extracted
  files, frozen retained verification, RR23 diagnostic source/log, terminal generic
  CI, focused regressions, parser closure, guidance, and campaign state were
  inspected. No implementation, campaign, lifecycle, or scientific state was
  changed. Operator and orchestrator are shared; session, worktree, and branch
  differ. Builder model is shared and external scientific validation is not claimed.
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
  CHANGES REQUESTED with one high-severity, fail-closed normalizer observability
  blocker. Run 32790475904 proves all six producers/caches, aggregation, and upload,
  but the new production compound predicate rejects before PATH sanitization or
  canonical output emission. Its single exception does not identify the failing
  term. A direct correction is not justified; one minimal Ubuntu-only diagnostic is
  required before production is changed again.

  Handoff c519e4f5b361d1a5ec01056ac302fe832714ef0c has exact tree
  783265daa602da24790a65ba9e01c16007edf012 and sole parent/content
  21deabc877e6a13a68f8973c333e2a98669680f3 with tree
  af06d574d367a1135bc7521720b28a1508848d25. Origin's campaign head matched.
  The handoff adds only the RR24 response and campaign/packet review metadata.
  Imported RR24 review commit b036813954c842afd945420e847423b75e9e9632
  preserves normalized SHA-256
  a092f15504e591f35377734a7873d27cae6ae2449e927da1203982d005a3384b;
  response SHA-256 is
  3bb61fbd833c483b5062983129184aa792b76cb35371082db3d852959cf650a9.
  Workflow/receipt and protocol/receipt are byte-identical with normalized SHA-256
  f6ba5c352564764d7b0537157fcd0d78594f43fd9e79164fd513c5ddc0d3d3d7
  and 75a79620a357361aabff256a66813e353d519b0f9fb3c9c0a65b29d6758cdd79.

  RR23 run 32783816053 directly logged MainModule
  `/opt/microsoft/powershell/7/pwsh`, PSHOME `/opt/microsoft/powershell/7`, version
  `7.6.5`, live initial PATH `/opt/microsoft/powershell/7:/usr/bin:/bin`, argument
  count six, and arguments 0-4 exactly as literal executable, `-NoLogo`,
  `-NoProfile`, `-NonInteractive`, and `-File`. Its `pass` value for version tested
  only a 7.x semantic-version regex, although the separately logged observed value
  was exactly 7.6.5. Critically, it replaced argument 5 with `<runner-script>` before
  logging and its pass expression checked only count and arguments 0-4. It never
  logged or validated the real script directory, basename, extension, case, or UUID
  grammar. Sealed RR24 nevertheless prescribed an `expected trusted runner temporary
  form`; the remediation concretized that into two newly unobserved predicates:
  exact parent equality with RUNNER_TEMP and a lowercase UUID-plus-`.ps1` regex.

  Both runs used runner 2.336.0, Ubuntu 24.04.4 image 20260816.277.1, the same literal
  shell declaration, and logged pre-launch PATH `/usr/bin:/bin`. Thus argument 5 is
  the evidence-weighted leading suspect, but current run 32790475904 exposes only the
  compound exception. It cannot distinguish directory equality from filename regex
  or formally re-observe the other runtime values. Guessing which new constraint to
  relax would repeat the unsupported inference that produced this failure.

  Proof run 32790475904 is exact at the handoff. Ubuntu pdf/mutation/candidate jobs
  97630871827/97630871997/97630872028 and Windows candidate/mutation/pdf jobs
  97630872037/97630872038/97630872064 passed. Aggregate 97631014006 restored and
  validated all six exact caches, produced the seven-file proof, and uploaded exact
  artifact 9542964254, size 48018, API digest
  sha256:ddaf3f2fcd147f68ecf8c3c46d100bb4c181c7009fcbdc2241ac69484079c17e.
  It then failed only the compound normalizer predicate; verifier 97631143954 skipped.
  Independent download contained exactly seven ordinary non-link files. Frozen
  `--verify-retained` returned zero for the exact source/ref/workflow/run/attempt.
  Aggregate SHA-256 is 002bd93436be81dd02318582bf70e2fbe0a9be267dc47c8e5f9c62ff1ef2a546;
  six envelope SHA-256 values are
  f53df42ad50ea34c3ae2bc5ff14f95422b29d5abfdac8cd609cc37e6077e57cd,
  2abc8328ad36f6b7d2832b94d11096ab16d864827427bd8b4da4d438ff8d88eb,
  6f477cefd2568490130c8dab2962c5174bc847addc18a261edba7d935ccd330d,
  0b81bc288b6548163c865641403db9e6f1c5d99d314109ef647ef81a35167381,
  37e686cc33ce3c0f2b4531f079d4eb774b46c3b53c7f5356d835678db53d1ee4,
  and df77f8cdb84a0151c39c14989ab77eed298f79f80ed5ae5ac77bcfa8fb539b9e.
  The aggregate explicitly records non-scientific and every no-campaign/no-candidate/
  no-lifecycle/no-custody/no-commitment-or-reveal assertion.

  Generic CI run 32790475972/job 97630871498 completed success independently at the
  exact handoff: Ruff, manuscript validation, 448 passed/2 skipped tests, documented
  example regeneration, and committed validation/visual contract checks all passed.
  It produced zero artifacts because failure retention correctly skipped. This is
  positive evidence for the shared validator remediation and repository baseline,
  but generic CI does not execute the containment normalizer and cannot resolve or
  authorize around the separate proof failure.
findings:
  - id: "VIA000-R3-RR25-NORMALIZER-ARGUMENT5-OBSERVABILITY-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:807-880 and mirrored containment-proof-workflow.yml receipt"
    evidence: |-
      Production combines MainModule, PSHOME, exact 7.6.5, exact initial PATH, argc,
      argv 0-4, runner-script parent, and runner-script basename regex into one throw.
      The hosted failure reports only that throw. RR23 independently observed every
      term except the real argument-5 value and the two argument-5 predicates; it
      normalized that operand to `<runner-script>` and excluded it from `pass`.
      Same runner/image evidence makes argument 5 likely but not proven. The new run
      did not log any individual production observation.
    finding: "RR24 production added two runner-script predicates that were not evidenced by RR23 and retained a compound failure that cannot identify which current runtime predicate rejected."
    failure_scenario: |-
      A guessed relaxation may accept an unintended runner-script location/name or
      may leave the actual failing version/PATH/argv condition unchanged. Repeated
      full six-cell runs then generate valid retained artifacts but cannot reach the
      verifier.
    consequence: "The hosted proof gate remains red; merge and all signer, custodian, refreeze, activation, and holdout actions remain unauthorized."
    required_action: |-
      Before editing production, run exactly one minimal Ubuntu-24.04 diagnostic on a
      fresh experiment branch with the same literal shell and pre-launch PATH, minimal
      read-only permissions, no checkout, secrets, cache, artifact, candidate,
      campaign, lifecycle, custody, or scientific path. Do not redact argument 5.

      Log nonsecret values and independent booleans for MainModule, PSHOME, exact
      PSVersion, initial live PATH, argc, each argv 0-4, raw argument-5 full-path
      status, normalized argument-5 directory, exact RUNNER_TEMP, directory equality,
      basename, extension, case, UUID-regex result, and ordinary-file/non-link/link-
      count metadata. Guard every index and path operation so one mismatch cannot
      prevent later observations. Log only path/metadata, never script contents.
      Record the exact runner/image identities and test ordered PATH assignment to
      `/usr/bin:/bin`. Emit one final vector of predicate booleans and then fail if
      any required observation differs. No GITHUB_OUTPUT or retained artifact is
      needed for this diagnostic.

      Apply a direct production correction only to the term the diagnostic disproves.
      If the runner-script spelling differs, bind the smallest observed invariant
      that proves one absolute ordinary non-link runner-created script inside exact
      RUNNER_TEMP; do not require an invented UUID/case/extension grammar. If version,
      initial PATH, executable/PSHOME, argc, or argv 0-4 differs, update only after
      separately binding the exact trusted image/tool fact and fail closed on drift.
      Split production checks into safe named assertions or log the same nonsecret
      boolean vector before throwing so future failures remain attributable.

      Preserve immediate PATH sanitization and all artifact ID/digest/URL,
      GITHUB_OUTPUT identity/link/byte-growth, receipt/hash, and redownload verifier
      gates. Add a regression drawn from the actual observed argument-5 value plus
      negative path-escape/link/extra-argument controls. Then rerun the exact proof-
      only 2x3 workflow from a new handoff. Acceptance requires six green producers,
      green aggregate/normalizer, exact canonical outputs, one seven-file artifact,
      and a fresh green redownload verifier.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001"
    description: "Independently log every current compound normalizer predicate without redacting argv[5], identify its exact failing term, bind only the observed safe runner-script/runtime invariant, and pass a fresh exact-handoff 2x3 artifact plus redownload verifier."
    rationale: "RR23 proved arguments 0-4 but not the script operand; the new compound failure cannot safely select a direct correction."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Hosted producer boundaries pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-resolved, evidence: "Six producer identities and retained subjects pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Six contexts and retained bytes pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-resolved, evidence: "Six post-quiescence subjects pass retained verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Retained proof validates, but hosted normalizer/verifier is red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR25-NORMALIZER-ARGUMENT5-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 export/cache/retain exact envelopes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six canonical envelopes survive transport and verification.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six digest-bound cache saves/restores pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "Supported Windows and Ubuntu literal shells execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "No-LUA low-integrity children execute.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: verified-resolved, evidence: "Windows 3/3 low-IL containment and retention pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "Eight scripts parse; hosted workflow reaches normalizer.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu 3/3 subjects aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: verified-resolved, evidence: "Windows 3/3 export boundaries retain exact evidence.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Preserved.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", outcome: verified-resolved, evidence: "Native descriptors pass all Windows cells.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-resolved, evidence: "Windows 3/3 exact zstd/cache identity passes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", outcome: verified-resolved, evidence: "All live inner JSON subjects aggregate and retain.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Production normalization/verifier remain blocked.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR25-NORMALIZER-ARGUMENT5-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: unresolved, evidence: "Literal launcher passes, but current compound identity remains red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR25-NORMALIZER-ARGUMENT5-OBSERVABILITY-001", notes: ""}
  - {finding_id: "VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-OBSERVABILITY-001", outcome: verified-resolved, evidence: "RR23 isolated its original PATH mismatch but did not observe argument 5.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR24-UBUNTU-PWSH-INPROCESS-PATH-001", outcome: unresolved, evidence: "PATH remediation is source-correct, but added unobserved argument-5 predicates and hosted normalizer remains red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR25-NORMALIZER-ARGUMENT5-OBSERVABILITY-001", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Hosted producer boundaries pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-satisfied, evidence: "Six producer identities pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Six contexts pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-satisfied, evidence: "Six post-quiescence attestations validate.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted normalizer/verifier remains red.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-satisfied, evidence: "Windows 3/3 export and retain exact envelopes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six canonical envelopes pass retained verification.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six exact caches and retained bytes pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: verified-satisfied, evidence: "Supported Windows and Ubuntu literal shells execute.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: verified-satisfied, evidence: "Windows 3/3 low-IL subjects retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Eight scripts parse.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu 3/3 subjects aggregate and retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: verified-satisfied, evidence: "Windows 3/3 export boundaries retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", outcome: verified-satisfied, evidence: "Native descriptors pass all Windows cells.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-satisfied, evidence: "Exact zstd/cache identity passes all Windows cells.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", outcome: verified-satisfied, evidence: "Live writers and retained bytes pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", outcome: unresolved, evidence: "Production normalization/verifier remain blocked.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001", outcome: unresolved, evidence: "Literal launcher passes, but compound identity remains red.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR23-UBUNTU-PWSH-IDENTITY-DIAGNOSTIC-001", outcome: superseded, evidence: "It isolated the original PATH mismatch but redacted and did not validate argument 5.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001", outcome: unresolved, evidence: "Source tests pass, but production compound identity rejects before sanitization/output.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC-001", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR25-NORMALIZER-ARGUMENT5-DIAGNOSTIC"
  predicted_outcome: "The one-job diagnostic will reconfirm the previously observed runtime terms and identify exactly which runner-script directory or basename predicate differs."
  predicted_failure_mode: "Any runtime or script-operand mismatch is retained as its own boolean without suppressing later observations; the job has no lifecycle or scientific path."
  confidence_statement: "High confidence that the architecture remains viable and that one bounded diagnostic will identify the current overconstraint. Argument 5 is the leading inference, not a proven current value. No scientific prediction or execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. Run one minimal Ubuntu observability diagnostic, correct only its disproven production term, preserve PATH sanitization and every artifact/verifier gate, add evidence-derived negative controls, and rerun the exact 2x3 proof through a green redownload verifier. Generic CI is independently green but does not authorize around the proof failure. Merge and all lifecycle gates remain unauthorized; architecture remains viable."
```

## Verification ledger

- Exact handoff/content/import/tree/remote identities and artifact-only history matched. The handoff contains only the RR24 response and campaign/packet review metadata over exact remediation content.
- All proof jobs/logs/API and artifact metadata for run `32790475904` were inspected. Six producers passed, aggregate validated/uploaded, only the compound normalizer failed, and verifier skipped. Exact artifact `9542964254` independently verifies as seven ordinary files under the frozen aggregator.
- RR23 diagnostic source and run log were replayed predicate-by-predicate. It proves MainModule, PSHOME, observed 7.6.5, initial PATH, argc, and argv 0-4; it does not prove the new argv[5] directory or basename grammar.
- Generic CI `32790475972` reached terminal success: Ruff, manuscript check, 448 passed/2 skipped tests, six example regenerations, and validation/visual contracts passed. This supports the validator baseline only.
- Focused RR21/RR22/RR24 regressions passed 3/3. Review guidance passed 9/9. Ruff passed changed Python, eight R3 source/receipt PowerShell scripts parsed with zero errors, campaign validator reported valid, source/receipt equality and diff checks passed. Full local suites, TeX, and actionlint were not rerun because terminal generic CI and the decisive hosted normalizer failure bind this narrow review.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending. No tag, signer/key, custody access, lifecycle/refreeze, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
