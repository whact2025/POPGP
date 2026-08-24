# VIA-000 R3 recovery-protocol independent re-review 21

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-21"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-21"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "3066adfce03997cbc357895f69aff3814fe37529"
baseline_commit: "96a8a084cf06b3f4ec4b37acdafa65bb9a072957"
prior_review_ref: "96a8a084cf06b3f4ec4b37acdafa65bb9a072957:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-20.md"
builder_response_ref: "3066adfce03997cbc357895f69aff3814fe37529:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-20-RESPONSE-1.md"
context_hash: "dfb9f55c0ef17437e25de8ab516387e6179ed92e"
context_hash_method: 'git rev-parse "3066adfce03997cbc357895f69aff3814fe37529^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-ENVELOPE.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-protocol.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-aggregator.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-20.md"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-20-RESPONSE-1.md"
  - "tests/unit/test_via000_r3_identity.py"
  - "scripts/check_viability_campaign.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository, GitHub run 32771982270 jobs/logs/API, and retained non-scientific artifact 9536553478; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-21 at exact handoff
  3066adfce03997cbc357895f69aff3814fe37529. Exact origin, remote head,
  sole-parent content ancestry, trees, review/response history, proof and receipt
  sources, every hosted job timeline/log, artifact API metadata, all seven retained
  files, frozen retained verification, focused regressions, parser closure, guidance,
  and campaign state were inspected. No implementation, campaign, lifecycle, or
  scientific state was changed. Operator and orchestrator are shared; session,
  worktree, and branch differ. Builder model is shared and external scientific
  validation is not claimed.
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
  CHANGES REQUESTED with one high-severity, fail-closed artifact-digest
  canonicalization blocker. RR20's inner-JSON byte defect is verified resolved and
  the six-cell containment architecture remains viable. All six exact producers and
  caches passed, aggregate job 97574418177 passed, retained artifact 9536553478
  contains exactly the frozen seven ordinary files, upload and redownload computed
  the same archive SHA-256, and the frozen aggregator independently accepted the
  downloaded bytes. Verifier job 97574603082 failed before invoking that aggregator
  only because actions/upload-artifact supplied bare lowercase 64-hex output while
  the verifier correctly required the API's canonical `sha256:<64hex>` form.

  Handoff 3066adfce03997cbc357895f69aff3814fe37529 has exact tree
  dfb9f55c0ef17437e25de8ab516387e6179ed92e and sole parent/content
  e5abbed1d77d934e01cc6b839cabcfea0c7c9e20 with tree
  249a62ae3425985995e84504ad7fc2c971bd3c14. Origin's campaign head matched.
  The handoff adds only the RR20 response and campaign/packet review metadata over
  the remediation content. Imported RR20 review Git-normalized SHA-256 is
  147c4f67571d9319b5454b9eb9216ca7ea96a902d96e221a67109075f069a1bd;
  its response is e68c6a6231c5d96f9653525f43be9dd999a5c776e0359836b973977ad0415704.

  Hosted push run 32771982270 attempt 1 was exact at the handoff. Windows producer
  jobs 97574174006/97574174281/97574174289 and Ubuntu producer jobs
  97574174224/97574174378/97574174702 passed every containment, envelope, digest,
  cache-save, and post-save check. Aggregate job 97574418177 required six green jobs
  and distinct digests, restored six exact hit/key-equal caches, canonical-validated
  all six outer envelopes and twelve inner JSON subjects, wrote exactly seven files,
  and uploaded artifact ID 9536553478 named
  via000-r3-containment-proof-retained. The upload log reports bare digest
  b791cdde77e43116ff341d4f70fdb9bee74cb440f7c7ca87d26598e270b25595.

  The artifact API reports exactly one unexpired artifact for the run, ID/name/size
  9536553478/via000-r3-containment-proof-retained/48018 and canonical digest
  sha256:b791cdde77e43116ff341d4f70fdb9bee74cb440f7c7ca87d26598e270b25595.
  The pinned download action identified that exact ID, size, and canonical expected
  digest, computed the identical bare hex over the downloaded archive, and succeeded
  with digest-mismatch:error. Its seven extracted files are ordinary non-link files;
  their exact SHA-256 values are 74dc4efe76b86800c042b9b37333e8f6986d9e86b52a4527389c835b370b8d11
  for aggregate.json, c241b76852193469e98441da90e8dd644edbd1cb458c9c787307d1d97c13e508,
  a0e185fffe29a5f6c9460ccb350ebae8291b2ace54f1d311934231eed0bf8fd3,
  e4c82bac903f91d52a287f130971876c79d646a71dbc88d74004f1b759fbf85d,
  a0dd95608d7607be03c4318d902314fa0f5178dd2823a5b81e4b9ebcaca30218,
  410e9316307b8067863b1a891e6ad0a8c5a549caa128972a4c196ca935093dba,
  and 040d02f43df360a4d37ab12b24ba669a413a6f8b1878a16accbe26963defd4cf
  for the six canonical cell envelopes in platform/stage order recorded by
  aggregate.json. The exact frozen aggregator returned zero on those downloaded
  files under the handoff's source/ref/workflow/run/attempt identity.

  The verifier environment proves the only mismatch: VIA000_ARTIFACT_ID was
  9536553478, URL exactly bound run 32771982270 and that ID, but
  VIA000_ARTIFACT_DIGEST was the bare upload-action output. The source exports the
  raw `steps.upload.outputs.artifact-digest` directly as a job output, then requires
  `^sha256:[0-9a-f]{64}$`. The resulting exception was exactly "retained artifact
  identity or digest is malformed" before the aggregator call. This is a bounded
  representation defect at a trusted action/workflow boundary, not an artifact-byte,
  cache, containment, or scientific failure.
findings:
  - id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:691-694,799-805,835-847 and corresponding containment-proof-workflow.yml receipt"
    evidence: |-
      The pinned upload action logged archive digest b791...b25595 and exported that
      bare value through `artifact-digest`; the aggregate job forwards it unchanged.
      The public artifact API and the pinned download action's expected metadata expose
      the identical digest as sha256:b791...b25595, and download recomputation matched.
      The verifier received correct artifact ID and URL plus the bare digest and failed
      its intentionally canonical prefixed grammar before executing the otherwise
      successful frozen aggregator. Independent execution of that exact aggregator on
      the downloaded seven files returned zero.
    finding: "The trusted aggregate job forwards a bare upload-action digest into a verifier contract that accepts only the canonical scheme-prefixed Actions API representation."
    failure_scenario: |-
      Every containment and byte-integrity control succeeds and the artifact is
      retained, but every verifier run deterministically rejects the valid bare action
      output before revalidating extracted bytes. Relaxing the verifier to accept both
      forms would create an unnecessary multi-encoding identity contract.
    consequence: "The required end-to-end hosted proof remains red, so merge and every signer, custodian, refreeze, activation, or holdout gate remain unauthorized."
    required_action: |-
      Add one trusted post-upload normalization step and make the aggregate job export
      only that step's values. Take `steps.upload.outputs.artifact-digest` as data;
      before any prefixing require an ordinal exact match to `[0-9a-f]{64}` with no
      whitespace, newline, uppercase, scheme, or other bytes. Construct exactly once
      `sha256:` plus that value. In the same step strictly validate the positive decimal
      artifact ID and exact URL formed from immutable repository/run ID/artifact ID,
      then write the canonical values to the protected GitHub output control file.
      Keep the verifier's single `^sha256:[0-9a-f]{64}$` grammar; do not accept bare,
      uppercase, whitespace-padded, alternate algorithm, or double-prefixed values.

      Preserve exact artifact name/ID/source SHA/ref/repository/workflow/run ID/attempt
      bindings. Require the canonical value to equal the Actions artifact API digest
      for that exact ID and run. Retain the pinned download action's fail-on-digest-
      mismatch archive-byte check and the frozen aggregator's exact seven extracted-
      file, inner-hash, 2x3 identity, and canonical-byte validation. Do not attempt to
      recompute the service ZIP digest from re-zipped extracted members; compare the
      service/archive digest through exact action/API metadata and validate extracted
      bytes through the frozen aggregate contract.

      Mirror the workflow receipt byte-for-byte and update its protocol/packet/manifest
      hashes. Add a focused grammar test where bare lowercase action output becomes
      exactly one prefixed canonical value and equals the prefixed API digest; reject
      uppercase, malformed length/hex, leading/trailing whitespace, embedded newline,
      already-prefixed input, double prefix, bare or uppercase API metadata, mismatched
      ID/name/run/attempt/source, and digest mismatch. Then rerun the exact proof-only
      2x3 workflow from the reviewed handoff successor. Acceptance requires six green
      producers/caches, green aggregate/upload, one exact seven-file artifact, and a
      green fresh redownload verifier. Do not reuse run 32771982270 as lifecycle proof.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001"
    description: "Prove strict one-way bare upload-action SHA-256 canonicalization, exact equality to prefixed Actions API/download metadata, malformed/alternate-encoding rejection, and a green exact-handoff six-cell aggregate/upload/redownload verifier."
    rationale: "The retained bytes are valid, but an untested representation mismatch keeps the authoritative hosted gate red; the fix must retain one canonical downstream identity rather than broaden acceptance."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "All hosted command boundaries executed.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-resolved, evidence: "Six producer tool identities and exact retained subjects pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Six contexts, aggregate identities, and retained bytes pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-resolved, evidence: "Six teardown/staging gates and retained post-quiescence subjects pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Exact retained proof exists and validates independently, but the required hosted verifier job is red.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-resolved, evidence: "Windows 3/3 export, cache, aggregate, and retain exact envelopes.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-resolved, evidence: "All six canonical envelopes survive cache, aggregate, upload, and download exactly.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-resolved, evidence: "Six distinct digest-bound cache saves and exact hit/key-equal restores pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-resolved, evidence: "Built-in control-plane steps execute.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001", outcome: verified-resolved, evidence: "Supported built-in pwsh executes all Windows steps.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR13-WINDOWS-LOW-IL-CHILD-INIT-001", outcome: superseded, evidence: "No-LUA low-integrity children execute.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", notes: ""}
  - {finding_id: "VIA000-R3-RR14-WINDOWS-LUA-TOKEN-PRODUCTION-001", outcome: verified-resolved, evidence: "Windows 3/3 pass production low-IL containment, teardown, and retained aggregation.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR15-POWERSHELL-PARSE-GATE-001", outcome: verified-resolved, evidence: "Scripts parse and execute through hosted proof.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-UBUNTU-EMPTY-ARRAY-CALLER-001", outcome: verified-resolved, evidence: "Ubuntu 3/3 aggregate and retain exact subjects.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR16-WINDOWS-EXPORT-ACL-001", outcome: verified-resolved, evidence: "Windows 3/3 pass exact descriptor checks through retention.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-OBSERVABILITY-001", outcome: verified-resolved, evidence: "Preserved exact diagnostic and production assertions.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001", outcome: verified-resolved, evidence: "Native descriptors pass Windows 3/3 through retained aggregation.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-resolved, evidence: "Windows 3/3 exact zstd identity and cache transport pass.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR20-WINDOWS-INNER-JSON-CANONICAL-BYTES-001", outcome: verified-resolved, evidence: "All twelve live inner JSON subjects pass aggregate canonical-byte validation and are retained.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "All hosted command boundaries executed.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: verified-satisfied, evidence: "Six producer tool manifests and retained subjects pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Six exact contexts and retained subjects pass.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: verified-satisfied, evidence: "Six post-quiescence attestations are retained and validate.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "The hosted redownload verifier remains red at digest representation precheck.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: verified-satisfied, evidence: "Windows 3/3 export and retain exact envelopes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six exact canonical envelopes survive through independent retained verification.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: verified-satisfied, evidence: "Six digest-bound caches, exact restores, and retained bytes pass independently.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001", outcome: verified-satisfied, evidence: "Built-in pwsh passes and produces retained exact subjects.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR13-WINDOWS-NATIVE-CHILD-FACTOR-MATRIX-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR14-NO-LUA-LOW-IL-PRODUCTION-001", outcome: verified-satisfied, evidence: "Windows 3/3 low-IL production subjects aggregate and retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR15-R3-POWERSHELL-PARSE-CLOSURE-001", outcome: verified-satisfied, evidence: "Eight source/receipt R3 PowerShell scripts parse and hosted execution passes.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-UBUNTU-CANONICAL-EMPTY-ARRAY-001", outcome: verified-satisfied, evidence: "Ubuntu 3/3 exact subjects aggregate and retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR16-WINDOWS-MEDIUM-EXPORT-BOUNDARY-001", outcome: verified-satisfied, evidence: "Windows 3/3 exact export descriptors aggregate and retain.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR17-WINDOWS-EXPORT-DESCRIPTOR-DIAGNOSTIC-001", outcome: verified-satisfied, evidence: "Preserved.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001", outcome: verified-satisfied, evidence: "Native descriptors pass all Windows cells and retained aggregation.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001", outcome: verified-satisfied, evidence: "Exact zstd/cache identity passes all Windows cells and retained aggregation.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR20-CANONICAL-INNER-JSON-BYTES-001", outcome: unresolved, evidence: "All writer/byte/2x3 artifact clauses pass, but its required hosted redownload verifier is red at a distinct digest-format precheck.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001", notes: ""}
predictions:
  experiment_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION"
  predicted_outcome: "Strictly validating the bare lowercase upload-action digest and prefixing it exactly once will make it equal the canonical API digest and allow the unchanged verifier plus frozen seven-file aggregator to pass."
  predicted_failure_mode: "Malformed or alternate raw output, API mismatch, archive mismatch, or extracted-byte/identity mutation rejects before acceptance; no lifecycle path runs."
  confidence_statement: "High confidence that this is a bounded deterministic control-plane representation defect and that architecture remains viable. No scientific prediction or execution occurred."
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Bounded CHANGES REQUESTED. Add strict one-way digest canonicalization at the trusted post-upload boundary, mirror hashes/receipt, add representation mutations, and rerun the exact proof-only 2x3 gate through a green redownload verifier. Run 32771982270 and artifact 9536553478 are scientifically/lifecycle inert and cannot authorize merge, signer/custodian work, refreeze, activation, or holdout."
```

## Verification ledger

- Exact handoff/content/tree/sole-parent/origin identities matched. Handoff scope is RR20 response plus campaign/packet review metadata. Workflow/receipt and aggregator/receipt are byte-identical with Git-normalized SHA-256 `a63f616626b5aae80876771c018433180e4c505afcdeccc19fd2f48530d085af` and `03a77df68c86477d6fb74d86b8aa540f6f96c2d8dce8ff26a7326e35d2f439c3` respectively.
- All eight job records and relevant logs for run `32771982270` were inspected. Six producer jobs and aggregate job `97574418177` were green. Verifier `97574603082` downloaded exact artifact `9536553478` with matching archive digest, then failed only its bare-versus-prefixed digest grammar before calling the aggregator.
- The Actions API reported exactly one artifact, the exact ID/name/run/head binding, size `48018`, and digest `sha256:b791cdde77e43116ff341d4f70fdb9bee74cb440f7c7ca87d26598e270b25595`. The artifact was independently downloaded; it contained exactly seven ordinary files and the frozen `--verify-retained` invocation returned zero under the exact source/ref/workflow/run/attempt identity.
- Focused RR10 cache, RR19 zstd, and RR20 canonical-inner-byte regressions passed 3/3 in 25.91 seconds. Review-guidance tests passed 9/9. Eight R3 source/receipt PowerShell scripts parsed with zero errors. Ruff passed on changed Python, campaign validator reported valid, and diff check passed. The complete R3/full suites, TeX, and actionlint were not rerun because this narrow artifact-only review changes no implementation and the decisive hosted representation failure plus independently valid retained bytes fully bound the verdict; actionlint was unavailable locally.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending. The proof artifact contains only synthetic non-scientific containment evidence (`no_campaign_execution`, `no_candidate_checkout`, `no_lifecycle_mutation`, `no_custody_access`, and `no_commitment_or_reveal` are all true). No R3 production tag, signer/key, lifecycle/refreeze, custody access, holdout, scientific execution, commitment, or reveal was created, accessed, performed, or authorized.
