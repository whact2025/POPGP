# VIA-000 R3 recovery-protocol independent re-review 11

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-11"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-11"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "d2ce792a71002310f9ef760962bcb1141ec99437"
baseline_commit: "05508ed54a3031ff1970cdaf8a2d917d06151421"
prior_review_ref: "05508ed54a3031ff1970cdaf8a2d917d06151421:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10.md"
builder_response_ref: "d2ce792a71002310f9ef760962bcb1141ec99437:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10-RESPONSE-1.md"
context_hash: "8973a54cf0bdc7f33becedbbb14f3af50066075e"
context_hash_method: 'git rev-parse "d2ce792a71002310f9ef760962bcb1141ec99437^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "tests/unit/test_via000_r3_identity.py"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10.md"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository and GitHub runs 32718921969/32720581939 logs, job metadata, artifact metadata, and cache metadata; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-11. Exact commit/tree/parent/origin,
  Git-normalized response/prior-review hashes, the supplied and actual hosted job
  identities/logs, artifact API results, cache API results, and frozen workflow and
  runner sources were independently inspected. No implementation or external state
  was changed. Operator/orchestrator are shared; session/worktree/branch differ.
  Builder model is shared and external validation is not claimed.
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
  CHANGES REQUESTED with one high-severity hosted control-plane blocker; the
  architecture remains viable. The supplied job IDs are not jobs of the requested
  run. Jobs 97405985825/97405986037/97405985752, 97405985916/97405985555/
  97405985828, 97406088331, and 97406139565 belong to run 32718921969 at ancestor
  bec02c5a3660a315b315466fe37532b9e71ebc15. In that run the three Windows
  digest steps failed with `Second path fragment must not be a drive or UNC name`,
  the three Ubuntu archive checks failed, and zero caches/artifacts were created.

  Run 32720581939 is the non-scientific push at the exact reviewed handoff. Its
  actual Windows jobs are 97410945232/97410945352/97410945403, Ubuntu jobs are
  97410945405/97410945431/97410945264, aggregate is 97411086226, and verifier is
  97411129608. All six containment, staged-envelope validation, digest, archive-tool,
  cache-preflight, and cache-save steps returned success. Ubuntu runner finalization
  logged three accepted digest outputs and the cache API retained three corresponding
  caches. Windows runner finalization logged no accepted digest output, all three
  cache-save actions warned that their staged path did not exist, and no Windows cache
  exists. Aggregate saw three valid Ubuntu digests and three empty Windows digests,
  rejected before restore or retained output, and verifier was skipped. Artifact count
  is zero. No campaign, lifecycle, signing, custody, holdout, scientific execution,
  commitment, or reveal action ran.

  The record does not distinguish (a) Windows environment-file/outer-shell output
  loss, (b) a path removed or hidden during cache preflight, or (c) Node path discovery
  failing on the protected newly created directory. The digest producer neither has a
  following same-job expression assertion nor uses the documented native cmd syntax;
  staging occurs before lookup-only cache preflight; and no trusted post-preflight
  path-existence check runs before the Node save action. Claiming any one root cause
  would therefore exceed the evidence. One bounded synthetic Windows-to-Ubuntu probe
  can separate them safely before another full six-cell proof.
findings:
  - id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:93-218,341-435,437-535; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:202-208,437-527"
    evidence: |-
      The exact-handoff Windows jobs returned success from their digest scripts, but
      runner 2.336.0 emitted no `Set output` line and aggregate received three empty
      Windows digest values. The same jobs validated the envelope before lookup-only
      preflight, which reported cache misses with digest-free trailing-hyphen keys;
      each later pinned cache/save action warned `Path(s) ... do(es) not exist` yet
      returned success. There is no intervening filesystem assertion. In contrast,
      the three exact-handoff Ubuntu jobs logged `Set output`, produced full-digest
      cache keys, and retained caches 6931994887/6931995394/6931994154. The artifact
      API returned zero. These observations prove two Windows control-plane failures
      but do not identify which of the three candidate causes is primary.
    finding: "The exact reviewed source still cannot transport or retain any Windows cell evidence, and its hosted record lacks the observations required to diagnose the failing Windows output and path boundaries without speculation."
    failure_scenario: "A Windows digest command-file write is not promoted to a step/job output, or the already-validated staged file disappears or becomes undiscoverable before cache save; the save action swallows the missing-path warning and the aggregate receives no Windows evidence."
    consequence: "The mandatory retained 2x3 proof remains absent, so RR4/RR6/RR8-RR10 cannot authorize the scientific lifecycle."
    required_action: |-
      Before another full matrix, run one minimal read-only, non-scientific diagnostic
      with one Windows producer and one dependent Ubuntu consumer. Prefer a temporary
      experiment branch/workflow, or a separately gated diagnostic-only path in the
      proof workflow. It must use fixed synthetic canonical bytes only and must not
      invoke containment candidate code, lifecycle code, repository writes, secrets,
      or production execution.

      First, after synthetic trusted teardown, use the literal native system cmd.exe
      with fixed script text and no interpolated untrusted value to emit a known
      lowercase 64-hex digest using the documented form
      `echo name=value>>%GITHUB_OUTPUT%`. A following same-job expression must check
      only strict length/regex and fail if absent; it must never print the value.
      GITHUB_OUTPUT must remain absent from every untrusted environment. This preserves
      the boundary because only trusted deterministic digest bytes cross after all
      untrusted descendants are dead.

      Second, perform the pinned cache lookup-only preflight before final staging.
      Only afterward, trusted code must create/copy exactly one bounded ordinary,
      single-link, non-reparse file at a fixed child path in an already checked-out,
      Git-verified, medium-integrity workspace directory. Do not overwrite tracked
      source. Recheck every ancestry component, destination absence, file type/link
      count, exact bytes/hash, and directory contents. A separate following trusted
      step must again prove path existence and hash before pinned cache/save. This
      order prevents preflight from being blamed for deleting final staged evidence
      and tests whether the Node action can discover an ordinary checked-out-root path.

      Use an exact fresh deterministic probe key containing repository/workflow/source,
      run ID/attempt, probe schema, Windows producer, and full digest; no restore
      prefixes and cross-OS mode enabled. Cache/save warnings are non-fatal, so only a
      dependent Ubuntu job's exact-key restore is success. That job must require exact
      primary/matched-key equality, reject missing/extra/link/corrupt bytes, and verify
      the fixed bytes/hash without logging content. A prior exact-key hit/collision
      must fail the producer preflight; a restored collision is harmless only if its
      canonical bytes match the full digest and bound identity.

      Record only pass/fail and the two path/output boundary outcomes. Treat the probe
      as diagnostic and non-authoritative, then remove or permanently gate it and rerun
      the exact six production cells. The final protocol still requires six distinct
      exact cache restores, canonical 2x3 validation, one retained seven-file artifact,
      and a green dependent redownload verifier.

      The narrow archive-tool correction is acceptable: it invokes fixed absolute
      tools, captures each exit code immediately after that tool, and validates each
      first-line banner independently. It passed in all six exact-handoff cells and
      needs no broader redesign.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001"
    description: "One Windows/one Ubuntu synthetic probe: native cmd digest environment-file write and same-job regex-only assertion; cache preflight before staging; trusted creation in an existing verified workspace directory; next-step path/hash assertion; pinned exact digest-bound save; dependent exact Ubuntu restore and byte/hash check; no content logs or candidate/lifecycle path. Then remove/gate the diagnostic and rerun the exact six-cell retained proof."
    rationale: "The current log proves empty Windows outputs and missing save paths but contains no observation between those boundaries; the bounded probe distinguishes environment-file loss, preflight removal, and Node path discovery without repeating the scientific or full production path."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Six cells executed, but no retained 2x3 artifact exists.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Blocked by RR11."}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Windows envelopes were not retained.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Blocked by RR11."}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: verified-resolved, evidence: "Safe non-scientific hosted path exists and ran.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Exact-handoff Windows save retained zero envelopes.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", notes: ""}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "Exact-handoff run retained no aggregate artifact.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", notes: ""}
  - {finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Ubuntu digest/cache transport worked; all three Windows outputs and caches were absent.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", notes: "Architecture remains viable pending bounded diagnosis and retained replay."}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Retained 2x3 evidence absent.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Windows envelopes were not retained.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted path ran but no complete artifact was retained.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Windows retained zero envelopes.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "No retained aggregate artifact exists.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", outcome: unresolved, evidence: "Only Ubuntu digest/cache transport succeeded.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR11-WINDOWS-CONTROL-PLANE-DIAGNOSTIC-001", notes: ""}
predictions: {experiment_id: "", predicted_outcome: "", predicted_failure_mode: "", confidence_statement: "No scientific execution occurred."}
recommendation: {approve: false, blocking_findings: 1, rationale: "Fixable CHANGES REQUESTED. Run the bounded diagnostic, implement the evidenced Windows correction, and rerun/retain/revalidate the exact six-cell proof; architecture remains viable."}
```

## Verification ledger

- Exact handoff `d2ce792a71002310f9ef760962bcb1141ec99437`, tree `8973a54cf0bdc7f33becedbbb14f3af50066075e`, sole parent/content `4c8fbeee32e45848e307d4c2b54c058c9072ff22`, content tree `9d30e9629db8c4965ccce7823b22da6b1c898237`, and origin branch head matched. Git-normalized response SHA-256 was `9c815dcc242c515bf00f740ca12e661da52fa8d680ab2a16c694ba6648db2667`; prior review SHA-256 was `88146bb6051b48e7fa1ba668eee7ca7a9f15212b5288aba914b636485b4f621b`.
- GitHub run/job API proved the supplied IDs belong to run `32718921969` at `bec02c5a3660a315b315466fe37532b9e71ebc15`, while requested run `32720581939` is exact handoff and uses jobs `97410945232`, `97410945352`, `97410945403`, `97410945405`, `97410945431`, `97410945264`, `97411086226`, and `97411129608`.
- Exact-handoff logs and metadata proved six green producer jobs; three accepted Ubuntu digest outputs; three empty Windows job outputs; three Windows missing-path save warnings; aggregate fail-closed; verifier skipped; and artifact count zero. Cache API proved exactly three Ubuntu caches (`6931994887`, `6931995394`, `6931994154`) and no Windows caches for that run.
- Frozen source inspection proved that current staging precedes lookup-only preflight, the Windows digest producer has no following expression-only assertion, and no post-preflight filesystem assertion precedes cache/save. Therefore environment-file syntax, preflight path effects, and Node discovery remain distinguishable hypotheses rather than findings.
- The corrected immediate GNU tar/Zstandard exit-code and banner checks passed in all six exact-handoff cells and are accepted as narrowly correct.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending. No signer, key, tag, refreeze, custody, lifecycle, holdout, scientific execution, commitment, or reveal was performed or authorized.
