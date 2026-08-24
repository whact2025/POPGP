# VIA-000 R3 recovery-protocol independent re-review 6

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-6"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-6"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "032611faafd2c3a3288b987c95d78fa5a03381d5"
baseline_commit: "7596948eab9b339cdb97332cd1ee047d40adeb27"
prior_review_ref: "7596948eab9b339cdb97332cd1ee047d40adeb27:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-5.md"
builder_response_ref: "032611faafd2c3a3288b987c95d78fa5a03381d5:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-5-RESPONSE-1.md"
context_hash: "45aefe8c3f0411f3c4fc883d4d27d1b758872ccc"
context_hash_method: 'git rev-parse "032611faafd2c3a3288b987c95d78fa5a03381d5^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-protocol.yml"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/PROJECT_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-5-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-5.md"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/PROTOCOL_MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/authorization/VIA-000-AUTHORIZED-SIGNERS"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/assembler-protocol.py"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_review_guidance.py"
  - "tests/unit/test_via000_r2_assembler.py"
  - "tests/unit/test_via000_r3_identity.py"
  - "tests/unit/test_viability_raw_evidence_contract.py"
access_level: "public-repository-only plus local Windows execution; no custody, reveal, external invalid-package, signer-key, tag, hosted-run, or handoff-memo access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-6 on dedicated branch
  review/via000-r3-protocol-rereview-6. Exact content/handoff commits and trees,
  sole parent, origin head, Git-normalized RR5 response SHA-256, complete review
  ancestry, and prior artifact immutability were checked first. Builder tests and
  response claims were hypotheses. No implementation, protocol, packet, signer,
  ref, lifecycle, evidence, commitment, custody, result, or reveal bytes were changed.
  Operator and orchestrator are shared; session/worktree/branch are distinct. Builder
  model is unknown, so model separation and external scientific validation are not claimed.
independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "unknown"
  builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false
hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
summary: |-
  CHANGES REQUESTED with two critical blockers. The six fresh hosted-job topology is
  real: Ubuntu and Windows each have separate candidate, PDF, and mutation matrix
  jobs; TeX exists only in PDF jobs; all six stage summaries/manifests are separately
  attested; the assembler requires the exact stage/platform set and reconciles run,
  attempt, repository, workflow, source, candidate, authorization, evidence, tool,
  and attestation identity. Environment scrubbing, TeX no-shell-escape, complete TeX
  closure before/after, undeclared-file rejection, failure cleanup, and success-only
  uploads materially resolve RR5's cross-stage execution-context finding.

  The response nevertheless explicitly leaves RR4's intra-stage same-path check/use
  race unresolved and blocking. No protocol may be approved while its accepted
  critical predecessor remains unresolved.

  A distinct post-validation race is also candidate reachable. The runner waits only
  for each direct child process and has no descendant-tree termination or quiescence
  proof. It removes RUNNER_TEMP from child environments but sets TEMP to the live
  stage workspace's temp directory, so candidate/mutation code can derive and access
  sibling evidence paths. A detached descendant can return a clean direct exit, wait
  beyond repository/environment/TeX closure checks and schema validation, then rewrite
  evidence, stage-summary, or evidence-manifest before the later attestation action.
  The builder's background-watcher test only inserts a static undeclared file into a
  completed fixture; it never starts a descendant or races production validation and
  attestation. Six VMs prevent cross-stage state sharing but do not close this same-job
  post-validation window.

  Focused 46/46, Ruff, response schema, TeX, and campaign validation pass. R2 and all
  prior review artifacts are immutable. R3 remains drafted, holdout-false, unrevealed,
  pending, and signer-blocked; no key/tag/run/output/commitment exists. No merge,
  signer amendment, custodian carry-forward, refreeze, activation, preregistration,
  holdout, assembly, reveal, or scientific claim is authorized.
findings:
  - id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001"
    severity: critical
    category: code
    location: "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1:115-194,284-309,448-536,584-738; .github/workflows/via000-r3-protocol.yml:386-519,521-627,627-669; tests/unit/test_via000_r3_identity.py:1821-1857"
    evidence: |-
      `Invoke-RetainedCommand` starts one Process, reads its streams, and calls only
      `$process.WaitForExit()`. Production source contains no process-tree kill,
      descendant enumeration, job object/cgroup, or post-child quiescence gate.
      It removes RUNNER_TEMP but assigns TEMP/TMP to `$stageTemp`, located directly
      under the live `$WorkspaceRoot`; arbitrary tested Python can compute the parent
      and address `evidence/`. Candidate pytest/generators and mutation subprocesses
      therefore have both execution and a discoverable target.

      On success the runner writes final status, evidence manifest, and stage summary
      and returns. The workflow validates the summary in that run step, then executes
      `actions/attest` in a separate later step against the still-live two files. A
      descendant that outlives its direct parent can wait for those files and replace
      them after the last local check but before attestation. The failure cleanup does
      not run on nominal success and upload follows attestation.

      The sole new negative control named `background-watcher-marker.txt` does not
      execute the runner or spawn background work. It edits an already constructed
      assembler fixture, restores one manifest, adds one undeclared file, and observes
      the expected undeclared-file rejection. It cannot falsify descendant survival,
      timing after schema validation, or attestation of replaced subjects.
    finding: |-
      Successful stages do not prove that candidate-controlled descendants are gone
      before their evidence subjects are attested. Validation and signing are separated
      by a writable, asynchronous same-job window.
    failure_scenario: |-
      A candidate test or mutation target launches a detached helper with its tracking
      marker removed, then exits zero. The helper derives the workspace from TEMP,
      waits until stage-summary/evidence-manifest exist, and replaces evidence plus
      those subjects self-consistently after schema validation but before the pinned
      attestation action reads them. Repository/environment checks have already passed;
      the trusted action signs the substituted bytes and upload retains them.
    consequence: |-
      Attested stage evidence need not be the evidence that passed the runner's final
      boundary and schema checks, defeating the claimed exact execution/evidence chain.
    required_action: |-
      Run every untrusted command in an OS-enforced process container whose complete
      descendant tree is synchronously terminated and proven empty before any final
      check. Deny candidate access to evidence/tool/output roots. Recompute complete
      closure after termination and make validation plus attestation subject capture
      atomic or revalidate attested/uploaded digests afterward. Add real Windows and
      Ubuntu production-runner payloads that detach/self-untrack, wait for summary,
      mutate/restore/replace subjects at each boundary, and must leave zero signed or
      uploaded output.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001"
    description: |-
      Execute exact candidate and mutation stage runners on Windows and Ubuntu with
      payloads that spawn detached descendants, remove runner tracking, derive the
      workspace through TEMP/TMP, and attempt replacement before/after final status,
      manifest, summary schema validation, attestation, and upload. Include delayed,
      self-restoring, symlink/reparse/hardlink, inherited-handle, and child-of-child
      variants. Prove the whole descendant tree is terminated before final closure and
      that no marker, subject, attestation, artifact, temporary output, or commitment
      survives. Preserve the exact six-fragment happy path.
    rationale: "A static completed-fixture test cannot establish process-tree quiescence across validation and attestation steps."
    blocking: true
prior_finding_results:
  - finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001"
    outcome: verified-resolved
    evidence: "Focused signed authority, lifecycle, source, packet, manifest, and cleanup cases pass."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No regression."
  - finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001"
    outcome: verified-resolved
    evidence: "Git-object verifier closure, real packet, source/dependency substitution, cross-run, and cleanup cases pass."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No regression."
  - finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001"
    outcome: verified-resolved
    evidence: "Normalized blob/receipt/autocrlf controls remain green."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No regression."
  - finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001"
    outcome: verified-resolved
    evidence: "Captured-OID races and immutable-object path remain green."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No regression."
  - finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001"
    outcome: verified-resolved
    evidence: "Replacement-disabled Git-object and hostile configuration cases remain green."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No regression."
  - finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001"
    outcome: verified-resolved
    evidence: "Fixed no-profile shells, environment transport, grammar rejection, and argument arrays remain green."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No regression."
  - finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001"
    outcome: unresolved
    evidence: "The builder response itself states the prior intra-stage same-path executable check/use race remains unresolved and blocking; stage separation does not remove races inside a stage."
    verification: read-only
    superseding_finding_id: ""
    notes: "Approval is impossible with this accepted critical predecessor open."
  - finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001"
    outcome: verified-resolved
    evidence: "Six fresh stage jobs, PDF-only TeX, scrubbed environments, no-shell-escape, TeX before/after closure, exact six attestations, and artifact rejection controls resolve its cross-stage/tool-context scope."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Post-validation descendant survival is a distinct later boundary assigned RR6."
prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001"
    outcome: verified-satisfied
    evidence: "Focused authority cases pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001"
    outcome: verified-satisfied
    evidence: "Focused verifier closure cases pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001"
    outcome: verified-satisfied
    evidence: "Normalized cross-platform hash cases pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001"
    outcome: verified-satisfied
    evidence: "Captured-object race cases pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001"
    outcome: verified-satisfied
    evidence: "Replacement-object controls pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001"
    outcome: verified-satisfied
    evidence: "Exact workflow hostile-input controls pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001"
    outcome: unresolved
    evidence: "Same-path use-time replacement remains explicitly unresolved."
    verification: read-only
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001"
    outcome: verified-satisfied
    evidence: "Six-stage topology and five static cross-stage payload cases pass; RR6 requests the distinct live-descendant test."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
predictions:
  experiment_id: ""
  predicted_outcome: ""
  predicted_failure_mode: ""
  confidence_statement: "No scientific experiment or hosted execution was run or inspected; this artifact assesses pre-holdout protocol/governance machinery only."
recommendation:
  approve: false
  blocking_findings: 2
  rationale: |-
    CHANGES REQUESTED. Close RR4's accepted intra-stage same-path race and RR6's
    descendant-to-attestation race, then obtain fresh independent rereview. This does
    not authorize merge, signer amendment, custodian verification/carry-forward,
    refreeze, activation, tag, preregistration, holdout, assembly, reveal, or claims.
```

## Verification ledger

- Exact handoff/tree `032611faafd2c3a3288b987c95d78fa5a03381d5` / `45aefe8c3f0411f3c4fc883d4d27d1b758872ccc`; sole parent content/tree `3a7d55e4797627765cd01801310a39fe58c655eb` / `49221426e9d2d539611fe21917d4a3c06fe13824`; origin head matched.
- RR5 response Git-normalized SHA-256 exactly `292859b9bbe29bb811269328a2b58f3603520ae41ba7d5776d438050b291d517`; response schema had zero errors.
- Focused R3 identity suite: 46 passed in 460.29 seconds.
- Changed-Python Ruff, TeX source, and campaign validator: passed.
- Required 22 compatibility suite and full 425 suite were not rerun after two critical blockers were bounded and sealing was requested; prior builder reports are not adopted as independent results.
- Decisive source proof: runner line 148 exposes stage-root-relative TEMP; line 172 waits only the direct process; no tree kill exists; workflow line 627 begins a later attestation step. The claimed watcher test contains no subprocess launch.
- Original through RR5 review commits are ancestors and prior reviewer artifacts are byte-unchanged. R2 scoped diff from `6e0e0c8ebaecef6d129c68666f113fbd47af4ce7` is empty.
- Comment-only signer remains; local and origin R3 tag queries are empty. Campaign is pending; packets remain holdout-false and unrevealed.
