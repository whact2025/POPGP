# VIA-000 R3 recovery-protocol independent re-review 10

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-10"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "9e28297b803d0570aa2fc62390267f9fb0f2c530"
baseline_commit: "7f28f7fb30823fa2081308fe12e3e51a493d0068"
prior_review_ref: "7f28f7fb30823fa2081308fe12e3e51a493d0068:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-9.md"
builder_response_ref: "9e28297b803d0570aa2fc62390267f9fb0f2c530:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-9-RESPONSE-1.md"
context_hash: "eee2ea50ff99dffe7e72712eb7effae41bfca533"
context_hash_method: 'git rev-parse "9e28297b803d0570aa2fc62390267f9fb0f2c530^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-containment-proof.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
  - "tests/unit/test_via000_r3_identity.py"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-9-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-9.md"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository, GitHub run 32714269336 logs/metadata, and public official GitHub runner/cache sources; no signer, custody, lifecycle, holdout, scientific execution, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-10. Exact commit/tree/parent/origin,
  Git-normalized response/prior-review hashes, all seven hosted job logs, artifact
  API results, and upstream runner/cache behavior were independently inspected.
  No implementation or external state was changed. Operator/orchestrator are shared;
  session/worktree/branch differ. Builder model is shared and external validation is
  not claimed.
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
  CHANGES REQUESTED with one high-severity transport blocker; the architecture
  remains viable. Run 32714269336 was a non-scientific push at the exact handoff.
  All six production containment jobs passed. Aggregate input proved that the three
  Ubuntu envelopes crossed as 9324, 9236, and 9308 base64 characters, while all
  three Windows values were empty. Runner 2.336.0 logged `Set output` for each
  Ubuntu job but neither `Set output` nor the warning `Skip output ... since it may
  contain secret` for any Windows job. Under the exact runner finalizer source, this
  means the Windows job-output expression evaluated empty; secret masking is not an
  evidence-supported diagnosis. The narrower supported diagnosis is a Windows
  step-output command-file loss between the producer's successful flush/readback
  and runner finalization. In addition, GitHub printed all three Ubuntu envelope
  values in the aggregate step environment preamble, so the large-output channel
  violated the no-content-log requirement.

  Aggregate job 97392196293 rejected the first empty Windows value before creating
  retained output, upload was skipped, verifier 97392296312 was skipped, and the
  artifact API returned zero. No campaign/lifecycle/signing/custody/holdout/
  scientific/commitment/reveal action ran.

  A cache-backed closure is bounded and viable within the declared trusted GitHub
  control-plane threat model, but not with a predictable key alone. The exact key
  must include the post-containment envelope SHA-256; only that 64-hex digest may
  cross as a small job output. Each cache contains one canonical envelope whose
  bytes must hash to the key. A different first-writer cache then rejects, and an
  identical pre-existing cache is byte-equivalent rather than a substitution.
findings:
  - id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001"
    severity: high
    category: code
    location: ".github/workflows/via000-r3-containment-proof.yml:27-387; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1:141-211,600-601"
    evidence: |-
      Jobs 97392081274, 97392081475, 97392081512, 97392081546, 97392081599,
      and 97392081632 passed. Runner finalization accepted all three Ubuntu outputs
      and accepted none of the Windows outputs. The aggregate environment exposed
      the complete Ubuntu base64 values and three empty Windows values, then failed
      closed. Official runner 2.336.0 JobExtension.cs lines 711-734 distinguishes
      empty, secret-skipped, and accepted outputs; the corresponding secret warning
      is absent. No retained artifact exists.
    finding: "The reviewed large job-output transport loses every Windows envelope and discloses every successful Ubuntu envelope in logs, so exact retained 2x3 evidence remains absent."
    failure_scenario: "A Windows step output remains empty, an envelope is logged through the aggregate environment preamble, or a predictable cache key is first-written with forged bytes before the genuine cell saves."
    consequence: "RR4/RR6 production-boundary claims remain unretained and cannot authorize the scientific lifecycle."
    required_action: |-
      Remove envelope bytes from job outputs and environments. Keep six explicit
      cells. After teardown, quiescence, cleanup, canonical validation, and subject
      recheck, stage exactly one bounded regular/single-link/non-reparse envelope in
      a fresh medium-integrity, workspace-relative, cell-specific ordinary path.
      Use forward-slash path literals that are identical in the saving and restoring
      jobs; cache version binds path and compression, so RUNNER_TEMP absolute paths
      are not cross-platform-equivalent. Require `enableCrossOsArchive: true`.

      Pin both actions/cache/save and actions/cache/restore to reviewed current
      v6.1.0 commit 55cc8345863c7cc4c66a329aec7e433d2d1c52a9. Use six keys under
      512 characters containing repository ID, workflow SHA, source SHA, run ID,
      run attempt, schema version, platform, stage, and the full canonical-envelope
      SHA-256. Transport only that strict 64-hex digest as a unique small cell job
      output; a missing/transformed digest must fail. Use no restore keys or prefix
      fallback. The Ubuntu aggregate must require `cache-hit == true`, require the
      reported matched key to equal the primary key byte-for-byte, restore the six
      keys into six fresh distinct cell-relative roots, and reject missing/extra/
      duplicate/link/reparse/hardlink/path-escape/oversized/corrupt/cross-cell bytes.
      Recompute each envelope hash against its key, then perform the existing
      canonical, inner-hash, and exact run/attempt/repository/workflow/source 2x3
      validation in memory. Only after all six pass may it write and upload the six
      envelopes plus canonical aggregate/manifest; a dependent job must download
      and revalidate that retained artifact.

      The official save action catches upload/collision errors and exits success;
      therefore cell success is not evidence of a save. Exact aggregate restore and
      validation are mandatory and authoritative. Run/attempt/cell keys prevent
      ordinary reuse; fork repositories have distinct cache scope, the workflow has
      no hostile parallel writer, and the containment helper removes ACTIONS_*,
      GITHUB_*, and RUNNER_* from untrusted children. If a hostile parallel job with
      a cache runtime token is admitted to the threat model, the standard save action
      cannot prove producer provenance and must be replaced by a separately reviewed
      direct runtime client that exposes and binds the reservation/cache identity.
      Cache retention is transport only; the consolidated artifact remains the
      authoritative record and immediate cache eviction must not affect verification.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001"
    description: "Hosted exact-source proof: six green cells; six unique digest-bound cache keys; Windows-to-Ubuntu cross-OS exact restores from identical cell-relative paths; cache-hit and matched-key equality; missing/save-warning/pre-existing/different-first-writer/prefix/default-branch/fork/truncated/link/path-escape/corrupt/cross-cell rejection; no envelope in logs; one retained seven-file artifact and green redownload verifier."
    rationale: "Only retained end-to-end evidence can distinguish a working cache transport from another green producer with missing or substituted cross-job state."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "All six cells passed but no retained 2x3 artifact exists.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Blocked by RR10."}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Preserved.", verification: read-only, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "All six cells passed but exact retained envelopes are absent.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Blocked by RR10."}
  - {finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: verified-resolved, evidence: "Safe non-scientific hosted path exists and ran.", verification: confirmed-by-execution, superseding_finding_id: "", notes: ""}
  - {finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Windows transport again retained zero bytes.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", notes: "Remedy superseded."}
  - {finding_id: "VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "All Windows outputs were empty and Ubuntu contents entered logs.", verification: confirmed-by-execution, superseding_finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", notes: ""}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Retained 2x3 evidence absent.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Retained 2x3 evidence absent.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001", outcome: unresolved, evidence: "Hosted path runs but has not retained a complete artifact.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001", outcome: unresolved, evidence: "Windows retained zero envelopes.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR9-CANONICAL-ENVELOPE-TRANSPORT-001", outcome: unresolved, evidence: "Hosted RR9 transport produced no artifact.", verification: confirmed-by-execution, superseding_requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001", notes: ""}
predictions: {experiment_id: "", predicted_outcome: "", predicted_failure_mode: "", confidence_statement: "No scientific execution occurred."}
recommendation: {approve: false, blocking_findings: 1, rationale: "Fixable CHANGES REQUESTED. Implement digest-bound exact cache transport and retain/revalidate the consolidated artifact; architecture remains viable."}
```

## Verification ledger

- Exact handoff/tree/content/tree/sole parent/origin matched. Git-normalized response SHA-256 was `a026c73be8d54d661b68047370484df2453d6ffd4e1f32a4bec52e68c2d32c9b`; prior review SHA-256 was `5d2b8aa6ab35e2f23481e6eac727082c61216f488ccfeffd7fa53ade69deb436`.
- Run `32714269336`, all named jobs, and artifact API inspected. Six cells passed; aggregate failed closed; verifier skipped; artifact count zero.
- Official runner `v2.336.0` FileCommandManager/JobExtension and official cache `v6.1.0` at `55cc8345863c7cc4c66a329aec7e433d2d1c52a9` inspected. Cache save warnings are non-fatal; cache version binds path/compression; keys are branch/version scoped and capped at 512 characters.
- `uv run --frozen --no-editable python -m pytest -q tests/unit/test_via000_r3_identity.py::test_r3_rr9_canonical_envelope_transport_is_exact_and_retained`: `1 passed`. This local test does not override the hosted failure.
- R3 remains drafted, `holdout_started: false`, unrevealed, and pending; no VIA-000 tag exists locally or remotely.
- No signer, key, tag, refreeze, custody, lifecycle, holdout, scientific execution, commitment, or reveal was performed or authorized.
