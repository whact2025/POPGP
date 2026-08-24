# VIA-000 R3 recovery-protocol independent re-review 7

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-7"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-7"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-24"
commit_reviewed: "55df162d7a44fc69674550d7804135c9c527ad09"
baseline_commit: "ae3c56708136525bf31a6efd97b22e3b4b819a99"
prior_review_ref: "ae3c56708136525bf31a6efd97b22e3b4b819a99:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-6.md"
builder_response_ref: "55df162d7a44fc69674550d7804135c9c527ad09:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-6-RESPONSE-1.md"
context_hash: "aee2d7d0123a277b3764463daa83ea5f5e577b4e"
context_hash_method: 'git rev-parse "55df162d7a44fc69674550d7804135c9c527ad09^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-protocol.yml"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-6-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-6.md"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_via000_r3_identity.py"
  - "schemas/viability/independent-rereview-v2.schema.json"
access_level: "public repository plus local Windows containment test; no custody, signer, tag, hosted scientific execution, assembly, commitment, or reveal access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-7 and dedicated review branch. Exact
  commits, trees, sole parent, origin, supplied Git-normalized hashes, review history,
  and artifact immutability were verified first. No implementation or lifecycle state
  was changed. The operator/orchestrator are shared; session/worktree/branch differ.
  Builder model is unknown; model separation and external validation are not claimed.
independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "unknown"
  builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false
hidden_access_declaration: {final_labels_seen: false, secret_seed_seen: false, private_evaluator_seen: false}
summary: |-
  CHANGES REQUESTED with one critical proof blocker. Source inspection shows serious
  remediation: Windows uses a restricted low-integrity primary token, suspended
  creation, Job Object assignment before resume, kill-on-close/explicit termination,
  and zero-active-process accounting; Ubuntu specifies a DynamicUser transient
  systemd service, protected paths, an explicit writable root, and control-group kill.
  The local exact Windows hostile replace/restore detached-tree test passes. The
  workflow records containment fields, captures subjects after teardown, rechecks
  their hashes after signing, and the assembler rejects malformed containment identity.

  These mechanisms depend on hosted-runner facts that local/static review cannot
  establish: GitHub Windows nested-job behavior and ACL/integrity enforcement, and
  Ubuntu passwordless transient DynamicUser service creation, protected-path access,
  cgroup emptiness, double-fork teardown, and inability to migrate/escape. The protocol
  makes its six hosted containment self-tests mandatory, but exposes no separate safe
  containment-only workflow. Its sole self-test is inside the production VIA-000
  workflow after snapshot/authorization guards and requires campaign tags/signing.
  Running it would require forbidden lifecycle actions or bypasses. Therefore no exact
  hosted run/job/artifact IDs exist, and the required 2x3 production proof is absent.

  RR4 and RR6 are source-remediated but cannot be declared verified-resolved until
  the mandatory safe hosted proof exists and passes. R3 remains drafted, pending,
  holdout-false, unrevealed, and signer-blocked. No merge, signer/custodian gate,
  refreeze, activation, preregistration, holdout, or claim is authorized.
findings:
  - id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001"
    severity: critical
    category: governance
    location: ".github/workflows/via000-r3-protocol.yml:1-205,410-516; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1:250-580"
    evidence: |-
      Repository workflow enumeration found containment only in
      via000-r3-protocol.yml. That workflow has only workflow_dispatch with the
      required authorization input, then enforces the content-addressed source tag,
      signed authorization tag, signer file, and lifecycle guard before reaching
      `Prove production containment against detached replacement payload`. There is
      no containment-only event, job, reusable workflow, or safe branch-ref gate.
      The task forbids creating tags/keys or bypassing those guards, so the mandatory
      hosted Windows and Ubuntu tests cannot be run independently of VIA-000.

      The exact local Windows production test passes, but it cannot prove GitHub's
      hosted nested-job state. No local Linux systemd host can prove hosted Ubuntu
      sudo/DynamicUser/cgroup/protected-path behavior. Static tokens and fixture
      manifests are claims, not live platform evidence.
    finding: "A mandatory security premise has no safe pre-campaign hosted proof path, so containment viability is unestablished."
    failure_scenario: "The hosted Windows runner is already job-contained or denies atomic assignment, or Ubuntu forbids the transient DynamicUser/protection configuration; the production workflow discovers this only after lifecycle activation, yielding an invalid attempt rather than reviewed viability."
    consequence: "RR4/RR6 closure and the six scientific stage evidence chain remain unproven on their required platforms."
    required_action: "Add an independently reviewable containment-only workflow on immutable reviewed source that creates no tag, authorization, campaign workspace, scientific execution, evidence, or commitment. Require all 2x3 hosted self-tests and retain run/job/artifact IDs plus signed self-test subjects for rereview."
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001"
    description: "Dispatch a separate containment-only reviewed workflow on Ubuntu 24.04 and Windows 2025 for candidate/pdf/mutation roles; prove the exact Windows and systemd controls, hostile descendant/link/restore/ACL/read-write payloads, zero process/cgroup residue, atomic subject capture, and no campaign/lifecycle artifacts. Retain exact run, six job, and artifact identifiers."
    rationale: "Platform-specific containment claims cannot be approved from source markers or a local machine alone."
    blocking: true
prior_finding_results:
  - {finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001", outcome: verified-resolved, evidence: "Prior focused authority controls remain preserved.", verification: read-only, superseding_finding_id: "", notes: "No regression found."}
  - {finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001", outcome: verified-resolved, evidence: "Frozen validator closure remains preserved.", verification: read-only, superseding_finding_id: "", notes: "No regression found."}
  - {finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001", outcome: verified-resolved, evidence: "Normalized blob and receipt bindings remain preserved.", verification: read-only, superseding_finding_id: "", notes: "No regression found."}
  - {finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001", outcome: verified-resolved, evidence: "Captured-object controls remain preserved.", verification: read-only, superseding_finding_id: "", notes: "No regression found."}
  - {finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001", outcome: verified-resolved, evidence: "Replacement-disabled Git closure remains preserved.", verification: read-only, superseding_finding_id: "", notes: "No regression found."}
  - {finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-resolved, evidence: "No-profile/environment-data command boundary remains preserved.", verification: read-only, superseding_finding_id: "", notes: "No regression found."}
  - {finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Source introduces OS containment and local Windows hostile test passes, but mandatory hosted 2x3 proof is unavailable.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Blocked by RR7 proof-path finding."}
  - {finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-resolved, evidence: "Six-job topology and context closure remain intact.", verification: read-only, superseding_finding_id: "", notes: "No regression found."}
  - {finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Source contains tree containment and atomic subject recheck, but mandatory hosted 2x3 proof is unavailable.", verification: confirmed-by-execution, superseding_finding_id: "", notes: "Blocked by RR7 proof-path finding."}
prior_requested_test_results:
  - {requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001", outcome: verified-satisfied, evidence: "Preserved in ancestry.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved in ancestry.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001", outcome: verified-satisfied, evidence: "Preserved in ancestry.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001", outcome: verified-satisfied, evidence: "Preserved in ancestry.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001", outcome: verified-satisfied, evidence: "Preserved in ancestry.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001", outcome: verified-satisfied, evidence: "Preserved in ancestry.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001", outcome: unresolved, evidence: "Hosted containment proof unavailable.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001", outcome: verified-satisfied, evidence: "Preserved six-stage controls.", verification: read-only, superseding_requested_test_id: "", notes: ""}
  - {requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001", outcome: unresolved, evidence: "Local Windows passes; hosted 2x3 proof unavailable.", verification: confirmed-by-execution, superseding_requested_test_id: "", notes: ""}
predictions: {experiment_id: "", predicted_outcome: "", predicted_failure_mode: "", confidence_statement: "No scientific or hosted campaign execution was performed."}
recommendation:
  approve: false
  blocking_findings: 1
  rationale: "CHANGES REQUESTED. Add and independently run a safe hosted containment-only proof gate. No merge or downstream signer/custodian/refreeze/activation/holdout gate is authorized."
```

## Verification ledger

- Exact handoff/content/tree/origin/sole-parent identities and all five supplied Git-normalized SHA-256 values matched.
- Exact local Windows hostile containment test: 1 passed in 6.22 seconds.
- Workflow enumeration: no separate containment-only safe hosted gate; no hosted run was authorized or executed.
- Deferred 22/full/focused suites were not run after the mandatory hosted-proof-path blocker was confirmed; their builder results are not adopted independently.
