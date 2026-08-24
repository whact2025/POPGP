# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-18-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-18-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "OpenAI Codex"
builder_model_version: "GPT-5"
builder_operator: "NVIDIA.COM\\rfuoco"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-18"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-18.md"
review_commit: "1265b92ade51fd56a066a5cc0e465c8fbd87a338"
candidate_commit_reviewed: "e3f9ad3463a6967c6fea17870d8fad38d4b1fa5b"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: |-
    R3 remains drafted, holdout_started=false, unrevealed, and pending. No custody,
    signer, lifecycle, hidden, or scientific-result material was accessed or created.
    No key, tag, refreeze, activation, campaign execution, commitment, adjudication,
    or reveal was performed. The hosted workflow remains proof-only and non-scientific.

summary: |-
  RR18 is implemented and bound at content commit
  3daeab1305a90430824257b78e6053d383c2687e (tree
  c74ae5f6abc9c3e6561a2159dd8e1493204e0804). The sealed review was imported
  artifact-only at campaign commit 962370ce46d0acf48382ca994038aa6d000317a1;
  no RR17 experiment workflow or script entered the campaign tree.

  Windows export roots now receive owner, exact one-ACE protected DACL, and medium
  mandatory label through native security APIs. The canonical envelope receives only
  an explicit native owner correction, preserving its exact single inherited runner
  FullControl ACE and inherited medium label. Native and managed descriptor facts are
  re-queried and compared at root creation, file close/hash capture, digest, and
  pre-cache-save boundaries. A workspace-relative live restricted-token replay proved
  trusted access, all seven low-integrity export attacks denied, descriptor/hash
  stability, and zero active descendants after teardown.

finding_responses:
  - finding_id: "VIA000-R3-RR18-WINDOWS-ROOT-FILE-DESCRIPTOR-APPLICATION-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Native owner/DACL application removes dependency on parent ACL shape and the
      elevated token's default owner. Exact raw and managed re-query preserves the
      reviewed descriptor rather than relaxing any ACL, owner, label, or link rule.
    changed_files:
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF.schema.json"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-protocol.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
      - "tests/unit/test_via000_r3_identity.py"
    fix_commits:
      - "3daeab1305a90430824257b78e6053d383c2687e"
    verification:
      - command: "focused RR14 through RR18 and cache identity tests"
        result: "6 passed in 26.11 seconds."
      - command: "complete R3 shared-identity test file"
        result: "66 passed in 756.84 seconds."
      - command: "exact native descriptor hostile replay plus eight-file PowerShell/YAML parser closure"
        result: "2 passed in 34.20 seconds."
      - command: "Ruff, actionlint 1.7.12, JSON/YAML/schema, receipt equality, diff check, and TeX Live 2026 compile"
        result: "passed before content sealing."
    residual_risk: |-
      The fresh exact-handoff hosted replay and independent rereview remain required.
      GitHub's hosted control plane and pinned actions remain trusted principals.
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR18-NATIVE-ROOT-FILE-DESCRIPTOR-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py::test_r3_rr18_native_root_file_descriptor_and_workspace_hostile_replay"
      - "tests/unit/test_via000_r3_identity.py::test_r3_rr16_windows_medium_export_boundary_denies_low_integrity_writes"
    verification:
      - command: "production helper workspace-relative live low-integrity replay"
        result: "Trusted create/write/read/hash passed; create, write, hardlink, reparse, rename, delete, and replace attacks were denied; teardown reported zero active processes."
      - command: "fresh proof-only hosted replay"
        result: "Pending exact handoff branch push; acceptance requires six green cells, six distinct caches, retained seven-file artifact, and green redownload verifier."
    rationale: |-
      The regression exercises the production native descriptor setter and restricted-
      token containment helper under the same nested workspace-relative export shape,
      while frozen workflow assertions independently cover digest and cache boundaries.
    disagreement_ref: ""

new_or_changed_risks:
  - "The exact hosted Windows control flags are bound to root 37892 and file 33796; any hosted divergence rejects before digest/cache rather than weakening the descriptor."
  - "Prior RR16 and RR17 runs are superseded and provide no reusable production acceptance artifact."

external_actions:
  - action: "Inspect or independently replay the fresh exact-handoff six-cell proof and retained redownload artifact."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: "fresh run recorded outside this artifact after branch push"
  - action: "Perform a fresh independent rereview of RR18 and all retained prior controls before any lifecycle action."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Verify native exact root/file descriptors, live low-integrity denial, receipt/hash bindings, fresh six-cell cache transport, retained aggregate redownload, and all prior R3 controls."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, scientific execution, commitment, reveal, or viability claims from this response alone."
```
