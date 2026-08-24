# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-8-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-8-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-8"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-8.md"
review_commit: "12158dd52288a08642d7d29376fbb94cd7809b25"
candidate_commit_reviewed: "611b39cc2d2bb68ee4a8705f32f8d36b2d53d5dd"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: |-
    R3 remains drafted, holdout_started=false, unrevealed, and pending. No custody
    directory, sealed manifest, hidden label, secret seed, reveal material, external
    invalid assembled package, or handoff memo was accessed. No signer key, tag,
    refreeze, activation, campaign execution, raw result, commitment, adjudication,
    or scientific candidate/baseline operation was created. The changed workflow is
    synthetic, non-scientific, read-only, and has no secret or environment access.

summary: |-
  RR8 is accepted and implemented at content commit
  2397160788ce4c5b0a0fe79995d5045c1d788738 (tree
  16a62f5645061db02386794d311e97d0ae8a1b26). The six-job containment topology and
  production Windows Job Object / Ubuntu systemd ephemeral-user boundaries are
  unchanged. After the contained descendant tree is terminated, quiescence is proven,
  and the four live proof subjects are validated as regular, single-link, bounded,
  hash-bound files, the trusted runner creates a fresh direct child of RUNNER_TEMP.
  Windows applies the explicit ordinary medium-integrity label and requires inherited
  user DACL state; Ubuntu requires mode 0700. Neither location exists while untrusted
  code runs.

  Each cell writes exactly one canonical UTF-8/LF/no-BOM JSON envelope. Its sorted,
  compact object binds repository, workflow/ref, source ref/SHA, run/attempt, platform,
  stage, cell, and artifact name, and contains exactly containment-result.json,
  proof.json, stderr.txt, and stdout.txt. Each case-sensitive member carries exact byte
  length, SHA-256, and canonical base64. Per-member, decoded-total, and encoded-envelope
  limits are frozen. The upload action receives only the exact runner-temp envelope
  file; no archive, extraction path, labelled multi-file tree, or wildcard is used.

  The frozen aggregate verifier requires exactly six ordinary one-file artifact
  directories. It rejects BOM/CR/non-UTF-8 or noncanonical JSON, duplicate or
  case-fold-colliding keys, missing/extra members, malformed/noncanonical base64,
  length/hash/expansion disagreement, symlink/hardlink substitution, inner proof or
  containment/transcript hash disagreement, and any cross-cell/run/source identity.
  Decoded subjects remain in memory and are passed to the existing exact 2x3 proof
  predicates. Failure removes aggregate output. A fresh safe hosted branch-push replay
  and independent rereview remain mandatory before RR4, RR6, or RR8 can be resolved.

finding_responses:
  - finding_id: "VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The transport no longer asks the artifact action to discover four files inside
      the security-labelled live evidence tree. One fresh, ordinary, post-quiescence
      runner-temp file contains the exact validated bytes without weakening the live
      protected closure. The aggregate treats the envelope as untrusted input and
      performs canonical, bounded, hash, identity, and inner-proof validation before
      creating aggregate output.
    changed_files:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF.schema.json"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-ENVELOPE.schema.json"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "tests/unit/test_via000_r3_identity.py"
      - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
      - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/README.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/amendments/VIA-000-R3-RECOVERY-DESIGN-1.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-aggregator.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof.schema.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-envelope.schema.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/PROTOCOL_MANIFEST.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
    fix_commits:
      - "2397160788ce4c5b0a0fe79995d5045c1d788738"
      - "5e74c230ebdc03e63d625f224035cce6ef45a09d"
      - "65a5795fdaca82f55ad485bbfa2f959b58e42c4f"
    verification:
      - command: "stable RR8 canonical-envelope production-path test"
        result: "1 passed in 7.17 seconds; exact six-envelope happy path and malformed, BOM/CR, noncanonical, duplicate/case-collision, oversized, base64, size/hash, inner-hash, hardlink, missing-cell, and identity failures were fail-closed with no aggregate output."
      - command: "Ruff, PowerShell, JSON schema, and YAML parsers"
        result: "Changed Python passed Ruff; runner PowerShell parsed; both proof schemas passed Draft 2020-12 checks; workflow and packet YAML parsed."
      - command: "review guidance, TeX, and campaign validation"
        result: "9 review-guidance tests passed in 11.31 seconds with the 432-test ledger reconciled; TeX source validation passed; the drafted campaign contract is valid at binding content 5e74c230ebdc03e63d625f224035cce6ef45a09d."
      - command: "official actionlint 1.7.12"
        result: "The proof-only workflow passed with zero parse or expression errors using the regular release binary (SHA-256 54ca21be3de4c7cfa26914aa8b61bd76bf573ef3caac5f80d110558cdf241718)."
      - command: "complete R3 identity file and focused correction"
        result: "53 passed in 588.32 seconds after replacing one stale RR7 source assertion for the deliberately removed InaccessiblePaths property with the retained NoNewPrivileges control; the exact corrected case plus stable RR8 export case also passed 2/2 in 6.41 seconds."
    residual_risk: "The one-file Windows artifact boundary and exact hosted 2x3 aggregate require a fresh branch-push replay and independent inspection. Comment-only signer and fresh custody carry-forward verification still block activation."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR8-WINDOWS-PROOF-EXPORT-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-ENVELOPE.schema.json"
      - "tests/unit/test_via000_r3_identity.py::test_r3_rr8_windows_proof_export_is_canonical_and_exact_2x3"
    verification:
      - command: "local exact six-envelope and hostile corpus"
        result: "The exact 2x3 set aggregates; all declared malformed/ambiguous/oversized/substituted/cross-identity forms reject with zero aggregate output."
      - command: "safe hosted feature-branch replay"
        result: "Pending the handoff push; run/job/artifact IDs and digests will be recorded outside this frozen response for independent rereview."
    rationale: "The stable test exercises the frozen aggregator on the exact production envelope shape and the workflow invokes the same frozen producer and verifier bytes on both hosted operating systems."
    disagreement_ref: ""

new_or_changed_risks:
  - "The proof envelope temporarily duplicates four small non-scientific subjects as base64, bounded to 262144 bytes per member, 524288 decoded bytes total, and 1048576 encoded bytes."
  - "The proof workflow still intentionally runs on qualifying feature/review branch pushes, with contents-read permission only and no secrets, environments, or scientific/lifecycle paths."
  - "A local Windows test cannot establish GitHub artifact-action behavior; the automatically triggered hosted replay and independent artifact inspection remain blocking evidence."

external_actions:
  - action: "Inspect or independently replay the safe hosted RR8 run; require all three Windows and three Ubuntu one-envelope artifacts and a green exact aggregate before resolving RR4/RR6/RR8."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: "recorded outside this artifact after branch push"
  - action: "Propose and independently review an amendment freezing exactly one Ed25519 authorization public key."
    owner: "maintainer-and-independent-reviewer"
    status: pending
    evidence_ref: ""
  - action: "Perform fresh fourteen-of-fourteen R3 custody carry-forward verification before preregistration or holdout."
    owner: "evaluator-custodian-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Verify the one-file runner-temp export is created only after containment teardown; exact medium-integrity/DACL or Unix-mode, regular/single-link/ADS, canonical JSON, member/inner hash, limits, identity, source/receipt bindings, six hosted artifacts, and exact 2x3 aggregate; preserve all prior R3 controls."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, scientific execution, commitment, reveal, or viability claims from this response alone."
```
