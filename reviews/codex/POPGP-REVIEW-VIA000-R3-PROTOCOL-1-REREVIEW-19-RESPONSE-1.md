# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-19-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-19-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "OpenAI Codex"
builder_model_version: "GPT-5"
builder_operator: "NVIDIA.COM\\rfuoco"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-19"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-19.md"
review_commit: "32d38636edc3229bbe279a367b8a51b8a6c117f2"
candidate_commit_reviewed: "c39626d098f6874216fbd0e3a8d78224569d4e82"

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
  RR19 is implemented and bound at content commit
  c11b678f20da5a0f4312c30fd23f7eb69f4db089 (tree
  a312bbb2ee68b10a6c12a1b1cd2bd04bb8575810). The sealed review was imported
  artifact-only at campaign commit a910a27c6e2c2cd02724ae4445da21d581b0335a.

  All three proof-only Windows cells now use one frozen zstd identity:
  C:\tools\zstd\zstd.exe version 1.5.7 under an exact sanitized PATH. The trusted
  archive step verifies ordinary non-reparse ancestors/file, one data stream, one
  hard link, unique exact Get-Command resolution, exact version, and SHA-256. The
  captured hash is transported as control-plane data and the complete identity is
  repeated immediately before and after the pinned cache-save action. Ubuntu and all
  containment, export-descriptor, envelope, digest, key, and topology rules are
  unchanged.

finding_responses:
  - finding_id: "VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The image-owned executable has one literal reviewed location. Pinning it and
      proving that sanitized PATH resolution, version, bytes, and filesystem identity
      remain identical around cache save closes the assertion/action divergence while
      preserving fail-closed image drift.
    changed_files:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "tests/unit/test_via000_r3_identity.py"
    fix_commits:
      - "c11b678f20da5a0f4312c30fd23f7eb69f4db089"
    verification:
      - command: "focused RR19/RR18/cache/RR12/RR15 identity gate"
        result: "5 passed, 62 deselected in 25.05 seconds."
      - command: "complete R3 shared-identity test file"
        result: "67 passed in 718.49 seconds."
      - command: "actionlint 1.7.12, Ruff, JSON/YAML parsing, receipt equality, and diff checks"
        result: "passed before content sealing."
    residual_risk: |-
      The fresh exact-handoff hosted replay and independent rereview remain required.
      GitHub's hosted control plane and pinned cache action remain trusted principals.
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR19-WINDOWS-CACHE-ZSTD-IDENTITY-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py::test_r3_rr19_windows_cache_zstd_identity_is_exact_and_rechecked"
    verification:
      - command: "exact source and mutation corpus"
        result: "Missing, wrong-version, changed-hash, PATH-shadow, reparse, alias, Git-relative, and multi-allowlist substitutions reject; exact workflow/receipt identity passes."
      - command: "fresh proof-only hosted replay"
        result: "Pending exact handoff branch push; acceptance requires six green producers, six exact caches, one retained seven-file artifact, and a green redownload verifier."
    rationale: |-
      The regression freezes the exact image-owned path, PATH position, 1.5.7 banner,
      hash capture/recheck order, pinned action adjacency, and receipt equality.
    disagreement_ref: ""

new_or_changed_risks:
  - "A future hosted-image zstd path, version, filesystem identity, resolution, or byte change rejects and requires a reviewed amendment."
  - "Run 32763190366 is superseded and supplies no reusable retained acceptance artifact."

external_actions:
  - action: "Inspect or independently replay the fresh exact-handoff six-cell proof, caches, retained artifact, and redownload verifier."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: "fresh run recorded outside this artifact after branch push"
  - action: "Perform a fresh independent rereview of RR19 and all retained prior controls before any lifecycle action."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Verify exact Windows zstd path/version/hash/resolution identity around pinned cache save, receipt/hash bindings, fresh six-cell exact-cache transport, retained aggregate redownload, and all prior R3 controls."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, scientific execution, commitment, reveal, or viability claims from this response alone."
```
