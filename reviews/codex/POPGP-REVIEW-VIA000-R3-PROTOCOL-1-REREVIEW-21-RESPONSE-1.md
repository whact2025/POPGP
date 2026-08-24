# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-21-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-21-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "OpenAI Codex"
builder_model_version: "GPT-5"
builder_operator: "NVIDIA.COM\\rfuoco"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-21"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-21.md"
review_commit: "43826b7cdcb3024d53d1f264ab17e4a225027a10"
candidate_commit_reviewed: "3066adfce03997cbc357895f69aff3814fe37529"

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
  RR21 is implemented and bound at content commit
  914ca1ae2e9cb3b916ae2231ef288d8f3a597ed8 (tree
  5b060b1fa08a468854f4fa0eca214f81a936c1ec). The sealed review was imported
  artifact-only at campaign commit bd3bcaf79c7b1e67dd8a290ce497583253308c62.

  The trusted aggregate job now consumes the pinned uploader outputs in one
  immediately adjacent built-in-pwsh step. It accepts only an exact bare lowercase
  64-hex digest, canonical positive artifact ID, exact repository/run identity, and
  exact repository/run/ID URL; constructs one `sha256:` prefix; and exposes only those
  normalized values. The verifier remains prefixed-only and independently recomputes
  the expected URL before invoking the unchanged retained-byte aggregator.

finding_responses:
  - finding_id: "VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Strict one-way normalization at the trusted action/workflow boundary preserves
      one downstream digest representation without relaxing the verifier or changing
      the archive download and extracted-byte contracts.
    changed_files:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "tests/unit/test_via000_r3_identity.py"
    fix_commits:
      - "914ca1ae2e9cb3b916ae2231ef288d8f3a597ed8"
    verification:
      - command: "focused RR21/RR20/artifact transport gate"
        result: "5 passed, 64 deselected in 47.81 seconds."
      - command: "RR15 parser plus RR21 exact regression"
        result: "2 passed, 67 deselected in 6.30 seconds after updating the expected parsed workflow-block count from 40 to 41."
      - command: "complete R3 shared-identity test file"
        result: "69 passed in 739.42 seconds."
      - command: "actionlint 1.7.12, Ruff, JSON/YAML parsing, receipt equality, response schema, review guidance, and campaign validator"
        result: "passed before handoff sealing."
    residual_risk: |-
      The exact-handoff hosted replay and independent rereview remain required.
      GitHub's hosted control plane and pinned upload/download/cache actions remain
      trusted principals.
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR21-ARTIFACT-DIGEST-CANONICALIZATION-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py::test_r3_rr21_artifact_digest_canonicalization"
    verification:
      - command: "workflow-source and representation mutation corpus"
        result: "Exact bare lowercase action digest becomes one API-prefixed value; uppercase, whitespace, newline, malformed hex/length, pre-prefixed, double-prefixed, noncanonical ID, and URL/repository/run substitutions reject."
      - command: "prefixed API equality corpus"
        result: "Only exact canonical `sha256:<hex>` equality passes; bare, uppercase, whitespace, double-prefix, and digest mismatch reject."
      - command: "fresh proof-only hosted replay"
        result: "Pending exact handoff branch push; acceptance requires six green producers and caches, one exact retained seven-file artifact, and a green redownload verifier."
    rationale: |-
      The regression binds the exact action-output inputs, immediate normalization
      ordering, protected output names, strict downstream grammar, exact URL identity,
      pinned downloader, and receipt equality.
    disagreement_ref: ""

new_or_changed_risks:
  - "A future uploader digest/ID/URL representation change rejects and requires a reviewed amendment."
  - "Run 32771982270 and artifact 9536553478 remain superseded and cannot serve as lifecycle evidence because the hosted verifier was red."

external_actions:
  - action: "Inspect or independently replay the fresh exact-handoff six-cell proof, caches, retained artifact, and redownload verifier."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: "fresh run recorded outside this artifact after branch push"
  - action: "Perform a fresh independent rereview of RR21 and all retained prior controls before any lifecycle action."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Verify strict one-way artifact digest canonicalization, exact ID/repository/run/URL and API equality, receipt/hash bindings, fresh six-cell cache transport, retained aggregate redownload, and all prior R3 controls."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, scientific execution, commitment, reveal, or viability claims from this response alone."
```
