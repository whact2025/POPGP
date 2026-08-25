# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-28-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-28-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "OpenAI Codex"
builder_model_version: "GPT-5"
builder_operator: "NVIDIA.COM\\rfuoco"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-28"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-28.md"
review_commit: "a3c7d1ca699a1f964e7ba4817e39a5eb52634e4a"
candidate_commit_reviewed: "0acec59f60ca1fdf60a66b25b60459e3c044bde9"

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
  RR28 is implemented and bound at content commit
  2c605ff2894eed22c61f2292699a27fd8d8bff41 (tree
  b030fb61f138439fdf706eee77c89af985508274). The sealed review was imported
  artifact-only at campaign commit e1c7d0a5a68cb8f653b2d752d3d2503cd94f0462
  without importing the RR27 experiment workflow or changing the review bytes.

  The Ubuntu aggregate normalizer now binds the exact five ordered initial PATH
  components to fixed PowerShell and Python 3.11.15 toolcache anchors. It requires
  exact raw-string and ordinal elementwise agreement, exact setup-python outputs and
  root variables, then immediately resets PATH to /usr/bin:/bin and reasserts that
  value before every artifact, runner-script, stat, and GITHUB_OUTPUT operation.
  The shared raw-results validator also preserves the strict R2 key contract while
  accepting only the frozen R3 extensions, closing the previously observed generic-CI
  compatibility failures without changing any R2 packet or interpretation.

finding_responses:
  - finding_id: "VIA000-R3-RR28-SETUP-PYTHON-PATH-BINDING-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      RR27 established the exact setup-python-aware hosted PATH. Binding that fixed
      vector and independently cross-checking all action and environment identities
      removes the false rejection while preserving fail-closed tool identity.
    changed_files:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "tests/unit/test_via000_r3_identity.py"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
    fix_commits:
      - "2c605ff2894eed22c61f2292699a27fd8d8bff41"
    verification:
      - command: "focused RR28, PowerShell parse-closure, and review-guidance gates"
        result: "11 passed in 34.89 seconds."
      - command: "complete R3 shared-identity test file"
        result: "73 passed in 649.49 seconds."
      - command: "complete repository test suite"
        result: "452 passed in 2691.50 seconds."
      - command: "actionlint 1.7.12, Ruff, receipt equality, two-pass TeX, and campaign validator"
        result: "All bounded local source, parser, document, receipt, and drafted-campaign gates passed."
    residual_risk: |-
      A fresh exact-handoff hosted proof, green generic CI, and independent rereview
      remain required. GitHub's hosted control plane and pinned actions remain trusted
      principals.
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR28-EXACT-FIVE-COMPONENT-PATH-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py::test_r3_rr28_exact_five_component_path"
    verification:
      - command: "exact five-component PATH and setup-python identity corpus"
        result: "The exact fixed vector passes; missing, extra, duplicated, reordered, empty, relative, alternate version/architecture/root, case-varied, trailing, injected, swapped, and manipulated action/root fixtures reject."
      - command: "fresh proof-only hosted replay"
        result: "Pending exact handoff branch push; acceptance requires six green producers/caches, aggregate and normalizer, one exact seven-file artifact, and a green redownload verifier."
    rationale: |-
      The regression binds both source and receipt bytes, fixed setup-python pin and
      inputs, exact output/root cross-checks, raw and ordinal PATH equality, immediate
      sanitization, and every required post-reset reassertion.
    disagreement_ref: ""

new_or_changed_risks:
  - "Any hosted change to the fixed PowerShell or Python 3.11.15 toolcache PATH vector rejects and requires review."
  - "The RR27 diagnostic remains non-authoritative evidence and is not part of the campaign closure."

external_actions:
  - action: "Inspect or independently replay the fresh exact-handoff six-cell proof, caches, normalized retained artifact identity, and redownload verifier."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: "fresh run recorded outside this artifact after branch push"
  - action: "Perform a fresh independent rereview of RR28 and retained prior controls before any lifecycle action."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Verify the exact five-component PATH, setup-python output/root anchors, immediate sanitization and rechecks, unchanged artifact/verifier gates, bindings, generic CI, and fresh retained six-cell proof."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, scientific execution, commitment, reveal, or viability claims from this response alone."
```
