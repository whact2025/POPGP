# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "OpenAI Codex"
builder_model_version: "GPT-5"
builder_operator: "NVIDIA.COM\\rfuoco"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-24.md"
review_commit: "887c656b89a2f37f64434fa36fc33af7b6151a92"
candidate_commit_reviewed: "768167269def1d6689ee38b0051127f51d670579"

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
  RR24 is implemented and bound at content commit
  21deabc877e6a13a68f8973c333e2a98669680f3 (tree
  af06d574d367a1135bc7521720b28a1508848d25). The sealed review was imported
  artifact-only at campaign commit b036813954c842afd945420e847423b75e9e9632
  without changing its bytes.

  The exact literal launcher and runtime identity remain fixed. The normalizer now
  accepts only the observed initial PATH `/opt/microsoft/powershell/7:/usr/bin:/bin`,
  immediately resets it to `/usr/bin:/bin`, and reasserts that sanitized value before
  artifact identity, stat, and output operations. The invalid profile-file-absence
  assumption is removed; exact six-argument `-NoProfile` launch remains mandatory.

  The same iteration repairs the shared validator regression exposed by generic CI:
  immutable R2 keeps its exact platform-summary contract without R3 stage/tool fields,
  while R3 requires the complete staged-plus-dispatch extension. Partial mixtures fail.

finding_responses:
  - finding_id: "VIA000-R3-RR24-UBUNTU-PWSH-INPROCESS-PATH-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Binding the one observed trusted runtime prefix and sanitizing it immediately
      closes the deterministic hosted mismatch without accepting arbitrary PATH data
      or weakening raw artifact, output-control, or redownload verification.
    changed_files:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "tests/unit/test_via000_r3_identity.py"
    fix_commits:
      - "21deabc877e6a13a68f8973c333e2a98669680f3"
    verification:
      - command: "focused RR24/RR22/RR21 identity gates"
        result: "3 passed, 68 deselected in 0.73 seconds."
      - command: "complete R3 shared-identity test file"
        result: "71 passed in 750.33 seconds."
      - command: "complete repository test suite"
        result: "450 passed in 2291.24 seconds, including all eight former R2 failures."
      - command: "actionlint 1.7.12, Ruff, JSON/YAML parsing, TeX, receipt/hash audit, review guidance, and campaign validator"
        result: "Local source/hash gates passed; campaign validator is rerun against the final bound handoff."
    residual_risk: |-
      A fresh exact-handoff hosted proof and independent rereview remain required.
      GitHub's hosted control plane and pinned actions remain trusted principals.
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR24-UBUNTU-PWSH-PATH-NORMALIZATION-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py::test_r3_rr24_ubuntu_pwsh_path_normalization"
    verification:
      - command: "exact initial/sanitized PATH and argv mutation corpus"
        result: "Exact initial prefix and reset pass; missing, doubled, reordered, alternate, injected, case-varied, post-reset, flag, argv-count, and script-location variants reject."
      - command: "fresh proof-only hosted replay"
        result: "Pending exact handoff branch push; acceptance requires six green producers/caches, aggregate and normalizer, one exact seven-file artifact, and a green redownload verifier."
    rationale: |-
      The regression binds source and receipt order, exact runtime identity/argv, four
      post-reset checks, and removal only of profile-file existence assumptions.
    disagreement_ref: ""

new_or_changed_risks:
  - "Any Ubuntu PowerShell runtime PATH or 7.6.5 launch-layout change rejects and requires review."
  - "Runs 32782295879 and 32783816053 are superseded diagnostic/failure evidence and cannot authorize lifecycle action."

external_actions:
  - action: "Inspect or independently replay the fresh exact-handoff six-cell proof, caches, normalized retained artifact identity, and redownload verifier."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: "fresh run recorded outside this artifact after branch push"
  - action: "Perform a fresh independent rereview of RR24 and retained prior controls before any lifecycle action."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Verify exact initial and sanitized Ubuntu PATH order, literal launcher/MainModule/PSHOME/7.6.5/six argv, unchanged artifact/verifier gates, R2/R3 validator compatibility, bindings, and the fresh retained six-cell proof."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, scientific execution, commitment, reveal, or viability claims from this response alone."
```
