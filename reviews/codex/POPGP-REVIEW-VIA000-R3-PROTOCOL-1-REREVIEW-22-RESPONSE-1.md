# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "OpenAI Codex"
builder_model_version: "GPT-5"
builder_operator: "NVIDIA.COM\\rfuoco"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-22.md"
review_commit: "ef34dd4e9a760d7c6941f5a37d728fcb91c5290a"
candidate_commit_reviewed: "77de37b64fbb5030f90da0b9207371e59d440f44"

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
  RR22 is implemented and bound at content commit
  48e9986a3d9e83965584a1eddee32a303cc39ad3 (tree
  32374c00f177d076482f6d5faa4baa5305b8c691). The sealed review was imported
  artifact-only at campaign commit f20a7e48ddbf51b7e0e59bce14647517fbdec441
  without changing its bytes.

  The aggregate post-upload normalizer now launches through the exact already-proven
  `/opt/microsoft/powershell/7/pwsh -NoLogo -NoProfile -NonInteractive -File {0}`
  contract. Its exact MainModule, PSHOME, PS7, absent-profile, sanitized-PATH,
  artifact ID/digest/URL, GITHUB_OUTPUT, and downstream verifier predicates are
  unchanged. Built-in, `/usr/bin`, alternate-target, missing-flag, and command-mode
  launchers reject in the frozen regression.

finding_responses:
  - finding_id: "VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Selecting the already-green literal launcher makes the process identity and
      launch contract agree without accepting a symlink path or weakening any
      attestation, artifact, output-control, or verifier gate.
    changed_files:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "tests/unit/test_via000_r3_identity.py"
    fix_commits:
      - "48e9986a3d9e83965584a1eddee32a303cc39ad3"
    verification:
      - command: "focused RR22 and RR21 launcher/canonicalization gate"
        result: "2 passed, 68 deselected in 0.60 seconds."
      - command: "focused RR22/RR21/RR20/RR15 compatibility gate"
        result: "4 passed, 66 deselected in 11.69 seconds."
      - command: "complete R3 shared-identity test file"
        result: "70 passed in 743.04 seconds after correcting one stale expected built-in-shell count."
      - command: "actionlint 1.7.12, Ruff, JSON/YAML and PowerShell parsing, receipt equality, review guidance, hash audit, and campaign validator"
        result: "passed before handoff sealing; the campaign validator was rerun against the final bound handoff."
    residual_risk: |-
      A fresh exact-handoff hosted proof and independent rereview remain required.
      GitHub's hosted control plane and pinned actions remain trusted principals.
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR22-UBUNTU-PWSH-LAUNCH-IDENTITY-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py::test_r3_rr22_ubuntu_pwsh_launch_identity"
    verification:
      - command: "exact normalizer launcher mutation corpus"
        result: "Only the literal `/opt/.../pwsh -NoLogo -NoProfile -NonInteractive -File {0}` form passes; built-in, `/usr/bin`, alternate target, missing flags, quoted placeholder, and command-mode forms reject."
      - command: "fresh proof-only hosted replay"
        result: "Pending exact handoff branch push; acceptance requires six green producers/caches, green normalizer, one seven-file artifact, and a green redownload verifier."
    rationale: |-
      The regression binds the source and receipt to the launcher proven by prior
      hosted steps while retaining the exact runtime and output side-effect checks.
    disagreement_ref: ""

new_or_changed_risks:
  - "Any Ubuntu PowerShell installation or launch-layout change rejects and requires a reviewed amendment."
  - "Run 32777599858 and artifact 9538560907 remain synthetic and superseded because the hosted verifier was skipped."

external_actions:
  - action: "Inspect or independently replay the fresh exact-handoff six-cell proof, caches, normalized retained artifact identity, and redownload verifier."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: "fresh run recorded outside this artifact after branch push"
  - action: "Perform a fresh independent rereview of RR22 and retained prior controls before any lifecycle action."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Verify exact literal Ubuntu normalizer launch identity, unchanged canonical artifact and verifier predicates, receipt/hash bindings, fresh six-cell retained artifact, and all prior R3 controls."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, scientific execution, commitment, reveal, or viability claims from this response alone."
```
