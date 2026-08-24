# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-3-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-3-RESPONSE-1"
response_round: 1
response_date: "2026-08-23"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-3"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-3.md"
review_commit: "8c2d4c4c4d037bb57b0a462c86db34546e972925"
candidate_commit_reviewed: "2cf2ac2b3b2fe2a22673eee49cd2275048aa38c7"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: |-
    R3 remains drafted, holdout_started=false, unrevealed, and pending. No custody
    directory, sealed manifest, hidden label, secret seed, reveal material, external
    invalid assembled package, or untracked handoff memo was accessed. No production
    authorization key, tag, execution, raw result, commitment, or adjudication was
    created.

summary: |-
  The workflow command-boundary finding is accepted and fixed in content commit
  aab992457139f22fd0c1279f54cb5139e0bad915. Every GitHub expression consumed by a
  PowerShell run step now crosses the boundary only through step environment data.
  The authorization ref is rejected unless it has the exact full-ref prefix, exact
  length, and sixty-four-lowercase-hex suffix, before Git, checkout mutation, guard
  execution, or authorization output. Native commands and Python receive argument
  arrays; the workflow contains no Invoke-Expression, command-string construction,
  Get-Command discovery, or GitHub-expression interpolation in any run source.

  Windows uses only C:\Program Files\Git\cmd\git.exe and
  C:\Windows\System32\OpenSSH\ssh-keygen.exe; Ubuntu uses only /usr/bin/git and
  /usr/bin/ssh-keygen. The base interpreter is supplied only by the exactly pinned
  actions/setup-python Python 3.11.15 output and is validated as an absolute file.
  Missing or wrong-platform tools fail before authorization-object access. PATH is
  not consulted for Git, ssh-keygen, Python, or PowerShell runner selection.

  Two executable controls extract and run the exact production authorization step.
  Fourteen malformed and executable-looking inputs, including the reviewer's exact
  quote-closing payload, reject without a marker, side effect, temporary file, or
  authorization JSON. Empty PATH, one malicious tool-shim directory, and two shim
  directories plus the normal PATH all preserve the exact signed happy path without
  invoking git, ssh-keygen, or python shims. The five earlier remediations and their
  requested controls remain retained without reopening their scopes. The comment-only
  signer placeholder still blocks activation.

finding_responses:
  - finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Untrusted dispatch values are now data, validated before use, and delivered as
      single array elements. Tool identities are explicit reviewed operating-system
      paths rather than dispatcher PATH results. The same invariant covers the first
      dispatch gate, later cleanliness and mutation gates, and all other PowerShell
      run steps receiving GitHub values.
    changed_files:
      - ".github/workflows/via000-r3-protocol.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_via000_r3_identity.py"
      - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
      - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/README.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/amendments/VIA-000-R3-RECOVERY-DESIGN-1.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/PROTOCOL_MANIFEST.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
    fix_commits:
      - "aab992457139f22fd0c1279f54cb5139e0bad915"
    verification:
      - command: "python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r3_identity.py"
        result: "35 passed in 433.39 seconds, including both exact production-step command-boundary controls."
      - command: "reviewer payload and malformed-value corpus against the extracted production PowerShell step"
        result: "All fourteen inputs rejected; the injected stdout marker, filesystem marker, authorization JSON, and runner temporary residue were absent."
      - command: "empty, single-shim, and multiple-shim PATH matrix with a real signed authorization repository"
        result: "All three exact signed paths passed through the reviewed literal tools; no Git, ssh-keygen, or Python shim was invoked."
    residual_risk: "The reviewed system paths and pinned setup-python release are platform trust roots. A hosted-image tool relocation or interpreter-version change requires a separately reviewed protocol amendment. The Ubuntu branch is covered by source assertions here and still requires independent execution on an Ubuntu hosted runner."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py"
    verification:
      - command: "exact workflow run-source extraction with quote, statement, subexpression, variable, newline, backtick, comment, whitespace, option-like, Unicode, control, pipe, ampersand, and quote payloads"
        result: "Every payload remained inert and rejected at the grammar boundary before authorization access, with no marker or side effect."
      - command: "real signed happy path under zero, one, and multiple PATH tool-shim layouts"
        result: "The exact signed path passed three times using fixed binaries, while every shim remained unexecuted."
      - command: "static parse of every workflow run source and both operating-system branches"
        result: "No run source contains a GitHub expression or Get-Command; exact Windows and Ubuntu Git/ssh-keygen paths, pinned Python output transport, and argument arrays are required."
    rationale: "The controls exercise the exact production step source instead of a handwritten approximation and retain the prior authorization, replacement-object, ref-race, cleanup, cross-run, and LF-object coverage in the 35-case aggregate."
    disagreement_ref: ""

new_or_changed_risks:
  - "Supported hosted images must retain the reviewed absolute Git and ssh-keygen paths."
  - "The production authorization signer remains a separate governance trust root requiring a public-key amendment and independent review."
  - "The comment-only allowed-signers artifact deliberately prevents production activation in this draft."

external_actions:
  - action: "Independently rereview the hosted command boundary, literal tool paths, exact extracted-step controls, frozen hashes, and this response."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""
  - action: "Execute the exact workflow boundary controls on the supported Ubuntu hosted image and confirm its reviewed absolute tool paths."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""
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
  scope: "Verify data-only workflow expression transport, pre-Git grammar rejection, literal trusted tool paths, argument arrays, exact shell-source execution, no side effects, and the signed PATH-shim happy path."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, execution, commitment, reveal, or scientific claims from this response alone."
```
