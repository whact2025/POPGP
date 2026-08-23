# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-RESPONSE-1"
response_round: 1
response_date: "2026-08-23"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1.md"
review_commit: "beba0f19ea7979c1676b45e2912b1ad6a2c3ba4f"
candidate_commit_reviewed: "46f33080eae18f4692624d3172027635d72aff97"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: |-
    R3 remains drafted, holdout_started=false, unrevealed, and pending. No custody
    directory, sealed manifest, secret seed, hidden label, reveal material, external
    invalid assembled package, or untracked handoff memo was accessed. No production
    signer key, tag, attack, raw result, output commitment, or adjudication was made.

summary: |-
  All three reported blockers are accepted and remediated in content commit
  f00115330166a77ccc9630b4773dc21700b76377. A lightweight source tag now identifies
  a protocol snapshot, while a distinct SSH-signed annotated authorization tag is
  content-addressed by its canonical record and points to immutable campaign, packet,
  and manifest bytes that independently authorize that exact snapshot. The guard and
  assembler reject a later correctly suffixed lifecycle tag because it is not the
  snapshot selected by the signed authorized packet.

  The assembler no longer imports the semantic validator from a caller-selected
  repository path. It extracts the validator and its complete local dependency
  closure from the authorized source snapshot, verifies each byte against both the
  primary protocol and authorized manifest, supplies the real authorized packet, and
  executes the extracted bundle under Python isolated mode before any output rename
  or commitment. Worktree substitution of every declared local dependency fails.

  Targeted LF attributes now make the workflow and validator dependency blobs exact
  on Windows with core.autocrlf both true and false. Frozen source/receipt hashes and
  the R3 protocol manifest were regenerated from Git-normalized bytes. The seven
  drafted packets and campaign bind the remediation content commit only as a draft
  handoff. This is not activation, preregistration, or approval.

  The allowed-signers artifact is deliberately comment-only. Activation remains
  blocked until a separately reviewed amendment freezes exactly one Ed25519 public
  authorization key. No production authorization tag can pass in the current draft.

finding_responses:
  - finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The workflow accepts only manual dispatch and takes a signed authorization ref,
      not a caller-authored snapshot commit. The snapshot guard loads its authorization
      contract and signer from the source Git object, verifies the signed canonical
      record and immutable target, hashes authorized campaign/packet/manifest blobs,
      and requires both campaign and packet protocol_commit fields to select the exact
      source-tag commit. The assembler independently repeats this guard and derives
      its source identity from the authorized bytes.
    changed_files:
      - ".github/workflows/via000-r3-protocol.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-DISPATCH-GUARD.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/authorization/VIA-000-AUTHORIZED-SIGNERS"
      - "tests/unit/test_via000_r3_identity.py"
    fix_commits:
      - "f00115330166a77ccc9630b4773dc21700b76377"
    verification:
      - command: "python -m pytest -q tests/unit/test_via000_r3_identity.py"
        result: "23 R3 identity cases passed, including real disposable SSH-signed authorization repositories."
      - command: "exact later-lifecycle and four authorization-ref mutation cases"
        result: "A later correctly suffixed tag and moved, deleted, wrong-suffix, or wrong-kind refs were rejected; no output survived."
    residual_risk: "The authorization signer is a privileged governance root and must be selected and reviewed separately before activation."
    disagreement_ref: ""

  - finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The frozen assembler extracts the validator, validation-artifact checker,
      reproduction-boundary checker, diagnostics module, and package initializers
      from the authorized snapshot. It checks exact primary and authorized-manifest
      hashes, rejects a differing worktree copy, and invokes the isolated extracted
      checker with the real packet blob before output commitment.
    changed_files:
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-VALIDATOR-PACKAGE-INIT.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "scripts/check_viability_campaign.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/PROTOCOL_MANIFEST.json"
      - "tests/unit/test_via000_r3_identity.py"
    fix_commits:
      - "f00115330166a77ccc9630b4773dc21700b76377"
    verification:
      - command: "validator source-closure substitutions plus isolated extracted-bundle execution"
        result: "Four dependency substitutions were rejected 4/4; the exact extracted bundle executed with the real packet and rejected invalid raw results 1/1."
      - command: "source/receipt/manifest SHA-256 reconciliation"
        result: "Every R3 execution and validator artifact matches its frozen receipt; the authorized manifest binds the complete local validator closure."
    residual_risk: "The isolated checker still trusts the declared base Python and installed third-party packages under the existing frozen environment contract."
    disagreement_ref: ""

  - finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Targeted .gitattributes rules force LF for the hosted workflow and every frozen
      local validator dependency. The fixture now verifies the committed Git blob
      against fresh checkout bytes under core.autocrlf=true and false, and the exact
      signed authorization happy path succeeds on Windows.
    changed_files:
      - ".gitattributes"
      - ".github/workflows/via000-r3-protocol.yml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
      - "tests/unit/test_via000_r3_identity.py"
    fix_commits:
      - "f00115330166a77ccc9630b4773dc21700b76377"
    verification:
      - command: "cross-platform Git-blob identity fixtures with core.autocrlf=true,false"
        result: "Both variants passed 2/2; checkout bytes equaled LF Git object bytes."
      - command: "python -m pytest -q tests/unit/test_review_guidance.py"
        result: "9 passed; the documented complete-suite count is 402."
    residual_risk: "A Linux independent re-review run remains required before any approval."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py"
    verification:
      - command: "signed authorization happy path, later-lifecycle attack, and authorization-ref mutation matrix"
        result: "The unique authorized snapshot passed; every tested self-consistent later commit or substituted ref failed closed."
    rationale: "Real temporary Git repositories, Ed25519 keys, lightweight source tags, signed annotated authorization tags, and unmocked identity guards exercise the authority boundary."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py"
    verification:
      - command: "four dependency substitutions and exact extracted validator execution"
        result: "All substituted worktree dependencies failed and the accepted path executed snapshot bytes with the real packet before commitment."
    rationale: "The tests cover the complete declared local validator closure, source-manifest checks, actual isolated process execution, and output-free rejection."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py"
      - ".gitattributes"
    verification:
      - command: "fresh clone byte comparison with core.autocrlf=true and core.autocrlf=false"
        result: "Both checkout variants exactly matched the LF workflow Git blob; the R3 exact signed-snapshot happy path passed on Windows."
    rationale: "The regression uses Git materialization rather than assuming worktree line endings and freezes the workflow and validator source byte model."
    disagreement_ref: ""

new_or_changed_risks:
  - "The authorization signer becomes a governance trust root; key selection, storage, rotation, and public-key amendment require separate independent review."
  - "The production allowed-signers file is intentionally empty, so the current draft cannot be activated or dispatched successfully."
  - "The isolated semantic validator continues to rely on the existing pinned Python and dependency environment boundary."

external_actions:
  - action: "Independently re-review the remediation content, this response, and the draft binding commit."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""
  - action: "Propose and independently review an amendment freezing exactly one Ed25519 authorization public key."
    owner: "maintainer-and-independent-reviewer"
    status: pending
    evidence_ref: ""
  - action: "After approval only, construct the final snapshot, authorization commit, lightweight source tag, and signed annotated authorization tag."
    owner: "authorized-maintainer-and-signer"
    status: pending
    evidence_ref: ""
  - action: "Perform fresh R3 custody carry-forward verification before any preregistration or holdout transition."
    owner: "evaluator-custodian-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Re-review the three accepted remediation findings, requested negative controls, signer-key activation blocker, Git-normalized frozen hashes, and drafted campaign bindings."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not approve activation, preregistration, tags, custody transition, execution, commitment, reveal, or scientific claims from this response alone."
```
