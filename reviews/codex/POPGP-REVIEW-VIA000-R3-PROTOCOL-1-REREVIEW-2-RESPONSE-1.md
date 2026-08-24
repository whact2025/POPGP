# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2-RESPONSE-1"
response_round: 1
response_date: "2026-08-23"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2.md"
review_commit: "766b147438e5e5ba8da5aa87cb20a721b79aeb62"
candidate_commit_reviewed: "7e194f89f789a89ec459dfb7504dbdf6a3f901ed"

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
  The Git replacement-object finding is accepted and fixed in content commit
  47fb5477ac4d8a44b8e5edc22230f383583b5409. Every security-critical Git operation
  in the workflow, frozen dispatch guard, unmodified assembler, snapshot extraction,
  campaign/packet/manifest reads, validator extraction, and validator precommit path
  now uses --no-replace-objects and GIT_NO_REPLACE_OBJECTS=1. Each subprocess removes
  all inherited GIT_* controls before installing a minimal fail-closed Git environment
  that disables system/global config, prompts, replacement objects, and optional
  locks. Repository access is supplied only by the resolved explicit repo root.

  Git and SSH verification programs are selected from fixed operating-system paths,
  not caller PATH. SSH tag verification additionally overrides gpg.format,
  gpg.ssh.allowedSignersFile, and gpg.ssh.program on the exact command line. Thus
  caller Git config, local gpg.ssh.program, replacement namespaces, repository/worktree
  redirection, object directories, alternates, namespace/discovery controls, SSH
  commands, and Git exec paths cannot alter the authenticated bytes or verifier.

  Five new real-Git controls cover the reviewer's invalid-tag/default-replacement
  reproducer in both the guard and unmodified assembler, a custom replacement
  namespace, broad environment/config/program injection, and replacements over the
  source commit plus campaign, packet, manifest, and validator blobs. Invalid original
  tags reject without authorization JSON, temporary assembly, output, or commitment;
  valid originals and source blobs remain accepted because replacements are ignored.
  The four prior resolved findings and four prior satisfied requested tests remain
  unchanged. The comment-only signer placeholder still blocks activation.

finding_responses:
  - finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      A full object ID is now interpreted only through no-replacement Git plumbing.
      The same scrubbed environment and absolute system Git executable cover type
      checks, record parsing, peeling, signature verification, campaign/packet/
      manifest reads, verifier extraction, and precommit reads. The separately pinned
      system ssh-keygen and command-line allowed-signers configuration prevent local or
      inherited signature-program substitution.
    changed_files:
      - ".github/workflows/via000-r3-protocol.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-DISPATCH-GUARD.py"
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
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/assembler-protocol.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/dispatch-guard.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
    fix_commits:
      - "47fb5477ac4d8a44b8e5edc22230f383583b5409"
    verification:
      - command: "python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r3_identity.py"
        result: "33 passed in 367.76 seconds, including all five new real-Git replacement and environment controls."
      - command: "invalid canonical tag replaced by the original valid signed tag"
        result: "The frozen guard and unmodified assembler both rejected the invalid original; no authorization JSON, temporary assembly, output, or commitment survived."
      - command: "custom replacement namespace, repository/object/config/program injection, and source/blob replacement matrix"
        result: "The invalid custom-namespace tag rejected; valid exact objects passed while every injected replacement and Git control was ignored."
    residual_risk: "The fixed operating-system Git and ssh-keygen paths are deliberate platform trust roots. Changing supported runner images or tool locations requires a separately reviewed protocol amendment. The production authorization signer remains a separate privileged governance root."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py"
    verification:
      - command: "real default and custom replacement-ref controls"
        result: "Default and environment-selected replacement mappings could not lend valid bytes or a valid signature to an invalid captured tag OID."
      - command: "real source/campaign/packet/manifest/validator replacement controls"
        result: "The guard, assembler, and isolated validator used the original Git objects; replacement bytes did not enter identity or precommit validation."
      - command: "caller Git environment/config/program injection control"
        result: "Repository/worktree/index/object/alternate/namespace/discovery/config/SSH/exec/PATH injection did not alter the exact signed-object happy path or invoke the configured local program."
    rationale: "The tests use real repositories, tag objects, Ed25519 signatures, replacement refs, custom replacement namespaces, commit/blob replacements, and the unmodified production guard/assembler subprocess boundaries."
    disagreement_ref: ""

new_or_changed_risks:
  - "The supported Ubuntu and Windows system Git/ssh-keygen locations are now explicit fail-closed platform requirements."
  - "The production authorization signer remains a governance trust root requiring a separate public-key amendment and independent review."
  - "The comment-only allowed-signers artifact deliberately prevents production activation in this draft."

external_actions:
  - action: "Independently rereview no-replacement Git plumbing, environment/config isolation, frozen hashes, tests, and this response."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""
  - action: "Propose and independently review an amendment freezing exactly one Ed25519 authorization public key."
    owner: "maintainer-and-independent-reviewer"
    status: pending
    evidence_ref: ""
  - action: "After protocol approval only, create the final snapshot, authorization commit, lightweight source tag, and signed authorization tag."
    owner: "authorized-maintainer-and-signer"
    status: pending
    evidence_ref: ""
  - action: "Perform fresh fourteen-of-fourteen R3 custody carry-forward verification before preregistration or holdout."
    owner: "evaluator-custodian-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Verify that all authorization and extraction Git operations disable replacements and caller Git controls, that signature verification uses the fixed program/config, and that real default/custom replacement attacks reject without output."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, execution, commitment, reveal, or scientific claims from this response alone."
```
