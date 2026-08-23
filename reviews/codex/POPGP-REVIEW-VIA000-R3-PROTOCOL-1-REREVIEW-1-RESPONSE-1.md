# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1-RESPONSE-1"
response_round: 1
response_date: "2026-08-23"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1.md"
review_commit: "e083dc79c204d40d218e10436e65ebfa06e79345"
candidate_commit_reviewed: "38ae35619ca8027c570d87868c48564dc109e7f6"

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
  The new authorization-reference race is accepted and fixed in content commit
  7a0cf295d874e6d181399d98ca77f4edd9d12a26. The workflow and assembler each
  capture the authorization ref as one full tag object ID before invoking the frozen
  guard. The guard requires that expected object, then uses only the captured ID for
  object type checking, message parsing, target peeling, SSH signature verification,
  and returned identity. It finally proves the ref still resolves to that same object.
  Workflow runner and mutation steps also compare the current ref with the verified
  returned object immediately before execution.

  The reviewer-shaped regression constructs one same-name/same-target tag object with
  the canonical authorization record and an invalid signature, plus a different valid
  signed tag object with an unrelated message. Swapping the ref after the invalid
  object is parsed cannot borrow the second object's signature: verification remains
  pinned to the invalid captured object and rejects. Deletion after parsing, movement
  after peeling, substitution before verification, and an assembler post-capture swap
  also reject without output. The unmoved exact signed-object control passes.

  The rereview's three prior finding outcomes remain verified-resolved, and its three
  prior requested-test outcomes remain verified-satisfied. This response changes none
  of those scientific, validator-source, or cross-platform byte contracts. All seven
  packets and the campaign bind the new content commit only as a drafted rereview
  handoff. The comment-only signer placeholder remains unchanged, so activation is
  still blocked pending a separate signer-key amendment and independent review.

finding_responses:
  - finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      No security-critical operation re-resolves the mutable authorization ref after
      capture. The exact captured tag OID supplies the parsed record, tag target,
      verified SSH signature, and retained identity. The ref name is used afterward
      only for equality checks against that OID. The assembler independently captures
      an OID and requires the extracted guard to return the same object.
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
    fix_commits:
      - "7a0cf295d874e6d181399d98ca77f4edd9d12a26"
    verification:
      - command: "python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r3_identity.py"
        result: "28 passed in 314.53 seconds, including five new deterministic race cases."
      - command: "reviewer-shaped invalid-object/same-name valid-object swap plus direct valid captured-object control"
        result: "The swapped invalid captured object rejected; the exact unmoved signed object passed."
      - command: "parse deletion, peel movement, signature-stage swap, and assembler post-capture swap"
        result: "All four paths rejected; no authorization JSON, assembled directory, temporary assembly, raw result, or commitment survived."
    residual_risk: "A repository principal may still move a ref, but movement cannot change the already authenticated record or object identity; the workflow aborts if movement is observed before execution. The separately selected production signer remains a privileged governance root."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py"
    verification:
      - command: "deterministic real-Git captured-object race matrix"
        result: "One invalid canonical-record object could not borrow a different valid object's signature; delete/move/swap stages rejected and the exact valid object passed."
      - command: "extracted assembler guard post-capture substitution"
        result: "The independently captured assembler OID disagreed with the moved ref and the frozen guard rejected before output."
    rationale: "The regression uses real Git tag objects, a disposable Ed25519 trust root, a corrupted signature, a separately valid unrelated signature, exact object peeling, and the production guard decision logic."
    disagreement_ref: ""

new_or_changed_risks:
  - "The ref may move after the final equality check, but all authorization semantics and retained identity are already bound to the verified immutable object ID and no later component derives authority from the ref name alone."
  - "The production authorization signer remains a governance trust root requiring a separate public-key amendment and independent review."
  - "The comment-only allowed-signers artifact deliberately prevents production activation in this draft."

external_actions:
  - action: "Independently rereview the captured-object implementation, deterministic race tests, frozen hashes, and this response."
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
  scope: "Verify that all authorization operations remain on one captured tag object, the final and pre-execution ref checks fail closed, and the deterministic race reproducer can no longer combine two objects."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, execution, commitment, reveal, or scientific claims from this response alone."
```
