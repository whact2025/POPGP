# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-6-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-6-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-6"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-6.md"
review_commit: "ae3c56708136525bf31a6efd97b22e3b4b819a99"
candidate_commit_reviewed: "032611faafd2c3a3288b987c95d78fa5a03381d5"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: |-
    R3 remains drafted, holdout_started=false, unrevealed, and pending. No custody
    directory, sealed manifest, hidden label, secret seed, reveal material, external
    invalid assembled package, or handoff memo was accessed. No signer key, tag,
    activation, execution, raw result, commitment, or adjudication was created.

summary: |-
  The descendant-attestation finding is accepted and addressed in implementation
  commit e35a5618b4a220d3c581248ad15f08c2651e1806 and final content commit
  6036cab8c021f5fbef7e51a3662ce4f147a633bc (tree
  33c4cbc1fd0dfb8d6da26b5a83a193d7a1c8c027). The existing six fresh hosted stage
  jobs remain. Within each job, every candidate- or mutation-controlled command now
  executes behind an OS-enforced privilege and process-tree boundary.

  Windows creates a restricted low-integrity token, creates the process suspended,
  assigns it to a kill-on-close Job Object before resume, terminates the Job Object,
  and requires zero active processes. Ubuntu uses a systemd transient service with
  DynamicUser, control-group kill, and explicit post-stop cgroup emptiness checks.
  Both paths fail closed if their required primitive is unavailable.

  Untrusted processes can write only fresh mutable staging. Trusted tool/configuration
  and evidence roots are not writable to that identity; Windows trusted evidence also
  uses no-read-up/no-write-up integrity policy. Child TEMP/TMP/RUNNER_TEMP, home,
  cache, loader, Git, Python, uv, and TeX surfaces are scrubbed or redirected to
  mutable staging. Trusted evidence creation, closure validation, summary/manifest
  production, and attestation subject capture happen only after whole-tree teardown
  and a zero-descendant proof. Subject digests are rechecked after attestation and
  after the retained bundle is formed.

  The same boundary closes the previously accepted RR4 same-path window: adversarial
  code cannot write the protected executable/configuration closure, while the trusted
  runner verifies closure before and after use and only records evidence after the
  untrusted tree is gone. The assembler requires every real retained containment
  result and rejects forged privilege, teardown, quiescence, tool protection, evidence
  protection, command count, or platform primitive claims before output commitment.

  The production workflow's first execution gate on both hosted OS families launches
  a real child-of-child delayed writer that attempts direct, hard-link, and
  replace-then-restore writes against live tool/evidence paths after the direct child
  returns. The exact Windows production helper was also executed locally and killed
  the detached tree with no marker or subject change. Ubuntu enforcement is encoded as
  a mandatory hosted gate but could not be executed on this Windows workstation.

finding_responses:
  - finding_id: "VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The final trusted evidence boundary no longer depends on waiting for a direct
      child. Whole-tree OS containment and lower-privilege write separation prevent a
      candidate-controlled descendant from surviving into, or writing through, final
      evidence creation and attestation. The exact production wrapper retains and the
      assembler recomputes command-by-command quiescence evidence.
    changed_files:
      - ".github/workflows/via000-r3-protocol.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
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
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-protocol.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/mutation-runner.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/assembler-protocol.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/raw-results.schema.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
    fix_commits:
      - "e35a5618b4a220d3c581248ad15f08c2651e1806"
      - "6036cab8c021f5fbef7e51a3662ce4f147a633bc"
    verification:
      - command: "focused production containment, exact assembly, verifier, tool identity, and hostile stage aggregate"
        result: "15 passed in 246.79 seconds; live Windows containment, three forged-boundary cases, five stage-injection cases, and the exact six-fragment happy path passed."
      - command: "exact post-format workflow-source and live Windows production containment controls"
        result: "2 passed in 8.28 seconds; the child-of-child delayed writer left no marker and changed neither protected subject."
      - command: "full R3 identity test file followed by exact correction rerun"
        result: "Initial run found two narrow validator-compatibility assertions after 49 passes; both were corrected and their exact two-test rerun passed in 108.38 seconds."
      - command: "Ruff plus PowerShell, Python, JSON, and YAML parsing"
        result: "All changed program sources linted or parsed successfully."
      - command: "python -m pytest -q -p no:cacheprovider tests/unit/test_review_guidance.py"
        result: "9 passed in 10.08 seconds with the 430-test ledger and negative-control registry reconciled."
      - command: "python scripts/check_tex.py"
        result: "TeX source is balanced and valid."
      - command: "python -m scripts.check_viability_campaign reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
        result: "Viability campaign contract is valid against content commit 6036cab8c021f5fbef7e51a3662ce4f147a633bc."
    residual_risk: "Fresh hosted Windows and Ubuntu execution plus independent rereview are still required. The Ubuntu systemd/DynamicUser path was source-parsed but cannot run locally on this Windows host. The GitHub control plane and pinned action revisions remain declared bootstrap principals. The comment-only signer and fresh custody verification still block activation."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR6-DESCENDANT-ATTESTATION-TOCTOU-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py::test_r3_workflow_requires_production_containment_and_atomic_subject_capture"
      - "tests/unit/test_via000_r3_identity.py::test_r3_windows_production_containment_kills_detached_replace_restore_tree"
      - "tests/unit/test_via000_r3_identity.py::test_r3_assembler_rejects_forged_execution_boundary"
      - ".github/workflows/via000-r3-protocol.yml"
    verification:
      - command: "production workflow hostile gate on the six hosted stage jobs"
        result: "The required first execution gate starts a real detached child-of-child replace/restore payload and requires containment, protected subjects, zero descendants, and no marker before candidate execution."
      - command: "local exact Windows Job Object production wrapper"
        result: "The direct child returned after its grandchild was live; Job Object teardown killed the tree and preserved protected tool/evidence subjects."
      - command: "unmodified assembler boundary falsification"
        result: "Missing privilege separation, false descendant quiescence, or nonzero active descendants were rejected without changing later-stage subjects or creating output/commitment."
    rationale: "The new controls exercise the production wrapper and live descendant timing, not a completed-fixture approximation. The workflow makes the same hostile gate mandatory on Ubuntu; fresh hosted execution is assigned to rereview because the current workstation cannot execute that OS primitive."
    disagreement_ref: ""

new_or_changed_risks:
  - "The security boundary now depends on Windows restricted-token/Job Object integrity enforcement and Ubuntu systemd DynamicUser/control-group enforcement; unavailable primitives intentionally invalidate a run."
  - "Fresh hosted Ubuntu and Windows execution is required to validate runner-image permissions and service-manager policy."
  - "The GitHub control plane and exact pinned action revisions remain trusted bootstrap principals."
  - "No full 430-test suite was rerun after the surgical remediation; the fresh rereviewer should run it if feasible."

external_actions:
  - action: "Independently rereview the real containment boundary, live hostile workflow gate, retained quiescence evidence, subject-digest chronology, and prior controls on fresh hosted Windows and Ubuntu runners."
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
  scope: "Verify real Windows restricted-token/Job Object and Ubuntu systemd DynamicUser/control-group containment, lower-privilege tool/evidence protection, live detached child-of-child and replace/restore attacks, complete teardown before evidence/subject capture, subject digest rechecks after attestation and bundle retention, assembler containment-proof enforcement, exact six-stage happy path, and preservation of all earlier identity and stage-isolation controls."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, execution, commitment, reveal, or scientific claims from this response alone."
```
