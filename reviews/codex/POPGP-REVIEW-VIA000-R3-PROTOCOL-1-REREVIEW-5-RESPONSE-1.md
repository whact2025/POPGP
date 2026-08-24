# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-5-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-5-RESPONSE-1"
response_round: 1
response_date: "2026-08-23"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-5"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-5.md"
review_commit: "7596948eab9b339cdb97332cd1ee047d40adeb27"
candidate_commit_reviewed: "2c4882f61662cb5f6f7362ee23c182c11176cd57"

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
  The execution-context finding is accepted and addressed in content commit
  3a7d55e4797627765cd01801310a39fe58c655eb (tree
  49221426e9d2d539611fe21917d4a3c06fe13824). Candidate tests/generators, PDF
  production, and frozen mutation verification now occupy six fresh GitHub-hosted
  matrix jobs: three independent VM contexts on each supported platform. No writable
  tool, configuration, environment, cache, temporary path, or process state crosses
  a stage boundary.

  The PDF stage checks out the exact candidate without building or executing a
  candidate Python environment. It clears TeX/TEXMF/kpathsea/font/native-loader
  selectors, disables shell escape, and retains byte-identical before/after SHA-256
  manifests of the complete pinned TeX Live tree. The mutation stage independently
  clones the candidate and rebuilds the frozen environment after the candidate job
  has ended. Each stage signs a stage summary and evidence manifest bound to the same
  repository, workflow, source, authorization, run, attempt, candidate, platform,
  and explicit stage identity.

  The assembler requires all three stages per platform, verifies all six attestations
  and manifests, rejects missing or mixed identity, and accepts no undeclared,
  executable, configuration, symlink, or reparse artifact state. It derives each
  final platform record only from the assigned candidate, PDF, and mutation evidence
  and cleans all temporary output on failure. Five hostile cross-stage payload
  variants and the six-fragment exact-snapshot happy path pass. All prior identity,
  authorization, replacement-object, command-boundary, validator-source, LF, and
  cleanup controls remain green in the 46-case R3 aggregate.

  The earlier same-path executable replacement finding remains unresolved inside an
  individual stage. Stage separation prevents candidate state from affecting the PDF
  or mutation stages, but this response does not claim atomic check/use identity for
  every repeated executable open within one stage. Activation therefore remains
  blocked pending fresh independent review and separate resolution or explicit
  disposition of that earlier blocker.

finding_responses:
  - finding_id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Fresh VM boundaries remove candidate access to later PDF and mutation tools,
      state, caches, configuration, temporary directories, and processes. The PDF
      stage has no candidate Python/install step and binds the complete TeX root plus
      relevant environment closure. Signed stage fragments and evidence-only
      transport make the assembler independently require the complete same-run set.
    changed_files:
      - ".github/workflows/via000-r3-protocol.yml"
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
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/mutation-runner.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/assembler-protocol.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/raw-results.schema.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
    fix_commits:
      - "15af20e1333d8aad7dfba4de2696caf2c73b0bc8"
      - "f20d0918f71e991c7b75ec448b9c906d9d87fb99"
      - "3a7d55e4797627765cd01801310a39fe58c655eb"
    verification:
      - command: "python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r3_identity.py"
        result: "46 passed in 469.89 seconds, including all prior R3 controls and the new staged cases."
      - command: "focused stage topology, five hostile cross-stage payloads, exact six-fragment happy path, and identity/attestation/cross-run negatives"
        result: "10 passed in 192.50 seconds; every hostile artifact rejected with no output or temporary commitment and later-stage hashes unchanged."
      - command: "Ruff plus JSON, YAML, Python, and PowerShell source parsing"
        result: "All selected sources passed lint and parsed successfully."
      - command: "python -m pytest -q -p no:cacheprovider tests/unit/test_review_guidance.py"
        result: "9 passed in 11.72 seconds after reconciling the 425-test ledger and negative-control registry."
      - command: "python scripts/check_tex.py"
        result: "TeX source is balanced and valid."
      - command: "python -m scripts.check_viability_campaign reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
        result: "Viability campaign contract is valid against final content commit 3a7d55e4797627765cd01801310a39fe58c655eb."
    residual_risk: "Fresh hosted Windows and Ubuntu execution is required during independent rereview. GitHub control plane and exact pinned actions remain declared bootstrap principals. Hosted-image base configuration outside the action TeX root remains trusted and is recorded by image identity. The prior intra-stage same-path executable check/use race remains unresolved. The deliberately empty signer and fresh custody verification still block activation."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py::test_r3_workflow_separates_candidate_pdf_and_mutation_execution_contexts"
      - "tests/unit/test_via000_r3_identity.py::test_r3_stage_isolation_rejects_cross_stage_state_injection"
    verification:
      - command: "exact production workflow topology source check"
        result: "Both Windows and Ubuntu contain distinct candidate, PDF, and mutation hosted jobs; TeX setup is PDF-only and mutation execution is mutation-only."
      - command: "self-restored pdfTeX/tool, TeX configuration, background watcher, runner-temp closure, and executable artifact payload variants"
        result: "All five variants rejected before output/commitment; independently rooted PDF and mutation fragments remained byte-identical."
      - command: "exact six-fragment assembler control"
        result: "All same-run identity-bound candidate/PDF/mutation fragments assembled successfully in the controlled fixture."
    rationale: "The regression exercises the exact workflow stage topology and unmodified assembler artifact boundary, including complete cleanup, while preserving the existing signed-identity and exact-snapshot controls. Hosted cross-platform execution remains assigned to the fresh rereviewer."
    disagreement_ref: ""

new_or_changed_risks:
  - "The prior RR4 same-path check/use race within one stage remains unresolved and blocking."
  - "The stage model trusts GitHub to provide genuinely fresh hosted VMs and trusts the exact pinned action revisions."
  - "The complete action-managed TeX tree is byte-manifested; fresh hosted-image base libraries/configuration outside that tree remain part of the declared hosted principal."
  - "A missing stage, changed stage identity, undeclared artifact, or hosted-tool relocation intentionally invalidates execution and requires review."

external_actions:
  - action: "Independently rereview the six-job topology, TeX closure, stage attestations, assembler merge, cleanup, and this response on both supported hosted images."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""
  - action: "Resolve or explicitly adjudicate the still-open intra-stage same-path executable identity race before activation."
    owner: "builder-and-independent-reviewer"
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
  scope: "Verify fresh candidate/PDF/mutation hosted contexts on Windows and Ubuntu; no candidate Python in PDF; TeX/loader scrub, no-shell-escape, and complete before/after TeX manifest; same-run stage attestations; evidence-only artifact closure; assembler reconciliation and cleanup; preservation of prior controls; and the explicitly unresolved intra-stage same-path risk."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, execution, commitment, reveal, or scientific claims from this response alone."
```
