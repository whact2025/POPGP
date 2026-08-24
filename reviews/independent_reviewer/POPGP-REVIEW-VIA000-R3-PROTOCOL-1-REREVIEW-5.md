# VIA-000 R3 recovery-protocol independent re-review 5

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-5"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex (GPT-5)"
reviewer_model_version: "unknown"
reviewer_operator: 'NVIDIA.COM\rfuoco'
reviewer_session_id: "popgp-via000-r3-protocol-independent-rereview-session-5"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-23"
commit_reviewed: "2c4882f61662cb5f6f7362ee23c182c11176cd57"
baseline_commit: "4a966a6e8be15f04a2a2d1c4dffa251c0d0f2333"
prior_review_ref: "4a966a6e8be15f04a2a2d1c4dffa251c0d0f2333:reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-4.md"
builder_response_ref: "2c4882f61662cb5f6f7362ee23c182c11176cd57:reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-4-RESPONSE-1.md"
context_hash: "3159b7ecd5a540a21eaa270e53d5f9b8c73f59a8"
context_hash_method: 'git rev-parse "2c4882f61662cb5f6f7362ee23c182c11176cd57^{tree}"'
files_reviewed:
  - ".github/workflows/via000-r3-protocol.yml"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/PROJECT_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-ASSEMBLER.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-DISPATCH-GUARD.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1"
  - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
  - "reviews/codex/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-4-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-2.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-3.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-4.md"
  - "reviews/viability/POPGP-VIABILITY-R2-2026-08/attempts/VIA-000-R2-INVALID-ATTEMPT-1.md"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/PROTOCOL_MANIFEST.json"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/authorization/VIA-000-AUTHORIZED-SIGNERS"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1"
  - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_review_guidance.py"
  - "tests/unit/test_via000_r2_assembler.py"
  - "tests/unit/test_via000_r3_identity.py"
  - "tests/unit/test_viability_raw_evidence_contract.py"
access_level: "public-repository-only plus local Windows execution; no custody, reveal, external invalid-package, or handoff-memo access"
independence_statement: |-
  Fresh artifact-only review in isolated worktree
  C:\src\POPGP-via000-r3-protocol-rereview-5 and dedicated branch
  review/via000-r3-protocol-rereview-5. Exact commits, trees, ancestry, response
  schema, and prior-review immutability were checked before review. Builder claims
  were hypotheses. No implementation, protocol, packet, signer, ref, lifecycle,
  evidence, commitment, custody, result, or reveal material was changed or accessed.
  Operator and orchestrator are shared; session, branch, and worktree are distinct.
  Builder model is unknown, so model separation and external validation are not claimed.
independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "unknown"
  builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false
hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
summary: |-
  CHANGES REQUESTED. The remediation fixes the original bare-PATH defect: workflow
  shells are explicit/no-profile, tool roots are fixed, children receive absolute
  paths, leaf hashes and versions are retained, failure cleanup is present, upload is
  success-only, and the focused 40 tests pass. The first six findings remain resolved.

  RR4's explicitly requested same-path replacement case is still open. The execution
  step rechecks only Git, then invokes the previously hashed PowerShell path before
  runner validation. The mutation step invokes base Python without rechecking its
  digest and validates environment Python only after mutation. The runner hashes
  tools once, creates output, then repeatedly reopens mutable paths without atomic
  use-time identity. This keeps RR4 and its requested test unresolved.

  A distinct in-scope execution-context gap also remains. Candidate Python executes
  before both pdfTeX passes, inherits RUNNER_TEMP, and can write the TeX installation
  deliberately placed beneath RUNNER_TEMP. It can replace the already-hashed pdfTeX
  leaf or alter its unmeasured format/config/input closure, then self-clean candidate
  and environment paths. The later passes neither rehash the leaf nor bind loaded TeX
  closure. The short environment denylist also preserves TeX/loader configuration,
  and pdfTeX is not passed `-no-shell-escape`. The declared GitHub/action trust
  principal does not include candidate code, so this is not promoted merely from a
  hypothetical malicious control plane.

  R2 is immutable. R3 remains drafted, holdout_started=false, unrevealed, and pending.
  The placeholder signer blocks activation; no production key, tag, run, output, or
  commitment exists. This review authorizes no merge, signer amendment, activation,
  refreeze, tag, preregistration, holdout, assembly, reveal, or scientific claim.
findings:
  - id: "VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001"
    severity: critical
    category: code
    location: ".github/workflows/via000-r3-protocol.yml:128-164,259-347,377-430,494-549; protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1:78-109,111-169,226-240,265-310,486-501; scripts/check_viability_campaign.py:1465-1586"
    evidence: |-
      The exact runner installs TeX beneath RUNNER_TEMP, hashes only pdfTeX at
      preflight, then executes all candidate tests at command 006 before pdfTeX
      commands 014/015. Retained candidate processes inherit RUNNER_TEMP, and the TeX
      tree is runner-user writable. Candidate Python can therefore replace the
      already-accepted pdfTeX bytes or its adjacent format/config/input dependencies,
      without touching the candidate repository or snapshotted Python environment.
      The later pdfTeX calls perform no digest recheck and the validator binds only
      the pdfTeX leaf hash recorded before candidate execution.

      ProcessStartInfo also preserves TEXINPUTS/TEXMF*/shell_escape, LD_PRELOAD,
      LD_LIBRARY_PATH, and related configuration; pdfTeX has no `-no-shell-escape`.
      The evidence validator checks typed paths/banners/hash shape, not an independent
      frozen TeX/config/dependency closure. This reproducer uses candidate-controlled
      execution between check and use and does not assume compromise of the explicitly
      trusted GitHub control plane or pinned actions.
    finding: |-
      Absolute leaf paths and file hashes do not close the actual execution context;
      inherited manager, TeX, loader, and subprocess state can alter loaded code and
      behavior while every recorded executable digest remains unchanged.
    failure_scenario: |-
      A candidate pytest case writes a replacement pdfTeX or modified format/config
      into the known RUNNER_TEMP TeX tree, while leaving the repository and Python
      environment unchanged (or restoring them before boundary verification). The
      later PDF passes execute/load the replacement with no recheck. The signed
      manifest remains self-consistent because it retains the pre-candidate leaf hash.
    consequence: |-
      Candidate execution, dependencies, PDF output, and attestable evidence are not
      functions solely of authorization-bound bytes and declared principals.
    required_action: |-
      Prevent candidate writes to every later executable/config/dependency closure,
      or copy it into a protected reviewer-owned location and bind/recheck the complete
      closure immediately at use. Explicitly disable TeX shell escape and sanitize
      TeX/loader state. State which hosted libraries/configuration are trusted and
      align claims/validation. Add exact Windows/Linux candidate tests that replace
      pdfTeX and its format/config closure between preflight and commands 014/015;
      require rejection, cleanup, and zero upload/commitment.
    verification: confirmed-by-execution
    blocking: true
requested_tests:
  - id: "TST-VIA000-R3-RR5-EXECUTION-CONTEXT-CLOSURE-001"
    description: |-
      Run the exact production sequence with a candidate pytest payload that replaces
      the RUNNER_TEMP pdfTeX leaf, format, configuration, and input dependencies after
      preflight and self-cleans candidate/environment state. Also probe hostile TeX
      and native-loader variables with unchanged executable leaf bytes. Reject before
      the payload runs or before any PDF/output/upload/commitment; pass exact Windows
      and Ubuntu paths from a protected, independently bound TeX/tool closure.
    rationale: "A leaf digest cannot identify code selected or loaded through ambient configuration."
    blocking: true
prior_finding_results:
  - finding_id: "VIA000-R3-P1-SNAPSHOT-AUTH-001"
    outcome: verified-resolved
    evidence: "Signed authority, lifecycle, source, campaign, packet, manifest, signer, and cleanup cases remain green."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No regression."
  - finding_id: "VIA000-R3-P1-VALIDATOR-SOURCE-001"
    outcome: verified-resolved
    evidence: "Git-object verifier closure, real packet, dependency/worktree substitution, cross-run, and cleanup cases remain green."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Execution context precedes this closure."
  - finding_id: "VIA000-R3-P1-WINDOWS-HAPPY-PATH-001"
    outcome: verified-resolved
    evidence: "Git-normalized source/receipt hashes and autocrlf controls remain reconciled."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No LF regression."
  - finding_id: "VIA000-R3-RR1-AUTH-REF-TOCTOU-001"
    outcome: verified-resolved
    evidence: "Captured-OID parse/verify/peel/final-equality races and direct-object happy path remain green."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No ref regression."
  - finding_id: "VIA000-R3-RR2-GIT-REPLACE-OBJECT-001"
    outcome: verified-resolved
    evidence: "Replacement-disabled Git operations and hostile replacement/config/environment cases remain green."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No replacement regression."
  - finding_id: "VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001"
    outcome: verified-resolved
    evidence: "No-profile fixed shells, environment-only context transport, grammar rejection, and argument arrays remain green."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No command-input regression."
  - finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001"
    outcome: unresolved
    evidence: "Bare names/shims are fixed, but execution rechecks only Git before invoking prior-hashed PowerShell; mutation invokes prior-hashed base Python and checks environment Python after use."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Same-path byte replacement was explicitly required by RR4; no duplicate ID is assigned."
prior_requested_test_results:
  - requested_test_id: "TST-VIA000-R3-P1-SNAPSHOT-AUTHORITY-001"
    outcome: verified-satisfied
    evidence: "Focused authority/lifecycle cases pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-P1-VALIDATOR-SOURCE-CLOSURE-001"
    outcome: verified-satisfied
    evidence: "Focused closure/substitution/cleanup cases pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-P1-CROSS-PLATFORM-BLOB-001"
    outcome: verified-satisfied
    evidence: "Normalized blob/receipt compatibility cases pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-RR1-AUTH-OBJECT-IMMUTABILITY-001"
    outcome: verified-satisfied
    evidence: "Captured-object race stages and immutable happy path pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-RR2-GIT-OBJECT-REPLACEMENT-001"
    outcome: verified-satisfied
    evidence: "Default/custom replacement and sanitized Git controls pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-RR3-WORKFLOW-COMMAND-BOUNDARY-001"
    outcome: verified-satisfied
    evidence: "Exact hostile-input/no-profile shell controls pass."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: ""
  - requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001"
    outcome: unresolved
    evidence: "Path/root/hash/shim cases pass, but no use-time same-path replacement control protects PowerShell, base/environment Python, or repeated runner tool uses."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Same-path replacement was part of the exact RR4 request."
predictions:
  experiment_id: ""
  predicted_outcome: ""
  predicted_failure_mode: ""
  confidence_statement: "No scientific experiment was run or inspected; this artifact assesses pre-holdout protocol/governance machinery only."
recommendation:
  approve: false
  blocking_findings: 2
  rationale: |-
    CHANGES REQUESTED. Close RR4's use-time identity race and RR5's execution-context
    closure, then obtain another artifact-only rereview. No merge, signer amendment,
    activation, refreeze, tag, preregistration, holdout, assembly, commitment, reveal,
    or scientific claim is authorized.
```

## Verification ledger

- Exact handoff/tree `2c4882f61662cb5f6f7362ee23c182c11176cd57` / `3159b7ecd5a540a21eaa270e53d5f9b8c73f59a8`; content/tree `e9addbc4e03d07d692bce7217548da3d63518d8d` / `b0b7460f18eba5f433d128720649a270ffcf6efb`; prior `4a966a6e8be15f04a2a2d1c4dffa251c0d0f2333`.
- Focused R3 identity suite: exit 0; 40 passed in 370.69 seconds.
- Compatibility/guidance suite: started against all 22 cases, but interrupted after
  three early failures and no completed diagnostic summary when the decisive protocol
  blockers were already confirmed; the prior handoff reported 22/22 green, so no
  independent positive compatibility claim is made here.
- Full 419 suite skipped after two critical blockers were confirmed.
- Exact-source use probe: workflow lines 377-380 rehash Git only; line 430 invokes PowerShell; line 544 invokes base Python; lines 547-549 check environment Python after use.
- In-scope closure probe: candidate tests precede commands 014/015, inherit RUNNER_TEMP, and can write the RUNNER_TEMP TeX tree; later pdfTeX calls do not rehash the leaf or bind its format/config/input closure. TeX/loader state is also inherited and `-no-shell-escape` is absent.
- Original through RR4 reviews remain in ancestry and byte-unchanged; R2 scoped diff from `6e0e0c8ebaecef6d129c68666f113fbd47af4ce7` is empty.
- Placeholder signer remains; no local R3 tag was found. R3 remains drafted, holdout-false, unrevealed, and pending.
