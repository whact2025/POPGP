# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-4-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-4-RESPONSE-1"
response_round: 1
response_date: "2026-08-23"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-4"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-4.md"
review_commit: "4a966a6e8be15f04a2a2d1c4dffa251c0d0f2333"
candidate_commit_reviewed: "9fb95a12047d9d038cb1f9b62caa3a2abcc97e2d"

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
  The complete execution-tool finding is accepted and fixed in final content commit
  e9addbc4e03d07d692bce7217548da3d63518d8d (tree
  b0b7460f18eba5f433d128720649a270ffcf6efb), following implementation commits
  5e2c05b7c6200f8e4b7e6ec9e00a0ddbab5413a6 and
  5ee3f18d458488ba6658ffd249785653c69ac19b. The GitHub control plane and exact
  pinned action revisions are now explicit bootstrap principals. The matrix uses
  fixed ubuntu-24.04 and windows-2025 labels and an absolute per-platform PowerShell
  shell. The setup-python output is data-only and must name the single exact hosted
  tool-cache location for CPython 3.11.15; command/script files, wrong roots, multiple
  values, and link/reparse ancestry reject before authorization reads.

  The workflow derives uv 0.11.11 only through trusted Python sysconfig and canonical
  pdfTeX only beneath the pinned TeX Live 2026 root. Git, ssh-keygen, base and copied
  environment Python, uv, pdfTeX, and PowerShell are regular non-reparse absolute
  files under reviewed roots with exact version/banner checks and retained SHA-256
  identities. Pre-authorization tool hashes are rechecked after setup. PATH, PATHEXT,
  Git, Python, shell-startup, and caller child-process injection state are scrubbed.
  Runner and mutation commands receive only explicit executable paths and argument
  arrays. Each signed platform summary binds a typed tool-identity manifest.

  The production runner and mutation runner reject substituted tools before an
  experiment workspace, mutation output, or commitment can exist. Every failure
  deletes the complete platform workspace; failed jobs cannot upload it. Five new
  executable/static controls plus the retained first-shell PATH control are green on
  Windows. The first six review findings and requested controls remain preserved in
  ancestry and were not reinterpreted.

finding_responses:
  - finding_id: "VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      No experiment or mutation child resolves an executable through caller PATH or
      PATHEXT. Trusted control-plane/action outputs select one fixed platform family;
      all later tools are derived under exact roots, checked for regular/non-reparse
      identity, version/banner and SHA-256, passed explicitly, and retained in signed
      evidence. Setup-output, command-file, path, version, banner, hash, and environment
      substitutions fail before the affected untrusted executable or output boundary.
    changed_files:
      - ".github/workflows/via000-r3-protocol.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-MUTATION-RUNNER.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-RAW-RESULTS.schema.json"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_via000_r3_identity.py"
      - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
      - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/PROTOCOL_MANIFEST.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/runner-protocol.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/mutation-runner.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/raw-results.schema.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
    fix_commits:
      - "5e2c05b7c6200f8e4b7e6ec9e00a0ddbab5413a6"
      - "5ee3f18d458488ba6658ffd249785653c69ac19b"
      - "e9addbc4e03d07d692bce7217548da3d63518d8d"
    verification:
      - command: "python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r3_identity.py"
        result: "40 passed in 364.09 seconds after closing the synthetic tool-evidence path set; all prior authorization, ref-race, replacement-object, validator-closure, cross-run, LF, cleanup, and exact snapshot happy controls remain green."
      - command: "python -m pytest -q -p no:cacheprovider tests/unit/test_via000_r3_identity.py -k 'complete_execution_toolchain or setup_python_output_substitution or tool_identity_manifest_rejects or unmodified_runner_rejects_explicit_shim or unmodified_mutation_runner_rejects_command_shim or workflow_command_boundary_ignores_path_tool_shims'"
        result: "6 passed in 102.89 seconds; no shim marker, authorization output, experiment workspace, mutation output, or commitment was created."
      - command: "python -m pytest -q -p no:cacheprovider tests/unit/test_review_guidance.py"
        result: "9 passed in 12.61 seconds after registering the new executable tool-identity gate."
      - command: "python scripts/check_tex.py"
        result: "TeX source balanced and valid."
      - command: "python -m scripts.check_viability_campaign reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
        result: "Viability campaign contract is valid at the final content binding."
    residual_risk: "The GitHub control plane and exact pinned action revisions remain declared trust principals. Hosted image packages can change while a fixed runner label remains; the retained byte hashes make each execution auditable but are not preregistered package digests. The Ubuntu production path is source-checked here and requires fresh execution on the supported hosted image during independent rereview. The deliberately empty signer file still prevents activation."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR4-EXECUTION-TOOL-IDENTITY-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r3_identity.py"
    verification:
      - command: "exact workflow authorization source with an absolute python.cmd setup-output substitution and malicious PATH/PATHEXT layouts"
        result: "Rejected before the substituted file ran or authorization JSON existed; the marker remained absent."
      - command: "unmodified VIA-000-RUNNER.ps1 with explicit PATH/PATHEXT git/python/uv/pdfTeX/PowerShell shims"
        result: "Rejected at explicit trusted-root/type preflight before marker execution or workspace creation."
      - command: "unmodified VIA-000-MUTATION-RUNNER.py with an explicit command shim plus PATH/PATHEXT shims"
        result: "Rejected before marker execution and before mutation evidence/output changed."
      - command: "Windows and Ubuntu tool-identity evidence controls"
        result: "Exact hosted-label/root/version/hash fixtures pass; wrong setup root, Python/uv version, TeX banner, environment-Python hash, and architecture fixtures reject."
      - command: "static parse of the exact workflow, runner, and mutation-runner command boundaries"
        result: "All workflow run steps use the fixed matrix shell; no production child uses a bare git, python, uv, pdfTeX, or PowerShell name; upload is success-only and failure cleanup is explicit."
    rationale: "The requested test is mapped once and exercised across the exact first workflow source, unmodified production runner, unmodified mutation runner, evidence validator, and both declared platform contracts while retaining all prior hostile-input, ref-race, replacement-object, validator-closure, cross-run, LF, and cleanup controls."
    disagreement_ref: ""

new_or_changed_risks:
  - "The exact pinned actions and GitHub hosted control plane are explicit bootstrap trust principals."
  - "Fixed hosted-image labels do not freeze package bytes; each run retains actual executable hashes for independent comparison."
  - "A hosted tool relocation, link conversion, version/banner change, or setup-action output change intentionally invalidates execution and requires a reviewed amendment."
  - "The production authorization signer and fresh fourteen-of-fourteen custody verification remain separate unresolved governance gates."

external_actions:
  - action: "Independently rereview the full workflow/runner/mutation executable boundary, exact tool roots and hashes, cleanup, frozen receipts, and this response on both supported hosted images."
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
  scope: "Verify the declared control-plane/action trust boundary, fixed hosted labels, exact setup-python data output, explicit regular non-reparse tool paths, sysconfig-derived uv, pinned-root pdfTeX, version/banner/hash retention, sanitized child environments, production runner/mutation rejection and cleanup, tool-identity evidence validation, frozen hashes, and this response on Windows and Ubuntu."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, execution, commitment, reveal, or scientific claims from this response alone."
```
