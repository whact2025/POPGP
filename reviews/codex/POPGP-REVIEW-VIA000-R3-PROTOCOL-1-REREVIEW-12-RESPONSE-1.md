# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-12-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-12-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "OpenAI Codex"
builder_model_version: "GPT-5"
builder_operator: "NVIDIA.COM\\rfuoco"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-12"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-12.md"
review_commit: "80e6005542033843628639cce1a3a30fc05e723d"
candidate_commit_reviewed: "d2ce792a71002310f9ef760962bcb1141ec99437"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: |-
    R3 remains drafted, holdout_started=false, unrevealed, and pending. No hidden,
    sealed, signer, lifecycle, custody, or scientific result material was accessed or
    created. No key, tag, refreeze, activation, campaign execution, commitment,
    adjudication, or reveal was performed. The hosted proof workflow remains synthetic,
    read-only, and isolated from the scientific candidate and comparison baseline.

summary: |-
  RR12 is implemented at content commit b114b04c9b445fe7da9b7e6ef6e635736ba11eda
  (tree 397601d8b68d9cb48dc7d4c1c013f1f022b89219). All twelve Windows
  PowerShell run steps in the proof workflow and all eight trusted run-step types in
  the production workflow now use GitHub's built-in shell: pwsh. No custom absolute,
  dot-source, or spaced -File PowerShell shell template remains.

  Every trusted production pwsh body begins with an OS-conditional identity boundary.
  Windows requires the current process and PSHOME executable to be the canonical
  PowerShell 7 installation, a PS7 version, no applicable profile file, and the exact
  reviewed Windows PATH. Ubuntu independently requires the canonical hosted PowerShell
  7 installation, the same version/profile policy, and its exact non-workspace PATH.
  Proof jobs apply the exact Windows checks and PATH. Material authorization, tool,
  containment, execution, mutation, subject, bundle, cache, and cleanup effects receive
  same-step postconditions and a recheck at the next consumer boundary.

  All earlier Windows proof evidence whose security-critical checks used a custom shell
  is invalid and superseded. The five-run experiment remains non-authoritative diagnostic
  evidence only; none of its caches or artifacts is accepted. A fresh exact-source
  six-cell proof and an independent rereview are still required.

finding_responses:
  - finding_id: "VIA000-R3-RR12-WINDOWS-CUSTOM-SHELL-NOOP-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Security-critical hosted PowerShell code can no longer pass through the proven
      no-op custom shell boundary. Built-in pwsh is constrained and self-identifies at
      entry before material action. Same-step observability plus downstream persistence
      checks make an absent side effect a failure. The same literal built-in shell is
      used on Ubuntu with a separate canonical process, PSHOME, version, profile, and
      PATH assertion so cross-platform resolution does not fall back to a dispatcher-
      controlled executable.
    changed_files:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - ".github/workflows/via000-r3-protocol.yml"
      - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/PROTOCOL_MANIFEST.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/README.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/amendments/VIA-000-R3-RECOVERY-DESIGN-1.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/workflow-protocol.yml"
      - "tests/unit/test_via000_r3_identity.py"
    fix_commits:
      - "b114b04c9b445fe7da9b7e6ef6e635736ba11eda"
    verification:
      - command: "complete R3 shared-identity test file"
        result: "56 passed in 609.68 seconds, including the stable RR12 source and live persistence regression."
      - command: "focused proof safety, envelope/cache compatibility, hosted-shell, and Windows containment controls"
        result: "5 passed in 20.78 seconds; the corrected stale assertion plus RR12 test passed 2/2 in 5.27 seconds."
      - command: "official actionlint 1.7.12, Ruff, YAML/JSON parsers, and all embedded PowerShell run-body parsers"
        result: "Both workflows and all 37 embedded run bodies passed; changed Python passed Ruff."
      - command: "frozen source, receipt, packet-rule, manifest, and campaign checks"
        result: "Source/receipt pairs and declared SHA-256 bindings matched; campaign validation passed on the content-bound handoff bytes."
    residual_risk: |-
      Built-in pwsh execution and the exact six-cell cache/aggregate/redownload path
      must still be demonstrated on fresh hosted Windows and Ubuntu runners at this
      source. The declared boundary trusts GitHub's hosted control plane and pinned
      actions. Production execution remains blocked, and fresh independent rereview is
      required after proof-only evidence is available.
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR12-WINDOWS-BUILTIN-PWSH-EXECUTION-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - ".github/workflows/via000-r3-protocol.yml"
      - "tests/unit/test_via000_r3_identity.py::test_r3_rr12_windows_builtin_pwsh_execution_is_observable_and_persistent"
    verification:
      - command: "workflow-source enumeration and forbidden-shell regression"
        result: |-
          The test enumerates exactly twelve proof-workflow Windows uses and eight
          production trusted step types, requires literal built-in pwsh, rejects custom
          dot-source and spaced -File forms, and requires both OS identity assertions
          before the first material action.
      - command: "live local Windows PowerShell 7 execution and persistence fixture"
        result: |-
          The extracted policy executed under the canonical local PowerShell 7 process,
          created and same-step validated a sentinel, preserved it for a separate
          consumer, and rejected a hostile PATH before creating a side effect.
      - command: "fresh proof-only hosted replay"
        result: |-
          Pending branch push. Acceptance requires all six explicit cells, six distinct
          digest-bound cache restores, the canonical 2x3 aggregate, one retained
          seven-file artifact, and green dependent redownload verification.
    rationale: "The regression proves source coverage and local execution observability; the frozen proof workflow supplies the required hosted two-platform replay without touching scientific or lifecycle state."
    disagreement_ref: ""

new_or_changed_risks:
  - "The workflows depend on the reviewed GitHub-hosted canonical PowerShell 7 installation paths; a runner-image change fails closed and requires a new reviewed binding."
  - "The no-profile policy rejects a hosted image that creates an applicable profile file, even if that file is benign."
  - "Earlier Windows caches, artifacts, and green custom-shell step metadata are invalid and cannot be reused."

external_actions:
  - action: "Inspect or independently replay the fresh proof-only run; require six successful explicit cells, six exact cache restores, one retained seven-file artifact, and green redownload verification."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: "fresh run recorded outside this artifact after branch push"
  - action: "Perform a fresh independent rereview of the built-in pwsh execution boundary and every retained prior control before any production action."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Verify built-in pwsh source coverage, canonical Windows and Ubuntu process/PSHOME/version/profile/PATH assertions, same-step and cross-boundary effects, fresh exact six-cell cache transport, retained aggregate redownload, frozen hashes, and all prior R3 controls."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, scientific execution, commitment, reveal, or viability claims from this response alone."
```
