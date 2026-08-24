# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-7-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-7-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-7"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-7.md"
review_commit: "5c588f66e43bc9a100b91ebb9a6cf48b197369ce"
candidate_commit_reviewed: "55df162d7a44fc69674550d7804135c9c527ad09"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: |-
    R3 remains drafted, holdout_started=false, unrevealed, and pending. No custody
    directory, sealed manifest, hidden label, secret seed, reveal material, external
    invalid assembled package, or handoff memo was accessed. No signer key, tag,
    activation, campaign execution, raw result, commitment, or adjudication was
    created. The new workflow is synthetic and non-scientific.

summary: |-
  The RR7 proof-path finding is accepted and addressed in implementation commit
  6e1b3beccba3072a861f3452665e85d20f2f4c0f and final content commit
  75dcebfdad9012ab4b884191f37e2a5d36ded719 (tree
  a79e4bb7b49b414e0938391c6ed7d26c95286897). A separate safe workflow now runs the
  exact production containment helper against a frozen hostile synthetic fixture in
  six fresh GitHub-hosted cells: candidate, PDF, and mutation labels on Ubuntu 24.04
  and Windows 2025. It is triggered only by pushes to the narrow R3 campaign/review
  branch prefixes, has contents-read permission only, and has no secrets,
  environments, id-token, authorization, signer, lifecycle, custody, scientific
  candidate/baseline, assembly, commitment, reveal, or ref-write path.

  Each cell verifies Git-normalized helper/runner/fixture/schema/aggregator/workflow
  source and receipt bytes before creating its synthetic workspace. It then calls the
  exact production Invoke-Via000ContainedCommand primitive, establishes a live
  detached child-of-child, attempts direct and delayed protected evidence/tool
  writes, replace-and-restore, hard-link substitution, protected evidence reads, and
  control-plane environment discovery, and requires complete teardown, zero active
  descendants, absent delayed writes, and unchanged closure hashes. Only small
  non-scientific JSON/transcript artifacts are uploaded on success.

  The frozen aggregate verifier requires exactly the 2x3 cell set, one repository,
  workflow/ref/source/run/attempt identity, identical frozen bundle hashes, exact
  platform primitives, true security predicates, a byte-matching containment result,
  and matching stdout/stderr hashes. A missing, duplicated, substituted, cross-source,
  or false cell cannot produce an aggregate. Hosted run/job/artifact identifiers are
  necessarily recorded after the handoff branch push and remain evidence for fresh
  independent rereview, never approval or campaign evidence.

  The first branch-push attempt, run 32696509331 at handoff 6cb7d25a0ccec067aa1813a3cce0905f737e7974,
  was rejected by GitHub before job creation because the workflow used the matrix
  context in a step shell expression, where that context is unavailable. The
  correction replaces it with two conditionally selected steps whose shell values
  are literal reviewed absolute paths. Official actionlint 1.7.12 accepts the
  corrected workflow with zero parse or expression errors. This failed zero-job run
  is not containment evidence and is retained as transparent negative evidence.
  The second branch-push attempt, run 32697185085 at handoff
  f3ae178c62d19c9efdf81eaa5ce3b1b729b46372, created all six matrix jobs but each
  failed during action preparation before checkout because the declared
  upload-artifact revision did not exist upstream. The pin is corrected to the
  verified immutable v4.6.2 commit ea165f8d65b6e75b540449e92b4886f43607fa02;
  this second run likewise contains no containment result or campaign evidence.
  The third branch-push attempt, run 32697512012 at handoff
  357bc823265ebaf6837553175873a3f98b99f3f6, reached the exact six checked-out
  cells and exposed two bootstrap defects before hostile execution: PowerShell
  parsed an unparenthesized two-path preflight as duplicate LiteralPath parameters,
  and the GitHub Windows shell resolver could not accept a quoted executable path
  containing spaces. The preflight is parenthesized and the Windows step now uses
  the absolute built-in Windows PowerShell host solely to launch the separately
  verified absolute PowerShell 7 executable. This run is not containment evidence.
  The fourth branch-push attempt, run 32697922964 at handoff
  dc67f5fe153327e50f5277914f07a6b868efd103, reached the production helper in all
  Ubuntu cells and exposed an over-narrow terminal-state predicate: a stopped empty
  transient unit may report ActiveState failed, while the prior check admitted only
  inactive. The corrected predicate admits only inactive/failed terminal states and
  still requires dead/failed SubState plus an empty or removed cgroup. Its Windows
  cells exposed that Windows PowerShell -File rejects GitHub's extensionless runner
  temp file; the absolute host now dot-sources that fixed control-plane file with
  -Command before launching verified PowerShell 7. No cell uploaded proof evidence.

finding_responses:
  - finding_id: "VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Reviewers can now exercise the required production Windows Job Object and
      Ubuntu systemd DynamicUser/control-group primitives on an unmerged reviewed
      source without creating campaign authorization or weakening the production
      workflow's signer/lifecycle guard. Runtime unavailability or any failed cell
      fails the synthetic workflow and prevents its aggregate artifact.
    changed_files:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - ".gitattributes"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-HOSTILE.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF.schema.json"
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
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-protocol.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-hostile.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-aggregator.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof.schema.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
    fix_commits:
      - "6e1b3beccba3072a861f3452665e85d20f2f4c0f"
      - "18f1ff0e2f5767463d8766ced45f7df10b60006c"
      - "0f8fbbe08f0eb8bec66eab07f7028d93b923f3a2"
      - "27e5251a29df24c8f1986ed396a1b44078b2bbba"
      - "3f2187b1ac0496d452b0e29f2e6cf655b2db5431"
      - "75dcebfdad9012ab4b884191f37e2a5d36ded719"
    verification:
      - command: "exact hosted-proof source/safety/aggregate negative control"
        result: "1 passed in 3.86 seconds after final hash binding; complete six-cell aggregate accepted, missing cell rejected, and helper/hash substitution rejected before workspace/output."
      - command: "full R3 identity test file"
        result: "52 passed in 535.74 seconds, preserving all prior authorization, replacement-object, command-boundary, tool-identity, assembly, stage-isolation, and containment controls."
      - command: "Ruff plus PowerShell, JSON, and YAML parsing"
        result: "Changed Python sources passed Ruff; all three proof/containment PowerShell files and the proof JSON/YAML parsed successfully."
      - command: "official actionlint 1.7.12"
        result: "Corrected hosted workflow passed with zero parse or expression errors after replacing the unsupported matrix-derived shell field with literal OS-specific shell steps."
      - command: "review guidance and TeX validation"
        result: "9 review-guidance tests passed in 10.83 seconds with the 431-test ledger reconciled; TeX source validation passed."
      - command: "campaign validator"
        result: "The drafted campaign is validator-clean at the bound content commit; no lifecycle state advanced."
    residual_risk: "Fresh hosted execution must pass all six cells and be independently inspected. GitHub hosted image/service/ACL policy and the pinned action/control-plane principals remain external runtime premises. The comment-only signer and fresh custody verification continue to block activation."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR7-HOSTED-CONTAINMENT-PROOF-PATH-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-HOSTILE.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
      - "tests/unit/test_via000_r3_identity.py::test_r3_safe_hosted_containment_proof_path_is_bound_and_exact_2x3"
    verification:
      - command: "source-level exact production-path and safety regression"
        result: "The workflow has the exact six hosted cells, immutable action revisions, read-only permissions, safe branch trigger, exact production helper invocation, frozen receipts, and no campaign execution surface."
      - command: "real hosted branch-push replay"
        result: "Triggered only after the immutable handoff is pushed; exact run, six job, and artifact identifiers are recorded outside this pre-push artifact for independent replay."
    rationale: "The requested real platform execution is now reachable without campaign authorization. Local tests prove the workflow contract and fail-closed aggregator; the branch push supplies the real Windows/Ubuntu primitive evidence."
    disagreement_ref: ""

new_or_changed_risks:
  - "The proof workflow intentionally runs on qualifying feature/review branch pushes; its path filter, read-only permissions, synthetic-only closure, and no-secrets design bound that exposure."
  - "Hosted Windows nested Job Object behavior and Ubuntu passwordless systemd DynamicUser policy must still pass in the recorded run."
  - "Artifact retention is seven days, so the independent reviewer should inspect or download the small proof set promptly."
  - "The full 431-test suite and the slow 22-case compatibility suite were deferred because the focused 52-case R3 aggregate and guidance/schema checks were green; fresh rereview should run them if feasible."

external_actions:
  - action: "Inspect/replay the recorded hosted run, six jobs, six cell artifacts, and aggregate; verify exact source SHA and both OS containment primitives before resolving RR4, RR6, or RR7."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: "recorded outside this artifact after branch push"
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
  scope: "Verify the safe feature/review-branch trigger and permissions, exact frozen source/receipt closure, six hosted cells, live child-of-child teardown, protected evidence/tool denial, replace/restore and link attacks, control-plane environment scrub, zero Windows Job Object processes and empty Ubuntu cgroup, exact fragment/aggregate identity, and preservation of all prior R3 controls."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, scientific execution, commitment, reveal, or viability claims from this response alone."
```
