# Builder response: POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10-RESPONSE-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10-RESPONSE-1"
response_round: 1
response_date: "2026-08-24"

builder_seat: builder
builder_model_identity: "OpenAI Codex"
builder_model_version: "GPT-5"
builder_operator: "NVIDIA.COM\\rfuoco"
builder_session_id: "popgp-viability-r3-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R3-PROTOCOL-1-REREVIEW-10.md"
review_commit: "05508ed54a3031ff1970cdaf8a2d917d06151421"
candidate_commit_reviewed: "9e28297b803d0570aa2fc62390267f9fb0f2c530"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: |-
    R3 remains drafted, holdout_started=false, unrevealed, and pending. No hidden,
    sealed, signer, lifecycle, or scientific result material was accessed or created.
    No key, tag, refreeze, activation, campaign execution, commitment, adjudication,
    or reveal was performed. The hosted workflow remains synthetic, read-only, and
    isolated from the scientific candidate and baseline.

summary: |-
  RR10 is implemented at content commit 281f3e31c0087080bf906af60578a5c00da22720
  (tree eefa859aadbcc6cc9558129fe44a4151454744af). Envelope bytes no longer
  cross GitHub job-output or environment boundaries. Each of the six explicit cells
  stages one validated canonical envelope at an exact cell-specific workspace-relative
  path after the contained process tree is quiescent. A separate trusted outer step
  rechecks the ordinary, single-link, non-reparse subject and cell/source/run identity,
  computes its SHA-256, and emits only that lowercase 64-hex digest.

  Each cell preflights and rejects an existing exact cache key, then uses the pinned
  actions/cache v6.1.0 save action. Keys bind the fixed namespace, repository ID,
  workflow SHA, source SHA, run ID, run attempt, platform, stage, and full digest, use
  no restore prefixes, stay below 512 characters, and enable cross-OS archives. The
  Ubuntu aggregate requires six green jobs and six distinct digests, restores six exact
  keys to the identical six relative paths, requires cache-hit=true and primary/matched-
  key equality, and validates every restored byte in memory before creating exactly six
  envelopes plus one aggregate manifest. One retained artifact is uploaded and a
  dependent job redownloads and revalidates it. Cache save success alone is not evidence.

  Hosted run 32718921969 proved all six containment/staging cells and then stopped
  before cache save. GitHub rejected the custom Windows shell expression containing
  the spaced PowerShell 7 path, while all three Ubuntu archive preflights exposed the
  shared delayed exit-code check after two version pipelines. Correction content
  4c8fbeee32e45848e307d4c2b54c058c9072ff22
  (tree 9d30e9629db8c4965ccce7823b22da6b1c898237) uses the supported absolute
  system Windows PowerShell shell for only the outer digest/archive/cache checks,
  replaces PowerShell 7-only JSON/hash APIs in those snippets, and captures and
  validates GNU tar and Zstandard output and exit codes independently. Cache keys,
  pinned actions, topology, containment, transport, and the threat boundary are unchanged.

finding_responses:
  - finding_id: "VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The large, logged, Windows-lossy envelope channel is removed. Only a strict
      digest crosses job-output control. Exact digest-bound cache restore plus canonical
      byte validation is authoritative, so a swallowed save warning becomes a hard
      aggregate miss and a colliding first writer must supply bytes matching the full
      digest and cell identity. No prefix fallback or predictable digest-free key exists.
    changed_files:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000.json"
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_via000_r3_identity.py"
      - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
      - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
      - "docs/scientific_hardening/REPRODUCIBILITY.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/README.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/amendments/VIA-000-R3-RECOVERY-DESIGN-1.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/packets/VIA-000.yaml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-runner.ps1"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-aggregator.py"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/containment-proof-workflow.yml"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/receipts/VIA-000/protocol.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/PROTOCOL_MANIFEST.json"
      - "reviews/viability/POPGP-VIABILITY-R3-2026-08/CAMPAIGN.yaml"
    fix_commits:
      - "281f3e31c0087080bf906af60578a5c00da22720"
      - "c1ff66a7e96d821aa7c7958d9fc93da9cbead422"
      - "4c8fbeee32e45848e307d4c2b54c058c9072ff22"
    verification:
      - command: "complete R3 identity test file"
        result: "55 passed in 589.54 seconds, including the stable digest/cache transport test."
      - command: "focused safe boundary, prior transport, and cache transport controls"
        result: "3 passed in 18.49 seconds; the new case alone passed in 6.99 seconds."
      - command: "Ruff, parsers, JSON/YAML, and official actionlint 1.7.12"
        result: "Changed Python, PowerShell, structured files, and workflow expressions passed."
      - command: "focused hosted-preflight correction regression"
        result: |-
          2 passed in 14.89 seconds. The stable cache transport test executes the exact
          outer digest source locally under Windows PowerShell 5, verifies the direct
          GITHUB_OUTPUT line, rejects every spaced custom Windows shell path in the
          three outer steps per cell, and accepts hosted GNU tar 1.35 and Zstandard
          1.5.6/1.5.7 banner shapes with separately captured zero exit codes.
      - command: "frozen source, receipt, packet-rule, manifest, and campaign checks"
        result: "All source/receipt pairs and declared SHA-256 bindings matched; final campaign validation is recorded on the handoff."
    residual_risk: |-
      A fresh hosted push must still prove six digest outputs, six exact Windows/Ubuntu-
      to-Ubuntu cache restores, one retained artifact, and green redownload verification.
      The declared threat model trusts the GitHub control plane and pinned actions and
      excludes an additional hostile trusted job holding a cache runtime token. Fresh
      independent rereview remains required.
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R3-RR10-JOB-OUTPUT-CACHE-TRANSPORT-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - ".github/workflows/via000-r3-containment-proof.yml"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R3-2026-08/VIA-000-CONTAINMENT-PROOF-AGGREGATOR.py"
      - "tests/unit/test_via000_r3_identity.py::test_r3_rr10_job_output_cache_transport_is_digest_bound"
    verification:
      - command: "digest/cache hostile corpus and exact retained happy path"
        result: |-
          Exact six-digest, six-cache, and seven-file acceptance passed, including the
          SHA-256 of legitimate empty stdout/stderr. Missing, empty, duplicate, nonhex,
          uppercase, truncated, newline/control-injected, swapped, corrupt, oversized,
          hardlinked, extra, and cross-cell cache state rejected without retained output.
          Workflow checks freeze preflight miss, no prefixes, exact hit/primary/matched
          key equality, cross-OS mode, the pinned action commit, and retained revalidation.
      - command: "safe hosted feature-branch replay"
        result: |-
          Run 32718921969 proved six green containment/staging cells, then failed closed
          before cache save on the Windows spaced-shell expression and archive exit/banner
          preflight. A fresh corrected handoff run remains pending; run/job/cache/artifact
          identities will be recorded from GitHub for independent rereview.
    rationale: "The local suite invokes the frozen cache aggregator and the hosted workflow invokes the same bound source/receipt bytes on both supported operating systems."
    disagreement_ref: ""

new_or_changed_risks:
  - "The standard cache save action can report success after a collision or upload warning; aggregate exact restore and byte validation are therefore mandatory evidence."
  - "Cache transport is non-authoritative and may be evicted immediately after the retained artifact is created and redownloaded."
  - "The trusted-principal boundary excludes a new hostile trusted job with a cache runtime token; expanding that boundary requires a separately reviewed cache reservation protocol."

external_actions:
  - action: "Inspect or independently replay the hosted proof; require six successful explicit cells, six exact digest-bound cache restores, green retained verification, and one seven-file artifact with recorded ID/digest."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: "failed-closed diagnostic run 32718921969; corrected run recorded outside this artifact after branch push"
  - action: "Perform a fresh independent rereview of the cache transport boundary and all retained prior controls."
    owner: "independent-reviewer-seat"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "Verify digest-only outputs, exact cache miss/save/restore semantics, key identity and length, no prefixes/content logs, cross-OS six-cell completeness, canonical byte validation, one retained seven-file artifact, redownload verification, frozen hashes, and all prior R3 controls."
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Do not authorize merge, signer selection, tags, activation, preregistration, custody transition, scientific execution, commitment, reveal, or viability claims from this response alone."
```
