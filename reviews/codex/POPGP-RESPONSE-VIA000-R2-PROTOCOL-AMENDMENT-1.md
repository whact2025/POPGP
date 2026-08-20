# Builder response: POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1

```yaml
artifact_schema_version: 2
response_id: "POPGP-RESPONSE-VIA000-R2-PROTOCOL-AMENDMENT-1"
response_round: 1
response_date: "2026-08-20"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r2-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1.md"
review_commit: "755a6a4d20bac2f4ca35432a98027918a4972d94"
candidate_commit_reviewed: "9d82ddabceee934be8036386af7867c93badc6f9"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: |-
    R2 remains preregistered, holdout_started=false, and unrevealed. No custody file,
    hidden holdout, secret seed, final label, output commitment, raw runner result, or
    untracked builder handoff memo was accessed. This response covers public
    pre-holdout protocol and validator remediation only.

summary: |-
  All five review findings are accepted. Fix commit
  3876cc0738b5474f49e10468b1e1e675fa6fcba1 replaces assertion-driven raw results
  with typed executable evidence. The validator now cross-binds the exact platform,
  candidate, command-contract, artifact, and mutation sets; verifies command result
  records and retained streams; derives the pytest count from output; parses real PDF
  framing and two eleven-page build logs; reconciles the complete Git source manifest
  and typed environment manifest; and independently compares every retained
  validation JSON and raster to the frozen candidate while rerunning the semantic
  predicates. Stored count, gate, capability, and outcome Booleans are recomputed.

  Raw results now schema-require blocked=false. Missing infrastructure, early command
  failure, or partial evidence is an invalid attempt that cannot be committed; it is
  not an author-selected terminal scientific outcome.

  The new frozen VIA-000-ASSEMBLER.py accepts exactly the two platform fragments and
  two eighteen-family mutation packages, verifies hashes and identities, creates
  collision-free platform-qualified paths without changing source bytes, rejects
  incomplete/nonzero/accepted inputs, validates the assembled raw document, and
  atomically emits raw-results.json plus a distinct output-commitment.json. On error
  it removes only its private temporary output and creates no commitment. A pinned
  Windows/Ubuntu workflow runs the real frozen platform producer and validates each
  emitted fragment directly against the platform schema.

  The exact implementation-fix Git blob of scripts/check_viability_campaign.py is
  SHA-256 eedc4b7add380c9ea1d64174dab7ba4e289cc2dd8bf92065e768e50acd61ff70;
  the protocol manifest now records that value. The VIA-000 rule was recomputed as
  cf1a27bcffd1702fc7478377489a0dcfb2772028f282655590ae69adedcefa10 and
  the campaign binds the updated manifest. All protocol/receipt copies are identical.

  Focused typed-evidence and assembler tests passed 6/6 in 229.98 seconds. The exact
  repository suite passed 372/372 in 1262.21 seconds after correcting its documented
  count. Ruff, the 652-line TeX checker, PowerShell parsing, all six generators, the
  semantic/visual artifact checker, JSON/schema checks, and diff checks pass. These
  are pre-holdout protocol results, not scientific viability evidence. Refreeze,
  holdout start, reveal, and merge remain forbidden pending a fresh zero-blocker
  independent re-review and a new safe falsifier result.

finding_responses:
  - finding_id: "VIA000-R2-PA1-EVIDENCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Evidence roles, executable command contracts, real scientific artifacts, source
      and environment manifests, PDF framing/build logs, and typed mutation receipts
      are now validated rather than inferred from self-authored summary fields.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1"
      - "tests/unit/test_viability_raw_evidence_contract.py"
    fix_commits:
      - "3876cc0738b5474f49e10468b1e1e675fa6fcba1"
    verification:
      - command: "uv run pytest -q tests/unit/test_viability_raw_evidence_contract.py tests/unit/test_via000_r2_assembler.py"
        result: "6 passed in 229.98 seconds; plaintext/empty-artifact, stale command, visual, identity, blockage, mutation, partial assembly, and commitment controls reject."
      - command: "uv run pytest -q"
        result: "372 passed in 1262.21 seconds."
    residual_risk: |-
      Cryptographic receipts still rely on the declared trusted base interpreter,
      Git object database, operating system, uv, TeX engine, and hosted runner. Those
      infrastructure assumptions are explicit and do not establish physical truth.
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA1-IDENTITY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      Platform candidate commit/tree mismatches are unconditional errors. The frozen
      platform list is exactly reconciled among packet parameters, raw contract, and
      schema required/properties; the command count and ID-to-contract map must agree.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"
      - "tests/unit/test_viability_raw_evidence_contract.py"
    fix_commits:
      - "3876cc0738b5474f49e10468b1e1e675fa6fcba1"
    verification:
      - command: "test_raw_evidence_contract_rejects_identity_contract_and_blockage"
        result: "Wrong platform identity and contradictory contract/parameter platform sets reject independently of outcome Booleans."
    residual_risk: "A future platform or command requires a deliberate preregistered contract revision and new review."
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA1-BLOCKED-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      R2 does not need an author-controlled blockage state after its prerequisites
      were demonstrated. The raw schema fixes blocked=false; incomplete or unavailable
      execution is invalid and the assembler emits no result or commitment.
    changed_files:
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"
      - "tests/unit/test_viability_raw_evidence_contract.py"
    fix_commits:
      - "3876cc0738b5474f49e10468b1e1e675fa6fcba1"
    verification:
      - command: "set raw blocked=true in the typed positive fixture"
        result: "Draft 2020-12 validation rejects with 'False was expected'."
    residual_risk: |-
      A genuine later external outage yields no terminal R2 outcome. A new attempt may
      be authorized only before reveal and must create a new immutable execution
      record; it cannot overwrite committed evidence.
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA1-ASSEMBLY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The assembler is now a frozen reviewed artifact. It prefixes paths, verifies
      every source byte/hash/identity/set, requires all mutations, validates the final
      schema, and atomically creates the result and commitment. Failed or partial
      fragments leave no output directory.
    changed_files:
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-ASSEMBLER.py"
      - "tests/unit/test_via000_r2_assembler.py"
      - ".github/workflows/via000-r2-protocol.yml"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/launch/VIA-000_EXECUTION_BRIEF.md"
    fix_commits:
      - "3876cc0738b5474f49e10468b1e1e675fa6fcba1"
    verification:
      - command: "test_assembler_roundtrip_produces_valid_raw_results_and_commitment"
        result: "Two typed platform fragments plus 36 mutation receipts assemble, schema-validate, pass the authoritative validator, and bind the raw SHA-256."
      - command: "test_assembler_rejects_partial_or_failed_fragments_without_commitment"
        result: "A nonzero command or missing platform mutation file exits nonzero and creates no output directory/commitment."
    residual_risk: "The real two-platform workflow remains an external pre-refreeze action and must be independently audited before approval."
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA1-FREEZE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The manifest validator digest is now derived with the public Git-blob command at
      the exact implementation-fix commit. The packet rule and campaign manifest hash
      were recomputed from the corrected bytes. Final activation remains intentionally
      deferred until the re-review approves this handoff.
    changed_files:
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/PROTOCOL_MANIFEST.json"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/CAMPAIGN.yaml"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"
    fix_commits: []
    verification:
      - command: "uv run python scripts/check_viability_campaign.py --git-blob-sha256 3876cc0738b5474f49e10468b1e1e675fa6fcba1 scripts/check_viability_campaign.py"
        result: "eedc4b7add380c9ea1d64174dab7ba4e289cc2dd8bf92065e768e50acd61ff70, exactly matching the corrected manifest."
      - command: "uv run python scripts/check_viability_campaign.py --packet-rule-sha256 reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"
        result: "cf1a27bcffd1702fc7478377489a0dcfb2772028f282655590ae69adedcefa10, exactly matching packet and manifest."
    residual_risk: "The response-containing handoff commit still requires independent all-entry Git-blob audit before any snapshot/activation commit."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R2-PA1-DUMMY-EVIDENCE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_raw_evidence_contract.py"
      - ".github/workflows/via000-r2-protocol.yml"
    verification:
      - command: "focused typed raw-evidence suite"
        result: "Plaintext/empty-artifact and asserted-summary forms reject; real committed scientific/PDF bytes form the positive fixture."
    rationale: "The positive path now exercises typed evidence and the negative path preserves the review's lethal synthetic class."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA1-IDENTITY-CONTRACT-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_raw_evidence_contract.py"
    verification:
      - command: "identity/contract adversarial fixture"
        result: "Wrong commit and independently narrowed platform contract reject."
    rationale: "Identity and list consistency are validity errors, not failed-capability inputs."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA1-BLOCKAGE-001"
    disposition: partially-accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_raw_evidence_contract.py"
      - "tests/unit/test_via000_r2_assembler.py"
    verification:
      - command: "blocked flip plus partial/nonzero assembly controls"
        result: "blocked=true is schema-invalid; incomplete attempts produce no output or commitment."
    rationale: |-
      The review allowed defensible removal of unsupported terminal blockage. R2 uses
      that narrower design, so a positive typed blockage fixture is intentionally not
      part of this packet's raw-results language.
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA1-ASSEMBLY-ROUNDTRIP-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r2_assembler.py"
      - ".github/workflows/via000-r2-protocol.yml"
    verification:
      - command: "frozen assembler positive and fail-closed tests"
        result: "Typed two-platform round trip passes; partial and failed fragments create no commitment."
    rationale: "The transformation is frozen, deterministic, byte-preserving, collision-free, schema-checked, and atomic."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA1-MANIFEST-BINDING-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "scripts/check_viability_campaign.py"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/PROTOCOL_MANIFEST.json"
    verification:
      - command: "exact implementation-fix Git-blob and packet-rule audit"
        result: "Corrected validator and VIA-000 digests match the exact committed bytes and canonical rule."
    rationale: "Worktree hashing is no longer used for the corrected validator binding."
    disagreement_ref: ""

new_or_changed_risks:
  - "The validator now deliberately performs a complete 560-path Git source audit for each retained platform; validation is slower but bounded."
  - "Mutation evidence is typed and hash-bound, but scientific confidence still depends on a clean falsifier independently executing every frozen family."
  - "R2 removes terminal blockage from raw results; an infrastructure outage produces an invalid/uncommitted attempt, not a scientific result."
  - "No protocol evidence establishes the physical mechanism or Tier R until execution, reveal, independent audit, and adjudication complete."

external_actions:
  - action: "Run and retain the exact real Windows/Ubuntu frozen platform fragments at the response handoff."
    owner: "builder"
    status: pending
    evidence_ref: ".github/workflows/via000-r2-protocol.yml"
  - action: "Obtain fresh independent re-review of all five findings/tests, exact counterexamples, real platform artifacts, manifest bindings, and broader attacks."
    owner: "independent-reviewer"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "all five findings, all five requested tests, typed evidence semantics, exact identity/list binding, blockage removal, partial/failure assembly, manifest Git-blob binding, real Windows/Ubuntu fragments, regressions, and new attacks"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: |-
    Use a fresh isolated worktree at the response-containing handoff. Treat this
    response, tests, and hosted runs as hypotheses. Reproduce the prior synthetic,
    identity, blockage, assembly, and stale-manifest attacks; broaden typed command,
    artifact, mutation, PDF, manifest, path, and partial-output boundaries. Modify
    only a new independent re-review artifact. Do not authorize refreeze or holdout
    unless every prior finding/test is verified and blocker count is zero.
```
