# Builder response: POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2-RESPONSE-3

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2-RESPONSE-3"
response_round: 3
response_date: "2026-08-20"

builder_seat: builder
builder_model_identity: "unknown"
builder_model_version: "unknown"
builder_operator: "fuocor"
builder_session_id: "popgp-viability-r2-2026-08-via000-builder-session"
builder_orchestrator_id: "codex-desktop"
builder_organization: "popgp-internal"

review_id: "POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2.md"
review_commit: "4e262c2457be80f6b87544671832b3eed6a23db5"
candidate_commit_reviewed: "71fc1fd6381a79a799ed1d4cc61dda640a22f382"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: |-
    R2 remains preregistered, holdout_started=false, and unrevealed. No custody
    file, hidden holdout, secret seed, final label, private evaluator, raw result,
    output commitment, or untracked handoff memo was accessed. The changes affect
    only the public pre-holdout protocol boundary.

summary: |-
  This response retains every disposition and binding from response rounds 1 and 2.
  Both reported blockers remain accepted. Producer evidence is GitHub-OIDC/Sigstore
  attested, mutation receipts are derived from the registered verbose pytest nodes,
  typed Linux symlinks and calibrated numeric drift are accepted, and every public
  input remains bounded and semantically verified before atomic output commitment.

  Replacement hosted run 32441990273 at source commit
  fb8f0800d16868965b70fa318f2c54c78c0b45d1 completed both platform runners,
  eighteen-family mutation suites, attestations, and bounded uploads. Ordinary CI
  run 32441990289 also succeeded. All four downloaded summary/manifest subjects
  verified with the exact frozen GitHub CLI attestation command. Production assembly
  then correctly rejected the genuine Windows fragment because its candidate checkout
  contained CRLF-converted source bytes: `popgp/renderer.py` was 8,614 bytes instead
  of the 8,341-byte Git blob, with exactly 273 inserted carriage returns.

  Fix commit db66c9f0c1e2d1d4c00f0256cc34cf56f5c60542 preserves the literal source
  boundary and removes the cause. The retained clone now stores
  `core.autocrlf=false` before detached checkout; the public evidence validator binds
  that exact clone argument contract, and the signed family-17 mutation matrix includes
  a regression proving the runner and validator cannot omit it. A fresh Windows probe
  reproduced byte identity: the checkout and Git blob both hash to
  `a09270caa9eb9890862d26648bccdcada8277a1b01fa80c07080bdf744aa44aa`.

  The same audit found and repaired a pre-refreeze ledger inconsistency. The packet now
  declares exactly the same five protocol receipt IDs as its frozen artifact set,
  including the mutation runner, and all five source/campaign copies and raw SHA-256
  values reconcile. The primary protocol hash is
  `9bca5f0b852627e588591e133820232be86e84e827b126b299b7e927bbfb1131`,
  the validator Git-blob SHA-256 is
  `9b529c287454b929e4c63aa33f25bf9171f9836ff346ff72cfc2bb5056b78db9`,
  and the canonical packet-rule SHA-256 is
  `98bbe33c2c5f2a2a321675d12b47e074e83f92ee3463a131f398c81c21d3367b`.

  Local verification is green: the complete raw-evidence contract passed 9/9 in
  216.35 seconds; the exact checkout regression and calibrated manifest control passed
  2/2; Ruff, the 652-line TeX checker, and review-guidance 9/9 passed. A new exact-SHA
  hosted run, unmodified two-platform production assembly, and fresh independent
  re-review remain mandatory before approval. This response does not authorize
  refreeze, holdout, reveal, result commitment, or claim promotion.

finding_responses:
  - finding_id: "VIA000-R2-PA3-PORTABLE-ASSEMBLY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      File and symlink environment records now have explicit disjoint canonical
      shapes. The cross-platform recomputation tolerance is independently calibrated
      between the genuine platform delta and the registered attack delta. Windows
      checkout conversion is now disabled before materialization, so literal source
      bytes remain identical to the candidate Git blobs on both platforms.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "scripts/check_validation_artifacts.py"
      - "tests/unit/test_validation_artifact_contract.py"
      - "tests/unit/test_viability_raw_evidence_contract.py"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json"
    fix_commits:
      - "ebfce474925d5dac616d24bd95b10758e80d40b9"
      - "26de7bab2b9c15b21a96738b9518166b2b19c1a8"
      - "87c0616fbaf8efd592f541870b58c17485d8f184"
      - "db66c9f0c1e2d1d4c00f0256cc34cf56f5c60542"
    verification:
      - command: "uv run --no-sync --frozen --no-editable python -m pytest -q -p no:cacheprovider tests/unit/test_viability_raw_evidence_contract.py tests/unit/test_via000_r2_assembler.py"
        result: "11 passed in 623.97 seconds at exact fix commit."
      - command: "all six generators plus python -m scripts.check_validation_artifacts --enforce-change-boundary"
        result: "Passed at exact fix commit from the external locked environment; final normal and ignored Git state was empty."
      - command: "GitHub Actions run 32439213137 plus production assembler replay"
        result: "Both original signed platforms completed and all subjects verified; genuine-byte replay exposed the calibrated node/status representations, which are fixed in the replacement protocol and covered locally."
      - command: "GitHub Actions runs 32441990273/32441990289 plus exact downloaded-fragment assembly"
        result: "Both platforms and ordinary CI succeeded and all four subjects verified; the assembler correctly exposed Windows CRLF conversion, whose cause is removed and frozen by db66c9f."
      - command: "test_runner_disables_checkout_line_ending_conversion plus fresh Windows clone probe"
        result: "The frozen clone argument is required, the legacy argument is rejected, and the Windows checkout exactly matches the candidate Git object and SHA-256."
    residual_risk: "A new exact-SHA Ubuntu and Windows run must still be attested, downloaded, verified, and assembled without modification."
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA1-EVIDENCE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      The missing external producer binding is now supplied by GitHub OIDC/Sigstore
      attestations over the exact summary and manifest. Verification is bound to the
      frozen repository, workflow, source commit, provenance predicate, and hosted
      runner before semantic processing or output.
    changed_files:
      - ".github/workflows/via000-r2-protocol.yml"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-ASSEMBLER.py"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-MUTATION-RUNNER.py"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RUNNER.ps1"
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000.json"
      - "scripts/check_viability_campaign.py"
      - "tests/unit/test_via000_r2_assembler.py"
      - "tests/unit/test_viability_raw_evidence_contract.py"
    fix_commits:
      - "ebfce474925d5dac616d24bd95b10758e80d40b9"
      - "26de7bab2b9c15b21a96738b9518166b2b19c1a8"
      - "87c0616fbaf8efd592f541870b58c17485d8f184"
      - "db66c9f0c1e2d1d4c00f0256cc34cf56f5c60542"
    verification:
      - command: "test_raw_evidence_contract_requires_verified_producer_attestation"
        result: "A structurally valid package whose external attestation fails is rejected and cannot retain passing capability Booleans."
      - command: "staged Git-blob hash and receipt-copy reconciliation"
        result: "All five protocol source/receipt pairs are byte-identical; the validator Git-blob hash and VIA-000 canonical rule hash match the draft manifest."
    residual_risk: "The protocol cannot defeat compromise or collusion of GitHub's OIDC/Sigstore control plane or an administrator authorized to run the exact signer workflow; that narrower limitation is explicit."
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA1-IDENTITY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "The previously resolved exact commit/tree/platform/contract bindings remain unchanged and now also bind the protocol source commit in every attested platform record."
    changed_files:
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-RAW-RESULTS.schema.json"
      - "scripts/check_viability_campaign.py"
    fix_commits:
      - "ebfce474925d5dac616d24bd95b10758e80d40b9"
    verification:
      - command: "final 11-test raw/assembler suite"
        result: "All identity and protocol-source cross-bindings passed."
    residual_risk: "A future workflow or platform-set change requires a new preregistration."
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA1-BLOCKED-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "The prior fail-closed blockage rule is retained: incomplete, unattested, or unavailable inputs produce no scientific result or commitment."
    changed_files: []
    fix_commits: []
    verification:
      - command: "final 11-test raw/assembler suite"
        result: "Partial and failed fragments remained invalid and output-free."
    residual_risk: "Infrastructure failure remains an invalid attempt rather than an author-selected terminal scientific outcome."
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA1-ASSEMBLY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "The previously resolved semantic-before-commit and custody-compatible output behavior remains, now preceded by external attestation verification."
    changed_files:
      - "protocols/POPGP-VIABILITY-R2-2026-08/VIA-000-ASSEMBLER.py"
      - "tests/unit/test_via000_r2_assembler.py"
    fix_commits:
      - "ebfce474925d5dac616d24bd95b10758e80d40b9"
    verification:
      - command: "final 11-test raw/assembler suite"
        result: "Valid structural evidence round-tripped through custody; invalid or partial evidence created neither output nor commitment."
    residual_risk: "A genuine signed two-platform round trip remains required before approval."
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA1-FREEZE-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "All five protocol artifacts, receipt copies, validator Git-blob bytes, and the canonical VIA-000 packet rule are rebound in the draft preregistration."
    changed_files:
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/PROTOCOL_MANIFEST.json"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/protocol.json"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/runner-protocol.ps1"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/raw-results.schema.json"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/assembler-protocol.py"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/receipts/VIA-000/mutation-runner.py"
    fix_commits:
      - "ebfce474925d5dac616d24bd95b10758e80d40b9"
      - "26de7bab2b9c15b21a96738b9518166b2b19c1a8"
      - "87c0616fbaf8efd592f541870b58c17485d8f184"
      - "db66c9f0c1e2d1d4c00f0256cc34cf56f5c60542"
    verification:
      - command: "Git index blob SHA-256, protocol-copy, exact-envelope, and canonical-rule audit"
        result: "All five source/copy/receipt bindings reconcile; receipt and artifact ID sets are identical; primary and packet preregistration envelopes are strictly equal."
    residual_risk: "The active campaign deliberately remains on the older snapshot until a zero-blocker review authorizes refreeze."
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA2-VISUAL-XPLAT-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "The independently resolved direct Ubuntu/Windows raster relation remains enforced at maximum channel delta four."
    changed_files: []
    fix_commits: []
    verification:
      - command: "test_raw_evidence_contract_parses_pdf_and_compares_platform_rasters"
        result: "Direct opposed platform drift remains rejected."
    residual_risk: "Fresh signed platform rasters must be reconciled after hosted execution."
    disagreement_ref: ""

  - finding_id: "VIA000-R2-PA2-CI-HISTORY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "The full-history ordinary CI checkout remains unchanged; a new exact-response run is requested."
    changed_files: []
    fix_commits: []
    verification:
      - command: "GitHub Actions ordinary CI run 32441990289"
        result: "Succeeded at predecessor calibration commit fb8f080; exact response-3 CI remains pending."
    residual_risk: "Exact response-3 ordinary CI is pending."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-VIA000-R2-PA3-PORTABLE-ASSEMBLY-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_validation_artifact_contract.py"
      - "tests/unit/test_viability_raw_evidence_contract.py"
      - ".github/workflows/via000-r2-protocol.yml"
    verification:
      - command: "typed Linux symlink, 133,309-node manifest, status separator, potential moment, and literal-checkout controls"
        result: "Canonical records pass, malformed records fail, honest numeric drift remains separated from the attack, the runner freezes core.autocrlf=false, and the validator rejects the legacy clone contract."
    rationale: "All observed genuine hosted representations now have exact controls; one new signed Ubuntu/Windows assembly remains the acceptance criterion."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA1-DUMMY-EVIDENCE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_raw_evidence_contract.py"
      - "tests/unit/test_via000_r2_assembler.py"
      - ".github/workflows/via000-r2-protocol.yml"
    verification:
      - command: "verified-producer-attestation negative"
        result: "A coherent structural package with rejected producer attestation fails capability validation and cannot assemble."
    rationale: "External attestation, rather than a package-local digest, now distinguishes accepted producer output from transport substitution."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA1-IDENTITY-CONTRACT-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_raw_evidence_contract.py"
    verification:
      - command: "final 11-test raw/assembler suite"
        result: "Candidate, tree, platform, contract, and protocol-source identities remain fail closed."
    rationale: "The prior satisfied matrix is retained and strengthened by attested source identity."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA1-BLOCKAGE-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r2_assembler.py"
      - "tests/unit/test_viability_raw_evidence_contract.py"
    verification:
      - command: "partial, nonzero, unattested, and blocked-input controls"
        result: "Inputs reject before output; blocked=true remains unavailable to R2 raw results."
    rationale: "Incomplete execution cannot become a scientific terminal state."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA1-ASSEMBLY-ROUNDTRIP-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_via000_r2_assembler.py"
    verification:
      - command: "final assembler/custody round trip"
        result: "Canonical commitment bytes pass custody/reveal validation; invalid evidence creates no output."
    rationale: "The prior satisfied test remains green after provenance is inserted ahead of semantic assembly."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA1-MANIFEST-BINDING-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/PROTOCOL_MANIFEST.json"
      - "reviews/viability/POPGP-VIABILITY-R2-2026-08/packets/VIA-000.yaml"
    verification:
      - command: "staged Git-blob and canonical packet-rule audit"
        result: "Validator blob, five protocol source/copy/receipt bindings, primary envelope, and VIA-000 rule reconcile."
    rationale: "All changed execution-contract bytes are frozen before a future activation."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA2-VISUAL-XPLAT-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - "tests/unit/test_viability_raw_evidence_contract.py"
    verification:
      - command: "direct platform-raster negative"
        result: "Opposed candidate-relative drift remains rejected by direct platform comparison."
    rationale: "The previously satisfied cross-platform visual relation is unchanged."
    disagreement_ref: ""

  - requested_test_id: "TST-VIA000-R2-PA2-CI-HISTORY-001"
    disposition: accepted
    implementation_status: implemented
    test_locations:
      - ".github/workflows/ci.yml"
    verification:
      - command: "GitHub Actions ordinary CI run 32441990289"
        result: "The full-history predecessor calibration run succeeded; exact response-3 hosted CI remains pending."
    rationale: "Full-history checkout remains configured and will be reverified at the response handoff."
    disagreement_ref: ""

new_or_changed_risks:
  - "Accepted raw evidence now depends on GitHub CLI 2.97.0 or newer and on availability of GitHub's Sigstore verification material retained in the bundle."
  - "The execution claim excludes compromise or collusion of GitHub's OIDC/Sigstore control plane or an administrator authorized to execute the exact signer workflow."
  - "The potential-moment tolerance is calibrated narrowly between one observed honest platform delta and the smallest registered attack; future numeric-stack changes require recalibration and re-review."
  - "Literal source evidence depends on the frozen clone retaining core.autocrlf=false before checkout; the command contract and family-17 regression now make omission fail closed."

external_actions:
  - action: "Run ordinary CI and the exact signed Ubuntu/Windows protocol workflow at the response handoff; download, verify, and assemble the genuine fragments without modification."
    owner: "builder"
    status: pending
    evidence_ref: ""
  - action: "Perform a fresh independent re-review of all eight findings/tests, external attestation verification, mutation execution binding, portable genuine assembly, hashes, and regressions."
    owner: "independent-reviewer"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "all eight finding IDs and eight requested-test IDs; forged/missing/replayed/wrong-repository/wrong-workflow/wrong-source attestations; signed mutation node/count/hash binding; exact genuine Ubuntu/Windows symlink and numeric portability; semantic no-commitment and custody round trip; frozen receipt/hash/rule bindings; exact hosted artifacts; regressions and broader attacks"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: |-
    Use a fresh isolated worktree at the response-containing handoff. Treat this
    response, local fixtures, action configuration, and hosted results as hypotheses.
    Verify the real attestation bundles with the exact frozen gh command, assemble
    the exact downloaded platform bytes, and modify only a new independent re-review
    artifact. Do not approve refreeze or holdout unless every historical finding and
    requested test is reconciled and the blocker count is zero.
```
