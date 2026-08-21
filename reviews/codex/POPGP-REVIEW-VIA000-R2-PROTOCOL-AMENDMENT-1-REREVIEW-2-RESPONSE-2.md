# Builder response: POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2-RESPONSE-2

```yaml
artifact_schema_version: 2
response_id: "POPGP-REVIEW-VIA000-R2-PROTOCOL-AMENDMENT-1-REREVIEW-2-RESPONSE-2"
response_round: 2
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
  This response retains every disposition and binding from response round 1. Both
  reported blockers are accepted and implemented beginning at fix commit
  ebfce474925d5dac616d24bd95b10758e80d40b9. The protocol no longer treats
  self-contained hashes or caller-supplied identity text as producer attribution.
  Its exact GitHub Actions workflow signs each platform summary and evidence manifest
  with the pinned official GitHub OIDC/Sigstore attestation action. The frozen
  assembler and public validator independently verify both subjects against the
  registered repository, signer workflow, exact protocol source commit, SLSA
  predicate, and hosted-runner constraint before accepting transported evidence.

  Separately authored mutation packages are removed. A new frozen mutation runner
  executes the preregistered verbose pytest matrix and derives all eighteen receipts
  from observed node IDs, expected case counts, suite hashes, timestamps, and oracle
  IDs before the summary and manifest are attested. A coherent internally hash-closed
  replacement without the external attestation is therefore rejected before output.

  Portable assembly accepts the runner's canonical Linux symlink manifest shape.
  The potential-moment absolute tolerance is calibrated at 2e-18, admitting the
  measured genuine Ubuntu-to-Windows recomputation difference of about 1.24345e-18
  while retaining rejection of the smallest registered attack at about 4.46e-18.
  Exact hosted run 32439213137 then completed both platform runners, frozen mutation
  suites, producer attestations, bundle retention, and bounded uploads at source
  commit 25d72fb993b9e6212a734873f5be85f462fc3d6b. All four downloaded subjects
  independently verified with the frozen GitHub CLI command. Production assembly of
  those genuine bytes exposed two additional representations within the same portable
  assembly scope: the 22,219-entry Ubuntu manifest measured 133,309 expanded nodes,
  above the generic 100,000-node cap, and the signed Git porcelain receipt carried an
  empty separator line. Commit 26de7bab2b9c15b21a96738b9518166b2b19c1a8 adds a
  150,000-node ceiling only for typed attested environment manifests while retaining
  the default ceiling and all other parser bounds. Commit
  87c0616fbaf8efd592f541870b58c17485d8f184 makes the runner emit canonical UTF-8
  status bytes and makes the verifier ignore empty separators while parsing and
  allowlisting every nonempty record. The replacement protocol also registers the
  new environment-bound regression in its signed mutation matrix; consequently the
  old immutable signed suite is not eligible under the replacement mapping and a new
  exact-SHA hosted run is required.

  Receipt copies, primary/packet envelopes, packet-rule hash, and exact Git-blob
  manifest bindings are synchronized through those follow-up commits. Amendment 4
  records the complete trust and portability boundary.

  Local evidence is green: the complete repository suite passed 377/377 in 1813.84
  seconds; the final exact-commit raw/assembler suite passed 11/11 in 623.97 seconds;
  Ruff, the 652-line TeX checker, all six generators, and the semantic/visual/source
  change-boundary checker passed from an external locked environment with clean normal
  and ignored Git state. The added manifest-bound/status regressions passed 11/11 in
  35.04 seconds and the full raw contract passed 8/8 in 243.61 seconds. Replacement
  GitHub attestations and genuine two-platform assembly remain mandatory hosted
  evidence before independent approval. This response does
  not authorize refreeze, holdout, reveal, result commitment, or claim promotion.

finding_responses:
  - finding_id: "VIA000-R2-PA3-PORTABLE-ASSEMBLY-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: |-
      File and symlink environment records now have explicit disjoint canonical
      shapes. The cross-platform recomputation tolerance is independently calibrated
      between the genuine platform delta and the registered attack delta.
    changed_files:
      - "scripts/check_viability_campaign.py"
      - "scripts/check_validation_artifacts.py"
      - "tests/unit/test_validation_artifact_contract.py"
      - "tests/unit/test_viability_raw_evidence_contract.py"
    fix_commits:
      - "ebfce474925d5dac616d24bd95b10758e80d40b9"
      - "26de7bab2b9c15b21a96738b9518166b2b19c1a8"
      - "87c0616fbaf8efd592f541870b58c17485d8f184"
    verification:
      - command: "uv run --no-sync --frozen --no-editable python -m pytest -q -p no:cacheprovider tests/unit/test_viability_raw_evidence_contract.py tests/unit/test_via000_r2_assembler.py"
        result: "11 passed in 623.97 seconds at exact fix commit."
      - command: "all six generators plus python -m scripts.check_validation_artifacts --enforce-change-boundary"
        result: "Passed at exact fix commit from the external locked environment; final normal and ignored Git state was empty."
      - command: "GitHub Actions run 32439213137 plus production assembler replay"
        result: "Both original signed platforms completed and all subjects verified; genuine-byte replay exposed the calibrated node/status representations, which are fixed in the replacement protocol and covered locally."
    residual_risk: "Replacement exact-SHA Ubuntu and Windows fragments must still be attested, downloaded, verified, and assembled on the supported reviewer platform."
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
    verification:
      - command: "Git index blob SHA-256, protocol-copy, exact-envelope, and canonical-rule audit"
        result: "All bindings reconcile; primary and packet preregistration envelopes are strictly equal."
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
      - command: "local complete repository suite"
        result: "377 passed in 1813.84 seconds."
    residual_risk: "Exact response-commit ordinary CI is pending."
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
      - command: "typed Linux symlink, 133,309-node genuine manifest, status-separator, and calibrated potential-moment controls"
        result: "Canonical symlink/status records pass, malformed records fail, the genuine manifest is within the typed 150,000-node bound, 1.25e-18 passes, and 4.46e-18 fails."
    rationale: "The genuine hosted representations are now covered; replacement exact signed Ubuntu/Windows assembly remains the acceptance criterion."
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
        result: "Validator blob, five protocol copies, primary envelope, and VIA-000 rule reconcile."
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
      - command: "local exact-history repository suite"
        result: "All 377 collected tests passed; exact response-commit hosted CI remains pending."
    rationale: "Full-history checkout remains configured and will be reverified at the response handoff."
    disagreement_ref: ""

new_or_changed_risks:
  - "Accepted raw evidence now depends on GitHub CLI 2.97.0 or newer and on availability of GitHub's Sigstore verification material retained in the bundle."
  - "The execution claim excludes compromise or collusion of GitHub's OIDC/Sigstore control plane or an administrator authorized to execute the exact signer workflow."
  - "The potential-moment tolerance is calibrated narrowly between one observed honest platform delta and the smallest registered attack; future numeric-stack changes require recalibration and re-review."

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
