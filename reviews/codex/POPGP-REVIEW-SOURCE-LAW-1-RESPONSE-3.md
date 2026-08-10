# Builder response: POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-3

```yaml
artifact_schema_version: 1
response_id: "POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-3"
response_round: 3
response_date: "2026-08-10"

builder_seat: builder
builder_model_identity: "gpt-5.6-sol"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"

review_id: "POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-3"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-3.md"
review_commit: "20b6e9217582aefd59467bfccc55a4879548f085"
candidate_commit_reviewed: "763fd1857d0a1298aa9858bc7b7c698c38833ef7"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "Authenticated maintainer access was used only to push and verify the public exact-SHA GitHub Actions run. No hidden evaluator data or final labels were available."

summary: "All ten new findings and the carried SLAW-003 blocker were accepted and implemented in fix commit 0e8d7bee58a317a87cd59301c2bf34fb0db85a4f. Thirteen requested tests were accepted in full. TST-GOV-004 was implemented except for its requirement that a passing sweep vary by more than its own acceptance tolerance; that clause was not adopted because it contradicts robustness within tolerance. This is a builder report pending independent re-review."

finding_responses:
  - finding_id: "SLAW-003"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "The carried blocker remained open because the closure row could be read as local conservation. It now states the global invariant, profile spreading, identity-only status, and absence of a local continuity/Bianchi law."
    changed_files: ["docs/scientific_hardening/FALSIFICATION_MATRIX.md", "tests/unit/test_claim_wording.py"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/unit/test_claim_wording.py"
        result: "exit 0; 2 passed"
      - command: "uv run pytest -q"
        result: "exit 0; 118 passed"
    residual_risk: "Final outcome remains for the independent reviewer; no discrete continuity current was added."
    disagreement_ref: ""

  - finding_id: "LIB-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Pi_res leakage now obtains each cell generator from the backend's configured interaction terms instead of rebuilding a Heisenberg-only approximation. Both supported exact Hamiltonian families reconstruct H, and the four-qubit Ising control selects the contiguous partition at leakage 5.712651e-4."
    changed_files: ["popgp/backend.py", "popgp/coarse_grain.py", "popgp/simulator.py", "tests/unit/test_backend.py", "tests/unit/test_simulator.py", "docs/scientific_hardening/THEORY_CODE_GAP.md"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/unit/test_backend.py tests/unit/test_simulator.py -k 'cell_generators or local_energy_operators_split or ising_pi_res'"
        result: "exit 0; 5 passed, 32 deselected"
      - command: "uv run pytest -q"
        result: "exit 0; 118 passed"
    residual_risk: "Backends without a cell-Hamiltonian capability continue to fail explicitly rather than silently substituting a family."
    disagreement_ref: ""

  - finding_id: "GOV-003"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The GitHub Actions workflow is now named as authoritative, and README plus the launch runbook reproduce its exact uv command set. A regression parses all three and requires set equality."
    changed_files: ["README.md", "docs/governance/AGENT_REVIEW_WORKFLOW.md", "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md", "tests/unit/test_review_guidance.py"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/unit/test_review_guidance.py"
        result: "exit 0; 3 passed"
    residual_risk: "The regression covers uv quality commands; non-uv setup actions remain workflow implementation details."
    disagreement_ref: ""

  - finding_id: "SLAW-004"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The closure row now uses the review's proposed global-only wording. The guard semantically rejects sentences coupling local/site/profile/decomposition language to conservation unless globally qualified with spreading or explicitly disclaiming a local law."
    changed_files: ["docs/scientific_hardening/FALSIFICATION_MATRIX.md", "tests/unit/test_claim_wording.py"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/unit/test_claim_wording.py"
        result: "exit 0; 2 passed"
    residual_risk: "The semantic guard is sentence-based and deliberately scoped to live claim documents."
    disagreement_ref: ""

  - finding_id: "CI-005"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The artifact checker now enforces an exact informational-check allowlist, prohibits failed informational checks, and requires overall_pass to equal the non-informational conjunction for every artifact."
    changed_files: ["scripts/check_validation_artifacts.py", "tests/unit/test_validation_artifact_contract.py", "docs/scientific_hardening/DECISIONS.md"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/unit/test_validation_artifact_contract.py"
        result: "exit 0; 14 passed"
      - command: "uv run python scripts/check_validation_artifacts.py"
        result: "exit 0; validation artifact contracts and required visual outputs are valid"
    residual_risk: "Allowlist changes remain policy changes requiring review; current entries are pinned."
    disagreement_ref: ""

  - finding_id: "LIB-004"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "Degenerate eigenvector blocks are canonically framed before a requested dimension can truncate them, and rank-deficient outputs are canonicalized in their represented subspace then zero-padded. Re-mixed eigenspaces are invariant for every candidate dimension on grid and five-cycle fixtures."
    changed_files: ["popgp/simulator.py", "tests/unit/test_simulator.py", "docs/scientific_hardening/REPRODUCIBILITY.md", "examples/physics_qg/chain_1d/results/validation.json", "examples/physics_qg/grid_2d/results/validation.json", "examples/physics_qg/gravity_well/results/validation.json"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/unit/test_simulator.py -k 'mds or degenerate or rank_deficient'"
        result: "exit 0; 5 passed, 20 deselected"
    residual_risk: "Canonicalization is label-ordered by design; it fixes serialization frame, not physical coordinate labels."
    disagreement_ref: ""

  - finding_id: "SLAW-005"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "Global-H drift is relabelled generator/observable implementation consistency and removed as a source-law falsifier. A mutation proves it remains zero for a changed local split and unrelated state, and drifts only when the measured observable differs from the generator."
    changed_files: ["README.md", "docs/scientific_hardening/FALSIFICATION_MATRIX.md", "docs/scientific_hardening/DECISIONS.md", "examples/physics_qg/source_law_many_body/__main__.py", "examples/physics_qg/source_law_many_body/README.md", "tests/scientific/test_many_body_source_law.py", "tests/unit/test_claim_wording.py"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/scientific/test_many_body_source_law.py -k generator_observable"
        result: "exit 0; 1 passed, 18 deselected"
    residual_risk: "A genuine local continuity residual remains open and is not claimed."
    disagreement_ref: ""

  - finding_id: "RTV-004"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The symmetric endpoint split is now pinned directly against build_interaction_terms for Heisenberg and Ising two-site chains."
    changed_files: ["tests/unit/test_backend.py"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/unit/test_backend.py -k local_energy_operators_split"
        result: "exit 0; 2 passed, 10 deselected"
    residual_risk: "The split remains a declared microscopic convention, not a uniquely derived density."
    disagreement_ref: ""

  - finding_id: "STAT-002"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The quadratic asymptote is now fit in response space as y=c0*x^2+c1*x^3 with covariance from response residuals, equivalent to inverse-variance weighting in the divided domain under the declared constant absolute floor."
    changed_files: ["popgp/diagnostics.py", "tests/unit/test_diagnostics.py", "examples/physics_qg/source_law_many_body/README.md", "examples/physics_qg/source_law_many_body/results/validation.json"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/unit/test_diagnostics.py"
        result: "exit 0; 7 passed"
      - command: "uv run python -m examples.physics_qg.source_law_many_body"
        result: "exit 0; all checks retained their outcomes; coefficient relative error improved to 3.812e-8"
    residual_risk: "The fit remains a deterministic numerical diagnostic, not a sampling inference."
    disagreement_ref: ""

  - finding_id: "STAT-003"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "slope_standard_error was renamed slope_residual_scale throughout code and artifacts, uncertainty wording was removed, and example output now says residual scale."
    changed_files: ["popgp/diagnostics.py", "scripts/check_validation_artifacts.py", "examples/physics_qg/source_law/__main__.py", "examples/physics_qg/source_law_many_body/__main__.py", "tests/scientific/test_source_law_scaling.py", "tests/unit/test_claim_wording.py"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/unit/test_claim_wording.py tests/scientific/test_source_law_scaling.py"
        result: "exit 0; 5 passed"
    residual_risk: "The renamed public dataclass field is an intentional API correction."
    disagreement_ref: ""

  - finding_id: "STAT-004"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "assess_quadratic_response now rejects non-1D, mismatched, short, or non-strictly-increasing amplitudes before selecting nested windows; the permutation-invariant lower-level fit remains unrestricted."
    changed_files: ["popgp/diagnostics.py", "tests/unit/test_diagnostics.py"]
    fix_commits: ["0e8d7bee58a317a87cd59301c2bf34fb0db85a4f"]
    verification:
      - command: "uv run pytest -q tests/unit/test_diagnostics.py -k unsorted"
        result: "exit 0; 1 passed, 6 deselected"
    residual_risk: "Callers must order amplitude sweeps explicitly, matching the nested-window semantics."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-LIB-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_backend.py::test_cell_generators_and_intercell_terms_reconstruct_hamiltonian", "tests/unit/test_simulator.py::test_ising_pi_res_selects_contiguous_two_site_cells"]
    verification: [{command: "uv run pytest -q tests/unit/test_backend.py tests/unit/test_simulator.py -k 'cell_generators or ising_pi_res'", result: "exit 0; 3 passed"}]
    rationale: "Covers both families and the qualitative Ising selection counterexample."
    disagreement_ref: ""

  - requested_test_id: "TST-SLAW-004"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_claim_wording.py::test_energy_conservation_claim_distinguishes_global_sum_from_profile"]
    verification: [{command: "uv run pytest -q tests/unit/test_claim_wording.py", result: "exit 0; 2 passed"}]
    rationale: "Semantic regression covers the corrected closure row and injected local-conservation phrasing."
    disagreement_ref: ""

  - requested_test_id: "TST-GOV-003"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_documented_quality_commands_match_authoritative_ci"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 3 passed"}]
    rationale: "README, runbook, and CI uv command sets are equal and governance names CI as authoritative."
    disagreement_ref: ""

  - requested_test_id: "TST-CI-005"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_validation_artifact_contract.py::test_headline_must_equal_noninformational_check_conjunction", "tests/unit/test_validation_artifact_contract.py::test_failing_check_cannot_be_demoted_to_informational", "tests/unit/test_validation_artifact_contract.py::test_committed_validation_artifacts_are_internally_consistent"]
    verification: [{command: "uv run pytest -q tests/unit/test_validation_artifact_contract.py", result: "exit 0; 14 passed"}]
    rationale: "Pins all three requested semantic invariants."
    disagreement_ref: ""

  - requested_test_id: "TST-LIB-004"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_simulator.py::test_classical_mds_is_invariant_to_degenerate_eigenbasis_for_every_dimension", "tests/unit/test_simulator.py::test_rank_deficient_mds_canonical_frame_fixes_degenerate_rotation"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k 'degenerate_eigenbasis or rank_deficient'", result: "exit 0; 3 passed"}]
    rationale: "Re-mixes legitimate degenerate eigenbases on the two requested fixtures for every candidate dimension."
    disagreement_ref: ""

  - requested_test_id: "TST-SLAW-005"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/scientific/test_many_body_source_law.py::test_global_energy_drift_is_generator_observable_identity_not_local_gate"]
    verification: [{command: "uv run pytest -q tests/scientific/test_many_body_source_law.py -k generator_observable", result: "exit 0; 1 passed"}]
    rationale: "Covers changed decomposition, unrelated state, and mismatched observable."
    disagreement_ref: ""

  - requested_test_id: "TST-RTV-004"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_backend.py::test_two_site_local_energy_operators_split_interaction_equally"]
    verification: [{command: "uv run pytest -q tests/unit/test_backend.py -k local_energy_operators_split", result: "exit 0; 2 passed"}]
    rationale: "Pins the endpoint split for both supported interaction families."
    disagreement_ref: ""

  - requested_test_id: "TST-STAT-002"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_diagnostics.py::test_response_space_weighting_improves_committed_kms_coefficient"]
    verification: [{command: "uv run pytest -q tests/unit/test_diagnostics.py", result: "exit 0; 7 passed"}]
    rationale: "The response-space fit is compared directly with the historical transformed unweighted estimator."
    disagreement_ref: ""

  - requested_test_id: "TST-STAT-003"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_claim_wording.py::test_power_law_residual_scale_is_not_labelled_as_sampling_uncertainty", "tests/scientific/test_source_law_scaling.py::test_clock_potential_obeys_linear_solver_homogeneity_identity"]
    verification: [{command: "uv run pytest -q tests/unit/test_claim_wording.py tests/scientific/test_source_law_scaling.py", result: "exit 0; 5 passed"}]
    rationale: "Pins the field name/output wording and explains the deterministic 0.01 residual criterion."
    disagreement_ref: ""

  - requested_test_id: "TST-STAT-004"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_diagnostics.py::test_quadratic_assessment_rejects_unsorted_amplitudes"]
    verification: [{command: "uv run pytest -q tests/unit/test_diagnostics.py -k unsorted", result: "exit 0; 1 passed"}]
    rationale: "A permuted sweep is rejected at the assessment boundary."
    disagreement_ref: ""

  - requested_test_id: "TST-LIB-005"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_simulator.py::test_gravitational_redshift_is_independent_of_default_dtype"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k default_dtype", result: "exit 0; 1 passed"}]
    rationale: "Public float inputs now produce bitwise-equal results under float32 and float64 defaults."
    disagreement_ref: ""

  - requested_test_id: "TST-GOV-004"
    disposition: partially-accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_every_declared_gate_has_an_executable_negative_control", "docs/scientific_hardening/GATE_TEST_REGISTRY.md"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 3 passed"}]
    rationale: "Implemented the governance rule, stable gate IDs, registry, before/after mutation records, executable-node meta-test, and demonstrated negative controls. Did not require a passing robustness sweep's statistic range to exceed its own tolerance: that condition is incompatible with the stated goal of remaining stable within tolerance. D012 records this bounded exception."
    disagreement_ref: ""

  - requested_test_id: "TST-GOV-005"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_review_identity_schema_has_typed_independence_declaration", "docs/governance/REVIEWER_IDENTITY.md", "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 3 passed"}]
    rationale: "Adds all requested typed fields and the required external-validation rule while retaining prose disclosure."
    disagreement_ref: ""

  - requested_test_id: "TST-CI-003"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["scripts/check_validation_artifacts.py", "docs/scientific_hardening/REPRODUCIBILITY.md"]
    verification: [{command: "uv run python scripts/check_validation_artifacts.py", result: "exit 0; validation artifact contracts and required visual outputs are valid"}]
    rationale: "Adopted the request's explicit-documentation alternative: the checker and reproducibility record now say visual presence/nonempty checks do not prove current-run rewrites or cross-platform pixel identity."
    disagreement_ref: ""

new_or_changed_risks:
  - "Response-space quadratic fitting intentionally changes committed diagnostic values while preserving all scientific gate outcomes."
  - "MDS eigenspace canonicalization produces roundoff-level changes in geometry artifacts; the exact-SHA Linux workflow regenerated and accepted them."
  - "TST-GOV-004 is partial only for the sweep-range-greater-than-tolerance clause; all gate-registration and negative-control requirements were implemented."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact remediation SHA."
    owner: "Richard Fuoco"
    status: complete
    evidence_ref: "https://github.com/whact2025/POPGP/actions/runs/31406872597"

rereview_request:
  requested: true
  scope: "all round-3 findings, the carried SLAW-003 blocker, all requested tests, regressions, exact-SHA CI, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Please reproduce the counterexamples and assign independent outcomes; builder dispositions are not resolution labels."
```

The builder does not assign final resolution status. That determination belongs to
the independent re-review artifact.
