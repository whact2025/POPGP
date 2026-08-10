# Builder response: POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-5

```yaml
artifact_schema_version: 1
response_id: "POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-5"
response_round: 5
response_date: "2026-08-10"

builder_seat: builder
builder_model_identity: "gpt-5.6-sol"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"

review_id: "POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-5"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-5.md"
review_commit: "c78ffe662562ae3a16105fda1dc31b3af27b787f"
candidate_commit_reviewed: "3974bbfab89d97bac79ce298285a7ce8f36f69fd"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden evaluator data or final labels were available. The approved review was preserved unchanged before remediation, and the MDS and disjoint-Bell counterexamples were replayed independently."

summary: "The round-5 reviewer approved the candidate with zero blockers and verified every prior finding and requested test. All fifteen new low-severity findings and all fifteen requested tests were nevertheless accepted and implemented in 9e6baec102530fdeda54b2b60c80b6e6ebdf0e22. The authoritative suite now has 159 passing tests; all six examples regenerate with no committed artifact drift. This builder report awaits independent verification and does not alter the review's approval or constitute external scientific validation."

finding_responses:
  - finding_id: "MDS-006"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The anchor degeneracy tolerance now scales with the leading singular value instead of an absolute unit floor. The review's 1e-10, 1e-11, and 1e-13 chain counterexamples raised at the frozen candidate and all return finite coordinates after the fix."
    changed_files: ["popgp/simulator.py", "tests/unit/test_simulator.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k mds", result: "exit 0; 29 passed, 20 deselected"}]
    residual_risk: "The rank decision remains an explicit rtol=1e-8 numerical convention."
    disagreement_ref: ""

  - finding_id: "MDS-007"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "Tie tolerance is now relative to the current maximum residual, and a selected residual is checked before normalization. Positive homogeneity holds through 1e-15 and the bare StopIteration reproducer now succeeds."
    changed_files: ["popgp/simulator.py", "tests/unit/test_simulator.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k mds", result: "exit 0; 29 passed, 20 deselected"}]
    residual_risk: "Canonical orientation remains a deterministic serialization frame, not a physical observable."
    disagreement_ref: ""

  - finding_id: "MDS-008"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The summary docstring now names maximum-volume anchors with label-ordered ties, and a regression pins that algorithm description."
    changed_files: ["popgp/simulator.py", "tests/unit/test_simulator.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k docstring", result: "exit 0; 1 passed"}]
    residual_risk: "Documentation changes to the selection policy now require a corresponding test update."
    disagreement_ref: ""

  - finding_id: "CACHE-002"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "Backend construction snapshots every substrate field. Any later substrate mutation raises with the changed field names and requires a new backend, before any Hamiltonian or cell cache can return stale data."
    changed_files: ["popgp/backend.py", "tests/unit/test_backend.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_backend.py", result: "exit 0; 20 passed"}]
    residual_risk: "This intentionally makes post-construction substrate mutation a loud API error; other SimulatorConfig sections remain mutable."
    disagreement_ref: ""

  - finding_id: "CACHE-004"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The redundant public compute_leakage cache parameter and optimizer plumbing were removed. The backend cell cache alone bounds uncached constructions to one per distinct ordered cell."
    changed_files: ["popgp/coarse_grain.py", "popgp/backend.py", "tests/unit/test_backend.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_backend.py -k distinct_cell", result: "exit 0; 1 passed"}]
    residual_risk: "Callers that used the newly introduced round-4 keyword must remove it; no repository caller did."
    disagreement_ref: ""

  - finding_id: "BACKEND-501"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "Full-space interaction terms are transient again. Cell generators are constructed directly in the cell Hilbert space from the same pair-term constructor, preserving REG-003 performance without retaining one dense full-space tensor per edge."
    changed_files: ["popgp/backend.py", "tests/unit/test_backend.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_backend.py", result: "exit 0; retained backend tensors after N=10 Hamiltonian construction remain within 3x one Hamiltonian"}]
    residual_risk: "Explicit callers of build_interaction_terms still own the returned dense list for as long as they retain it."
    disagreement_ref: ""

  - finding_id: "GEOM-501"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The absolute 1e-15 stress shortcut was removed. Nonzero targets retain scale-invariant stress through 1e-12; exactly degenerate targets produce infinite stress and poor_fit rather than a perfect geometric declaration."
    changed_files: ["popgp/simulator.py", "tests/unit/test_simulator.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k stress_and_geometry_status", result: "exit 0; 1 passed"}]
    residual_risk: "Infinite stress is a deliberate degeneracy signal and may require handling by future consumers of PiGeomResult."
    disagreement_ref: ""

  - finding_id: "GOV-013"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The artifact checker now requires the exact non-informational check-name set for each example. An acceptance check added to code and regenerated into JSON is rejected until it is explicitly registered in the per-example policy. Registry prose was narrowed to gates declared there."
    changed_files: ["scripts/check_validation_artifacts.py", "tests/unit/test_validation_artifact_contract.py", "docs/scientific_hardening/GATE_TEST_REGISTRY.md"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_validation_artifact_contract.py", result: "exit 0; 15 passed, including rejection of relative_entropy_fit_quality_floor"}]
    residual_risk: "Registering a new check remains a human-reviewed policy edit; it is no longer hidden in regenerated output."
    disagreement_ref: ""

  - finding_id: "GOV-009"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "Only the two exact canonical CI-authority clauses are permitted in quality-authority sentences, both are positively pinned, and the review's concessive-clause bypass has its own negative regression."
    changed_files: ["tests/unit/test_review_guidance.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 9 passed"}]
    residual_risk: "The lexical trigger set cannot recognize every possible natural-language synonym, so the canonical wording is intentionally strict."
    disagreement_ref: ""

  - finding_id: "GOV-007"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The Markdown parser requires exactly one header, a valid separator, one contiguous table, and accounting for every pipe-prefixed line. A whitespace-truncation regression fails explicitly."
    changed_files: ["tests/unit/test_review_guidance.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 9 passed"}]
    residual_risk: "The strict parser intentionally rejects additional Markdown tables in these two single-table governance documents."
    disagreement_ref: ""

  - finding_id: "GOV-008"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "Negative-control markers require exact AST equality, cited controls cannot be skipped, and pytest now runs with --strict-markers. Exact, pending-prefix, and skipped fixtures pin all branches."
    changed_files: ["tests/unit/test_review_guidance.py", "pyproject.toml", "docs/scientific_hardening/GATE_TEST_REGISTRY.md"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q", result: "exit 0; 159 passed with strict marker enforcement"}]
    residual_risk: "A marker remains an auditable author assertion; scientific adequacy still requires review of the cited control."
    disagreement_ref: ""

  - finding_id: "FRAMEWORK-003"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The parity guard ignores LaTeX comments, requires exactly one active locality row per source, and compares each complete expected row rather than four loose tokens."
    changed_files: ["tests/unit/test_claim_wording.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_claim_wording.py", result: "exit 0; 3 passed"}]
    residual_risk: "Any intentional wording revision to either row now requires an explicit parity-policy update."
    disagreement_ref: ""

  - finding_id: "FRAMEWORK-004"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The falsification matrix now states that disjoint-Bell non-separability is an exact-degeneracy artifact. A registered negative control evolves to dt=0.05 and pins the separable declaration, gap ratio above 1e4, false D*=1, and geometric_candidate status."
    changed_files: ["docs/scientific_hardening/FALSIFICATION_MATRIX.md", "docs/scientific_hardening/GATE_TEST_REGISTRY.md", "tests/scientific/test_topology_recovery.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/scientific/test_topology_recovery.py", result: "exit 0; 8 passed"}]
    residual_risk: "The control deliberately records a complete non-geometric gate failure; no topology-inference repair is claimed."
    disagreement_ref: ""

  - finding_id: "TSTGUARD-001"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "Nested collection no longer uses check=True, includes child stdout/stderr in failures, has a 60-second timeout, and a broken temporary module proves the filename reaches the assertion."
    changed_files: ["tests/unit/test_review_guidance.py"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 9 passed"}]
    residual_risk: "The diagnostic subprocess is intentionally bounded to 60 seconds."
    disagreement_ref: ""

  - finding_id: "TSTGUARD-002"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The passed-count guard now proves the suite contains no skip or xfail markers before equating collected with passed; a temporary skipped test is a demonstrated negative control."
    changed_files: ["tests/unit/test_review_guidance.py", "docs/scientific_hardening/REPRODUCIBILITY.md"]
    fix_commits: ["9e6baec102530fdeda54b2b60c80b6e6ebdf0e22"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 9 passed"}]
    residual_risk: "Future intentional skip/xfail use requires changing the reproducibility-record policy rather than silently changing count semantics."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-MDS-006"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_simulator.py::test_mds_canonicalization_is_scale_covariant_below_unit_scale"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k mds", result: "exit 0; 29 passed"}]
    rationale: "Covers all requested chain scales and ill-conditioned 8x3 families with a 1e-14 isometry bound."
    disagreement_ref: ""
  - requested_test_id: "TST-MDS-007"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_simulator.py::test_mds_canonical_frame_is_positively_homogeneous", "tests/unit/test_simulator.py::test_mds_anchor_scan_never_selects_a_zero_residual_row"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k mds", result: "exit 0; 29 passed"}]
    rationale: "Pins positive homogeneity and the exact zero-residual StopIteration reproducer."
    disagreement_ref: ""
  - requested_test_id: "TST-MDS-008"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_simulator.py::test_mds_canonicalization_docstring_names_maximum_volume_selection"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k docstring", result: "exit 0; 1 passed"}]
    rationale: "Pins the corrected summary line."
    disagreement_ref: ""
  - requested_test_id: "TST-CACHE-002"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_backend.py::test_backend_rejects_substrate_mutation_after_construction"]
    verification: [{command: "uv run pytest -q tests/unit/test_backend.py", result: "exit 0; 20 passed"}]
    rationale: "Coupling, family, and boundary mutations all raise for cached and previously unseen cells."
    disagreement_ref: ""
  - requested_test_id: "TST-CACHE-004"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_backend.py::test_optimize_cells_constructs_each_distinct_cell_generator_once"]
    verification: [{command: "uv run pytest -q tests/unit/test_backend.py -k distinct_cell", result: "exit 0; 1 passed"}]
    rationale: "The backend-only cache is instrumented at the uncached construction boundary."
    disagreement_ref: ""
  - requested_test_id: "TST-BACKEND-501"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_backend.py::test_exact_backend_does_not_retain_full_interaction_decomposition"]
    verification: [{command: "uv run pytest -q tests/unit/test_backend.py", result: "exit 0; 20 passed"}]
    rationale: "Reachable tensor bytes at N=10 are bounded to three Hamiltonians."
    disagreement_ref: ""
  - requested_test_id: "TST-GEOM-501"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_simulator.py::test_mds_stress_and_geometry_status_are_scale_invariant"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k stress_and_geometry_status", result: "exit 0; 1 passed"}]
    rationale: "Pins stress, D_star, and status across 1 through 1e-12 and rejects exact degeneracy as poor_fit."
    disagreement_ref: ""
  - requested_test_id: "TST-GOV-013"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_validation_artifact_contract.py::test_unregistered_noninformational_check_is_rejected"]
    verification: [{command: "uv run pytest -q tests/unit/test_validation_artifact_contract.py", result: "exit 0; 15 passed"}]
    rationale: "Replays the exact relative_entropy_fit_quality_floor mutation against content-intrinsic semantics."
    disagreement_ref: ""
  - requested_test_id: "TST-GOV-003C"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_competing_quality_authority_sentence_is_rejected", "tests/unit/test_review_guidance.py::test_documented_quality_commands_match_authoritative_ci"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 9 passed"}]
    rationale: "Rejects the exact concessive-clause bypass and positively pins both canonical declarations."
    disagreement_ref: ""
  - requested_test_id: "TST-GOV-007"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_markdown_table_parser_rejects_truncated_coverage"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 9 passed"}]
    rationale: "The whitespace-truncated table fixture raises."
    disagreement_ref: ""
  - requested_test_id: "TST-GOV-008"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_negative_control_marker_must_be_exact_and_unskipped", "pyproject.toml"]
    verification: [{command: "uv run pytest -q", result: "exit 0; 159 passed under --strict-markers"}]
    rationale: "Exact, prefix-pending, skipped, and unregistered-marker routes are all enforced."
    disagreement_ref: ""
  - requested_test_id: "TST-FRAMEWORK-003"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_claim_wording.py::test_framework_locality_status_discloses_qcmi_gap_in_both_sources"]
    verification: [{command: "uv run pytest -q tests/unit/test_claim_wording.py", result: "exit 0; 3 passed"}]
    rationale: "Active-row cardinality and exact full-row text reject comment decoys and inverted claims."
    disagreement_ref: ""
  - requested_test_id: "TST-FRAMEWORK-004"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/scientific/test_topology_recovery.py::test_perturbed_disjoint_bell_control_becomes_separable_and_still_declares_geometry"]
    verification: [{command: "uv run pytest -q tests/scientific/test_topology_recovery.py", result: "exit 0; 8 passed"}]
    rationale: "Pins the measured dt=0.05 gate failure and registers it as a negative control."
    disagreement_ref: ""
  - requested_test_id: "TST-TSTGUARD-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_collection_failure_reports_the_offending_module"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 9 passed"}]
    rationale: "The child collection failure names test_broken.py."
    disagreement_ref: ""
  - requested_test_id: "TST-TSTGUARD-002"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_skip_marker_invalidates_documented_pass_count", "tests/unit/test_review_guidance.py::test_reproducibility_record_matches_collected_test_count"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 9 passed"}]
    rationale: "A skip marker invalidates the passed-count invariant."
    disagreement_ref: ""

new_or_changed_risks:
  - "ExactBackend now rejects all substrate-field mutation after construction; callers must construct a new backend for a substrate sweep."
  - "build_interaction_terms is intentionally transient, trading repeated explicit decomposition calls for bounded retained memory; the partition path constructs and caches only small cell-space generators."
  - "Exactly degenerate target geometry now reports infinite stress and poor_fit rather than perfect stress."
  - "Every new non-informational example check requires an explicit checker-policy update, making gate/check additions visible in code review."
  - "The perturbed disjoint-Bell result is recorded as a complete gate failure; no inference repair or stronger scientific claim is made."
  - "The source law remains a supplied convention/identity, dimension remains parameter-dependent, the KMS gate does not scale, and external cross-platform/scientific validation remains open."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "all fifteen round-5 findings, all fifteen requested tests, regressions, exact-SHA CI, artifact identity, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Round 5 already approved with zero blockers. This optional follow-up should independently verify the additional housekeeping remediation; builder dispositions are not resolution labels."
```

The builder does not assign final resolution status. That determination belongs to
an independent re-review artifact.
