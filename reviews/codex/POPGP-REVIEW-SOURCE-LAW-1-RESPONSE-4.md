# Builder response: POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-4

```yaml
artifact_schema_version: 1
response_id: "POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-4"
response_round: 4
response_date: "2026-08-10"

builder_seat: builder
builder_model_identity: "gpt-5.6-sol"
builder_model_version: "unknown"
builder_operator: "Richard Fuoco"

review_id: "POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-4"
review_artifact: "reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-4.md"
review_commit: "676e03d6de2405414df417a4d47c444733514256"
candidate_commit_reviewed: "4caa4f16b407a47a9531442751ccde1e446b8271"

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: "No hidden evaluator data or final resolution labels were available. The review artifact was committed mechanically before remediation and its reported counterexamples were reproduced independently."

summary: "All five new findings and all seven requested tests were accepted and implemented in fix commit 21baacd2c98f86533c51a38080294ad1a05ea73d. REG-001's mutual-recursion blocker is replaced by a terminating, isometric canonical-frame construction. The performance, governance, and reproducibility gaps are pinned by executable regressions. The review's unverified TeX/Markdown locality-status observation was also confirmed and corrected. This is a builder report pending independent re-review."

finding_responses:
  - finding_id: "REG-001"
    blocking_as_reported: true
    disposition: accepted
    implementation_status: implemented
    rationale: "Canonicalization now computes the represented rank once from global singular values, selects a label-stable maximum-volume row basis, and applies an SVD polar factor. It no longer recurses from a full represented rank back into the same failing greedy-anchor condition."
    changed_files: ["popgp/simulator.py", "tests/unit/test_simulator.py", "examples/physics_qg/chain_1d/results/validation.json", "examples/physics_qg/grid_2d/results/validation.json", "examples/physics_qg/grid_2d/results/embedding.png", "examples/physics_qg/gravity_well/results/validation.json", "examples/physics_qg/gravity_well/results/gravity_embedding.png"]
    fix_commits: ["21baacd2c98f86533c51a38080294ad1a05ea73d"]
    verification:
      - command: "uv run pytest -q tests/unit/test_simulator.py -k mds"
        result: "exit 0; 23 passed, 20 deselected"
      - command: "uv run pytest -q"
        result: "exit 0; 142 passed"
    residual_risk: "The label-stable frame is a serialization convention, not a claim that a degenerate MDS eigenspace has a unique physical orientation."
    disagreement_ref: ""

  - finding_id: "REG-003"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "ExactBackend now memoizes configured interaction terms and ordered cell Hamiltonians. optimize_cells shares a cell-generator cache across every candidate partition, reducing generator construction to at most once per distinct ordered cell."
    changed_files: ["popgp/backend.py", "popgp/coarse_grain.py", "tests/unit/test_backend.py"]
    fix_commits: ["21baacd2c98f86533c51a38080294ad1a05ea73d"]
    verification:
      - command: "uv run pytest -q tests/unit/test_backend.py"
        result: "exit 0; 16 passed"
      - command: "uv run python -m examples.physics_qg.chain_1d"
        result: "exit 0; selected cells, leakage, retention, topology, and gates retained their outcomes"
    residual_risk: "The exact backend remains exponentially scaling by design; memoization removes repeated construction but does not make dense exact evolution scalable."
    disagreement_ref: ""

  - finding_id: "MDS-005"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The inverse square root of an ill-conditioned anchor Gram was replaced by the SVD polar factor of the anchor matrix itself. The returned transform is orthogonal to floating-point precision through the rank threshold and in the zero-padded rank-deficient branch."
    changed_files: ["popgp/simulator.py", "tests/unit/test_simulator.py", "examples/physics_qg/chain_1d/results/validation.json", "examples/physics_qg/grid_2d/results/validation.json", "examples/physics_qg/grid_2d/results/embedding.png", "examples/physics_qg/gravity_well/results/validation.json", "examples/physics_qg/gravity_well/results/gravity_embedding.png"]
    fix_commits: ["21baacd2c98f86533c51a38080294ad1a05ea73d"]
    verification:
      - command: "uv run pytest -q tests/unit/test_simulator.py -k mds"
        result: "exit 0; 23 passed, 20 deselected; maximum relative distance-change assertions are below 1e-14 over all requested scales and both branches"
      - command: "uv run python scripts/check_validation_artifacts.py"
        result: "exit 0 after a second full example regeneration; contracts and required visual outputs valid"
    residual_risk: "The intended pure rotation changed serialized coordinates and two rendered embeddings; scientific gates and pairwise geometry are unchanged."
    disagreement_ref: ""

  - finding_id: "GOV-006"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "Gate IDs now live in individual falsification-matrix hypothesis rows. The registry test parses both tables row-wise, requires exactly one ID per gated row, pins the explicit ungated rows, requires every registry row to cite an existing test node, and AST-verifies a machine-readable negative_control marker on every cited control."
    changed_files: ["docs/scientific_hardening/FALSIFICATION_MATRIX.md", "docs/scientific_hardening/GATE_TEST_REGISTRY.md", "pyproject.toml", "tests/unit/test_review_guidance.py", "tests/scientific/test_many_body_source_law.py", "tests/scientific/test_source_law_controls.py", "tests/scientific/test_topology_recovery.py", "tests/unit/test_diagnostics.py"]
    fix_commits: ["21baacd2c98f86533c51a38080294ad1a05ea73d"]
    verification:
      - command: "uv run pytest -q tests/unit/test_review_guidance.py"
        result: "exit 0; 4 passed, including executable-control, authority-scope, and recorded-count checks"
      - command: "uv run pytest -q"
        result: "exit 0; 142 passed with no unknown-marker warnings"
    residual_risk: "The registry proves executable demonstrated controls exist; it does not by itself establish that a gate is scientifically sufficient."
    disagreement_ref: ""

  - finding_id: "REPRO-002"
    blocking_as_reported: false
    disposition: accepted
    implementation_status: implemented
    rationale: "The reproducibility table now records 142 passing tests, and a regression compares that number with pytest --collect-only at the current tree."
    changed_files: ["docs/scientific_hardening/REPRODUCIBILITY.md", "tests/unit/test_review_guidance.py"]
    fix_commits: ["21baacd2c98f86533c51a38080294ad1a05ea73d"]
    verification:
      - command: "uv run pytest -q tests/unit/test_review_guidance.py::test_reproducibility_record_matches_collected_test_count"
        result: "exit 0; 1 passed"
      - command: "uv run pytest -q"
        result: "exit 0; 142 passed"
    residual_risk: "The count is intentionally a current-tree invariant and must change whenever test collection changes."
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: "TST-REG-001"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_simulator.py::test_mds_canonicalization_terminates_when_first_label_is_symmetry_center", "tests/unit/test_simulator.py::test_mds_direct_anchor_scan_uses_global_rank_tolerance"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k mds", result: "exit 0; 23 passed, 20 deselected"}]
    rationale: "Covers five- and six-node stars for every candidate dimension, the relabelled 3x3 grid, and the review's direct rank-two fixture."
    disagreement_ref: ""

  - requested_test_id: "TST-REG-003"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_backend.py::test_backend_reuses_interaction_and_cell_hamiltonian_caches", "tests/unit/test_backend.py::test_optimize_cells_constructs_each_distinct_cell_generator_once"]
    verification: [{command: "uv run pytest -q tests/unit/test_backend.py", result: "exit 0; 16 passed"}]
    rationale: "Pins object reuse and bounds a full optimization run to one construction per distinct cell."
    disagreement_ref: ""

  - requested_test_id: "TST-MDS-005"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_simulator.py::test_mds_canonical_frame_is_isometric_near_rank_threshold"]
    verification: [{command: "uv run pytest -q tests/unit/test_simulator.py -k mds", result: "exit 0; 23 passed, 20 deselected"}]
    rationale: "Sweeps every requested delta with and without a zero-padded column and requires maximum relative pairwise-distance drift below 1e-14."
    disagreement_ref: ""

  - requested_test_id: "TST-GOV-006"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_every_declared_gate_has_an_executable_negative_control"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 4 passed"}]
    rationale: "The parser and negative_control markers reject missing/pending citations, unrelated unmarked tests, and active rows without a gate ID."
    disagreement_ref: ""

  - requested_test_id: "TST-GOV-003B"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_documented_quality_commands_match_authoritative_ci"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 4 passed"}]
    rationale: "Every sentence in README.md, docs/governance, or docs/reviews that designates an authoritative quality/pre-freeze suite must name .github/workflows/ci.yml."
    disagreement_ref: ""

  - requested_test_id: "TST-REPRO-002"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_review_guidance.py::test_reproducibility_record_matches_collected_test_count"]
    verification: [{command: "uv run pytest -q tests/unit/test_review_guidance.py", result: "exit 0; 4 passed"}]
    rationale: "The current documentation count is derived from and checked against live collection."
    disagreement_ref: ""

  - requested_test_id: "TST-LEAKGUARD-006"
    disposition: accepted
    implementation_status: implemented
    test_locations: ["tests/unit/test_backend.py::test_cell_hamiltonian_validates_indices_and_backend_capability", "tests/unit/test_backend.py::test_compute_leakage_requires_backend_edges_and_coupling"]
    verification: [{command: "uv run pytest -q tests/unit/test_backend.py", result: "exit 0; 16 passed"}]
    rationale: "Pins empty, duplicate, negative, and high indices, the base-backend capability error, and both substrate-consistency guards."
    disagreement_ref: ""

new_or_changed_risks:
  - "The canonical-frame correction intentionally changes serialized coordinate orientation and two rendered embeddings while preserving pairwise distances and all scientific gate outcomes."
  - "Cell-Hamiltonian caching trades bounded exact-backend memory for elimination of repeated exponential construction."
  - "The review's unverified manuscript observation was confirmed: framework.tex overstated QCMI implementation relative to framework.md. The TeX row now says Definition / partial prototype, identifies pairwise MI and blind routing as implemented, discloses that QCMI screening is not implemented, and names the non-geometric-control failure; a source-parity regression pins this disclosure."
  - "The review's six open scientific questions remain open except for the TeX/Markdown wording mismatch; no derivation, scalable KMS gate, intrinsic dimension uniqueness, local continuity law, or cross-platform pixel-identity claim was added."

external_actions:
  - action: "Run the complete GitHub Actions workflow on the exact response handoff SHA."
    owner: "Richard Fuoco"
    status: pending
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "all round-4 findings, all requested tests, regressions, manuscript-status parity, exact-SHA CI, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: "Please reproduce the original counterexamples and assign independent outcomes; builder dispositions are not resolution labels."
```

The builder does not assign final resolution status. That determination belongs to
the independent re-review artifact.
