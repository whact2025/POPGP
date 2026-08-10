# Independent re-review: POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-3

```yaml
artifact_schema_version: 1
review_id: "POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-3"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: claude-opus-5
reviewer_model_version: unknown
reviewer_operator: Richard Fuoco
review_date: 2026-08-10
commit_reviewed: 763fd1857d0a1298aa9858bc7b7c698c38833ef7
baseline_commit: c03800e47ba4988bd125b81acd3a1b6ae07728f7
prior_review_ref: "1dd86b3ba0c4a4aaea3b3027f2263a547c8a7eb6:reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-2.md"
builder_response_ref: "reviews/codex/POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-2.md at 763fd1857d0a1298aa9858bc7b7c698c38833ef7"
context_hash: 53ff112b28572dac16e66c0281307b047db2126c
context_hash_method: "git rev-parse \"763fd1857d0a1298aa9858bc7b7c698c38833ef7^{tree}\""
files_reviewed:
  - "Frozen reviewer worktree: C:/src/POPGP-review-source-law-rereview-3 at 763fd1857d0a1298aa9858bc7b7c698c38833ef7"
  - "Candidate diff: git diff --stat b519c7dafd4488d587a1edcbe14891f0c83aedb5 763fd1857d0a1298aa9858bc7b7c698c38833ef7 (7 paths, +622/-0)"
  - "Full frozen-tree diff from baseline: git diff --name-only c03800e47ba4988bd125b81acd3a1b6ae07728f7 adbfab58c20dc28f4cf05b601f81480207d74415 (74 paths)"
  - "C:/src/POPGP-review-source-law-rereview-3/README.md"
  - "C:/src/POPGP-review-source-law-rereview-3/.github/workflows/ci.yml"
  - "C:/src/POPGP-review-source-law-rereview-3/pyproject.toml"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/governance/REVIEWER_IDENTITY.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/templates/DISAGREEMENT_LOG_TEMPLATE.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/framework.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/framework.tex"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/simulator_analysis_and_plan.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/validation_report_v0.12.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/scientific_hardening/DECISIONS.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/scientific_hardening/PROJECT_PLAN.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/scientific_hardening/REPRODUCIBILITY.md"
  - "C:/src/POPGP-review-source-law-rereview-3/docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/__init__.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/backend.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/capacity.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/coarse_grain.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/config.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/diagnostics.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/information.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/renderer.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/simulator.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/geometry/closure.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/geometry/local_metric.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp/geometry/regge.py"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp_engine/renderer/src/regge.cu"
  - "C:/src/POPGP-review-source-law-rereview-3/popgp_engine/renderer/include/regge.h"
  - "C:/src/POPGP-review-source-law-rereview-3/scripts/check_tex.py"
  - "C:/src/POPGP-review-source-law-rereview-3/scripts/check_validation_artifacts.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/unit/test_simulator.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/unit/test_diagnostics.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/unit/test_claim_wording.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/unit/test_regge_proxy.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/unit/test_validation_artifact_contract.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/unit/test_backend.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/unit/test_local_metric.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/scientific/test_many_body_source_law.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/scientific/test_source_law_controls.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/scientific/test_source_law_scaling.py"
  - "C:/src/POPGP-review-source-law-rereview-3/tests/scientific/test_topology_recovery.py"
  - "C:/src/POPGP-review-source-law-rereview-3/examples/physics_qg/chain_1d/ (README.md, __main__.py, results/validation.json)"
  - "C:/src/POPGP-review-source-law-rereview-3/examples/physics_qg/grid_2d/ (README.md, __main__.py, results/validation.json)"
  - "C:/src/POPGP-review-source-law-rereview-3/examples/physics_qg/gravity_well/ (README.md, __main__.py, results/validation.json)"
  - "C:/src/POPGP-review-source-law-rereview-3/examples/physics_qg/source_law/ (README.md, __main__.py, results/validation.json)"
  - "C:/src/POPGP-review-source-law-rereview-3/examples/physics_qg/source_law_many_body/ (README.md, __main__.py, results/validation.json)"
  - "C:/src/POPGP-review-source-law-rereview-3/examples/physics_qg/ca_model/ (README.md, __main__.py, results/validation.json)"
  - "C:/src/POPGP-review-source-law-rereview-3/reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1.md"
  - "C:/src/POPGP-review-source-law-rereview-3/reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-1.md"
  - "C:/src/POPGP-review-source-law-rereview-3/reviews/codex/POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-1.md"
  - "C:/src/POPGP-review-source-law-rereview-3/reviews/codex/POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-2.md"
  - "git show 18aa5b3feea4f35e8bf89f1ae4cd34ce57fc279e:reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1.md"
  - "git show 1dd86b3ba0c4a4aaea3b3027f2263a547c8a7eb6:reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-2.md (absent from the candidate tree)"
  - "git show adbfab58c20dc28f4cf05b601f81480207d74415:popgp/ (pre-fix package, extracted to a scratch directory to reproduce the original SLAW-001 defect)"
access_level: local-repository-and-public-github-read-only
independence_statement: "This review is process separation with a different model. It is not external scientific independence, and on the fresh-session criterion it is weaker than the two prior rounds. Full disclosure, per REVIEWER_IDENTITY.md's requirement to declare shared prompts, context, operator, session, and conclusions. (1) MODEL: the reviewer model is claude-opus-5, a different model from the prior reviewer for this chain (gpt-5.6-terra) and from the builder (gpt-5.6-sol). Per REVIEWER_IDENTITY.md:17-18, changing the model occupying the seat does not by itself make a review independent. (2) OPERATOR: the human operator, Richard Fuoco, is the same operator as for the builder and for all prior reviews in this chain. (3) SESSION: this review was NOT run in a fresh session, contrary to REVIEWER_IDENTITY.md:38-39. It was orchestrated from a session that had already produced three earlier INFORMAL audits of ancestor commits - 590c68f/3b5812f, 47b8bae/aed5191, and a partial audit of a7a23ba/adbfab5 - and the conclusions of those audits were present in the orchestrator context before this review began. Those informal audits were not conducted under this governance process, are not part of the review record, and cannot be audited from the repository. This is substantial inherited context and it materially weakens the independence claim. (4) DIRECTED ATTENTION: the orchestrator relayed to every reviewing subagent, as prior context, the historical failure pattern that this repository's acceptance gates have repeatedly been insensitive to the physics they gate. That directed attention to a specific defect class rather than leaving it to be discovered independently, and any credit for probing gate sensitivity in this review must be discounted accordingly. (5) ORCHESTRATION: the reviewing subagents whose executed evidence this artifact consolidates were spawned by the same orchestrator and share its operator and access boundary. (6) BUILDER CONCLUSIONS: this is a re-review, so it necessarily received the prior review and the builder response; per REVIEWER_IDENTITY.md:47-48 the dispositions in those documents were re-derived rather than accepted, and every prior-finding result below states what was independently reproduced. (7) ACCESS: local read-only repository access plus unauthenticated public GitHub REST/HTML. Raw GitHub Actions job-log bodies returned HTTP 403 and were recorded as unavailable rather than inferred. No authenticated CLI, private log, final label, secret seed, private evaluator, credential, or private hardware profile was used."

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: "Changes requested. Two blockers: one prior blocking finding (SLAW-003) is downgraded to unresolved because a local-conservation claim survives at docs/scientific_hardening/FALSIFICATION_MATRIX.md:15 in a location the prior chain did not inspect (tracked as SLAW-004), and one new high-severity code defect (LIB-001) silently evolves the Pi_res leakage functional with the Heisenberg generator when substrate.hamiltonian is 'ising', inverting the contiguity result for that family. CI-001, SLAW-001 and SLAW-002 are verified resolved by direct re-derivation and counterexample replay, not by accepting the builder response; four of five prior requested tests are verified satisfied. The candidate diff from the approved b519c7da is documentation only (7 files, +622/-0), so neither blocker was introduced by this commit; both are pre-existing and neither was caught by three prior rounds. The central source-law gates were probed directly and are physics-sensitive: the quadratic-order gate rejects linear contamination down to c1=1e-8 against c2=0.0778, the Richardson gate fails a strictly quadratic response, and significance_ratio spans 7.4e7 to 5.7e9 across the twelve sweep configurations rather than sitting at a constant. The new governance layer is competent process documentation but is enforced entirely by human diligence and contains no rule requiring an acceptance gate to ship a demonstrated negative control."

findings:
  - id: "LIB-001"
    severity: high
    category: code
    location: "popgp/coarse_grain.py:88-121 (_build_local_hamiltonian) and :175-178 (compute_leakage), reached from popgp/simulator.py:365-382 (_pi_res_exact); contrast popgp/backend.py:166-191; over-broad claim at docs/scientific_hardening/THEORY_CODE_GAP.md:26"
    evidence: "_build_local_hamiltonian receives no Hamiltonian-family argument and unconditionally assembles coupling_J * (XX + YY + ZZ) for every intra-cell edge (popgp/coarse_grain.py:116-120), while ExactBackend.build_interaction_terms correctly emits ZZ-only terms for hamiltonian='ising' (popgp/backend.py:166-191). Simulator._pi_res_exact forwards only edges and coupling_J, so the family never reaches the leakage functional; the generator produced at coarse_grain.py:175-178 is used at :201 as the generator of the trace-then-evolve branch. Executed against the frozen tree: _build_local_hamiltonian([0,1], [(0,1),(1,2),(2,3)], 1.0) returns a matrix with Frobenius distance 0.7071067811865476 from J*ZZ and 0.0 from J*(XX+YY+ZZ), i.e. off-diagonal entries of 0.5 that an Ising in-cell generator cannot have. With SubstrateConfig(n_qubits=4, topology='chain', hamiltonian='ising', beta=1.0) and PiResConfig(cell_dim=2), optimize_cells as shipped ranks [[0,3],[1,2]] L_leak=1.494911e-03 < [[0,1],[2,3]] 1.779854e-03 < [[0,2],[1,3]] 1.798158e-03 and selects the non-contiguous partition; with a ZZ-only in-cell generator the ranking becomes [[0,1],[2,3]] 5.712651e-04 < [[0,3],[1,2]] 1.128399e-03 < [[0,2],[1,3]] 1.798158e-03 and the contiguous partition wins by 2.6x. The control partition [[0,2],[1,3]] has no intra-cell edge and scores 1.798158e-03 in both runs, which is exactly the invariance the generator swap predicts. Simulator.run() completes for this configuration with no error and no warning."
    finding: "For substrate.hamiltonian='ising' the Pi_res leakage functional evolves the trace-then-evolve branch with the Heisenberg generator XX+YY+ZZ instead of the substrate's actual ZZ interaction, so the commutator measures a channel mismatch against dynamics the system does not have and the selected cell decomposition E* is wrong. The unqualified statement at THEORY_CODE_GAP.md:26 that 'Hamiltonian family selection no longer silently ignores ising' is scoped in its authoring commit (590c68f) to ExactBackend.build_hamiltonian; a second, unguarded family-selection site remains in the leakage functional."
    failure_scenario: "A user runs Simulator(SimulatorConfig(substrate=SubstrateConfig(n_qubits=4, topology='chain', hamiltonian='ising', beta=1.0), pi_res=PiResConfig(cell_dim=2))).run(). No error or warning is raised. Pi_res returns the non-contiguous partition [[0,3],[1,2]] at L_leak=1.494911e-03, where the correct Ising generator selects the contiguous [[0,1],[2,3]] at 5.712651e-04. The qualitative headline result of Pi_res, that the leakage attractor is the contiguous locality-respecting decomposition, is inverted for one of the two supported interaction families."
    consequence: "The framework's 'locality emerges from the leakage functional' result is silently wrong for half of the supported Hamiltonian families, on a documented and reachable configuration, and every downstream stage (Pi_loc MI graph, Pi_geom embedding, Pi_time source) would inherit a cell net selected against the wrong generator. No committed example or validation artifact is currently affected, because chain_1d uses the default Heisenberg 8-qubit chain and source_law_many_body exercises 'ising' only through ExactBackend directly; the defect is a latent silent-wrong-physics path, not a corrupted published number. No existing test covers any Hamiltonian family other than the default."
    required_action: "Pass the Hamiltonian family into _build_local_hamiltonian, or preferably restrict the backend's own build_interaction_terms decomposition to intra-cell edges so the in-cell generator is by construction the restriction of the substrate Hamiltonian actually being evolved. Add a regression asserting, for both 'heisenberg' and 'ising', that the sum of intra-cell generators plus inter-cell terms reproduces backend.build_hamiltonian(), plus a Pi_res selection test on the 4-qubit Ising chain. Qualify the scope of the claim at THEORY_CODE_GAP.md:26."
    verification: confirmed-by-execution
    blocking: true

  - id: "GOV-003"
    severity: medium
    category: governance
    location: "docs/governance/AGENT_REVIEW_WORKFLOW.md:58; README.md:45-60; docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md:26-38; .github/workflows/ci.yml:16-34"
    evidence: "AGENT_REVIEW_WORKFLOW.md:58, the mandatory builder freeze step, reads 'Run the complete quality suite documented in the repository README.' The README's only quality-suite block is under '## Quick start' (README.md:45, block at :49-60) and contains uv sync / uv run pytest -q / uv run ruff check popgp tests examples / the six examples. Executed: grep -n 'check_validation_artifacts|check_tex' README.md returns nothing, so the README omits both repository gate scripts. LAUNCH_INDEPENDENT_REVIEW.md:26-38 lists a different, larger suite - uv sync --frozen, ruff check ., scripts/check_tex.py, pytest -q, the six examples, scripts/check_validation_artifacts.py - which matches the run steps of .github/workflows/ci.yml:16-34 exactly. The README suite is a strict subset that omits scripts/check_tex.py and scripts/check_validation_artifacts.py and weakens two commands (uv sync instead of uv sync --frozen; ruff check popgp tests examples instead of ruff check .). scripts/check_validation_artifacts.py is precisely the CI step named in prior blocking finding CI-001 and in requested tests TST-CI-001 and TST-CI-002."
    finding: "The single mandatory pre-freeze verification instruction in the governance document points builders at a command list that does not exist as described and that omits the two repository gate scripts, including the validation-artifact contract check whose failure was the only blocking CI finding in this review chain. Two mutually inconsistent quality suites are named in two documents introduced by the same commit, and the weaker, incomplete one is designated authoritative at the freeze step."
    failure_scenario: "A builder obeys AGENT_REVIEW_WORKFLOW.md:58 literally, runs the README Quick start block, sees it pass, and freezes the candidate. scripts/check_validation_artifacts.py never runs locally, so a validation.json contract violation or a missing/empty required visual is not detected before handoff, and a review round-trip is spent on a defect a local command would have caught. That is the CI-001 sequence: the initial review recorded a clean local Windows replay while GitHub Actions failed at the structured-artifact step for adbfab58c20dc28f4cf05b601f81480207d74415."
    consequence: "A documentation-consistency defect that costs review round-trips, not a bypassable gate. The same governance document cross-links the authoritative command list at AGENT_REVIEW_WORKFLOW.md:159-161, requires 'the complete quality suite and CI' again at step 4 (:105), and requires green CI before merge at :129-132, and .github/workflows/ci.yml runs on every push with check_validation_artifacts.py as its final step, so the artifact contract cannot actually be evaded as a merge gate. Blocking status is therefore false."
    required_action: "Name a single authoritative suite at AGENT_REVIEW_WORKFLOW.md:58 - either 'Run every command in .github/workflows/ci.yml, which is the authoritative quality suite; docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md section 1 reproduces it', or add the two missing script invocations and the --frozen / ruff-scope corrections to README.md. Add a regression asserting that the command list in LAUNCH_INDEPENDENT_REVIEW.md section 1 equals, as a set, the run steps of .github/workflows/ci.yml."
    verification: confirmed-by-execution
    blocking: false

  - id: "SLAW-004"
    severity: medium
    category: claim
    location: "docs/scientific_hardening/FALSIFICATION_MATRIX.md:15"
    evidence: "The 'Closure / conservation' row reads: 'Global microscopic energy and a declared local decomposition are conserved in one exact chain; covariant/discrete Bianchi closure is not implemented.' Executed: git blame -L 15,15 HEAD attributes it to cac51c33, and git show d527cb8bfe02b754c2f52724bce852ccfc28d584 --stat is 2 files, +8/-2, touching only line 7 of that table and tests/unit/test_claim_wording.py, so the SLAW-003 fix never reached line 15. Line 7 of the same table now says the opposite: 'the global Hamiltonian expectation is conserved while the local-energy profile spreads ... no discrete continuity current or local conservation law is implemented.' docs/framework.md:488-489 and docs/framework.tex:357 also disclaim any local conservation law. The repository's own gate requires the local decomposition NOT to be conserved: examples/physics_qg/source_law_many_body/__main__.py:583-585 passes only when evolved_outside_fraction > 0.05, and the committed results/validation.json records t1_endpoint_fraction = 0.12078184671097476 against initial_endpoint_fraction = 4.54123182112811e-16. No discrete current, flux, divergence or continuity operator exists anywhere in popgp/ or examples/. Guard coverage was tested by execution: tests/unit/test_claim_wording.py PASSES with this sentence present, and an injected mutation adding 'but the audited local-energy profile is conserved site by site' to the covered falsification-matrix text still satisfies all of its assertions. The textual defect itself is established by inspection; the git-history and guard-coverage halves are executed."
    finding: "SLAW-003's required action was to correct ALL affected wording. In the row a referee reads for conservation status, the falsification matrix still asserts on its natural reading that the declared local energy decomposition is conserved in the exact chain, contradicting line 7 of the same table three rows earlier. The wording regression added by the SLAW-003 fix is a three-literal-string denylist plus presence assertions scoped to row 7, so it does not cover this row."
    failure_scenario: "A referee reads FALSIFICATION_MATRIX.md line 15 and concludes that POPGP has demonstrated a conserved local energy decomposition, the property that would matter for a candidate source law or stress-energy density. Line 7 of the same table and docs/framework.md:489 state that no local conservation law is implemented and that the profile spreads. Both cannot be true; the code supports only the second, with 12 percent of the audited local energy having moved to the chain endpoints by t=1."
    consequence: "The document that defines this project's falsification criteria contains an internal contradiction in its conservation row, and re-review 2 closed SLAW-003 as verified-resolved on a repository-wide search whose regex could not match this phrasing. A charitable reading exists - 'the declared split remains an exact decomposition of a conserved total' is an implemented, tested property (tests/scientific/test_many_body_source_law.py:246) - which is why this is scoped as a residual wording defect rather than a false numerical result, and why it is non-blocking on its own merits. It counts toward the blocker total only because it leaves prior blocking finding SLAW-003 unresolved."
    required_action: "Restate FALSIFICATION_MATRIX.md:15 to say only what is implemented, for example 'the global microscopic energy is conserved under a declared local decomposition whose site profile spreads; no local conservation law, discrete continuity current, or covariant/Bianchi closure is implemented', and extend tests/unit/test_claim_wording.py to cover this row semantically rather than by three pinned phrases."
    verification: confirmed-by-execution
    blocking: false

  - id: "CI-005"
    severity: medium
    category: code
    location: "scripts/check_validation_artifacts.py:90-95 and :112-143; examples/physics_qg/ca_model/__main__.py:295-297; examples/physics_qg/chain_1d/__main__.py:412-414"
    evidence: "compare_validation_documents is a pure reference-versus-candidate diff and contains no semantic rule. Executed in memory against the real committed artifacts: compare_validation_documents(x, x) returns passed=True with zero errors for any document x, including one whose overall_pass contradicts its own checks[].passed values (checks[0].passed=False with overall_pass=True is accepted because both sides agree), and including ca_model with its failing population_growth check demoted to severity informational and overall_pass flipped to True. Executed: grep -rn 'overall_pass' tests/ returns exactly one hit, tests/unit/test_validation_artifact_contract.py:39, inside a synthetic fixture; no test reads any committed examples/physics_qg/*/results/validation.json. Two of six examples exclude informational checks from the headline verdict - ca_model/__main__.py:295-297 and chain_1d/__main__.py:412-414 both compute all(c['passed'] for c in report['checks'] if c.get('severity') != 'informational') - while grid_2d:359, gravity_well:501, source_law:244 and source_law_many_body:762 aggregate over all checks with no filter. The mechanism has already been used once: git show 47b8bae -- examples/physics_qg/ca_model/results/validation.json shows severity 'informational' being REMOVED from the failing population_growth check in the same hunk where overall_pass changes true to false. Present state audited and clean: for all six artifacts overall_pass equals both AND(non-informational passed) and AND(all passed), ca_model overall_pass is still false with population_growth failing, and the only two informational checks repo-wide (ca_model survivor_entropy_filter_regression, chain_1d blind_edge_recovery) both pass."
    finding: "No gate in the repository validates a validation artifact's internal consistency. The artifact contract only compares regenerated values against committed values, so any inconsistency produced by the code is reproduced identically and accepted, and no test asserts that overall_pass equals the conjunction of its constituent passed booleans or that no failing check carries severity informational. For ca_model and chain_1d the informational-demotion key that once concealed a failing ca_model check is a one-line, unguarded route to flipping the headline verdict."
    failure_scenario: "A later change adds severity 'informational' to ca_model's currently failing population_growth check. ca_model/__main__.py:295-297 recomputes overall_pass as True, the regenerated artifact matches the newly committed artifact, check_validation_artifacts.py reports success, pytest never inspects the artifact, and the repository's documented negative headline verdict flips from false to true with no gate objecting. This is the same edit that was reverted at 47b8bae, now invisible to CI. For grid_2d and gravity_well the same one-key edit is inert, because those examples aggregate over all checks."
    consequence: "The single most load-bearing boolean in each published artifact can be flipped by a one-line annotation with no automated objection, in a repository whose recorded history contains exactly that manipulation. The consequence is bounded: the contract forces the changed artifact to be recommitted, so both the severity key and the overall_pass flip appear in the pull-request diff, and detection falls to human diff review - which is what caught it at 47b8bae. No published claim is currently wrong."
    required_action: "Add a check to scripts/check_validation_artifacts.py, or a unit test over the committed artifacts, asserting for every examples/physics_qg/*/results/validation.json that (i) no check with passed false carries severity informational, and (ii) overall_pass equals the conjunction of the non-informational passed booleans with the informational set pinned to an explicit allow-list. Document the informational-demotion policy in docs/scientific_hardening/DECISIONS.md."
    verification: confirmed-by-execution
    blocking: false

  - id: "LIB-004"
    severity: medium
    category: code
    location: "popgp/simulator.py:766-803 (_canonicalize_embedding) and :805-818 (_canonicalize_embedding_signs), reached from :747-764 (_classical_mds, eigenvalue clamp at :762); contradicted claim at docs/scientific_hardening/REPRODUCIBILITY.md:100"
    evidence: "The docstring at popgp/simulator.py:768-774 claims the rule 'yields deterministic coordinates', and REPRODUCIBILITY.md:100 states that 'The canonical MDS frame removes arbitrary orientation changes from serialized tensors', with no carve-out. The main anchor branch genuinely delivers that: over 50 random proper and improper O(2) frames on the committed 3x3 grid embedding the maximum coordinate deviation is 1.776e-15, and the polar factor depends only on the anchor Gram matrix, so it is well defined under exact degeneracy. The fallback taken when represented_rank < dimension (:783) only flips column signs (:805-818), and sign flips are a strict subgroup of O(D), so it provably cannot remove a rotation inside a degenerate retained eigenspace. Executed by constructing a legitimate alternative eigenbasis inside the exactly degenerate leading eigenspace (residual 2.7e-15, still an exact eigenbasis) and passing both through Simulator._canonicalize_embedding: on the 3x3 grid hop metric, D=2/3/4 take the anchor branch and are invariant (1.8e-15, 2.2e-15, 2.8e-15) while D=5 and D=6 take the sign branch and give maximum coordinate difference 3.754 against a coordinate scale of 2.009, with pairwise distances agreeing to 8.9e-16. On the 5-cycle hop metric this fires at D=3, inside max_geometric_dimension=3 and with stress 0.1244 < max_geometric_stress=0.25 (a geometric_candidate), shifting coordinates by 1.147 against a scale of 1.079 and the serialized h_ab components by 0.194 while their eigenvalues agree to 1e-14 and stress changes by 2.8e-17."
    finding: "The rank-deficient fallback fixes only column signs and therefore does not remove the O(D) frame ambiguity that classical MDS leaves when the retained eigenvalues are degenerate, so the documented determinism invariant of _canonicalize_embedding does not hold whenever the selected embedding dimension includes an eigenvalue clamped to zero at :762 while the retained positive eigenvalues contain a degenerate multiplet."
    failure_scenario: "A configuration whose selected best_D exceeds the numerical rank of the MDS Gram while the retained positive eigenvalues are degenerate - the 5-cycle hop metric at D=3 is a concrete instance inside the geometric_candidate regime - returns pi_geom.coords in a frame fixed only by LAPACK's arbitrary choice inside the degenerate eigenspace. Two runs on different BLAS/LAPACK builds then emit O(1)-different coords and h_ab components while stress and the selection objective are identical to 1e-17, so nothing in the pipeline detects the divergence."
    consequence: "A silent cross-platform reproducibility gap in exactly the fields the committed validation JSONs record and that scripts/check_validation_artifacts.py compares at 1e-3 relative / 5e-9 absolute with coords absent from SENSITIVE_DIAGNOSTIC_TOLERANCES. Scope is bounded: simplices and deficit_angles cannot be affected, because they are computed only when best_D == 2 (popgp/simulator.py:672) and at D_eff=2 a clamped eigenvalue and a retained degenerate positive pair are mutually exclusive; the committed examples select D_star = 1, 1, 2, so no shipped artifact is currently wrong. The sole existing regression, tests/unit/test_simulator.py:130-150, uses a full-rank input and covers only the anchor branch."
    required_action: "Make _canonicalize_embedding_signs frame-fixing rather than sign-fixing: restrict the polar-anchor construction to the represented (numerically nonzero) column block and apply it there, or drop the clamped-to-zero columns before canonicalizing and re-pad. Add a regression that re-mixes degenerate eigenvectors and asserts coordinate equality for every candidate D in 1..D_max, not only the selected one."
    verification: confirmed-by-execution
    blocking: false

  - id: "SLAW-005"
    severity: low
    category: science
    location: "docs/scientific_hardening/FALSIFICATION_MATRIX.md:7; README.md:157; examples/physics_qg/source_law_many_body/__main__.py:432 and :583; tests/scientific/test_many_body_source_law.py:247"
    evidence: "FALSIFICATION_MATRIX.md:7 lists 'failure of global-energy conservation' as one of four failure thresholds for the modular-energy-localization hypothesis and README.md:157 reports 'exact finite-system energy conservation and profile spreading'. The gate is conservation_drift = float(np.ptp(evolved_total_energy)) < 1e-12 (__main__.py:432, :583) and max(total_energies) - min(total_energies) < 1e-13 (test_many_body_source_law.py:247). Both measure Tr[(rho(t)-sigma)H] where rho(t) = U rho U-dagger and U = exp(-iHt) is built from the same eigendecomposition of the same H at popgp/backend.py:271-279, so [U,H]=0 and the quantity is identically zero for any input state. Executed against the frozen tree: the real setup gives ptp(E)=0.0; replacing the local decomposition by a deliberately wrong split assigning all energy to site 0 still gives ptp(E)=0.0; three random density matrices unrelated to the experiment give 2.220e-16, 0.000e+00 and 2.220e-16, all far below the 1e-12 threshold; and the quantity becomes nonzero (7.63e-02) only when the measured observable is a different Hamiltonian (Ising) from the evolution generator."
    finding: "The global-energy-conservation leg of the source-law acceptance gate is an algebraic identity of unitary evolution generated by the observable being measured, not a physical result. It is insensitive to the local energy decomposition, to the excitation, to the source model, to source_scale and to epsilon. No wrong physics makes it fail; it can only fail if ExactBackend.evolve stops using the Hamiltonian it exposes."
    failure_scenario: "A wrong microscopic energy split (all pair energy assigned to one endpoint instead of split symmetrically), a wrong source model, or an arbitrary unrelated density matrix all yield conservation_drift = 0 and clear the 1e-12 threshold, so the declared criterion 'failure of global-energy conservation' cannot be triggered by a defective source law."
    consequence: "One declared falsification threshold and one README result sentence carry no falsification power, which inflates the apparent evidential content of the finite-chain quench. Nothing asserted is false - 'exact finite-system energy conservation' is literally exactly true, merely uninformative - the underlying limitation is already disclosed three times (the matrix readiness cell, docs/framework.md:486-490 with its explicit no-continuity-current disclaimer pinned by tests/unit/test_claim_wording.py:20-27, and examples/physics_qg/source_law_many_body/README.md:17-18), and the other legs of the same gate (initial_endpoint_fraction < 1e-12, t=1 fraction > 0.05, and the commuting-Ising control) are genuinely discriminating. Hence low severity and non-blocking."
    required_action: "Relabel this quantity in FALSIFICATION_MATRIX.md:7, README.md:157 and the example criterion string as an implementation/numerical-consistency check verifying that the evolution generator equals the measured Hamiltonian, matching the convention the repository already uses for its analytic-identity regressions, or replace it with a genuinely falsifiable quantity such as a discrete continuity residual whose failure would discriminate candidate decompositions. docs/framework.md already carries the honest global-only wording and needs no change."
    verification: confirmed-by-execution
    blocking: false

  - id: "RTV-004"
    severity: low
    category: code
    location: "popgp/backend.py:194 and :205-206; tests/unit/test_simulator.py:261-279; tests/scientific/test_many_body_source_law.py:340-352; tests/unit/test_backend.py:38"
    evidence: "Executed in a disposable sandbox copy: replacing local_terms[i] += 0.5 * interaction; local_terms[j] += 0.5 * interaction (popgp/backend.py:205-206) with a 0.7/0.3 split leaves the entire suite green (99 passed); 0.75/0.25 and 0.8/0.2 also pass. The two KMS-density tests build their expected values from the same build_local_energy_operators() call, so their aggregation assertions are tautological with respect to the split; their sum assertions equal -source_scale * modular_energy_delta for any per-edge weights summing to one (verified: a 0.9/0.1 split moved the per-cell source from [-3.4953e-03, -1.1457e-03] to [-4.4118e-03, -2.2914e-04], a 26 percent change, while the identity residual stayed at 8.67e-19); and tests/unit/test_backend.py:38 only asserts sum(local_terms) == build_hamiltonian(), the same weight-invariant. tests/unit/test_validation_artifact_contract.py compares synthetic in-test documents and never recomputes from the backend. The divergence is caught only downstream of pytest: after the 0.7/0.3 mutation, regenerating source_law_many_body made scripts/check_validation_artifacts.py exit 1 with drifts such as exact_kubo_mori_quadratic_coefficient 0.07784 -> 0.09465 (rel 0.178)."
    finding: "The declared microscopic convention 'split each pair term equally between endpoints' (popgp/backend.py:194, published as config.local_energy_convention in every validation.json) is not pinned by any test, so the code and the published convention string can diverge silently through pytest. The per-cell values of the KMS energy density are only loosely constrained: the localization assertion at tests/scientific/test_many_body_source_law.py:277 does fail at 0.9/0.1 and 1.0/0.0, but deviations up to roughly 0.8/0.2 pass silently."
    failure_scenario: "A refactor changes the endpoint split, or introduces an asymmetry, in build_local_energy_operators. Every unit and scientific test still passes at 0.7/0.3, 0.75/0.25 and 0.8/0.2. The mismatch between code and the convention string published in each validation.json is caught only indirectly, by numeric drift in the committed artifact contract during CI example regeneration."
    consequence: "The published local-energy convention and the code that implements it can diverge without any test failing, and the per-site energy density that the localization diagnostics report is constrained only very loosely by independent ground truth."
    required_action: "Add a unit test that pins the endpoint split directly - for an open two-site chain assert build_local_energy_operators()[0] == 0.5 * build_interaction_terms()[0][1] - so the code and the documented convention cannot diverge silently."
    verification: confirmed-by-execution
    blocking: false

  - id: "STAT-002"
    severity: low
    category: statistics
    location: "popgp/diagnostics.py:139-148"
    evidence: "fit_quadratic_asymptote forms scaled = y / x**2, fits [1, x] by unweighted np.linalg.lstsq, and reports covariance = residual_variance * inv(design.T @ design), the homoscedastic OLS form. The module's own error model (absolute_precision_floor) is additive roundoff in y, so the transformed response is heteroscedastic by (x_max/x_min)**2 = 1e4 across the declared window and the intercept's largest effective leverage falls on the noisiest, smallest-amplitude points. Verified at the candidate by exact-rational least squares on the committed epsilons and relative_entropy in examples/physics_qg/source_law_many_body/results/validation.json (no example run): OLS gives c0 = 0.07784175639247233, bit-identical to the committed coefficient, relative error 1.2541583585835188e-05 against exact_kubo_mori_quadratic_coefficient 0.07784273266361058, while inverse-variance WLS with weights proportional to x**4 on the identical nine points gives 0.077842729696115, relative error 3.812e-08 - a factor of 329. Subset fits reproduce the same picture (drop smallest 7.0723e-06, drop two smallest 1.3213e-06, four largest 2.6921e-07). Per-point absolute deviations of y from the fitted model are all within 2.3x the declared floor 6.175292551076229e-16, confirming roundoff rather than truncation dominates, and WLS beats OLS in all 13 committed configurations by factors of 8x to 330x."
    finding: "The quadratic-asymptote estimator is an unweighted OLS applied to a strongly heteroscedastic transformed response, so it discards one to two and a half orders of magnitude of achievable accuracy relative to the module's own noise model, and the choice is undisclosed anywhere in the code comments or documentation."
    failure_scenario: "n=5 Heisenberg beta=1.3, eps=np.logspace(-5,-3,9): the committed fit returns c0 = 0.07784175639247233 (1.2542e-05 relative error) while an inverse-variance weighted fit on exactly the same nine measurements returns 0.077842729696115 (3.812e-08). The lost accuracy shrinks the margin against the declared 5e-4 exact-Kubo-Mori gate: the worst committed case, n=5 Ising beta=0.3, sits at 1.285e-04, only about 3.9x inside the gate, against a 2e-4 absolute artifact-drift tolerance for that same field at scripts/check_validation_artifacts.py:41."
    consequence: "No published value is wrong and every committed configuration passes the 5e-4 exact-Kubo-Mori gate, so this is a CI-robustness and estimator-quality concern, not a correctness defect. Two parts of a broader version of this concern do not hold and are excluded: coefficient_residual_scale is already documented as descriptive rather than a sampling estimate (popgp/diagnostics.py:111-112) and empirically brackets the true intercept error within 0.43x to 3.2x in every committed case, and a regression comparing the fitted coefficient to the exact Kubo-Mori value already exists at tests/scientific/test_many_body_source_law.py:135-137."
    required_action: "Weight the fit by the module's own noise model (sigma_i proportional to absolute_precision_floor / x_i**2, equivalently fit y = c0 x**2 + c1 x**3 directly with constant-variance weights) and report a covariance consistent with that weighting; or, if the unweighted form is retained deliberately, document that the intercept is dominated by the smallest-amplitude points and state the resulting accuracy explicitly."
    verification: confirmed-by-execution
    blocking: false

  - id: "STAT-003"
    severity: low
    category: statistics
    location: "popgp/diagnostics.py:12, :16 and :86-99; examples/physics_qg/source_law/__main__.py:102 and :149; examples/physics_qg/source_law_many_body/__main__.py:39 and :445-446; scripts/check_validation_artifacts.py:51"
    evidence: "PowerLawFit is documented at popgp/diagnostics.py:12 as a 'Log-log power-law fit with elementary uncertainty diagnostics' and exposes slope_standard_error at :16, computed at :86-92 as the textbook homoscedastic-OLS sampling standard error sqrt((RSS/(n-2))/Sxx). The inputs are deterministic, noise-free responses from exact diagonalization on a fixed amplitude grid, so there is no sampling and RSS measures model misspecification. Executed: for a strictly power-law input y = 3.7*eps**2 on np.logspace(-5,-2,10) the field returns 3.5180234707272447e-16; for y = x**2 + 30*x**1.5 it returns 6.76e-05 alongside slope 1.500417, a tight 'error bar' attached to an exponent wrong by 0.5, so the quantity bounds nothing about the true exponent. It is nonetheless printed as 'slope=... +/- ...', stored under the sampling name in every committed validation.json, given a drift policy at scripts/check_validation_artifacts.py:51, and used as a live acceptance assertion at tests/scientific/test_source_law_scaling.py:93. The repository already corrected the identical mislabel for the sibling statistic: git show adbfab5 removes the 5.0 * slope_standard_error gates and renames coefficient_standard_error to coefficient_residual_scale with the docstring 'descriptive, not sampling estimates' (now popgp/diagnostics.py:25 and :111-112), while leaving PowerLawFit untouched."
    finding: "PowerLawFit.slope_standard_error applies a sampling-theory name, docstring and +/- presentation to a deterministic residual quantity, inconsistently with the repository's own corrected treatment of the same statistic in the quadratic fit."
    failure_scenario: "A reader takes examples/physics_qg/source_law/results/validation.json modular_energy slope = 0.9999999999995276 +/- 1.8074841977714659e-13 as a statistically significant determination of a unit exponent. It is not: the underlying relation is an exact algebraic identity in an affine mixture family and 1.8e-13 is the roundoff level of that identity on the chosen grid."
    consequence: "A published field carries a sampling-uncertainty name and a +/- presentation with no sampling content. The scientific content is already guarded elsewhere - examples/physics_qg/source_law/README.md:24-26 states that the modular and solver linearities are analytic identity regressions and not evidence for a physical source, and the check name and conclusion string repeat it - so what remains is a labeling inconsistency. The acceptance assertion at tests/scientific/test_source_law_scaling.py:93 is also not vacuous: inputs on its own grid do fail it (y = x**2 + 1e4*x**3 gives 0.0542, y = x**2*(1+0.5*sin(20*log x)) gives 0.0613, both above the 0.01 threshold), so it is a working log-log residual bound that is merely misnamed."
    required_action: "Rename the field to a residual-scale name consistent with coefficient_residual_scale, drop 'uncertainty' from the popgp/diagnostics.py:12 docstring and the +/- from the two example print sites, update the JSON key and scripts/check_validation_artifacts.py:51 accordingly, and state why a residual bound of 0.01 is the intended criterion at tests/scientific/test_source_law_scaling.py:93."
    verification: confirmed-by-execution
    blocking: false

  - id: "STAT-004"
    severity: low
    category: code
    location: "popgp/diagnostics.py:171-186 (assess_quadratic_response, no amplitude-ordering validation); contrast popgp/diagnostics.py:223-224"
    evidence: "richardson_first_order_limit validates its precondition explicitly at popgp/diagnostics.py:223-224 ('amplitudes must be positive and strictly increasing') and raises on a permuted input. assess_quadratic_response selects its lower window positionally as x[:lower_window_size] (:181-186) and never checks ordering. Executed against the frozen module: with y = c1*eps + c2*eps**2, eps = np.logspace(-5,-3,9), c2 = 0.07784273266361058 and lower_window_size=6, the largest first-order contaminant still accepted rises from c1 = 3.0869e-09 for ascending amplitudes to c1 = 6.1758e-09 for the interleaved order [0,3,6,1,4,7,2,5,8]; at c1 = 1e-7 the drift statistic falls from 3.0898e-02 to 1.5468e-02 on identical data, with no error or warning. The nested-window gate is the binding one on this data - slope_deviation is 7.2e-04 against a 0.02 limit and maximum normalized RMSE is 1.05e-03 against a 1e-02 limit at the ascending threshold - so nothing else takes up the slack."
    finding: "The nested-window agreement gate silently loses half its discriminating power when amplitudes are not sorted ascending, and assess_quadratic_response does not validate that precondition although the sibling Richardson estimator in the same module does."
    failure_scenario: "A caller passes the same nine (amplitude, response) pairs in the interleaved order [0,3,6,1,4,7,2,5,8]. assess_quadratic_response reports relative_coefficient_difference 1.5468e-02 instead of 3.0898e-02 for identical data, and the largest first-order contaminant it accepts doubles from 3.0869e-09 to 6.1758e-09. No error or warning is raised."
    consequence: "The documented nested-window agreement criterion is silently reinterpreted as a comparison of two overlapping full-range windows. Both orderings still reject at c1 = 1e-08, so this is a 2x sensitivity loss rather than a dead gate, and no in-repository caller violates the precondition (all pass np.logspace), making this a latent API-contract gap rather than a defect in the committed artifacts."
    required_action: "Add the strictly-increasing-amplitude validation used at popgp/diagnostics.py:223-224 to assess_quadratic_response (or sort internally and document it) and add a unit test that a permuted input is rejected. Do not impose the same restriction on fit_quadratic_asymptote, which is exactly permutation-invariant (verified: identical outputs to 1e-15 under the same permutation)."
    verification: confirmed-by-execution
    blocking: false

requested_tests:
  - id: "TST-LIB-001"
    description: "Add a Pi_res regression that (a) asserts, for every supported Hamiltonian family, that the sum over cells of the in-cell generator used by compute_leakage plus the inter-cell terms reproduces backend.build_hamiltonian() to machine precision, and (b) asserts the selected cell decomposition on the 4-qubit Ising chain with cell_dim=2. The test must fail on the current code, which returns [[0,3],[1,2]] at L_leak=1.494911e-03 instead of [[0,1],[2,3]] at L_leak=5.712651e-04."
    rationale: "LIB-001. The leakage functional evolves the trace-then-evolve branch with a generator the substrate does not have whenever hamiltonian='ising', inverting the contiguity result Pi_res is cited for, and no existing test covers any Hamiltonian family other than the default."
    blocking: true

  - id: "TST-SLAW-004"
    description: "Correct docs/scientific_hardening/FALSIFICATION_MATRIX.md:15 and extend tests/unit/test_claim_wording.py so that a sentence associating 'local', 'site', 'profile' or 'decomposition' with 'conserved' is rejected unless the same sentence also carries the global-Hamiltonian qualifier or the explicit no-local-law disclaimer. The regression must fail on the current line 15 text before the fix and pass after it."
    rationale: "SLAW-004. The existing wording regression is a three-literal-string denylist plus row-7 presence assertions: it passes today with an unqualified local-conservation claim present at line 15 of a file it reads, and it passes an injected 'the audited local-energy profile is conserved site by site' mutation."
    blocking: false

  - id: "TST-GOV-003"
    description: "Fix the divergence identified in GOV-003, then add a regression asserting that the command list in docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md section 1 is equal, as a set, to the run steps in .github/workflows/ci.yml, and that no other governance document designates a different suite as authoritative."
    rationale: "GOV-003. AGENT_REVIEW_WORKFLOW.md:58 makes the README the authoritative pre-freeze suite, but the README omits scripts/check_tex.py and scripts/check_validation_artifacts.py, the latter being the gate whose failure was blocking finding CI-001. A string-set equality test is the minimal guard against the two lists diverging again."
    blocking: false

  - id: "TST-CI-005"
    description: "Add a test over every committed examples/physics_qg/*/results/validation.json asserting (i) no check with passed false carries severity informational, and (ii) overall_pass equals the conjunction of the non-informational passed booleans, with the set of informational checks pinned to an explicit allow-list."
    rationale: "CI-005. No gate validates artifact self-consistency: a self-contradictory document is accepted with zero errors because the contract compares the artifact only against itself, and the informational-demotion key that once concealed a failing ca_model check at 47b8bae is unguarded."
    blocking: false

  - id: "TST-LIB-004"
    description: "Add a determinism regression that constructs a legitimate alternative eigenbasis inside each exactly degenerate MDS eigenspace and asserts that Simulator._classical_mds returns identical coordinates for every candidate D in 1..D_max, not only the selected one. On the 3x3 grid hop metric this currently passes for D=2, 3 and 4 (1.8e-15 to 2.8e-15) and fails for D=5 and D=6 (3.754 against a coordinate scale of 2.009); on the 5-cycle hop metric it fails at D=3 (1.147 against a scale of 1.079)."
    rationale: "LIB-004. The rank-deficient fallback fixes only column signs and cannot remove an O(D) mixing, so the documented determinism invariant silently does not hold whenever the selected dimension includes a clamped non-positive MDS eigenvalue while the retained positive eigenvalues are degenerate. The sole existing regression covers only the full-rank anchor branch."
    blocking: false

  - id: "TST-SLAW-005"
    description: "Add a mutation test demonstrating what the global-energy-conservation criterion can and cannot detect: assert that conservation_drift stays below 1e-12 for a deliberately wrong local energy decomposition and for an arbitrary unrelated density matrix, and that it becomes nonzero only when the measured observable differs from the evolution generator. Then relabel the criterion in FALSIFICATION_MATRIX.md:7, README.md:157 and the example criterion string as an implementation-consistency check rather than a falsification threshold."
    rationale: "SLAW-005. Executed evidence: ptp(Tr[(rho(t)-sigma)H]) is 0.0 for the real setup, 0.0 for a wrong decomposition and at most 2.22e-16 for random states, but 7.63e-02 when the observable is a different Hamiltonian. Naming the insensitivity in a test prevents the criterion from being re-presented as physical evidence."
    blocking: false

  - id: "TST-RTV-004"
    description: "Add a unit test pinning the declared endpoint split in ExactBackend.build_local_energy_operators against build_interaction_terms, for example h_i == 0.5 * interaction for a two-site open chain."
    rationale: "RTV-004. A 0.7/0.3, 0.75/0.25 or 0.8/0.2 split leaves all 99 tests passing, so the published config.local_energy_convention string and the code that implements it can diverge with no pytest failure."
    blocking: false

  - id: "TST-STAT-002"
    description: "Add a test that fits the committed headline data (n=5 Heisenberg beta=1.3, eps=np.logspace(-5,-3,9)) with the module's estimator and with an inverse-variance weighting sigma_i = absolute_precision_floor / eps_i**2, and asserts the weighted coefficient's relative error against the exact Kubo-Mori value is below the unweighted one by the demonstrated margin (unweighted 1.2542e-05 versus weighted 3.812e-08). Then either adopt the weighted estimator or record the accuracy cost of the unweighted one."
    rationale: "STAT-002. The transformed response y/eps**2 has noise variance varying by 1e4 across the declared window, the unweighted OLS is dominated by its noisiest points, and the worst committed configuration sits only 3.9x inside the 5e-4 acceptance gate."
    blocking: false

  - id: "TST-STAT-003"
    description: "Rename PowerLawFit.slope_standard_error to a residual-scale name matching coefficient_residual_scale, update the two example printers, the JSON keys and scripts/check_validation_artifacts.py:51, and add a claim-wording style regression asserting that no committed artifact or example output presents a power-law slope with a +/- or a sampling standard-error label."
    rationale: "STAT-003. On deterministic data the OLS residual standard error has no sampling interpretation, the repository already corrected the identical mislabel for the quadratic fit at adbfab5, and the affine-mixture artifact currently publishes slope 0.9999999999995276 +/- 1.8074841977714659e-13 for a relation it elsewhere calls an exact algebraic identity."
    blocking: false

  - id: "TST-STAT-004"
    description: "Add strictly-increasing-amplitude validation to assess_quadratic_response mirroring popgp/diagnostics.py:223-224, and a unit test that a permuted amplitude array is rejected. Leave fit_quadratic_asymptote unrestricted."
    rationale: "STAT-004. The nested-window gate's semantics depend entirely on ascending order; with the interleaved order [0,3,6,1,4,7,2,5,8] the same nine points halve the gate's detection power (largest accepted first-order contaminant 3.0869e-09 to 6.1758e-09) with no error raised."
    blocking: false

  - id: "TST-LIB-005"
    description: "Add a dtype-hygiene regression asserting that public library outputs are bitwise unchanged under torch.set_default_dtype(torch.float32) versus torch.float64 for float inputs. Simulator.gravitational_redshift currently fails it: popgp/simulator.py:954 calls torch.as_tensor on Python-float arguments, which adopts the ambient global dtype, so gravitational_redshift(-0.009963709390575564, 0.0022578960410844567) returns 0.012296594435271313 under a float64 default and 0.012296557426452637 under a float32 default (3.70e-08 absolute, 3.01e-06 relative)."
    rationale: "Every other numerically significant allocation in popgp pins its dtype explicitly; this is the single leak, and it sits in the function that produces the redshift number quoted in the gravity_well artifact and in the README status table."
    blocking: false

  - id: "TST-GOV-004"
    description: "For every acceptance gate named in docs/scientific_hardening/FALSIFICATION_MATRIX.md, commit a mutation test that demonstrably makes that gate fail, record the gated statistic before and after the mutation, and add a meta-test asserting that every gate row names an existing mutation-test id. For every numerical gate, additionally assert that the gated statistic's minimum and maximum across the declared parameter sweep differ by more than the gate's own tolerance."
    rationale: "This is the operational form of the guidance recommendation below. Nothing in docs/governance requires a gate to ship a demonstrated negative control, and this review found one gate leg that cannot fail for any physics input (SLAW-005) and one artifact-level gate that cannot fail at all (CI-005). The repository already applies the underlying discipline ad hoc - a first-order negative control, a precision floor and a commuting-Ising control in the source-law example - so this formalizes existing practice rather than inventing a new burden. Non-blocking: it is an additive enhancement to a documentation-only candidate."
    blocking: false

  - id: "TST-GOV-005"
    description: "Add to docs/governance/REVIEWER_IDENTITY.md's required-fields table and to docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md a typed block independence_declaration with keys shared_operator, shared_session, shared_orchestrator, builder_model_identity, reviewer_model_differs_from_builder and external_scientific_validation, retaining the prose field alongside it; and add the sentence 'external_scientific_validation is false for every review conducted under this workflow. Only a replication by an unaffiliated group with independent code can set it true; no agent review may set it true.'"
    rationale: "hidden_access_declaration is three typed booleans a script can check, while independence is a single free-text string that a vacuous sentence would satisfy. The review artifact also records no builder model identity, so the guidance's own preference for a different model cannot be verified from a review artifact in isolation. This round is a case in point: the shared-session disclosure in this artifact is a matter of reviewer conscience and nothing in the schema would have detected its omission."
    blocking: false

  - id: "TST-CI-003"
    description: "Make the required-visual leg of scripts/check_validation_artifacts.py regeneration-sensitive: capture each declared visual's digest before the 'Regenerate documented examples' CI step and require it to be rewritten, or regenerate into an empty scratch directory; and add a git-cleanliness assertion after regeneration. Alternatively state in the script and in REPRODUCIBILITY.md that visual regeneration is not verified on the runner."
    rationale: "Not raised as a finding: the tracked/present/nonempty policy is exactly what TST-CI-002 literally requested ('required nonempty files on the Linux runner') and it is accurately described in scripts/check_validation_artifacts.py:13 and docs/scientific_hardening/REPRODUCIBILITY.md:61-62. This is a hardening request grounded in executed evidence from this round's quality suite: changing one plot title in examples/physics_qg/source_law_many_body/__main__.py and regenerating changed many_body_source.png from 241734 to 242910 bytes while ruff, pytest (99 passed) and check_validation_artifacts.py all still exited 0. Relatedly, .github/workflows/ci.yml contains no git diff --exit-code step after regeneration, so byte-level artifact reproducibility is an observed property rather than an enforced CI invariant."
    blocking: false

prior_finding_results:
  - finding_id: "CI-001"
    outcome: verified-resolved
    evidence: "Re-derived at the candidate rather than accepting REREVIEW-2. (1) Scope: git diff --stat b519c7dafd4488d587a1edcbe14891f0c83aedb5 763fd1857d0a1298aa9858bc7b7c698c38833ef7 returns 7 files, +622/-0 (README.md +6 and six new docs/ files); the same diff filtered to popgp, tests, examples, scripts and .github returns nothing, so no CI configuration, generation code or committed artifact changed since the approved commit. (2) The byte-exact gate is gone: .github/workflows/ci.yml contains no git diff --exit-code and its final step (:33-34) runs uv run python scripts/check_validation_artifacts.py. (3) The replacement is a real gate, not a no-op: it compares each regenerated validation.json against git show HEAD:<path> with key sets, JSON types, list lengths, strings, ints, bools, nulls, check identities, criterion strings and all passed booleans exact, configuration floats at 8 machine epsilons, non-finite values forbidden, strict JSON rejecting NaN/Infinity, and a named allowlist of seven sensitive diagnostics; raise SystemExit(main()) at :342 with return 1 at :335. Executed on copies in a scratch repository: exit 0 clean, exit 1 with '$.checks[0].passed: changed from True to False' on a gate flip, exit 1 on a missing declared visual. Mutation of the committed source_law_many_body document confirmed the contract rejects a flipped check boolean, a flipped overall_pass, a 1 percent change to an exact Kubo-Mori value and a doubled local-energy profile array, and reproduced the factor-100 significance_ratio boundary exactly (x99 and x100 accepted, x100.5, x101 and sign reversal rejected). (4) The two named cross-platform causes are fixed at source and one has a live regression: removing the canonical sort at popgp/geometry/regge.py:22-25 makes tests/unit/test_regge_proxy.py::test_delaunay_proxy_has_canonical_simplex_order fail; the chain_1d scale-aware selection_driver tolerance change is scientifically correct rather than cosmetic, since stress_by_dimension is {1: 4.860e-16, 2: 6.640e-10, 3: 5.179e-16} against a selection_margin of 1.3219e-2, so the dimension choice really is driven by the spectral penalty. (5) Exact-SHA public evidence at the candidate: unauthenticated curl https://api.github.com/repos/whact2025/POPGP/actions/runs?head_sha=763fd1857d0a1298aa9858bc7b7c698c38833ef7 returns total_count 1, run 31380394043, branch codex/source-law-linear-response-remediation, event push, conclusion success; the jobs endpoint returns job 93429053764 with every step success, including step 8 Test, step 9 Regenerate documented examples and step 10 Check committed validation contracts and visual outputs - the exact step that failed at adbfab58c20dc28f4cf05b601f81480207d74415. (6) Independent local replay in a separate disposable worktree detached at the same candidate: uv sync --frozen, ruff check ., check_tex.py, pytest -q (99 passed), all six examples and check_validation_artifacts.py all exited 0, and all 18 tracked artifacts under examples/physics_qg/*/results/ regenerated bit-identically (git hash-object matched git ls-files -s for every one)."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Raw GitHub job-log bodies return HTTP 403 under the public-only boundary and were not fetched; the remote evidence is run/job/step conclusion metadata, and I say so rather than calling it my own execution. Residual policy limitations, unchanged and correctly disclosed: the comparator is semantic rather than byte-exact and its widest allowance is a factor-100 relative tolerance on significance_ratio, though every scientific gate boolean, check identity, criterion string, configuration float and array length remains exact. Two related observations are recorded elsewhere rather than reopening this finding: GOV-003 (the governance freeze step points builders at a README list omitting this very script) and TST-CI-003 (the visual leg validates repository state and CI has no post-regeneration cleanliness gate)."

  - finding_id: "SLAW-001"
    outcome: verified-resolved
    evidence: "Re-derived by counterexample replay, not accepted from the builder or the prior reviewer. The gate is popgp/simulator.py:1001-1004 calling _validate_kms_reference_state (:1035-1073), which rebuilds finite_gibbs_state(backend.build_hamiltonian(), beta_kms) independently of the supplied reference and compares nuclear-norm trace distance against KMS_REFERENCE_TRACE_DISTANCE_TOLERANCE = 1e-10 (:46); :990-995 additionally requires the supplied cells to be a nonempty disjoint partition of all sites, which is what makes the sum identity well posed. Executed against the frozen tree with popgp resolved into the review worktree: (a) the initial review's exact counterexample, SimulatorConfig.for_chain(n=4, beta=1.3) with beta_kms=0.5 and the beta=1.3 Gibbs reference, now raises ValueError 'reference_state must be the Gibbs/KMS state of the backend Hamiltonian at beta_kms=0.5; trace distance 2.741193e-01 exceeds KMS_REFERENCE_TRACE_DISTANCE_TOLERANCE=1.0e-10'; (b) a faithful non-Gibbs diag(0.4,0.3,0.2,0.1) reference is rejected at trace distance 2.627858e-01, and the maximally mixed I/32 reference on the n=5 chain at 4.679959e-01; (c) an incomplete cell list is rejected with 'cells must form a nonempty disjoint partition of all microscopic sites'; (d) a matched reference is accepted and the documented identity holds including source_scale, sum(source) = -0.004640973644968117 against -1.7 * Delta<K_reference> = -0.004640973644968115, difference 1.73e-18. The premise is enforced at the correct boundary - configuration construction cannot validate it because the reference arrives later - which is what the required action demanded. The gate is sharp rather than decorative: a 1e-11 trace-distance perturbation of the Gibbs state is accepted and 1e-10 is rejected, and five independent source mutations (early return before the trace-distance test, widening the tolerance to 1.0, flipping the sign of beta_kms, dropping source_scale, doubling the beta_kms fallback) each break exactly the tests that certify it. The identity is not vacuously circular: it also constrains the decomposition, and max|sum_i h_i - H| = 0.0 exactly. For contrast the pre-fix package was extracted from adbfab58c20dc28f4cf05b601f81480207d74415 into a scratch directory and the original defect reproduced exactly - the same beta_kms=0.5 call was accepted and returned sum(source)/(-Delta<K>) = 0.38461538461503 = 0.5/1.3, confirming the initial review's mechanism claim. Regressions exist by name and pass: tests/unit/test_simulator.py:200, :243, :287, :303, :347."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Residual limitation, unchanged and correctly disclosed by the builder: the dense trace-norm check forms a full Gibbs matrix and a nuclear norm, so it is appropriate only for the small exact backend and deliberately excludes alternate KMS representations. Per-cell densities remain convention-dependent and are only loosely pinned by the suite, which is reported separately as RTV-004."

  - finding_id: "SLAW-002"
    outcome: verified-resolved
    evidence: "popgp/simulator.py:1077-1104 declares run(self, evolve_steps: int | None = None, *, reference_state: object | None = None) and forwards the reference into run_pi_time at :1103, which forwards it into _compute_source_term at :859; the docstring at :1086-1088 names this as the route for reference-dependent candidates. Executed against the frozen tree: inspect.signature(Simulator.run) returns the keyword-only parameter; Simulator(SimulatorConfig.for_chain(n=4, beta=0.7)).run(reference_state=reference) with source_model='negative_kms_energy_density_candidate' completed and returned pi_time.source_model 'negative_kms_energy_density_candidate' with delta_rho_raw.sum() = -0.0027299844970400684 against the expected -modular_energy_delta(state, reference) = -0.0027299844970400675, difference 8.67e-19; with source_scale=1.7 the same route gives ||delta_rho_raw|| = 0.003678260069630865 and sum = -0.004640973644968117. Omitting the reference on the same public path still raises ValueError \"Source model 'negative_kms_energy_density_candidate' requires a reference_state\", so the mode fails loudly rather than degrading silently. A nonzero source is reachable through the unpatched public path without overriding prepare(): with beta=0.7 and beta_kms=1.1 the route returns delta_rho_raw = [-0.1393140712394197, -0.13931407123941975]. Sensitivity confirmed by mutation in a disposable copy: deleting the reference_state argument from the run() to run_pi_time() call makes tests/unit/test_simulator.py::test_run_propagates_reference_state_to_kms_candidate fail with the original 'requires a reference_state' ValueError (1 failed, 98 passed). No popgp/ or tests/ path changed between the approved b519c7dafd4488d587a1edcbe14891f0c83aedb5 and the candidate."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "One subagent proposed superseding this finding on the ground that no configuration of the public run path can produce a nonzero clock potential. That proposition was tested and refuted: the zero result it reported is an artifact of the default cell_dim=2 on an open 4-chain, whose mirror symmetry makes the two cell sources equal so the zero-mode removal annihilates them. With cell_dim=1 - the partition the repository's own KMS experiment uses at examples/physics_qg/source_law_many_body/__main__.py:390 - the public route gives delta_rho_raw = [-1.046253e-01, -1.901076e-01, -1.901076e-01, -1.046253e-01] and ||phi|| = 8.096e-01, and the same holds at n=3, 5, 6, 7 and for the Ising family. The finding is therefore resolved rather than superseded. The remaining ergonomic limitation is real and non-blocking: run() constructs its own state, so a localized excitation through the fully automatic path still requires the stage-wise route Simulator.run_pi_time(state, ...), which is public and documented at popgp/config.py:275-277."

  - finding_id: "SLAW-003"
    outcome: unresolved
    evidence: "The named locations from the original finding are genuinely fixed and I verified each by reading the current text: docs/framework.md:487-490 now reads 'the global Hamiltonian expectation is conserved while the audited site-energy profile changes and spreads. No discrete continuity current or local conservation law is established by this check'; docs/framework.tex:357 carries the matching sentence; docs/scientific_hardening/FALSIFICATION_MATRIX.md:7 was corrected by d527cb8bfe02b754c2f52724bce852ccfc28d584 and now separates the modular-density sum, global-energy conservation and profile spreading as distinct failure thresholds and ends 'no discrete continuity current or local conservation law is implemented'; README.md:157, REPRODUCIBILITY.md:78 and examples/physics_qg/source_law_many_body/README.md:17-18 agree; the example check is renamed localized_energy_is_globally_conserved_and_spreads (__main__.py:572) with the figure title 'Global energy conserved; profile spreads' (:489); and tests/unit/test_claim_wording.py rejects the three retired phrases and requires the corrected wording in three documents, passing inside the full 99-test run and failing when the retired phrase is reintroduced. However, the required action was to change ALL affected wording or implement a discrete continuity equation, and one affected statement was not changed. docs/scientific_hardening/FALSIFICATION_MATRIX.md:15, the Closure/conservation row, still reads 'Global microscopic energy and a declared local decomposition are conserved in one exact chain; covariant/discrete Bianchi closure is not implemented.' Executed: git blame -L 15,15 HEAD attributes it to cac51c33 and git show d527cb8b --stat is 2 files, +8/-2, touching only line 7 of that table and the unit test, so the round-2 fix never reached it. No discrete current, divergence or continuity residual exists anywhere in popgp/ or examples/, and the audited local decomposition demonstrably is not conserved: the example's own gate requires evolved_outside_fraction > 0.05 (__main__.py:583-585) and the committed artifact records t1_endpoint_fraction = 0.12078184671097476 against initial_endpoint_fraction = 4.54123182112811e-16. tests/unit/test_claim_wording.py was executed directly and PASSES with this sentence present; an injected mutation asserting site-by-site local conservation in the same covered text also passes. Re-review 2 closed this on a repository-wide search whose regex could not match the phrase 'local decomposition are conserved'."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Recorded as unresolved rather than superseded: SLAW-004 is the same defect at a location the prior chain did not inspect, not a replacement defect. This is a case where a prior re-review closed a blocking finding that is not in fact fully resolved, and per AGENT_REVIEW_WORKFLOW.md:35-36 it remains blocking until the required action is independently satisfied or a maintainer override is recorded. The residual is one sentence in one document, and a charitable reading exists under which 'a declared local decomposition is conserved' means only that the split remains an exact decomposition of a conserved total, which is implemented and tested at tests/scientific/test_many_body_source_law.py:246. That is why SLAW-004 itself is scoped medium and non-blocking; the blocker here is the unclosed prior required action, not a false numerical result. Correction rather than rewriting: per AGENT_REVIEW_WORKFLOW.md:33-34 this correction is issued as a new round and REREVIEW-2 is left intact."

prior_requested_test_results:
  - requested_test_id: "TST-CI-001"
    outcome: verified-satisfied
    evidence: "The obligation was to repair the cross-platform instability, add a regression catching the cause, and get the structured-artifact step green on a new frozen SHA. All three verified at the candidate. Green Linux CI at the exact candidate: public run 31380394043 / job 93429053764 at head_sha 763fd1857d0a1298aa9858bc7b7c698c38833ef7, with step 9 Regenerate documented examples and step 10 Check committed validation contracts and visual outputs both success. The repair is a declared semantic contract in place of the byte-exact diff, and its regressions are specific and directional: tests/unit/test_validation_artifact_contract.py:65-69 accepts a representative observed Windows-to-Linux drift, :72-79 rejects a changed gate outcome, :82-89 a weakened criterion string, :92-99 a changed array length, :102-109 a meaningful stable-config change, :112-119 a non-finite diagnostic, :122-139 excessive sensitive drift, :142-150 the specific observed significance_ratio drift; all 11 cases pass. I re-derived the cross-platform robustness argument myself: because comparison happens after json.loads, CRLF/LF, whitespace, key ordering and float-repr differences cannot break the gate, while a LAPACK sign or basis change under degeneracy would move the diagnostics far past the tolerances and flip the exactly-compared passed booleans. I additionally executed the contract on mutated copies of the real committed artifacts and confirmed it rejects gate-boolean, overall_pass, exact-value and array-length changes. Independent local replay of the entire CI job in a separate disposable worktree at the same candidate exited 0 at every step, with all 18 tracked artifacts regenerating bit-identically."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The literal sub-request to capture the exact regenerated validation.json diff from the failing Ubuntu runner at adbfab58 was never satisfied publicly, because raw job logs return HTTP 403 under the public-only boundary; the specific Linux field that failed is still unidentified from public evidence, and I record that as unavailable rather than inferring it. One named root cause, the chain_1d scale-aware selection_driver classification, has no unit test - grep over tests/ for selection_driver, stress_span and selection_margin returns nothing - and is protected only by the exactly compared selection_driver string in the committed artifact, which retains roughly a 20x margin (stress_span 6.639522486020637e-10 against selection_driver_tolerance 1.3219290882453315e-08)."

  - requested_test_id: "TST-SLAW-001"
    outcome: verified-satisfied
    evidence: "The requested acceptance-plus-rejection triad exists and I re-derived every leg by direct execution rather than by running the tests. tests/unit/test_simulator.py:243-284 covers matched-KMS acceptance over a full-site partition with source_scale=1.7 and the scaled modular-sum identity, :287-300 beta-mismatch rejection, :303-318 faithful non-Gibbs rejection, :347-361 partition completeness, plus tests/scientific/test_many_body_source_law.py:307-352. My independent replay: beta mismatch rejected at trace distance 2.741193e-01, faithful non-Gibbs rejected at 2.627858e-01, matched reference accepted with sum(source) equal to -source_scale * Delta<K_reference> to 1.73e-18 against an asserted tolerance of 2e-14 (relative 4e-12, i.e. tight rather than permissive). The acceptance leg asserts against an independent code path: popgp/information.py:228-231 modular_energy_delta forms K_sigma by matrix logarithm, not by the local-operator construction the source uses. A mutation battery in a disposable copy confirmed the enforcement is load-bearing: early return from _validate_kms_reference_state fails exactly the two rejection tests, widening the tolerance from 1e-10 to 1.0 fails the same two, flipping -beta_kms to +beta_kms fails 3, dropping cfg_time.source_scale fails 1, doubling the beta_kms fallback fails 3. The identity also survives a hostile ambient environment: with torch.set_default_dtype(torch.float32) set globally before import, the residual was still 1.73e-18 and the beta-mismatch rejection still raised."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "This triad is the only demonstrated negative control in the repository that exists because a reviewer asked for it rather than because any rule requires it, which is the substance of the guidance recommendation TST-GOV-004 below. Scope caveat recorded as RTV-004: the modular-sum identity is exactly invariant under any endpoint split whose per-edge weights sum to one, so it constrains sum_i h_i = H and the KMS premise but places no tight independent ground truth on the per-cell densities."

  - requested_test_id: "TST-SLAW-002"
    outcome: verified-satisfied
    evidence: "tests/unit/test_simulator.py:321-345 (test_run_propagates_reference_state_to_kms_candidate) drives the declared public API Simulator.run(reference_state=reference) with the KMS candidate selected and asserts result.pi_time.delta_rho_raw.sum() == approx(-modular_energy_delta(state, reference), abs=2e-14); it passes inside the full 99-test run. I reproduced the same end-to-end call independently and observed a residual of 8.67e-19, and confirmed the propagation chain by reading popgp/simulator.py:1098-1104 to :858-859 to :1001-1014. Mutation confirms sensitivity: deleting the forwarding argument makes exactly that test fail. tests/unit/test_simulator.py:200-206 additionally asserts that both reference-dependent source models raise ValueError matching 'requires a reference_state' when none is supplied."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The committed regression monkeypatches Simulator.prepare at tests/unit/test_simulator.py:333 to obtain a localized excited state, and one subagent proposed downgrading this requested test on that basis. Tested and refuted: the monkeypatch is a convenience for that test's stated purpose (propagation), not a necessity - the unpatched public route with cell_dim=1 yields a manifestly non-uniform source and ||phi|| = 8.096e-01, and the public stage-wise route Simulator.run_pi_time(excited_state, ..., reference_state=reference) yields ||phi|| = 2.605e-01 with no patching. The requested test is satisfied."

  - requested_test_id: "TST-SLAW-003"
    outcome: unresolved
    evidence: "The request offered two branches: test a specified lattice continuity equation if a local-conservation claim is retained, or otherwise regression-test the corrected global-conservation wording and labels. Neither is complete. (a) No discrete current, divergence or continuity residual exists in popgp/ or examples/; a search for continuity, divergence residual and current finds only prospective or disclaimer text, so the first branch was not taken. (b) A local-conservation claim IS retained, at docs/scientific_hardening/FALSIFICATION_MATRIX.md:15 ('Global microscopic energy and a declared local decomposition are conserved in one exact chain'), which is inside a document the wording regression reads. (c) The wording regression does not catch it: tests/unit/test_claim_wording.py was executed directly against the frozen tree and passes with that sentence present, and an injected mutation appending 'but the audited local-energy profile is conserved site by site' to the covered falsification-matrix text still satisfies all of its assertions. The regression that does exist is real and sensitive within its scope - appending 'the audited energy profile evolves conservatively' to FALSIFICATION_MATRIX.md makes it fail at line 19, and restoring the file makes it pass - and tests/scientific/test_many_body_source_law.py:233-279 does assert global drift below 1e-13 together with spreading above 0.05, matching the corrected row-7 wording. But the obligation as written was to leave no retained local-conservation claim unguarded, and one remains."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Prior re-review 2 marked this verified-satisfied. Downgrading to unresolved on executed evidence. Tracked forward by new requested test TST-SLAW-004 and new finding SLAW-004. The inherent limitation of an exact-string denylist - that it cannot catch a novel rewording - is generic and was disclosed by REREVIEW-1 and REREVIEW-2; that generic limitation is not what makes this unresolved. What makes it unresolved is a specific, currently present, unguarded claim in a file the guard already reads."

  - requested_test_id: "TST-CI-002"
    outcome: verified-satisfied
    evidence: "The request was to add a stable visual-artifact smoke policy, with 'required nonempty files on the Linux runner' given as the example, or else to state explicitly that CI attests structured metrics only. The first branch is implemented: scripts/check_validation_artifacts.py:55 defines VISUAL_SUFFIXES over .gif/.jpeg/.jpg/.png/.svg/.webp, and check_required_visuals at :203-236 errors when a document declares no visual artifact, when a declared path escapes the repository root, and when a declared visual is untracked by git ls-files, missing, or zero-length; it is invoked per example at :301 and wired as the final CI step, which is green at the candidate (run 31380394043 step 10). Its accept and reject paths are covered by tests/unit/test_validation_artifact_contract.py:153-185, and I confirmed by execution that deleting or truncating a declared visual produces exit 1. The policy is described accurately and without overstatement in the script docstring at :13 and at docs/scientific_hardening/REPRODUCIBILITY.md:61-62 ('Required visual outputs must be tracked, present, and nonempty. PNG pixels and metadata are not hashed across environments.'), and the CI step name itself says 'committed'."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Two subagents proposed downgrading this to unresolved on the ground that actions/checkout restores the committed figures, so the check cannot fail for a suppressed image write. That mechanical observation is correct and I reproduced it independently in this round's quality suite: changing one plot title in examples/physics_qg/source_law_many_body/__main__.py and regenerating changed many_body_source.png from 241734 to 242910 bytes while ruff, pytest and check_validation_artifacts.py all exited 0. But the downgrade does not survive: the implemented policy is the requester's own named example, and its scope is stated precisely in both the code and the reproducibility document rather than overclaimed. I therefore record the obligation as satisfied and file the hardening separately as non-blocking requested test TST-CI-003, so that a real residual weakness is carried forward without misrepresenting the prior rounds' dispositions."

predictions:
  experiment_id: "TST-LIB-001"
  predicted_outcome: "Plumbing the Hamiltonian family (or the backend's own intra-cell restriction of build_interaction_terms) into _build_local_hamiltonian will change the Pi_res selection on the 4-qubit Ising chain with cell_dim=2 from [[0,3],[1,2]] at L_leak = 1.494911e-03 to [[0,1],[2,3]] at L_leak = 5.712651e-04, while leaving the control partition [[0,2],[1,3]] at 1.798158e-03 unchanged because it contains no intra-cell edge, and leaving every committed validation artifact byte-identical because chain_1d, grid_2d and gravity_well all use the default Heisenberg family."
  predicted_failure_mode: "The likely inadequate fix is a family switch added only inside _build_local_hamiltonian with a regression written against the example rather than against the generator. That would pass without ever exercising the invariant that matters, namely that the sum of the in-cell generators plus the inter-cell terms equals backend.build_hamiltonian() for both families; a third family, a next-nearest-neighbour term, or an anisotropic coupling added later would silently reintroduce the same class of defect. A second predicted failure mode is that, because no committed artifact changes, the fix is judged unverifiable and downgraded to a documentation caveat - which is the disposition the initial review correctly pre-rejected for SLAW-001."
  confidence_statement: "High confidence in the numerical prediction: both leakage rankings were computed by direct execution against the frozen tree, and the invariance of the no-intra-cell-edge control partition is an independent internal check that the generator swap is the only thing changing. Moderate confidence in the failure-mode prediction, which is a judgement about remediation behaviour rather than a measurement."

recommendation:
  approve: false
  blocking_findings: 2
  rationale: "Two blockers, decomposed exactly as the arithmetic requires: one confirmed new blocking finding (LIB-001) plus one prior finding left unresolved (SLAW-003). No other new finding is blocking - GOV-003, SLAW-004, CI-005, LIB-004, SLAW-005, RTV-004, STAT-002, STAT-003 and STAT-004 are all non-blocking after verification, and every finding proposed in this round that did not survive an adversarial verification pass has been excluded from this artifact rather than softened. LIB-001 is blocking because Simulator.run() silently produces a wrong Pi_res cell decomposition for a documented, reachable, unwarned configuration (hamiltonian='ising'), inverting the qualitative locality result that Pi_res is cited for, with no error, no warning and no test covering any non-default family; it is a latent path rather than a corrupted published number, which is why the severity is high rather than critical. SLAW-003 is unresolved because its required action was to change all affected wording and one affected statement remains at docs/scientific_hardening/FALSIFICATION_MATRIX.md:15, in the row a referee reads for conservation status, contradicting line 7 of the same table; per AGENT_REVIEW_WORKFLOW.md:35-36 a blocking finding stays blocking until the required action is independently satisfied or a maintainer override is recorded, and neither has happened. Both blockers are pre-existing rather than introduced by this candidate: git diff --stat b519c7dafd4488d587a1edcbe14891f0c83aedb5 763fd1857d0a1298aa9858bc7b7c698c38833ef7 is 7 documentation files, +622/-0, touching no code, test, workflow or artifact, so this commit cannot have regressed anything - it is nevertheless the commit under review, and approve: true requires zero unresolved blocking findings at the reviewed commit regardless of which round introduced them. Everything else in the chain is in good order: CI-001, SLAW-001 and SLAW-002 are verified resolved by counterexample replay and mutation rather than by accepting the builder response, four of five prior requested tests are verified satisfied, the exact candidate SHA has a green public CI run, the full quality suite reproduces cleanly in an independent worktree with all 18 committed artifacts bit-identical, and the central source-law gates were probed directly and shown to be sensitive to the physics they adjudicate. On the guidance the recommendation is also changes requested, but only at low severity: GOV-003 should be fixed, and the governance layer should gain a rule requiring an acceptance gate to ship a demonstrated negative control (TST-GOV-004) and a typed independence declaration (TST-GOV-005). Finally, this recommendation should be weighted with the independence statement above: this review was not run in a fresh session, its orchestrator carried conclusions from three earlier informal audits of ancestor commits, and it relayed the historical gate-insensitivity pattern to every reviewing subagent. It is process separation with a different model, not external scientific validation."
```

## Method

### Frozen worktree and integrity verification

The review was conducted read-only in the frozen reviewer worktree
`C:/src/POPGP-review-source-law-rereview-3`. Before any review work began I verified the
worktree against the supplied handoff:

- `git rev-parse HEAD` returned `763fd1857d0a1298aa9858bc7b7c698c38833ef7` exactly.
- `git rev-parse "763fd1857d0a1298aa9858bc7b7c698c38833ef7^{tree}"` returned
  `53ff112b28572dac16e66c0281307b047db2126c`, matching the declared `context_hash`.
- `git status --porcelain` returned nothing, before and after the review.

I ran no example and modified no file in that worktree, in `C:/src/POPGP`, or in
`C:/src/POPGP-source-law-remediation-1`. Counterexample and mutation work was performed
either in memory (importing the frozen sources through `sys.path` with
`PYTHONDONTWRITEBYTECODE=1`, after confirming `popgp.__file__` resolved into the review
worktree) or on copies under the session scratchpad. The only file written into the
frozen worktree is this artifact.

### Quality suite in a separate disposable worktree

Per the workflow's requirement to keep the reviewer worktree clean while still verifying
the suite by execution, the complete CI job was replayed in a **separate disposable
worktree** detached at the same candidate,
`C:/src/POPGP/.claude/worktrees/wf_e8f619f6-0de-1`. Checkout output was
`HEAD is now at 763fd18 Add independent agent review guidance`, and I re-verified both
hashes there before running anything:

```
git rev-parse HEAD          -> 763fd1857d0a1298aa9858bc7b7c698c38833ef7
git rev-parse "HEAD^{tree}" -> 53ff112b28572dac16e66c0281307b047db2126c
git status --porcelain      -> (empty)
```

The `.github/workflows/ci.yml` `python` job was then reproduced step for step, in order:

| Step | Command | Exit | Observed |
|---|---|---|---|
| Sync locked environment | `uv sync --frozen` | 0 | CPython 3.11.15; 55 packages (numpy 2.4.2, scipy 1.17.0, torch 2.10.0, matplotlib 3.10.8, pytest 9.1.1, ruff 0.16.2) |
| Lint | `uv run ruff check .` | 0 | `All checks passed!` |
| Validate manuscript source | `uv run python scripts/check_tex.py` | 0 | 652 lines; no non-ASCII; final brace balance 0; all 9 environments matched; no Markdown remnants |
| Test | `uv run pytest -q` | 0 | `99 passed in 8.47s` |
| Regenerate documented examples | six `python -m examples.physics_qg.*` | 0 each | chain_1d, grid_2d, gravity_well, source_law, source_law_many_body, ca_model |
| Check committed contracts | `uv run python scripts/check_validation_artifacts.py` | 0 | `Validation artifact contracts and required visual outputs are valid.` |

`uv run pytest --collect-only -q` reported `99 tests collected`, equal to the 99 passed,
so there are no silent skips, deselects or collection errors. After regenerating all six
examples, `git status --porcelain` and `git diff --stat` were both empty, and
`git hash-object` on all 18 tracked artifacts under `examples/physics_qg/*/results/`
matched the corresponding `git ls-files -s` blob hashes exactly - byte-identical
regeneration, with zero numeric drifts accepted by the comparator, so no platform-drift
tolerance was exercised. This was on win32 / CPython 3.11.15; CI pins ubuntu-latest.

Gate-teeth probes were run in that same disposable worktree and reverted afterwards
(`git checkout --`, with `git hash-object` confirming the original blobs and
`git status --porcelain` empty). `scripts/check_tex.py` exits 1 on an injected brace
imbalance and on a deleted `\end{itemize}`. `scripts/check_validation_artifacts.py`
exits 1 on a flipped gate boolean, on a +1.0% change to
`modular_susceptibility.estimate`, and on a zero-byte declared figure; it exits 0 on
three simultaneous sub-tolerance changes, including a 50x change to `significance_ratio`
(whose declared `rel_tol` is 0.99, a documented and bounded exception). One probe is
worth recording plainly: changing a single plot title and regenerating changed
`many_body_source.png` from 241734 to 242910 bytes while ruff, pytest and the artifact
checker all exited 0, and `.github/workflows/ci.yml` has no `git diff --exit-code` step
after regeneration. That is carried forward as non-blocking requested test TST-CI-003,
not as a finding, for reasons given there.

### External evidence

GitHub Actions evidence was obtained through unauthenticated public REST endpoints:
`GET /repos/whact2025/POPGP/actions/runs?head_sha=763fd1857d0a1298aa9858bc7b7c698c38833ef7`
returned `total_count 1`, run `31380394043`, branch
`codex/source-law-linear-response-remediation`, event `push`, conclusion `success`; the
jobs endpoint returned job `93429053764` with every step successful. Raw job-log bodies
returned HTTP 403 and were **not** fetched. Throughout this artifact, remote CI outcomes
are described as run/job/step conclusion metadata read from a public API - never as my
own execution.

### Verification levels

Every `verification` value in this artifact is `confirmed-by-execution`, and each such
entry names the commands, inputs and observed numbers. Where a claim is partly
inferential the split is stated inside the evidence field rather than hidden behind the
label. Specifically: SLAW-004's textual defect is established by inspection while its
git-history and guard-coverage halves are executed; CI-001 and TST-CI-001 combine my own
local execution with public run metadata that I explicitly do not claim as execution;
and LIB-001's consequence for downstream stages is a read-only inference from the code
path, while the generator mismatch and the inverted leakage ranking are executed.

### Adversarial verification of this round's own findings

Every candidate finding produced in this round was put through a separate verification
pass whose default was refutation, requiring current (non-stale) quotes, independent
reproduction of the executed numbers, and - for any claim that a gate cannot fail -
either an algebraic independence argument or a named input that would make it fail.
Twenty-four candidate findings were refuted and are **not** reported here; ten survived
and appear above, several with severity or blocking status corrected downward. The
refuted set is not a secret list of suppressed concerns: it includes several superficially
compelling charges - that the governance layer's branching rule strands the approving
re-review, that `Simulator.run` can never produce a nonzero KMS source, that the
precision-floor threshold certifies a materially wrong coefficient, that the manuscript's
two renderings state contradictory scientific status - each of which failed on a specific,
checkable point. I mention this because a review that reports ten findings out of
thirty-four candidates is making a claim about its own discipline, and that claim should
be auditable.

## Guidance assessment

The project explicitly asked for an assessment of the review guidance introduced by this
candidate. What follows is my own judgement; where a charge from this round's analysis
did not survive verification I say so rather than repeating it.

### Overall

The six documents added by `763fd18` are a competent, unusually honest **process**
document set and a weak **scientific-adjudication** standard. They are worth adopting.
The corrections below are additive; none of them requires rethinking the role model.

### What is sound

The builder / reviewer / maintainer separation is coherent and the authority limits are
correctly non-overlapping and correctly asymmetric: `AGENT_REVIEW_WORKFLOW.md:14-16`
forbids the builder from marking its own findings resolved, forbids the reviewer from
authorising merge, and forbids the maintainer from converting model agreement into
empirical validation. The separation of `disposition` (agreement) from
`implementation_status` (action) at `:96-97`, and the sentence at `:101` - "The builder
may say a change is `implemented`; only a later independent reviewer may say
`verified-resolved`" - is exactly the right control, and it is mirrored correctly in
`REVIEW_RESPONSE_TEMPLATE.md:74-76` and `INDEPENDENT_REVIEW_TEMPLATE.md:85-86`.

The rule at `:35-36` that a disputed or deferred blocking finding stays blocking until
independent evidence or a recorded maintainer override is the right default, and it is
load-bearing in this very round: it is why SLAW-003 counts as a blocker rather than being
argued away. `DISAGREEMENT_LOG_TEMPLATE.md:31-35` correctly requires the discriminating
test to be frozen before it is run, and its closing sentence at `:49-50` - "A maintainer
override may authorize merge but does not change an unsupported scientific proposition
into a verified result" - is the single best line in the set. The finding lifecycle at
`:136-144` is complete for the states it defines and maps onto the three template outcome
enums with no gaps. The instruction at `:107` and `REVIEW_RESPONSE_TEMPLATE.md:83-84`
that a response artifact cannot contain its own commit hash is correct and non-obvious.

`REVIEWER_IDENTITY.md` is the most intellectually honest part of the set. Three statements
are exactly right and I applied all three to myself above: `:17-18` "Changing the model
occupying the seat does not change the role and does not by itself make a review
independent"; `:42-44` that same-operator runs are "useful process separation but not
external scientific independence"; and `AGENT_REVIEW_WORKFLOW.md:7-8` "Model agreement is
review evidence; it is not independent experimental confirmation of a physical claim."
The last is well placed, in the fourth and fifth lines of the primary document.

### Self-consistency

The two governance documents agree on branch names, artifact paths, remediation base
commits and the handoff record, and `LAUNCH_INDEPENDENT_REVIEW.md` sections 2, 4 and 5
reproduce the `AGENT_REVIEW_WORKFLOW.md:40-50` layout faithfully. One real mismatch
survives verification and is reported as **GOV-003**: the quality suite is specified twice
and differently, and the freeze step designates the weaker specification as authoritative.

Three further observations, none of which is a defect:

- `REVIEW_RESPONSE_TEMPLATE.md:22` introduces `legacy_requested_test_id_method:
  "not-applicable"` with no definition, no allowed values, and no mention in the field
  rules or in `AGENT_REVIEW_WORKFLOW.md`. It is copied verbatim into both existing
  responses. This is dead schema, worth either defining or deleting.
- I checked and reject the charge that `review_id` is ambiguous between a stable chain
  token and a per-round token. `LAUNCH_INDEPENDENT_REVIEW.md:73` already says
  `Review ID: <stable-review-id>`, and the round suffix appears only in the filename
  (`<review-id>.md` versus `<review-id>-REREVIEW-1.md`). The guidance is clear; only the
  pre-schema artifacts deviate.
- I also reject the charge that `RESPONSE-2` under-answered its obligations.
  `RESPONSE-2:14` sets `review_id: POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-1`, and
  `REREVIEW-1` carries `findings: []` and `requested_tests: []` with only SLAW-003 left
  open. `REVIEW_RESPONSE_TEMPLATE.md:74-75` ends "Use an empty list only when the review
  contains none." One finding response for one open finding, and an empty
  `requested_test_responses`, is exactly conformant.

### Does the project follow its own rules?

Mostly yes, and the parts that hold up were verified rather than assumed.

- **Reviewer-branch hygiene**: `git diff --stat` for the three review commits
  `18aa5b3feea4f35e8bf89f1ae4cd34ce57fc279e`, `088724989e69d427fdb63ad930787b5c9a3f47fb`
  and `1dd86b3ba0c4a4aaea3b3027f2263a547c8a7eb6` shows exactly one file each, all under
  `reviews/independent_reviewer/`, at +170 / +166 / +158 lines. The "reviewer branches
  contain review artifacts only" rule is fully honoured.
- **Remediation bases**: `cb004f08` descends from the initial-review commit `18aa5b3f`
  and `d527cb8b` descends from the re-review-1 commit `08872498`, matching `:48-49`.
- **Hash discipline**: every 40-character hash appearing in the four in-tree artifacts and
  in `REREVIEW-2` resolves - twelve commits and three trees checked with
  `git cat-file -t` - and each recorded `context_hash` matches its stated
  `git rev-parse <commit>^{tree}` command.

Two deviations are worth recording, neither of them a defect in the guidance:

- `REREVIEW-1` uses five `verification` values outside the committed enum
  (`confirmed-by-execution-and-inspection` twice,
  `confirmed-by-inspection-and-execution`,
  `confirmed-by-public-execution-and-local-execution`,
  `confirmed-by-public-execution-and-local-replay`), and the initial review omits
  `artifact_schema_version`. These artifacts **predate** the template, which is added by
  this candidate, so they cannot be charged with non-conformance; and per `:33-34` they
  must not be rewritten. The honest disposition is a provenance note in a future round,
  not a retroactive edit. Note also that the extra labels are strictly *more* informative
  than the enum, not less.
- `docs/governance/` does not exist in the tree at `adbfab5`, `18aa5b3`, `b25988c`,
  `b519c7d` or `1dd86b3`. The three prior artifacts were therefore produced against
  guidance held outside version control, and their conformance to the now-frozen text
  cannot be audited from the repository. That is unavoidable for a first commit of a
  process document, and it is not a reason to block one; a one-line note in the runbook
  saying the in-tree exemplars predate the template would close it cleanly.

I also examined, and reject, the charge that the branch layout structurally strands the
approving re-review. The approving artifact `1dd86b3` is not an ancestor of the candidate
because the builder branched the new candidate from the pre-review commit `b519c7da`, not
because rule `:48-50` prevents it - applying that rule would have carried `REREVIEW-2`
onto the builder line. `1dd86b3` is durably pushed as
`refs/heads/review/source-law-linear-response-rereview-2`, which is precisely the
mechanism `LAUNCH_INDEPENDENT_REVIEW.md:98` prescribes. A merge-readiness ancestry check
would still be a sensible addition, but its absence is a gap in completeness rather than a
defect. One plain observation does follow: `763fd18` adds 622 lines on top of the approved
`b519c7da` and no artifact names it as `commit_reviewed`, which is exactly why this round
exists and is the guidance working as designed (`:25-26`).

### Operationality

About half the rules are checkable in principle - hash length and resolution, branch base
commits, one-file review commits, one-response-per-open-item, enum membership, artifact
presence - and **none** of them is checked. `.github/workflows/ci.yml` has one job and
eight run steps and touches neither `reviews/` nor `docs/governance/`; `scripts/` contains
only `check_tex.py` and `check_validation_artifacts.py`; no test under `tests/` reads a
review artifact. Enforcement is therefore entirely human. That is a legitimate design
choice for a documentation-only commit that never claims machine validation - the workflow
explicitly assigns verification to the maintainer at `:129-132` and to the completion
checklist at `LAUNCH_INDEPENDENT_REVIEW.md:150-160` - so it is not a defect. But it is a
fragility worth naming, and the pre-schema drift above shows what unenforced schema looks
like after only three rounds.

The rules that are not checkable even in principle are the ones that matter most
scientifically. "Reproduce important behavior and construct counterexamples or mutations
where feasible" (`:75`) and "Add unit tests and scientific regression or falsification
tests as appropriate" (`:57`) are effort-bounded by design. The template does leave a
partial trace - a reviewer who constructed nothing must record `read-only` or `unverified`
with no commands, per `INDEPENDENT_REVIEW_TEMPLATE.md:42` and `:88` - which is more than
nothing, and I withdrew a stronger version of this charge on exactly that ground. But
there is no field that records *which* gates were probed or *what* mutation was attempted,
so a thorough gate audit and a cursory one produce structurally identical documents.

### What is missing

This is the substantive recommendation. Searched exhaustively: the six documents contain
no requirement that an acceptance gate ship a demonstrated negative control; no rule
distinguishing an analytic identity from a measurement; no numerical-precision-floor
policy; no requirement that a gate statistic vary with the physics across a declared
sweep; and no cross-platform reproducibility standard. None of the six mentions
`docs/scientific_hardening/`, `FALSIFICATION_MATRIX.md` or `CLAIMS_MATRIX.md` by name.

The last omission is the sharpest evidence, because the repository has already
independently invented most of these controls. `FALSIFICATION_MATRIX.md` row 1 gates on
"first-order negative control accepted" and "signal/floor <1000"; the same row states
"Under affine mixtures unit slopes are identities"; row 10 reports "direct order,
precision-floor, negative-control, Richardson, and Kubo--Mori comparison gates" across a
declared beta and N sweep; and the file closes with a binding rule that changing "the
source law, inference rule, or acceptance threshold requires a recorded decision and a
rerun of all relevant controls." The scientific discipline exists. The governance layer
simply does not require any of it, so nothing prevents the next gate from being introduced
without it.

Two findings in this round are instances of exactly that gap. **SLAW-005**: the
global-energy-conservation leg of the source-law gate is an algebraic identity of unitary
evolution generated by the observable being measured, and it is listed in
`FALSIFICATION_MATRIX.md:7` as a failure threshold; no rule in the new guidance would have
caught it. **CI-005**: no gate anywhere validates a validation artifact's internal
consistency, and the one-key demotion that once concealed a failing `ca_model` check is
unguarded; again, no rule addresses it. The requested tests TST-GOV-004 and TST-GOV-005
are the operational forms of the remedy: a committed mutation test per declared gate, a
meta-test that every gate row names one, a sweep-variation assertion per numerical gate,
and a typed `independence_declaration` block to replace the free-text field.

### `REVIEWER_IDENTITY.md`: is the disclosure requirement sufficient?

Partially. The mechanism has worked in practice - `independence_statement` is a required
field and all three prior artifacts disclosed the shared operator and orchestrated task
honestly. But it is unenforceable in a way that access declaration is not:
`hidden_access_declaration` is three typed booleans a script can check, while independence
is a single free-text string that a vacuous sentence would satisfy. Two concrete gaps
follow. First, the review artifact records no builder identity at all - there is no
`builder_model_identity` field - so the guidance's own preference for "a different model"
(`AGENT_REVIEW_WORKFLOW.md:18-19`, `REVIEWER_IDENTITY.md:38`) cannot be verified from a
review artifact in isolation. Second, `access_level: public-repository-only` is pre-filled
as a literal default rather than left blank with an enum, which invites a reviewer with
broader access to leave the wrong value in place; this round's `access_level` had to be
widened by hand.

This round is the strongest available argument for typing the field. The disclosure above
- that this review was not run in a fresh session, that its orchestrator carried
conclusions from three earlier informal audits, and that it relayed a specific defect
class to every subagent - is a matter of reviewer conscience. Nothing in the schema would
have detected its omission, and the resulting artifact would have looked exactly as
compliant.

### Recommendation on the guidance

**Changes requested, at low severity.** Fix GOV-003. Schedule TST-GOV-004 (gate mutation
tests) and TST-GOV-005 (typed independence declaration), and either define or delete
`legacy_requested_test_id_method`. Adopt the rest as written.

## Prior findings and requested tests: verification narrative

The re-review obligation is to assign an outcome to every prior finding and every prior
requested test, and to reproduce the evidence rather than accept the builder's
dispositions. `REREVIEW-2` carried `findings: []` and `requested_tests: []`, so the open
chain is four findings (CI-001, SLAW-001, SLAW-002, SLAW-003) and five requested tests
(TST-CI-001, TST-CI-002, TST-SLAW-001, TST-SLAW-002, TST-SLAW-003).

The candidate diff from the approved commit is documentation only - 7 files, +622/-0,
touching no code, test, workflow or committed artifact - so no round-2 conclusion can have
regressed. That established, I re-derived rather than carried forward.

**CI-001 (verified-resolved).** The byte-exact `git diff --exit-code` gate is gone and the
replacement is a real gate: I executed `scripts/check_validation_artifacts.py` on mutated
copies of the real committed artifacts and confirmed it rejects a flipped check boolean, a
flipped `overall_pass`, a 1% change to an exact Kubo-Mori value, a doubled profile array
and a missing declared visual, while reproducing the documented factor-100
`significance_ratio` boundary exactly (x99 and x100 accepted, x100.5 and a sign reversal
rejected). Both named cross-platform root causes are fixed at source, and the Delaunay
ordering fix has a regression that fails when the sort is removed. I also verified that
the `chain_1d` classification flip from `mixed_stress_and_penalty` to `spectral_penalty`
is the *more correct* label, not a cosmetic tolerance widening: the stresses are
4.860e-16, 6.640e-10 and 5.179e-16 against a selection margin of 1.3219e-2, so the choice
really is spectral. Public CI is green at the exact candidate SHA, and the whole job
replays cleanly in an independent worktree with bit-identical artifacts.

**SLAW-001 (verified-resolved).** This is the round where re-derivation mattered most, so
I did it in both directions. Against the frozen tree, the initial review's two
counterexamples are now rejected with quantitative trace distances (2.741193e-01 and
2.627858e-01), the matched case reproduces the modular-sum identity to 1.73e-18 against a
2e-14 assertion, and the tolerance is a real discriminator (1e-11 accepted, 1e-10
rejected). Against the *pre-fix* package extracted from `adbfab5` into a scratch
directory, the original defect reproduces exactly, returning the predicted ratio
`beta_kms/beta_reference = 0.5/1.3 = 0.38461538461503`. Five separate mutations of the
validator each break exactly the tests that certify it, so the guard is load-bearing
rather than decorative. Two details deserve credit: the enforcement is at the runtime
boundary where the state first exists, which is what the initial review demanded and what
it explicitly pre-rejected a cheaper fix for; and the gate is non-circular, because the
expected state is recomputed from `backend.build_hamiltonian()` independently of the
supplied reference, so a wrong Hamiltonian, a wrong beta and a non-Gibbs faithful state
each produce a nameable failure.

**SLAW-002 (verified-resolved).** The keyword exists on the advertised primary API and is
forwarded end to end; deleting the forwarding argument fails the regression. One subagent
proposed superseding this on the ground that the public path can never produce a nonzero
clock potential. I tested that proposition and it is false: the zero it reports is an
artifact of the default `cell_dim=2` on an open 4-chain, whose mirror symmetry makes the
two cell sources equal so the zero-mode removal annihilates them. With `cell_dim=1` - the
partition the repository's own KMS experiment uses - the unpatched public route gives a
manifestly non-uniform source and `||phi|| = 8.096e-01`, and the same holds at n=3, 5, 6,
7 and for Ising. Resolved, not superseded.

**SLAW-003 (unresolved).** This is the correction this round exists to make. Every
location named in the original finding is genuinely fixed, and the wording regression is
real and sensitive within its scope. But the required action was to change *all* affected
wording, and `FALSIFICATION_MATRIX.md:15` still asserts, in the row titled
"Closure / conservation", that "a declared local decomposition [is] conserved in one exact
chain". `git blame` puts that sentence at `cac51c3`, and `git show d527cb8b --stat` is 2
files / +8/-2 touching only row 7 and the unit test, so the round-2 fix never reached it.
Three rows earlier the same table says no local conservation law is implemented, and the
example's own gate *requires* the profile not to be conserved (`t1_endpoint_fraction =
0.12078184671097476`). Executing `tests/unit/test_claim_wording.py` shows it passes with
the sentence present, and an injected site-by-site local-conservation mutation in the same
covered text also passes. `REREVIEW-2` closed this on a search whose regex could not match
the phrasing.

I want to be precise about the weight of this. A charitable reading of line 15 exists -
"the split remains an exact decomposition of a conserved total" is implemented and tested
- which is why the new finding SLAW-004 is scoped medium and non-blocking on its own
merits. What is blocking is not the sentence; it is that a prior blocking finding's
required action is not satisfied, and `AGENT_REVIEW_WORKFLOW.md:35-36` says such a finding
stays blocking until it is. Per `:33-34` the correction is issued as this new round rather
than as an edit to `REREVIEW-2`.

**Requested tests.** TST-CI-001, TST-SLAW-001 and TST-SLAW-002 are verified satisfied by
independent execution, mutation and, for TST-SLAW-001, a five-mutation battery plus a
hostile-ambient-dtype probe. TST-SLAW-003 is downgraded to unresolved: neither branch of
the request is complete, because no continuity current was implemented *and* a retained
local-conservation claim is not covered by the wording regression. TST-CI-002 is verified
satisfied, over a proposed downgrade that I examined and rejected: the implemented
tracked/present/nonempty policy is the requester's own named example and its scope is
stated precisely in both the code (`scripts/check_validation_artifacts.py:13`) and the
reproducibility document (`:61-62`). The residual weakness that motivated the proposed
downgrade is real - I reproduced it - and is carried forward as non-blocking requested
test TST-CI-003 rather than by retroactively recharacterising the prior rounds.

## New findings

Ten findings survived verification; they are stated in full in the YAML above. Three
deserve narrative.

**LIB-001** is the one genuine physics defect found in this round and the only new
blocker. `_build_local_hamiltonian` takes no Hamiltonian-family argument and always builds
`XX + YY + ZZ`, while `ExactBackend.build_interaction_terms` correctly emits ZZ-only terms
for `hamiltonian='ising'`. The trace-then-evolve branch of the leakage commutator is
therefore generated by an operator the substrate does not have. On the 4-qubit Ising chain
the shipped code selects the non-contiguous `[[0,3],[1,2]]` at
`L_leak = 1.494911e-03`; with the correct generator the contiguous `[[0,1],[2,3]]` wins at
`5.712651e-04`, a factor of 2.6. The `[[0,2],[1,3]]` partition, which has no intra-cell
edge, scores `1.798158e-03` in both runs - exactly the invariance the generator swap
predicts, which is what convinced me the effect is the generator and not something else.
`Simulator.run()` completes on this configuration with no error and no warning. No
committed artifact is affected, because every shipped example uses the default Heisenberg
family, and no test covers any non-default family. That combination - a reachable,
documented, unwarned configuration producing silently wrong physics with zero test
coverage - is why it is blocking despite affecting no published number.

**SLAW-005** is the clearest instance of the defect class the guidance does not address.
`conservation_drift = np.ptp(Tr[(rho(t)-sigma)H])` is measured with `U = exp(-iHt)` built
from the same eigendecomposition of the same `H`, so `[U,H] = 0` and the quantity is
identically zero for any state, any split, any source model and any epsilon. I confirmed
it: `0.0` for the real setup, `0.0` for a deliberately wrong all-energy-at-site-0 split,
at most `2.22e-16` for random density matrices, and nonzero (`7.63e-02`) only when the
measured observable is a *different* Hamiltonian from the generator. It is listed at
`FALSIFICATION_MATRIX.md:7` as a failure threshold. Nothing asserted is false - it is
exactly true and completely uninformative - and the limitation is disclosed three times
elsewhere, which is why the severity is low. But a failure threshold that no physics can
cross should be relabelled as an implementation-consistency check, exactly as the
repository already labels its `*_identity_regression` checks.

**CI-005** is the artifact-level analogue. `compare_validation_documents(x, x)` returns
`passed=True` with zero errors for *any* `x`, including a self-contradictory document, and
no test reads any committed `validation.json`. For `ca_model` and `chain_1d` - the two
examples that filter informational checks out of the headline verdict - adding one
`"severity": "informational"` key to a failing check flips `overall_pass`. The repository's
own history at `47b8bae` contains that exact edit being reverted. The consequence is
bounded, because the contract forces the changed artifact to be recommitted so the diff is
visible, but detection currently depends entirely on a human noticing.

## What holds up

A great deal, and it should be said as plainly as the blockers.

**The central source-law gates are physics-sensitive.** This was the historical failure
mode and it is fixed in the reviewed candidate, confirmed by direct probing rather than by
reading criterion strings. `assess_quadratic_response` rejects linear contamination down to
`c1 = 1e-8` against `c2 = 0.0778`, a linear term only `1.3e-4` of the quadratic term at the
top of the epsilon window, and rejects cubic contamination at `c3 = 10`. The Richardson
gate correctly fails a strictly quadratic response (estimate 0.0, significance 0.0) and
fails a symmetry-vanishing perturbation such as `V = Z_center` (ratio 0.276). The
"constant of the sampling grid" statistic is gone: `significance_ratio` spans `7.4e7` to
`5.7e9` across the twelve sweep configurations and tracks the physics, non-monotonic in
beta for Heisenberg and monotonic for Ising. The committed first-order negative control
fails for the right reason (log-log slope 1.0000354, normalized RMSE 0.676), not by
tripping an unrelated exception.

**The Kubo-Mori cross-check is not circular.** The numerical route re-diagonalizes
`H + eps*V` nine times; the analytic route uses only `eigh(sigma)` and the logarithmic-mean
kernel. I verified `kubo_mori_covariance` independently two ways - central differences of
the actual Gibbs family (relative error 3.0e-10 at h=1e-4, consistent with O(h^2)) and
Gauss-Legendre quadrature of the integral representation using `matrix_exp`, a code path
with no eigendecomposition, agreeing to 1.5e-14. The degenerate branch at
`popgp/information.py:208-215` is the correct limit and was probed at 60 digits with
`mpmath` across the transition window with no catastrophic cancellation. The comparison is
an accuracy control against an independently computed exact limit; it is not two
evaluations of one expression.

**The remediations were correctly typed.** SLAW-001 was a code defect and got a runtime
code fix, not a documentation caveat. SLAW-002 was an API defect and got an API change.
SLAW-003 was a claim defect and got a wording fix plus a wording regression - and the
round-1 reviewer refused to close it while one document still said "the audited energy
profile evolves conservatively", which is the single best judgement call in the chain and
is precisely the discipline that this round is extending to line 15.

**Negative results are retained, not buried.** Three of six committed examples carry
`overall_pass: false`. `ca_model` still records its failing `population_growth` check.
`CLAIMS_MATRIX.md` volunteers an unresolved false positive under C08 (a disjoint-Bell
control receiving a false `D*=1 geometric_candidate` status). Identity-only results are
labelled as identities in the artifacts themselves
(`affine_linearity_identity_not_a_falsification_test`,
`isospectral_unitary_identity_regression`). The many-body artifact's
`scientific_status` is `feasible_candidate_not_validated_physical_law` with an explicit
`remaining_requirements` list naming state-independent localization, covariant
conservation, refinement behaviour and an independent clock observable. The MST degeneracy
of the four-cell chain recovery is disclosed in the artifact, in the README and in C06.

**Numerical claims match artifacts.** Every quantitative statement I could bind to an
artifact matched: the grid figure's 12 held-out edges at precision and recall 1.0 with
`D*=2`; the chain's 105 partitions, entropy increases 1.1125 and 1.2998, and near-zero
stress 4.86e-16; the gravity well's `Phi(center) = -0.0099637`, `1+z = 1.0122966` and
constraint residual 1.878e-18; the qutrit equal-energy control at modular energy
0.5752103826 for both states; `ca_model`'s 33-to-27 population. I recomputed the
dimension-selection objective by hand and reproduced the committed values to all printed
digits for both the chain and the grid.

**Solver and geometry internals are sound where they matter.** The graph Laplacian is PSD
with exactly one zero mode and `||L @ ones|| = 0.0`; the KKT solve makes the neutrality
constraint exact; the sign chain from `L = D - W` through the negated source to
`1+z = exp(Phi_obs - Phi_emit)` is correct and was checked against a deliberately flipped
source, which produces a blueshift as it should. The local-metric design basis is
Frobenius-orthonormal so the ridge penalty is an O(D) invariant, and the fitted tensor
obeys `h -> R^T h R` to 1.126e-13 over 30 random frames with scalar diagnostics invariant
to 2.4e-12. The Regge boundary convention is Gauss-Bonnet consistent (total deficit equals
`2*pi` to 2.5e-14). The identically zero placeholder source in `grid_2d` is correct physics
- SU(2) symmetry makes every single-site reduced state exactly `I/2` - and the artifact
gates it honestly as `placeholder_source_degeneracy`.

**Coverage of the original review was not shallow.** `git diff --name-only c03800e4
adbfab58` returns 74 paths and every one is covered by the initial review's
`files_reviewed` globs. Four findings across 9397 inserted lines invites the hypothesis
that the review was thin; I tested that hypothesis where it mattered and the answer is
that the work was strong. The two defects this round adds were both outside the diff's
central subject matter.

## Open scientific questions

These are limitations of the science, not defects, and most are already disclosed by the
project. I list them because a referee should see them in one place.

1. **The KMS source law is a family-dependent feasibility result.** The declared
   `remaining_requirements` - a state-independent localization prescription, covariant
   conservation, refinement behaviour and an independent clock observable - are all still
   open, and the artifact says so. Nothing here derives a family-independent first-order
   law, a gravitational source, covariance, or a continuum limit.

2. **The local energy split is a declared convention, not a stress tensor.** `sum_i h_i =
   H` holds exactly, and the modular-sum identity is exactly invariant under any endpoint
   split whose per-edge weights sum to one, so the per-cell densities have no tight
   independent ground truth (RTV-004). What "localized" means in this framework is
   currently a choice, not a measurement.

3. **No discrete continuity law exists.** The finite-chain result is a global invariant
   plus demonstrated profile spreading. The documents now say this correctly at row 7,
   `framework.md:488-489` and `framework.tex:357`; row 15 is the residual (SLAW-004).
   Whether a discrete current can be defined for which this decomposition is conserved is
   an open and, I think, the most interesting question in this file.

4. **The parameter sweep is thinner than its row suggests.** The `N` leg varies only at
   Heisenberg / beta=1.0 over {3,5,7} and the physical observable moves about 4%, against a
   115x range across beta. All five Ising configurations have `[H,V] = 0` exactly, so they
   exercise no noncommutative Kubo-Mori structure; the repository does label this family a
   commuting control in four places. The whole sweep's discriminating power effectively
   rests on the two beta=0.3 configurations, where the tightest margins are 1.8x
   (`minimum_signal_to_floor`), 3.9x and 9.7x; every configuration at beta >= 1.0 clears
   every gate by two to four orders of magnitude. Three gate legs - `significance_ratio`,
   `susceptibility_relative_error` and `normalized_rmse` - are never exercised anywhere in
   the sweep, with minimum margins of 7.4e6x, 561x and 43x respectively. This is not a
   defect; it is a statement about how much the sweep currently proves.

5. **Extrapolation accuracy is controlled by comparison, not by convergence order.** The
   Richardson `significance_ratio` certifies nonzero-ness against its own iterate spread;
   the accuracy control is the exact Kubo-Mori comparison at 1e-6. That is a legitimate and
   strong control for the analytic family in use, and I verified the error estimate is
   conservative in every committed run (actual deviation is 0.047x to 0.43x of the reported
   `total_error`). But it does not generalize to a family whose leading correction the
   extrapolant does not model.

6. **The curvature diagnostic is a proxy, and is labelled as one.** On an
   embedding-Delaunay complex the interior deficits are identically zero and the total is
   `2*pi` by Gauss-Bonnet; the boundary entries do vary with the inferred connectivity and
   embedding, so the field is not inert, but no deficit is cited as curvature evidence
   anywhere and `CLAIMS_MATRIX.md` C13 records Einstein/Regge closure as unimplemented.

7. **Non-geometric controls are incomplete and one of them fails.** `FALSIFICATION_MATRIX`
   declares four control families; only Bell, Petersen/expander and uniform fixtures exist,
   with no random-regular and no shuffled-MI control. The disjoint-Bell control receives a
   false `D*=1 geometric_candidate` status, disclosed in the matrix, in C08 and in
   `REPRODUCIBILITY.md`. A reader who reaches only `README.md` or `framework.tex` learns
   that controls remain open but not that a specific one currently fails; I considered
   raising that as a finding and did not, because both documents state the non-passing
   status plainly and the direction of any imprecision is understatement.

8. **Cross-platform artifact reproducibility is a policy, not a proof.** The contract is
   semantic with a documented factor-100 allowance on `significance_ratio` and a 1e-3 / 5e-9
   default; the exact Linux artifact diff that failed at `adbfab58` was never recovered
   publicly. Regeneration reproduced bit-identically on win32 in this round, but CI has no
   post-regeneration cleanliness gate, so that property is observed rather than enforced.

9. **The public `Simulator.run` path is ergonomically narrow.** A localized excitation
   through the fully automatic route still requires the stage-wise
   `Simulator.run_pi_time(state, ...)` entry point, which is public and documented but is
   not the API the class docstring leads with. This is a usability observation, not a
   defect; the physics is reachable and regression-tested.
