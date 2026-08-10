# Independent review: POPGP-REVIEW-SOURCE-LAW-1

```yaml
review_id: "POPGP-REVIEW-SOURCE-LAW-1"
review_kind: initial
reviewer_seat: independent-reviewer
reviewer_model_identity: gpt-5.6-terra
reviewer_model_version: unknown
reviewer_operator: Richard Fuoco
review_date: 2026-08-10
commit_reviewed: adbfab58c20dc28f4cf05b601f81480207d74415
prior_review_ref: ""
builder_response_ref: ""
context_hash: 31506d0eccf2f753bbcae5ef1bc0ac4cd2ac46b5
context_hash_method: "git rev-parse adbfab58c20dc28f4cf05b601f81480207d74415^{tree}"
files_reviewed:
  - ".github/workflows/ci.yml"
  - "LICENSE"
  - "README.md"
  - "docs/framework.md"
  - "docs/framework.tex"
  - "docs/playbook.md"
  - "docs/questions_engine.md"
  - "docs/scientific_hardening/{CLAIMS_MATRIX,DECISIONS,FALSIFICATION_MATRIX,PROJECT_PLAN,REPRODUCIBILITY,THEORY_CODE_GAP}.md"
  - "docs/{simulation_engine_whitepaper,simulator_analysis_and_plan,validation_report_v0.12}.md"
  - "examples/physics_qg/{ca_model,chain_1d,grid_2d,gravity_well,source_law,source_law_many_body}/"
  - "popgp/{__init__,backend,capacity,coarse_grain,config,diagnostics,information,renderer,simulator}.py"
  - "popgp/geometry/{__init__,closure,local_metric,regge}.py"
  - "pyproject.toml"
  - "scripts/check_tex.py"
  - "tests/scientific/{test_many_body_source_law,test_source_law_controls,test_source_law_scaling,test_topology_recovery}.py"
  - "tests/unit/{test_backend,test_closure,test_diagnostics,test_information,test_local_metric,test_regge_proxy,test_simulator}.py"
  - "uv.lock"
access_level: public-repository-only
independence_statement: "I am a separate agent/model variant with no prior-review content supplied. I share the repository, human operator, and orchestrated Codex task; therefore this is process independence, not external scientific validation. I did not inspect prior reviewer artifacts or accept builder conclusions."

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: "The frozen diff is substantial (74 changed paths). Local Windows checks, test execution, and structured-artifact regeneration succeeded, but the exact frozen SHA has failing public GitHub Actions runs. The KMS-density candidate also accepts inputs for which its documented modular-sum identity is false, and documentation overstates global-energy conservation as conservation of a changing local profile."

findings:
  - id: "CI-001"
    severity: high
    category: governance
    location: ".github/workflows/ci.yml:25-34; public Actions runs 31343184026 and 31343185458"
    evidence: "Unauthenticated public GitHub API reports both runs with head_sha adbfab58c20dc28f4cf05b601f81480207d74415 and conclusion failure. In each run, setup, sync, lint, manuscript-source validation, tests, and example regeneration succeeded; step 10, named 'Check committed structured artifacts', has conclusion failure. The workflow executes 'git diff --exit-code -- examples/physics_qg/*/results/validation.json' after regeneration. Public raw job-log access returned HTTP 403, so the exact artifact diff was not available under this review's public-only boundary."
    finding: "The frozen candidate does not pass its required GitHub Actions CI gate. The failure is specifically a Linux regeneration mismatch in a committed validation.json artifact, despite clean local Windows replay."
    failure_scenario: "On GitHub Actions Ubuntu for the frozen SHA, all six examples regenerate successfully, then the structured-artifact git diff exits nonzero. This occurs for both the push and pull_request event runs."
    consequence: "The published reproducibility claim is not established across the only configured CI platform, and the candidate cannot clear its declared quality gate."
    required_action: "Recover the CI artifact diff using authorized maintainer access, make the JSON generation and committed artifacts platform-stable (or deliberately scope the policy), add a regression that catches the cause, and rerun CI on a new frozen remediation SHA until the structured-artifact check passes."
    verification: confirmed-by-execution
    blocking: true

  - id: "SLAW-001"
    severity: high
    category: science
    location: "popgp/simulator.py:976-1006; docs/framework.md:472-485; README.md:163-167"
    evidence: "The candidate requires only reference_state is not None, selects beta_kms independently (lines 992-996), and computes -beta_kms*Tr[(state-reference_state) h_cell] (lines 998-1006). In contrast, the documented identity requires a faithful KMS state with K_sigma=beta H+log Z. Direct execution with a beta=1.3 prepared reference but beta_kms=0.5 was accepted and produced sum_source=-0.003537452571797346 versus -Delta<K_reference>=-0.009197376686673116 (ratio 0.3846153846153839=0.5/1.3). A distinct accepted faithful non-Gibbs reference produced sum_source=4.5102810375396986e-18 versus -Delta<K_reference>=-0.01791759469228053."
    finding: "negative_kms_energy_density_candidate permits reference/beta combinations that violate its stated KMS premise, while repository claims say its density sums to minus the global modular-energy change. That statement is false for accepted inputs."
    failure_scenario: "A caller passes any faithful non-Gibbs reference, or a Gibbs reference whose temperature differs from cfg_time.beta_kms. The code emits a source labelled KMS energy density, but its sum is only -beta_kms Delta<H>, not -Delta<K_reference>."
    consequence: "A central source-law invariant can be reported as satisfied by an invalidly parameterized calculation, undermining the candidate's physical interpretation and its documented validation boundary."
    required_action: "At the runtime boundary, validate that reference_state is the Gibbs/KMS state of this backend Hamiltonian at beta_kms (within a recorded tolerance), or relabel the source and restrict every modular-sum claim to validated KMS inputs. Add accepted matched-KMS and rejected beta-mismatch/non-Gibbs regression tests. Configuration construction alone cannot validate this because the reference state is supplied later."
    verification: confirmed-by-execution
    blocking: true

  - id: "SLAW-002"
    severity: medium
    category: code
    location: "popgp/simulator.py:832-855, 976-980, 1029-1043; examples/physics_qg/source_law_many_body/__main__.py:388-415"
    evidence: "run_pi_time exposes reference_state, but the primary full-pipeline entry point calls run_pi_time(state, pi_res, pi_loc, pi_geom) with no reference. Direct execution of Simulator(config).run() with the KMS candidate selected reaches Pi_time then raises ValueError: Source model 'negative_kms_energy_density_candidate' requires a reference_state. The stage-wise public path does work when given a reference: run_pi_time completed with source norm 0.0018151979236086676 and residual 2.9193036915438557e-19. The documented example instead calls the private _compute_source_term and _solve_clock_constraint separately."
    finding: "Reference-dependent source modes are not usable through the advertised complete Simulator.run() path, and the example does not test the public end-to-end integration that would be needed to support that path."
    failure_scenario: "A user sets PiTimeConfig.source_model to the advertised KMS candidate and invokes the documented primary API, Simulator(config).run()."
    consequence: "The candidate has a viable stage-wise route but lacks a complete primary-API route, increasing the risk that the tested source/solver composition diverges from the published pipeline interface."
    required_action: "Either add an explicit reference_state parameter to Simulator.run() and an end-to-end candidate test, or document and test the stage-wise call sequence as the sole supported public API for reference-dependent sources."
    verification: confirmed-by-execution
    blocking: false

  - id: "SLAW-003"
    severity: medium
    category: claim
    location: "docs/framework.md:482-485; examples/physics_qg/source_law_many_body/__main__.py:428-439, 571-585; tests/scientific/test_many_body_source_law.py:245-252"
    evidence: "The documentation states 'A separate finite-chain quench demonstrates conservation of the audited local-energy profile.' The implementation computes only conservation_drift=np.ptp(evolved_total_energy), i.e. the global Hamiltonian expectation. Its own scientific test asserts an initially negligible endpoint profile and then requires endpoint fraction >0.05 at t=1; the regenerated example reports an endpoint fraction 0.120782 at t=1. Thus the profile demonstrably changes and spreads while its global sum is conserved."
    finding: "The source-law documentation calls a changing/spreading local-energy profile 'conserved' without a local continuity equation or current. The actual demonstrated invariant is global energy conservation under the declared local decomposition."
    failure_scenario: "A reader treats profile conservation as evidence of a locally conserved source density, even though the checked profile values move between sites and no discrete divergence/current is calculated."
    consequence: "This overstates the conservation evidence relevant to a candidate source law and conflicts with the repository's stated need for covariant/conservation validation."
    required_action: "Change all affected wording and visual labels to 'globally conserved energy with profile spreading', or implement and test a declared discrete continuity equation/current before claiming local-profile conservation."
    verification: confirmed-by-execution
    blocking: true

requested_tests:
  - id: "TST-CI-001"
    description: "On Ubuntu CI, capture the exact regenerated validation.json diff for the frozen candidate, repair the cross-platform instability, and require the structured-artifact step to pass."
    rationale: "The local Windows replay is clean but the exact GitHub Actions push and PR runs fail after regeneration."
    blocking: true
  - id: "TST-SLAW-001"
    description: "Add matched-KMS acceptance plus beta-mismatch and faithful non-Gibbs rejection tests for negative_kms_energy_density_candidate; assert its sum equals -source_scale*Delta<K_reference> only after KMS validation."
    rationale: "The current public input boundary accepts counterexamples that falsify the documented identity."
    blocking: true
  - id: "TST-SLAW-002"
    description: "Exercise the selected reference-dependent candidate through its declared public API: either a complete Simulator.run(..., reference_state=...) path or a documented stage-wise run_pi_time contract."
    rationale: "The current example uses private helpers while Simulator.run() cannot carry the required reference."
    blocking: false
  - id: "TST-SLAW-003"
    description: "If retaining a local-conservation claim, test a specified lattice continuity equation with current/divergence residuals; otherwise regression-test corrected global-conservation wording and labels."
    rationale: "Global-energy constancy and profile spreading do not establish local conservation."
    blocking: true
  - id: "TST-CI-002"
    description: "Add a stable visual-artifact smoke policy (for example, required nonempty files on the Linux runner) or explicitly state that CI attests structured metrics only."
    rationale: "CI regenerates figures/GIFs but diffs only validation.json, so suppressed image writes would not fail the present final gate."
    blocking: false

prior_finding_results: []
prior_requested_test_results: []

predictions:
  experiment_id: "IR-SLAW-CI-KMS-COUNTEREXAMPLES-01"
  predicted_outcome: "Without remediation, GitHub Actions continues to fail at the structured-artifact diff; a beta_kms/reference mismatch produces sum(source)/(-Delta<K_reference>)=beta_kms/beta_reference when both states otherwise share the configured Hamiltonian."
  predicted_failure_mode: "Faithful non-Gibbs references and beta mismatches are accepted and invalidate the modular-sum claim; an unchanged API raises the missing-reference error from Simulator.run()."
  confidence_statement: "High: each prediction follows from direct execution against the frozen tree and the public CI metadata for the exact SHA."

recommendation:
  approve: false
  blocking_findings: 3
  rationale: "Changes requested. CI-001, SLAW-001, and SLAW-003 are unresolved blocking findings. SLAW-002 and the non-blocking requested tests should also be addressed in the builder response."
```

## Review method and scope

I read the required governance, reviewer-identity, and independent-review-template
documents before review. The reviewed worktree was on
`review/source-law-linear-response-1`, its `HEAD` exactly matched
`adbfab58c20dc28f4cf05b601f81480207d74415`, and the baseline was
`c03800e47ba4988bd125b81acd3a1b6ae07728f7`. The stated context command recomputed
the recorded tree hash `31506d0eccf2f753bbcae5ef1bc0ac4cd2ac46b5`.

I inspected the complete frozen diff: `git diff --name-only` returned 74 paths. This
included every changed source, test, documentation, workflow, lockfile, validation
JSON, and regenerated visual artifact listed in `files_reviewed`. Source-law and
geometry figures were visually inspected; structured artifacts were regenerated and
compared without changes locally. No implementation file was changed by this review.

The review lead authorized two bounded, read-only audit fan-outs: one science/statistics
audit and one CI/reproducibility audit. Neither agent modified files or supplied any
prior-review artifact; their concrete counterexamples and checks were independently
reproduced before inclusion above. No local quality command stalled or timed out.
Public Actions run metadata was read without credentials. Raw public job logs were
unavailable (HTTP 403 from the public job-log endpoint), so the exact Linux artifact
diff is recorded as unavailable rather than guessed.

## Executed quality and counterexample record

| Command / check | Outcome |
|---|---|
| `uv sync` | Passed; CPython 3.11.15 environment created and 55 packages installed. |
| `uv run pytest -q` | Passed: `83 passed in 9.57s`. |
| `uv run ruff check popgp tests examples` | Passed: `All checks passed!` |
| `uv run ruff check .` | Passed: `All checks passed!` |
| `uv run python scripts/check_tex.py` | Exit 0; 652 lines, brace balance 0, matching environments and no Markdown remnants. It reports six advisory wide-equation lines. |
| Six README examples in order | All exited 0: `chain_1d`, `grid_2d`, `gravity_well`, `source_law`, `source_law_many_body`, and `ca_model`. The source-law outputs reproduced the committed reports; the grid/gravity scripts log their disclosed singleton Pi_res inadmissibility. |
| `git diff --exit-code -- 'examples/physics_qg/*/results/validation.json'` after local replay | Passed with no output. |
| `uv lock --check` and `uv sync --frozen --dry-run` | Passed; lock resolves 74 packages and frozen sync would make no changes. |
| Valid KMS stage-wise integration | Passed: source model `negative_kms_energy_density_candidate`, source norm `0.0018151979236086676`, constraint residual `2.9193036915438557e-19`. |
| beta/reference and non-Gibbs source counterexamples | Reproduced the numerical mismatches quoted in SLAW-001. |
| `Simulator(config).run()` with the KMS candidate selected | Reproduced `ValueError: Source model 'negative_kms_energy_density_candidate' requires a reference_state`. |
| Public GitHub Actions API, exact frozen SHA | Push run `31343184026` and PR run `31343185458` both completed with conclusion `failure`; all quality steps through example regeneration passed, while `Check committed structured artifacts` failed. |

The nearby PR1 public run `31342573926` (head `a7a23ba1745f8b1b4595f61e0b4f6713de64103f`, not the review target) also fails at the same structured-artifact step. That corroborates a pre-existing cross-platform reproducibility defect but does not replace the frozen-SHA evidence above.
