# Independent re-review: POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-1

```yaml
artifact_schema_version: 1
review_id: "POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-1"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: gpt-5.6-terra
reviewer_model_version: unknown
reviewer_operator: Richard Fuoco
review_date: 2026-08-10
commit_reviewed: b25988cb7b1accdf6b90216ac567749bf0ed8df8
prior_review_ref: "18aa5b3feea4f35e8bf89f1ae4cd34ce57fc279e:reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1.md"
builder_response_ref: "reviews/codex/POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-1.md at b25988cb7b1accdf6b90216ac567749bf0ed8df8"
context_hash: 66ebc16791414053185a5c6db381dc462034e9bb
context_hash_method: "git rev-parse b25988cb7b1accdf6b90216ac567749bf0ed8df8^{tree}"
files_reviewed:
  - "Complete remediation diff: git diff --name-only 18aa5b3feea4f35e8bf89f1ae4cd34ce57fc279e b25988cb7b1accdf6b90216ac567749bf0ed8df8 (23 paths)"
  - ".github/workflows/ci.yml"
  - "scripts/check_validation_artifacts.py"
  - "popgp/{simulator,config}.py"
  - "popgp/geometry/regge.py"
  - "tests/unit/{test_simulator,test_validation_artifact_contract,test_claim_wording,test_regge_proxy}.py"
  - "tests/scientific/test_many_body_source_law.py"
  - "README.md; docs/framework.{md,tex}; docs/scientific_hardening/{CLAIMS_MATRIX,DECISIONS,FALSIFICATION_MATRIX,PROJECT_PLAN,REPRODUCIBILITY,THEORY_CODE_GAP}.md"
  - "examples/physics_qg/{chain_1d,grid_2d,gravity_well,source_law,source_law_many_body,ca_model}/ including regenerated validation artifacts and required visuals"
  - "reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1.md"
  - "reviews/codex/POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-1.md"
access_level: public-repository-only
independence_statement: "This re-review necessarily inspected the supplied immutable prior review and builder response to verify their stated dispositions. I am a separate agent/model variant and did not accept builder conclusions as evidence. I share the repository, human operator, and orchestrated Codex task; therefore this is process independence, not external scientific validation. Remote Actions evidence was read only through unauthenticated public GitHub API endpoints; no authenticated CLI, raw private log, final-label, secret-seed, or private-evaluator access was used."

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: "CI-001, SLAW-001, and SLAW-002 are independently verified resolved, and all five requested test obligations have direct passing evidence. The exact handoff SHA has a public green Ubuntu Actions run. SLAW-003 remains unresolved: the corrected framework, README, example, plot, and regression test distinguish global conservation from spreading, but the unchanged falsification matrix still states that the audited energy profile 'evolves conservatively' despite no implemented discrete current or local conservation law. This residual claim wording is a blocking documentation defect under the prior finding's required action to correct all affected wording."

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "CI-001"
    outcome: verified-resolved
    resolution_reasoning: "The byte-exact Linux artifact gate was replaced by an explicit semantic validation contract and required-visual smoke check. The contract compares keys, JSON types, list lengths, strings, integers, booleans, nulls, configuration floats, check identities/criteria, and all gate booleans exactly; it rejects non-finite values. Only named numeric diagnostics receive bounded tolerances. The exact frozen handoff SHA passed every Ubuntu CI step, including regeneration and the new contract step."
    verification: "confirmed-by-public-execution-and-local-replay"
    residual_risk: "Artifact comparison is intentionally semantic rather than byte-exact. The widest allowance is significance_ratio at factor 100, but it is limited to that named finite diagnostic; its check criterion, configured threshold, check passed boolean, and overall_pass remain exact. Public raw job logs returned HTTP 403, so the numerical field that caused the intervening pre-f40 failure cannot be independently identified under the public-only boundary."
    blocking: false
    evidence: "Public unauthenticated GitHub API: run 31376803027, head_sha b25988cb7b1accdf6b90216ac567749bf0ed8df8, conclusion success; job 93417770201, steps 5-10 (locked sync, lint, TeX, pytest, example regeneration, validation-contract/visual-output check) all success. Predecessor run 31376568187, head_sha f40fc6d7aa3710f335ee561d02fa8d862f5d9ec6, conclusion success; its job 93417034274 likewise passed all steps. Local `uv run python scripts/check_validation_artifacts.py` passed after all six examples regenerated, and `git diff --exit-code -- 'examples/physics_qg/*/results/validation.json'` passed."

  - finding_id: "SLAW-001"
    outcome: verified-resolved
    resolution_reasoning: "negative_kms_energy_density_candidate now requires ExactBackend, a complete nonempty site partition, and a supplied tensor whose trace distance from finite_gibbs_state(H, beta_kms) is at most 1e-10 before constructing -beta_kms Delta<h_i>. The reference is checked at the runtime boundary where the supplied state is available. Thus the stated modular-energy identity has its KMS premise enforced instead of merely documented."
    verification: "confirmed-by-execution"
    residual_risk: "This dense Gibbs-state/trace-norm validation is appropriate only for the present small exact backend, and it deliberately excludes alternate KMS representations. It is not a scalable large-system validation strategy."
    blocking: false
    evidence: "`uv run pytest -q tests/unit/test_simulator.py tests/unit/test_validation_artifact_contract.py tests/unit/test_claim_wording.py tests/unit/test_regge_proxy.py tests/scientific/test_many_body_source_law.py` passed 55 tests. Independent counterexample: config beta=1.3, beta_kms=0.5 with the beta=1.3 prepared reference raised ValueError with trace distance 2.741193e-01 > 1e-10. A faithful diagonal non-Gibbs two-qubit reference raised ValueError with trace distance 2.627858e-01 > 1e-10. A matched KMS source with source_scale=1.7 passed the modular-sum test in test_simulator.py; an independent public-path calculation agreed to 8.673617379884035e-19."

  - finding_id: "SLAW-002"
    outcome: verified-resolved
    resolution_reasoning: "Simulator.run now has a keyword-only reference_state argument and forwards it to run_pi_time, which forwards it to the selected source. The API documentation in the method and PiTimeConfig names this route, and the public-path regression uses the KMS candidate with a nonzero injected excitation."
    verification: "confirmed-by-execution-and-inspection"
    residual_risk: "A caller must still supply a reference for reference-dependent modes; omitting it remains an explicit ValueError. The full run API constructs its state internally, so custom-state experiments may still use the documented stagewise interfaces or a prepared-state hook."
    blocking: false
    evidence: "popgp/simulator.py:1077-1104 defines and propagates run(..., reference_state=...). Independent execution set the KMS candidate, supplied an excited state via the test's public-path preparation hook and reference_state, then obtained source_model=negative_kms_energy_density_candidate and source sum -0.0027299844970400684 versus -Delta<K_reference>=-0.0027299844970400675. The regression test test_run_propagates_reference_state_to_kms_candidate passed."

  - finding_id: "SLAW-003"
    outcome: unresolved
    resolution_reasoning: "The remediation correctly changed framework.md/.tex, README, reproducibility material, the example check and its plot title to say global energy is conserved while the profile spreads; test_claim_wording.py protects those specific strings. However, docs/scientific_hardening/FALSIFICATION_MATRIX.md:7 remains unchanged and states that 'the audited energy profile evolves conservatively' and treats 'failure to sum/conserve' as its failure condition. The code checks only global conservation_drift=np.ptp(evolved_total_energy) plus profile spreading; it still implements no discrete current/divergence. In this context, the unchanged wording continues to imply an unestablished conservation property of the profile and falls short of the prior required action to correct all affected wording."
    verification: "confirmed-by-inspection-and-execution"
    residual_risk: "Readers of the falsification matrix can infer local conservative evolution from an exact-chain global-energy check, overstating the evidence for a localized source law."
    blocking: true
    evidence: "The regenerated source_law_many_body example reported energy-conservation drift 0.000e+00 and t=1 endpoint fraction 0.120782; its plot title visibly reads 'Global energy conserved; profile spreads.' examples/physics_qg/source_law_many_body/__main__.py:571-586 gates only global drift and endpoint spreading. docs/framework.md:486-489 and docs/framework.tex:357 explicitly disclaim a discrete continuity current/local law, while FALSIFICATION_MATRIX.md:7 retains the conflicting 'profile evolves conservatively' wording."

prior_requested_test_results:
  - requested_test_id: "TST-CI-001"
    outcome: verified-satisfied
    evidence: "Exact handoff Actions run 31376803027 is public success at b25988cb7b1accdf6b90216ac567749bf0ed8df8, with the regeneration and validation-contract steps both green. tests/unit/test_validation_artifact_contract.py includes representative Linux/Windows drift acceptance, excessive-drift rejection, schema/type/gate rejection, and f40 adds observed significance-ratio drift coverage."
    verification: "confirmed-by-public-execution-and-local-execution"
    blocking: false

  - requested_test_id: "TST-SLAW-001"
    outcome: verified-satisfied
    evidence: "tests/unit/test_simulator.py tests matched KMS acceptance and scaled modular-sum identity, beta mismatch rejection, and faithful non-Gibbs rejection. Independent executions reproduced both rejections and the public-path identity agreement."
    verification: "confirmed-by-execution"
    blocking: false

  - requested_test_id: "TST-SLAW-002"
    outcome: verified-satisfied
    evidence: "test_run_propagates_reference_state_to_kms_candidate passed, and independent execution of Simulator.run(reference_state=...) with the selected KMS candidate produced a nonzero source and the expected modular-energy sum."
    verification: "confirmed-by-execution"
    blocking: false

  - requested_test_id: "TST-SLAW-003"
    outcome: verified-satisfied
    evidence: "tests/unit/test_claim_wording.py passed and protects corrected framework/plot wording; tests/scientific/test_many_body_source_law.py passed with global conservation and spreading behavior. This requested regression is satisfied, but it does not cover the remaining FALSIFICATION_MATRIX.md wording; that separate residual keeps SLAW-003 open."
    verification: "confirmed-by-execution-and-inspection"
    blocking: false

  - requested_test_id: "TST-CI-002"
    outcome: verified-satisfied
    evidence: "scripts/check_validation_artifacts.py requires every declared visual to be tracked, present, and nonempty; its visual-path containment check rejects path escapes. Its unit tests passed, and the checker passed after regeneration of all six documented examples."
    verification: "confirmed-by-execution"
    blocking: false

predictions:
  experiment_id: "IR-SLAW-REREVIEW-CHECK-01"
  predicted_outcome: "A correction of the remaining FALSIFICATION_MATRIX.md wording to explicitly limit conservation to the global Hamiltonian expectation, followed by a regression that includes that document, will resolve the sole blocker without altering the source-law calculation."
  predicted_failure_mode: "Until then, a wording-only review or future documentation change can continue to present profile evolution as conservative despite the absence of a discrete continuity current."
  confidence_statement: "High: the unresolved phrase is present in the frozen handoff and the implemented check demonstrably measures only the global sum and endpoint spreading."

recommendation:
  approve: false
  blocking_findings: 1
  rationale: "Changes requested. CI-001, SLAW-001, and SLAW-002 are resolved and all requested tests are satisfied, but SLAW-003 remains open because the falsification matrix retains an unqualified profile-conservation implication unsupported by the code or a local continuity equation."
```

## Review method and evidence scope

I re-read the required governance workflow, reviewer identity, and independent-review
template before acting. The worktree was clean on
`review/source-law-linear-response-rereview-1`; `HEAD` equaled the requested handoff
`b25988cb7b1accdf6b90216ac567749bf0ed8df8`; and the stated context command produced
tree hash `66ebc16791414053185a5c6db381dc462034e9bb`. I read the complete 23-path
remediation diff from immutable review commit
`18aa5b3feea4f35e8bf89f1ae4cd34ce57fc279e`, all affected source, tests,
documentation, validation artifacts, workflow, prior review, and builder response.
No implementation or test file was modified. No audit fan-out was used.

Public Actions evidence was obtained with unauthenticated GitHub REST requests only.
The exact handoff push run [31376803027](https://github.com/whact2025/POPGP/actions/runs/31376803027)
is successful at `b25988c`; the predecessor remediation run
[31376568187](https://github.com/whact2025/POPGP/actions/runs/31376568187) is successful
at `f40fc6d`; and the intervening run
[31376275587](https://github.com/whact2025/POPGP/actions/runs/31376275587) failed at
`d7a4d87` solely in step 10, `Check committed validation contracts and visual outputs`.
The latter's setup, locked sync, lint, TeX check, tests, and example regeneration all
passed. Commit `f40fc6d` changes the `significance_ratio` tolerance from 0.5 to 0.99
and adds an observed-drift regression; the later successful runs establish the amended
contract passes. The public raw-log endpoints for jobs 93416115236, 93417034274, and
93417770201 each returned HTTP 403, so I do not claim public independent access to the
precise drift printed by the failed runner.

## Executed quality, API, and mutation record

| Command / check | Outcome |
|---|---|
| `uv sync --frozen` | Passed; fresh locked 55-package environment installed. |
| `uv run ruff check .` | Passed: `All checks passed!` |
| `uv run python scripts/check_tex.py` | Exit 0; 652 lines, balanced braces and environments, no Markdown remnants; six wide-equation notices are advisory. |
| `uv run pytest -q` | Passed: `99 passed in 7.95s`. |
| Focused source/contract/claim/regge suite | Passed: `55 passed in 3.81s`. |
| All six README examples, then `uv run python scripts/check_validation_artifacts.py` | All exited 0; checker reported `Validation artifact contracts and required visual outputs are valid.` |
| `git diff --exit-code -- 'examples/physics_qg/*/results/validation.json'` after replay | Passed. Worktree and index remained clean. |
| `uv lock --check`; `uv sync --frozen --dry-run`; `git diff --check 18aa... b259...` | All passed. |
| Comparator mutations | Changing a check `passed`, check criterion, `overall_pass`, stable config threshold, or array length each failed. A factor-99 `significance_ratio` change passed; a factor-101 change and sign reversal failed, exactly matching the documented factor-100 policy. |
| KMS runtime counterexamples | beta mismatch and faithful non-Gibbs references both rejected; a matched beta=0.5 Gibbs reference accepted. Configuration alone cannot reject a reference/source pairing because reference_state is supplied later; the runtime boundary has the required data and rejects the invalid pair. |
| Public full-pipeline reference path | `Simulator.run(reference_state=...)` with a nonzero excitation propagated the KMS reference and produced sum(source) equal to `-Delta<K_reference>` within `8.673617379884035e-19`. |

The factor-100 `significance_ratio` exception is not a new blocker in this candidate.
It accepts only a finite numeric field whose minimum required margin remains the exact
configuration value 10.0, whereas the observed values are orders of magnitude higher.
The comparator's exact string/boolean/config protections were additionally exercised
above. Its scope is nevertheless a reviewable reproducibility policy rather than
byte-for-byte numerical identity.
