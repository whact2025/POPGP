# Independent re-review: POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-2

```yaml
artifact_schema_version: 1
review_id: "POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-2"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: gpt-5.6-terra
reviewer_model_version: unknown
reviewer_operator: Richard Fuoco
review_date: 2026-08-10
commit_reviewed: b519c7dafd4488d587a1edcbe14891f0c83aedb5
prior_review_ref: "reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-1.md"
builder_response_ref: "reviews/codex/POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-2.md"
context_hash: 03966666af543c398307d5832ccb6e82360c1344
context_hash_method: "git rev-parse b519c7dafd4488d587a1edcbe14891f0c83aedb5^{tree}"
files_reviewed:
  - "Complete round-2 remediation diff: git diff --name-only 088724989e69d427fdb63ad930787b5c9a3f47fb b519c7dafd4488d587a1edcbe14891f0c83aedb5 (3 paths)"
  - "Exact fix diff d527cb8bfe02b754c2f52724bce852ccfc28d584^..d527cb8bfe02b754c2f52724bce852ccfc28d584"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "tests/unit/test_claim_wording.py"
  - "README.md; docs/; examples/; popgp/; tests/ repository-wide conservation-language audit"
  - ".github/workflows/ci.yml; scripts/check_validation_artifacts.py; all six README examples and validation artifacts"
  - "reviews/independent_reviewer/POPGP-REVIEW-SOURCE-LAW-1-REREVIEW-1.md at immutable commit 088724989e69d427fdb63ad930787b5c9a3f47fb"
  - "reviews/codex/POPGP-REVIEW-SOURCE-LAW-1-RESPONSE-2.md"
access_level: public-repository-only
independence_statement: "This re-review inspected the supplied immutable prior re-review and builder response to verify their stated dispositions. I am a separate agent/model variant and did not accept builder conclusions as evidence. I share the repository, human operator, and orchestrated Codex task; therefore this is process independence, not external scientific validation. Remote Actions evidence was read only through unauthenticated public GitHub API endpoints; no authenticated CLI, raw private log, final-label, secret-seed, or private-evaluator access was used."

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: "The sole open blocker, SLAW-003, is verified resolved. Exact fix commit d527cb8b changes the falsification matrix to distinguish the modular-density sum, global Hamiltonian conservation, and profile spreading, explicitly disclaiming a discrete continuity current/local conservation law. The claim-wording regression now covers that document. The complete local quality replay and both exact public Actions runs succeeded. No regression or new finding was identified."

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "CI-001"
    outcome: verified-resolved
    evidence: "No regression is present in the two-file fix. Public exact-SHA handoff run 31378004579 at b519c7dafd4488d587a1edcbe14891f0c83aedb5 completed successfully; its locked sync, lint, TeX validation, test, example regeneration, and validation-contract/visual-output steps all succeeded. Local replay also passed the semantic artifact checker after all six examples."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The comparator policy was unchanged in round 2. Public raw job-log endpoints returned HTTP 403, but public run/job metadata establishes all steps succeeded."

  - finding_id: "SLAW-001"
    outcome: verified-resolved
    evidence: "No source or runtime-validation code changed in this two-file documentation/test remediation. The full local suite passed 99 tests, retaining the KMS matched-reference identity and beta-mismatch/non-Gibbs rejection regressions verified in round 1."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No regression found."

  - finding_id: "SLAW-002"
    outcome: verified-resolved
    evidence: "No Simulator.run or reference-state propagation code changed in this two-file remediation. The full local suite passed 99 tests, retaining the public reference_state propagation regression verified in round 1."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "No regression found."

  - finding_id: "SLAW-003"
    outcome: verified-resolved
    evidence: "Exact commit d527cb8bfe02b754c2f52724bce852ccfc28d584 replaces the residual FALSIFICATION_MATRIX.md:7 phrase 'the audited energy profile evolves conservatively' with 'the global Hamiltonian expectation is conserved while the local-energy profile spreads.' It separately names failure of the modular-density sum, global-energy conservation, or profile spreading, and states 'no discrete continuity current or local conservation law is implemented.' tests/unit/test_claim_wording.py now reads this matrix, rejects the residual phrase, and requires the global/no-local-law wording. `uv run pytest -q tests/unit/test_claim_wording.py tests/scientific/test_many_body_source_law.py` passed 19 tests. Repository-wide searches found no remaining statement that the changing profile itself is locally conserved; the remaining matches either state global conservation plus spreading, explicitly state the local-law gap, or are prospective requirements."
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The underlying physical limitation remains correctly documented: this is a finite-chain global-energy and profile-spreading diagnostic, not a local continuity or covariant-conservation demonstration."

prior_requested_test_results:
  - requested_test_id: "TST-CI-001"
    outcome: verified-satisfied
    evidence: "Public exact handoff CI run 31378004579 succeeded at b519c7dafd4488d587a1edcbe14891f0c83aedb5, including regeneration and the validation-contract/visual-output step; local six-example replay and `uv run python scripts/check_validation_artifacts.py` also passed."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No CI-contract regression in the round-2 diff."

  - requested_test_id: "TST-SLAW-001"
    outcome: verified-satisfied
    evidence: "Full `uv run pytest -q` passed 99 tests, retaining the matched-KMS acceptance and invalid reference rejection regressions."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No affected source code changed."

  - requested_test_id: "TST-SLAW-002"
    outcome: verified-satisfied
    evidence: "Full `uv run pytest -q` passed 99 tests, retaining test_run_propagates_reference_state_to_kms_candidate."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No affected API code changed."

  - requested_test_id: "TST-SLAW-003"
    outcome: verified-satisfied
    evidence: "The corrected claim regression now explicitly includes FALSIFICATION_MATRIX.md, rejects 'audited energy profile evolves conservatively', and requires both global-Hamiltonian conservation and the no-local-law disclaimer. It and the scientific many-body source-law tests passed together: 19 passed in 8.78s."
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The requested wording-regression alternative is now complete across the residual affected document."

  - requested_test_id: "TST-CI-002"
    outcome: verified-satisfied
    evidence: "All six examples regenerated successfully; `uv run python scripts/check_validation_artifacts.py` reported `Validation artifact contracts and required visual outputs are valid.`"
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "No visual-artifact contract change in round 2."

predictions:
  experiment_id: "IR-SLAW-REREVIEW2-CLAIM-01"
  predicted_outcome: "A future edit that restores the exact residual phrase to FALSIFICATION_MATRIX.md will fail tests/unit/test_claim_wording.py; the present wording will continue to state only global conservation and profile spreading."
  predicted_failure_mode: "A different unqualified conservation phrase outside the three documents covered by the unit test would require human repository-wide claim review; current targeted and broad searches found none that asserts the local profile is conserved."
  confidence_statement: "High for the protected phrase and current frozen tree, based on direct inspection, passing regression execution, and repository-wide search."

recommendation:
  approve: true
  blocking_findings: 0
  rationale: "Approve. Every carried prior finding and requested test is verified resolved or satisfied; SLAW-003's sole residual claim has been corrected and guarded, exact fix and handoff CI runs are successful, the complete local replay is green, and no new blocker was found."
```

## Review method and exact CI evidence

I re-read the required governance workflow, reviewer identity, and independent-review
template before review. The worktree was clean on
`review/source-law-linear-response-rereview-2`; its `HEAD` exactly matched
`b519c7dafd4488d587a1edcbe14891f0c83aedb5`; and the stated context command produced
tree hash `03966666af543c398307d5832ccb6e82360c1344`. I inspected the entire three-path
round-2 diff from prior review commit
`088724989e69d427fdb63ad930787b5c9a3f47fb`, the exact two-path fix commit
`d527cb8bfe02b754c2f52724bce852ccfc28d584`, the prior re-review, builder response,
claim regression, complete current claim language, and quality workflow. No
implementation, documentation, or test file was changed by this review. No audit
fan-out was used.

Unauthenticated public GitHub API metadata reports successful Ubuntu CI for both
[fix run 31377797933](https://github.com/whact2025/POPGP/actions/runs/31377797933) at
`d527cb8bfe02b754c2f52724bce852ccfc28d584` (job 93420901485) and
[handoff run 31378004579](https://github.com/whact2025/POPGP/actions/runs/31378004579)
at `b519c7dafd4488d587a1edcbe14891f0c83aedb5` (job 93421535309). Each has successful
locked sync, lint, manuscript-source validation, tests, six-example regeneration, and
validation-contract/visual-output steps. Public raw job-log endpoints returned HTTP
403, recorded as unavailable rather than inferred.

## Executed quality and claim audit

| Command / check | Outcome |
|---|---|
| `uv sync --frozen` | Passed. |
| `uv run pytest -q tests/unit/test_claim_wording.py tests/scientific/test_many_body_source_law.py` | Passed: `19 passed in 8.78s`. |
| `uv run ruff check .` | Passed: `All checks passed!` |
| `uv run python scripts/check_tex.py` | Exit 0; balanced structure/no Markdown remnants; six wide-equation notices advisory. |
| `uv run pytest -q` | Passed: `99 passed in 5.67s`. |
| Six README examples, then `uv run python scripts/check_validation_artifacts.py` | All exited 0; checker reported validation contracts and required visual outputs valid. |
| `git diff --exit-code -- 'examples/physics_qg/*/results/validation.json'` after replay | Passed; worktree/index remained clean. |
| `uv lock --check`; `uv sync --frozen --dry-run`; `git diff --check 0887249... b519c7d...` | All passed. |
| Repository-wide conservation-language searches across README, docs, examples, popgp, and tests | No residual assertion that the changing local-energy profile is conserved. Remaining relevant statements qualify global Hamiltonian/energy conservation, profile spreading, or the absence of local/covariant conservation. |

The new FALSIFICATION_MATRIX statement matches the actual many-body check:
`conservation_drift=np.ptp(evolved_total_energy)` checks a global sum while endpoint
fractions demonstrate spreading. It neither claims nor introduces a local continuity
current. This resolves the limited claim defect without elevating the finite diagnostic
to a physical conservation law.
