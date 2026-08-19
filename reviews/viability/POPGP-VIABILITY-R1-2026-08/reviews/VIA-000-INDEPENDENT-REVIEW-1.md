# VIA-000 formal independent review 1

```yaml
artifact_schema_version: 2
review_id: "POPGP-VIABILITY-R1-2026-08-VIA-000-INDEPENDENT-REVIEW-1"
review_kind: initial
reviewer_seat: independent-reviewer
reviewer_model_identity: "unknown"
reviewer_model_version: "unknown"
reviewer_operator: "fuocor"
reviewer_session_id: "popgp-viability-r1-2026-08-via000-independent-review-session-1"
reviewer_orchestrator_id: "codex-desktop"
review_date: "2026-08-19"
commit_reviewed: "9a29e05f803666bf0e3a28417ea399e3e26769fc"
baseline_commit: "70c867552279b74d5ce1a7bc5c50d5a980cf81e6"
prior_review_ref: ""
builder_response_ref: ""
context_hash: "358fb1af6ca587b6c71ff2ef0fb87e335163eeaf"
context_hash_method: "git rev-parse \"9a29e05f803666bf0e3a28417ea399e3e26769fc^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - "pyproject.toml"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "schemas/viability/requirements-v2.json"
  - "schemas/viability/packet-v2.schema.json"
  - "schemas/viability/independent-review-v2.schema.json"
  - "scripts/check_validation_artifacts.py"
  - "scripts/check_viability_campaign.py"
  - "protocols/POPGP-VIABILITY-R1-2026-08/VIA-000.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/CAMPAIGN.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/README.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/packets/VIA-000.yaml"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/attacks/VIA-000-ATTACK-PLAN-7.md"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/protocol.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/environment.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/run-log.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/raw-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/mutation-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/output-commitment.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/reveal-record.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/statistical-audit.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/claim-diff.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/github-runs.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/linux-attempt-1/command-index.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/linux-attempt-1/platform-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/linux-attempt-2/command-index.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/linux-attempt-2/platform-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/linux-mutation-continuation/command-index.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/linux-mutation-continuation/mutation-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/windows/command-index.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/windows/platform-results.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/windows-mutation-continuation/command-index.json"
  - "reviews/viability/POPGP-VIABILITY-R1-2026-08/receipts/VIA-000/runner/windows-mutation-continuation/mutation-results.json"
  - "reviews/independent_reviewer/POPGP-VIABILITY-R1-2026-08-VIA-000-CLAIM-AUDIT-1.md"
  - "reviews/independent_reviewer/POPGP-VIABILITY-R1-2026-08-VIA-000-CLAIM-AUDIT-1-CLOSURE.json"
access_level: "public-repository-post-reveal-static-evidence-only"
independence_statement: |-
  This is a fresh independent-reviewer task and session, separate from the builder,
  falsifier, runner, statistical-auditor, claim-auditor, custodian, and future
  adjudicator sessions. The same human operator and Codex Desktop orchestrator are
  shared with the builder, so this is internal process separation rather than
  external scientific validation. The runtime exposes neither an exact served-model
  identifier nor a model snapshot; both reviewer model fields are therefore unknown.
  The packet also records the builder model identity as unknown, so model separation
  cannot be established and is declared false.

  The review was static and receipt-only. It bound the scientific review context to
  candidate 9a29e05f803666bf0e3a28417ea399e3e26769fc and its tree, while auditing the
  retained post-reveal campaign evidence at handoff
  82c2e2a0b5eb3cfcb70ce4b1d4d4573203116a5a (tree
  da654bb00a2c70b0b0e2fc1ba8df12dbaaa996e7). The candidate was not rerun after
  reveal. No private custody path, private evaluator, credentials, or local
  POPGP_Codex_Handoff.md file was accessed. Public revealed-manifest bytes were
  hash-checked without reading their field values; the review read the public reveal
  record and audits but did not see a secret seed value or an unrevealed final label.
independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "unknown"
  builder_session_id: "popgp-viability-r1-2026-08-via000-builder-session"
  builder_orchestrator_id: "codex-desktop"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false
hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
summary: |-
  Immutable identity and receipt integrity were independently verified. The scientific
  candidate resolves to tree 358fb1af6ca587b6c71ff2ef0fb87e335163eeaf and the audited
  campaign handoff resolves to tree da654bb00a2c70b0b0e2fc1ba8df12dbaaa996e7.
  The primary protocol and protocol receipt are byte-identical with SHA-256
  66d9cc3325a9758c7443b29df612e2b27b8db14de3b224489dee5e2e04a718f8;
  Plan 7 has SHA-256
  735783112cf90451551cbaec9de4d0bafd15754400b479a0be888fc670a725fb;
  and the recomputed packet-rule SHA-256 is
  e5ac4103188f652a460b4a0cc74d641d694e0043b6f2d62dcf2484b92de8acca.
  Every one of the 296 entries in the runner evidence manifest matches its retained
  byte count and SHA-256, all three aggregate receipt references match, all 34 JSON
  documents below the retained VIA-000 receipt tree strict-parse without duplicate
  keys, and every indexed stdout/stderr stream matches its declared bytes and hash.
  The unmodified handoff passes the authoritative campaign validator.

  Ordering and custody are consistent. raw-results.json has SHA-256
  7a221ea1a11dc1033df3dd7d5732a3693ee620dcf6b429d243afa5be6bff2641,
  exactly the output commitment. Raw generation at 2026-08-19T16:22:43.110961Z and
  output commitment at 2026-08-19T16:22:43.113962Z precede custodian reveal at
  2026-08-19T16:35:04.810794Z. The raw result and commitment are unchanged from
  runner commit bd1a8d28717e493cd62053d32bd38062f293c2cd. The revealed holdout and
  seed repository copies hash to their frozen commitments, and the reveal receipt,
  statistical audit, claim audit, claim-diff, and standalone claim-audit closure all
  retain matching immutable references and hashes.

  Windows satisfies the full frozen protocol: 187 tests pass, all six examples and
  semantic validation complete, the exact pdfTeX 1.40.29 / TeX Live 2026 engine makes
  a nonempty 535368-byte PDF, Git diff and postflight exit zero, and the complete
  continuation rejects all ten mutation families. The retained initial Windows
  mutation-harness interruption is not hidden; the independent continuation supplies
  the complete ten-family evidence.

  Linux supplies a valid negative result. Two complete retained attempts run the
  exact locked protocol, pass 187 tests, all six examples, semantic validation, and
  two PDF passes with the exact TeX engine. Both attempts nevertheless produce the
  same 16 tracked JSON/PNG drift paths, so Git diff exits one. Postflight also exits
  one because the locked environment contains
  _cuda_bindings_redirector.pth in addition to the only frozen expected surface,
  _virtualenv.pth. The mutation continuation rejects G01 through G09. For G10 it
  reproduces the historical editable self-cleaning carrier, then removes that carrier
  with non-editable sync and observes no injected marker, but the complete frozen
  oracle still fails because the unrelated baseline CUDA redirector prevents a clean
  final postflight. Under the strict preregistered composite oracle this is nine of
  ten rejections and mutation-rejection=false. All four retained Linux GitHub attempts
  conclude failure; no contrary passing Linux attempt was found.

  The frozen bindings are therefore evidence-contract=false,
  cross-platform-reproduction=false, mutation-rejection=false, failed=true, and
  blocked=false. The pass expression is false, the failed expression is true, and the
  blocked expression is false. This is a tested-capability failure: Linux had the
  required runner, locked Python, exact TeX engine, time budget, tests, examples, and
  artifact outputs, so the observed cleanliness and startup-surface violations cannot
  be relabeled as unavailable infrastructure. The known absence of retained peak
  memory and accelerator measurements and the run-log omission of two non-primary
  Linux diagnostic attempts remain disclosed limitations; neither supplies favorable
  evidence or alters the independently decisive failures.

  All cumulative E3 receipt kinds other than this independent review and the future
  adjudication are present and hash-valid: environment, run-log, raw-results,
  protocol, attack-plan, statistical-audit, mutation-results, and claim-diff. The
  additional output-commitment, reveal-record, and revealed-manifest custody receipts
  are also valid. The claim audit's sole administrative finding is verified resolved
  by its immutable standalone closure. Existing claims remain narrowly worded and no
  claim promotion is supported. Approval below means only that the evidence chain is
  ready for an adjudicator to record the valid frozen packet outcome failed. It does
  not mean VIA-000 passed, does not establish any physical mechanism, and does not
  permit Tier-R advancement or downstream holdout execution.
findings: []
requested_tests: []
prior_finding_results: []
prior_requested_test_results: []
predictions:
  experiment_id: "VIA-000-FROZEN-RULE-ADJUDICATION"
  predicted_outcome: |-
    A receipt-preserving adjudication of this round will evaluate pass=false,
    fail=true, and blocked=false and will record packet_outcome=failed.
  predicted_failure_mode: |-
    A passed or blocked outcome, any VIA-000 scientific-pass wording, any claim
    promotion, or any Tier-R advance would contradict the immutable raw bindings and
    the frozen conjunctive cross-platform and mutation rules.
  confidence_statement: |-
    High confidence for the Boolean outcome because it follows from independently
    hash-verified retained values and exact rule evaluation. This is an internal
    receipt review, not adjudication, a candidate rerun, or external validation.
recommendation:
  approve: true
  blocking_findings: 0
  rationale: |-
    Approve the integrity and completeness of the retained evidence chain for formal
    adjudication with packet outcome failed. There are no unresolved review blockers
    or requested tests. This approval expressly does not approve a scientific pass,
    Tier-R promotion, downstream holdout execution, or post-reveal repair of this
    frozen round.
```
