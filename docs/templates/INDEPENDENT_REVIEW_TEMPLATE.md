# Independent review template

Use this template for initial reviews and re-reviews. Store completed artifacts under
`reviews/independent_reviewer/` and commit them on a reviewer branch.

Completed initial reviews must conform to
[`independent-review-v2.schema.json`](../../schemas/viability/independent-review-v2.schema.json);
completed re-reviews must conform to
[`independent-rereview-v2.schema.json`](../../schemas/viability/independent-rereview-v2.schema.json).
The viability validator rejects duplicate YAML/JSON keys and validates the immutable
`commit:path` receipt reference before using any result.

```yaml
artifact_schema_version: 2
review_id: ""
review_kind: initial|re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: ""       # exact model id supplied by the operator
reviewer_model_version: ""        # version/snapshot, or "unknown"
reviewer_operator: ""
reviewer_session_id: ""
reviewer_orchestrator_id: ""
review_date: ""
commit_reviewed: ""               # full 40-character hash
baseline_commit: ""               # full baseline hash
prior_review_ref: ""              # required for re-review
builder_response_ref: ""          # required for re-review
context_hash: ""
context_hash_method: ""           # exact reproducible command/algorithm
files_reviewed: []
access_level: public-repository-only
independence_statement: ""

independence_declaration:
  shared_operator: false
  shared_session: false
  shared_orchestrator: false
  builder_model_identity: ""
  builder_session_id: ""
  builder_orchestrator_id: ""
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: ""

findings:
  - id: ""
    severity: critical|high|medium|low
    category: science|statistics|code|energy|hardware|governance|claim
    location: ""
    evidence: ""
    finding: ""
    failure_scenario: ""
    consequence: ""
    required_action: ""
    verification: confirmed-by-execution|read-only|unverified
    blocking: true

requested_tests:
  - id: ""
    description: ""
    rationale: ""
    blocking: false

prior_finding_results:             # empty for an initial review
  - finding_id: ""
    outcome: verified-resolved|unresolved|superseded
    evidence: ""
    verification: confirmed-by-execution|read-only|unverified
    superseding_finding_id: ""
    notes: ""

prior_requested_test_results:      # empty for an initial review
  - requested_test_id: ""
    outcome: verified-satisfied|unresolved|superseded
    evidence: ""
    verification: confirmed-by-execution|read-only|unverified
    superseding_requested_test_id: ""
    notes: ""

predictions:
  experiment_id: ""
  predicted_outcome: ""
  predicted_failure_mode: ""
  confidence_statement: ""

recommendation:
  approve: false
  blocking_findings: 0
  rationale: ""
```

## Field rules

- `commit_reviewed`, `baseline_commit`, and review-commit handoffs use full hashes.
- `review_kind: re-review` requires the prior review, builder response, one result for
  every prior finding, and one result for every prior requested test.
- Findings and requested tests use stable IDs that persist across all rounds.
- Only the independent reviewer assigns `verified-resolved`, `unresolved`,
  `superseded`, or `verified-satisfied`.
- Evidence identifies exact files/lines, commands, inputs, and observed results. Do not
  claim execution for a read-only inference.
- `context_hash_method` must be reproducible, for example
  `git rev-parse "<commit>^{tree}"`.
- In a viability campaign, the validator requires the canonical command above and
  reconciles `shared_operator`, `builder_model_identity`, and
  `reviewer_model_differs_from_builder`, plus builder/reviewer session and orchestrator
  declarations, to the packet's builder seat.
- `external_scientific_validation` is always false for an agent review under this
  workflow, even when builder and reviewer use different models.
- `approve: true` requires zero unresolved blocking findings.
- Remove unused example list entries from a completed artifact; do not leave ambiguous
  placeholder objects in `findings`, `requested_tests`, or prior-result lists.
