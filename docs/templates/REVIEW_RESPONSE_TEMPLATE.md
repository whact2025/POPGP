# Review response template

This is a builder artifact, not a re-review and not proof that a finding is resolved.
Store completed responses under `reviews/codex/`. The independent reviewer assigns
final finding outcomes in a later artifact.

Completed responses used by a viability campaign must conform to
[`review-response-v1.schema.json`](../../schemas/viability/review-response-v1.schema.json)
and be named by an immutable `commit:path` reference in the packet review chain.

```yaml
artifact_schema_version: 1
response_id: ""
response_round: 1
response_date: ""

builder_seat: builder
builder_model_identity: ""
builder_model_version: ""
builder_operator: ""

review_id: ""
review_artifact: ""
review_commit: ""                 # full hash containing immutable review
candidate_commit_reviewed: ""     # full candidate hash named by review

access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
  notes: ""

summary: ""

finding_responses:
  - finding_id: ""
    blocking_as_reported: true
    disposition: accepted|partially-accepted|disputed|deferred
    implementation_status: implemented|not-implemented|external-action-required
    rationale: ""
    changed_files: []
    fix_commits: []                 # full hashes; fixes precede response artifact
    verification:
      - command: ""
        result: ""
    residual_risk: ""
    disagreement_ref: ""

requested_test_responses:
  - requested_test_id: ""
    disposition: accepted|partially-accepted|disputed|deferred
    implementation_status: implemented|not-implemented|external-action-required
    test_locations: []
    verification:
      - command: ""
        result: ""
    rationale: ""
    disagreement_ref: ""

new_or_changed_risks: []

external_actions:
  - action: ""
    owner: ""
    status: pending|complete
    evidence_ref: ""

rereview_request:
  requested: true
  scope: "all findings, requested tests, regressions, and new findings"
  handoff_commit: "recorded outside this artifact after it is committed"
  notes: ""
```

## Field rules

- Include one response for every finding and requested-test ID, including non-blocking
  or disputed items. Use an empty list only when the review contains none.
- `accepted` describes agreement. `implemented` describes builder action. Neither means
  independently resolved.
- A disputed blocking finding requires a disagreement record. A deferred or external
  blocking action stays open until the reviewer verifies evidence.
- Record exact commands and outcomes; “tests pass” is insufficient.
- `fix_commits` contains full hashes. If a fix and response share a commit, leave the
  list empty and identify changed files; the re-review binds the final handoff hash.
- Do not guess the response-containing commit's hash inside itself. Put that hash in
  the handoff or pull request.
