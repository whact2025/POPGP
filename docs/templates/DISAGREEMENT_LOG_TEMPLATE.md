# Disagreement log template

Use this artifact when the builder disputes or only partly accepts a finding. Store the
completed record under `reviews/disagreements/` and preserve both positions even after
a discriminating result is available.

```yaml
artifact_schema_version: 1
disagreement_id: ""
date: ""
review_id: ""
commit_or_protocol: ""

proposition: ""

builder_position:
  model_identity: ""
  statement: ""
  evidence: []
  predicted_result: ""

independent_reviewer_position:
  model_identity: ""
  statement: ""
  evidence: []
  predicted_result: ""

shared_assumptions: []
points_of_disagreement: []

discriminating_test:
  experiment_id: ""
  frozen_before_run: true
  procedure_ref: ""
  outcome_rule: ""

result:
  status: pending|complete
  evidence_ref: ""
  interpretation: ""

resolution:
  status: unresolved|builder-position-supported|reviewer-position-supported|superseded
  claim_status_change: ""
  unresolved_items: []
  authorized_by: ""
```

A maintainer override may authorize merge but does not change an unsupported scientific
proposition into a verified result. Record the override, rationale, and residual risk.
