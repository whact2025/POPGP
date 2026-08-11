# Reviewer identity and independence

POPGP review roles are defined by duties and access boundaries, not by a vendor or
model name.

## Independent-reviewer seat

The independent reviewer:

- audits equations, assumptions, source-law premises, statistical methods, energy and
  conservation boundaries, code, tests, CI, validation artifacts, and claims;
- proposes counterexamples, mutations, and falsifiers;
- reviews a frozen commit rather than a moving branch;
- does not remediate implementation on its review branch; and
- must not receive final labels, secret seeds, private evaluator logic, or credentials.

Changing the model occupying the seat does not change the role and does not by itself
make a review independent.

## Required identity fields

Every review artifact records these fields separately:

| Field | Meaning |
|---|---|
| `reviewer_seat` | `independent-reviewer` |
| `reviewer_model_identity` | Exact model identifier supplied by the operator |
| `reviewer_model_version` | Version/snapshot string, or `unknown` |
| `reviewer_operator` | Human accountable for the run |
| `reviewer_session_id` | Fresh task/session identifier |
| `reviewer_orchestrator_id` | Orchestrator identifier, or a disclosed standalone value |
| `access_level` | Information and systems available to the reviewer |
| `independence_statement` | Shared prompts, context, operator, session, or conclusions |
| `independence_declaration` | Typed shared-role and model-separation facts defined below |

A seat name is not a model identity. If an exact identifier or version is unavailable,
record `unknown`; never guess.

## Independence declaration

A strong review uses a different model and a fresh task/session and initially receives
only the frozen tree, baseline, scope, quality commands, and access restrictions. It
does not receive builder conclusions before forming its initial assessment.

If the same human operates both roles, or an orchestrator relays information between
them, say so. This is useful process separation but not external scientific
independence. A single model role-playing both seats in one response is not an
independent agent review.

Re-review necessarily receives the prior review and builder response. It must still
reproduce evidence rather than accepting the builder's dispositions as proof.

Every review also records the typed declaration:

```yaml
independence_declaration:
  shared_operator: false
  shared_session: false
  shared_orchestrator: false
  builder_model_identity: ""
  builder_session_id: ""
  builder_orchestrator_id: ""
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false
```

`external_scientific_validation` is false for every review conducted under this
workflow. Only a replication by an unaffiliated group with independent code can set it
true; no agent review may set it true.

For machine-adjudicated viability campaigns, these declarations are cross-document
facts rather than unchecked labels: the named builder model must match the packet
builder seat, model separation must equal the actual model-identity comparison, and
shared operator/session/orchestrator flags must equal their corresponding identity
comparisons. Contradictory values invalidate the campaign receipt.

## Hidden-access declaration

Every review records whether the reviewer saw:

```yaml
hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false
```

If any value is true, explain the exposure and its effect on independence. Never hide
an access limitation or infer unavailable evidence.

The operational branch, artifact, and handoff rules are defined in
[the agent review workflow](AGENT_REVIEW_WORKFLOW.md).
