# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-2

```yaml
artifact_schema_version: 1
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-2"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
review_date: "2026-08-10"
commit_reviewed: "5cd803a190da704bf7295974a8e481e46562211a"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "3c84fc6c0d5ef31d314da5fd848b9f54f66e65dd:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-1.md"
builder_response_ref: "5cd803a190da704bf7295974a8e481e46562211a:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-2.md"
context_hash: "a3d3a9d84247ad1941971c3e96b8928f091955ed"
context_hash_method: "git rev-parse \"5cd803a190da704bf7295974a8e481e46562211a^{tree}\""
files_reviewed:
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/governance/REVIEWER_IDENTITY.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-1.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-1.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-2.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/schemas/viability/campaign-v2.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-2/schemas/viability/packet-v2.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-2/schemas/viability/protocol-manifest-v2.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-2/schemas/viability/requirements-v2.json"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
  - "C:/src/POPGP-review-viability-plan-rereview-2/scripts/check_viability_campaign.py"
  - "C:/src/POPGP-review-viability-plan-rereview-2/tests/unit/test_viability_campaign_contract.py"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/scientific_hardening/PROJECT_PLAN.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/scientific_hardening/DECISIONS.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/scientific_hardening/REPRODUCIBILITY.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/docs/framework.md (empirical-compatibility, falsifier, and claim-scope sections)"
  - "C:/src/POPGP-review-viability-plan-rereview-2/.github/workflows/ci.yml"
  - "C:/src/POPGP-review-viability-plan-rereview-2/README.md"
  - "C:/src/POPGP-review-viability-plan-rereview-2/pyproject.toml"
  - "C:/src/POPGP-review-viability-plan-rereview-2/uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..5cd803a190da704bf7295974a8e481e46562211a (complete, all 20 changed files)"
access_level: local/public-repository-only
independence_statement: |-
  This is a fresh reviewer session under the same human operator and root orchestrator
  as the builder. It is process separation, not external scientific independence. The
  builder and reviewer identities exposed to this run are in the same OpenAI Codex GPT-5
  model family; no exact snapshot/version is exposed, so the version is `unknown` and
  model separation is false. Re-review necessarily received both earlier reviews and
  both builder responses. Builder claims were not accepted as proof: the frozen diff,
  schemas, validator, tests, and surrounding scientific/governance contracts were read,
  and every prior mutation family was independently executed. No final labels, secret
  seeds, private evaluator logic, credentials, or private hardware profiles were
  available. This review is not an unaffiliated replication, empirical validation, or
  external scientific confirmation of POPGP.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "OpenAI Codex GPT-5"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  Changes requested with two unresolved blocking findings. The v2 remediation is a
  substantial improvement: canonical raw Boolean capability gates reject the Tier G
  countermodel and strict Boolean/integer substitutions; evaluator-only reveal,
  reproduced-phase ordering, runner/output/reveal byte reconciliation, all requested
  outcome classes, the complete eight-row Boolean truth table, the dependency DAG and
  every dependent holdout target, all E4/E5 floors, real Git identities, candidate-tree
  identity, packet-rule snapshots, canonical requirements, contract checkout identity,
  path containment, and explicit v1 rejection all passed independent mutation checks.
  VPLAN-CUSTODY-001, VPLAN-SCI-001, VPLAN-OUTCOME-001, VPLAN-DEP-001,
  VPLAN-EVIDENCE-001, and VPLAN-FREEZE-002 are verified-resolved, and their six
  requested tests are verified-satisfied.

  VPLAN-SCHEMA-001 remains unresolved. The validator reconciles item IDs and outcome
  summaries, but it does not validate initial review, response, or re-review artifacts
  against their governance templates or bind their commit/reference/evidence fields.
  A four-field response and a seven-field re-review, neither a conforming workflow
  artifact, can mark a blocker and blocking requested test resolved and yield a passing
  campaign. A hash-valid re-review whose outcome is an object crashes the public API,
  and duplicate YAML keys are silently accepted with last-key-wins semantics.

  New blocker VPLAN-FREEZE-003 shows that the campaign-specific protocol receipt and
  its measurement/statistical/resource content are not frozen by the packet-rule hash
  or protocol manifest. Rewriting that receipt after holdout and updating the mutable
  receipt hashes leaves the frozen packet-rule hash unchanged and validates. The green
  authoritative suite (170 tests), all six examples, and artifact checker do not close
  these two accepted counterexamples. No POPGP viability tier is demonstrated here.

findings:
  - id: "VPLAN-FREEZE-003"
    severity: high
    category: governance
    location: "scripts/check_viability_campaign.py:38-64, 149-160, 472-507; schemas/viability/packet-v2.schema.json:345-378; docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md:112-114, 630-638"
    evidence: |-
      `PACKET_FREEZE_FIELDS` freezes the hypothesis, capability/outcome expressions,
      threat prose, and manifest commitments, but excludes the receipt array. The
      `protocol` receipt's path and SHA-256 therefore are neither part of
      `packet_rule_sha256` nor present in `PROTOCOL_MANIFEST.json`. The packet schema
      also has no explicit frozen-parameter, statistics/uncertainty, command, or
      resource-budget fields, although the lifecycle definition says those are
      committed at preregistration.

      Starting from the real temporary-Git passing fixture, I rewrote the file named by
      the required `protocol` receipt from `{"evidence": true}` to
      `{"evidence":true,"threshold":"changed-after-holdout"}` and updated every
      mutable receipt entry sharing those bytes to the new raw SHA-256. I did not change
      the packet rule, protocol manifest, or protocol commit. `validate_campaign(...)`
      returned `[]` and printed
      `POST_FREEZE_PROTOCOL_RECEIPT_SUBSTITUTION_ACCEPTED=True`. In the control,
      changing a genuinely frozen packet field and recomputing only the self hash was
      rejected with `packet rules differ from protocol snapshot`.
    finding: |-
      The protocol manifest freezes decision expressions but not the campaign-specific
      experimental protocol bytes that define how those Boolean inputs are produced.
      A receipt advertised as immutable can be substituted after holdout without
      changing any frozen identity.
    failure_scenario: |-
      After seeing holdout output, an author edits the protocol receipt to change the
      measurement procedure, exclusion rule, uncertainty method, threshold provenance,
      command, or resource ceiling while retaining the same raw Boolean rule. The
      author updates the receipt SHA-256 in PACKET.yaml. Because receipts are post-freeze
      fields, the validator accepts the altered protocol as the preregistered one.
    consequence: |-
      A passing tier cannot prove that raw result fields were produced under the
      protocol, statistics, and resource rules frozen before holdout. This reopens
      post-selection at the boundary immediately upstream of the otherwise strict
      Boolean gates.
    required_action: |-
      Add an explicit preregistration content block for parameters, measurement and
      uncertainty procedures, resource budget, commands, mutations, and protocol
      artifacts. Include its canonical values or content references in
      `packet_rule_sha256` and the protocol manifest, require each referenced protocol
      blob to exist at the named protocol commit, and reject later receipt path/hash
      substitution. Add TST-VPLAN-FREEZE-003.
    verification: confirmed-by-execution
    blocking: true

requested_tests:
  - id: "TST-VPLAN-FREEZE-003"
    description: |-
      Starting from a passing preregistered campaign, alter the campaign-specific
      protocol/parameter/statistics/resource/command bytes after the protocol commit,
      update every mutable receipt SHA-256, and leave the frozen outcome expressions
      unchanged. The campaign command must reject the mutation because the original
      protocol bytes are bound to the protocol commit. Include path substitution,
      same-path byte substitution, and a protocol document with a changed threshold or
      resource ceiling.
    rationale: |-
      Freezing only the final Boolean expression does not prevent post-selection in the
      procedure that creates its raw Boolean inputs.
    blocking: true

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The v2 schemas and validator reject the requested lifecycle, evidence, ID,
      receipt, raw-binding, strict-type, dependency, blocker-count, omitted-item, and
      dangling-supersession mutations. They do not validate the hashed governance
      artifacts themselves. A synthetic initial review with F-1/T-1, a response whose
      only keys were `response_id`, `review_id`, `finding_responses`, and
      `requested_test_responses`, and a re-review whose only keys were `review_id`,
      `review_kind`, prior-result lists, new-item lists, and `recommendation` marked
      both blockers resolved. Those files omit the required commit hashes, artifact
      refs, dispositions, implementation status, evidence, verification level,
      independence declarations, and reviewer identity; validation nevertheless
      returned `[]`.

      A hash-valid re-review with `outcome: {}` raised uncaught
      `TypeError: unhashable type: 'dict'` at the outcome membership check instead of
      returning validation errors. The public `validate_requirements` helper likewise
      raised `AttributeError: 'list' object has no attribute 'get'` for a known
      dependency whose packet entry had the wrong type. A valid campaign document with
      duplicate `decision` keys was accepted using PyYAML's last-key-wins behavior.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Review item coverage is real, but a blocker can still be self-closed by minimal
      nonconforming receipt documents. Require versioned schemas for review/response/
      re-review receipts; verify review IDs, rounds, candidate/commit refs, evidence and
      independence fields; use duplicate-rejecting loaders; type-check before set
      operations; and make every public validation entry point return errors for
      malformed external input.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      Independent campaigns rejected builder-authorized reveal, reveal while
      preregistered/holdout-not-started, wrong output committer, output SHA mismatch,
      a recomputed output-commitment receipt whose `committed_by` bytes disagreed, and a
      recomputed reveal receipt whose `authorized_by` bytes disagreed. The committed
      custody test also retains exposure, shared-session, custodian-role,
      canonicalization, reveal-time, post-reveal manifest, retention, and missing-seat
      mutations.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Declared human identity/exposure truth remains an auditable assertion, not external validation."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      A complete Tier G fixture validated. Binding-free pass, alternate capability
      pointer, alternate capability rule, literal integer/Boolean, and omitted-gate
      mutations were rejected. Raw results setting three-dimensional recovery and
      same-source lensing false produced `passing outcome has failed capabilities`.
      All nine canonical VIA-700 capability names remain required.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "This prevents false promotion; it is not evidence that POPGP passes Tier G."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      Scientific negative, implementation-capability failure, tested resource
      exhaustion, external-replication disagreement, toolchain/hardware/external-access/
      authorization blockage, all five invalid causes, missing receipt, wrong cause,
      zero-true, simultaneous fail/block, and valid/pending cases were independently
      exercised. All eight pass/fail/block Boolean vectors were total: exactly one true
      predicate validated with its correct outcome/cause; every zero/multiple case
      returned errors without raising. Failed-over-blocked campaign precedence held.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The separate malformed-review crash belongs to VPLAN-SCHEMA-001, not outcome classification."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      The canonical registry has 41 dependency edges, all strictly lower-wave, no
      cycles, and complete tier transitive closure. Independent pending-prerequisite
      campaigns rejected holdout start for every dependent target VIA-010, VIA-100,
      VIA-150, VIA-200, VIA-400, VIA-500, VIA-600, VIA-700, VIA-800, and VIA-900.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The validator orders prerequisites but does not provision external execution infrastructure."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      Achieved-evidence downgrades were rejected for every E4 packet VIA-300, VIA-400,
      VIA-500, VIA-600, VIA-700, VIA-800 and the E5 packet VIA-900. A VIA-900 declared
      downgrade, unknown E9 value, and achieved-below-declared case were rejected; the
      positive VIA-000 fixture successfully used E4 above its E3 campaign floor.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Semantic truth of evidence-class labels remains a human review obligation; the mutable-floor defect is closed."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      Nonexistent candidate, baseline, and protocol commits; wrong candidate tree;
      altered packet rule with recomputed self hash; supplied and actual same-version
      requirements downgrades; wrong manifest path/hash; requirements path/hash
      substitutions; post-protocol executing-validator alteration; packet/receipt/Git
      path escapes; and v1 silent migration were all rejected. Actual canonical
      requirements Git-blob SHA-256 was
      `632528e8c4b19d746253719e308b3a676b5a19cffc3a734a670d1c878c161d20`, matching the
      validator pin.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Git provenance still lacks external timestamp/custody; the distinct unfrozen campaign-protocol receipt is VPLAN-FREEZE-003."

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The 11 committed contract tests and independent mutations cover the named
      cross-field, receipt, ID, binding/type, review-item coverage/count, and
      supersession cases. They do not reject minimal nonconforming response/re-review
      artifacts, duplicate YAML keys, or malformed artifact result types without a
      crash. The requested complete review/response/re-review chain therefore remains
      unsatisfied.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Add artifact-schema, strict-loader, reference-pairing, and malformed-type totality rows."

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Committed and independent cases cover blind exposure, role/session reuse,
      custodian authority, reproduced-phase reveal, runner identity, runner/raw-output
      hash binding, structured output/reveal receipt bytes, manifest substitution,
      canonicalization, retention, and missing custody data.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the v2 local cryptographic receipt contract."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: verified-satisfied
    evidence: |-
      The requested raw-false Tier G countermodel, binding-free/alternate rules,
      capability omission, and integer-as-Boolean cases all fail the campaign command.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The test establishes gating behavior only, not physical viability."

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    outcome: verified-satisfied
    evidence: |-
      The committed test persists every requested cause family, and independent
      execution covered all cause classes plus the complete eight-vector Boolean truth
      table. Valid/pending returned a deterministic error and no outcome case raised.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-DEP-001"
    outcome: verified-satisfied
    evidence: |-
      Canonical missing/cycle/wave mutations fail, all 41 edges are lower-wave, tier
      transitive closure holds, and every dependent packet target rejects a pending
      prerequisite at holdout start.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      Every E4/E5 floor, declared and achieved downgrade, unknown level, and stricter
      packet declaration requested by the test was independently exercised.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-FREEZE-002"
    outcome: verified-satisfied
    evidence: |-
      Real temporary Git commits reject every requested nonexistent-object, wrong-tree,
      rule, requirement, contract-checkout, manifest/path/hash, path-escape, and version
      migration mutation. The exact packet-rule and Git-blob hash helpers were also
      exercised.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the mutation scope requested in re-review 1."

predictions:
  experiment_id: "TST-VPLAN-SCHEMA-001 and TST-VPLAN-FREEZE-003"
  predicted_outcome: |-
    A complete remediation will reject nonconforming or ambiguously parsed governance
    receipts, return validation errors rather than exceptions for every malformed
    external type, and reject any post-protocol change to the experiment-defining
    protocol/parameter/statistics/resource bytes even when mutable receipt hashes are
    recomputed.
  predicted_failure_mode: |-
    Adding only more ID coverage assertions will still allow a two-field result to
    impersonate an independent resolution. Adding the current protocol receipt hash
    only to mutable PACKET.yaml will still allow its post-holdout replacement; the hash
    or content reference must cross the packet-rule/protocol-manifest freeze boundary.
  confidence_statement: |-
    High. Both blockers are accepted/crashing counterexamples against the exact frozen
    public validator. This is confidence in contract behavior, not in any POPGP
    physical claim.

recommendation:
  approve: false
  blocking_findings: 2
  rationale: |-
    One prior blocker remains open because hashed but nonconforming review receipts can
    self-close blocking findings/tests and malformed artifact types crash validation.
    One new blocker permits post-holdout substitution of the campaign-specific protocol
    bytes upstream of the frozen Boolean gates. All other prior findings and requested
    tests are independently resolved/satisfied, but approval requires both remaining
    blockers and TST-VPLAN-FREEZE-003 to be remediated and re-reviewed.
```

## Frozen-tree, history, and diff audit

The worktree was clean and on the requested branch before review. The supplied original
re-review commit deserves one provenance note: `3c84fc6c...` is not the literal ancestor
of this line, but the ancestor `ffb5b3e2...` has the same parent, the same tree
`25978d1f13b703d64fb9de0123cca4815acf0422`, and the same re-review blob
`f8f1698ab53e0277eafb8ffab542094a118d69a5`. The named original object exists and is
byte-equivalent; no review content was substituted.

| Command | Exit | Observed result |
|---|---:|---|
| `git branch --show-current` | 0 | `review/adversarial-viability-runbook-rereview-2` |
| `git rev-parse HEAD` | 0 | `5cd803a190da704bf7295974a8e481e46562211a` |
| `git rev-parse "HEAD^{tree}"` | 0 | `a3d3a9d84247ad1941971c3e96b8928f091955ed` |
| `git status --porcelain=v1 --untracked-files=all` | 0 | empty before review execution |
| `git merge-base --is-ancestor 0bdff136c3c5fba8d8868fdd6355f3f824245a8e 5cd803a190da704bf7295974a8e481e46562211a` | 0 | original candidate is an ancestor |
| `git merge-base --is-ancestor 064a233c426f1c620184809bad3a06d8a531a7d3 5cd803a190da704bf7295974a8e481e46562211a` | 0 | initial review is an ancestor |
| `git diff --stat 0bdff136..5cd803a` | 0 | 20 files; 5,596 insertions; 142 deletions |
| `git diff --name-status 0bdff136..5cd803a` | 0 | complete v2 implementation, tests, docs, and preserved artifacts enumerated |
| `git diff --check 0bdff136..5cd803a` | 0 | no output |
| local relative-Markdown-link scan over changed Markdown | 0 | `LOCAL_MARKDOWN_LINK_CHECK: PASS` |

The complete diff was read, including all added files and every deletion from the original
portable prose contract. Builder summaries were used only to enumerate claimed fixes.

## Authoritative suite and repository checks

Every command from `.github/workflows/ci.yml` was run at the frozen handoff.

| Command | Exit | Observed result |
|---|---:|---|
| `uv sync --frozen` | 0 | locked environment; 60 packages checked |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | 652 lines; balanced braces/environments; no Markdown remnants |
| `uv run pytest -q` | 0 | `170 passed in 69.75s` |
| `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | 0 | `11 passed in 49.05s` |
| `uv run python -m examples.physics_qg.chain_1d` | 0 | contiguous cells, D*=1, artifacts regenerated |
| `uv run python -m examples.physics_qg.grid_2d` | 0 | precision/recall 1.0, D*=2, artifacts regenerated |
| `uv run python -m examples.physics_qg.gravity_well` | 0 | Green-function diagnostic pass |
| `uv run python -m examples.physics_qg.source_law` | 0 | relative-entropy slope 1.999684; modular slope 1.0 |
| `uv run python -m examples.physics_qg.source_law_many_body` | 0 | quadratic, Kubo--Mori, Richardson, spreading diagnostics reproduced |
| `uv run python -m examples.physics_qg.ca_model` | 0 | PNG/GIF/JSON regenerated; documented analogy limitation retained |
| `uv run python scripts/check_validation_artifacts.py` | 0 | validation contracts and required visual outputs valid |
| `git status --porcelain=v1 --untracked-files=all` after regeneration | 0 | empty |

`pdflatex` and `nvcc` were unavailable. They are not authoritative CI commands, and the
remediation makes no native/PDF campaign-pass claim. Their absence limits this review's
external/native coverage but does not explain either validator counterexample.

## Independent mutation evidence

All campaign mutations used disposable `TemporaryDirectory` trees, real temporary Git
commits/blobs, the candidate's fixture constructor, and the same public
`validate_campaign(..., repo_root=...)` API used by the committed tests. The shell form
was an inline Python here-string piped to `uv run python -`; no candidate file was
changed. Representative exact mutation operations were:

```python
# Review-chain counterexample: hash each document, add the three typed receipts,
# declare F-1/T-1 resolved, then call the public validator.
response = {
    "response_id": "RESPONSE-1", "review_id": "REVIEW-1",
    "finding_responses": [{"finding_id": "F-1"}],
    "requested_test_responses": [{"requested_test_id": "T-1"}],
}
rereview = {
    "review_id": "REREVIEW-1", "review_kind": "re-review",
    "prior_finding_results": [{"finding_id": "F-1", "outcome": "verified-resolved"}],
    "prior_requested_test_results": [
        {"requested_test_id": "T-1", "outcome": "verified-satisfied"}
    ],
    "findings": [], "requested_tests": [],
    "recommendation": {"approve": True, "blocking_findings": 0},
}
assert validate_campaign(campaign_path, repo_root=frozen_root) == []

# Protocol-receipt counterexample: change shared protocol bytes and all mutable hashes.
protocol_path.write_text(
    '{"evidence":true,"threshold":"changed-after-holdout"}\n', encoding="utf-8"
)
for receipt in packet["receipts"]:
    if resolved(receipt["path"]) == protocol_path:
        receipt["sha256"] = sha256(protocol_path)
assert validate_campaign(campaign_path, repo_root=frozen_root) == []
```

| Mutation/check | Observed result |
|---|---|
| Positive Tier G fixture | accepted |
| Binding-free pass, alternate capability pointer/rule | all rejected |
| Raw 3D/lensing false | rejected |
| Raw integer for Boolean; Boolean versus integer literal | rejected |
| Hashed review omission; incomplete response/re-review ID coverage | rejected |
| Blocker-count/approval mismatch; dangling supersession | rejected |
| Minimal nonconforming response plus re-review | **accepted** |
| Re-review result `outcome: {}` | **uncaught `TypeError`** |
| Duplicate YAML campaign key | **accepted, last key wins** |
| Evaluator authorization, premature reveal, wrong runner/hash | all rejected |
| Output/reveal structured receipt byte mismatches | both rejected |
| Full valid/invalid outcome matrix and eight Boolean vectors | deterministic; intended rows accepted/rejected |
| Canonical DAG/lower-wave/tier closure | pass; 41 edges |
| Pending prerequisite for every dependent target VIA-010 through VIA-900 | all 10 rejected |
| Every E4 packet and VIA-900 E5 downgrade | all 7 rejected |
| Nonexistent Git objects, wrong tree, altered frozen rule/self-hash | all rejected |
| Same-version requirements downgrade, altered executing contract | rejected |
| Manifest/path/hash substitutions and path escapes | rejected |
| v1 campaign/version input | rejected; no silent migration |
| Post-freeze protocol receipt rewritten with updated mutable hashes | **accepted** |

## Limitations and recommendation

The synthetic mutations test contract logic, not honesty of off-system declarations or
scientific adequacy of future thresholds. Git provenance is immutable locally but is not
an independent timestamp or institutional custody service. No hidden campaign, external
operator, private evaluator, native CUDA environment, TeX build, or empirical data were
available. The same operator/orchestrator/model-family limitation is disclosed above.

Changes requested: **two unresolved blocking findings**. VPLAN-SCHEMA-001 and its test
remain open; VPLAN-FREEZE-003 and TST-VPLAN-FREEZE-003 are new. The other six prior
findings and six prior requested tests are independently resolved/satisfied. A later
fresh re-review must verify artifact-schema/totality hardening and a real
protocol-content freeze; builder statements alone cannot close either blocker.
