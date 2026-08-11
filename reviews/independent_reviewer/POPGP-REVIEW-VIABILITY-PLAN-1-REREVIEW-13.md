# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-13

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-13"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_13"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "ef96a6481a91a31f87cd58b46b1057ef4f009584"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "940284c93e77c2044d027b4d91024d71f9e50fc0:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-12.md"
builder_response_ref: "ef96a6481a91a31f87cd58b46b1057ef4f009584:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-13.md"
context_hash: "569ec3701879002a2f76cbf03adf9ca73d5125e2"
context_hash_method: "git rev-parse \"ef96a6481a91a31f87cd58b46b1057ef4f009584^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - "README.md"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "docs/templates/DISAGREEMENT_LOG_TEMPLATE.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
  - "docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml"
  - "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
  - "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
  - "docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "docs/scientific_hardening/DECISIONS.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/PROJECT_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "schemas/viability/campaign-v2.schema.json"
  - "schemas/viability/packet-v2.schema.json"
  - "schemas/viability/protocol-manifest-v2.schema.json"
  - "schemas/viability/primary-protocol-v1.schema.json"
  - "schemas/viability/requirements-v2.json"
  - "schemas/viability/independent-review-v1.schema.json"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/independent-rereview-v1.schema.json"
  - "schemas/viability/independent-rereview-v2.schema.json"
  - "schemas/viability/review-response-v1.schema.json"
  - "schemas/viability/review-response-v2.schema.json"
  - "scripts/check_viability_campaign.py"
  - "tests/unit/test_viability_campaign_contract.py"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-1.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-1.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-2.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-2.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-3.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-3.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-4.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-4.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-5.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-5.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-6.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-6.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-7.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-7.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-8.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-8.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-9.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-9.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-10.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-10.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-11.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-11.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-12.md"
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-12.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-13.md"
  - "pyproject.toml"
  - "uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..ef96a6481a91a31f87cd58b46b1057ef4f009584 (complete 53-file diff)"
  - "git diff 940284c93e77c2044d027b4d91024d71f9e50fc0..ef96a6481a91a31f87cd58b46b1057ef4f009584 (exact five-file remediation-and-response diff)"
access_level: local/public-repository-only
independence_statement: |-
  This was a fresh focused adversarial re-review in the required isolated worktree and
  reviewer session. It remained under the builder's human operator and root Codex
  orchestrator, and both seats identify as OpenAI Codex GPT-5; exact model snapshots
  are unavailable, so model separation is false. Immutable history and the builder
  response were visible, but their claims were treated as hypotheses. I independently
  bound the candidate and tree, read the complete validator, campaign tests, contract,
  governance, templates, schemas, and latest review/response, reconciled the complete
  history, executed both merge happy paths and hostile mappings, and ran the repository
  quality suite. I did not inspect or receive any later builder remediation. No final
  labels, secret seed, private evaluator, credentials, unaffiliated implementation,
  private hardware result, or empirical campaign result was available. This is
  process-separated local contract review, not external scientific validation.

independence_declaration:
  shared_operator: true
  shared_session: false
  shared_orchestrator: true
  builder_model_identity: "OpenAI Codex GPT-5"
  builder_session_id: "019fe641-5da6-7f91-8c77-567db6d5c4a0"
  builder_orchestrator_id: "codex-multi-agent-root"
  reviewer_model_differs_from_builder: false
  external_scientific_validation: false

hidden_access_declaration:
  final_labels_seen: false
  secret_seed_seen: false
  private_evaluator_seen: false

summary: |-
  Changes requested with one unresolved blocking finding and no new ID. The candidate
  fixes the original shipped-template contradiction: the authoritative loader matches
  `yaml.safe_load` for every template field; all distinct seat identity/session
  overrides and inherited fields survive; two CLI and direct hashes equal
  a549585ebff920eba000f529d6d157c1031924750edb0c157e9f61f51ac009bb;
  legal merge sequences preserve standard earlier-source precedence; ordinary aliases
  remain positive; direct and destination-local repeated data keys fail; unhashable
  keys fail; and the 100,000-node, depth-128, alias-DAG, and cycle boundaries are
  unchanged.

  VPLAN-SCHEMA-002 and TST-VPLAN-SCHEMA-002 are nevertheless unresolved because
  `_construct_unique_mapping` checks only the destination mapping's key nodes before
  `flatten_mapping`. PyYAML flattening copies pairs from an inline mapping used solely
  as a `<<` source without invoking that source's mapping constructor, so repeated
  explicit data keys inside the source bypass the check. A direct authoritative probe
  accepted `<<: &inline_duplicate_source` containing `threshold: 0.1` followed by
  `threshold: 0.95`. More decisively, the same source replaced the preregistration
  parameters in a fully frozen Tier-R fixture; the loader retained 0.95, the packet
  hash remained consistent, and public `validate_campaign` returned `[]` in 4.204
  seconds. This is the exact genuine-duplicate rejection required by the prior finding
  and test, so it is not new scope and does not receive another ID.

  The uncontended full repository suite passed 179/179 in 1101.59 seconds. An earlier
  concurrent run observed another suite's same-prefix Windows temporary checkout and
  failed one global residue assertion; after the other run exited, the isolated
  resource test passed in 142.17 seconds, zero temporary checkouts/processes remained,
  and the uncontended full rerun passed. Focused tests passed 3/3 in 98.84 seconds.
  Ruff, TeX source validation, review-guidance tests, six examples, artifact checking,
  regeneration cleanliness, history/schema/ref checks, both diff checks, and final
  residue inspection passed. No Tier R, G, or E campaign, external replication,
  native CUDA result, PDF build, or exact-candidate Linux CI result was produced.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-002"
    outcome: unresolved
    evidence: |-
      The shipped template and legal merge-sequence positives now work, and direct or
      destination-local repeated data keys reject. However, `_load_yaml_text` accepted
      `o: {<<: &b {a: 1, a: 2}}` as `{'o': {'a': 2}}`. A fully frozen Tier-R fixture
      with the same pattern in `preregistration.parameters` retained the last duplicate
      threshold and returned no public campaign-validation errors. Source inspection at
      lines 141-164 shows the pre-flatten key pass skips the merge node, after which
      `flatten_mapping` imports the inline source pairs without applying this constructor
      to the source mapping itself.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Unresolved in its existing scope. The prior required action explicitly required
      genuine repeated explicit keys to remain invalid, so no new finding ID is added.

  - finding_id: "VPLAN-SCHEMA-001"
    outcome: verified-resolved
    evidence: |-
      Exact graph probes accepted 100,000 logical nodes and depth 128, rejected the next
      node/depth, rejected a cycle, and rejected the 42-level binary YAML alias DAG in
      0.016 seconds. The focused production/graph test matrix and both full-suite runs
      retained controlled packet, campaign, requirements, schema, and receipt handling.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the declared byte/depth/node/numeric/termination contract."

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The focused contract tests and uncontended 179-test suite retained blind exposure,
      prohibited-session separation, custodian-only reveal, chronology, output/manifest
      commitments, path/byte distinction, retention, and structured receipt checks.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for local custody consistency; off-system custody remains external."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      Tier-G positive and negative controls remained green in the 179-test suite,
      including required 3D, same-source gravity, laboratory, Lorentz, and no-signaling
      capability enforcement and strict raw Boolean rules.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the executable gate contract only; no Tier-G result exists."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      The full suite retained deterministic pass/fail/block/invalid truth vectors,
      cause classes, ambiguity rejection, evidence errors, and campaign precedence.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for locally computed adjudication."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      Canonical lower-wave dependencies, cycles, unknown edges, same-wave prerequisites,
      malformed values, tier closure, and premature holdout negatives remained green.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for dependency topology and execution order."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      E4/E5 floors, declared/achieved downgrade rejection, unknown levels, cumulative
      receipt kinds, and stricter declarations remained green in the full suite.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Receipt-class semantic truth still requires substantive review."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      The full suite retained real-Git commit/tree/rule/requirements/contract/schema,
      manifest, bundle, and fix-ancestry controls. Fix commit
      90f749762c57b31668da4688d622cd5da0d5f80d is an ancestor of the reviewed
      candidate; the REREVIEW-12 and RESPONSE-13 immutable refs resolve to their exact
      tracked blobs.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for frozen object identity."

  - finding_id: "VPLAN-FREEZE-003"
    outcome: verified-resolved
    evidence: |-
      Packet parameters, procedures, budgets, commands, mutations, protocol paths, and
      bytes remained bound to the protocol commit after mutable hashes were recomputed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for preregistration content freeze."

  - finding_id: "VPLAN-PROTOCOL-001"
    outcome: verified-resolved
    evidence: |-
      The closed primary-protocol schema, competing-authority negatives, exact-envelope
      positive, and recursive Boolean/integer/float type substitutions remained green.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Scientific adequacy of frozen protocol choices remains outside this result."

  - finding_id: "VPLAN-INDEPENDENCE-001"
    outcome: verified-resolved
    evidence: |-
      Repository aliases, bundle identity/provenance, output reuse, orchestrator reuse,
      chronology, comparison typing, percent encodings, and invalid host/path controls
      remained green with the typed clean-room positive fixture.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Organization, authorship, exposure, and affiliation remain external facts."

  - finding_id: "VPLAN-RESOURCE-001"
    outcome: verified-resolved
    evidence: |-
      After a disclosed cross-run `%TEMP%` interference event, the exact isolated
      Tier-E bundle/resource test passed in 142.17 seconds, immediate inspection found
      zero matching temporary checkout directories and no Git process, and the complete
      uncontended suite passed 179/179. Existing descendant-kill and bounded-process
      tests also passed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved on Windows; POSIX process-group execution remains external."

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-002"
    outcome: unresolved
    evidence: |-
      The persisted test proves the shipped template, intended overrides, inherited
      fields, deterministic CLI hashing, and a top-level duplicate. Independent probes
      additionally proved legal sequence precedence, direct/destination-local duplicate
      rejection, ordinary aliases, and unhashable-key rejection. It does not exercise
      an inline merge-source mapping with repeated explicit data keys; that concrete
      case is accepted by both `_load_yaml_text` and complete public campaign validation.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Unresolved in the original test scope. The regression must cover duplicate keys
      recursively in every mapping node that flattening can consume.

  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: verified-satisfied
    evidence: |-
      Persisted focused tests and independent exact boundary probes retained prompt
      DAG/cycle rejection and exact logical-node/depth behavior across public inputs.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the declared structured-input expansion matrix."

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Exposure, session/identity reuse, custodian authority, reveal order, commitments,
      path/byte distinction, structured receipts, retention, and field mutations passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for locally enforceable custody facts."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: verified-satisfied
    evidence: |-
      Tier-G positive/raw-false, missing/false capability, rule/pointer, binding, and
      strict Boolean/number cases remained covered and green.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    outcome: verified-satisfied
    evidence: |-
      Cause families, truth vectors, missing evidence, ambiguity, valid/pending states,
      campaign precedence, and external disagreement remained covered and green.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-DEP-001"
    outcome: verified-satisfied
    evidence: |-
      Canonical DAG/wave/tier closure, cycle, unknown edge, malformed value, and
      premature-holdout cases remained covered and passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      E4/E5 floor, declared/achieved downgrade, unknown-level, cumulative-receipt, and
      stricter-declaration cases remained covered and passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-FREEZE-002"
    outcome: verified-satisfied
    evidence: |-
      Real Git object/tree, packet rule, requirements, contract, schema, manifest,
      bundle content/commit/tree, and candidate-reuse controls remained green.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for frozen Git and bundle provenance."

  - requested_test_id: "TST-VPLAN-FREEZE-003"
    outcome: verified-satisfied
    evidence: |-
      Canonical preregistration fields, paths, bytes, budgets, commands, and mutations
      remained bound to the protocol commit and rejected when altered.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-PROTOCOL-001"
    outcome: verified-satisfied
    evidence: |-
      Competing authority and recursive exact-type controls remained covered and green.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for exact primary-protocol envelope equality."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      Typed organization/operator/implementation, exposure, prediction, commitment,
      receipt-envelope negatives, and the positive clean-room fixture remained green.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for typed clean-room identity and receipt scope."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    outcome: verified-satisfied
    evidence: |-
      Custody output, path/byte distinction, bundle provenance, external orchestrator,
      comparison type/order/tolerance, alias, percent, encoding, and host cases passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for locally enforceable repository/comparison integrity."

  - requested_test_id: "TST-VPLAN-RESOURCE-001"
    outcome: verified-satisfied
    evidence: |-
      The isolated Tier-E bundle/resource test passed, no checkout/process remained,
      and the uncontended complete suite plus descendant-termination test passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied on Windows; POSIX process-group execution remains external."

predictions:
  experiment_id: "recursive-merge-source-duplicate-remediation"
  predicted_outcome: |-
    A complete remediation will recursively reject repeated explicit data keys in every
    mapping node reachable through merge flattening, including an inline merge source,
    while the shipped template, legal merge sequences, explicit overrides, ordinary
    aliases, and current graph/resource limits remain unchanged.
  predicted_failure_mode: |-
    Checking only the destination node before `flatten_mapping`, or relying on normal
    mapping construction for a node consumed only by merge flattening, will continue to
    accept the frozen Tier-R duplicate-threshold counterexample.
  confidence_statement: |-
    High for the local parser contradiction: the exact authoritative loader and full
    public campaign validator accepted the same duplicate-key source deterministically,
    and source control flow explains the bypass. Confidence does not extend to
    off-system identity, POSIX cleanup, native hardware, external experiments, or POPGP
    physics.

recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    The intended packet template and standard legal merge behavior now work, and all
    eleven other historical findings and twelve other historical requested tests remain
    independently resolved/satisfied. Approval still fails closed because the candidate
    accepts repeated explicit data keys in a mapping used solely as an inline YAML merge
    source, including through complete public campaign validation. This is a concrete
    violation of the documented duplicate-key rule and the prior finding's required
    action. No new ID or broader contract demand is introduced. No scientific viability
    tier, external validation, merge authority, native CUDA result, PDF build, or
    exact-candidate Linux CI success follows from this re-review.
```

## Frozen identity and immutable history

The reviewer branch began clean at
`ef96a6481a91a31f87cd58b46b1057ef4f009584` with tree
`569ec3701879002a2f76cbf03adf9ca73d5125e2`; the baseline is an ancestor. The prior
review ref resolves to blob `64d23cbca4a16d1b2b65c8f09e1475d7c095aec2`, and the
builder response ref resolves to blob `7747259767c947e66cce2276c1a509c30dd47482` at
the exact candidate. Fix commit `90f749762c57b31668da4688d622cd5da0d5f80d` is an
ancestor of the candidate.

All 13 review and 13 response fenced artifacts through REREVIEW-12/RESPONSE-13 were
parsed with duplicate-key rejection and validated against the Draft 2020-12 schema
matching each declared v1/v2 version. Every prior-review and builder-response immutable
ref resolved to bytes identical to its tracked artifact, and every review context hash
matched its commit tree. REREVIEW-12 contains exactly 12 unique historical finding IDs
and 13 unique historical requested-test IDs; this artifact assigns each exactly one
outcome. RESPONSE-13 covers exactly the one then-active finding and test. The complete
baseline diff has 53 files, 17,382 insertions, and 147 deletions. The exact prior-review
to candidate diff has five files, 169 insertions, and 11 deletions. Both diffs passed
`git diff --check`. No candidate file was changed.

## Independent execution

| Command or probe | Observed result |
|---|---|
| `uv sync --frozen` | exit 0; CPython 3.11.15 environment created; 60 packages installed in 25.4 s |
| packet-template CLI, twice | both exit 0; identical SHA-256 `a549585ebff920eba000f529d6d157c1031924750edb0c157e9f61f51ac009bb` |
| authoritative template/reference comparison | complete documents equal; eight identity/session overrides and inherited fields retained |
| legal merge-sequence probe | equal to `yaml.safe_load`; earlier source won overlaps; explicit destination value won its override |
| ordinary alias/unhashable/direct-duplicate probes | ordinary alias accepted with identity preserved; unhashable and direct/destination-local duplicates rejected |
| inline merge-source duplicate probe | accepted `a: 1` then `a: 2` as `a == 2` |
| complete frozen Tier-R inline-source counterexample | authoritative threshold became 0.95; `validate_campaign` returned `[]` in 4.204 s |
| exact graph boundaries | 100,000 nodes/depth 128 accepted; next values, direct cycle, and level-42 alias DAG rejected promptly |
| focused template/review-artifact/graph tests | exit 0; 3 passed in 98.84 s |
| uncontended `uv run pytest -q` | exit 0; 179 passed in 1101.59 s |
| isolated Tier-E bundle/resource test | exit 0; 1 passed in 142.17 s; no checkout or Git process remained |
| `uv run ruff check .` | exit 0; all checks passed |
| `uv run python scripts/check_tex.py` | exit 0; 652 lines, balanced braces/environments, no Markdown remnants |
| review-guidance tests | exit 0; 9 passed in 15.29 s |
| all six documented examples | all exit 0 in 88.7 s; expected finite diagnostics/artifacts regenerated |
| `uv run python scripts/check_validation_artifacts.py` | exit 0; contracts and required visuals valid |
| regeneration, schema/history/ref, diff, and residue checks | all pass; no candidate diff, checkout directory, or Git process remained |

One earlier full-suite attempt overlapped another repository worktree's suite. Both
used global Windows `%TEMP%`; the first run therefore observed the other run's
same-prefix checkout and reported 178 passed/one residue assertion failed in 791.76
seconds. The named path disappeared once the other run ended. The isolated regression,
immediate residue inspection, and full uncontended rerun above all passed, so this was
treated as cross-run test interference rather than candidate-owned residue.

`pdflatex` and `nvcc` were unavailable, so no PDF or native-CUDA build success is
claimed. Tests use synthetic campaign fixtures, process execution was Windows-only,
and Linux CI for the exact candidate remains an external gate.

## Enforceability boundary

The blocker is locally decidable and inside the declared executable contract: every
explicit YAML data-key repetition must fail regardless of whether its mapping is
constructed normally or consumed only through merge flattening. A local validator
still cannot prove unaffiliated organization, independent authorship, truthful
exposure declarations, or off-system custody. Green suites and examples do not close
the source, locality, geometry, continuum, closure, Lorentz, hardware, or empirical
gaps named by the scientific plan.
