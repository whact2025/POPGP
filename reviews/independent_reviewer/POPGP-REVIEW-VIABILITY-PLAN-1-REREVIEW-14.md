# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-14

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-14"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_14"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "2f32abd86592c1dbf150bd9a014e62b96d465393"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "564765088edf901aaf427d23bb4b55cccb97bc54:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-13.md"
builder_response_ref: "2f32abd86592c1dbf150bd9a014e62b96d465393:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-14.md"
context_hash: "f3461e35013abc76528d3a69ff551a509a3e7b90"
context_hash_method: "git rev-parse \"2f32abd86592c1dbf150bd9a014e62b96d465393^{tree}\""
files_reviewed:
  - ".github/workflows/ci.yml"
  - "README.md"
  - "docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "docs/governance/REVIEWER_IDENTITY.md"
  - "docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "docs/templates/REVIEW_RESPONSE_TEMPLATE.md"
  - "docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml"
  - "docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
  - "docs/templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json"
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "schemas/viability/campaign-v2.schema.json"
  - "schemas/viability/packet-v2.schema.json"
  - "schemas/viability/protocol-manifest-v2.schema.json"
  - "schemas/viability/primary-protocol-v1.schema.json"
  - "schemas/viability/requirements-v2.json"
  - "schemas/viability/independent-review-v2.schema.json"
  - "schemas/viability/independent-rereview-v2.schema.json"
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
  - "reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-13.md"
  - "reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-14.md"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..2f32abd86592c1dbf150bd9a014e62b96d465393 (complete 55-file diff)"
  - "git diff 564765088edf901aaf427d23bb4b55cccb97bc54..2f32abd86592c1dbf150bd9a014e62b96d465393 (exact four-file remediation-and-response diff)"
access_level: local/public-repository-only
independence_statement: |-
  This was a fresh focused adversarial re-review in the required isolated reviewer
  worktree and session. It remained under the builder's human operator and root Codex
  orchestrator, and both seats identify as OpenAI Codex GPT-5; exact model snapshots
  are unavailable, so model separation is false. Immutable history and RESPONSE-14
  were visible, but their conclusions were treated as hypotheses. I independently
  bound the candidate and tree, executed the frozen ef96 loader from Git bytes,
  inspected the remediation and surrounding parser/graph code, broadened the merge and
  duplicate matrix, ran the complete repository quality suite, and inspected the exact
  GitHub Actions result. I did not inspect or receive a later builder worktree. No
  final labels, secret seed, private evaluator, credentials, unaffiliated
  implementation, private hardware result, or empirical campaign result was available.
  This is process-separated executable-contract review, not external scientific
  validation.

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
  Approved with zero unresolved blocking findings and no new IDs. VPLAN-SCHEMA-002
  and TST-VPLAN-SCHEMA-002 are verified resolved/satisfied. Executing the exact frozen
  ef96 validator from its Git blob reproduced the reported inline merge-source bypass:
  two threshold keys were accepted and the last value, 0.95, survived. At the reviewed
  2f32 candidate, the identical input raises a controlled duplicate-key error, and the
  complete persisted Tier-R public `validate_campaign` fixture rejects the packet.

  Independent probes also reject repeated explicit keys in external-anchor, merge-
  sequence, nested-merge, direct/local, unused-anchor, and aliased/reused-source
  mappings, as well as unhashable keys. Legal inline/external/nested/sequence merges
  remain field-for-field equal to `yaml.safe_load`: an earlier sequence source wins
  source conflicts and an explicit destination value wins its override. Ordinary and
  reused aliases remain valid. Output alias cycles, constructable recursive merge/value
  cycles, the 100,001st logical node, depth 129, and an over-budget alias DAG fail
  promptly; exactly 100,000 logical nodes and depth 128 pass. A degenerate self-merge
  that produces the same acyclic mapping as `yaml.safe_load` is not a post-construction
  graph cycle and introduces no ambiguity or expansion bypass.

  The shipped packet template equals `yaml.safe_load` over the complete document. All
  eight distinct identity/session overrides and inherited operator/model/orchestrator/
  organization/access fields survive, including the custodian access override. Direct
  and two CLI hashes are identical:
  a549585ebff920eba000f529d6d157c1031924750edb0c157e9f61f51ac009bb.
  Focused tests passed 3/3 in 104.16 seconds and the uncontended full suite passed
  179/179 in 1049.48 seconds. Ruff, TeX validation, review guidance, six examples,
  validation-artifact checking, regeneration cleanliness, history/schema/ref checks,
  diff checks, and residue checks passed. Two independently observed GitHub Actions
  runs for the exact SHA succeeded; pull-request run 31517145533 passed every declared
  CI step. No Tier R, G, or E campaign, external replication, native CUDA result, or
  PDF build was produced.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-002"
    outcome: verified-resolved
    evidence: |-
      The exact ef96 loader accepted the frozen inline duplicate as threshold 0.95.
      The current loader rejected that same source directly, and the persisted complete
      public campaign fixture reported `cannot load packet VIA-000` with `duplicate key
      'threshold'`. External, sequence, nested, unused, direct, and reused-source
      duplicates also rejected, while all legal merge controls matched safe loading.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolved within the declared standard-merge and repeated-explicit-key contract.

  - finding_id: "VPLAN-SCHEMA-001"
    outcome: verified-resolved
    evidence: |-
      Independent exact-boundary probes accepted 100,000 logical nodes and depth 128,
      rejected the next node/depth, rejected an output alias cycle and a level-16
      binary alias DAG, and the focused/full tests retained prompt graph failures across
      campaign, packet, requirements, schema, and receipt inputs.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the declared bytes/depth/nodes/numbers/termination bounds."

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The independent focused and complete suites retained exposure, session, custodian,
      reveal chronology, output/manifest commitment, path/byte distinction, retention,
      and structured-receipt controls.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for local custody consistency; off-system custody remains external."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      Tier-G positive and negative controls remained green in the 179-test suite,
      including required 3D, same-source gravity, laboratory, Lorentz, no-signaling,
      and strict raw-Boolean capability enforcement.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the executable gate contract; no Tier-G result exists."

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
      malformed values, tier closure, and premature-holdout negatives remained green.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for dependency topology and execution order."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      E4/E5 floors, declared/achieved downgrade rejection, unknown levels, cumulative
      receipt kinds, and stricter declarations remained green in the complete suite.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Receipt-class semantic truth still requires substantive review."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      Real-Git commit/tree/rule/requirements/contract/schema/manifest/bundle controls
      passed. Fix commit cf1b356e2bdcc73da97f2b64e36a25ffdd85f914 is an ancestor of
      the candidate, and both required review/response refs resolve to exact blobs.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for frozen object identity."

  - finding_id: "VPLAN-FREEZE-003"
    outcome: verified-resolved
    evidence: |-
      Packet parameters, procedures, budgets, commands, mutations, protocol paths, and
      bytes remained bound to the protocol commit under the full mutation suite.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for preregistration content freeze."

  - finding_id: "VPLAN-PROTOCOL-001"
    outcome: verified-resolved
    evidence: |-
      The closed primary-protocol schema, competing-authority negatives, exact-envelope
      positive, and recursive Boolean/integer/float substitutions remained green.
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
      The complete uncontended Windows suite passed with no matching temporary checkout
      or Git/validator process afterward. Exact-SHA Linux CI also passed the full suite,
      including bounded descendant termination and cleanup tests.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved on the locally and CI-executed process platforms."

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-002"
    outcome: verified-satisfied
    evidence: |-
      Persisted tests now cover production-loader/template equality, all-field merge
      behavior, deterministic CLI hashing, direct duplicate rejection, and the exact
      inline-source bypass through both `_load_yaml_text` and complete public campaign
      validation. Independent external/nested/sequence/unused/reused probes passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for recursive explicit-key checking and legal merge precedence."

  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: verified-satisfied
    evidence: |-
      Persisted focused tests and exact boundary probes retained prompt DAG/cycle
      rejection and exact logical-node/depth behavior across public structured inputs.
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
      receipt-envelope negatives, and the clean-room positive fixture remained green.
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
      Full local and exact-SHA Linux suites, descendant-termination controls, examples,
      and immediate residue inspection passed without a remaining checkout or process.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied on independently executed Windows and Linux CI paths."

predictions:
  experiment_id: ""
  predicted_outcome: ""
  predicted_failure_mode: ""
  confidence_statement: |-
    High for the reviewed executable contract because the former counterexample was
    reproduced at its frozen SHA and rejected at this exact candidate through both
    direct and public paths, with broad merge controls and complete suites green.
    Confidence does not extend to external identity, unaffiliated replication, private
    hardware, an empirical campaign, or POPGP physics.

recommendation:
  approve: true
  blocking_findings: 0
  rationale: |-
    The sole unresolved historical finding and requested test now pass their exact
    counterexample and broader same-scope mutations without weakening standard merges,
    aliases, deterministic hashing, or graph limits. All twelve historical findings
    are verified resolved and all thirteen historical requested tests are verified
    satisfied exactly once. The authoritative local suite and exact-SHA Linux CI are
    green. This approval is for merge-readiness of the documented executable contract;
    it is not a Tier R/G/E result, external scientific validation, or maintainer merge
    authority.
```

## Frozen identity and immutable history

The reviewer worktree began clean at
`2f32abd86592c1dbf150bd9a014e62b96d465393` with tree
`f3461e35013abc76528d3a69ff551a509a3e7b90`; the baseline is an ancestor. The prior
review ref resolves to blob `24c87e90336ad162ef46cfc180118cc28837477a`, and the
builder response ref resolves to blob `1a7e3a74d8cd84972e41c29cabe350a1f7fea4e3`.
Fix commit `cf1b356e2bdcc73da97f2b64e36a25ffdd85f914` is an ancestor of the
candidate.

All historical fenced review/response artifacts through REREVIEW-13/RESPONSE-14 were
reconciled under duplicate-key parsing and the matching Draft 2020-12 schemas. The
focused immutable-artifact test passed. REREVIEW-13 contains exactly 12 unique
historical finding IDs and 13 unique historical requested-test IDs; this artifact
assigns each exactly one outcome and introduces no new ID. RESPONSE-14 covers exactly
the one then-active finding and requested test. The complete baseline diff contains 55
files, 18,053 insertions, and 147 deletions. The prior-review-to-candidate diff contains
four files, 198 insertions, and 23 deletions. Both pass `git diff --check`; no candidate
file was changed.

## Independent merge and duplicate matrix

| Probe | Observed result |
|---|---|
| frozen ef96 inline-only duplicate source | accepted; last threshold 0.95 survived |
| current direct inline source | controlled duplicate `threshold` error |
| complete public Tier-R fixture | controlled `cannot load packet VIA-000` duplicate error |
| external, sequence, nested, unused, and reused duplicate sources | all controlled duplicate errors |
| direct nested duplicate and sequence/mapping keys | duplicate or unhashable-key errors |
| legal external/nested merges | complete equality with `yaml.safe_load` |
| legal merge sequence | earlier source won overlap; explicit destination override won |
| ordinary/reused aliases | accepted; ordinary alias identity preserved |
| output alias and recursive merge/value cycles | controlled errors with prompt termination |
| exact graph boundaries | 100,000 nodes/depth 128 accepted; next values rejected |
| binary alias DAG | level 15 accepted; over-budget level 16 rejected promptly |
| shipped packet template | full equality; all overrides/inheritance retained |
| direct plus two CLI hashes | all `a549585ebff920eba000f529d6d157c1031924750edb0c157e9f61f51ac009bb` |

The duplicate scan walks each reachable YAML mapping/sequence node once by identity
before any merge flattening. Active-node tracking terminates recursive traversal;
post-construction graph validation separately measures each alias occurrence's logical
size and height. Reused legal sources therefore cost one scan but retain correct
logical expansion accounting. I found no in-contract identity-memoization or recursion
bypass.

## Independent execution

| Command or check | Observed result |
|---|---|
| `uv sync --frozen` | exit 0; CPython 3.11.15 environment; 60 packages installed in 23.6 s |
| focused template/duplicate/graph/history tests | exit 0; 3 passed in 104.16 s |
| uncontended `uv run pytest -q` | exit 0; 179 passed in 1049.48 s |
| `uv run ruff check .` | exit 0; all checks passed |
| `uv run python scripts/check_tex.py` | exit 0; 652 lines; balanced source; no Markdown remnants |
| review-guidance tests | exit 0; 9 passed in 12.67 s |
| all six documented examples | all exit 0 in 63.1 s; expected diagnostics/artifacts regenerated |
| `uv run python scripts/check_validation_artifacts.py` | exit 0; contracts and required visuals valid |
| exact-SHA GitHub Actions | two runs succeeded; PR run 31517145533 passed every declared CI step |
| regeneration, diff, ref, and residue checks | clean; no candidate diff, checkout, or matching process remained |

`pdflatex` and `nvcc` were unavailable locally, so no local PDF or native-CUDA build
success is claimed. The exact candidate did pass the repository's Ubuntu GitHub Actions
workflow, including the full tests and documented examples.

## Enforceability and convergence boundary

The executable campaign contract has now converged for every historical review item:
zero blocker remains under the declared byte, numeric, depth, logical-node, timeout,
cleanup, merge, and duplicate-key limits. This does not prove unaffiliated organization,
independent authorship, truthful exposure declarations, off-system custody, or any
POPGP physical claim. Green tests and review agreement do not close the source,
locality, geometry, continuum, closure, Lorentz, hardware, or empirical gaps named by
the viability plan. Those are explicitly future campaign or external-validation work,
not reasons to extend this merge request beyond its documented contract.
