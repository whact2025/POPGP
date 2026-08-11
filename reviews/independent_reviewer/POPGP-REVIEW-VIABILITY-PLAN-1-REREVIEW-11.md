# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-11

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-11"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_11"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "06561b3e83fedb52707b42e89b278d411311154c"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "3813260cca3bef7bd902569026f52bce37327fe2:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-10.md"
builder_response_ref: "06561b3e83fedb52707b42e89b278d411311154c:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-11.md"
context_hash: "4356d784c6ab3c3dcf62eaf10932a550904f4d22"
context_hash_method: "git rev-parse \"06561b3e83fedb52707b42e89b278d411311154c^{tree}\""
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
  - "pyproject.toml"
  - "uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..06561b3e83fedb52707b42e89b278d411311154c (complete 49-file diff)"
  - "git diff 3813260cca3bef7bd902569026f52bce37327fe2..06561b3e83fedb52707b42e89b278d411311154c (exact five-file review-and-response diff)"
access_level: local/public-repository-only
independence_statement: |-
  This was a fresh adversarial re-review in the required dedicated worktree and
  reviewer session. It remained under the builder's human operator and root Codex
  orchestrator, and both seats identify as OpenAI Codex GPT-5; exact model snapshots
  are unavailable, so model separation is false. Immutable history was visible and
  builder claims were treated only as hypotheses. I independently bound the candidate
  commit and tree, audited the complete baseline and remediation diffs, ran the locked
  suites, created disposable real-Git campaigns, and supervised hostile-input process
  trees. No final labels, secret seed, private evaluator, credentials, unaffiliated
  implementation, private hardware result, or empirical campaign result was available.
  This is process-separated local contract review, not external scientific validation.

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
  Changes requested with one unresolved historical parser-totality blocker. The
  response's receipt-specific remediation is independently verified. The exact old
  42-level binary YAML alias DAG remained live beyond a seven-second supervisor. On
  the candidate, the same complete Tier-R receipt campaign returned two controlled
  expanded-node errors and exited in 5.25 seconds; a fully frozen four-level packet
  alias campaign validated in 5.25 seconds; list cycles returned controlled errors.
  Direct probes retained ordinary, scalar, unequal-depth, merge, mixed mapping/list,
  and wide-fanout aliases, accepted logical node counts through 100,000 and depth
  through 128, and rejected 100,001 nodes and the next depth. The 16 MiB byte ceiling
  was inclusive and 16 MiB plus one byte failed before decode, including multibyte
  UTF-8 accounting. Numeric finite/subnormal/zero and 256-character boundaries passed;
  overflow, nonzero underflow, non-finite constants, duplicates, and 257-character
  numbers failed closed.

  The resource budget is not applied at every structured-input boundary. A complete
  campaign with the exact DAG in schema-valid packet preregistration parameters wrote
  READY but not RETURNED within seven seconds; supervision killed the process tree.
  A campaign document with the same graph in a schema-invalid decision field also
  wrote READY but not RETURNED, consistent with unbounded expansion during schema-error
  rendering. Campaign requirements DAG and cycle overrides returned controlled mismatch
  errors. Receipt-only identity memoization therefore does not make public campaign
  validation total before JSON Schema and packet canonicalization.

  The separate contract file passed 20/20 and the separately invoked full repository
  suite passed 179/179. Ruff, TeX source validation, six examples, artifact validation,
  regeneration, both diff checks, numeric controls, and Windows process-tree cleanup
  were green. Those results are local executable-contract evidence only. No Tier R, G,
  or E campaign, external scientific replication, native CUDA result, or PDF build was
  produced.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The exact former receipt campaign hung under the old validator and returned
      bounded expanded-node errors on the candidate. However, the identical 42-level
      DAG in schema-valid packet preregistration parameters reached the public validator,
      wrote READY, and did not return inside seven seconds. The same graph in a campaign
      decision value also stalled the schema-error path. The supervised processes were
      killed. Direct receipt probes verified identity-memo correctness for shared nodes,
      unequal depths, merges, cycles, mixtures, fanout, and exact node/depth/byte/numeric
      boundaries. The remaining accepted hangs are before receipt traversal, at packet
      canonicalization and campaign schema-error rendering.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Numeric and receipt-graph totality are repaired. Public campaign/packet graph
      totality remains in this historical scope, so no duplicate finding was created.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      Two complete contract executions retained blind exposure, distinct prohibited
      sessions, custodian-only reveal, chronology, output and manifest commitments,
      resolved path/byte distinction, retention, and structured reconciliation.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for local custody consistency; off-system custody remains external."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      Tier-G positive and raw-false controls remained green. Missing or false 3D,
      acceleration/geodesic, same-source lensing and Shapiro, two-potential, laboratory,
      Lorentz, and no-signaling capabilities and altered typed rules remained rejected.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the executable gate contract only; no Tier-G result exists."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      Scientific, capability, resource, access, invalid, and disagreement cause classes;
      pass/fail/block truth vectors; missing evidence; valid/pending states; precedence;
      and campaign outcomes remained deterministic in both complete suite executions.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for locally computed adjudication."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      Lower-wave acyclic tier-closed dependencies remained enforced. Unknown edges,
      cycles, same-wave prerequisites, malformed values, and premature dependent holdout
      start remained rejected.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for dependency topology and execution order."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      Campaign-owned E4 floors and the external packet's E5 floor remained immutable.
      Declared and achieved downgrades, unknown levels, and missing cumulative receipt
      kinds failed while a deliberately stricter declaration remained valid.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Receipt-class semantic truth still requires substantive review."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      Real-Git tests retained nonexistent-commit, candidate-tree, packet-rule,
      requirements, executing-contract, manifest/path/hash, schema migration, bundle
      commit/tree/content, and fix-ancestry controls. The response fix is an ancestor of
      the exact candidate and both immutable prior-review carrier blobs are identical.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for frozen object identity."

  - finding_id: "VPLAN-FREEZE-003"
    outcome: verified-resolved
    evidence: |-
      Same-path changes to parameters, procedures, budgets, commands, mutations,
      protocol paths, and bytes remained bound to protocol-commit blobs and the packet
      freeze after mutable hashes were recomputed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for preregistration content freeze."

  - finding_id: "VPLAN-PROTOCOL-001"
    outcome: verified-resolved
    evidence: |-
      The closed primary-protocol schema retained extra-authority negatives, while
      scalar, list, and nested mapping Boolean/integer/float substitutions and unequal
      key sets remained rejected by recursive exact-type comparison.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolved for exact recursive JSON envelope identity; scientific adequacy of the
      preregistered choices is outside this local result.

  - finding_id: "VPLAN-INDEPENDENCE-001"
    outcome: verified-resolved
    evidence: |-
      The persisted matrix rejected repository aliases, output reuse, bundle identity
      defects, orchestrator reuse, chronology defects, invalid disagreement, strict
      comparison substitutions, residual percent forms, multi-encoding, and invalid
      hosts while retaining the positive typed clean-room fixture.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolved for local URI/provenance consistency only. Organization, authorship,
      exposure, affiliation, and control remain externally verified facts.

  - finding_id: "VPLAN-RESOURCE-001"
    outcome: verified-resolved
    evidence: |-
      The complete contract matrix twice exercised the bounded malformed-bundle path
      and valid bundle control. No bundle checkout or matching Git process remained.
      Three additional direct Windows timeouts returned in 2.48-2.66 seconds under a
      0.8-second execution deadline, killed both recorded parent and child PIDs, and
      froze heartbeat bytes. Exit-zero, exit-seven, and spawn-error paths were prompt.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolved for the historical Git/process-tree scope on Windows. The POSIX branch
      was unavailable locally; structured YAML boundary hangs remain under parser totality.

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The persisted test proves the exact receipt DAG, ordinary alias, cycle, numeric,
      and oversized-campaign cases. Independent breadth added media, unequal-depth,
      merge, mixture, fanout, exact node/depth/UTF-8 byte, and requirements controls.
      The requested portable fail-closed matrix remains incomplete because exact DAGs
      at packet canonicalization and campaign schema-error boundaries did not return.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Add supervised public regressions for every YAML-bearing entry document, not only
      receipt payloads, and prove prompt deterministic failure without an outer kill.

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Exposure, session/identity reuse, custodian authority, reveal order, output and
      manifest hashes, path/byte distinction, structured receipts, retention, and
      missing-field mutations remained covered and passed twice.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for locally enforceable custody facts."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: verified-satisfied
    evidence: |-
      The Tier-G positive, raw-false, missing/false capability, alternate rule/pointer,
      missing-binding, and strict Boolean/number cases remained persisted and green.
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
      Canonical DAG, wave, tier closure, cycle, missing-edge, malformed-value, and
      premature-holdout cases remained covered and passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      E4/E5 declared and achieved downgrades, unknown levels, cumulative receipt kinds,
      and stricter declarations remained covered and passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-FREEZE-002"
    outcome: verified-satisfied
    evidence: |-
      Real Git object/tree, rule, requirements, executing-contract, manifest/path/hash,
      schema-version, bundle content, bundle commit/tree, and candidate-reuse mutations
      remained covered and green.
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
      Extra-authority controls and the exact-envelope positive remained green; scalar
      and nested/list Boolean-integer-float substitutions remained rejected.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for exact primary-protocol envelope equality."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      Distinct typed organization/operator/implementation, exposure, prediction,
      commitment, and generic/malformed receipt negatives remained green; the positive
      clean-room fixture still validates.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for typed clean-room identity and receipt scope."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    outcome: verified-satisfied
    evidence: |-
      Candidate custody output, path/byte distinction, bundle provenance, external
      orchestrator, typed comparison, pointer/tolerance/order, accumulated aliases,
      residual percent, multi-encoding, and invalid-host cases failed closed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for locally enforceable repository and comparison integrity."

  - requested_test_id: "TST-VPLAN-RESOURCE-001"
    outcome: verified-satisfied
    evidence: |-
      Both complete contract runs exercised malformed and valid Git-bundle paths. Direct
      repeated helper runs independently verified descendant termination, stopped output,
      prompt zero/nonzero/spawn-error controls, and absence of checkout/process residue.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied on Windows; POSIX process-group execution remains external."

predictions:
  experiment_id: "cross-boundary-structured-graph-remediation"
  predicted_outcome: |-
    A complete remediation will apply one identity-aware logical work/depth/cycle budget
    to every parsed campaign, packet, and receipt document before JSON Schema, error
    formatting, canonical hashing, or recursive comparison. Both exact public hangs will
    return deterministic errors under supervision while ordinary aliases and all exact
    positive boundaries remain valid.
  predicted_failure_mode: |-
    Leaving the graph budget inside receipt parsing will continue to permit shallow
    alias DAGs to expand exponentially during campaign schema reporting or packet JSON
    canonicalization even though byte size and logical depth are bounded.
  confidence_statement: |-
    High for local executable behavior: both entry-document processes wrote READY but
    not RETURNED, receipt controls were deterministic, and source inspection identifies
    unbudgeted pre-receipt consumers. Confidence does not extend to off-system identity,
    POSIX process cleanup, native hardware, external experiments, or POPGP physics.

recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    Approval remains fail closed. Ten historical findings are independently resolved
    and eleven historical requested tests are satisfied. The remaining historical
    parser-totality blocker is repaired for structured receipts but not for public
    campaign and packet YAML object graphs; two small 42-level inputs still stall under
    supervision before the receipt budget runs. No scientific viability tier, external
    validation, merge authority, native CUDA result, or PDF build follows from this review.
```

## Frozen identity and immutable history

The required reviewer branch began clean at
`06561b3e83fedb52707b42e89b278d411311154c` with tree
`4356d784c6ab3c3dcf62eaf10932a550904f4d22`; the baseline is an ancestor. The prior
review ref resolves at `3813260cca3bef7bd902569026f52bce37327fe2`, and the candidate's
ancestry carries the same review bytes at `f8ca46961dfc1310140371287f2c007a4948b072`;
both resolve to blob `e04fb358203c6d082eb1e12d55f2374d57e738ef`. The response ref
resolves at the reviewed candidate, and fix commit
`e4d43b0a507bba770968e51f2d71ae6f8c481431` is its ancestor.

The complete baseline diff is 49 files with 16,030 insertions and 147 deletions. The
exact prior-review-to-response diff is five files with 251 insertions and 18 deletions.
Both complete diffs were audited and passed `git diff --check`. No candidate file was
changed.

## Independent execution

| Command or probe | Observed result |
|---|---|
| `uv sync --frozen` | exit 0; CPython 3.11.15 environment created; 60 packages installed |
| `uv run ruff check .` | exit 0; all checks passed |
| `uv run python scripts/check_tex.py` | exit 0; 652 lines, balanced braces/environments, no Markdown remnants |
| separate `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | exit 0; 20 passed in 959.25 s |
| separate `uv run pytest -q` | exit 0; 179 passed in 984.06 s |
| all six documented `python -m examples.physics_qg.*` commands | all exit 0; finite diagnostics and expected artifacts regenerated |
| `uv run python scripts/check_validation_artifacts.py` | exit 0; contracts and required visuals valid |
| regeneration `git diff --exit-code` | exit 0 before this sole review artifact was created |
| old exact complete receipt-DAG campaign | READY written; no RETURNED marker inside 7 s; process tree killed |
| candidate exact complete receipt-DAG campaign | two stable expanded-node errors; returned and exited in 5.25 s |
| fully frozen ordinary packet-alias campaign | returned `[]` and exited in 5.25 s |
| direct alias node/depth/media matrix | ordinary/scalar/merge/mixed/unequal-depth positives; cycles and over-budget graphs failed |
| exact logical-node boundaries | 99,999 and 100,000 accepted; 100,001 rejected |
| exact nesting boundaries | depth 128 accepted; depth 129 rejected for JSON/YAML structured documents |
| exact byte boundaries | 16 MiB minus one and 16 MiB accepted; 16 MiB plus one rejected before decode |
| exact numeric controls | finite max/subnormal/zero/256 accepted; overflow/underflow/nonfinite/duplicate/257 rejected |
| requirements override DAG and cycle | both returned one controlled frozen-requirements mismatch |
| packet exact DAG | READY written; no RETURNED marker inside 7 s; supervised process tree killed |
| campaign exact invalid DAG | READY written; no RETURNED marker inside 7 s; supervised process tree killed |
| Windows bounded-process matrix | three parent/child kills stable; zero/nonzero/spawn-error prompt; no retained output/residue |

Final inspection found no `popgp-external-bundle-*` checkout and no matching Git or
probe process. `pdflatex` and `nvcc` were unavailable, so no PDF or native-CUDA build
success is claimed. Tests use synthetic campaign fixtures, and process execution was
Windows-only; the POSIX branch and Linux CI remain external actions.

## Enforceability boundary

The remaining blocker is locally decidable termination of untrusted structured input.
A local validator cannot prove unaffiliated organization, independent authorship,
truthful exposure declarations, or off-system custody. Green suites and examples do
not close the source, locality, geometry, continuum, closure, Lorentz, hardware, or
empirical gaps named by the scientific plan.
