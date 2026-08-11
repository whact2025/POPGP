# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-12

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-12"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_12"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "9ae2e83f3d54a1e5adabf24da5aa21f222cfe55e"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "1839eca1ebcbe4ee40d25044bb5c8b5d30fdab04:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-11.md"
builder_response_ref: "9ae2e83f3d54a1e5adabf24da5aa21f222cfe55e:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-12.md"
context_hash: "cc61a42ef49269f2753b55f9e996cdd4d27afa0b"
context_hash_method: "git rev-parse \"9ae2e83f3d54a1e5adabf24da5aa21f222cfe55e^{tree}\""
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
  - "pyproject.toml"
  - "uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..9ae2e83f3d54a1e5adabf24da5aa21f222cfe55e (complete 51-file diff)"
  - "git diff 1839eca1ebcbe4ee40d25044bb5c8b5d30fdab04..9ae2e83f3d54a1e5adabf24da5aa21f222cfe55e (exact five-file remediation-and-response diff)"
access_level: local/public-repository-only
independence_statement: |-
  This was a fresh adversarial re-review in the required isolated worktree and
  reviewer session. It remained under the builder's human operator and root Codex
  orchestrator, and both seats identify as OpenAI Codex GPT-5; exact model snapshots
  are unavailable, so model separation is false. Immutable history and the builder
  response were visible, but their claims were treated as hypotheses. I independently
  bound the candidate commit and tree, audited the full source and tests, ran the two
  locked suites separately, reproduced graph and process boundaries, regenerated all
  examples, and tested the shipped packet template through the authoritative loader.
  No final labels, secret seed, private evaluator, credentials, unaffiliated
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
  Changes requested with one newly discovered executable-template blocker. The sole
  REREVIEW-11 graph-totality blocker is independently resolved. The exact 42-level
  shared binary YAML alias DAG now returns controlled expanded-node errors at receipt,
  schema-valid packet preregistration, schema-invalid campaign decision,
  validate_requirements, and validate_campaign requirements-override boundaries.
  Ordinary direct aliases and a complete four-level packet/protocol alias campaign
  remain valid. Direct probes accepted exactly 100,000 logical nodes, depth 128,
  256-character JSON integers, finite maximum/subnormal values, and exactly 16 MiB;
  they promptly rejected the next node, depth, number, and byte boundaries, cycles,
  overflow, underflow, and non-finite values. Three Windows timeout probes killed both
  parent and child and stopped output in 2.24-2.34 seconds under a 0.8-second deadline.

  The shipped packet template is nevertheless not executable under that same public
  loader. Its documented standard YAML merge pattern combines `<<: *seat_protocol`
  with explicit per-seat identity/session overrides. `_construct_unique_mapping`
  calls `flatten_mapping` before duplicate detection, so the legal overrides become
  apparent duplicate keys. The documented packet hash command exits 1 at the builder
  seat's `agent_identity`, even though `yaml.safe_load` and Draft 2020-12 schema
  validation accept the template. The current template test uses `yaml.safe_load`, not
  the authoritative loader, and therefore misses this contradiction. A user following
  the runbook cannot hash or preregister a populated copy without manually rewriting
  every merged seat. This is inside the explicit portable packet and ordinary-alias
  contract, so approval cannot be granted despite all other checks being green.

  The separate campaign-contract file passed 20/20 in 914.67 seconds and the separate
  full repository suite passed 179/179 in 983.80 seconds. Ruff, TeX source validation,
  review-guidance tests, six examples, artifact validation, regeneration cleanliness,
  history/schema/ref checks, both diff checks, and process-residue inspection passed.
  No Tier R, G, or E campaign, external scientific replication, native CUDA result,
  PDF build, or Linux/POSIX cleanup result was produced.

findings:
  - id: "VPLAN-SCHEMA-002"
    severity: high
    category: code
    location: "scripts/check_viability_campaign.py:138-161; docs/templates/VIABILITY_PACKET_TEMPLATE.yaml:47-88; tests/unit/test_viability_campaign_contract.py:1266-1287"
    evidence: |-
      `uv run python scripts/check_viability_campaign.py --packet-rule-sha256
      docs/templates/VIABILITY_PACKET_TEMPLATE.yaml` exited 1 with `found duplicate
      key 'agent_identity'` at the builder mapping. Direct `_load_yaml_text` reproduced
      the same ConstructorError. `yaml.safe_load` accepts the document and applies its
      explicit builder identity/session overrides, which is why the shipped-template
      schema test passes. Source inspection shows the custom constructor flattens YAML
      merges before checking duplicates and therefore cannot distinguish a legal
      explicit override from a true same-level duplicate.
    finding: |-
      The authoritative duplicate-key loader rejects the legal YAML merge overrides
      used by the repository's own packet template, so the claimed executable template
      and the validator disagree.
    failure_scenario: |-
      A campaign author copies and populates the shipped packet template exactly as the
      runbook instructs, then invokes the documented packet-rule hash command. Loading
      stops at the first merged seat override before schema validation or hashing.
    consequence: |-
      The portable campaign cannot be preregistered from its authoritative template,
      and the existing template test gives a false assurance because it bypasses the
      production loader.
    required_action: |-
      Make the authoritative representation consistent: either preserve standard YAML
      merge precedence while still rejecting true duplicate explicit keys, or remove
      merge overrides from the shipped template. Add a production-loader/CLI regression
      that hashes the shipped template deterministically, verifies the seat overrides,
      retains ordinary aliases, and continues to reject genuine duplicate keys.
    verification: confirmed-by-execution
    blocking: true

requested_tests:
  - id: "TST-VPLAN-SCHEMA-002"
    description: |-
      Exercise the shipped packet template with the authoritative unique-key loader and
      packet-hash CLI; assert legal merge overrides produce the intended distinct seat
      fields and a deterministic hash, while an actual duplicate explicit key remains
      a controlled error.
    rationale: |-
      Draft-schema validation through `yaml.safe_load` cannot detect a contradiction in
      the production parser, and the template is the documented campaign entry point.
    blocking: true

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: verified-resolved
    evidence: |-
      The focused public regression passed 2/2 in 96.64 seconds. Complete campaign tests
      returned controlled expanded-node errors for the exact 42-level receipt, packet,
      campaign, requirements-validation, and requirements-override graphs; a matching
      four-level packet/protocol alias campaign returned no errors. Direct probes
      independently confirmed exact node, depth, byte, numeric, finite-value, and cycle
      boundaries. Graph validation now follows every JSON/YAML parse and precedes schema
      error formatting and canonicalization; direct packet hashing and requirements APIs
      have the same guard.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      Resolved for the historical expansion/termination scope. VPLAN-SCHEMA-002 is a
      distinct production-template merge-semantics contradiction, not a continuation of
      the alias-DAG resource failure.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The separate 20-test contract suite and 179-test repository suite retained blind
      exposure, prohibited-session separation, custodian-only reveal, chronology, output
      and manifest commitments, path/byte distinction, retention, and structured receipt
      reconciliation.
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
      commit/tree/content, and fix-ancestry controls. Fix commit 70039a2733dd7ae2a3884004dcc4f5eb13587ba5
      is an ancestor of the exact candidate, and the immutable REREVIEW-11 ref resolves
      to the same blob as the candidate copy.
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
      Both complete suites exercised malformed and valid Git-bundle paths. Three direct
      Windows process-tree probes returned in 2.24-2.34 seconds under a 0.8-second
      execution deadline, killed recorded parent and child PIDs, and froze heartbeat
      bytes. Exit-zero, exit-seven, and spawn-error paths were prompt; final inspection
      found no external-bundle checkout or matching Git process.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved on Windows; the POSIX process-group branch remains externally untested."

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: verified-satisfied
    evidence: |-
      Persisted and independent execution covered the exact hostile DAG at receipt,
      packet, campaign, validate_requirements, and requirements-override boundaries,
      with matching ordinary alias positives and exact byte/node/depth/numeric controls.
      All returned deterministically without an outer supervisor kill.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: |-
      Satisfied for the requested expansion-totality matrix. TST-VPLAN-SCHEMA-002 covers
      the newly identified legal merge-override behavior of the shipped template.

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
      Tier-G positive, raw-false, missing/false capability, alternate rule/pointer,
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
      Both complete contract executions exercised malformed and valid Git-bundle paths;
      direct repeated helper runs independently verified descendant termination, stopped
      output, prompt zero/nonzero/spawn-error controls, and absence of checkout/process
      residue.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied on Windows; POSIX process-group execution remains external."

predictions:
  experiment_id: "executable-template-merge-alias-remediation"
  predicted_outcome: |-
    A complete remediation will make the shipped packet template load and hash through
    the authoritative path with the documented per-seat overrides intact, while direct
    duplicate explicit keys still fail and the current graph limits remain unchanged.
  predicted_failure_mode: |-
    Continuing to flatten merge keys before undifferentiated duplicate detection will
    reject every seat that legally overrides the shared anchor. Testing only with
    `yaml.safe_load` will continue to conceal the production-loader failure.
  confidence_statement: |-
    High for the local executable contradiction: the repository's exact documented CLI
    fails deterministically on its own tracked template and source inspection identifies
    the flatten-before-duplicate cause. Confidence does not extend to off-system identity,
    POSIX cleanup, native hardware, external experiments, or POPGP physics.

recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    The REREVIEW-11 alias-DAG blocker and all eleven older findings are independently
    resolved, and all twelve older requested tests are satisfied. Approval still fails
    closed because the authoritative loader rejects the legal YAML merge overrides in
    the shipped packet template, contradicting the documented executable campaign entry
    point. One bounded parser/template fix and production-path regression should permit
    convergence. No scientific viability tier, external validation, merge authority,
    native CUDA result, PDF build, or Linux CI success follows from this review.
```

## Frozen identity and immutable history

The reviewer branch began clean at
`9ae2e83f3d54a1e5adabf24da5aa21f222cfe55e` with tree
`cc61a42ef49269f2753b55f9e996cdd4d27afa0b`; the baseline is an ancestor. The prior
review ref resolves to blob `a493f4a3726177f51df3e92b9d0cb965da42a898`, identical to
the candidate's REREVIEW-11 copy. The builder response ref resolves at the exact
candidate, and fix commit `70039a2733dd7ae2a3884004dcc4f5eb13587ba5` is its ancestor.

All initial-review through REREVIEW-11 and RESPONSE-1 through RESPONSE-12 fenced
artifacts were parsed with duplicate-key rejection and validated against the Draft
2020-12 schema matching each artifact's declared v1 or v2 version. Every prior-review
and builder-response immutable ref resolved to bytes identical to its tracked artifact.
The complete baseline diff is 51 files with 16,706 insertions and 147 deletions. The
exact prior-review-to-response diff is five files with 251 insertions and 47 deletions.
Both diffs passed `git diff --check`. No candidate file was changed.

## Independent execution

| Command or probe | Observed result |
|---|---|
| `uv sync --frozen` | exit 0; CPython 3.11.15 environment created; 60 packages installed |
| `uv run ruff check .` | exit 0; all checks passed |
| `uv run python scripts/check_tex.py` | exit 0; 652 lines, balanced braces/environments, no Markdown remnants |
| separate focused graph-boundary tests | exit 0; 2 passed in 96.64 s |
| separate `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | exit 0; 20 passed in 914.67 s |
| separate `uv run pytest -q` | exit 0; 179 passed in 983.80 s |
| separate review-guidance tests | exit 0; 9 passed in 15.95 s |
| all six documented `python -m examples.physics_qg.*` commands | all exit 0; finite diagnostics and expected artifacts regenerated |
| `uv run python scripts/check_validation_artifacts.py` | exit 0; contracts and required visuals valid |
| regeneration `git diff --exit-code` | exit 0 before this sole review artifact was created |
| exact DAG public boundary matrix | receipt, packet, campaign, requirements, and override returned controlled expanded-node errors |
| fully frozen ordinary packet/protocol aliases | returned `[]`; direct ordinary aliases accepted |
| exact logical-node boundaries | 100,000 accepted in 0.047 s; 100,001 rejected in 0.063 s |
| exact nesting boundaries | depth 128 accepted; depth 129 rejected |
| exact byte boundaries | 16 MiB accepted; 16 MiB plus one rejected before decode |
| exact numeric controls | finite maximum/subnormal and 256 characters accepted; overflow/underflow/nonfinite/257 rejected |
| Windows bounded-process matrix | three parent/child kills stable; zero/nonzero/spawn-error prompt; no retained output/residue |
| packet-template production loader | exit 1; legal merged builder override reported as duplicate `agent_identity` |

Final inspection found no `popgp-external-bundle-*` checkout and no matching Git or
probe process. `pdflatex` and `nvcc` were unavailable, so no PDF or native-CUDA build
success is claimed. Tests use synthetic campaign fixtures, and process execution was
Windows-only; the POSIX branch and Linux CI remain external actions.

## Enforceability boundary

The remaining blocker is locally decidable consistency between the shipped executable
packet template and the authoritative strict YAML loader. A local validator cannot
prove unaffiliated organization, independent authorship, truthful exposure
declarations, or off-system custody. Green suites and examples do not close the source,
locality, geometry, continuum, closure, Lorentz, hardware, or empirical gaps named by
the scientific plan.
