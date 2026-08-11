# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-5

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-5"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_5"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "c5f2528bb72bd267d79448ad17c9a2d9c3b4d650"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "1647e4f118a773256edb838e1d4f61bd853d5548:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-4.md"
builder_response_ref: "c5f2528bb72bd267d79448ad17c9a2d9c3b4d650:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-5.md"
context_hash: "8f6d72d21232202c234283f1602e2cb711c8f8de"
context_hash_method: "git rev-parse \"c5f2528bb72bd267d79448ad17c9a2d9c3b4d650^{tree}\""
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
  - "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "docs/scientific_hardening/DECISIONS.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/PROJECT_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
  - "docs/scientific_hardening/THEORY_CODE_GAP.md"
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
  - "pyproject.toml"
  - "uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..c5f2528bb72bd267d79448ad17c9a2d9c3b4d650 (complete 37-file diff)"
access_level: local/public-repository-only
independence_statement: |-
  This was a fresh adversarial re-review in a dedicated worktree and session, but it
  remained under the same human operator and root Codex orchestrator as the builder.
  Builder and reviewer are both identified as OpenAI Codex GPT-5; no exact model
  snapshot is exposed, so model separation is false and the version is `unknown`.
  The preserved review/response chain was necessarily available. Builder claims were
  treated only as hypotheses: every status below was rechecked against the frozen
  tree, the complete baseline diff, authoritative execution, and independent
  temporary-Git mutations. No final labels, secret seed, private evaluator, private
  implementation, unaffiliated maintainer, credentials, physical hardware result, or
  empirical campaign result was available. This is process-separated contract review,
  not external scientific validation or proof of off-system affiliation/authorship.

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
  Changes requested with two unresolved blocking findings. The full authoritative
  sequence and all 17 campaign-contract tests pass, and the fifth remediation closes
  the intended output-commitment, resolved-path/byte, Git-bundle, external
  orchestrator, typed-receipt chronology, and valid-disagreement cases represented by
  its persisted tests. All prior custody, scientific-gate, outcome, dependency,
  evidence-floor, freeze, and exact-primary-protocol results remain closed.

  Two independently enforceable gaps remain. First, a schema-valid, honestly frozen
  campaign containing repository URL `file:///%00bad` reaches public validation and
  raises uncaught `ValueError: stat: embedded null character in path`, so the totality
  contract has regressed. Second, the typed comparison check uses Python equality and
  therefore accepts JSON numeric `1` or `1.0` as Boolean agreement and accepts Boolean
  measured values in place of numeric 1/1.0. Canonical candidate-repository separation
  also accepts the aliases `https://github.com:443/whact2025/POPGP`,
  `https://github.com./whact2025/POPGP`,
  `https://github.com/whact2025/x/../POPGP`, and
  `https://github.com/whact2025/POPGP/.git`. Each counterexample began as a complete
  passing Tier-E campaign, was frozen before execution, and returned an empty error
  list except the URL-totality case, which escaped the API. These local consistency
  failures are distinct from real organization, exposure, or authorship truth, which
  the plan correctly leaves to an external maintainer.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      A complete real-Git Tier-E campaign was first frozen and validated successfully.
      Before a second honest freeze its schema-valid external implementation repository
      was changed to `file:///%00bad`, with all affected packet rules and hashes
      recomputed. The public `validate_campaign(..., repo_root=...)` call raised
      `ValueError: stat: embedded null character in path` instead of returning a list
      of validation errors. `_canonical_repository_identity` URL-decodes the path and
      passes the embedded NUL to `Path.resolve()` while catching `OSError` only. Other
      malformed URLs, malformed bundles, 5,000-digit JSON/YAML numbers, NaN receipt
      values, and structured-receipt parse failures returned controlled errors.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      This is a regression in the public totality requirement, not an off-system truth
      problem. One escaping schema-valid input is sufficient to keep the blocker open.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The 17-test contract suite and independent Tier-E mutations retained blind-seat
      exposure controls, distinct session/role constraints, evaluator-only reveal,
      prediction/output/reveal ordering, reproduction-runner output commitment,
      manifest substitution detection, canonical receipt hashing, and retained
      structured output/reveal receipt reconciliation. Missing, malformed, swapped,
      or externally substituted commitments failed closed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for locally enforceable custody consistency; off-system custody still needs external verification."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      The complete Tier-G positive fixture and canonical raw-Boolean countermodel pass.
      Missing 3D, same-source, or laboratory capability; false raw 3D/lensing values;
      altered rule IDs or JSON pointers; missing bindings; and Boolean/integer
      substitution remain rejected by the full suite.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "This verifies only the gate contract; no Tier-G scientific result was produced."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      The full suite retains scientific, capability, resource, access, and invalidity
      cause families; all eight pass/fail/block truth vectors; missing-receipt,
      valid/pending, and failed-over-blocked precedence cases; and the Tier-E valid
      disagreement case. A genuine external disagreement is accepted only as failed
      with `external-replication-disagreed`, never promoted or silently discarded.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for deterministic local outcome adjudication."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      Canonical requirements validation returns no errors and dependencies remain
      lower-wave and tier-transitively closed. Unknown edges, cycles, same-wave
      prerequisites, and premature dependent holdout execution are rejected. Extreme
      or mistyped requirement values also return controlled errors except for the
      separately recorded repository-URL totality regression.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for dependency topology and execution ordering."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      Campaign-owned E4 floors and the VIA-900 E5 floor remain immutable. Declared or
      achieved downgrades, unknown evidence levels, and absent cumulative required
      receipt kinds are rejected, while a stricter packet declaration remains valid.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Evidence-class semantic truth remains a substantive review obligation."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      Real-Git tests reject nonexistent candidate, baseline, or protocol objects;
      incorrect candidate tree; frozen packet-rule, requirements, executing-contract,
      manifest, path, and hash substitutions; path escape; and schema migration. The
      implementation commit `35a8b9979e18b5147187ec7f16ec6385d89ba64e`
      exists and is an ancestor of the reviewed candidate. External bundle mutations
      for missing, malformed, wrong-media, truncated, nonexistent-commit, wrong-tree,
      content replacement, hash/ref substitution, and candidate commit/tree reuse all
      returned errors without escaping.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for candidate/protocol Git identity and the implemented external-bundle cases."

  - finding_id: "VPLAN-FREEZE-003"
    outcome: verified-resolved
    evidence: |-
      Protocol-content tests retain same-path changes to every canonical
      preregistration field family, resources, thresholds, receipt/self-hashes, and
      campaign paths. Protocol-commit blobs and packet-freeze-v4 remain decisive after
      mutable hashes are recomputed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for frozen preregistration content."

  - finding_id: "VPLAN-PROTOCOL-001"
    outcome: verified-resolved
    evidence: |-
      The closed primary-protocol schema still requires the complete canonical
      nine-field envelope; the validator checks that schema and compares the full
      document to the packet-derived object. Extra threshold, exclusion, measurement,
      command, and resource fields remain rejected, while exact positive and all
      post-freeze substitutions behave as specified.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Protocol uniqueness does not establish scientific adequacy."

  - finding_id: "VPLAN-INDEPENDENCE-001"
    outcome: unresolved
    evidence: |-
      The positive real-Git Tier-E fixture and the complete 17-test suite were rerun.
      They reject same explicit agent/operator/organization/model/session/orchestrator,
      copied-core declarations, unrelated custody outputs, identical resolved paths or
      hashes, malformed/generic/swapped receipts, false computed agreement, chronology
      errors, missing/malformed/wrong Git bundles, unresolved or reused commit/tree,
      and the canonical aliases already persisted by the suite.

      Independently frozen variants nevertheless returned `[]` when agreement in the
      hash-verified comparison JSON was numeric `1` or `1.0`, when numeric computed
      candidate value 1 was represented as Boolean `true`, and when numeric computed
      external value 1.0 was represented as Boolean `true`. Python mapping equality
      conflates those values despite the contract requiring typed Boolean agreement
      and typed numeric measured values. Four further complete variants returned `[]`
      with the external repository redirected to the candidate through an explicit
      default HTTPS port, a DNS trailing dot, a dot-segment path, or `/repo/.git`.
      Standard `.git`, `.git/`, credential, HTTPS/SSH/SCP, case, duplicate-slash, and
      percent-encoded `.git` aliases were correctly rejected.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      These are enforceable type and canonical-identity gaps. A valid bundle plus
      consistent declarations still cannot prove independent real-world affiliation,
      authorship, exposure history, or institutional custody; those facts remain an
      explicit external-maintainer responsibility and are not asserted as local bugs.

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      The committed suite and broad independent malformed-input matrix cover nested
      requirements, receipts, bundles, numeric extremes, identity, provenance, and
      schema-version cases, but the embedded-NUL file URL causes the public validator
      to raise. The requested totality property therefore is not satisfied.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Add the honestly pre-frozen NUL-decoding URL counterexample and require a deterministic error list."

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Blind exposure, seat/session reuse, custodian authorization, commitment/reveal
      order, output/manifest hashes, retention, and structured receipt parse and
      reconciliation negatives remain executable and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for local custody consistency."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: verified-satisfied
    evidence: |-
      The Tier-G positive, raw-false countermodel, missing-capability, alternate-rule,
      pointer, binding, and type mutations remain persisted and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    outcome: verified-satisfied
    evidence: |-
      Every required cause family, all truth vectors, missing evidence, ambiguity,
      valid/pending, failed precedence, and correctly caused external disagreement
      remain deterministic in the 17-test suite.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-DEP-001"
    outcome: verified-satisfied
    evidence: |-
      Canonical DAG, wave and tier closure, cycle/unknown/same-wave mutations, and
      premature holdout execution remain covered and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      E4/E5 declared and achieved downgrades, unknown levels, missing required kinds,
      and stricter declarations remain covered and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-FREEZE-002"
    outcome: verified-satisfied
    evidence: |-
      Real Git object/tree, rules, requirements, contract checkout, manifest/path/hash,
      escape, schema-version, and external-bundle mutations remain covered. Independent
      missing, malformed, wrong-media, truncated, replaced-content, nonexistent-commit,
      wrong-tree, and candidate-identity substitutions all returned controlled errors.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the persisted freeze and provenance matrix."

  - requested_test_id: "TST-VPLAN-FREEZE-003"
    outcome: verified-satisfied
    evidence: |-
      Canonical preregistration field, path, and byte substitutions remain frozen to
      the protocol commit and rejected after mutable hashes are recomputed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-PROTOCOL-001"
    outcome: verified-satisfied
    evidence: |-
      The exact-envelope positive and extra threshold, exclusion, measurement, command,
      and resource negatives are persisted and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    outcome: unresolved
    evidence: |-
      The expanded clean-room suite now covers the previously missing custody-output,
      byte/path, bundle, orchestrator, chronology, comparison, and disagreement cases.
      It does not cover Python Boolean/number equality in computed receipt fields or
      the four accepted candidate-repository aliases. The positive clean-room objective
      is therefore still promotable through enforceable local substitutions.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Retain all existing cases and add strict JSON-type and canonical alias negatives."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    outcome: unresolved
    evidence: |-
      The narrowed test now rejects output diversion, identical IDs/paths/bytes,
      ordinary aliases, candidate commit/tree reuse, missing/malformed/replaced bundle
      evidence, external orchestrator reuse, false agreement, pointer/tolerance/order
      mutations, and accepts valid disagreement only as failed with the required cause.
      It still omits the accepted numeric agreement, Boolean-as-number, explicit
      default-port, DNS trailing-dot, dot-segment, and nested `/.git` cases.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The requested typed comparison and canonical repository properties remain incomplete."

predictions:
  experiment_id: "strict comparison typing and repository canonicalization regression"
  predicted_outcome: |-
    A complete remediation will keep the current positive Tier-E campaign valid while
    rejecting numeric agreement, Boolean measured values, default-port and DNS-dot URL
    aliases, normalized dot segments, nested terminal `.git`, and the embedded-NUL URL
    with a controlled error list.
  predicted_failure_mode: |-
    Mapping equality without explicit JSON-type checks will continue to accept
    Boolean/numeric substitution. String normalization that strips only a final suffix
    and does not normalize authority defaults, DNS spelling, and path segments will
    continue to accept candidate-repository reuse. Path resolution before rejecting
    decoded control characters will continue to escape the public API.
  confidence_statement: |-
    High. Every decisive counterexample began from a passing, schema-valid campaign,
    was frozen before validation in a disposable real-Git repository, and either
    returned an empty error list or reproduced the named uncaught exception. Confidence
    concerns executable contract behavior only, not POPGP physics or external identity.

recommendation:
  approve: false
  blocking_findings: 2
  rationale: |-
    Approval is fail-closed. The remediation materially improves Tier-E provenance,
    custody, comparison, and disagreement handling, and the remaining historical
    controls are green. However, public validation is not total for a schema-valid URL,
    and Tier E still accepts mistyped comparison values and canonical aliases of the
    candidate repository. Both are locally enforceable blocking gaps. No new stable
    finding or requested-test ID is needed because the failures fall exactly within
    the two existing unresolved scopes.
```

## Frozen identity, history, and full diff

The dedicated worktree was on the exact requested branch and was clean before any
review command. The response-containing commit, tree, and ancestry were independently
resolved rather than inferred from the handoff text.

| Command | Exit | Observed result |
|---|---:|---|
| `git branch --show-current` | 0 | `review/adversarial-viability-runbook-rereview-5` |
| `git rev-parse HEAD` | 0 | `c5f2528bb72bd267d79448ad17c9a2d9c3b4d650` |
| `git rev-parse "HEAD^{tree}"` | 0 | `8f6d72d21232202c234283f1602e2cb711c8f8de` |
| `git status --short --branch` | 0 | branch header only; no changes |
| `git merge-base --is-ancestor 0bdff136c3c5fba8d8868fdd6355f3f824245a8e HEAD` | 0 | baseline is an ancestor |
| `git cat-file -e 1647e4f118a773256edb838e1d4f61bd853d5548:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-4.md` | 0 | exact prior artifact exists |
| `git cat-file -e c5f2528bb72bd267d79448ad17c9a2d9c3b4d650:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-5.md` | 0 | exact response artifact exists |
| `git diff --check 0bdff136c3c5fba8d8868fdd6355f3f824245a8e c5f2528bb72bd267d79448ad17c9a2d9c3b4d650` | 0 | no output |
| `git diff --stat 0bdff136c3c5fba8d8868fdd6355f3f824245a8e c5f2528bb72bd267d79448ad17c9a2d9c3b4d650` | 0 | 37 files; 11,363 insertions; 147 deletions |

The complete diff and every changed or added file were audited. The review included the
entire 2,505-line validator, 2,604-line contract suite, 870-line viability plan, every
viability schema/requirements/template, all governance/launch/identity instructions,
both v2 review/response schemas and templates, all named scientific matrices and
reproducibility documents, lock/dependency changes, and the complete preserved
review/response chain through response round 5. Builder summaries supplied no evidence.

## Authoritative execution

Every authoritative command named by README and `.github/workflows/ci.yml` ran at the
frozen candidate. Regeneration was checked before this review artifact was created.

| Command | Exit | Observed result |
|---|---:|---|
| `uv sync --frozen` | 0 | 60 locked packages checked |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | 652 lines; braces/environments balanced; no Markdown remnants |
| `uv run pytest -q` | 0 | `176 passed in 312.46s` |
| `uv run python -m examples.physics_qg.chain_1d` | 0 | contiguous blocks; D*=1; artifacts regenerated |
| `uv run python -m examples.physics_qg.grid_2d` | 0 | precision/recall 1.0; D*=2; known Pi_res inadmissibility retained |
| `uv run python -m examples.physics_qg.gravity_well` | 0 | Green-function diagnostic passed |
| `uv run python -m examples.physics_qg.source_law` | 0 | relative-entropy slope 1.999684; modular identity slope 1.0 |
| `uv run python -m examples.physics_qg.source_law_many_body` | 0 | quadratic, Kubo--Mori, Richardson, and spreading diagnostics reproduced |
| `uv run python -m examples.physics_qg.ca_model` | 0 | PNG/GIF/JSON artifacts regenerated |
| `uv run python scripts/check_validation_artifacts.py` | 0 | validation contracts and required visual outputs valid |
| `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | 0 | `17 passed in 279.96s` |
| `git diff --exit-code` and `git status --porcelain=v1 --untracked-files=all` after regeneration | 0 | no generated diff and no untracked files |

`pdflatex` and `nvcc` were unavailable. They are optional rather than authoritative CI
commands, and no PDF/native or physical viability result is claimed here.

## Independent adversarial matrix

All custom campaigns lived in disposable `TemporaryDirectory` repositories, used real
candidate/protocol commits and Git bundles, were fully hash-consistent, and called the
public validator. The positive fixture returned `[]` before mutation. Mutations that
were meant to represent honest declarations were applied before regenerating the
packet rule and protocol commit; they were not post-freeze tampering.

| Mutation/check | Observed result |
|---|---|
| Comparison absent, malformed, generic, or wrong typed receipt kind | rejected; controlled errors |
| Comparison redirected away from blind-custody output | rejected |
| Same IDs, same resolved path, copied identical bytes, or hard-linked output | rejected |
| File symlink alias | unavailable on Windows (`WinError 1314`); resolved-path code and other alias cases exercised |
| External output/commitment/hash or comparison ID/hash swaps | rejected |
| Agreement `false` when recomputation is true | rejected |
| Agreement `NaN` or huge integer | rejected; controlled errors |
| Agreement numeric `1` or `1.0` | **accepted** |
| Numeric candidate 1 represented as Boolean `true` | **accepted** |
| Numeric external 1.0 represented as Boolean `true` | **accepted** |
| Metric pointer `/missing` or invalid `~2` escape | rejected |
| Negative tolerance or NaN tolerance | rejected; controlled errors |
| Wrong comparison actor or comparison before reveal | rejected |
| Valid external disagreement as failed with required cause | accepted as required |
| Exact, `.git`, `.git/`, credential, HTTPS/SSH/SCP, case, duplicate-slash, percent-encoded alias | rejected |
| Explicit default port, DNS trailing dot, path dot segment, or `/repo/.git` alias | **accepted** |
| Candidate commit/tree reuse; nonexistent commit; wrong tree | rejected |
| Missing, malformed, wrong-media, truncated, content-replaced, hash/ref-substituted bundle | rejected; controlled errors |
| Safely bounded 8 MiB malformed bundle | rejected without exception; validator has no explicit evidence-byte ceiling |
| External orchestrator reuse and prior identity/provenance/exposure negatives | rejected |
| External scalar `NaN` or 5,000-digit JSON number | rejected; controlled errors |
| 5,000-digit YAML tolerance | rejected; controlled packet-load error |
| Repository `file:///%00bad` in an honestly pre-frozen campaign | **uncaught `ValueError`** |

The comparison receipt is hash-verified and its values are recomputed, but Python
equality is not JSON type equality. For example, these independently frozen mutations
were accepted without changing the underlying metric values:

```python
comparison["agreement"] = 1       # also 1.0
comparison["candidate_value"] = True  # computed candidate value is numeric 1
comparison["external_value"] = True   # computed external value is numeric 1.0
assert validate_campaign(campaign, repo_root=frozen_repo) == []
```

The 8 MiB malformed bundle establishes controlled failure only, not a general resource
bound. Bundle storage limits remain an operator-side risk already disclosed by the
builder and are not promoted to a new contract blocker in this review.

## Recommendation

Changes requested: **two unresolved blocking findings**. The current campaign contract
substantially improves external evidence consistency and retains every older resolved
control, but approval must wait for total public validation, strict JSON typing of the
comparison computation, and canonical rejection of the demonstrated repository
aliases. No Tier R, G, or E scientific campaign, native/continuum result, or empirical
validation is established by this artifact.
