# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-7

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-7"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_7"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "1e378d9dc30be2aa070997e6d04e48f2d808c3eb"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "ec7bafa83c87fcbb4c61dd52c3d8a71f39913f40:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-6.md"
builder_response_ref: "1e378d9dc30be2aa070997e6d04e48f2d808c3eb:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-7.md"
context_hash: "e2ce69a8a87091d345d53de616eae0f481fd2c0e"
context_hash_method: "git rev-parse \"1e378d9dc30be2aa070997e6d04e48f2d808c3eb^{tree}\""
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
  - "pyproject.toml"
  - "uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..1e378d9dc30be2aa070997e6d04e48f2d808c3eb (complete 41-file diff)"
access_level: local/public-repository-only
independence_statement: |-
  This was a fresh adversarial re-review in a dedicated worktree and session, but it
  remained under the same human operator and root Codex orchestrator as the builder.
  Builder and reviewer are both identified as OpenAI Codex GPT-5; no exact model
  snapshot is exposed, so model separation is false and the version is `unknown`.
  The preserved review/response chain was necessarily available. Builder claims were
  treated only as hypotheses: every status below was rechecked against the exact
  frozen tree, the complete baseline diff, independent execution, and fresh disposable
  real-Git Tier-E campaigns. No final labels, secret seed, private evaluator,
  unaffiliated implementation, credentials, private hardware result, or empirical
  campaign result was available. This is process-separated contract review, not
  external scientific validation or proof of off-system affiliation or authorship.

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
  Changes requested with two unresolved blocking findings. RESPONSE-7 repairs every
  exact REREVIEW-6 counterexample. Fresh complete Tier-E campaigns rejected `%ZZ` and
  `%00` repositories with identical deterministic invalid-identity lists; rejected all
  nine default-port, DNS-dot, dot-segment, terminal `/.git`, percent-host, IPv6,
  Windows-file, leading-zero IPv4, and ssh/git+ssh reuse spellings; and rejected all
  four Boolean/number comparison substitutions. The full historical contract suite,
  including output commitment, resolved path/byte distinction, Git bundle,
  orchestrator, chronology, disagreement, custody, schema totality, freeze,
  review-chain, primary-protocol, outcome, dependency, evidence, and Tier-G cases,
  remains green.

  The public boundary is still not total. One honestly frozen Tier-E campaign whose
  external repository host contained a 5,000-digit dotted-decimal label raised an
  uncaught Python integer-conversion `ValueError`. Another complete campaign with a
  NUL in a receipt path raised an uncaught `ValueError` from `Path.resolve`/`stat`.
  Both inputs pass the applicable JSON Schema layer and should produce deterministic
  error lists. VPLAN-SCHEMA-001 therefore remains unresolved.

  Tier-E identity enforcement is also incomplete beyond the fixed examples. Complete
  campaigns returned `[]` when candidate `C:/src/same-repository` was reused as
  `file:C:/src/same-repository`, and when candidate IPv4 `127.0.0.1` was reused through
  IPv4-mapped IPv6 `::ffff:127.0.0.1`. These are locally reconcilable identities, so
  VPLAN-INDEPENDENCE-001 remains unresolved. They do not ask the validator to infer
  affiliation or authorship. No new finding or test ID is needed because the four
  counterexamples fall within the two existing scopes.

  The authoritative suite is otherwise green: the separate campaign-contract run
  passed 18 tests, the full repository run passed 177 tests, all six documented
  examples regenerated, validation artifacts passed, and regeneration left the tree
  clean. These results do not demonstrate Tier R, G, or E or validate POPGP physics.

findings: []

requested_tests: []

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      RESPONSE-7's `%ZZ` fix is real. Fresh, fully frozen Tier-E campaigns made
      `file:///%00bad` and `https://example.com/%ZZ/repo` return the same deterministic
      invalid-identity list on repeated calls. The dedicated 18-test suite retains the
      earlier control, host, port, bracket, duplicate-key, receipt, nested-type, and
      governance totality cases. However, an external repository with a 5,000-digit
      dotted-decimal hostname label passed schema validation and reached the decimal
      IPv4 branch, where `int(part, 10)` raised an uncaught `ValueError`. Independently,
      a schema-valid NUL receipt path reached `_resolve_inside`, where `Path.resolve`
      raised `ValueError: stat: embedded null character in path`. The public
      `validate_campaign` API returned no error list in either case.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The exact malformed-percent remediation is verified, but fail-closed totality is
      incomplete for bounded hostile strings already admitted by the schemas. This is
      a local enforceability defect, not an off-system identity-truth question.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The complete campaign suite retains blind-seat exposure, prohibited role/session
      reuse, evaluator-only reveal, reproduced-phase order, runner output commitment,
      manifest substitution, resolved-path/byte distinction, retention, and structured
      commitment/reveal receipt reconciliation. The full suite passed without an
      escaping custody exception.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for locally enforceable custody consistency; off-system custody remains externally verified."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      The dedicated suite retains the complete Tier-G positive and raw-false
      countermodel. Missing 3D, acceleration/geodesic, same-source lensing/Shapiro,
      two-potential, and laboratory capability gates; false raw 3D/lensing values;
      alternate rules or pointers; missing bindings; and Boolean/number substitutions
      are rejected at the public validator boundary.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "This verifies the gate contract only; no Tier-G scientific result exists."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      The 18-test suite retains scientific, capability, resource, access, and invalid
      cause families; all pass/fail/block truth vectors; missing evidence; valid/pending;
      failed-over-blocked precedence; and valid external disagreement. The separate
      execution returned deterministic results and no outcome exception.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for deterministic local adjudication."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      Canonical requirements validation and the dedicated suite retain the lower-wave,
      acyclic, tier-transitively-closed dependency registry and reject unknown edges,
      cycles, same-wave prerequisites, malformed dependency types, and dependent
      holdout start before every prerequisite passes.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for dependency topology and holdout ordering."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      Campaign-owned E4 floors and the VIA-900 E5 floor remain immutable. Declared or
      achieved downgrades, unknown evidence levels, and missing cumulative receipt
      kinds are rejected, while a stricter packet declaration remains valid.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Evidence-class semantic truth remains a substantive independent-review obligation."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      Real-Git tests retain nonexistent candidate/baseline/protocol rejection,
      candidate-tree identity, packet-rule/requirements/executing-contract freeze,
      manifest/path/hash containment, v1 migration rejection, and Git-bundle
      commit/tree/content binding. Fix commit
      `2bcbb2047fdecde358ca67ccb9199326ca5d9457` exists and is an ancestor of the
      reviewed candidate.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the frozen candidate/protocol and external bundle objects."

  - finding_id: "VPLAN-FREEZE-003"
    outcome: verified-resolved
    evidence: |-
      The protocol-content regression retains same-path changes to every canonical
      preregistration field family, resources, thresholds, receipt/self hashes, and
      campaign paths. Protocol-commit blobs and packet-freeze-v4 remain decisive after
      mutable hashes are recomputed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for preregistration content freeze."

  - finding_id: "VPLAN-PROTOCOL-001"
    outcome: verified-resolved
    evidence: |-
      The closed primary-protocol schema still requires the exact canonical nine-field
      envelope. Extra threshold, exclusion, measurement, command, and resource
      authorities are rejected, and all post-freeze protocol substitutions remain
      rejected in the dedicated suite.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Protocol uniqueness does not establish scientific adequacy."

  - finding_id: "VPLAN-INDEPENDENCE-001"
    outcome: unresolved
    evidence: |-
      RESPONSE-7 closes every exact prior case. Fresh complete Tier-E campaigns
      rejected candidate reuse through default HTTPS port, DNS trailing dot, dot
      segment, terminal `/.git`, unreserved percent-encoded host, expanded IPv6,
      Windows drive/file URI, leading-zero IPv4, and ssh/git+ssh port-22 spellings.
      Output commitment identity, distinct paths and bytes, external Git-bundle
      commit/tree, orchestrator separation, chronology, honest disagreement, and
      recursive typed receipt comparison remain covered and green.

      Two independently constructed, honestly frozen real-Git Tier-E campaigns still
      returned `[]` for candidate repository reuse. On Windows,
      `C:/src/same-repository` and the valid file URI
      `file:C:/src/same-repository` identify the same drive path, but the latter is
      mistaken for SCP-like syntax before URL parsing. Separately,
      `https://127.0.0.1/repo` and `https://[::ffff:127.0.0.1]/repo` identify the same
      IPv4 endpoint through the standardized IPv4-mapped IPv6 form, but the
      canonicalizer leaves the mapped form distinct. The broader probe also exercised
      encoded host dots and separators, percent-decoding depth, Unicode/IDNA, legacy
      IPv4 integer/octal/hex spellings, IPv6 zone IDs, file localhost, UNC and drive
      case, empty/dot paths, query/fragment, ssh/git+ssh/SCP, controls, and malformed
      ports/brackets without another promoted alias.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      These are locally reconcilable repository aliases. Actual affiliation,
      independent authorship, and control of a remote repository remain off-system
      facts for an external maintainer and are not claimed here.

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      Exact NUL/control repository identities, `%ZZ`, malformed host/port/bracket,
      duplicate-key, nested type, structured-receipt, immutable-ref, and governance
      provenance cases now return deterministic error lists without raising. The
      5,000-digit dotted host and the NUL receipt-path campaigns instead raise uncaught
      `ValueError`, so the requested fail-closed malformed-input boundary is incomplete.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Retain the existing matrix and add adversarial length plus NUL/path-operation cases."

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Blind exposure, seat/session reuse, custodian authority, reveal order,
      output/manifest hashes, resolved path and byte distinction, structured receipts,
      retention, and missing custody fields remain executable negative controls.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for local custody consistency."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: verified-satisfied
    evidence: |-
      The Tier-G positive, raw-false countermodel, missing capability, alternate rule,
      pointer, binding, and Boolean/number cases remain persisted and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    outcome: verified-satisfied
    evidence: |-
      Every requested cause family, truth vector, missing-evidence, ambiguity,
      valid/pending, campaign precedence, and external disagreement row remains
      deterministic in the dedicated suite.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-DEP-001"
    outcome: verified-satisfied
    evidence: |-
      Canonical DAG/wave/tier closure, cycles, missing dependencies, malformed values,
      and premature holdout cases remain covered and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      E4/E5 declared and achieved downgrades, unknown levels, cumulative receipt kinds,
      and stricter declarations remain covered and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-FREEZE-002"
    outcome: verified-satisfied
    evidence: |-
      Real Git object/tree, rules, requirements, executing-contract, manifest/path/hash,
      escape, schema-version, bundle content, commit/tree, and candidate reuse mutations
      remain covered and pass. A fresh honestly frozen 16 MiB malformed bundle also
      produced a controlled repository-bundle error rather than an exception.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the frozen Git/provenance scope."

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
      Exact-envelope positives and extra threshold, exclusion, measurement, command,
      and resource negatives are persisted and pass.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    outcome: unresolved
    evidence: |-
      The expanded clean-room suite and fresh campaigns now cover the exact REREVIEW-5
      and REREVIEW-6 aliases and strict scalar/nested JSON substitutions while retaining
      custody, byte/path, bundle, orchestrator, chronology, and disagreement cases. It
      does not cover the accepted opaque Windows file URI or IPv4-mapped IPv6 spellings,
      so candidate repository reuse remains promotable.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Retain all current cases and extend canonical identity negatives."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-002"
    outcome: unresolved
    evidence: |-
      Output commitment identity, distinct paths/bytes, bundle commit/tree, external
      orchestrator, typed comparison, pointer/tolerance/order, and all nine exact
      historical aliases pass. The two broader repository aliases still validate, so
      the requested canonical repository property remains incomplete.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Add opaque/hierarchical file-URI equivalence and IPv4-mapped IPv6 equivalence."

predictions:
  experiment_id: "TST-VPLAN-SCHEMA-001, TST-VPLAN-INDEPENDENCE-001, and TST-VPLAN-INDEPENDENCE-002"
  predicted_outcome: |-
    A complete remediation will preserve the positive clean-room campaign and every
    repaired historical case while returning deterministic errors for overlong dotted
    host labels and NUL receipt paths, and treating opaque Windows file URIs and
    IPv4-mapped IPv6 endpoints as candidate repository reuse.
  predicted_failure_mode: |-
    Leaving integer conversion and path resolution outside guarded boundaries will
    continue to let schema-admitted hostile strings escape the public API. Applying
    SCP rewriting before recognizing every file-URI form and preserving mapped IPv6
    text without projecting its IPv4 endpoint will continue to accept the two aliases.
  confidence_statement: |-
    High. Every decisive counterexample was a complete schema-valid campaign built
    with real candidate/protocol commits and a real external Git bundle, frozen before
    validation, and either accepted with an empty error list or observed to raise at
    the public validator boundary. Confidence concerns executable contract behavior
    only, not POPGP physics or external identity truth.

recommendation:
  approve: false
  blocking_findings: 2
  rationale: |-
    Approval is fail closed. RESPONSE-7 repairs every exact REREVIEW-6 mutation and all
    older scientific, custody, outcome, dependency, evidence, freeze, review-chain,
    protocol, output, comparison, and Tier-G controls remain green. Nevertheless, the
    public validator still raises for two schema-admitted inputs and Tier E still
    accepts two demonstrably equivalent candidate repository identities.
    VPLAN-SCHEMA-001 and VPLAN-INDEPENDENCE-001 therefore remain blocking; their three
    active requested tests remain unresolved. No Tier R, G, or E result or external
    scientific validation is established here.
```

## Frozen identity, history, and complete diff

The dedicated reviewer worktree was exact and clean before review execution:

| Command | Exit | Observed result |
|---|---:|---|
| `git branch --show-current` | 0 | `review/adversarial-viability-runbook-rereview-7` |
| `git rev-parse HEAD` | 0 | `1e378d9dc30be2aa070997e6d04e48f2d808c3eb` |
| `git rev-parse "HEAD^{tree}"` | 0 | `e2ce69a8a87091d345d53de616eae0f481fd2c0e` |
| `git status --short --branch` | 0 | branch header only; no changes |
| `git merge-base --is-ancestor 0bdff136c3c5fba8d8868fdd6355f3f824245a8e HEAD` | 0 | baseline is an ancestor |
| `git merge-base --is-ancestor 2bcbb2047fdecde358ca67ccb9199326ca5d9457 HEAD` | 0 | RESPONSE-7 fix commit is an ancestor |
| `git diff --check 0bdff136c3c5fba8d8868fdd6355f3f824245a8e HEAD` | 0 | no output |
| `git diff --stat 0bdff136c3c5fba8d8868fdd6355f3f824245a8e HEAD` | 0 | 41 files; 12,956 insertions; 147 deletions |

The exact prior review ref and response ref both exist. The prior-review blob at
`ec7bafa...` is byte-identical to the blob in the candidate. Candidate commit
`1196787...` is equivalent to the named prior-review commit: it has the same parent,
tree, subject, and review blob, although the named reviewer-branch commit itself is not
a literal ancestor. The immutable ref remains locally resolvable.

The complete baseline-to-candidate diff, complete validator, complete contract suite,
every viability schema/template/requirements object, all named scientific and
reproducibility matrices, and the complete preserved review and response chain through
RESPONSE-7 were read. Builder summaries were not used as proof.

## Authoritative execution

Every command in README, CI, and the launch runbook ran against the frozen candidate.
The campaign-contract suite was also run separately as requested.

| Command | Exit | Observed result |
|---|---:|---|
| `uv sync --frozen` | 0 | CPython 3.11.15; fresh environment; 60 locked packages installed |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | 652 lines; braces and environments balanced; no Markdown remnants |
| `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | 0 | `18 passed in 456.20s` |
| `uv run pytest -q` | 0 | `177 passed in 474.63s` |
| `uv run python -m examples.physics_qg.chain_1d` | 0 | contiguous blocks; D*=1; artifacts regenerated |
| `uv run python -m examples.physics_qg.grid_2d` | 0 | precision/recall 1.0; D*=2; known Pi_res inadmissibility retained |
| `uv run python -m examples.physics_qg.gravity_well` | 0 | Green-function diagnostic passed |
| `uv run python -m examples.physics_qg.source_law` | 0 | relative-entropy slope 1.999684; modular identity slope 1.0 |
| `uv run python -m examples.physics_qg.source_law_many_body` | 0 | quadratic/Kubo--Mori/Richardson/spreading diagnostics reproduced |
| `uv run python -m examples.physics_qg.ca_model` | 0 | PNG/GIF/JSON artifacts regenerated |
| `uv run python scripts/check_validation_artifacts.py` | 0 | validation contracts and required visual outputs valid |
| `git diff --exit-code` and `git status --porcelain=v1 --untracked-files=all` after regeneration | 0 | no generated diff and no untracked files |

`pdflatex` and `nvcc` were unavailable. They are optional rather than authoritative CI
commands, so no PDF/native success is claimed. No hidden campaign, private evaluator,
unaffiliated group, native hardware result, continuum result, or empirical data were
available.

## Independent adversarial evidence

All decisive campaigns lived under disposable `TemporaryDirectory` repositories, used
real candidate/protocol commits and external Git bundles, were hash-consistent, and
called the public `validate_campaign(..., repo_root=...)` boundary. Mutations intended
as honest declarations were applied before packet-rule and protocol freezing.

| Mutation/check | Observed result |
|---|---|
| Exact REREVIEW-5 four repository aliases | all rejected as candidate reuse |
| Exact RESPONSE-7 encoded-host, IPv6, Windows-file, leading-zero IPv4, and ssh/git+ssh aliases | all rejected as candidate reuse |
| `%00` and invalid `%ZZ` external repository | each returned a deterministic invalid-identity list on two calls; no exception |
| Agreement numeric `1`/`1.0`; Boolean candidate/external measured values | all rejected by recursive strict equality |
| Candidate drive path versus external `file:C:/...` | **accepted with `[]`** |
| Candidate IPv4 versus external IPv4-mapped IPv6 | **accepted with `[]`** |
| 5,000-digit dotted-decimal repository hostname label | **uncaught Python integer-conversion `ValueError`** |
| NUL in `output-commitment` receipt path | **uncaught `ValueError: stat: embedded null character in path`** |
| Encoded dots/separators/depth, Unicode/IDNA, legacy IPv4, zone IDs, localhost/UNC, case, empty/dot paths, query/fragment, ssh/git+ssh/SCP, controls, malformed ports/brackets | probed; no additional promotable mutation |
| Output commitment, path/byte distinction, bundle commit/tree, orchestrator, chronology, disagreement, custody, freeze, review-chain, primary protocol, outcome, dependency, evidence, Tier G | retained in the separately green complete contract suite |
| Honestly frozen 16 MiB malformed Git bundle with reconciled hashes | controlled repository-bundle clone timeout error; no exception |

The malformed bundle demonstrates controlled failure at that size, not a general
evidence-byte ceiling. The validator hashes local evidence by streaming and bounds Git
clone by 30 seconds, but it has no explicit receipt-byte ceiling; this remains an
operator-side resource limitation rather than a promoted finding in this round.

## Enforceability boundary and recommendation

Changes requested: **two unresolved blocking findings**. Both are local executable
contract defects: total public-API behavior and normalization of repository identities
that the validator itself compares. The reviewer does not infer unaffiliated status,
authorship, organizational control, or actual remote-repository custody from local
strings and bundles; those remain external maintainer attestations. Likewise, green CI,
example regeneration, and a sounder runbook do not supply a Tier R/G/E result, validate
the proposed physics, close continuum or scaling gaps, or establish scientific
viability.
