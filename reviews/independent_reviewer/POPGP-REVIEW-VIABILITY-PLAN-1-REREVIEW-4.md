# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-4

```yaml
artifact_schema_version: 2
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-4"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
reviewer_session_id: "codex-subtask:/root/independent_viability_plan_rereview_4"
reviewer_orchestrator_id: "codex-multi-agent-root"
review_date: "2026-08-11"
commit_reviewed: "dccccfb685c2181192896c70c225b4076e8d8eca"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "7f24eb727d4be69ddb145e2080664fc591936dfd:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-3.md"
builder_response_ref: "dccccfb685c2181192896c70c225b4076e8d8eca:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-4.md"
context_hash: "d2aae5ae008ff0accb97e28c2ac2384f43354e65"
context_hash_method: "git rev-parse \"dccccfb685c2181192896c70c225b4076e8d8eca^{tree}\""
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
  - "docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "docs/scientific_hardening/PROJECT_PLAN.md"
  - "docs/scientific_hardening/REPRODUCIBILITY.md"
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
  - "pyproject.toml"
  - "uv.lock"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..dccccfb685c2181192896c70c225b4076e8d8eca (complete 35-file diff)"
access_level: local/public-repository-only
independence_statement: |-
  This is a fresh re-review subtask under the same human operator and root Codex
  orchestrator as the builder. It is process separation, not external scientific
  independence. The builder and reviewer are both identified as OpenAI Codex GPT-5;
  no exact snapshot/version is exposed, so the version is `unknown` and model
  separation is false. Re-review necessarily received the complete preserved review
  and response history. Builder summaries were used only to identify claimed fixes:
  every prior result below was checked against the frozen source and independently
  executed mutations. No final labels, secret seeds, private evaluator logic,
  credentials, unaffiliated implementation, private hardware, or empirical campaign
  results were available. This review is neither external replication nor physical
  validation of POPGP.

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
  Changes requested with one unresolved blocking finding. The fourth remediation
  closes VPLAN-SCHEMA-001 and VPLAN-PROTOCOL-001: malformed nested requirements and
  malformed JSON/YAML/Markdown structured receipts return deterministic validation
  errors; v2 governance schemas reconcile identity, session, orchestrator, response,
  candidate, context-tree, and fix-commit provenance while all three v1 schemas remain
  byte-unchanged from the prior candidate; and the primary protocol is a closed schema
  whose complete object must equal the canonical preregistration envelope. All prior
  custody, Tier G, outcome, dependency, evidence, and freeze regressions remain closed.

  VPLAN-INDEPENDENCE-001 remains unresolved. The new typed Tier-E contract rejects
  explicit same operator/organization/agent/model/session values, an exact candidate
  repository string, `candidate_core_derived: true`, generic or malformed structured
  receipts, false or mistyped agreement, and broken prediction/output/reveal order.
  However, an honestly pre-frozen Tier E campaign still validates when its comparison
  points at an unrelated extra candidate raw-results receipt, when the candidate and
  external receipts name the exact same path and SHA-256 bytes under different IDs,
  when the candidate repository is reused through a `.git` URL alias, and when the
  external implementation commit/tree are nonexistent self-declarations. The external
  operator schema has no orchestrator field, so reuse of the root orchestrator is not
  representable or reconcilable even though the runbook says same-orchestrator evidence
  remains internal. Both accepted campaigns returned `[]` from `validate_campaign`.
  These are repository/process contract failures, not evidence about POPGP physics.

findings: []

requested_tests:
  - id: "TST-VPLAN-INDEPENDENCE-002"
    description: |-
      Extend the typed Tier-E regression so the candidate comparison receipt must be
      exactly the raw-results receipt named by `blind_custody.output_commitment`, and
      candidate/external outputs must be distinct immutable paths and bytes. Resolve
      the external repository commit/tree against a retained independently retrievable
      Git bundle, archive, or equivalent content-addressed provenance receipt; reject
      nonexistent/mismatched objects and canonical aliases of the candidate repository.
      Add an external orchestrator identity and reject reuse of any internal
      orchestrator. Require a typed comparison receipt that binds both candidate and
      external output hashes and the preregistered comparison computation, rather than
      accepting a self-authored `/agreement: true` field. Retain the current positive
      clean-room fixture plus the same-identity, copied-core, generic/malformed receipt,
      false/mistyped agreement, and chronology negatives.
    rationale: |-
      Tier E is defined by external code/output provenance and an actual
      cross-implementation comparison. Distinct labels and self-declared hashes do not
      establish those facts, and an output commitment cannot protect a comparison that
      is allowed to select unrelated or reused bytes.
    blocking: true

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: verified-resolved
    evidence: |-
      `uv run pytest -q tests/unit/test_viability_campaign_contract.py` passed all 16
      tests. Independent totality cases supplied a root list, scalar evidence order,
      list-valued packet entry, integer dependencies, object capabilities, integer tier,
      and integer receipt-kind list; all returned error arrays without raising.
      Malformed JSON, YAML, and fenced-Markdown-YAML output commitments likewise
      returned `output commitment receipt cannot be parsed`. A schema-complete response
      with the builder model/operator/session/orchestrator/organization mutations was
      rejected, a nonexistent fix commit is covered by the committed suite, and an
      existing protocol commit that was not an ancestor of the re-reviewed candidate
      independently produced `fix commit is not an ancestor`. Canonical context-tree
      and typed independence comparisons remain enforced.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The public entry points were total for the expanded malformed-input matrix.
      Off-system truth cannot be inferred from a schema-valid declaration; the specific
      external-evidence reconciliation defect is retained under VPLAN-INDEPENDENCE-001.

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: verified-resolved
    evidence: |-
      The full and dedicated suites retain blind-seat exposure, prohibited session and
      custodian-role reuse, evaluator-only authorization, reproduced-phase reveal,
      reproduction-runner output commitment, manifest substitution, canonicalization,
      retention, and structured output/reveal receipt reconciliation. Independent
      malformed structured commitments failed closed rather than escaping the API.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for local packet custody; it does not prove truthful off-system custody."

  - finding_id: "VPLAN-SCI-001"
    outcome: verified-resolved
    evidence: |-
      The complete Tier G positive fixture and the canonical raw-Boolean countermodel
      regression passed. Missing 3D/same-source/laboratory capabilities, false raw 3D
      and lensing values, alternate capability rules/pointers, binding-free rules, and
      Boolean/integer substitution remain rejected.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "This is gate-contract evidence only; POPGP has not passed Tier G."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: verified-resolved
    evidence: |-
      The committed 16-test suite retains the full scientific/capability/resource/
      access/invalidity matrix, all eight pass/fail/block truth vectors, missing receipt,
      valid/pending, and failed-over-blocked precedence cases. No outcome case raised.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for deterministic local adjudication."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      Canonical requirements validation returns no errors; all dependencies remain
      lower-wave and tier-transitively closed. The suite still rejects unknown edges,
      cycles, same-wave prerequisites, and dependent holdout execution before pass.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Malformed dependency types are also total under VPLAN-SCHEMA-001."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      The campaign-owned E4 floors and VIA-900 E5 floor remain immutable; declared and
      achieved downgrades, unknown levels, and missing cumulative receipt kinds are
      rejected while a stricter packet declaration is allowed.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Evidence-class semantic truth still requires substantive independent review."

  - finding_id: "VPLAN-FREEZE-002"
    outcome: verified-resolved
    evidence: |-
      The real-Git regression still rejects nonexistent candidate/baseline/protocol
      objects, wrong candidate tree, changed packet rules, requirements downgrade,
      executing-contract change, manifest/path/hash substitution, path escape, and v1
      migration. The response-4 fix commit `4a069d58e95d828701a09129a677d987c04b5e87`
      exists and is an ancestor of the reviewed candidate.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The distinct unverified external-repository hashes remain under VPLAN-INDEPENDENCE-001."

  - finding_id: "VPLAN-FREEZE-003"
    outcome: verified-resolved
    evidence: |-
      The protocol-content regression retains same-path changes to every canonical
      preregistration field family, resource/threshold changes, receipt/self-hash
      recomputation, and campaign-path substitution. Protocol-commit Git blobs and the
      packet-freeze-v3 manifest remain decisive.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Resolved for the candidate campaign protocol."

  - finding_id: "VPLAN-PROTOCOL-001"
    outcome: verified-resolved
    evidence: |-
      `primary-protocol-v1.schema.json` has `additionalProperties: false` and requires
      the complete nine-field envelope. `_validate_preregistration` validates that
      schema and compares the entire document to the packet-derived object. The
      committed exact-envelope negative rejects extra threshold, exclusion,
      measurement, command, and resource keys, while the positive and all post-freeze
      substitutions pass/fail as intended.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "A unique frozen protocol can still be scientifically inadequate; that is a review obligation."

  - finding_id: "VPLAN-INDEPENDENCE-001"
    outcome: unresolved
    evidence: |-
      The positive typed Tier-E fixture and its explicit negative controls were rerun.
      Same internal agent/operator/organization/model/session plus exact candidate
      repository produces every intended distinctness error; generic receipts and
      `candidate_core_derived: true` are rejected. Independently frozen variants also
      rejected Boolean false and numeric `1` agreement, output after reveal, a
      raw-results receipt substituted for the blinded-prediction kind, and malformed
      external contract JSON.

      Two independent positive-by-validator counterexamples remain. First, the fixture
      was frozen with an extra unrelated `raw-results` receipt, the comparison's
      `candidate_output_receipt_id` was changed to it, the external repository was the
      candidate URL with only `.git` appended, and implementation commit/tree were
      `111...`/`222...`; `git cat-file -e` returned 128 for both objects while
      `validate_campaign` returned `[]`. Second, candidate `raw-results` bytes were
      given `/agreement: true` and the external raw receipt was pointed at the exact
      same path and SHA-256 under a different ID; this also returned `[]`. The candidate
      comparison ID is checked only for kind, external/candidate IDs only for string
      inequality, repository inequality only as a raw string, external Git objects are
      never resolved, and the external operator contract has no orchestrator field.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: |-
      The remediation is substantive but does not yet bind the clean-room provenance
      and comparison facts that define Tier E. Real affiliation/authorship will always
      need external verification; the accepted local substitutions are independently
      enforceable contract gaps, not merely unknowable honesty claims.

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: verified-satisfied
    evidence: |-
      The 16-test contract suite plus independent nested-requirement, malformed
      JSON/YAML/Markdown, identity, nonexistent/non-ancestor fix-commit, and v1/v2
      version checks cover the requested fail-closed schema and provenance matrix.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied; the remaining external contract gap has its own stable test IDs."

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: verified-satisfied
    evidence: |-
      Blind exposure, role/session reuse, custodian authority, commitment/reveal order,
      output and manifest hashes, structured receipt parsing, retention, and missing
      custody fields remain executable negative controls.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for local custody consistency."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: verified-satisfied
    evidence: |-
      The raw-false Tier G countermodel and missing/alternate/type gate cases remain
      persisted and rejected.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    outcome: verified-satisfied
    evidence: |-
      Every requested cause family, all Boolean truth vectors, missing evidence,
      ambiguity, valid/pending, and campaign precedence remain deterministic.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-DEP-001"
    outcome: verified-satisfied
    evidence: |-
      The canonical DAG, wave, tier closure, cycle/missing mutations, and premature
      holdout cases remain covered.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      E4/E5 declared and achieved downgrades, unknown evidence, missing required kinds,
      and stricter declarations remain covered.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-FREEZE-002"
    outcome: verified-satisfied
    evidence: |-
      Real Git object/tree, frozen rules, requirements, contract checkout,
      manifest/path/hash, escape, and version mutations remain covered.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for candidate/protocol provenance."

  - requested_test_id: "TST-VPLAN-FREEZE-003"
    outcome: verified-satisfied
    evidence: |-
      All canonical preregistration field/path/byte substitutions remain frozen to the
      protocol commit and rejected after mutable hashes are recomputed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-PROTOCOL-001"
    outcome: verified-satisfied
    evidence: |-
      The exact-envelope positive and extra threshold/exclusion/measurement/command/
      resource negatives are persisted and passed.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied."

  - requested_test_id: "TST-VPLAN-INDEPENDENCE-001"
    outcome: unresolved
    evidence: |-
      The test now rejects explicit same identity fields, exact candidate repository,
      declared copied core, and generic typed receipts, and the independent matrix adds
      false/mistyped agreement, malformed receipt, and chronology rejection. Its
      positive fixture itself uses nonexistent `111...` and `222...` external Git
      identities. It does not test unrelated candidate comparison receipts, identical
      candidate/external paths and bytes under different IDs, candidate repository URL
      aliases, resolvable external Git provenance, or root-orchestrator reuse. Both
      accepted counterexamples therefore survive the requested clean-room objective.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Retain this test and add TST-VPLAN-INDEPENDENCE-002 for the narrowed missing cases."

predictions:
  experiment_id: "TST-VPLAN-INDEPENDENCE-001 and TST-VPLAN-INDEPENDENCE-002"
  predicted_outcome: |-
    A complete remediation will keep all current Tier-E negatives closed and reject
    unrelated or byte-reused candidate/external results, canonical aliases of the
    candidate repository, nonexistent or tree-mismatched external implementation
    objects, root-orchestrator reuse, and an agreement assertion not derived by a
    frozen typed comparison bound to both output hashes.
  predicted_failure_mode: |-
    Checking only receipt IDs, raw repository strings, 40-hex shape, or a Boolean
    `/agreement` field will keep accepting labels that are internally distinct while
    their underlying repository or output evidence is identical, unrelated, or absent.
  confidence_statement: |-
    High. Both remaining counterexamples were honestly constructed before the protocol
    commit, hash-consistent throughout, and returned an empty error list from the exact
    frozen public validator. This confidence concerns contract behavior only.

recommendation:
  approve: false
  blocking_findings: 1
  rationale: |-
    VPLAN-SCHEMA-001 and VPLAN-PROTOCOL-001 are independently resolved, and all eight
    older findings remain closed. VPLAN-INDEPENDENCE-001 is still blocking because the
    validator can promote Tier E with unrelated or reused comparison bytes,
    string-aliased candidate repository reuse, nonexistent external Git provenance,
    and no representable external orchestrator identity. Approval requires those
    fail-open paths and TST-VPLAN-INDEPENDENCE-002 to be closed in a new frozen round.
```

## Frozen identity, history, and full diff

The requested worktree was exact and clean before review execution:

| Command | Exit | Observed result |
|---|---:|---|
| `git branch --show-current` | 0 | `review/adversarial-viability-runbook-rereview-4` |
| `git rev-parse HEAD` | 0 | `dccccfb685c2181192896c70c225b4076e8d8eca` |
| `git rev-parse "HEAD^{tree}"` | 0 | `d2aae5ae008ff0accb97e28c2ac2384f43354e65` |
| `git status --short --branch` | 0 | branch header only; no changes |
| `git diff --check 0bdff136c3c5fba8d8868fdd6355f3f824245a8e dccccfb685c2181192896c70c225b4076e8d8eca` | 0 | no output |
| `git diff --stat 0bdff136c3c5fba8d8868fdd6355f3f824245a8e dccccfb685c2181192896c70c225b4076e8d8eca` | 0 | 35 files; 10,125 insertions; 146 deletions |

The complete baseline-to-candidate diff and every current changed/added file were read,
including the 2,297-line validator, 2,255-line contract suite, 860-line plan, all schemas,
templates, lock changes, and complete preserved review/response history. The named prior
review commit `7f24eb7...` exists; candidate ancestor `ad8707f...` has the same parent,
tree, and re-review blob. The response-4 blob is present at the exact candidate ref.
`4a069d5...` exists and is an ancestor of the candidate. `git diff --exit-code
e8f7862... dccccfb... --` over all three v1 review schemas returned 0.

## Authoritative execution

Every command in `.github/workflows/ci.yml`, README, and the launch runbook was run at
the frozen candidate:

| Command | Exit | Observed result |
|---|---:|---|
| `uv sync --frozen` | 0 | CPython 3.11.15; 60 locked packages installed |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | 652 lines; braces/environments balanced; no Markdown remnants |
| `uv run pytest -q` | 0 | `175 passed in 176.55s` |
| `uv run python -m examples.physics_qg.chain_1d` | 0 | contiguous blocks; D*=1; artifacts regenerated |
| `uv run python -m examples.physics_qg.grid_2d` | 0 | precision/recall 1.0; D*=2; artifacts regenerated |
| `uv run python -m examples.physics_qg.gravity_well` | 0 | Green-function diagnostic passed |
| `uv run python -m examples.physics_qg.source_law` | 0 | relative-entropy slope 1.999684; modular identity slope 1.0 |
| `uv run python -m examples.physics_qg.source_law_many_body` | 0 | quadratic/Kubo--Mori/Richardson/spreading diagnostics reproduced |
| `uv run python -m examples.physics_qg.ca_model` | 0 | PNG/GIF/JSON regenerated; retained analogy limitation unchanged |
| `uv run python scripts/check_validation_artifacts.py` | 0 | validation contracts and required visual outputs valid |
| `uv run pytest -q tests/unit/test_viability_campaign_contract.py` | 0 | `16 passed in 149.06s` |
| `git status --porcelain=v1 --untracked-files=all` after regeneration | 0 | empty |

`pdflatex` and `nvcc` were unavailable. They are not authoritative CI commands, and
the candidate makes no native/PDF campaign-pass claim. No result was invented for them.

## Independent adversarial matrix

All custom campaigns lived under disposable `TemporaryDirectory` repositories, used
real candidate/protocol Git commits generated by the repository fixture, and called the
public `validate_campaign(..., repo_root=...)` entry point. No candidate file was edited.
The decisive accepted operations were equivalent to:

```python
# Redirect the candidate side of the comparison to unrelated raw bytes.
packet["receipts"].append(unrelated_raw_results_receipt)
external["comparison"]["candidate_output_receipt_id"] = (
    "unrelated-candidate-output"
)
external["implementation"]["repository"] = candidate_repository + ".git"
packet["protocol_rule_sha256"] = packet_rule_sha256(packet)
# Freeze this exact packet rule and all exact receipt bytes in the protocol fixture.
assert validate_campaign(campaign, repo_root=frozen_repo) == []

# Reuse the candidate output as the purported external result under another ID.
external_receipt["path"] = candidate_receipt["path"]
external_receipt["sha256"] = candidate_receipt["sha256"]
external["reproduction"]["output_sha256"] = candidate_receipt["sha256"]
assert validate_campaign(campaign, repo_root=frozen_repo) == []
```

| Mutation/check | Observed result |
|---|---|
| Expanded malformed requirements matrix | all returned errors; no exception |
| Malformed JSON/YAML/Markdown structured commitments | all returned parse errors; no exception |
| Existing non-ancestor fix commit | rejected |
| v1 schemas changed from prior candidate | no; byte-unchanged |
| Primary protocol extra threshold/exclusion/measurement/command/resource keys | rejected |
| Same external agent/operator/organization/model/session | rejected |
| Exact candidate repository reuse | rejected |
| Declared `candidate_core_derived: true` | rejected |
| Generic, swapped-kind, or malformed external receipts | rejected |
| Agreement `false` or numeric `1` | rejected |
| External output committed after reveal | rejected |
| Unrelated candidate comparison raw receipt | **accepted** |
| Same candidate/external path and SHA-256 under distinct IDs | **accepted** |
| Candidate repository reused through `.git` alias | **accepted** |
| Nonexistent external commit/tree (`git cat-file` exit 128) | **accepted** |
| Shared external root orchestrator | cannot be represented; external schema has no field |

The accepted Tier-E campaigns are not external scientific validation. They demonstrate
only that the local contract can currently overstate external provenance. POPGP itself
still has finite toy-model evidence and the open physical/scaling/closure/clock gates
described by the claims and falsification matrices; this re-review supplies no Tier R,
G, E, native, continuum, or empirical result.

## Recommendation

Changes requested: **one unresolved blocking finding**. The schema/totality and exact
primary-protocol remediations are independently verified, and all earlier resolved
controls remain green. Tier E must remain non-passing until its typed contract binds the
actual custody output, distinct external bytes, canonical repository and resolvable
implementation provenance, orchestrator separation, and a hash-bound comparison result.
