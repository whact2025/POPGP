# Independent re-review: POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-1

```yaml
artifact_schema_version: 1
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1-REREVIEW-1"
review_kind: re-review
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "Richard Fuoco"
review_date: "2026-08-10"
commit_reviewed: "fba32f03ba92998e587560793f6118cf618a41f7"
baseline_commit: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
prior_review_ref: "064a233c426f1c620184809bad3a06d8a531a7d3:reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1.md"
builder_response_ref: "fba32f03ba92998e587560793f6118cf618a41f7:reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-1.md"
context_hash: "8b91abe42545921595791d454f585bf68c4c7e4a"
context_hash_method: "git rev-parse \"fba32f03ba92998e587560793f6118cf618a41f7^{tree}\""
files_reviewed:
  - "C:/src/POPGP-review-viability-plan-rereview-1/.github/workflows/ci.yml"
  - "C:/src/POPGP-review-viability-plan-rereview-1/README.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/framework.md (viability and empirical-compatibility sections)"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/governance/REVIEWER_IDENTITY.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/scientific_hardening/DECISIONS.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/scientific_hardening/PROJECT_PLAN.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/scientific_hardening/REPRODUCIBILITY.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml"
  - "C:/src/POPGP-review-viability-plan-rereview-1/docs/templates/VIABILITY_PACKET_TEMPLATE.yaml"
  - "C:/src/POPGP-review-viability-plan-rereview-1/pyproject.toml"
  - "C:/src/POPGP-review-viability-plan-rereview-1/uv.lock"
  - "C:/src/POPGP-review-viability-plan-rereview-1/reviews/independent_reviewer/POPGP-REVIEW-VIABILITY-PLAN-1.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/reviews/codex/POPGP-REVIEW-VIABILITY-PLAN-1-RESPONSE-1.md"
  - "C:/src/POPGP-review-viability-plan-rereview-1/schemas/viability/campaign-v1.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-1/schemas/viability/packet-v1.schema.json"
  - "C:/src/POPGP-review-viability-plan-rereview-1/schemas/viability/requirements-v1.json"
  - "C:/src/POPGP-review-viability-plan-rereview-1/scripts/check_viability_campaign.py"
  - "C:/src/POPGP-review-viability-plan-rereview-1/tests/unit/test_viability_campaign_contract.py"
  - "C:/src/POPGP-review-viability-plan-rereview-1/examples/physics_qg/* (all six documented example entry points and regenerated validation artifacts)"
  - "git diff 0bdff136c3c5fba8d8868fdd6355f3f824245a8e..fba32f03ba92998e587560793f6118cf618a41f7 (complete, all 16 changed files)"
access_level: local/public-repository-only
independence_statement: |-
  This is a fresh reviewer session under the same human operator and root orchestrator
  as the builder. It is process separation, not external scientific independence. The
  reviewer and builder identities exposed in this record are from the same OpenAI Codex
  GPT-5 model family; no snapshot/version is exposed, so the reviewer version is recorded
  as `unknown` and model separation is false. Re-review necessarily received the initial
  review and builder response, but no builder summary was accepted as proof: every prior
  finding and requested test was traced to the frozen source and independently exercised.
  No final labels, secret seeds, private evaluator logic, credentials, or private hardware
  profiles were available. This review is not an unaffiliated replication, empirical
  validation, or external scientific confirmation of POPGP.

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
  Changes requested. Two original findings, VPLAN-DEP-001 and VPLAN-EVIDENCE-001,
  are verified-resolved, and their requested tests are verified-satisfied. The canonical
  requirements file has a closed lower-wave DAG, all target tiers include their transitive
  dependencies, dependent holdout starts are rejected until prerequisites pass, E4/E5
  floors cannot be lowered, and all required Tier G capability names are present.

  Four original blocking findings remain unresolved. The validator accepts binding-free
  literal capability rules even when hash-verified raw results record failed Tier G
  comparisons; Python equality also permits integer 1 to satisfy a Boolean-true rule.
  Review closure is self-declared rather than reconciled to the hashed review bytes, so an
  omitted unresolved blocker or a dangling supersession passes. Reveal authorization is
  not bound to the evaluator/custodian, and a packet may be revealed while still only
  preregistered, before holdout start, attack, or reproduction. A valid-round/pending-outcome
  contradiction crashes `validate_campaign` with `KeyError` instead of returning a
  deterministic validation error, and the committed outcome truth-table test omits several
  cases explicitly requested by the initial review.

  One new blocking finding, VPLAN-FREEZE-002, records that candidate, baseline, tree, and
  protocol hashes are checked only for 40-hex shape and packet/campaign equality. A fully
  passing synthetic Tier G campaign uses nonexistent Git objects and an unrelated tree hash.
  The validator also accepts removal of a mandatory Tier G capability from the same-version
  requirements document. Thus the executable contract does not yet prove that its rules and
  registry are the frozen objects named by the campaign.

  The complete authoritative suite passes with 168 tests, every documented example exits 0,
  validation artifacts conform, and regeneration leaves the frozen tree clean. Those green
  results do not close the accepted adversarial mutations. Recommendation: changes requested,
  five unresolved blocking findings (four prior and one new). Four prior blocking requested
  tests remain unresolved; two are satisfied; one new blocking test is requested.

findings:
  - id: "VPLAN-FREEZE-002"
    severity: high
    category: governance
    location: "schemas/viability/campaign-v1.schema.json:20-41; scripts/check_viability_campaign.py:510-519, 691-782; tests/unit/test_viability_campaign_contract.py:19-24, 236-253"
    evidence: |-
      The campaign schema constrains candidate, baseline, tree, and protocol values only to
      40 lowercase hexadecimal characters. `_validate_packet` checks only that packet copies
      equal campaign strings. `validate_campaign` never resolves the Git objects, recomputes
      the candidate tree, or verifies that preregistered rule bytes are present at the named
      protocol commit.

      The repository's positive test fixture intentionally uses `a` times 40 for the candidate,
      `b` times 40 for the baseline, `c` times 40 for the tree, and `d` times 40 for the protocol.
      `git cat-file -e` rejected the candidate, baseline, and protocol values as nonexistent,
      while `validate_campaign` accepted the complete synthetic Tier G campaign with no errors.
      A second mutation removed `three-dimensional-recovery` from VIA-700 while retaining
      `popgp-viability-requirements-v1`; `validate_requirements` returned no errors.
    finding: |-
      The executable contract does not bind campaign identity or the authoritative requirements
      to the frozen Git objects it names. Shape-valid invented hashes and same-version registry
      downgrades can pass.
    failure_scenario: |-
      After observing holdout results, an author changes outcome/capability rules or weakens the
      requirements registry, retains the v1 version strings, and fills campaign hashes with
      self-consistent 40-hex placeholders. The local validator accepts the campaign even though
      no preregistered protocol object exists and the recorded tree is unrelated to the candidate.
    consequence: |-
      A passing packet or tier cannot establish that it ran the preregistered candidate, protocol,
      or minimum-capability registry. The freeze and no-post-selection properties remain manual
      assertions at the central decision boundary.
    required_action: |-
      For the local v1 validator, require candidate, baseline, and protocol commits to resolve;
      require `tree_hash` to equal the candidate commit tree; bind campaign/packet protocol rules
      and the requirements document to content present at the named protocol commit (or to explicit
      verified content hashes); and require a version change when authoritative requirements
      semantics change. Add TST-VPLAN-FREEZE-002.
    verification: confirmed-by-execution
    blocking: true

requested_tests:
  - id: "TST-VPLAN-FREEZE-002"
    description: |-
      Starting from a passing campaign, mutate candidate, baseline, protocol, and tree hashes to
      nonexistent or mismatched Git objects; change packet outcome/capability rules relative to
      the named protocol snapshot; and remove or rename a required Tier G capability without a
      requirements-version change. The same campaign-validation command must reject every case.
    rationale: "Demonstrates that the executable campaign is bound to the preregistered Git and requirements content rather than self-consistent labels."
    blocking: true

prior_finding_results:
  - finding_id: "VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      Versioned schemas and a validator now exist, and unknown IDs, missing/hash-mismatched
      receipts, empty decisive receipts, malformed pointers, basic lifecycle contradictions,
      evidence ordering, and declared outcome mismatches are rejected. The central binding and
      closure guarantees remain bypassable. A Tier G fixture whose raw receipt states
      `three_dimensional_recovery: false` and `same_source_lensing: false` is accepted because all
      required capability rules are allowed to be `literal: true` and use no binding, contrary to
      VIABILITY_DEMONSTRATION_PLAN.md:545-550. Changing raw `passed` from Boolean true to integer 1
      is also accepted by the supposedly typed equality comparator. Finally, a hash-valid review
      file containing an unresolved blocker is accepted when the packet self-declares empty result
      arrays, and a superseded blocker can name a nonexistent successor.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The remediation is substantial, but raw-result binding, typed comparison, and byte-reconciled review closure remain fail-open."

  - finding_id: "VPLAN-CUSTODY-001"
    outcome: unresolved
    evidence: |-
      The evaluator/custodian seat, raw-byte manifest hashes, per-seat records, separate-session
      rules, output commitment, reveal timestamps, post-reveal receipts, and retention fields now
      exist. Independent matrices rejected all 9 blind-seat exposure mutations, all 10 prohibited
      session-sharing pairs, all 4 evaluator role-reuse cases, both post-reveal byte substitutions,
      and missing commitment/retention/archive fields. However, setting `authorized_by` to the
      blind builder identity is accepted. More importantly, a packet with revealed manifests is
      accepted at lifecycle `preregistered` with `holdout_started: false` and `not-run/pending`,
      before attack or reproduction. The output-commitment receipt/timestamp is not bound to the
      reproduction runner or to the committed raw output bytes, so the original early-reveal
      failure scenario remains representable as valid.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Hash, retention, exposure, and session enforcement are real partial fixes; reveal authority and execution-order binding remain open."

  - finding_id: "VPLAN-SCI-001"
    outcome: unresolved
    evidence: |-
      The current requirements file names all nine minimum Tier G capabilities, the plan adds the
      missing 3D/same-source/laboratory comparisons, omission of the names is rejected, and an
      explicitly false capability expression prevents pass. But the positive fixture implements
      every capability as `literal: true`. Hash-verified raw results can explicitly record failed
      three-dimensional recovery and failed same-source lensing while those unbound literals let
      the complete Tier G campaign pass. The minimum comparisons therefore exist as registry names
      but are not executable evidence gates over the raw observations.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Tier G wording and capability completeness improved, but the countermodel still passes when false raw results are ignored by literal rules."

  - finding_id: "VPLAN-OUTCOME-001"
    outcome: unresolved
    evidence: |-
      Independent mutations confirmed the intended cause classes for scientific negative,
      implementation-capability failure, unavailable TeX, unavailable CUDA hardware, unavailable
      external access, tested resource exhaustion, protocol-invalid and receipt-invalid rounds,
      simultaneous fail/block ambiguity, and campaign failed-over-blocked precedence. However,
      `packet_outcome: pending` is schema-valid. When paired with `round_status: valid` and a true
      pass rule, `validate_campaign` appends the mismatch and then indexes its allowed-cause table
      by `pending`, raising `KeyError` at scripts/check_viability_campaign.py:649-655 instead of
      returning fail-closed errors. The committed truth-table test also exercises only tested
      exhaustion, toolchain blockage, simultaneous fail/block, and invalid protocol, not the full
      requested matrix.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Core precedence and cause classification are largely correct, but an allowed cross-field contradiction crashes the public validator and its requested regression matrix is incomplete."

  - finding_id: "VPLAN-DEP-001"
    outcome: verified-resolved
    evidence: |-
      The canonical requirements file uses packet IDs only, has no cycles, places every dependency
      in a strictly lower wave, and includes every transitive prerequisite in each tier. Waves now
      run VIA-010/VIA-300 before VIA-100/VIA-150/VIA-200 and preserve VIA-500 before VIA-600.
      `validate_requirements` rejects unknown dependencies, cycles, and same-wave prerequisites;
      campaign validation rejects VIA-100 and VIA-200 holdout start while VIA-010 is pending.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "The current canonical DAG and dependent-holdout ordering satisfy the original action. Requirements-content version integrity is separated into VPLAN-FREEZE-002."

  - finding_id: "VPLAN-EVIDENCE-001"
    outcome: verified-resolved
    evidence: |-
      The campaign-owned registry fixes E4 floors for VIA-300/400/500/600/700/800 and E5 for
      VIA-900. The validator enforces declared >= floor and, for passes, achieved >= declared plus
      cumulative receipt kinds. Independent mutations downgraded all six E4/E5 packets in a Tier E
      campaign and VIA-800 in an extension campaign; all seven were rejected. Unknown levels fail
      schema validation and stricter packet declarations are accepted only with matching achieved
      evidence.
    verification: confirmed-by-execution
    superseding_finding_id: ""
    notes: "Evidence labels remain governance classifications whose semantic truth needs human review, as the builder response discloses; the specific mutable-floor defect is closed."

prior_requested_test_results:
  - requested_test_id: "TST-VPLAN-SCHEMA-001"
    outcome: unresolved
    evidence: |-
      `test_schema_contract_rejects_cross_field_and_receipt_mutations` and the requirements/review
      tests pass, but they do not reconcile declared review outcomes to review receipt bytes, reject
      dangling supersessions, require capability/outcome decisions to depend on raw bindings, or
      enforce strict JSON operand types. The test fixture itself proves that all required
      capabilities may be binding-free literals and every receipt kind may point to the same
      generic evidence JSON.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The requested complete-chain and raw-binding mutations are not satisfied."

  - requested_test_id: "TST-VPLAN-CUSTODY-001"
    outcome: unresolved
    evidence: |-
      The committed custody test rejects one exposure flag, one shared session, custodian/builder
      identity reuse, unsupported canonicalization, late output-commitment time, post-reveal hash
      substitution, missing custodian, and blank retention. Independent expansion confirmed broad
      exposure/session/hash coverage. It does not reject builder-authorized reveal, reveal before
      attack/reproduction/holdout, or a commitment receipt not bound to runner output bytes.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The original leaked-runner/reveal-order counterexample is not closed by the timestamp-only mutation."

  - requested_test_id: "TST-VPLAN-SCI-001"
    outcome: unresolved
    evidence: |-
      The committed test removes capability names and sets one capability rule to `literal: false`,
      both of which are rejected. It does not encode the requested countermodel in raw results and
      require the capability expressions to consume those results. The independently constructed
      raw-false Tier G countermodel is accepted.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "A false expression is rejected; false evidence ignored by a literal-true expression is not."

  - requested_test_id: "TST-VPLAN-OUTCOME-001"
    outcome: unresolved
    evidence: |-
      The committed test covers only four of the requested truth-table families. Independent
      execution showed the implementation handles the omitted named cause classes and precedence,
      but the regression test requested by the review was not added, and a valid/pending
      cross-field case crashes the validator API with `KeyError`.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The test must persist every requested row and assert returned errors for malformed cross-field states."

  - requested_test_id: "TST-VPLAN-DEP-001"
    outcome: verified-satisfied
    evidence: |-
      The committed tests parse the canonical requirements, reject missing dependency IDs, cycles,
      and same-wave prerequisites, and prevent VIA-100/VIA-200 holdout start until VIA-010 passes.
      An independent canonical closure scan found no missing transitive dependency or wave violation.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "Satisfied for the frozen canonical registry and holdout-order invariant."

  - requested_test_id: "TST-VPLAN-EVIDENCE-001"
    outcome: verified-satisfied
    evidence: |-
      The committed test rejects VIA-300/VIA-400 below E4, VIA-900 below E5, achieved below
      declared, and unknown values while allowing a stricter declaration. Independent mutations
      extended the downgrade check to VIA-500/600/700/800; every campaign-owned floor held.
    verification: confirmed-by-execution
    superseding_requested_test_id: ""
    notes: "The requested floor-downgrade cases are satisfied."

predictions:
  experiment_id: "TST-VPLAN-SCHEMA-001, TST-VPLAN-CUSTODY-001, TST-VPLAN-SCI-001, TST-VPLAN-OUTCOME-001, and TST-VPLAN-FREEZE-002"
  predicted_outcome: |-
    A complete remediation will reject binding-free passing capability rules, strict-type
    mismatches, review results not reconciled to hashed artifacts, unauthorized or premature
    reveals, valid/pending contradictions without crashing, nonexistent Git identities, wrong
    trees, post-protocol rule changes, and same-version requirements downgrades.
  predicted_failure_mode: |-
    Patching only the named unit assertions will leave another self-declared label unbound to its
    authoritative bytes. In particular, checking that a binding object exists is insufficient if
    decisive capability expressions may still be literals, and checking timestamps is insufficient
    if the output commitment is not tied to the runner's frozen outputs.
  confidence_statement: |-
    High. Every unresolved condition and the new finding was reproduced as an accepted mutation or
    uncaught exception against the exact frozen commit. This is confidence in contract behavior,
    not confidence in or validation of POPGP's physical hypotheses.

recommendation:
  approve: false
  blocking_findings: 5
  rationale: |-
    Four prior blockers remain open at the executable decision boundary, and one new freeze-integrity
    blocker permits invented or downgraded authoritative objects. The canonical dependency and
    evidence-floor remediations are verified, but approval requires fail-closed raw bindings,
    artifact-reconciled review closure, custody/reveal ordering and authority, total outcome
    validation, and Git/protocol/requirements identity binding. Four prior blocking requested tests
    remain unresolved, two are satisfied, and one new blocking test is requested.
```

## Frozen-tree and diff audit

The worktree was clean and on the requested review branch before any review action.

| Command | Exit | Observed result |
|---|---:|---|
| `git status --short --branch` | 0 | `## review/adversarial-viability-runbook-rereview-1`; no changes |
| `git rev-parse HEAD` | 0 | `fba32f03ba92998e587560793f6118cf618a41f7` |
| `git rev-parse "fba32f03ba92998e587560793f6118cf618a41f7^{tree}"` | 0 | `8b91abe42545921595791d454f585bf68c4c7e4a` |
| `git merge-base --is-ancestor 0bdff136c3c5fba8d8868fdd6355f3f824245a8e fba32f03ba92998e587560793f6118cf618a41f7` | 0 | original candidate is an ancestor |
| `git merge-base --is-ancestor 064a233c426f1c620184809bad3a06d8a531a7d3 fba32f03ba92998e587560793f6118cf618a41f7` | 0 | initial review commit is an ancestor |
| `git diff --name-status 0bdff136..fba32f0` | 0 | 16 files: 8 modified, 8 added |
| `git diff --check 0bdff136..fba32f0` | 0 | no output |

The complete remediation diff was read, including the schemas, requirements registry,
validator, templates, test fixture and mutations, documentation, dependency declarations,
lockfile, preserved initial review, and builder response. The initial-review blob at
`064a233c` is byte-identical to the one at the reviewed commit. Builder dispositions and
reported command summaries were not used as resolution evidence.

## Adversarial mutation evidence

Scratch campaigns were created only under disposable `TemporaryDirectory` paths by importing
the candidate test fixture. Each mutation was validated through the same public
`validate_campaign(..., repo_root=ROOT)` function used by the candidate tests. No candidate file
was changed.

| Mutation/check | Observed result |
|---|---|
| Canonical DAG, tier transitive closure, and lower-wave order | pass |
| Canonical E4/E5 floors and all nine VIA-700 capabilities | pass |
| Explicit `same-source-lensing: {literal: false}` | rejected |
| Raw results say 3D recovery/lensing false; capability rules remain literal true | **accepted** |
| Raw integer `passed: 1` compared `eq` to Boolean true | **accepted** |
| Hashed review bytes contain unresolved blocker; packet results empty | **accepted** |
| Blocking finding superseded by nonexistent ID with arbitrary typed response/re-review receipts | **accepted** |
| Blind exposure flags across three blind seats | 9/9 rejected |
| Shared sessions across five separated seats | 10/10 rejected |
| Evaluator/custodian role reuse with four prohibited execution seats | 4/4 rejected |
| Post-reveal holdout and seed byte substitutions | 2/2 rejected |
| Missing output commitment, retention policy, or archive location | 3/3 rejected |
| Reveal authorized by the builder identity | **accepted** |
| Revealed while preregistered, not run, and holdout not started | **accepted** |
| Scientific/code/resource/access outcome cases and simultaneous ambiguity | intended classifications observed |
| Failed packet plus blocked packet | campaign failure correctly takes precedence |
| Valid round with declared pending outcome | **uncaught `KeyError: 'pending'`** |
| Current E4/E5 packet floor downgrades, including VIA-800 | 7/7 rejected |
| Nonexistent candidate/baseline/protocol objects and unrelated tree | **accepted** |
| Same-version requirements document with 3D capability removed | **accepted** |

The candidate's nine dedicated contract tests pass. Their positive fixture is itself decisive
for several findings: every required capability is a literal true expression, every evidence
kind points to one generic JSON receipt, review result arrays are empty, and all Git hashes are
nonexistent repeated characters, yet the fixture represents a passing campaign.

## Authoritative quality suite

Every command from `.github/workflows/ci.yml` was run against the exact frozen tree.

| Command | Exit | Observed result |
|---|---:|---|
| `uv sync --frozen` | 0 | uv 0.11.11; CPython 3.11.15; locked environment installed |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | 652 lines; balanced braces/environments; no Markdown remnants |
| `uv run pytest -q` | 0 | `168 passed in 47.20s` |
| `uv run python -m examples.physics_qg.chain_1d` | 0 | contiguous blocks; D*=1; artifacts regenerated |
| `uv run python -m examples.physics_qg.grid_2d` | 0 | precision/recall 1.0; D*=2; known Pi_res inadmissibility retained |
| `uv run python -m examples.physics_qg.gravity_well` | 0 | Green-function diagnostic pass; known Pi_res inadmissibility retained |
| `uv run python -m examples.physics_qg.source_law` | 0 | relative-entropy slope 1.999684; modular identity slope 1.0 |
| `uv run python -m examples.physics_qg.source_law_many_body` | 0 | quadratic/Kubo--Mori/Richardson diagnostics reproduced |
| `uv run python -m examples.physics_qg.ca_model` | 0 | PNG/GIF/JSON regenerated; analogy limitations unchanged |
| `uv run python scripts/check_validation_artifacts.py` | 0 | validation contracts and required visuals valid |
| `git status --short` after regeneration | 0 | empty |

`pdflatex` and `nvcc` are unavailable in this Windows reviewer environment. They are not
commands in the authoritative CI suite, and no remediation result claims a native/PDF campaign
pass. Their absence remains an external execution limitation, not evidence for or against the
portable-contract findings above.

## Recommendation

Changes requested: **five unresolved blocking findings** — four prior findings remain open and
one new freeze-integrity finding is added. Four prior blocking requested tests remain unresolved,
two are verified-satisfied, and one new blocking test is requested. The green repository suite and
the real DAG/evidence-floor improvements should be retained, but they do not make the campaign
contract fail closed under the accepted binding, review, custody, outcome, and freeze mutations.
