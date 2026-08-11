# Independent review: POPGP-REVIEW-VIABILITY-PLAN-1

```yaml
artifact_schema_version: 1
review_id: "POPGP-REVIEW-VIABILITY-PLAN-1"
review_kind: initial
reviewer_seat: independent-reviewer
reviewer_model_identity: "OpenAI Codex GPT-5"
reviewer_model_version: "unknown"
reviewer_operator: "same human operator as the builder; name not separately supplied to this reviewer"
review_date: "2026-08-10"
commit_reviewed: "0bdff136c3c5fba8d8868fdd6355f3f824245a8e"
baseline_commit: "70c867552279b74d5ce1a7bc5c50d5a980cf81e6"
prior_review_ref: ""
builder_response_ref: ""
context_hash: "5b73512063229752ca44d0c32e3ee4f545ce096c"
context_hash_method: "git rev-parse \"0bdff136c3c5fba8d8868fdd6355f3f824245a8e^{tree}\""
files_reviewed:
  - "C:/src/POPGP-review-viability-plan-1/docs/governance/AGENT_REVIEW_WORKFLOW.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/governance/REVIEWER_IDENTITY.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/templates/INDEPENDENT_REVIEW_TEMPLATE.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/scientific_hardening/CLAIMS_MATRIX.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/scientific_hardening/FALSIFICATION_MATRIX.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/scientific_hardening/THEORY_CODE_GAP.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/scientific_hardening/GATE_TEST_REGISTRY.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/scientific_hardening/PROJECT_PLAN.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/scientific_hardening/DECISIONS.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/scientific_hardening/REPRODUCIBILITY.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/reviews/LAUNCH_INDEPENDENT_REVIEW.md"
  - "C:/src/POPGP-review-viability-plan-1/docs/framework.md (relevant viability, matching, weak-field, and falsification sections)"
  - "C:/src/POPGP-review-viability-plan-1/.github/workflows/ci.yml"
  - "C:/src/POPGP-review-viability-plan-1/README.md"
  - "git diff 70c867552279b74d5ce1a7bc5c50d5a980cf81e6..0bdff136c3c5fba8d8868fdd6355f3f824245a8e (complete, all four changed files)"
access_level: local/public-repository-only
independence_statement: |-
  This is a fresh reviewer session under the same human operator and root orchestrator as
  the builder. It is process separation, not external scientific independence. The exact
  reviewer identity exposed to this session is OpenAI Codex GPT-5; no version or snapshot
  is exposed, so the version is recorded as `unknown`. The builder identity supplied in the
  handoff is also OpenAI Codex GPT-5, so model separation is false. The review did not receive
  final labels, secret seeds, private evaluator logic, credentials, or private hardware
  profiles. It received the frozen hashes, scope, access boundary, and an instruction not to
  accept builder validation summaries. I independently read the candidate and surrounding
  material and reproduced the authoritative quality suite. None of this constitutes an
  unaffiliated replication or empirical confirmation of a physical claim.

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
  Changes requested with six blocking findings. The candidate is unusually candid about
  POPGP's present scientific status: it preserves the raw-relative-entropy and reduced-modular
  negative results, counts the perturbed-Bell false geometry as a failure, states that Tier R
  has not been demonstrated, reserves external validation for an unaffiliated clean-room
  replication, and keeps singularity claims behind VIA-800. The complete authoritative quality
  suite passes, every new relative Markdown link resolves locally, and the candidate diff has
  no whitespace error.

  The remaining defects are in campaign executability and claim gating. The portable YAML is
  a prose template rather than a validated contract: terminal lifecycle state, decision outcome,
  evidence level, dependencies, receipt existence, and review resolution can contradict one
  another, while no campaign schema or validator exists. Blind custody lacks an evaluator or
  custodian seat and records exposure only for the builder. The fail/blocked rules overlap and
  have no invalid-run outcome or precedence. The recommended waves schedule three packets in
  parallel with prerequisites they formally depend on. Evidence floors are not bound per packet,
  so refinement and external-replication packets can inherit the template's E3 value. Finally,
  Tier G can pass without the framework's own minimum acceleration, lensing/time-delay, and
  laboratory quantum-statistics comparisons and does not require three-dimensional recovery.

findings:
  - id: "VPLAN-SCHEMA-001"
    severity: high
    category: governance
    location: "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md:474-553, 560-586, 670-680"
    evidence: |-
      Section 7 calls the YAML a portable orchestration contract and section 10 calls the tier
      decision mechanical, but the repository contains no schema, parser, validator, or test for
      CAMPAIGN.yaml or PACKET.yaml. A repository-wide search for `CAMPAIGN.yaml`, `PACKET.yaml`,
      `hidden_holdout_manifest_hash`, `blockage_rule`, and `evidence_level_required`, excluding the
      plan itself, returned no implementation or test reference.

      The template duplicates terminal status in `state` and `decision.outcome` without an
      invariant. It also leaves claims, gates, dependencies, evidence levels, Boolean rules,
      artifact paths, and review evidence as unconstrained scalar/list fields. No CAMPAIGN.yaml
      schema is provided even though the tier rule needs campaign-wide packet closure and review
      status. The receipt contract has one `independent_review` path but no response/re-review
      chain, no finding/test outcomes, and no content hashes for ordinary receipts.

      Concrete malformed instance: `state: passed`, `decision.outcome: pending`,
      `evidence_level_required: E4-convergent-replication`,
      `decision.evidence_level_achieved: E0-proposal`, `claims: [C99]`,
      `dependencies: [VIA-999]`, empty decisive receipts, and nonexistent receipt paths all use
      the documented field shapes. The document defines no executable operation that rejects
      this instance or prevents it from being counted by a downstream orchestrator.
    finding: |-
      The packet format cannot enforce the fail-closed, mechanical campaign decision it claims
      to support. It is a useful checklist, but it is not yet an executable packet contract.
    failure_scenario: |-
      Two agents complete otherwise identical packets. One treats `state` as authoritative and
      counts the malformed instance above as passed; the other treats `decision.outcome` and the
      evidence fields as authoritative and rejects it. Both behaviors are compatible with the
      document because no validation algorithm, schema, or source-of-truth rule is defined.
    consequence: |-
      A tier can be promoted with missing evidence, dangling dependencies, stale review blockers,
      or nonexistent receipts. Independent adjudicators cannot reproduce the claimed mechanical
      decision, so the central actionability claim fails closed only by convention, not by contract.
    required_action: |-
      Add versioned machine-readable packet and campaign schemas plus a repository validator.
      Define one authoritative lifecycle/outcome representation; a versioned expression language
      and raw-result bindings for pass/fail/block rules; referential integrity for claim, gate,
      packet, and receipt identifiers; dependency closure; ordered evidence-level comparison;
      receipt existence and content hashing; and the complete independent review/response/re-review
      chain with blocker/test outcomes. Make campaign adjudication run this validator before a
      packet or tier can pass and add the mutations in TST-VPLAN-SCHEMA-001.
    verification: read-only
    blocking: true

  - id: "VPLAN-CUSTODY-001"
    severity: high
    category: governance
    location: "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md:82-84, 123-138, 503-505, 517-558, 625-645"
    evidence: |-
      The plan requires evaluator-only holdout labels and secret seeds, but `seats` has no
      evaluator or custodian. `access_declaration` records final-label, seed, and evaluator
      exposure only for the builder. It has no per-seat session/model/operator/access records,
      no identity or custody chain for the holder of the manifests, no manifest hash algorithm or
      canonicalization, no output-commitment receipt before reveal, and no typed reveal/retention
      record. Lines 555-558 mention custodian and retention policy only as prose after adjudication.

      A targeted field search returned no `evaluator:`, `custodian:`, `hash_algorithm:`,
      `canonicalization:`, `reveal_`, `session_id:`, or `model_identity:` field in the plan.
      The required-seat prose prohibits self-adjudication but lines 136-138 merely say that the
      four execution seats "should" use separate sessions; the schema permits every seat to name
      the same identity while declaring `shared_session: false` once at packet scope.
    finding: |-
      Hidden-label/seed custody and core-seat separation are not auditable from a completed
      packet. The access declaration can be true for the builder while the falsifier, runner,
      statistical auditor, claim auditor, or adjudicator has already seen the secret material.
    failure_scenario: |-
      One operator fills builder, evaluator, falsifier, runner, and adjudicator under separate
      display names. The evaluator reveals labels before the runner commits outputs. The completed
      template still records all builder exposure booleans as false and contains no field in which
      the leak, custodian, reveal event, or shared session must appear.
    consequence: |-
      Holdout results may be tuned or adjudicated with foreknowledge while producing a superficially
      complete packet. Tier E's blinded-prediction and unaffiliated-replication claims are especially
      vulnerable, but the same defect compromises Tier R/G adversarial evidence.
    required_action: |-
      Add an evaluator/custodian seat and per-seat identity, model, operator, session, access, and
      exposure declarations. Define prohibited role combinations as validated invariants. Record
      manifest hash algorithm/canonicalization, custodian, immutable location, output commitment,
      reveal authorization/time, post-reveal manifest hash, and ongoing-secrecy retention policy.
      Require adjudication to verify these receipts and add TST-VPLAN-CUSTODY-001.
    verification: read-only
    blocking: true

  - id: "VPLAN-SCI-001"
    severity: high
    category: claim
    location: "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md:29-38, 328-430; docs/framework.md:1006-1013, 1029-1039, 1041-1048"
    evidence: |-
      Tier G is said to support a `gravitational-framework claim`. VIA-500 requires an
      independently measured clock/redshift and model comparison among 3D 1/r, 2D logarithmic,
      screened, finite-volume, and alternative potential laws. It does not require gravitational
      acceleration/geodesic motion, light bending, Shapiro time delay, or the framework's
      two-potential same-source consistency test. VIA-700 asks that assumed and derived probability
      content be separated, but it does not require standard laboratory interference and
      entanglement statistics to be reproduced.

      The surrounding framework calls acceleration plus redshift, light bending plus time delay,
      clock mapping, and laboratory quantum statistics "A minimal set of comparisons that any
      instantiation must pass" at lines 1006-1013. Its practical matrix additionally names
      emergent 3D space and weak-field redshift/lensing at lines 1041-1048. The candidate plan
      neither incorporates these gates nor explicitly narrows Tier G below that framework
      viability standard. Because VIA-400 accepts "one physical geometry" and VIA-500 treats the
      2D logarithmic law as an admissible alternative, a stable two-dimensional candidate can
      satisfy the written Tier G rules.
    finding: |-
      Tier G is not complete enough to justify its advertised claim scope and is inconsistent with
      the repository's own minimum empirical-compatibility requirements.
    failure_scenario: |-
      A candidate recovers a stable 2D intrinsic complex, predicts the correct 2D logarithmic
      clock potential and redshift, closes its chosen 2D tensors, meets dispersion/no-signaling
      bounds for the quantities it elects to report, and explicitly says the Born rule is assumed.
      It therefore passes VIA-400/500/600/700 as written even if it predicts wrong acceleration,
      wrong lensing and Shapiro delay for the same source, and wrong laboratory interference or
      entanglement statistics. The tier can still be labeled gravitationally viable.
    consequence: |-
      Campaign success can promote a scientifically incomplete or lower-dimensional toy mechanism
      beyond the claim scope permitted by the framework and claims matrix.
    required_action: |-
      Either narrow and rename Tier G so it does not imply the framework's accessible-regime
      gravitational viability, or add frozen gates for three-dimensional recovery, acceleration
      and geodesic response, lensing and Shapiro delay for the same source, two-potential
      consistency under one calibration, and standard laboratory interference/entanglement
      statistics. Bind each to a packet, pass/failure rule, evidence floor, control/mutation, and
      claim wording. Add the countermodel test in TST-VPLAN-SCI-001.
    verification: read-only
    blocking: true

  - id: "VPLAN-OUTCOME-001"
    severity: high
    category: governance
    location: "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md:87-96, 102-121, 178-183, 320-326, 480-553, 630-645, 648-668"
    evidence: |-
      The plan correctly says scientific failure and infrastructure blockage differ, but it does
      not make failure, blockage, and run invalidity mutually exclusive. The schema has no
      `invalid` outcome. The adjudicator prompt says an invalid run requires a new round, yet the
      only decision values are pending/passed/failed/blocked and campaign completion requires a
      final terminal outcome for every packet.

      The rules can overlap. VIA-000 says inability to regenerate an artifact is failure, while
      the global rule says unavailable toolchains/hardware/access are blocked. VIA-300 says failure
      includes inability to reach the resource-bounded ladder, while the blockage definition also
      covers evidence unobtainable inside the resource ceiling. `failure_rule` and `blockage_rule`
      are independent free-text expressions with no evaluation order or exclusivity check.
    finding: |-
      Outcome adjudication is not deterministic for invalid or resource-limited runs, so the
      promised separation of theory failure from infrastructure blockage is not operational.
    failure_scenario: |-
      A frozen VIA-000 run cannot rebuild the required PDF because its declared TeX image cannot
      be obtained within the access ceiling. `artifact cannot be regenerated` triggers the packet
      failure rule; `evidence cannot be obtained inside the resource/access ceiling` triggers the
      blockage rule. A second example is a VIA-300 backend that exhausts the frozen budget before
      level four: the packet failure rule and generic blockage rule again admit different outcomes.
      If the protocol itself is malformed, the adjudicator is told to start a new round but cannot
      encode `invalid` as the run's terminal status.
    consequence: |-
      Equivalent evidence can be labeled failed, blocked, or left pending by different
      adjudicators. Scientific negatives may be laundered as infrastructure blocks, while genuine
      access limitations may be misreported as theory falsifications.
    required_action: |-
      Define a mutually exclusive decision procedure with explicit precedence and cause codes.
      Represent invalid execution/invalid protocol as a round outcome distinct from the scientific
      packet outcome, and state when candidate defects count as a valid capability failure versus
      invalid evidence. Require failure and blockage expressions to be disjoint or have a frozen
      priority rule. Add the truth-table cases in TST-VPLAN-OUTCOME-001.
    verification: read-only
    blocking: true

  - id: "VPLAN-DEP-001"
    severity: medium
    category: governance
    location: "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md:140-155, 686-705"
    evidence: |-
      The packet register declares VIA-100 and VIA-200 dependent on VIA-010, and VIA-150 dependent
      on VIA-300. The recommended execution table nevertheless labels VIA-010/VIA-100/VIA-200 as
      parallel in wave 1 and VIA-150/VIA-300 as parallel in wave 2. A line-by-line PowerShell
      parser mapping registry dependencies to wave ordinals reported:

        DEPENDENCY_WAVE_CHECK: FAIL
        VIA-100 scheduled wave 1 depends on VIA-010 scheduled wave 1
        VIA-150 scheduled wave 2 depends on VIA-300 scheduled wave 2
        VIA-200 scheduled wave 1 depends on VIA-010 scheduled wave 1

      The same mechanical check also flags VIA-600/VIA-500 in wave 4, but the human-readable cell
      explicitly says `VIA-500, then VIA-600`, so that fourth match is not treated as a defect.
      VIA-800 and VIA-900 use conceptual dependencies (`Tier G`, `target tier complete internally`)
      rather than packet IDs, and the schema does not define how those predicates resolve.
    finding: |-
      The execution schedule violates its own packet dependency order and mixes packet-ID
      dependencies with undefined tier predicates.
    failure_scenario: |-
      VIA-100 and VIA-200 attack holdouts while VIA-010 is still discovering time/topology leakage.
      VIA-010 then fails or changes the dependency ledger, invalidating both downstream runs. In
      wave 2, VIA-150 freezes a refinement/capacity protocol before VIA-300 proves which observables
      and ladder are available. The campaign must discard and rerun work that the runbook explicitly
      described as safe parallel execution.
    consequence: |-
      Clean agents can follow the recommended waves exactly and still produce dependency-invalid
      receipts or consume holdouts before prerequisite rules are frozen.
    required_action: |-
      Make dependencies machine-resolvable packet/outcome predicates, require prerequisite packets
      to be adjudicated passed before dependent holdout execution, and split waves 1 and 2 into
      design versus execution subwaves or move the dependent packets later. Preserve the explicit
      VIA-500-then-VIA-600 ordering. Add TST-VPLAN-DEP-001.
    verification: confirmed-by-execution
    blocking: true

  - id: "VPLAN-EVIDENCE-001"
    severity: medium
    category: claim
    location: "docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md:50-66, 302-353, 450-472, 480-553, 670-684"
    evidence: |-
      Lines 64-66 require E4 for scalable or continuum assertions. VIA-300 asserts scalable-backend
      equivalence and a four-level size ladder, VIA-400 asserts refinement/convergence, and VIA-900
      asserts unaffiliated independent replication. Yet the only packet template value is the
      literal `evidence_level_required: E3-adversarial-suite`; no packet-to-minimum-evidence matrix
      binds VIA-300/400 to E4 or VIA-900 to E5. The decision fields accept any achieved level, and
      no order or comparison algorithm is defined.

      This is not cured by the prose sentence alone: a completed VIA-400 packet copied from the
      provided contract can retain E3, satisfy its custom pass rule, and be counted by the tier rule
      because the tier checks the packet's self-declared required level rather than a campaign-owned
      floor.
    finding: |-
      Required evidence levels are mutable packet inputs instead of tier-owned minimum gates, so a
      packet can downgrade the evidence needed for the claim it adjudicates.
    failure_scenario: |-
      A VIA-400 author leaves the template at E3 and reports a four-point trend without the
      independently implemented cross-check required by E4. The packet's frozen pass rule returns
      true and records E3 achieved. Every conjunct in the tier rule can then be true even though the
      plan's evidence taxonomy says the refinement claim requires E4. VIA-900 can be similarly
      mislabeled below E5.
    consequence: |-
      Tier R, G, or E can pass with evidence weaker than the plan's own claim hierarchy requires.
    required_action: |-
      Define immutable minimum evidence levels per packet and tier in CAMPAIGN.yaml/schema, at
      least E4 for VIA-300/400 and E5 for VIA-900 unless the packet/level definitions are revised.
      Validate `achieved >= campaign minimum >= packet-declared requirement`; a packet may raise but
      never lower the campaign floor. Define the evidence-level ordering and required receipt types,
      then add TST-VPLAN-EVIDENCE-001.
    verification: read-only
    blocking: true

requested_tests:
  - id: "TST-VPLAN-SCHEMA-001"
    description: |-
      Add executable schema/validator mutations for contradictory lifecycle/decision values,
      achieved-below-required evidence, unknown claim/gate/packet IDs, dependency cycles or missing
      prerequisites, nonexistent or hash-mismatched receipts, empty decisive receipts, unresolved
      blocking reviews, and an incomplete review/response/re-review chain. Every mutation must make
      the same campaign-adjudication command fail closed.
    rationale: "Demonstrates that the portable contract, not human convention, enforces the mechanical tier rule."
    blocking: true

  - id: "TST-VPLAN-CUSTODY-001"
    description: |-
      Exercise a blinded mock packet with separate builder, evaluator/custodian, falsifier, runner,
      and adjudicator receipts. Mutate each per-seat exposure flag, reuse a prohibited session/role,
      change manifest bytes/canonicalization, reveal before output commitment, omit custodian or
      retention data, and substitute the post-reveal manifest. Each mutation must be rejected.
    rationale: "Shows that hidden labels/seeds and role independence have a verifiable custody chain."
    blocking: true

  - id: "TST-VPLAN-SCI-001"
    description: |-
      Encode a Tier G countermodel that passes the present clock, 2D logarithmic potential, closure,
      dispersion, and no-signaling clauses but deliberately fails three-dimensional recovery,
      acceleration/geodesic response, same-source lensing/Shapiro/two-potential consistency, and
      laboratory interference/entanglement statistics. The revised campaign validator must prevent
      a gravitational-viability pass or require explicitly narrowed tier wording.
    rationale: "Pins the advertised Tier G scope to the repository's own minimum compatibility tests."
    blocking: true

  - id: "TST-VPLAN-OUTCOME-001"
    description: |-
      Add an adjudication truth table covering valid scientific negative; code defect that validly
      fails a capability claim; unavailable TeX/CUDA/external access; resource exhaustion that is
      itself the tested scalability result; protocol-invalid run; missing receipt; and simultaneous
      failure/blockage predicates. Assert one deterministic round status and packet outcome for each.
    rationale: "Prevents scientific failure, infrastructure blockage, and invalid evidence from being interchanged."
    blocking: true

  - id: "TST-VPLAN-DEP-001"
    description: |-
      Parse the packet registry and execution waves as a DAG. Reject same/later-wave prerequisites
      unless an explicit intra-wave `then` edge exists; reject conceptual dependency strings without
      a defined resolver; reject cycles; and prevent dependent holdout execution until prerequisite
      adjudication is passed.
    rationale: "Makes the recommended parallel schedule consistent with the declared scientific dependencies."
    blocking: true

  - id: "TST-VPLAN-EVIDENCE-001"
    description: |-
      Mutate VIA-300 and VIA-400 required/achieved evidence below E4 and VIA-900 below E5 while all
      scientific Boolean clauses are true. The campaign decision must remain non-passing. Also test
      unknown evidence values and a packet requirement stricter than the campaign floor.
    rationale: "Prevents self-declared packet metadata from downgrading the evidence required by a tier."
    blocking: true

prior_finding_results: []
prior_requested_test_results: []

predictions:
  experiment_id: "TST-VPLAN-SCHEMA-001 through TST-VPLAN-EVIDENCE-001"
  predicted_outcome: |-
    The current document-only contract has no command that rejects the malformed packet, custody,
    outcome, dependency, countermodel, or evidence-downgrade cases. A remediated executable contract
    will reject every named mutation before a packet or tier is marked passed.
  predicted_failure_mode: |-
    A validator that checks only YAML syntax, required keys, or individual enum membership will still
    accept cross-field contradictions, leaked custody, dependency-invalid schedules, insufficient
    scientific gates, and downgraded evidence.
  confidence_statement: |-
    High for the governance and dependency counterexamples because the relevant fields and validator
    are absent and the dependency mismatch was mechanically reproduced. Moderate-to-high for the
    exact Tier G remedy: the omitted comparisons are explicit framework requirements, while maintainers
    may instead choose to narrow the tier wording rather than add every proposed gate.

recommendation:
  approve: false
  blocking_findings: 6
  rationale: |-
    The candidate is scientifically candid and repository-consistent, but it is not yet an executable,
    fail-closed adversarial campaign contract. Six unresolved blockers allow contradictory packet/tier
    decisions, unauditable blinded custody, incomplete Tier G promotion, ambiguous failure classification,
    dependency-invalid execution, or evidence-level downgrade. Approval requires independent verification
    of all six findings and requested tests after remediation.
```

## Method and executed evidence

The reviewer branch was clean and pointed at the exact candidate before review:

| Command | Exit | Observed result |
|---|---:|---|
| `git status --short --branch` | 0 | `## review/adversarial-viability-runbook-1` and no file changes |
| `git rev-parse HEAD` | 0 | `0bdff136c3c5fba8d8868fdd6355f3f824245a8e` |
| `git rev-parse "0bdff136c3c5fba8d8868fdd6355f3f824245a8e^{tree}"` | 0 | `5b73512063229752ca44d0c32e3ee4f545ce096c` |
| `git diff --stat 70c867552279b74d5ce1a7bc5c50d5a980cf81e6 0bdff136c3c5fba8d8868fdd6355f3f824245a8e` | 0 | 4 files, 742 insertions, 2 deletions |
| `git diff --name-status 70c867552279b74d5ce1a7bc5c50d5a980cf81e6 0bdff136c3c5fba8d8868fdd6355f3f824245a8e` | 0 | README, review launch runbook, project plan modified; viability plan added |
| `git diff --check 70c867552279b74d5ce1a7bc5c50d5a980cf81e6 0bdff136c3c5fba8d8868fdd6355f3f824245a8e` | 0 | no output |

All eleven files required by the handoff were read completely. The complete candidate diff was
read, not inferred from its summary. Relevant framework, decision, reproducibility, and CI material
was then inspected to test consistency and claim completeness.

### Authoritative quality suite

| Command | Exit | Observed result |
|---|---:|---|
| `uv sync --frozen` | 0 | CPython 3.11.15; locked environment installed |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run python scripts/check_tex.py` | 0 | 652 lines; brace balance 0; environments matched; no Markdown remnants |
| `uv run pytest -q` | 0 | `159 passed in 35.37s` |
| `uv run python -m examples.physics_qg.chain_1d` | 0 | selected contiguous blocks; D*=1; artifacts regenerated |
| `uv run python -m examples.physics_qg.grid_2d` | 0 | precision/recall 1.0; D*=2; known Pi_res inadmissibility retained |
| `uv run python -m examples.physics_qg.gravity_well` | 0 | Green-function diagnostic pass; known Pi_res inadmissibility retained |
| `uv run python -m examples.physics_qg.source_law` | 0 | relative-entropy slope 1.999684; modular identity slope 1.0 |
| `uv run python -m examples.physics_qg.source_law_many_body` | 0 | quadratic/Kubo--Mori/Richardson diagnostics reproduced |
| `uv run python -m examples.physics_qg.ca_model` | 0 | artifacts regenerated; retained campaign-level analogy limitations unchanged |
| `uv run python scripts/check_validation_artifacts.py` | 0 | `Validation artifact contracts and required visual outputs are valid.` |
| `git status --short` after the suite | 0 | empty |

`pdflatex` and `nvcc` were unavailable in this Windows review environment. They are not commands in
the authoritative CI suite and the candidate contains no native/PDF claim result to validate. Their
absence is a documented review limitation, not a candidate finding; the proposed VIA-000 packet
correctly requires those receipts when a campaign makes the corresponding claims.

### Local Markdown-link check

A PowerShell scan extracted every non-image Markdown link from the four changed Markdown files,
skipped `http`, `mailto`, and fragment-only targets, resolved relative targets against each source
file, and applied `Test-Path`. Result:

```text
LOCAL_MARKDOWN_LINK_CHECK: PASS
all relative link targets in changed Markdown files exist
```

### Dependency mutation/omission check

The packet table and wave table were parsed directly from the candidate document. Packet IDs were
mapped to wave ordinals and every packet-ID dependency was required to have a lower ordinal. Exact
decisive output:

```text
DEPENDENCY_WAVE_CHECK: FAIL
VIA-100 scheduled wave 1 depends on VIA-010 scheduled wave 1
VIA-150 scheduled wave 2 depends on VIA-300 scheduled wave 2
VIA-200 scheduled wave 1 depends on VIA-010 scheduled wave 1
VIA-600 scheduled wave 4 depends on VIA-500 scheduled wave 4
```

The last row is not a finding because its wave cell explicitly specifies `VIA-500, then VIA-600`.
No such ordering qualification exists for the other three rows. A second repository search found
no packet/campaign validator outside the candidate plan. A schema-field omission scan found no
typed evaluator, custodian, hash algorithm/canonicalization, reveal, per-seat session/model,
builder-response, re-review, or blocking-finding field.

## Adversarial counterexamples

The following examples distinguish limitations already documented by the candidate from defects in
the runbook itself.

1. **Contradictory but shape-valid packet.** A packet can say `state: passed` and
   `decision.outcome: pending`, require E4 but achieve E0, cite C99/VIA-999, omit decisive receipts,
   and point at nonexistent paths. The current contract has no rejection operation. This supports
   VPLAN-SCHEMA-001; it is not merely the documented limitation that the contract is vendor-neutral.
2. **Blindness declaration with a leaked runner.** All builder exposure booleans remain false after
   a reproduction runner sees labels before committing outputs. There is no runner exposure field,
   evaluator seat, or reveal receipt. This supports VPLAN-CUSTODY-001.
3. **Tier G countermodel.** A stable 2D model with correct logarithmic clock behavior, closed 2D
   tensors, and no signaling can pass current written rules despite wrong acceleration, lensing,
   Shapiro delay, and lab quantum statistics. This supports VPLAN-SCI-001; it does not dispute the
   candidate's honest statement that present POPGP has not reached Tier R.
4. **One observation, two outcomes.** Missing TeX access makes a PDF non-regenerable. The VIA-000
   failure rule says failed, while the global access-ceiling rule says blocked. No invalid/precedence
   outcome exists. This supports VPLAN-OUTCOME-001.
5. **Dependency-invalid clean execution.** Agents follow the advertised parallel waves exactly;
   VIA-010 later changes or fails the dependency ledger and invalidates already exposed VIA-100/200
   holdouts. This supports VPLAN-DEP-001.
6. **Evidence downgrade.** A copied VIA-400 packet retains E3, passes its own rule, and is accepted
   without the independent numerical cross-check that the taxonomy requires at E4. This supports
   VPLAN-EVIDENCE-001.

## What is not a defect

- The plan preserves and prominently names the perturbed-Bell false geometry, raw-relative-entropy
  failure, reduced-modular localization failure, mean-field MI limitation, missing independent
  clock, absent closure, and absent external replication. No negative result was silently removed.
- Tier R, G, E, and the high-curvature extension are separated, and VIA-800 is correctly required
  before singularity-resolution or causal-completeness wording.
- The plan consistently says internal agent consensus is not external scientific validation.
- Existing claims remain scoped as finite toy-model results; the candidate does not claim that
  publishing this plan demonstrates viability.
- A vendor-neutral portable contract is reasonable. The defect is the absence of a defined
  executable validation/translation conformance target, not the absence of a Crucible-specific CLI.

## Recommendation

Changes requested: **six unresolved blocking findings**. The candidate should retain its tier
structure, negative-result honesty, adversarial roles, and receipt-first approach while making the
contract executable and closing the six fail-closed gaps above. A later independent reviewer must
verify every finding and requested test against a new frozen candidate; builder statements alone do
not resolve this review.
