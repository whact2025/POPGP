# Adversarial viability demonstration plan

This runbook converts POPGP's open scientific questions into falsifiable work packets
that can be assigned to clean adversarial agents. It is compatible with gated
multi-agent systems such as Crucible, but it does not assume a vendor-specific
manifest or command-line interface. An orchestrator may translate the portable packet
schema below into its native format without changing the scientific decision rules.

This document complements, rather than replaces:

- [the claims matrix](CLAIMS_MATRIX.md), which limits what may currently be said;
- [the falsification matrix](FALSIFICATION_MATRIX.md), which records active gates;
- [the theory-to-code gap register](THEORY_CODE_GAP.md), which names missing objects;
- [the acceptance-gate mutation registry](GATE_TEST_REGISTRY.md), which binds gates
  to executable negative controls; and
- [the independent agent review workflow](../governance/AGENT_REVIEW_WORKFLOW.md),
  which governs frozen commits, findings, responses, and re-review.

Agent consensus is not scientific validation. A gate passes only when its frozen
outcome rule is satisfied by inspectable evidence. A review performed by another
model under the same operator or orchestrator remains internal adversarial evidence,
not external replication.

## 1. The decision this campaign must make

"Viability" has three progressively stronger meanings. Every campaign must name one
target before work begins.

| Decision tier | Meaning | Required packets |
|---|---|---|
| **R — mechanism viability** | The proposed source/locality/geometry mechanism survives preregistered controls, is not an artifact of hidden priors or one exact toy instance, and shows a controlled path to refinement. This justifies continued research; it is not a GR result. | `VIA-000`, `VIA-010`, `VIA-100`, `VIA-150`, `VIA-200`, `VIA-300`, `VIA-400` |
| **G — gravitational viability** | Tier R plus an independently measured clock response, a controlled weak-field limit, convergent discrete closure/conservation, and accessible-regime consistency. This supports a gravitational-framework claim, not a completed theory of nature. | Tier R plus `VIA-500`, `VIA-600`, `VIA-700` |
| **E — externally supported core framework** | Tier G has been reproduced by an unaffiliated group using independently written code and a blinded prediction package. | Tier G plus `VIA-900` |

`VIA-800` tests the optional high-curvature/singularity program. It is required for a
singularity-resolution or causal-completeness claim, but not for Tier R. Standard
Model, Born-rule derivation, junction-access deviations, and physical-AI claims remain
separate research programs unless a campaign explicitly adds them.

### Current determination

At the repository state documented in August 2026, POPGP has reproducible finite toy
benchmarks and useful negative results. It has **not demonstrated Tier R mechanism
viability**. The microscopic KMS-energy source is family- and decomposition-dependent;
the perturbed Bell control can receive a false geometric declaration; no refinement
limit, independent clock, intrinsic closure tensor, full-pipeline scalable backend, or
external replication exists. A campaign must retain these as failures or open gates,
not average them into a positive score.

## 2. Evidence levels

Every claimed result carries one evidence level. Higher levels include, rather than
erase, lower-level receipts.

| Level | Required evidence | What it cannot establish |
|---|---|---|
| `E0-proposal` | Equation, definition, or qualitative argument | Executability or correctness |
| `E1-implementation` | Executable code plus unit/identity tests | Scientific discrimination |
| `E2-selected-benchmark` | Reproducible finite example with recorded configuration | Robustness, universality, or continuum behavior |
| `E3-adversarial-suite` | Preregistered families, holdouts, negative controls, mutations, error model, and frozen pass rule | Refinement or external independence unless included explicitly |
| `E4-convergent-replication` | Refinement/scaling evidence and an independently implemented numerical cross-check | Empirical confirmation in nature |
| `E5-external-empirical` | Unaffiliated replication and, where applicable, comparison with experimental bounds | Claims outside the measured regime |

Tier R packets require at least `E3-adversarial-suite`; packets that assert scalable
or continuum behavior require `E4-convergent-replication`. Identity checks, green CI,
and a second agent reading the same code never raise evidence above E1 by themselves.

## 3. Non-negotiable campaign constraints

These constraints come from the framework's own viability conditions and apply to
every packet.

1. **No hidden substrate time.** Every use of `dt`, ordering, phase, action distance,
   wall-clock time, or causal order must appear in a dependency ledger. Ordinary
   unitary time cannot be renamed phase order and counted as a derivation.
2. **Disclose structural priors.** Inference must not read coordinates, labels,
   reference edges, target dimension, or evaluator-only topology. Interaction
   adjacency encoded in a Hamiltonian or state must be declared.
3. **No regional or instance tuning.** Couplings, scale-setting constants, response
   functions, thresholds, and regularizers are frozen globally or selected by a
   preregistered training rule. Holdout cases cannot receive individual tuning.
4. **Separate inference from evaluation.** Ground truth is available only to the
   evaluator. Builders and inference agents receive public calibration cases but not
   holdout labels or secret seeds.
5. **Preserve negative results.** A failed gate remains a first-class artifact. A new
   candidate or narrower claim may supersede it only through a new frozen round.
6. **Distinguish theory failure from infrastructure blockage.** A valid negative
   experiment is `failed`; unavailable scale, precision, hardware, or external access
   is `blocked`. Neither state is a pass.
7. **No threshold fitting after observation.** Statistics, tolerances, exclusion
   rules, resource budgets, and family weights are committed before holdout execution.
8. **Mutations must cross the boundary.** Every new acceptance gate ships with a
   demonstrated negative control that fails the same implementation of the gate.
9. **No vote-based adjudication.** Multiple agents may propose or challenge evidence,
   but the frozen outcome rule decides. The adjudicator may reject an invalid run; it
   may not waive a scientific failure by majority vote.
10. **Agent review is not external validation.** Only `VIA-900` can satisfy external
    replication, and only under the independence conditions in that packet.

## 4. Crucible-compatible campaign lifecycle

Each packet moves through a fail-closed state machine:

```text
drafted -> preregistered -> implemented -> attacked -> reproduced -> adjudicated
                                                                -> passed
                                                                -> failed
                                                                -> blocked
```

- `preregistered` means the hypothesis, null, parameters, holdouts, statistics,
  resource ceiling, and outcome rule are committed at a full Git hash.
- `implemented` means the builder has produced a frozen candidate and public
  calibration evidence.
- `attacked` means a separate falsifier attempted the packet's named attacks and
  recorded both successful and unsuccessful attack attempts.
- `reproduced` means a clean runner executed the protocol from the locked environment
  without using the builder's runtime state or generated files.
- `adjudicated` means a non-builder checked receipts against the frozen rule.
- Any code, data, threshold, or interpretation change after preregistration creates a
  new candidate and invalidates unfinished downstream states.

### Required seats

| Seat | Responsibility | Prohibited action |
|---|---|---|
| Protocol designer | Defines the hypothesis, null, measurement model, resources, and outcome rule | Editing the rule after holdout results are visible |
| Builder | Implements the candidate and public calibration tests | Adjudicating its own packet |
| Falsifier | Constructs counterexamples, boundary cases, mutations, and alternative explanations | Repairing the candidate on its attack branch |
| Statistical auditor | Checks power, uncertainty, multiplicity, leakage, and post-selection | Replacing a missing measurement model with intuition |
| Reproduction runner | Executes the frozen protocol in a clean environment | Reusing builder caches or accepting summary-only evidence |
| Claim auditor | Maps outcomes to `CLAIMS_MATRIX.md` wording | Promoting a failed or blocked gate |
| Adjudicator | Verifies receipts and assigns `passed`, `failed`, or `blocked` | Deciding by model vote or unpublished information |
| Maintainer/PI | Authorizes resources, external gates, and claim changes | Relabeling internal agent agreement as external replication |

One agent may fill multiple design seats before preregistration, but builder,
falsifier, reproduction runner, and adjudicator should use separate clean sessions.
Shared operators, models, prompts, and orchestration must be disclosed.

## 5. Viability work-packet register

| Packet | Question | Current state | Primary dependencies | Tier |
|---|---|---|---|---|
| `VIA-000` | Can an evaluator reproduce the candidate and trust its evidence contracts? | Internal CI/reproduction passes; native CUDA and current PDF evidence remain limited | none | R |
| `VIA-010` | Does the mechanism avoid hidden time, target leakage, and undisclosed topology priors? | Open; code uses ordinary `dt` and Hamiltonians encode interaction graphs | `VIA-000` | R |
| `VIA-100` | Is there a nontrivial, localized, conserved, and sufficiently convention-stable source response? | Partial finite KMS result; localization, averaging, covariance, and refinement open | `VIA-000`, `VIA-010` | R |
| `VIA-150` | Does projection/capacity selection remain nontrivial and universal as scale changes? | Selected finite partition benchmark; retention budget and capacity law are imposed | `VIA-010`, `VIA-300` | R |
| `VIA-200` | Can blind inference recover geometric families while rejecting non-geometric controls? | Selected chain/grid pass; perturbed Bell false positive and broader families open | `VIA-000`, `VIA-010` | R |
| `VIA-300` | Can a scalable backend reproduce the exact observables needed by the full pipeline? | Open; current mean-field path cannot supply MI/QCMI | `VIA-000` | R |
| `VIA-400` | Do intrinsic geometry, dimension, topology, metric, and curvature converge under refinement? | Not demonstrated | `VIA-100`, `VIA-150`, `VIA-200`, `VIA-300` | R |
| `VIA-500` | Does the candidate source predict an independently measured clock and weak-field law? | Numerical graph solve only; source/clock/Newtonian limit open | `VIA-100`, `VIA-300`, `VIA-400` | G |
| `VIA-600` | Do source and intrinsic geometry satisfy convergent closure and conservation? | Not implemented | `VIA-100`, `VIA-400`, `VIA-500` | G |
| `VIA-700` | Are accessible-regime Lorentz, quantum-statistical, and no-signaling constraints satisfied? | Mostly unimplemented or assumed | `VIA-300`, `VIA-400`, `VIA-600` | G |
| `VIA-800` | Does a controlled high-curvature solution remain causally extendible without case tuning? | Not implemented | Tier G | extension |
| `VIA-900` | Can an unaffiliated group reproduce the frozen core prediction independently? | Not attempted | target tier complete internally | E |

## 6. Detailed work packets

### VIA-000 — evidence integrity and reproducible baseline

**Objective:** prove that later failures and passes refer to the same immutable
candidate and can be regenerated without builder state.

**Required work:**

- Freeze the candidate, baseline, tree hash, lockfile, operating-system image, hardware
  profile, random-seed policy, and authoritative CI commands.
- Run the full quality suite from a fresh clone on at least Linux and one independent
  environment; build the manuscript PDF with a declared TeX toolchain.
- Regenerate all JSON and visual artifacts, validate their semantic contracts, and
  archive raw stdout/stderr plus environment metadata.
- For native/scalable claims, run the declared CUDA/native tests on named hardware and
  retain compiler, driver, and device details.
- Mutate a check identity, pass outcome, shape, stable configuration value, visual
  output, and scientific threshold to demonstrate that the evidence checker rejects
  each corruption.

**Pass rule:** every required command succeeds at the frozen hash; clean reruns produce
contract-equivalent artifacts; every corruption is rejected; no required evidence is
available only through an agent summary.

**Failure rule:** an artifact cannot be regenerated, a mutation survives, a run depends
on undeclared local state, or the documented and CI command sets disagree.

### VIA-010 — hidden-prior and atemporality audit

**Objective:** determine whether the claimed pre-geometric/atemporal mechanism relies
on the structures it claims to derive.

**Required work:**

- Produce a machine-readable dependency ledger for every output: coordinates, labels,
  interaction edges, target family, target dimension, `dt`, boundary conditions,
  evaluator labels, and calibration constants.
- Trace these fields dynamically through `Pi_res`, `Pi_loc`, `Pi_geom`, and `Pi_time`.
- Run label permutations, graph isomorphisms, equivalent Hamiltonian encodings,
  boundary changes, and time/order reparameterizations.
- State an operational distinction between phase/action order and external time, or
  narrow the claim to ordinary unitary-time dynamics.
- Give the falsifier evaluator-only topology and ask it to find any leakage into
  inference, logging data-flow evidence rather than only output correlations.

**Pass rule:** inference has no path from evaluator-only data; every encoded structural
prior is disclosed; global reparameterizations behave according to a preregistered
covariance rule; the atemporality claim has an operational test that can fail.

**Failure rule:** target labels or edges reach inference, a claimed derived structure
is supplied as an unreported input, or ordinary `dt` is necessary while the claim still
states that substrate time has been eliminated.

### VIA-100 — source law, localization, and conservation

**Objective:** decide whether any proposed source is more than a finite-family identity
or a convention-dependent energy split.

**Required work:**

- Preregister at least two noncommuting and two commuting Hamiltonian families, affine
  and non-affine perturbations, isospectral controls, equal-energy/different-entropy
  controls, multiple temperatures, boundaries, sizes, locations, and held-out
  couplings.
- Compare raw relative entropy, reduced modular energy, microscopic KMS-energy density,
  and at least one independently derived alternative. Keep known negative candidates.
- Derive or measure a local current and test a discrete continuity equation. Global
  energy conservation alone does not satisfy this requirement.
- Vary admissible local Hamiltonian decompositions and preregister a quantitative
  convention-sensitivity bound or an invariant equivalence class.
- Add temporal coarse-graining and demonstrate that the response survives an interval
  selected without seeing holdout outcomes.
- Cross-check the weakest nonzero susceptibilities using independently implemented
  extended precision and a separate eigensolver or linear-response implementation.
- Carry source normalization, units, sign, uncertainty, and numerical floor through
  the clock solve without refitting per case.

**Pass rule:** one candidate has nonzero discriminating response above the declared
floor, passes its continuity and global-sum laws, remains within the preregistered
convention/temporal/parameter bounds, and predicts holdout behavior with a single
global calibration. The exact tolerances must be justified and frozen before holdout
execution.

**Failure rule:** the signal is an algebraic identity only, disappears under a valid
decomposition or family, violates continuity, requires per-case tuning, or cannot be
distinguished from entropy or numerical-floor confounders.

### VIA-150 — projection stability and capacity universality

**Objective:** show that resolution selection and capacity are nontrivial mechanisms,
not chosen tolerances that encode the desired answer.

**Required work:**

- Replace or derive the retention budget and capacity scale from independently stated
  assumptions; freeze them before topology outcomes are computed.
- Compare exhaustive minima with the proposed local flow/attractor on overlap sizes.
- Sweep probe count, system size, perturbations, interaction family, and resolution;
  report degeneracy and uncertainty in the selected partition.
- Test whether cut capacity follows area-like rather than volume-like scaling on blind
  geometric holdouts and rejects the same law on non-geometric controls.
- Ask the falsifier to construct trivial information-destroying partitions, probe-seed
  attacks, and Hamiltonians for which the desired partition is encoded by the search
  constraints.

**Pass rule:** a preregistered global rule selects nontrivial stable partitions across
holdouts, agrees with the local-flow construction on overlap cases, and produces a
capacity scaling law with model-comparison evidence that survives refinement.

**Failure rule:** success depends on a selected probe seed, imposed target partition,
regional retention tolerance, or a capacity law that is equally supported by volume
scaling or non-geometric controls.

### VIA-200 — blind locality and non-geometric rejection

**Objective:** recover encoded relational structure without evaluator leakage while
refusing low-dimensional geometry where none is identifiable.

**Required work:**

- Freeze training families and hold out sizes, temperatures, couplings, and graph
  families. Include larger chains, square/triangular/3D lattices, long-range models,
  expanders, random regular graphs, shuffled MI, uniform correlations, exact and
  perturbed Bell pairs, and topological-entanglement controls where feasible.
- Implement QCMI/Markov screening or record a justified alternative before evaluating
  the holdouts.
- Report edge precision/recall/F1, distance distortion, dimension, stress, fit status,
  calibration curves, and abstention rate. Ground-truth edges remain evaluator-only.
- Require an explicit `non-geometric` or `not-identifiable` outcome; forcing a best
  dimension on every input is not permitted.
- Predeclare family-balanced aggregation and a maximum false-geometric rate. The known
  perturbed-Bell false positive must count as a failure unless the frozen rule rejects
  it for a reason independent of its label.
- Attack thresholds, dimension penalty, MI scaling, tie handling, node permutations,
  small noise, and adversarial correlation matrices.

**Pass rule:** the frozen inference rule meets preregistered geometric recovery and
distance criteria on holdouts, rejects or abstains on non-geometric controls within the
false-positive bound, and remains permutation-equivariant and perturbation-stable.

**Failure rule:** target geometry appears only at tuned parameters, non-geometric
controls receive stable geometric declarations, evaluator data reaches inference, or
abstention hides one family's failures.

### VIA-300 — scalable-backend equivalence

**Objective:** enable the sizes needed for robustness and refinement without changing
the scientific observable.

**Required work:**

- Select a scalable entangling method that can compute controlled approximations to
  the MI/QCMI, source, current, and geometry inputs used by the exact pipeline.
- On every feasible overlap size, compare states or reduced states, MI/QCMI matrices,
  inferred edges, source profiles, currents, dimension/status, clock potential, and
  gate outcomes against the exact backend.
- Preregister observable-level error budgets and verify convergence with bond dimension,
  truncation threshold, timestep, precision, and hardware.
- Run CPU/native or independent-library cross-checks and CUDA race/determinism tests.
- Measure memory/runtime scaling and show that the backend reaches at least the four
  refinement levels specified by `VIA-400` within the frozen resource budget.

**Pass rule:** all decision-relevant observables and pass/fail outcomes agree within
their frozen error budgets on overlap systems, errors decrease with computational
refinement, and the target size ladder completes reproducibly.

**Failure rule:** the backend substitutes product-state proxies for entangling
observables, changes a scientific gate, lacks a decreasing error estimate, or cannot
reach the preregistered refinement ladder.

### VIA-400 — intrinsic geometry and refinement

**Objective:** show that dimension, metric, topology, and curvature approach stable
intrinsic quantities as resolution increases.

**Required work:**

- Define one physical geometry and source configuration across at least four resolution
  levels, including physical scale matching, boundary conditions, and error transport.
- Construct intrinsic neighborhoods/complexes, edge lengths, hinge deficits, and dual
  volumes without using an embedding-space Delaunay complex as the claimed geometry.
- Track dimension plateaus, distance distortion, metric rank/condition/residual,
  topology invariants, curvature integrals, boundary contributions, and uncertainty.
- Fit a preregistered convergence model and reserve at least one finer level as a
  holdout prediction. A visual trend alone is insufficient.
- Include flat, constant-curvature, boundary, noisy, and non-geometric controls.
- Attack scale matching, complex threshold, neighborhood choice, regularization,
  boundary removal, and the number of levels included in the fit.

**Pass rule:** the same global construction gives stable topology and dimension,
decreasing metric/curvature errors with a supported convergence model, and an accurate
holdout-level prediction; non-geometric controls fail or abstain.

**Failure rule:** no stable window exists, results change qualitatively with admissible
resolution choices, boundary/regularization effects dominate, or the holdout level
does not follow the frozen convergence prediction.

### VIA-500 — operational clock and weak-field limit

**Objective:** connect the source and geometric constraint to an independently
constructed observable clock and a controlled weak-field law.

**Required work:**

- Define a clock observable from subsystem dynamics or an operational protocol; do not
  substitute the simulation step `dt` or the solved potential itself.
- Calibrate coupling and normalization once on public cases, then hold them fixed for
  source magnitude, sign, location, geometry, boundary, and refinement holdouts.
- Measure clock-rate contrasts and redshift independently and compare them with
  `exp(Phi)` predictions including propagated uncertainty.
- Use enough radial scales and refinement levels to discriminate the relevant 3D
  `1/r`, 2D logarithmic, screened, finite-volume, and alternative laws.
- Verify linear superposition in the weak regime and identify the preregistered point
  at which nonlinear corrections become measurable.
- Attack zero-mode policy, boundary image effects, source smearing, calibration
  leakage, sign conventions, and shell selection.

**Pass rule:** a single frozen normalization predicts holdout clock observables and the
correct dimensional weak-field law within stated uncertainty, while alternatives are
rejected by preregistered model comparison.

**Failure rule:** the clock is not independent, the law cannot be distinguished at the
available scales, calibration changes per case, or sign/magnitude disagree beyond the
combined error budget.

### VIA-600 — discrete closure and conservation

**Objective:** test the central gravitational matching condition rather than a
tensor-agnostic placeholder.

**Required work:**

- Define intrinsic discrete geometric and source tensors with units, orientations,
  dual volumes, boundary terms, and the coupling fixed independently of holdouts.
- Implement the discrete closure residual and Bianchi/divergence residual on flat,
  constant-curvature, weak-source, multi-source, and boundary fixtures.
- Demonstrate the source continuity law from `VIA-100` on the same complexes.
- Run the `VIA-400` refinement ladder and preregister the expected convergence orders
  for closure and conservation residuals.
- Mutate tensor signs, incidence orientation, dual volumes, boundary terms, source
  localization, and coupling to show each error is detected.

**Pass rule:** closure, Bianchi/divergence, and source-continuity residuals decrease
under refinement at their frozen rates with one global coupling and correct control
limits; every structural mutation fails.

**Failure rule:** only an integrated scalar can be matched, residuals do not converge,
conservation requires post-hoc projection, or coupling/boundary terms are retuned per
fixture.

### VIA-700 — accessible-regime consistency

**Objective:** ensure the viable core does not violate physics already tested in
accessible regimes.

**Required work:**

- Measure excitation dispersion by direction, polarization/species, scale, and
  boost-like frame construction; estimate limiting speeds and infrared anisotropy.
- Translate deviations into the appropriate current experimental bounds with an
  uncertainty and units audit. Bounds and external data versions must be frozen.
- Verify operational no-signaling for every accessible junction configuration.
- State which quantum probabilities are assumed and which are derived. Standard
  density-matrix sampling cannot be counted as a Born-rule derivation.
- Attack preferred-frame choices, finite-volume dispersion, species-dependent
  calibration, post-selected regimes, and signaling hidden by averaged marginals.

**Pass rule:** all accessible-regime observables meet frozen experimental bounds
without region/species tuning; no-signaling holds in each tested stratum; assumed and
derived quantum-statistical content remains explicitly separated.

**Failure rule:** a robust excluded deviation appears, a common limiting behavior does
not emerge, controllable signaling occurs, or agreement depends on post-selection.

### VIA-800 — controlled high-curvature completeness extension

**Objective:** test causal extendibility in a model whose classical counterpart becomes
incomplete.

**Required work:** first pass the same model's low-curvature `VIA-500` and `VIA-600`
gates; freeze the finite-capacity modification; evolve through the high-curvature
regime; measure invariants, constraints, determinism, causal trajectories, and which
singularity-theorem assumption changes. Include classical recovery, resolution,
initial-data, and alternative-regularization controls.

**Pass rule:** low-curvature behavior converges correctly and one universal rule gives
deterministic continuation with bounded declared invariants, controlled constraints,
and extendible causal trajectories across held-out initial data.

**Failure rule:** continuation requires solution-specific tuning, constraints lose
control, the low-curvature limit is wrong, or causal trajectories still terminate.

### VIA-900 — external clean-room replication

**Objective:** distinguish internal agent hardening from independent scientific
support.

**Required work:**

- Deposit a frozen theory/protocol package, input data, public calibration cases,
  cryptographic hashes, and blinded holdout predictions.
- Recruit an unaffiliated group with a different operator, no shared private evaluator,
  and independently written core implementation. Declare all prior exposure.
- Have that group choose its own numerical methods within the frozen observable and
  outcome contract, reveal holdouts only after its outputs are committed, and publish
  code, raw data, failures, and environment details.
- Resolve discrepancies with a preregistered discriminating experiment; do not merge
  implementations until independent outputs are frozen.

**Pass rule:** the unaffiliated implementation reproduces every target-tier decisive
observable and gate outcome within the frozen cross-implementation uncertainty budget.

**Failure rule:** a decisive outcome disagrees, independence conditions are not met,
or replication depends on copying the original implementation. Agent-only re-review
must be labeled internal even when it uses a different model.

## 7. Portable packet contract

The following is a portable orchestration contract, not an assertion about any
particular Crucible schema. Store one completed file per packet and replace every
placeholder before preregistration.

```yaml
schema_version: 1
campaign_id: POPGP-VIABILITY-<n>
target_tier: R|G|E|extension
packet_id: VIA-<nnn>
state: drafted|preregistered|implemented|attacked|reproduced|adjudicated|passed|failed|blocked

repository: https://github.com/whact2025/POPGP
candidate_commit: <40-hex>
baseline_commit: <40-hex>
tree_hash: <40-hex>
protocol_commit: <40-hex>

claims: [C00]
existing_gates: [GATE-NAME]
dependencies: [VIA-000]
evidence_level_required: E3-adversarial-suite

hypothesis: <single falsifiable statement>
null_or_competitors: []
known_failure_to_retain: <existing negative result or none>
threat_model: []

public_calibration_cases: []
hidden_holdout_manifest_hash: <hash held by evaluator>
secret_seed_manifest_hash: <hash held by evaluator>
frozen_parameters: {}
statistics_and_uncertainty: <procedure>
pass_rule: <executable Boolean rule>
failure_rule: <executable Boolean rule>
blockage_rule: <conditions that mean evidence unavailable rather than pass/fail>
resource_budget: <compute, time, memory, and external-access ceiling>

commands: []
required_artifacts: []
mutation_tests: []

seats:
  protocol_designer: <identity>
  builder: <identity>
  falsifier: <identity>
  statistical_auditor: <identity>
  reproduction_runner: <identity>
  claim_auditor: <identity>
  adjudicator: <identity>
  operator: <human>

access_declaration:
  shared_operator: false
  shared_session: false
  shared_orchestrator: false
  final_labels_seen_by_builder: false
  secret_seeds_seen_by_builder: false
  private_evaluator_seen_by_builder: false
  external_scientific_validation: false

receipts:
  protocol: <path>
  attack_plan: <path>
  raw_results: <path>
  run_log: <path>
  environment: <path>
  statistical_audit: <path>
  independent_review: <path>
  claim_diff: <path>
  adjudication: <path>

decision:
  outcome: pending|passed|failed|blocked
  evidence_level_achieved: E0-proposal
  decisive_receipts: []
  residual_risks: []
  authorized_by: <maintainer-or-PI>
```

The evaluator stores secret manifests outside the builder worktree and commits their
hashes before execution. After adjudication, reveal and archive the manifests unless
an ongoing benchmark requires continued secrecy; in that case record the custodian
and retention policy.

## 8. Required receipt layout

Use immutable, round-numbered artifacts. Do not overwrite a failed run.

```text
reviews/viability/<campaign-id>/
  CAMPAIGN.yaml
  DECISION.md
  <packet-id>/
    PACKET.yaml
    PROTOCOL.md
    ATTACK-PLAN.md
    ENVIRONMENT.json
    RUN-<round>.log
    RESULTS-<round>.json
    STATISTICAL-AUDIT-<round>.md
    CLAIM-DIFF-<round>.md
    ADJUDICATION-<round>.md
```

Large raw arrays may live in a versioned external archive if the repository stores a
content hash, schema, license, retrieval command, and immutable identifier. Plots are
diagnostic views; the decision must be reproducible from machine-readable raw results.

Implementation changes discovered by a packet still use the existing review workflow:
freeze a candidate, commit an independent review artifact, respond finding by finding,
and obtain a clean re-review. Packet adjudication does not bypass code review.

## 9. Copy/paste prompts for clean adversarial agents

### Falsifier prompt

```text
You occupy the falsifier seat for POPGP packet <packet-id> in campaign <campaign-id>.

Read completely:
- docs/scientific_hardening/VIABILITY_DEMONSTRATION_PLAN.md
- <packet-yaml>
- <frozen-protocol>
- docs/governance/AGENT_REVIEW_WORKFLOW.md

Candidate commit: <40-hex>
Protocol commit: <40-hex>
Worktree: <absolute-clean-worktree>
Access declaration: <declaration>

Your objective is to make the frozen scientific gate fail or show that the protocol
cannot decide the stated hypothesis. Do not improve or remediate the candidate. Attack
the assumptions, numerical floor, parameter boundaries, hidden priors, controls,
statistics, invariances, resource ceiling, and artifact contract named in the packet.
Run executable counterexamples where feasible. Record unsuccessful attacks as attack
receipts, but promote only evidence-supported failures to findings. Do not request or
infer holdout labels, secret seeds, or private evaluator logic.

Create <packet-dir>/ATTACK-PLAN.md and an independent review artifact using the
repository template. Commit only review/attack artifacts on the review branch. Return
the full commit hash, commands, observed results, successful falsifiers, protocol
defects, and unresolved attacks. Do not assign the final packet outcome.
```

### Reproduction-runner prompt

```text
You occupy the reproduction-runner seat for POPGP packet <packet-id>.

Start from the exact candidate and protocol commits in PACKET.yaml. Use a clean clone
or verified clean worktree and a newly created locked environment. Do not use builder
caches, generated artifacts, informal instructions, or expected numerical outputs.
Execute the commands exactly within the frozen resource budget. Preserve stdout,
stderr, environment details, raw results, artifact hashes, and any deviation from the
protocol. A command that cannot run is blocked or failed according to PACKET.yaml; it
is never silently skipped. Do not repair code or adjudicate the hypothesis.
```

### Adjudicator prompt

```text
You occupy the adjudicator seat for POPGP packet <packet-id>.

Do not modify implementation or thresholds. Verify commit hashes, access declarations,
protocol freeze, required seats, complete receipts, mutations, hidden-holdout handling,
and reproduction evidence. Recompute the executable pass and failure rules from raw
results. Assign exactly one outcome: passed, failed, or blocked. Model agreement is not
an outcome rule. If the run is invalid, identify the violated preregistration clause
and require a new round. Map the outcome to claims-matrix wording without extending the
claim beyond the evidence level achieved.
```

## 10. Adjudication and stop rules

A packet is **passed** only when every conjunct in its frozen pass rule is true. It is
**failed** when a frozen failure rule is triggered by a valid run. It is **blocked**
only when the declared evidence cannot be obtained inside the resource/access ceiling
and no scientific outcome follows.

The campaign stops and reports the result instead of recursively tuning when:

- a non-negotiable structural constraint fails;
- a decisive holdout falsifies the packet under a valid protocol;
- the measured effect remains below the preregistered precision or power floor;
- no decreasing error or convergence window appears within the frozen refinement
  ladder and resource budget;
- a required full-pipeline observable cannot be computed by the scalable backend; or
- external replication disagrees on a decisive target-tier outcome.

Further research may start a new campaign with a changed hypothesis or mechanism. The
old failure remains in the ledger. A failed auxiliary implementation may be repaired
without abandoning the hypothesis only when the packet's frozen rules classify the
failure as implementation invalidity rather than a valid scientific observation.

### Tier decision rule

The campaign decision is mechanical:

```text
target tier passes
  iff every required packet is adjudicated passed
  and every required evidence level is achieved
  and no non-negotiable constraint is unresolved
  and every blocking independent-review finding is verified resolved.
```

There is no partial-credit viability score. Report packet-level progress separately.
If one required packet is failed, the target tier is failed for that candidate. If one
is blocked, the tier is not demonstrated.

## 11. Recommended execution waves

The ordering below lets independent agents work concurrently without letting later
claims outrun their dependencies.

| Wave | Parallel packets | Exit condition |
|---|---|---|
| 0 — freeze | `VIA-000`, campaign manifest, holdout custody | Reproducible baseline and immutable protocol hashes |
| 1 — foundations | `VIA-010`, `VIA-100`, `VIA-200`, scalable-method design for `VIA-300` | Structural audit complete; source/topology candidates and attacks frozen |
| 2 — scale | `VIA-150`, `VIA-300` | Nontrivial projection rule and full-pipeline size ladder available |
| 3 — convergence | `VIA-400` plus independent numerical cross-checks | Intrinsic refinement gate adjudicated |
| 4 — gravity | `VIA-500`, then `VIA-600` | Independent clock/weak-field and closure gates adjudicated |
| 5 — compatibility | `VIA-700` | Accessible-regime bounds adjudicated |
| 6 — independence | `VIA-900` | Clean-room prediction reveal and replication adjudicated |
| extension | `VIA-800` | Required only for high-curvature claims |

Start with the smallest counterexample capable of refuting a claim, but do not call a
packet passed at that scale unless the packet explicitly requires only that scale.
Parallel agents should receive bounded packet context, the same frozen hashes, and no
other agent's conclusions until their initial artifacts are committed.

## 12. Definition of a completed viability campaign

A campaign is complete when:

- its target tier, candidate, baseline, tree, protocol, holdout-manifest, and seed hashes
  are immutable and recorded;
- every required packet has a final `passed`, `failed`, or `blocked` adjudication;
- commands, environments, raw results, mutations, attack attempts, reviews, and claim
  diffs are durable and independently readable;
- every builder finding and requested test has a later independent outcome;
- negative and blocked results remain visible in the campaign decision;
- the claims matrix, falsification matrix, gate registry, reproducibility record, and
  manuscript wording agree with the adjudicated evidence;
- exact candidate CI and every applicable external/native gate are recorded; and
- a maintainer or PI signs the decision without converting agent consensus into
  empirical validation.

Until those conditions hold, the accurate statement is: **POPGP is an internally
reproducible finite research prototype with explicit falsifiers; its mechanism and
gravitational viability remain to be demonstrated.**
