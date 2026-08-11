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
| **G — gravitational viability** | Tier R plus 3D recovery, an independently measured clock, one-calibration acceleration/geodesic/redshift/lensing/Shapiro comparisons, convergent closure/conservation, laboratory quantum-statistics compatibility, and accessible-regime Lorentz/no-signaling consistency. This supports a gravitational-framework claim, not a completed theory of nature. | Tier R plus `VIA-500`, `VIA-600`, `VIA-700` |
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

Each packet has a lifecycle phase separate from its adjudicated outcome:

```text
drafted -> preregistered -> implemented -> attacked -> reproduced -> adjudicated

adjudicated round_status: valid | invalid
valid packet_outcome:     passed | failed | blocked
invalid packet_outcome:   pending (a new round is required)
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
- `invalid` records unusable evidence or protocol without turning it into a scientific
  failure or infrastructure blockage. It is a round status, not a packet outcome.
- Valid pass, fail, and blockage expressions must be mutually exclusive. If zero or
  more than one evaluates true, the round is invalid with `rule-ambiguous`.
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
| Evaluator/custodian | Commits hidden manifest hashes, protects labels/seeds, receives output commitments, and authorizes reveal | Building, running, falsifying, or adjudicating the same packet |
| Maintainer/PI | Authorizes resources, external gates, and claim changes | Relabeling internal agent agreement as external replication |

One agent may fill multiple design seats before preregistration, but builder,
falsifier, reproduction runner, adjudicator, and evaluator/custodian **must** use
separate validated session IDs. The evaluator/custodian cannot reuse the agent identity
of any of those execution seats. Every seat records model, version, operator, session,
access, and hidden-data exposure separately; shared operators, prompts, and
orchestration remain disclosed.

## 5. Viability work-packet register

| Packet | Question | Current state | Dependencies | Wave | Floor |
|---|---|---|---|---:|---|
| `VIA-000` | Can an evaluator reproduce the candidate and trust its evidence contracts? | Internal CI/reproduction passes; native CUDA and current PDF evidence remain limited | none | 0 | E3 |
| `VIA-010` | Does the mechanism avoid hidden time, target leakage, and undisclosed topology priors? | Open; code uses ordinary `dt` and Hamiltonians encode interaction graphs | `VIA-000` | 1 | E3 |
| `VIA-100` | Is there a nontrivial, localized, conserved, and sufficiently convention-stable source response? | Partial finite KMS result; localization, averaging, covariance, and refinement open | `VIA-000`, `VIA-010` | 2 | E3 |
| `VIA-150` | Does projection/capacity selection remain nontrivial and universal as scale changes? | Selected finite partition benchmark; retention budget and capacity law are imposed | `VIA-010`, `VIA-300` | 2 | E3 |
| `VIA-200` | Can blind inference recover geometric families while rejecting non-geometric controls? | Selected chain/grid pass; perturbed Bell false positive and broader families open | `VIA-000`, `VIA-010` | 2 | E3 |
| `VIA-300` | Can a scalable backend reproduce the exact observables needed by the full pipeline? | Open; current mean-field path cannot supply MI/QCMI | `VIA-000` | 1 | E4 |
| `VIA-400` | Do intrinsic geometry, dimension, topology, metric, and curvature converge under refinement? | Not demonstrated | `VIA-100`, `VIA-150`, `VIA-200`, `VIA-300` | 3 | E4 |
| `VIA-500` | Does the candidate source predict an independently measured clock and weak-field law? | Numerical graph solve only; source/clock/Newtonian limit open | `VIA-100`, `VIA-300`, `VIA-400` | 4 | E4 |
| `VIA-600` | Do source and intrinsic geometry satisfy convergent closure and conservation? | Not implemented | `VIA-100`, `VIA-400`, `VIA-500` | 5 | E4 |
| `VIA-700` | Does a 3D candidate satisfy gravitational and accessible-regime comparisons? | Mostly unimplemented or assumed | `VIA-300`, `VIA-400`, `VIA-600` | 6 | E4 |
| `VIA-800` | Does a controlled high-curvature solution remain causally extendible without case tuning? | Not implemented | all Tier G packets | 7 | E4 |
| `VIA-900` | Can an unaffiliated group reproduce the frozen core prediction independently? | Not attempted | all Tier G packets | 7 | E5 |

The authoritative dependency lists, waves, capabilities, tier membership, evidence
ordering, and floors are machine-owned by
[`requirements-v2.json`](../../schemas/viability/requirements-v2.json). Packet authors
may raise an evidence requirement but cannot lower that campaign floor.

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

### VIA-700 — three-dimensional gravity and accessible-regime consistency

**Objective:** require the Tier G candidate to recover three-dimensional behavior and
the framework's own minimum same-source gravitational and laboratory comparisons.

**Required work:**

- Demonstrate a stable three-dimensional intrinsic geometry under the `VIA-400`
  refinement and non-geometric-control rules. A successful 2D logarithmic model can be
  a lower-dimensional result but cannot pass Tier G.
- Predict acceleration/geodesic response and clock redshift from one frozen source and
  calibration, rather than testing the potential alone.
- Predict light bending and Shapiro time delay for that same source. Test both metric
  potentials and their consistency without separately fitting lensing and clock data.
- Measure excitation dispersion by direction, polarization/species, scale, and
  boost-like frame construction; estimate limiting speeds and infrared anisotropy.
- Translate deviations into the appropriate current experimental bounds with an
  uncertainty and units audit. Bounds and external data versions must be frozen.
- Verify operational no-signaling for every accessible junction configuration.
- Reproduce preregistered standard laboratory interference and entanglement
  statistics. If the Born rule is assumed, say so; matching under an assumed sampling
  rule is compatibility evidence, not a derivation.
- State which quantum probabilities are assumed and which are derived. Standard
  density-matrix sampling cannot be counted as a Born-rule derivation.
- Attack dimensional post-selection, preferred-frame choices, finite-volume
  dispersion, species-dependent calibration, independent retuning of the two metric
  potentials, post-selected regimes, and signaling hidden by averaged marginals.

**Pass rule:** the same globally calibrated 3D candidate predicts acceleration,
geodesics, redshift, lensing, and Shapiro delay within frozen uncertainty; its two
metric potentials obey the preregistered consistency relation; laboratory interference
and entanglement statistics match their frozen controls; all Lorentz/no-signaling
observables meet current bounds without region/species tuning; and assumed versus
derived quantum-statistical content remains explicit.

**Failure rule:** 3D recovery fails, any same-source gravitational comparison disagrees,
the metric potentials require separate tuning, laboratory quantum statistics disagree,
a robust excluded deviation appears, a common limiting behavior does not emerge,
controllable signaling occurs, or agreement depends on post-selection.

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
  no shared orchestrator, and independently written core implementation. Declare all
  prior exposure.
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

The portable contract is executable and vendor-neutral. It is not an assertion about
any particular Crucible schema. Its authoritative components are:

- [`campaign-v2.schema.json`](../../schemas/viability/campaign-v2.schema.json), which
  defines `CAMPAIGN.yaml`;
- [`packet-v2.schema.json`](../../schemas/viability/packet-v2.schema.json), which
  defines every `PACKET.yaml`, seat, custody record, receipt, review chain, expression,
  and adjudication;
- [`protocol-manifest-v2.schema.json`](../../schemas/viability/protocol-manifest-v2.schema.json),
  which binds the campaign, canonical requirements, contract files, and every packet's
  preregistered rules to bytes present at the named protocol commit;
- [`primary-protocol-v1.schema.json`](../../schemas/viability/primary-protocol-v1.schema.json),
  which closes the authoritative experiment envelope and rejects competing top-level
  thresholds, exclusions, commands, or resource overrides;
- [`independent-review-v2.schema.json`](../../schemas/viability/independent-review-v2.schema.json),
  [`review-response-v2.schema.json`](../../schemas/viability/review-response-v2.schema.json),
  and [`independent-rereview-v2.schema.json`](../../schemas/viability/independent-rereview-v2.schema.json),
  which make reviewer identity, independence, evidence, builder disposition,
  verification, references, and recommendation fields executable rather than optional
  prose;
- [`requirements-v2.json`](../../schemas/viability/requirements-v2.json), which owns
  tier membership, dependencies, execution waves, required capabilities, evidence
  ordering/floors, and receipt kinds;
- [`VIABILITY_CAMPAIGN_TEMPLATE.yaml`](../templates/VIABILITY_CAMPAIGN_TEMPLATE.yaml)
  [`VIABILITY_PACKET_TEMPLATE.yaml`](../templates/VIABILITY_PACKET_TEMPLATE.yaml), and
  [`VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json`](../templates/VIABILITY_PROTOCOL_MANIFEST_TEMPLATE.json);
  and
- [`check_viability_campaign.py`](../../scripts/check_viability_campaign.py), the
  fail-closed validator and campaign decision implementation.

Contract v2 is a breaking replacement for the reviewed v1 draft. The v1 schemas were
withdrawn because their literals, review summaries, custody order, and hash fields were
not fail closed; no v1 packet or campaign may be promoted or silently translated to v2.
The current `popgp-packet-freeze-v4` digest also freezes preregistered seat organization
and typed external-replication content; older draft freeze digests must be recomputed
before holdout execution.

Copy the templates into a new campaign and duplicate the packet template for every
required packet. Populate the candidate/baseline/tree identities, rules, raw-result
bindings, seat assignments, hidden-manifest commitments, and the complete
`preregistration` block before any holdout work. That block records canonical
parameters, measurement and uncertainty procedures, statistical analysis, resource
budget, exact commands, mutation plan, and one or more protocol-artifact content
references. Each referenced blob must already be stored at its repository-relative
`protocol_path` with the recorded raw-byte SHA-256. Exactly one reference is the
`primary-protocol`; its JSON parameters, procedures, analysis, budget, commands, and
mutation plan must equal the complete canonical packet envelope byte-for-semantics;
the versioned schema has `additionalProperties: false`. Other references are explicitly
`supporting` rather than competing protocol definitions.
Print each canonical packet-rule hash with:

```powershell
uv run python scripts/check_viability_campaign.py `
  --packet-rule-sha256 reviews/viability/<campaign-id>/packets/<packet-id>.yaml
```

Put those hashes and the Git-blob SHA-256 values for every required contract file into
`PROTOCOL_MANIFEST.json`. Commit the manifest, all referenced contract files, and every
packet-specific protocol artifact in one immutable protocol snapshot. Then record the
resulting full `protocol_commit` and manifest hash in the campaign and all packets. The
helper below hashes a file exactly as stored in that Git commit, avoiding checkout
line-ending differences:

```powershell
uv run python scripts/check_viability_campaign.py `
  --git-blob-sha256 <protocol-commit> <repository-relative-path>
```

Only then validate the preregistered campaign:

```powershell
uv run python scripts/check_viability_campaign.py `
  reviews/viability/<campaign-id>/CAMPAIGN.yaml
```

The validator enforces JSON Schema structure and cross-document invariants. It rejects:

- lifecycle/adjudication contradictions and nonexclusive pass/fail/block rules;
- evidence achieved below the packet declaration or campaign-owned floor;
- unknown claim, gate, packet, capability, or evidence identifiers;
- missing dependencies, cycles, same-wave prerequisites, and holdout execution before
  every dependency is adjudicated `passed`;
- missing, out-of-tree, or SHA-256-mismatched receipts and empty decisive evidence;
- review summaries that do not exactly reconcile to schema-valid initial-review,
  builder-response, and re-review artifact bytes at immutable `commit:path` refs,
  including incomplete evidence, missing identity/independence fields, wrong candidate
  bindings, omitted items, and dangling supersessions;
- contradictory reviewer/builder operator or model-separation declarations,
  noncanonical context-tree methods, unrelated response identities, and fix commits
  that do not exist or precede the candidate under re-review;
- prohibited seat/session reuse or hidden-data exposure by a blind seat;
- reveal not authorized by the evaluator/custodian, reveal before reproduced holdout
  execution, output commitments not made by the reproduction runner or not bound to its
  raw-result bytes, changed post-reveal manifests, missing custody metadata, or
  unsupported canonicalization;
- nonexistent candidate/baseline/protocol commits, candidate/tree mismatches,
  post-protocol packet-rule changes, protocol receipt path/hash substitution,
  packet-specific protocol bytes absent or changed at the protocol commit, same-version
  requirements changes, or execution with contract files different from the protocol
  commit;
- duplicate YAML/JSON mapping keys, malformed external types, invalid percent escapes,
  non-finite JSON constants, excessive structured nesting, unsafe decoded URI/path
  control characters, adversarial host lengths, and invalid repository or receipt paths
  without raising an uncaught validator exception; and
- a campaign outcome inconsistent with its required packet outcomes.

For `VIA-900`, a raw Boolean named `unaffiliated-operator` is necessary but not
sufficient. The packet must also carry a frozen typed `external_replication` contract.
That contract records an external organization and operator, independent repository
commit/tree provenance resolved from a content-addressed Git bundle, zero prior
candidate/output/conclusion exposure, a blinded prediction commitment and custodian
reveal, an external output commitment, and a typed candidate/external comparison.
Organization, operator, agent, model, session, and orchestrator must all be distinct
from every internal campaign seat. Canonical repository aliases—including default
ports, DNS trailing dots, percent-encoded hostnames, canonical and IPv4-mapped IP
literals, equivalent hierarchical or opaque Windows/file-URI paths, dot segments,
credentials, protocol spelling, and terminal `.git` path forms—candidate commit/tree
reuse, nonexistent bundle commits, and mismatched trees are rejected. Invalid,
malformed-percent, overlong-host, or control-bearing repository identities and receipt
paths fail closed before filesystem access. Residual percent escapes after one URI
decode are invalid rather than reinterpreted. The candidate side of the comparison is
exactly the custody-committed raw result; candidate/external
paths and bytes must differ. After reveal, the adjudicator records both hashes, the
frozen JSON pointer and absolute tolerance, both measured values, and the recomputed
agreement Boolean. Structured receipts use strict JSON type equality: Boolean and
numeric representations are not interchangeable. A disagreement is valid failure
evidence only when adjudication is `failed` with `external-replication-disagreed`; it
is not discarded as an invalid run.
The named structured receipts must reproduce the contract exactly. Therefore the local
validator can check a clean-room evidence package without converting a same-operator
assertion into Tier E; an external maintainer must still verify that off-system
affiliation and independent authorship declarations are truthful.

Every structured receipt is bounded to 128 nested mapping/array levels and finite JSON
numbers. JSON uses strict RFC constants, so `NaN` and infinities are invalid. Exact
protocol-envelope and external-receipt comparisons use recursive JSON type equality;
Boolean, integer, and floating-point values cannot substitute for one another.

Outcome rules use the versioned `popgp-bool-v2` expression language. Bindings name a
SHA-256-verified `raw-results` JSON receipt, a JSON Pointer, and an exact JSON type.
Boolean and numeric values are not interchangeable. Expressions combine Boolean
`literal`, `all`, `any`, `not`, and typed `compare` nodes (`eq`, `ne`, `gt`, `ge`,
`lt`, `le`), but every pass/fail/block expression must consume at least one verified
binding. Every campaign-owned capability has a same-named Boolean binding at
`/capabilities/<capability>` and the canonical rule `<capability> == true`; a passing
packet requires every such raw gate to be true. Binding-free literals, alternate
pointers, or merely listing a capability name are rejected. Free-form prose is
explanatory only and cannot adjudicate a packet.

The evaluator/custodian commits raw-byte SHA-256 hashes for the hidden holdout and seed
manifests. Builder, falsifier, and reproduction runner remain blind. The reproduction
runner signs an immutable output-commitment receipt that identifies and hashes its
`raw-results` receipt. Only the evaluator/custodian identity may authorize reveal, and
reveal is rejected until holdout execution reaches the reproduced phase. Adjudication
requires a hash-verified reveal receipt, matching post-reveal manifests, archive
location, and retention policy. A packet cannot pass on builder-only access booleans.

## 8. Required receipt layout

Use immutable, round-numbered artifacts. Do not overwrite a failed run.

```text
reviews/viability/<campaign-id>/
  CAMPAIGN.yaml
  PROTOCOL_MANIFEST.json
  DECISION.md
  manifests/
    HOLDOUT-MANIFEST.json
    SEED-MANIFEST.json
  packets/
    <packet-id>.yaml
  <packet-id>/
    PACKET.yaml
    PROTOCOL.md
    ATTACK-PLAN.md
    ENVIRONMENT.json
    RUN-<round>.log
    RESULTS-<round>.json
    STATISTICAL-AUDIT-<round>.md
    MUTATION-RESULTS-<round>.json
    INDEPENDENT-REVIEW-<round>.md
    BUILDER-RESPONSE-<round>.md
    INDEPENDENT-REREVIEW-<round>.md
    EXTERNAL-IMPLEMENTATION-PROVENANCE.json
    EXTERNAL-REPOSITORY.bundle
    BLINDED-PREDICTION.json
    PREDICTION-REVEAL.json
    EXTERNAL-OUTPUT.json
    EXTERNAL-OUTPUT-COMMITMENT.json
    CROSS-IMPLEMENTATION-COMPARISON.json
    EXTERNAL-REPLICATION.json
    CLAIM-DIFF-<round>.md
    ADJUDICATION-<round>.md
    OUTPUT-COMMITMENT-<round>.json
    REVEAL-<round>.json
```

Every receipt entry has an ID, kind, path, media type, and raw-byte SHA-256 hash. A
`protocol` receipt must exactly match a content reference in the frozen
`preregistration.protocol_artifacts` array; changing either its campaign path or bytes
after the protocol snapshot is rejected. Review, response, and re-review receipts also
carry parallel immutable Git refs in `review_chain`; the validator compares their raw
bytes to the named blobs before accepting any lifecycle outcome. Large
raw arrays may live in a versioned external archive only after a retrieval adapter can
verify the same fields; the v2 validator otherwise rejects unavailable paths rather
than trusting a URI. Plots are diagnostic views; the decision must be reproducible
from machine-readable raw results.

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
infer holdout labels, secret seeds, or private evaluator logic. Record your own model,
operator, session, access, and exposure fields; do not reuse the evaluator/custodian or
builder session.

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
protocol. Commit the output receipt hash before the evaluator reveals holdouts. A
command that cannot run is classified by the frozen cause-code rules; it is never
silently skipped or automatically called blocked. Do not repair code or adjudicate the
hypothesis.
```

### Adjudicator prompt

```text
You occupy the adjudicator seat for POPGP packet <packet-id>.

Do not modify implementation or thresholds. Verify commit hashes, access declarations,
protocol freeze, required seats, complete receipts, mutations, hidden-holdout handling,
and reproduction evidence. Run `scripts/check_viability_campaign.py` and recompute the
versioned pass/fail/block expressions from hash-verified raw results. A valid round has
exactly one outcome: passed, failed, or blocked. An invalid protocol, receipt, custody
chain, dependency state, or ambiguous rule is recorded as `round_status: invalid` with
`packet_outcome: pending`; it requires a new round and is not scientific failure.
Model agreement is not an outcome rule. Map the result to claims-matrix wording without
extending the claim beyond the evidence level achieved.
```

## 10. Adjudication and stop rules

A packet is **passed** only when its pass expression alone is true. It is **failed**
when its failure expression alone is true in a valid run. It is **blocked** only when
its blockage expression alone is true because a prerequisite outside the tested claim
was unavailable. Zero or multiple true expressions make the round **invalid** and
leave the packet outcome pending.

Cause codes pin the distinction:

| Round/outcome | Allowed cause class | Example |
|---|---|---|
| invalid / pending | protocol, receipt, rule, custody, or dependency invalid | malformed protocol or hash mismatch |
| valid / failed | scientific or tested capability failure | a scalable backend exhausts the budget that its packet claims it can meet |
| valid / blocked | unavailable untested prerequisite | named CUDA hardware was not authorized or external access was unavailable |
| valid / passed | pass rule satisfied | exactly the frozen pass expression is true |

Resource exhaustion is failed when operating within that resource ceiling is itself a
tested capability; it is blocked only when a separately declared prerequisite was not
authorized or available. Missing/invalid evidence is never a scientific failure or a
blockage. The validator enforces these mutually exclusive classes.

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
  and achieved evidence >= packet declaration >= campaign floor
  and no non-negotiable constraint is unresolved
  and every blocking independent-review finding/test is independently resolved
  and scripts/check_viability_campaign.py reports no error.
```

There is no partial-credit viability score. Report packet-level progress separately.
If one required packet is failed, the target tier is failed for that candidate. If no
packet failed but one is blocked, the tier is blocked/not demonstrated. Pending or
invalid rounds keep the campaign pending.

## 11. Recommended execution waves

The ordering below lets independent agents work concurrently without letting later
claims outrun their dependencies.

| Wave | Parallel packets | Exit condition |
|---|---|---|
| 0 — freeze | `VIA-000`, campaign manifest, holdout custody | Reproducible baseline and immutable protocol hashes |
| 1 — enabling foundations | `VIA-010`, `VIA-300` | Structural audit passed and full-pipeline scalable ladder available |
| 2 — mechanism attacks | `VIA-100`, `VIA-150`, `VIA-200` | Source, projection/capacity, and blind-locality packets adjudicated |
| 3 — convergence | `VIA-400` plus independent numerical cross-checks | Intrinsic refinement gate adjudicated |
| 4 — clock/weak field | `VIA-500` | Independent clock and weak-field gate adjudicated |
| 5 — closure | `VIA-600` | Tensor closure/conservation gate adjudicated |
| 6 — 3D compatibility | `VIA-700` | 3D same-source gravity, laboratory statistics, Lorentz, and no-signaling gates adjudicated |
| 7 — independence | `VIA-900` | Clean-room prediction reveal and replication adjudicated |
| 7 — extension | `VIA-800` | Required only for high-curvature claims |

Start with the smallest counterexample capable of refuting a claim, but do not call a
packet passed at that scale unless the packet explicitly requires only that scale.
Parallel agents should receive bounded packet context, the same frozen hashes, and no
other agent's conclusions until their initial artifacts are committed. Design work may
start earlier, but hidden holdout execution is rejected until every machine-declared
dependency is adjudicated `passed`.

## 12. Definition of a completed viability campaign

A campaign is complete when:

- its target tier, candidate, baseline, tree, protocol, complete preregistration content,
  review artifacts, holdout-manifest, and seed hashes are immutable and recorded;
- every required packet has a final `passed`, `failed`, or `blocked` adjudication;
- the campaign, packets, dependency DAG, evidence floors, custody chain, receipts,
  review chain, and computed decision pass `scripts/check_viability_campaign.py`;
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
