# Scientific hardening decisions

## D001 — claim strength

Use: implemented, benchmark-supported, negative result, placeholder, conjecture, or
unimplemented. “Proof,” “derivation,” and “emergence” require the corresponding
mathematical or controlled numerical result, not a successful chosen-equation solve.

## D002 — intrinsic inference versus visualization

Inference receives states/correlations and configuration only. Hamiltonian reference
edges may be read after inference for metrics. Plotting reference edges must be
labeled validation overlay. MDS coordinates are an embedding diagnostic, not by
themselves an intrinsic metric or equivalence-principle test.

## D003 — topology language

The chain/grid Hamiltonians encode interaction topology without coordinates. Current
results are “blind recovery of encoded locality,” not creation of topology without a
spatial prior. Pairwise MI remains provisional until non-geometric/QCMI controls pass.

## D004 — spectral dimension

Finite graphs have zero heat-kernel dimension in UV and IR limits. Report the full
curve and finite-graph peak time/value. Do not call a 4- or 9-node peak a continuum
spectral dimension.

## D005 — source law

The pipeline von Neumann source is non-physical. A manually injected source is a
Green-function diagnostic. Raw relative entropy is rejected as a standalone linear
mass source in the tested perturbative regime. Modular energy is a candidate only;
its affine-mixture unit slope is an algebraic identity, not a falsification test. No
final source law is selected without a non-affine response, many-body localization,
and conservation tests.

## D006 — clock and redshift convention

Use `dτ ∝ exp(Φ)` and `1+z=exp(Φ_observer-Φ_emitter)`. A negative source must make Φ
smaller near the source, clocks slower there, and boundary-observed redshift positive.
Finite unscreened graph sources must sum to zero or explicitly use a neutralizing
background; never pin a node as a hidden sink.

## D007 — singularity criterion

Bounded capacity, bounded fitted curvature, or a divergent reconstruction condition
number does not establish singularity resolution. Require causal-trajectory
extendibility or a precise deterministic discrete analogue and identify which
singularity-theorem assumption is modified.

## D008 — reproducibility and negative results

Record configuration, seed, raw numerical values, uncertainties, commands, runtime,
and unavailable toolchains. Committed validation JSON is evidence only for its named
criterion. Retain negative results and do not change thresholds after seeing output
without recording the change.

## D009 — backend evidence tiers

Exact density-matrix, approximate mean-field, and native CUDA results are not
interchangeable. The mean-field backend cannot represent entanglement or MI and must
fail explicitly at `Π_loc`. The native clock identity stub is not evidence for a
clock solve.

## D010 — branches and review

Use `codex/` branches and descriptive commits. Prefer staged PRs for source,
geometry/closure, paper, and high-curvature work. Do not stage the private handoff
document unless the owner explicitly chooses to publish it.
