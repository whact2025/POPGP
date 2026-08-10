# A Phase-Ordered Pre-Geometric Projection Framework
*v1.0 (Submission Draft)*  
Author: Richard Fuoco  
Date: 2026-02-21

> **Scientific-status note (2026-08-09):** This is a theoretical submission draft.
> Its unique projection, 3+1-dimensional recovery, GR/QFT closure, Lorentz recovery,
> and singularity claims are hypotheses or requirements, not completed results. The
> companion code currently supports finite exact tests of partition ranking, blind
> recovery of encoded chain/grid locality, MDS dimension selection, and a numerical
> graph clock constraint, local SPD embedding fits, and a 2D angle-deficit proxy. It
> does not implement QCMI filtering, intrinsic Regge/Einstein closure, or a validated
> physical Araki/KMS source. Perturbative tests reject raw
> relative entropy as a standalone linear source. See
> `docs/scientific_hardening/CLAIMS_MATRIX.md` for the current evidence boundary.

## Abstract
We propose a relational/algebraic substrate and a candidate finite-capacity projection
program. Current exact toy models recover interaction locality encoded in selected
chain/grid Hamiltonians, choose one- and two-dimensional finite embeddings, and test
graph clock constraints. Raw relative entropy fails as a standalone linear source;
a microscopic KMS-energy density remains a finite-model candidate. General
Relativity, Lorentz recovery, a three-dimensional continuum limit, and singularity
resolution are matching goals or conjectures, not derived results.

## Contents
1. Motivation and scope  
2. Design constraints  
3. Core postulates (axioms)  
4. Mathematical primitives  
5. Projection outputs and emergent time  
6. Finite distinguishability and Planck-scale resolution  
7. Quantum statistics as projection-limited inference  
8. Emergent geometry and GR matching  
9. Standard Model compatibility (program)  
10. Entanglement and junction accessibility (open module)  
11. Evaluation, falsification, and test program  
12. Discussion and open problems  
13. Worked examples (spherical mass, lensing, cosmology)


## Reviewer-facing summary

### What this framework commits to
- **Substrate:** an atemporal, pre-geometric relational/algebraic structure $(\mathcal{A}, \omega)$ with a compact internal symmetry and a preferred **phase/action ordering flow** $\sigma_s$ (order parameter $s$, not time).
- **Projection:** a **unique and necessary** physical map $\Pi$ from substrate to effective descriptions. $\Pi$ is defined operationally by (i) a stability-selected coarse-graining into finite-capacity “cells”, (ii) correlation-defined locality, (iii) an embedding procedure into 3D, and (iv) a phase-order-to-clock-time mapping.
- **Finite distinguishability:** any finite projected region has bounded distinguishable information, scaling with **boundary area** (Planck-area-like scale $\ell_*^2$), interpreted as a **relational encoding boundary** rather than a geometric boundary in the substrate.
- **Emergence:** discrete quantum “types” arise from stable representation content under projection constraints; effective geometry arises from correlation structure and the clock mapping, not from substrate curvature.

### What this framework does *not* claim (yet)
- It does **not** assume a complete derivation of the Standard Model spectrum, coupling constants, or cosmological parameters in this draft.
- It does **not** assert superluminal signaling. “Junction access” is defined as a constrained open module with parameters required to match existing no-signaling constraints in all empirically accessed regimes.
- It does **not** claim a closed-form fundamental “constitutive law” $g = F(\rho, \ldots)$; the effective metric is defined by the explicit projection construction (graph distances → embedding → reconstructed $h_{ab}$ and $d\tau$ mapping).

### Scientific status and evaluation
- The framework is **empirically anchored** by matching requirements: in accessible regimes, the induced effective description must reproduce (to stated tolerance) standard GR tests and standard quantum statistics.
- The framework becomes **strictly falsifiable** when it asserts: (i) universal area-law capacity bounds outside their known domain, (ii) saturation/no-singularity behavior in regimes where GR predicts divergence, or (iii) any nonzero operational “junction access” parameter in regimes already constrained by experiment.
- Absent additional predictive commitments, the framework is evaluated by: (a) internal consistency (no hidden time/geometry in substrate), (b) uniqueness/minimality of $\Pi$ under stated selection principles, and (c) whether GR/QFT emerge as stable effective closures with a small, auditable parameter budget.


---

## 1. Motivation and scope
This document introduces a conceptual framework intended for comprehensive evaluation and iterative refinement. The goal is to provide a minimal set of primitives and postulates from which (i) an effective 3+1 spacetime description, (ii) quantum statistical behavior, and (iii) General Relativity in tested regimes can be recovered as projection-level physics. The framework is explicitly pre-geometric at the substrate level and treats time as emergent from a phase/action ordering structure.

Scope control: the present draft focuses on structural definitions, matching requirements, and evaluation criteria. It does not claim a completed derivation of the Standard Model or the Born rule; instead it identifies the minimal mathematical objects needed to attempt such derivations.

## 2. Design constraints
- No substrate time: the substrate admits no fundamental temporal parameter.
- No substrate spatial geometry: the substrate is not a manifold with metric/curvature; geometric notions are projection outputs.
- Analog intuition with bounded physics: continuous symmetry/phase structure is allowed as an idealization, but no unbounded physical observables (no divergent densities; no infinite recursion depth).
- Unique and necessary projection: the mapping from substrate to projection is physical and not contingent among many alternatives.
- Finite distinguishability: any finite projected region has a bounded number of distinguishable states; capacity scales with boundary area (area law) via a relational encoding boundary.
- Empirical recovery: in currently tested regimes the framework must reproduce standard GR and quantum predictions to within experimental bounds.
- FTL signaling remains an open module: the existence of operational superluminal channels is not assumed; it is parameterized and constrained.

## 3. Core postulates (axioms)

### 3.1 Substrate postulates
**S1 (Atemporal substrate).** There exists a substrate $\mathcal{S}$ that is not spacetime and has no intrinsic time parameter.  
**S2 (Pre-geometric substrate).** The substrate carries relational/algebraic structure but no spatial metric, curvature, or geometric topology in the usual sense.  
**S3 (Compact internal symmetry).** The substrate admits a compact internal SU(2)-like symmetry acting on its relational degrees of freedom.  
**S4 (Phase/action ordering generator).** A distinguished generator induces an ordering structure (phase/action order) that is not time but is used by projection to construct time-order.

### 3.2 Projection postulates
**P1 (Equivalence-Class Uniqueness).** There exists a physical projection map $\Pi$ from substrate structure to projection-level effective physics. While the specific microscopic cell net may only be unique up to spontaneous symmetry breaking (attractor basin selection, analogous to a gauge choice), the *macroscopic predictions and geometric invariants* output by $\Pi$ are uniquely determined, establishing equivalence-class uniqueness.  
**P2 (Emergent spatial structure; dimension as output).** $\Pi$ yields an emergent metric space with a manifold-like realization of some effective dimension $D^*$; in the empirically accessed regime of our universe, $D^* \approx 3$ (treated as an output/matching condition rather than an assumed input).  
**P3 (Emergent time-order).** $\Pi$ yields an objective time-order in projection as a metric on phase/action order; non-conscious systems inherit this time-order.  
**P4 (Finite distinguishability; boundary-capacity scaling).** For any finite projected region $R$, physically distinguishable information is bounded by a boundary *cut-capacity* functional (defined on the correlation graph). In manifold-like regimes this scaling is expected to reproduce an area-law.  
**P5 (Finite-capacity regularization conjecture).** Projection-level observables may
saturate rather than diverge. This is not yet demonstrated dynamically and does not by
itself imply causal or geodesic completeness.

### 3.3 Emergence postulates
**E1 (Quantum discreteness from representations).** Discrete quantum types arise from stable representation content of the internal symmetry under projection constraints, not from fundamental substrate discreteness.  
**E2 (Geometry as projection output).** The projection is intended to construct an effective metric from correlation distances, an embedding/local-metric diagnostic, and a clock mapping. The current code implements finite prototypes of those pieces but not a complete spacetime metric or tensor dynamics. The source of the graph potential remains experimental.
**E4 (Lorentz recovery conjecture).** SU(2)-equivariance constrains internal/spatial rotations only; it does not implement boosts or establish Lorentz covariance. A possible infrared recovery mechanism remains conjectural until explicit dispersion, common-speed, anisotropy, boost, and renormalization-group calculations are supplied.
**J (module) (Junction accessibility).** In all known regimes, operations available to agents obey no-signaling in local marginals. Additional junction access is treated as a constrained open module.

## 4. Mathematical primitives

### 4.1 Substrate representation (analog-friendly)
Represent the substrate as a pair:
$$
S = (A, ω)
$$
where $\mathcal{A}$ is an algebra of relational observables and $\omega$ is a state on $\mathcal{A}$. No manifold, metric, or time parameter is assumed at this level.

**Analog-friendly choice.** To support continuous internal symmetry/phase structure while preserving *finite physical realizability* (via projection-level finite distinguishability), take $\mathcal{A}$ to be a separable operator algebra that is well-approximated by finite matrix algebras at any finite resolution. A convenient template is:
$$
A = A_rel ⊗ A_F
$$
- $\mathcal{A}_{\text{rel}}$ is the “relational bulk” algebra (pre-geometric), chosen so that it can be approximated by an increasing sequence of finite-dimensional matrix algebras:
$$
A_rel ≈ closure( ⋃_n M_{k_n}(C) )
$$
This supports the **analog** intuition (continuous symmetry/phase) while remaining compatible with the principle that *only a finite number of states are physically distinguishable in any finite projected region* (Section 6).

- $\mathcal{A}_F$ is an optional finite “internal” algebra used to encode gauge/representation structure in a purely algebraic way (Section 9). A minimal candidate that naturally contains $U(1)$, $SU(2)$, and $SU(3)$-type unitary structure is:
$$
A_F = C ⊕ H ⊕ M_3(C)
$$
with $H$ the quaternions.

**Remark (no infinities physically instantiated).** Continuous groups and infinite-dimensional algebras are treated as *idealized descriptions*: the framework’s finiteness claim is that no *projection-level observable* diverges and that any finite region has finite operational capacity (area-law distinguishability), not that the descriptive mathematics cannot employ limits.

### 4.2 Internal symmetry action
Let $\alpha$ be an action of the internal symmetry group on $\mathcal{A}$ as *-automorphisms:
$$
α: SU(2) → Aut(A)
$$
Symmetry invariants are quantities unchanged under $\alpha_U$ for all $U \in SU(2)$.

### 4.3 Phase/action ordering generator
The substrate has no time, but it provides an **ordering structure** that projection uses to construct time-order.

Two equivalent presentations are allowed in this draft:

**(i) Generator form (working placeholder).** Let $H \in \mathcal{A}$ be a distinguished self-adjoint element ($H = H^\dagger$) defining a one-parameter family of inner automorphisms:
$$
A ↦ e^{isH} A e^{-isH}
$$
The parameter $s$ is a phase/action **order parameter**, not substrate time.

**(ii) Canonical-flow form (parameter-reducing option).** Impose that the pair $(\mathcal{A}, \omega)$ determines a canonical one-parameter automorphism flow $\sigma_s$ on $\mathcal{A}$. Projection-time is then constructed from monotone “distance” along this flow. In this view, $H$ is not chosen freely; it is the generator associated with the canonical flow in a suitable representation.

The framework’s finiteness claim applies to *projection-level observables* and distinguishability (Section 6), not to the mathematical use of continuous $s$.

### 4.4 Projection map (explicit construction template)

In this framework, the projection map $\Pi$ is not treated as an arbitrary “fit function.”  
Instead it is defined as a *constrained construction* that turns substrate relational structure into:

- an emergent 3D locality structure,
- an effective spatial metric,
- an objective time-order (from phase/action order),
- an encoding-density field and finite distinguishability bounds,
- effective subsystem states used for quantum predictions.

Mathematically, it is useful to write $\Pi$ as a composition of four stages:
$$
Π = Π_time ∘ Π_geom ∘ Π_loc ∘ Π_res
$$
where:

- $\Pi_{\text{res}}$ implements finite distinguishability (resolution-limited coarse-graining),
- $\Pi_{\text{loc}}$ extracts emergent locality from correlation structure,
- $\Pi_{\text{geom}}$ builds an effective 3D geometry/metric from locality data,
- $\Pi_{\text{time}}$ maps phase/action order into projection time.

The definitions below form a *template*. Each stage is explicit enough to be evaluated, but still leaves room for later refinement.

#### 4.4.1 Stage 0: represent the substrate state (GNS representation)
Given $(\mathcal{A}, \omega)$, choose the GNS triple $(\pi_\omega, \mathcal{H}_\omega, |\Omega\rangle)$ such that:
$$
ω(a) = ⟨Ω | π_ω(a) | Ω⟩   for all a ∈ A.
$$
All projection-level quantities are ultimately defined from this representation, not from any prior notion of space.

#### 4.4.2 $\Pi_{\text{res}}$: resolution-limited coarse-graining via split-property funnels
A projection must provide a notion of subsystems/regions, but subsystems are not substrate primitives. Furthermore, strict local algebras in Quantum Field Theory are typically **Type III factors** (entanglement is ubiquitous), meaning they do not admit pure states or density matrices, and cannot be factorized into tensor products $\mathcal{A}_R \otimes \mathcal{A}_R^c$.

To resolve this **Type III incompatibility** without abandoning the continuum, we define emergent "cells" explicitly as **split-property funnels** (Type I approximants across algebraic buffer zones).

**The Funnel Definition.**
Instead of a single algebra $\mathcal{A}_i$, the projection selects a **funnel structure** for each emergent site $i$:
$$
A_{i, inner} ⊂ N_i ⊂ A_{i, buffer} ⊂ A
$$
where $N_i$ is a **Type I factor** (isomorphic to $B(\mathcal{H})$ or a finite matrix algebra $M_d(\mathbb{C})$ in toy models) that acts as a local "distinguishable semantic shell" between the inner core and the buffer zone.

The effective local algebra is identified with the Type I factor $N_i$. This guarantees the existence of well-defined local density matrices, entropies, and particle-like excitations, even if the underlying substrate algebra is Type III.

**Coarse-Graining.**
The coarse-graining maps $E_i$ are conditional expectations onto these Type I factors:
$$
{A_i ⊂ π_ω(A)'' }_{i ∈ V},   with   A_i ≅ M_{d_i}(C)   (Type I approximants),
$$
together with conditional expectations (coarse-graining maps):
$$
E_i : π_ω(A)'' → A_i
$$
that are completely positive, unital, and idempotent ($E_i \circ E_i = E_i$).

For any finite set of cells $R \subset V$, define the induced effective algebra and reduced state:
$$
A_R  ≈  ⊗_{i∈R} A_i,        (exact in toy models; approximate “split” factorization in the continuum/QFT limit)
E_R  := ⊗_{i∈R} E_i,
ω_R  := ω ∘ E_R .
$$
### Finite distinguishability constraint (non-geometric form)
To avoid circularity, the capacity bound is imposed **before** any geometric embedding.

Let $\mathrm{Cap}(\partial R)$ denote a **cut-capacity** functional on the cell net:
$$
Cap(∂R) := Σ_{i∈R, j∉R} κ(I_{ij}),
$$
where $I_{ij}$ is the mutual information between cells $i$ and $j$, and $\kappa$ is a fixed monotone map to dimensionless edge capacities.

**Finite distinguishability constraint (Araki-relative form).**
For Type III compatibility, we eschew the naive von Neumann entropy. The bound is imposed on the **Araki relative entropy** of the region's state $\omega|_R$ against the canonical **KMS vacuum baseline** $\omega^{\text{vac}}|_R$ (defined by the global modular flow):
$$
S_{Araki}( \omega|_R || \omega^{vac}|_R ) ≤ η · Cap(∂R)   for all finite regions R ⊂ V.
$$
This is where “Planck-scale resolution” enters operationally: the projection cannot support arbitrarily many independent degrees of freedom in finite regions relative to the vacuum. High-resolution structure is bounded by a cut-capacity rather than a geometric area.

A geometric area scaling $\mathrm{Cap}(\partial R) \propto A_{\text{geo}}(\partial R)/\ell_*^2$ is treated as an emergent behavior in the low-stress embedding regime (§4.4.4), not as an input constraint.

#### 4.4.2a Selection Principle $E$: symmetry-commuting, phase-flow stable cell net

The choice of the effective cell algebras $\{\mathcal{A}_i\}$ and coarse-graining maps $\{E_i\}$ is fixed by a *stability principle* tied to the phase/action ordering flow $\sigma_s$ (§4.3), rather than chosen ad hoc. Intuitively: the correct decomposition is the one for which phase/action ordering does not continually “slosh” information across emergent cell boundaries.

Let $\alpha_g$ denote the internal SU(2)-like symmetry action on $\mathcal{A}$ (or on $\pi_\omega(\mathcal{A})^{\prime\prime}$ after the GNS embedding). Let $\sigma_s$ denote the phase/action ordering flow.

**Admissible coarse-grainings.** Define $\mathfrak{E}_{\text{adm}}$ as the set of families $\{E_i\}$ (and their associated cell algebras $\{\mathcal{A}_i\}$) satisfying all of:

1) **Finite-capacity cells.** Each cell algebra is finite-dimensional:
$$
A_i ≅ M_{d_i}(C),
$$
with either a fixed common dimension $d_i = d$ for all cells (simplest), or an explicit local bound $d_i \leq d_{\max}$ (bounded capacity). In all cases the finite distinguishability constraint in §4.4.2 is required to hold for all finite regions $R$.

2) **Symmetry commutation (internal isotropy constraint).** Each coarse-graining map commutes with the SU(2)-like symmetry:
$$
E_i ∘ α_g = α_g ∘ E_i    for all g ∈ SU(2) and all i.
$$
Equivalently (often easier to verify): the subalgebra $\mathcal{A}_i$ is invariant under $\alpha_g$, and $E_i$ is SU(2)-equivariant.

**Lorentz status.** This constraint supplies a tested spatial-rotation equivariance
condition in the toy coarse graining. The proposed identification with frame rotations
or a custodial/spin-connection mechanism is unimplemented. No conclusion about boosts,
Lorentz-violating operators, RG flow, or experimental suppression follows from the
current SU(2) test.

3) **Information retention (anti-triviality constraint).** The coarse-graining must not erase essentially all substrate information. Impose a global bound on relative-entropy loss:
$$
D( ω || ω ∘ E ) ≤ ε,
$$
where $E = \bigotimes_i E_i$ and $D(\cdot\|\cdot)$ is the quantum relative entropy. (Other retention constraints are possible, but this one is explicit and auditable.)

**Primary stability functional (phase-flow leakage).** For each admissible $\{E_i\}$, define the phase-flow leakage:
$$
L_leak(E) := ∫ ds w(s) · Σ_i  || E_i ∘ σ_s  -  σ_s ∘ E_i ||^2,
$$
where $w(s) \geq 0$ is a weight over a chosen phase-order interval, and $\|\cdot\|$ is a norm on superoperators (channels). Suitable choices include:
- the diamond norm $\|\cdot\|_\diamond$ (operationally strongest),
- or a Hilbert–Schmidt / Frobenius norm in a fixed representation (computationally simpler).

$\mathcal{L}_{\text{leak}}$ measures how strongly the phase/action ordering flow mixes degrees of freedom across the selected cell boundaries. Exact co-motion corresponds to $\mathcal{L}_{\text{leak}} = 0$.

**Optional tie-breaker (local information drift).** Among decompositions that minimize leakage, prefer those for which local cell information varies smoothly (minimally) along phase-order. Define $\omega_s := \omega \circ \sigma_s$, and let $\rho_i(s)$ be the density operator corresponding to the reduced state $\omega_s \circ E_i$ on the cell Hilbert space $\mathcal{H}_i$. For a small step $\delta$, define:
$$
L_drift(E) := ∫ ds w(s) · Σ_i  (1/δ^2) · D( ρ_i(s+δ) || ρ_i(s) ).
$$
This penalizes rapid change of local reduced information under phase-order advance. It is optional; it is intended as a *tie-breaker* and not the primary selector (to avoid freezing dynamics by choosing overly coarse cells).

**Cell selection as causal gradient flow.**

The cell net $E^* = \{E_i^*\}$ is **not** defined by a global $argmin$ (which would require an acausal, instantaneous computation over all possible coarse-grainings). Instead, it is defined as the **stable fixed-point (attractor)** of a local, causally-bounded gradient flow on the space of admissible coarse-grainings.

**Local update channel.** For each cell $i$, define the local leakage gradient:
$$
δ_i L_leak := ∂ L_leak / ∂ E_i |_{E_{j≠i} fixed}
$$
This is the functional derivative of $\mathcal{L}_{\text{leak}}$ with respect to the coarse-graining map at site $i$, holding all other cells fixed. To strictly avoid circularity with the emergent geometry $d_G$, this gradient depends only on cells $j$ within the **Algebraic Lieb-Robinson neighborhood** $N_{\text{LR}}^{\text{alg}}(i)$ of cell $i$. This neighborhood is defined intrinsically on the pre-geometric substrate using operator commutators under the canonical flow, completely independent of the correlation graph:
$$
N_{\text{LR}}^{\text{alg}}(i) := \{ j \in V : \| [\sigma_{\Delta s}(\mathcal{A}_i), \mathcal{A}_j] \| > \epsilon \}
$$
where $d_G$ is the graph-geodesic distance on the correlation graph and $\Delta s$ is the phase-order window width.

**Causal flow.** The coarse-graining maps evolve under a local descent:
$$
∂_s E_i = -η_flow · Π_{adm} [ δ_i L_leak + λ_drift · δ_i L_drift ]
$$
where:
- $\eta_{\text{flow}} > 0$ is a flow rate (not a physical observable; absorbed into the parametrization of the relaxation trajectory),
- $\Pi_{\text{adm}}$ projects each update back onto the admissible set $\mathfrak{E}_{\text{adm}}$ (enforcing SU(2)-equivariance, finite capacity, and retention bound at every step),
- $\lambda_{\text{drift}} \ll 1$ is a small coupling that implements the lexicographic hierarchy: $\mathcal{L}_{\text{leak}}$ dominates and $\mathcal{L}_{\text{drift}}$ acts as a tie-breaker in near-degenerate regions.

**Causality guarantee.** Because $\delta_i \mathcal{L}_{\text{leak}}$ depends only on $E_j$ for $j \in N_{\text{LR}}^{\text{alg}}(i)$, updates at cells $i$ and $k$ with non-overlapping algebraic neighborhoods **strictly commute**:
$$
[ \partial_s E_i ,  \partial_s E_k ] = 0 \quad \text{whenever} \quad N_{\text{LR}}^{\text{alg}}(i) \cap N_{\text{LR}}^{\text{alg}}(k) = \emptyset.
$$
This is the discrete analog of the causal (hyperbolic) structure of the Einstein constraint equations: the cell net evolves locally, with information propagating at finite speed $v_{\text{LR}}$ on the correlation graph.

**Definition of the selected cell net.** The projection-level cell net is the fixed point of this flow:
$$
E* := lim_{s → ∞} E(s),    where   ∂_s E_i |_{E=E*} = 0   for all i.
$$
This fixed point is the unique attractor within each basin of the local flow (uniqueness within a basin follows from the convexity of $\mathcal{L}_{\text{leak}}$ in each $E_i$ at fixed neighbors; global uniqueness is a dynamical selection — the substrate's actual relaxation trajectory determines which basin is reached).

This implements a stability principle in the precise sense requested: emergent subsystems are selected to be as invariant as possible under the phase/action ordering flow (primary), while also exhibiting minimal local information churn along phase-order (tie-breaker), all while respecting SU(2)-equivariance, finite distinguishability, and causal locality.


**Variational interpretation status.** A local causal relaxation toward stable
coarse-grainings is a proposed physical mechanism. The current exact code exhaustively
ranks a finite set of partitions with sampled objectives; it does not implement that
flow or prove that its attractors equal the global ranking. The CA cooling analogy
motivates an open-system control. It has no matched no-cooling arm, so it establishes
neither a cooling effect nor a universal thermodynamic precondition, a Type III buffer
mechanism, or spatial collapse. Those implications remain open questions.

**Implementation notes (default choices for toy models).**
- Weight $w(s)$: choose a finite phase-order window of width $\Delta s$ and set $w(s) = 1/\Delta s$ on that window (and $0$ outside), or use a Gaussian centered at $s_0$. $\Delta s$ should be small enough that the selected cell net is approximately stable across the window.
- Norm $\|\cdot\|$: use a Hilbert–Schmidt / Frobenius norm in a fixed representation as a computational proxy in toy models; treat the diamond norm $\|\cdot\|_\diamond$ as the operationally strongest target definition.
- Retention budget $\varepsilon$: define the capacity of a reference region $R0$ by $S_{\text{cap}}(R_0) := \eta \cdot \mathrm{Cap}(\partial R_0)$ and choose $\varepsilon = \kappa \cdot S_{\text{cap}}(R_0)$ with $\kappa \ll 1$ (or impose per-cell bounds). This prevents the trivial “erase everything” minimizer.
- Drift step $\delta$: choose $\delta \ll \Delta s$ (e.g., $\delta = \Delta s / N$ with large $N$) so that $\mathcal{L}_{\text{drift}}$ approximates a local drift rate.
- Lexicographic hierarchy: the flow uses $\lambda_{\text{drift}} \ll 1$ so that $\mathcal{L}_{\text{leak}}$ dominates; in exhaustive toy-model enumeration, this is equivalent to sorting by $\mathcal{L}_{\text{leak}}$ first and breaking ties by $\mathcal{L}_{\text{drift}}$.

#### 4.4.3 $\Pi_{\text{loc}}$: locality from correlations (graph metric from mutual information)
Given the effective cell net produced by $\Pi_{\text{res}}$, the reduced states $\omega_i$, $\omega_j$, and $\omega_{i \cup j}$ are well-defined on the Type I funnel factors $N_i$, $N_j$, and $N_i \otimes N_j$ (Section 4.4.2). All entropic quantities below are evaluated on these finite-dimensional approximants, never on the raw Type III substrate algebras.

Define the mutual information between cells $i$ and $j$ via **Araki relative entropy**:
$$
I_{ij} = S_Araki(ω_{i ∪ j} ‖ ω_i ⊗ ω_j)
$$
where $S_Araki(φ ‖ ψ)$ is the Araki relative entropy of the joint state against the product of marginals. This is the unique, UV-finite measure of total correlations that:
- is well-defined for arbitrary (including Type III) von Neumann algebras,
- reduces to the familiar $S(\rho_i) + S(\rho_j) - S(\rho_{ij})$ in finite-dimensional (toy model) representations,
- satisfies monotonicity under completely positive maps (data processing inequality).

The naive von Neumann formula $S(\rho_i) + S(\rho_j) - S(\rho_{ij})$ is **not used** at the foundational level because individual von Neumann entropies $S(\rho)$ diverge for Type III factors. The Araki formulation avoids this by never computing absolute entropies — only relative ones.

Interpret $I_{ij}$ as a **relational coupling kernel**: higher mutual information indicates tighter relational dependence.

### Distance kernel
Define an emergent pairwise distance by a fixed monotone decreasing map $f$:
$$
d_{ij} = ℓ_* · f(I_{ij} / I_0),
$$
where $I_0$ is a reference mutual-information scale (e.g., typical strong-coupling value), and $f$ satisfies:
- $f(u) \geq 0$,
- $f'(u) < 0$,
- $f(u \to 0) \to +\infty$ (very weak coupling → far apart).

A simple low-parameter choice is:
$$
f(u) = max{0, -log u }.
$$
### Markovian Geometric Filtering & Weighted Graph
Raw mutual information can falsely identify highly entangled but geometrically distant nodes (e.g., Bell pairs, error-correcting codes) as "adjacent." To ensure strict geometric proximity, the edge weight kernel applies a **Quantum Conditional Mutual Information (QCMI)** filter. If the correlations between $i$ and $j$ are entirely mediated by an intermediate neighborhood $k$, their QCMI vanishes: $I(i:j | k) \approx 0$. By restricting continuous edges to pairs with strictly non-Markovian (direct) QCMI, the framework systematically separates true geometric proximity from long-range topological entanglement.

Define the final **continuous** edge weight kernel (no hard threshold) on unscreened pairs:
$$
w_{ij} := \kappa(I_{ij}) \quad \text{with} \quad \kappa \text{ monotone increasing and } \kappa(0)=0.
$$
In principle this defines a fully connected weighted graph. For computational sparsification in toy models, use
a connectivity-preserving rule (e.g., k-nearest neighbors by smallest $d_{ij}$ **plus** a minimum-spanning-tree backbone)
rather than a hard $I_{ij}$ cutoff, to prevent “islands” created by rapid decay.

### Graph geodesic distance
With edge lengths $d_{ij}$ (or equivalently weights $w_{ij}$), define graph-geodesic distance:
$$
d_G(i,j) = inf_{paths i→j} Σ_{(a,b) in path} d_{ab}.
$$
This defines a metric space $(V, d_G)$ without assuming any background geometry.

#### 4.4.4 $\Pi_{\text{geom}}$: emergent geometry via complexity-stress minimization
The spatial manifold is not a background container. It is a compression scheme for the correlation graph.

**Ontic vs. Epistemic distinction.** A critical conceptual clarification is required before the construction. The physical (ontic) reality in this framework is solely the **discrete correlation graph** $(V, d_G)$ — the set of cells, their MI-derived edge weights, and the raw graph-geodesic distances between them. This graph is the output of $\Pi_{\text{loc}}$ and requires no coordinates, no embedding space, and no notion of dimension.

The embedding into $\mathbb{R}^{D^*}$ described below is strictly an **epistemic coordinate chart**: a lossy compression constructed by macroscopic observers (or their computational proxies) to extract the effective dimension $D^*$ and to interface the discrete structure with continuum language. The universe does not globally compute MDS — such a computation would require instantaneous access to all pairwise distances, violating the Lieb-Robinson causal bound that governs information propagation on the graph (§4.4.2a). In the physical substrate, what "selects" the effective dimensionality is the intrinsic diffusive topology of the graph (spectral dimension $D_S$), not a global optimization.

### 1. Dimension-selection diagnostic
The finite pipeline does not hard-code $D=3$. It defines a selected embedding
dimension $D^*$ as the integer that minimizes a declared **Complexity-Stress
Functional** $F(D)$:
$$
D* := argmin_{D \in \mathbb{N}} [ Stress(D) + \lambda_{dim} \cdot |D - D_S|^2 ]
$$
where:
- $\mathrm{Stress}(D)$ is the standard MDS stress (distortion of graph distances $d_G$ vs Euclidean distances in $\mathbb{R}^D$).
- $D_S$ is the **Spectral Dimension** of the graph (derived from the heat kernel trace $\mathrm{Tr}(e^{-t\Delta}) \sim t^{-D_S/2}$), representing the graph's intrinsic diffusive topology.
- $\lambda_{\text{dim}}$ is an explicit tunable penalty weight. It can change the
  selected dimension in non-geometric controls and therefore must be reported and
  swept; the code does not implement a $\lambda_{\text{dim}}\to0^+$ limiting procedure.

In the physical regime of our universe, we require $D^* = 3$ as a stable minimum. If $D^*$ diverges or collapses to 1, the framework predicts a non-geometric phase (see Falsifiers F1).

### 2. Relational Embedding (Epistemic Coordinates)
Given $D^*$, the coordinates $\{x_i\}$ are determined by the stress-minimizing configuration in $\mathbb{R}^{D^*}$. This embedding is unique only up to isometries; physical observables must be relational (diffeomorphism invariant).

**Local embedding status.** MDS seeks a low-stress Euclidean representation of finite
graph distances. Its ability to flatten a neighborhood is an embedding property, not
a test or derivation of the Einstein equivalence principle. An EEP claim would require
operational freely falling clocks/trajectories and local nongravitational dynamics.

**Curvature status.** The current Python path triangulates two-dimensional MDS
coordinates with Delaunay and reports boundary-aware angle-deficit proxies. It does not
yet construct an intrinsic Vietoris–Rips complex, dual volumes, a Regge tensor, or a
refinement limit. Intrinsic graph-length curvature remains planned work.

### 3. Local Metric Reconstruction Diagnostic
The implemented prototype fits an SPD matrix in the selected MDS coordinates and
reports rank, condition number, and residual. It uses a declared ridge penalty toward
the coordinate identity; a relational covariance regularizer is not implemented.

Solve for $h_{ab}(x_i)$:
$$
min_{h \succ 0} \sum_{j \in N(i)} ( d_G(i,j)^2 - \Delta x^T h \Delta x )^2  +  \lambda_{spd} || h - I ||_F^2
$$
The fit is an embedding-coordinate diagnostic and can be underdetermined at boundaries.
It is not yet an intrinsic continuum metric.

### 4. Metric-reconstruction breakdown diagnostic
A large condition number $\kappa$ of the local design matrix $M$,
$$
\kappa(M) \to \infty
$$
indicates that this particular local fit is poorly identifiable. It is neither an event
horizon nor a spacetime singularity. Singularity resolution requires continued
physical evolution and causal/geodesic extendibility under a controlled model.


#### 4.4.5 $\Pi_{\text{time}}$: clock-rate constraint candidate
The current implementation treats the clock field as a graph-wide elliptic diagnostic.

### 1. The Clock-Rate Constraint Ansatz
The framework proposes a graph **Clock-Rate Potential** $\Phi$. In the current code,
$\Phi_i$ is a numerical diagnostic whose interpretation as gravitational depth is a
matching hypothesis, not a derived result.

$\Phi$ is the unique solution to the **Weighted Graph-Laplacian** equation:
$$
(\Delta_w + \mu^2 I) \Phi = \delta\rho
$$
where:
- $\Delta_w$ is the graph Laplacian on the mutual-information weighted graph: $(\Delta_w \Phi)_i := \sum_j w_{ij}(\Phi_i - \Phi_j)$.
- $\delta\rho$ is the source term (defined below).
- $\mu$ is an optional screening parameter. On a finite connected graph, $\mu=0$
  requires an explicit zero-mode compatibility policy; $\mu>0$ makes the operator
  invertible. Neither choice establishes a continuum $1/r$ law without a dimensional,
  refinement, boundary, and normalization analysis.

*Status:* the solver is a well-tested elliptic graph constraint. Its relationship to
the ADM Hamiltonian constraint remains conjectural because no tensor closure,
conservation identity, or continuum matching has been demonstrated.

### 2. Source candidates and a retained falsification result
The original proposal used a temporally averaged local Araki contrast,
$$
\delta\rho_i := -\,\frac{1}{s_0} \left[ \bar{S}_{Araki}(\omega_{i} \| \omega_{i}^{vac}) \right]
$$
where the bar denotes a proposed short phase-window average. The minus sign fixes a
well convention, but positivity alone does not make relative entropy a mass density.
For $\rho(\epsilon)=(1-\epsilon)\sigma+\epsilon\rho_{exc}$, the implemented exact
tests find $D(\rho(\epsilon)\|\sigma)=O(\epsilon^2)$ and entropy dependence at fixed
energy. Raw relative entropy is therefore rejected as a standalone weak-field linear
source in the tested regime.

For a faithful finite KMS state, the global modular Hamiltonian obeys
$K_\sigma=\beta H+\log Z$, so
$$
\Delta\langle K_\sigma\rangle=\beta\,\Delta\langle H\rangle.
$$
The exact backend now tests a separately named microscopic candidate
$$
\delta\rho_i^{(E)}=-g\,\beta\,\Delta\langle h_i\rangle,
\qquad \sum_i h_i=H,
$$
using a declared symmetric split of pair interactions. In the non-affine family
described below it has a nonzero first-order susceptibility. The KMS-labelled runtime
source is accepted only when the supplied reference is within trace distance $10^{-10}$
of the Gibbs state for the same backend Hamiltonian and $\beta$; under that validated
premise, its source density sums to minus the global modular-energy change. A separate
finite-chain quench demonstrates
that the global Hamiltonian expectation is conserved while the audited site-energy
profile changes and spreads. No discrete continuity current or local conservation law
is established by this check.

The order statement is excitation-family dependent. Under the affine interpolation
$\rho(\epsilon)=(1-\epsilon)\sigma+\epsilon\rho_{exc}$, every fixed-observable
expectation and the linear graph solve are exactly proportional to $\epsilon$; those
legs are analytic-identity regressions, not falsification tests. A genuinely
non-affine family,
$$
\rho(\epsilon)=\frac{\exp[-\beta(H+\epsilon V)]}
{\operatorname{Tr}\exp[-\beta(H+\epsilon V)]},
$$
with localized $V=-h_{center}$ shows a nonzero signed, Richardson-extrapolated modular
susceptibility in the declared finite sweep ($\beta\leq3$, including $\beta=2.5$),
matching the exact Kubo--Mori value. Relative coefficient agreement, an absolute
slope band, normalized residual, a precision floor, and a first-order negative control
separately gate $D=O(\epsilon^2)$. Physical-energy response then follows from the exact
KMS identity rather than an independent slope test. Conversely, an isospectral
local unitary family has $\Delta S=0$ and
$D(\rho\|\sigma)=\Delta\langle K_\sigma\rangle=\beta\Delta\langle H\rangle
=O(\epsilon^2)$. No family-independent first-order law is claimed.

By contrast, a naive one-site reduced modular Hamiltonian is proportional to the
identity in the symmetric KMS control and gives zero local response. The energy-density
candidate is therefore a feasible test object, not a unique or covariant source law;
its dependence on the microscopic interaction graph and $h_i$ decomposition is open.
No particle/black-hole duality, Newtonian field law, or universal attraction follows
from these finite tests.

### 3. The Baseline Vacuum (Solving the Zero-Mode)
For $\mu = 0$, the graph Laplacian has a kernel (zero mode) corresponding to constant shifts. To fix this mode without introducing instantaneous non-local spatial averaging (which would violate causality), we define the baseline vacuum $\omega^{\text{vac}}$ strictly algebraically:

**Finite-model definition:** $\omega^{\text{vac}}=e^{-\beta H}/Z$ is the Gibbs/KMS
reference for the chosen Hamiltonian. A representation-independent construction and
proof that this is the unique physical vacuum remain open.

### 4. Emergent Proper Time
The local proper time $\tau$ along a phase trajectory is then constructed by integrating the **Laplacian-derived clock rate**:
$$
d\tau(i) = \beta_0 \cdot \exp(\Phi_i) \cdot dS_{\text{act}}
$$
where $dS_{\text{act}}$ is proposed as an operational action-distance increment. The
current simulator substitutes its numerical step `dt`; an independent clock observable
and gravitational matching test remain to be implemented.

### Minimal emergent spacetime line element (preferred slicing)
In the preferred foliation induced by phase/action order (a physically distinguished slicing, not assumed to be fundamental time),
define:
$$
ds² = -c² dτ² + h_ab(x) dx^a dx^b,
$$
where $a, b = 1, \ldots, D^*$ and $D^*$ is the emergent embedding dimension from §4.4.4 (empirically $D^*\approx 3$ in our regime).

**Lorentz covariance status (explicit).**  
A preferred slicing exists at the projection-construction level because $\sigma_s$ defines a distinguished order.
Local Lorentz covariance is a required matching target, not a current result. SU(2)
equivariance covers spatial rotations only. The project must compute dispersion,
directional limiting speeds, boost-sensitive observables, and scale dependence before
an infrared fixed-point claim can be evaluated.

#### 4.4.6 Summary: the projection map as output object
With the above stages, the projection map can be summarized as:
$$
Π(A, ω, α) = ( M^{D*}, {x_i}, h_ab(x), τ(x), Φ(x), {ρ_R}, Cap(∂R), … )
$$
with:
- $D^*$ and $\{x_i\}$ from stress-minimizing embedding of the correlation metric space (dimension selected by distortion/complexity),
- $h_{ab}(x)$ from SPD-constrained local metric reconstruction on the embedded point cloud,
- $\mathrm{Cap}(\partial R)$ from the MI-weighted graph cut functional (primitive “boundary size” for capacity bounds),
- $\delta\rho$ (entropy-contrast) from local reduced states relative to a vacuum baseline,
- $\Phi$ from the screened Poisson equation on the correlation graph $(\Delta_w + \mu² I)\Phi = \delta\rho$,
- $\tau$ from $d\tau = \beta_0 \cdot \exp(\Phi)\, dS_{\text{act}}$ (phase/action order converted to clock time).

**Where freedom remains (auditable).**  
The remaining degrees of freedom are explicit and constrained:
- coarse-graining selection tolerances ($\varepsilon$, choice of channel norm, phase-window weight $w(s)$, drift step $\delta$),
- MI→distance map $f$ and MI→weight map $\kappa$ (both fixed monotone families, not fitted per region),
- embedding criterion ($\varepsilon_{\text{embed}}$ or $λ_dim$) and neighborhood rule $N(i)$,
- SPD regularization strength $λ_spd$,
- screening mass $\mu$ (default $\mu = 0$; treated as an optional controlled IR-modification module).

These are the only places “parameter fitting” could enter; the framework requires universality (same choices everywhere) and
empirical matching constraints to keep them from becoming arbitrary.

### 4.5 Symbol glossary
| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | Substrate object |
| $\mathcal{A}$ | Substrate algebra of relational observables |
| $\mathcal{A}_{\text{rel}}$ | Relational “bulk” algebra (pre-geometric; approximable by finite matrices) |
| $\mathcal{A}_F$ | Finite internal algebra encoding gauge/representation structure (optional program) |
| $\omega$ | State on $\mathcal{A}$ |
| $\alpha$ | Internal symmetry action on $\mathcal{A}$ |
| $H$ | Phase/action ordering generator (order parameter; not substrate time) |
| $\Pi$ | Unique physical projection map |
| $M^3$ | Emergent 3D spatial structure |
| $g_{\mu\nu}$ | Emergent spacetime metric in projection |
| $t$ | Emergent time coordinate/order metric in projection |
| $\rho(x)$ | Encoding density field in projection |
| $\ell_*$ | Minimal encoding resolution scale (Planck-like) |
| $\eta$ | Dimensionless coefficient in the area-law capacity bound |
| $\gamma$ | Junction-access gating parameter (open module) |
| $\Delta$ | Junction-access correction functional (open module) |


### 4.6 Substrate algebra and parameter budget (review-facing)

This section makes explicit (i) what is fixed by structural choice versus (ii) what remains to be calibrated to match empirical reality. The goal is to prevent “free-form projection fitting” by keeping the degrees of freedom auditable.

#### 4.6.1 Algebraic architecture (what the substrate “looks like”)
**Substrate data.** The substrate is specified by:
$$
S = (A, ω, α)
$$
where:
- $\mathcal{A} = \mathcal{A}_{\text{rel}} \otimes \mathcal{A}_F$ is a relational algebra with optional finite internal factor,
- $\omega$ is a state on $\mathcal{A}$,
- $\alpha: SU(2) \to \mathrm{Aut}(\mathcal{A})$ is the compact internal symmetry action (Section 4.2).

**Key point.** $\mathcal{A}_{\text{rel}}$ is not “functions on a space.” It is a non-spatial relational algebra. Any appearance of locality/geometry arises only after projection.

#### 4.6.2 Where continuous structure enters (and why this does not force physical infinities)
- Continuous symmetry (SU(2)-like) and continuous phase/action order are treated as idealized descriptions.
- Physical finiteness is enforced at the projection level via the area-law bound and bounded intensities:
  - finite distinguishability in any finite region (Section 6),
  - saturation rather than divergence for densities/curvatures (Postulate P5).

In other words: *the model forbids unbounded physical observables, not the use of continuous mathematics.*

#### 4.6.3 Parameter budget (what must be fixed or fitted)

| Category | Examples in this framework | Status / how constrained |
|---|---|---|
| **Structural (discrete) choices** | Choice of algebra class for $\mathcal{A}_{\text{rel}}$; choice of internal algebra $\mathcal{A}_F$; compact internal symmetry group; equivalence-class uniqueness of projection postulate | Architecture decisions (not continuously fitted). Must be defended by conceptual minimality and consistency. |
| **Empirical scale-setting constants** | $\ell_*$ (minimal encoding patch); $\eta$ (area-law coefficient); effective constants matching $(c, \hbar, G, \Lambda)$ | Expected to be measured unless derived from a deeper principle; treated like fundamental constants in early drafts. |
| **Projection response functions** | Mapping from encoding density to metric scaling; mapping from phase-order to clock time (e.g., monotone/saturating functions) | Primary risk of “parameter fitting.” Must be restricted to minimal families with clear constraints and matching requirements. |
| **Substrate state selection** | Choice/characterization of $\omega$ | Must be constrained by a selection principle (symmetry, equilibrium relative to canonical flow, minimal information, etc.) to avoid embedding arbitrary structure by hand. |
| **Open junction module** | $\gamma(\text{regime})$ and $\Delta(\ldots)$ | Must satisfy $\gamma \approx 0$ in all tested regimes; left open otherwise, but constrained by normalization/positivity and by compatibility with known no-signaling bounds. |

#### 4.6.4 Anti-overfitting constraints on the projection map $\Pi$
To keep the framework falsifiable and reviewable, $\Pi$ should be restricted by explicit principles such as:

1) **Universality:** the same mapping rules apply across all regions and epochs (no ad hoc patchwork).  
2) **Correlation-based inference:** reconstruction reads correlations rather than
   coordinate labels, while every microscopic interaction graph or other structural
   prior is disclosed and controlled.
3) **Minimal functional freedom:** response functions are chosen from low-parameter families (monotone, saturating) with their parameters tied to known constants.  
4) **Matching constraints:** $\Pi$ must reproduce GR in weak-field and tested strong-field regimes and reproduce standard quantum statistics in laboratory regimes.

These constraints convert “projection” from a free fitting function into a tightly parameterized construction with a transparent parameter budget.

#### 4.6.5 Optional parameter reduction via a canonical phase-order flow
To reduce arbitrariness in the choice of the phase/action ordering generator, one may impose:

- the phase-order flow is **canonical** given $(\mathcal{A}, \omega)$ (i.e., derived from the substrate state rather than chosen as an additional free input).

This provides a principled origin for the preferred ordering used to construct projection-time while preserving substrate atemporality.

## 5. Projection outputs and emergent time
Time does not exist at the substrate level. Instead, projection constructs an objective time-order by mapping phase/action order to a temporal metric. Conceptually: substrate provides an ordering structure; projection interprets it as time.

### 5.1 Order-to-time mapping
A minimal requirement is a monotone mapping from phase order to time:
$$
t = f(Φ) with f' > 0
$$
where $\Phi$ is a projection-accessible phase/order functional.

## 6. Finite distinguishability and Planck-scale resolution
The framework adopts an analog substrate intuition while enforcing bounded physical realizability in projection. Planck-scale resolution is treated as a limit on physically distinguishable information, not necessarily as a lattice of spacetime points.

### 6.1 Finite distinguishability via algebraic cut-capacity
To completely eliminate geometric circularity, we define the distinguishability bound using a strictly pre-geometric **algebraic cut-capacity** functional. This ensures that "area" is an output of the theory, not an input.

**1. Algebraic Cut-Capacity.**
Let $R \subset V$ be a finite set of emergent funnels. Define the **cut-capacity** $\mathrm{Cap}(\partial R)$ purely from the correlation graph weights:
$$
Cap(∂R) := Σ_{i∈R, j∉R} κ(I_{ij})
$$
where $I_{ij}$ is the mutual information between funnels $i$ and $j$, and $\kappa$ is the edge-weight kernel. This quantity measures the total "information flux" connecting $R$ to its complement, with no reference to spatial geometry.

**2. The Araki Entropy Bound.**
For rigorous compatibility with Type III limits, we replace the naive von Neumann entropy with the **Araki relative entropy** evaluated on the buffer zones. Let $N_R = \bigotimes_{i \in R} N_i$ be the composite Type I factor for region $R$.
The bound requires that the distinguishable information content of $R$ (measured relative to the vacuum state $\omega_{\text{vac}}$ on the buffer) is limited by the cut-capacity:
$$
S_{Araki}( ω |_R || ω_vac |_R ) ≤ η · Cap(∂R)
$$
where $\eta$ is a dimensionless constant ($O(1)$).

**3. Area-scaling conjecture.**
The following is an unproved matching hypothesis, not a theorem: if the
stability-selected correlation graph admits a stable low-distortion three-dimensional
limit, then $\mathrm{Cap}(\partial R)$ may scale proportionally to the geometric
surface area $A_{\text{geo}}(\partial R)$ defined in that limit.
    $$
    Cap(∂R) ∝ A_{geo}(∂R) / ℓ_*^2
    $$
This reverses the standard logic: we do not assume an Area Law to get geometry; we assume a Cut-Capacity Bound, and recover the Area Law only in regimes where a stable geometry emerges.

### 6.2 Finite distinguishability
The operational consequence of the capacity bound (§6.1) is a limit on the effective dimension of the Type I funnel factor $N_R$ (§4.4.2) associated with any finite region $R$:
$$
dim(N_R) ≲ exp( η · Cap(∂R) )
$$
This is stated entirely in terms of the algebraic cut-capacity — no absolute von Neumann entropy $S(R)$ appears. In manifold-like regimes where $\mathrm{Cap}(\partial R) \propto A_{\text{geo}}/\ell_*^2$, the bound reproduces the familiar area-law scaling of distinguishable degrees of freedom.

## 7. Quantum statistics as projection-limited inference
Quantum probabilities are framed as a consequence of limited distinguishability: projection exposes only coarse-grained effective states for subsystems. Many substrate micro-configurations correspond to the same effective projection state.

### 7.1 Effective states
For a subsystem associated with region $R$, projection yields an effective density operator $\rho_R$ acting on $\mathcal{H}_R$. Measurement outcomes are computed by the Born-rule form in empirically accessed regimes:
$$
P(i) = Tr(ρ_R E_i)
$$
### 7.2 Program toward a derivation
A derivation program (not completed in this draft) is to show that the trace-rule probability assignment is uniquely selected by a combination of:
1) additivity for exclusive outcomes,  
2) invariance under unitary transformations induced by internal symmetry actions, and  
3) finite distinguishability constraints that prevent access to deeper microstructure.

## 8. Emergent geometry and GR matching
The effective spacetime metric $g_{\mu\nu}$ is treated as a projection output, not a substrate primitive. Geometry is determined by projection-accessible fields including encoding density $\rho(x)$.

### 8.1 Metric definition as projection output (replacing an abstract constitutive law)
In this framework, we do not postulate a fundamental constitutive law of the form $g_{\mu\nu} = F(\rho, ...)$. Instead, the effective spacetime metric is the **output of the projection construction** defined in Section 4.4.

The projection pipeline replaces the role of a constitutive law by constructively generating geometry from correlation.
1.  **Locality:** Defined by the stability-selected cell net and mutual information (Section 4.4.2–4.4.3).
2.  **Spatial Metric $h_{ab}$:** Defined by the relational embedding and local SPD reconstruction (Section 4.4.4).
3.  **Lapse (Clock-Rate Potential $\Phi$):** Defined by the Laplacian clock-rate potential on the correlation graph (Section 4.4.5).
4.  **Shift Vector:** Extracted via Gromov-Wasserstein optimal transport of the correlation graph between adjacent phase-order slices (Section 4.4.5).

Items 2–4 are intended to supply analogues of the fields in an **ADM
(Arnowitt–Deser–Misner) decomposition**. At present, $h_{ab}$ is an embedding-space
fit, $\Phi$ is a scalar graph diagnostic, and the shift is unimplemented. Without
extrinsic curvature, Hamiltonian and momentum constraints, tensor evolution, and
covariance tests, they do not constitute an ADM gravitational solution.

**Entanglement-shear proposal.** A future dynamical model may couple clock-rate
contrasts to changing correlations and hence to reconstructed spatial diagnostics. No
such evolution law, shift vector, extrinsic curvature, or tensor constraint is
implemented. “Entanglement shear” is therefore a proposed mechanism, not a derivation
of nonlinear curvature.

### 8.2 Proposed GR matching protocol (unimplemented)
GR closure is a required future test on a reconstructed effective geometry. The current
code provides only a tensor-agnostic mismatch function and a 2D embedding-space
angle-deficit proxy.

#### 8.2.1 Candidate intrinsic discrete geometry

1.  **Intrinsic discrete geometry:** construct and test a Vietoris–Rips or other
    justified complex from $(V,d_G)$. The current Delaunay triangulation uses MDS
    coordinates and is only a visualization/diagnostic proxy; it is not guaranteed
    homeomorphic or evidence-equivalent to an intrinsic complex.
2.  **Discrete Metric:** Assign edge squared-lengths $l_{ij}^2 = h_{ab}(x_i) \Delta x^a \Delta x^b$ consistent with the reconstructed local metric.
3.  **Discrete Curvature (Regge):** Compute the curvature using **Regge Calculus**. The curvature is concentrated on the $(D^*-2)$-dimensional bones (hinges). For $D^*=3$, these are edges. The deficit angle $\epsilon_h$ at hinge $h$ is:
    $$
    \epsilon_h = 2\pi - \sum_{cell \supset h} \theta_{cell}(h)
    $$
    where $\theta_{\text{cell}}(h)$ is the dihedral angle of the tetrahedron at hinge $h$.
    Discrete conservation/Bianchi residuals must be computed explicitly with boundary,
    dual-volume, and refinement effects included.

#### 8.2.2 Proposed closure mismatch
If validated geometric and source tensors become available, compare them using:

Define the **discrete Regge Einstein tensor** $G_h$ on each hinge $h$:
$$
G_h \cdot l_h := \epsilon_h - \Lambda V_h
$$
(normalized appropriately by Voronoi dual volumes).

Define the **closure mismatch** $M(L)$ over a region $V_L$:
$$
M(L) := \left\| G_{Regge} - 8\pi G \, \Pi_{graph}(T_{\mu\nu}^{eff}) \right\|_L
$$
Neither $G_{\mathrm{Regge}}$, a conserved $T_{\mu\nu}^{eff}$, nor
$\Pi_{\mathrm{graph}}$ is implemented. Raw Araki contrast has failed the tested linear
source criterion, and scalar clock data cannot substitute for the missing tensor
components.

**Matching requirement:** under controlled refinement, $M(L)$ and discrete
conservation residuals must converge within a predeclared tolerance while known
weak-field and dynamical observables are recovered.

#### 8.2.3 Proposed weak-field (Newtonian) consistency test
A future weak-field claim must be evaluated under controlled refinement, with a
physical source and boundary conditions declared independently of the desired result.

1.  **Discrete Scalar Diagnostic:** The clock-rate potential $\Phi_i$ is defined on the graph. It may be compared with a Newtonian potential only after its source, normalization, dimensions, and continuum behavior have been validated.
2.  **Inferred Source Diagnostic:** Given an intrinsic discrete Laplacian, infer a comparison quantity using
    $$
    4\pi G \rho_i \approx (\Delta_{\text{VR}} \Phi)_i
    $$
    where $\Delta_{\text{VR}}$ is the cotan-weighted Laplacian on the simplicial complex.
3.  **Consistency:** Test whether the inferred quantity matches the independently supplied coarse-grained source and whether the relation converges with refinement. The present code has not passed this test.

#### 8.2.4 Parameter calibration versus parameter fitting
Matching GR introduces *scale-setting* constants (analogous to $c$, $G$, $\Lambda$) through:
- the unit choice $\ell_*$ (minimal encoding patch scale),
- the clock-rate field coupling $\beta_0$ and potential $\Phi_i$ (controlling gravitational time dilation),
- the distance mapping $f(I/I_0)$ (sets correlation-to-distance conversion).

To avoid unconstrained fitting, the framework treats these as:
1) **universal functions/constants** (no region-by-region tuning), and
2) **minimally parameterized** (e.g., monotone/saturating families) with calibration fixed by a small set of benchmark tests (Newtonian limit, gravitational redshift, lensing).

Any additional freedom beyond this limited set should be declared explicitly as a model extension.


## 9. Standard Model compatibility (program)
A full embedding of the Standard Model is not derived in this draft. However, an algebraic route is identified that is compatible with a pre-geometric substrate.

### 9.1 Candidate internal algebra
This framework treats Standard Model unification as a **programmatic module** that should not alter the core projection machinery
(cells → correlations → geometry → clock-rate). The role of an internal finite algebra is to supply a compact representation structure
for gauge/charge degrees of freedom that can be attached to the already-defined emergent cell net.

A commonly studied finite algebra whose unitary structure can realize the Standard Model gauge group is:
$$
A_F = C ⊕ H ⊕ M_3(C).
$$
### How this integrates non-ad-hoc with the projection
A natural integration point is at the cell level: treat each effective cell as carrying an internal factor,
schematically:
$$
A_i  ≈  M_{d_i}(C) ⊗ A_F,
$$
(with the same “approximate split” caveats as in §4.4.2 for continuum/QFT limits).

- The compact SU(2)-like structure is naturally supported by the quaternionic component $H$.
- $M_3(\mathbb{C})$ supports $SU(3)$-type color structure.
- $\mathbb{C}$ supports a $U(1)$-type phase.

Gauge structure then appears as **internal automorphisms** (unitaries) acting on the $\mathcal{A}_F$ factor that
(i) commute with the symmetry-commuting constraints of the projection and
(ii) can be promoted to edge data (a connection) on the emergent locality graph if desired.
Matter excitations correspond to stable representation labels of the internal algebra across the cell net.

**Status.**  
- This section is not required for the GR/QM recovery claims of the core draft.
- It provides a concrete place to attach Standard Model degrees of freedom *once* the geometry/clock pipeline is fixed,
so it is not “bolted on” to define spacetime itself.

## 10. Entanglement and junction accessibility (open module)
Entanglement is treated as a non-factorizable relational structure that may be naturally nonlocal at the substrate level. Operationally, however, currently observed physics obeys no-signaling: local marginals do not depend on distant choices.

### 10.1 Empirical constraint: no-signaling in accessible regimes
For accessible operations and regimes:
$$
P(b | x, y) = P(b | x)   (within experimental bounds)
$$
### 10.2 Parameterized junction-access extension
To keep the possibility of additional operational access open without asserting it, introduce a gating parameter $\gamma(regime) \in [0, 1]$ and write:
$$
P(b | x, y) = Tr(ρ_B E_b^{(x)}) + γ · Δ_b(x, y; ρ_AB)
$$
Constraints: (i) $\sum_b \Delta_b = 0$ for normalization, (ii) probabilities remain in $[0,1]$, and (iii) $\gamma \approx 0$ in all regimes tested so far.

## 11. Evaluation, falsification, and test program

### 11.1 What “prove or disprove” means here
This framework is structured as (i) **definitions** of substrate and projection primitives, plus (ii) **matching requirements** to known physics in empirically accessed regimes, plus (iii) optional **extension modules** (e.g., junction access) that may introduce new testable deviations. In the scientific sense, the framework is:
- **disprovable** if it cannot satisfy its own matching requirements without introducing hidden time/geometry, or if it predicts deviations already excluded by experiment;
- **supported** if a small-parameter instantiation reproduces broad classes of GR/QFT phenomena while remaining internally consistent and non-arbitrary.

### 11.2 Non-negotiable constraints (must hold in accessible regimes)
The following are hard requirements for viability:
- **No hidden substrate time:** all ordering must be definable as phase/action order; no background time parameter may be reintroduced implicitly.
- **Disclosed structural priors:** inference must not read coordinate labels or
  reference edges, and any topology encoded in the Hamiltonian/state must be reported.
- **Operational no-signaling (junction module):** in all regimes already probed experimentally, the junction-access gate must satisfy $\gamma \approx 0$ so that local marginals do not depend on distant choices.
- **Universality:** scale-setting constants and response functions (e.g., $f$, $\beta$) must be global (not tuned region-by-region).
- **GR closure on a controlled discrete complex:** intrinsic geometry, source tensor,
  conservation, boundary treatment, and refinement convergence must all pass before a
  GR-matching claim is made.

### 11.3 Mathematical status table (definition / theorem / conjecture / matching)

The table below classifies the framework’s most important claims by **mathematical status**. This is intended to make reviewer evaluation precise and to prevent “definition vs prediction” ambiguity.

**Legend**
- **Definition:** fixed by the construction; not an empirical claim by itself.
- **Theorem (toy model):** provable/verifyable in finite-dimensional approximations and explicit numerical realizations.
- **Conjecture / open:** plausibly true but not yet proven in generality; may require additional technical assumptions.
- **Matching condition:** required to hold in empirically accessed regimes; falsified if it cannot be satisfied without ad hoc tuning.

| Item | Status | What it asserts | Validation route / failure mode |
|---|---|---|---|
| Substrate triple \((\mathcal A,\omega,\alpha)\) | Definition | The substrate is specified as an atemporal, pre-geometric relational algebra with a compact internal symmetry action. | Pure definition; failure only if later steps implicitly add substrate spacetime structure. |
| Phase/action ordering flow \(\sigma_s\) | Definition (if postulated) / Conjecture (if derived) | A distinguished ordering flow exists; \(s\) is an order parameter, not time. | If derived (canonical/modular flow): show existence conditions on \((\mathcal A,\omega)\). If postulated: keep minimal and symmetry-compatible. |
| Cell net \(\{\mathcal A_i,E_i\}\) at finite distinguishability | Definition | Projection-resolution stage outputs effective “cells” (Type I approximants / toy matrix factors) and coarse-graining maps. | Toy models: explicit construction. General AQFT: implement via split-property funnels; failure if no consistent finite-resolution approximation exists. |
| Symmetry-commutation of coarse-graining | Definition / constraint | \(E_i\circ\alpha_g=\alpha_g\circ E_i\) for all \(g\in SU(2)\). | Check directly in constructions; failure if recovered physics requires symmetry-breaking at the projection-selection level. |
| Stability selection via leakage minimization \(\mathcal L_{\text{leak}}\) | Definition | Among admissible coarse-grainings, select those minimizing phase-flow leakage across cell boundaries (optionally tie-break by drift). | Toy models: compute minimizers numerically; failure if the minimizer is always trivial (information-destroying) despite retention constraints. |
| Existence of minimizers for \(\mathcal L_{\text{leak}}\) under constraints | Theorem (toy model) / Conjecture (general) | Minimizers exist for finite toy systems and well-posed admissible sets; general operator-algebraic existence is open. | Toy models: compactness/finite search; general case: requires technical assumptions on admissible channel set and topology. |
| Capacity bound using algebraic cut capacity \(\mathrm{Cap}(\partial R)\) | Definition | Finite distinguishability is bounded by a non-geometric boundary-capacity functional computed from correlation weights. | Pure definition; failure only if it cannot be made compatible with recovered continuum behavior. |
| Emergent “area law” scaling | Matching output (and/or theorem in restricted classes) | In manifold-like regimes, \(\mathrm{Cap}(\partial R)\propto A_{\text{geo}}(\partial R)/\ell_*^2\). | Validate in toy models that yield low-stress embeddings; falsified if scaling is generically volume-like even in 3D-like regimes. |
| Emergent locality via QCMI-screened mutual information \(I_{ij}\) and graph metric \(d_G\) | Definition / partial prototype | Pairwise MI and blind graph routing are implemented; QCMI screening is not. The tested Hamiltonians contain chain/grid interaction graphs. | Fails if non-geometric controls produce stable geometric declarations or encoded locality is not robustly recovered. |
| Dimension \(D^*\) from embedding distortion | Finite benchmark + matching target | The criterion selects \(D^*=1\) and \(D^*=2\) in selected exact chain/grid cases whose Hamiltonians encode those interactions. | Requires parameter robustness, non-geometric rejection, and refinement; no continuum dimension has been measured. |
| Spatial metric \(h_{ab}(x)\) from SPD-constrained local fit | Definition | The emergent spatial metric is reconstructed from neighbor distances with explicit positive-definiteness constraints. | Validate numerically (SPD fit succeeds and is stable under refinement of the cell net); failure if SPD enforcement destroys the ability to match observables. |
| Clock-rate field \(\beta\) and proper time \(d\tau=\beta\,dS_{\text{act}}\) | Definition / numerical diagnostic | A graph constraint and sign convention are implemented; the physical source and independent action-distance clock are open. | Requires source validation, operational clock comparison, and continuum weak-field matching. |
| Spacetime metric \(g_{\mu\nu}\) | Proposed construction | Embedding/local-metric and clock diagnostics exist, but shift, extrinsic curvature, constraints, and tensor evolution do not. | Requires a complete discrete geometry and convergent GR closure test. |
| GR closure mismatch functional \(M\) | Definition + matching condition | The discrete Regge Einstein tensor \(G_{Regge}\) on the Vietoris-Rips simplicial complex must satisfy Einstein closure within tolerance (§8.2). | Falsified if no small-parameter instantiation yields small \(M\) across standard tests. |
| Born-rule statistics in accessible regimes | Matching condition (derivation open) | Outcome statistics must match standard QM in all currently tested regimes. | Falsified by any predicted deviation already excluded; derivation from projection-limited inference is an open program item. |
| Operational no-signaling in accessible regimes | Matching condition | Local marginals must not depend on distant choices (junction gate \(\gamma\approx 0\)). | Falsified by any proposal that enables controllable signaling in ordinary Bell-test regimes. |
| Junction-access deviations \(\gamma(\cdot)>0\) | Open module | Whether additional operational access exists in extreme regimes is left open and parameterized. | Must remain consistent with existing constraints; becomes testable if a concrete activation regime is specified. |
| Local Lorentz covariance | Matching condition / conjecture | SU(2) equivariance tests spatial rotations only; boost recovery and spin-connection locking are not implemented. | Measure dispersion, common limiting speeds, anisotropy, boost-sensitive observables, and compatibility with experimental bounds. |


### 11.4 Internal validation tests (toy-model / simulation tests of Π)
These tests evaluate finite pieces of the proposed projection. T1 and T2 are selected
toy benchmarks, T4 is a graph-solver/sign diagnostic with a manual source, and T5 is
a phenomenological cellular-automaton analogy. None constitutes completion of the
full projection. T3 remains open.

**T1 — Recovery of encoded locality in selected states. [FINITE BENCHMARKS]**
Two exact simulations test correlation-graph inference and finite embedding diagnostics.
They do not validate the full projection or clock-source physics.

**T1a — 1D Chain ($N = 8$ qubits).** The full thermal density matrix $\rho = e^{-\beta H}/Z$ of an 8-qubit nearest-neighbour Heisenberg chain was computed exactly. From this state, the mutual information $I_{ij}$ was evaluated for all qubit pairs using the exact von Neumann reduction (the finite-dimensional proxy for the Araki formulation, §4.4.3). The resulting MI graph exhibited the expected exponential decay with chain distance, confirming that the correlation structure encodes the underlying 1D topology.

For the declared inference settings, the reconstructed graph preserves the chain
ordering and the dimension criterion selects $D^*=1$. The nearest-neighbor interaction
graph is encoded in the Hamiltonian even though coordinate labels are not read during
inference. At four coarse cells the three inferred edges are also exactly the minimum
spanning tree, so perfect edge recovery is non-discriminating in this benchmark.

The example also runs the clock solver with the explicitly non-physical von Neumann
placeholder. That output is a pipeline diagnostic, not a gravitational observable.

**T1b — 2D Grid ($N = 9$ qubits, $3 \times 3$).** The exact thermal state
comes from a Hamiltonian with a square-grid interaction graph. The blind MI-gap rule
recovers the 12 reference nearest-neighbor edges in the selected benchmark, and MDS
selects $D^*=2$. Coordinate labels are hidden from inference; grid adjacency is not.

These selected finite results show that the criterion can distinguish one- and
two-dimensional encoded interaction families. Parameter sensitivity, non-geometric
controls, and refinement are required before interpreting $D^*$ as an emergent
continuum dimension.

The grid heatmap is generated from the placeholder source. SU(2)-invariant one-site
Gibbs marginals make that source constant; zero-mode removal therefore gives
$\Phi=0$ up to round-off. The panel is retained as an explicit degeneracy diagnostic,
not as a clock-potential validation.

**T2 — Stability under phase-flow (locality from leakage minimization). [FINITE TOY BENCHMARK]**
An exhaustive combinatorial search evaluated **all 105 possible 4-cell coarse-graining partitions** of the 8-qubit Hilbert space (each partition dividing 8 qubits into 4 cells of 2 qubits each). For each candidate partition $\{E_i\}$, the phase-flow leakage $\mathcal{L}_{\text{leak}}$ was ranked with an unnormalized common-Haar-probe mean proportional to the Hilbert--Schmidt channel norm at fixed Hilbert-space dimension. The omitted $d(d+1)$ factor prevents comparison of the reported values across dimensions but does not change the within-system ranking.

The $\mathcal{L}_{\text{leak}}$ ranking selected the contiguous 1D blocks
$\{(1,2), (3,4), (5,6), (7,8)\}$ for the tested configuration. Non-local
partitions had larger sampled leakage values.

The nearest-neighbor Heisenberg Hamiltonian already encodes the chain interaction
graph, and the code performs exhaustive ranking rather than the proposed causal
gradient flow. This benchmark therefore supports stability-based recovery of encoded
locality in one small system; it is not a proof of topology-free emergence or of
large-system convergence.

**T3 — Robustness of discrete geometry under refinement. [OPEN]**
Show that the Vietoris-Rips simplicial complex, deficit angles, and discrete Regge Einstein tensor $G_{\text{Regge}}$ are stable when the cell net is refined (increasing cell count $|V|$ while holding the physical source configuration fixed). Specifically: the integrated closure mismatch $M$ (§8.2) must converge rather than diverge, and topological invariants of the simplicial complex (Euler characteristic, homology) must remain stable.

**T4 — Graph-potential and redshift-sign diagnostic. [NUMERICAL CHECK; PHYSICAL SOURCE OPEN]**
A localized negative point source was manually injected into a 2D MI-weighted graph
and the screened graph equation $(\Delta_w + \mu^2 I)\Phi = \delta\rho$ was solved.
The source was not produced by an entropy or KMS law. The diagnostic found:

1. **Monotonic shell ordering:** $\Phi$ decreased from the boundary toward the
   source. The small graph has too few shells to establish a $1/r$ law.
2. **Exact symmetry preservation:** the potential respected the full lattice symmetries of the underlying graph — nodes at equal graph-geodesic distance from the source received identical $\Phi$ values, confirming that the Laplacian solver introduces no spurious anisotropy.
3. **Clock-sign consistency:** under the chosen mapping $d\tau\propto e^\Phi$, the
   defined ratio $1+z=\exp(\Phi_{\text{boundary}}-\Phi_{\text{source}})$ is positive
   for an emitter in the well.

This validates the finite graph solver, symmetry, and internal sign convention only.
It does not establish a physical source, Newtonian limit, or gravitational redshift.

**T5 — Phenomenological stability and radiative cooling. [CA ANALOGY]**
A cellular automaton on a supplied 2D grid tests one chosen rule in which purity decay,
a death threshold, and optional probabilistic cooling control cell survival. It is not
the quantum leakage dynamics implemented by the exact backend.

Only one cooling-enabled run is committed. It retains a nonzero population but declines
from 33 to 27 cells and therefore fails its stable-or-growing population criterion.
No matched no-cooling artifact or multi-seed sweep exists. The spatial grid and update
law are inputs, so the run neither measures a cooling effect nor demonstrates emergent
geometry.

The committed examples are reproducible at their stated evidence tiers: exact finite
benchmarks, graph-solver diagnostics, or CA analogies. T3 requires intrinsic Regge and
refinement infrastructure.

#### 11.4.1 Toy Model Validation: Stability and Radiative Cooling (Detail)
This phenomenological cellular automaton probes a possible thermodynamic analogy for
$\Pi_{\text{res}}$; it is not a derivation of a substrate requirement.

**Setup.** A 2D grid of cells evolves under nearest-neighbour interaction (purity decay proportional to Bloch-vector misalignment), a selection rule that removes cells whose entropy exceeds a leakage death-threshold, and optional probabilistic entropy export ("radiative cooling") to the environment.

**Current result.** For the selected seed and parameters, the cooling-enabled run
survives but loses population. The survivor-entropy criterion is partly structural:
cells above the threshold are culled and newborns are pure. Whether cooling changes
survival requires a matched multi-seed control that has not been implemented.

**Implication for the framework.** The result motivates an explicit open-versus-closed
control in future quantum dynamics. It does not establish a theorem about fixed points
of the unimplemented causal coarse-graining flow.

#### 11.4.2 Methodological Note on Finite-Dimensional Reductions
The finite T1/T2 benchmarks use standard von Neumann entropy and mutual information.
For finite matrices, mutual information equals the corresponding relative entropy
$D(\rho_{ij}\|\rho_i\otimes\rho_j)$. This is an exact finite identity, but it does not
establish the existence or scale consistency of the proposed Type III construction.

The exhaustive T2 search locates the best sampled finite objective among enumerated
partitions. Equivalence to a causal gradient-flow attractor is an unproved conjecture
and requires an implemented flow plus convergence and basin analysis.

### 11.5 Empirical compatibility checks (must reproduce known physics)
A minimal set of comparisons that any instantiation must pass:
- **Newtonian limit / weak field:** recover standard gravitational acceleration and redshift behavior from the discrete clock-rate potential $\Phi$ on the correlation graph (§8.2.3).
- **Light bending and time delay:** after intrinsic geometry and a physical source are
  available, compare refined discrete causal trajectories with weak-field predictions.
- **Consistency of clock mapping:** the proper-time formula $d\tau = \beta_0 \cdot \exp(\Phi) \cdot dS_{\text{act}}$ must reproduce observed gravitational time dilation in weak-field regimes without ad hoc spatial dependence.
- **Quantum statistics:** for laboratory-scale systems, projection must reproduce standard interference and entanglement statistics (Born-rule form at the operational level).
- **No-singularity claims (if asserted):** if the model claims bounded curvature universally (deficit angles saturate rather than diverge), it must not conflict with any observed high-density astrophysical phenomena.

### 11.6 Distinctive empirical commitments and lethal falsifiers
The framework becomes strictly falsifiable when it asserts universal constraints that can be violated by specific mathematical or empirical counterexamples. The following are severe falsification tests:

**F1 — Dimensional Collapse.**
The framework would fail this test if controlled families systematically select a
trivial or divergent dimension instead of a stable continuum value. Current finite
chain/grid benchmarks return $D^*=1$ and $D^*=2$, respectively, but those Hamiltonians
already encode the tested adjacency and the samples are too small to establish
stability, topology-free emergence, or a continuum result. Higher-dimensional,
topologically non-trivial, null, and refinement controls remain required.

**F2 — Volume-Law Saturation.**
Falsified if numerical simulations of split-property funnels show cut-capacity scaling with volume rather than boundary cuts in low-distortion regimes.

**F3 — Lorentz-recovery failure.**
The conjecture fails if a specified large-system limit retains direction-dependent
dispersion, unequal limiting speeds, or preferred-frame observables incompatible with
current bounds. The project has not yet derived a suppression law or a numerical
experimental prediction.

**F4 — The Two-Potential Disconnect.**
Falsified if matching the Shapiro delay ($\Phi_t$) and spatial lensing ($\Phi_s$) for the exact same source requires fundamentally different parameter calibrations.

**F5 — Junction Accessibility (Open Module).**
If $\gamma(\text{regime})$ is asserted nonzero in any accessible regime, then it predicts detectable signaling via marginal shifts. Conversely, existing no-signaling constraints imply $\gamma \approx 0$ in all currently accessed regimes; any model variant predicting otherwise is excluded.

### 11.7 A practical “test matrix” (what to measure and what would count as failure)
| Target claim/module | Observable proxy | What must hold | What would disprove it |
|---|---|---|---|
| Stability-selected locality | Leakage/drift metrics | $\mathcal{L}_{\text{leak}}$ small, drift small at fixed capacity | No stable decomposition exists without collapsing capacity |
| Emergent 3D space | Dimensionality of embedding | Best-fit embedding dimension ≈ 3 across scales | No stable low-dimensional embedding; dimension drifts wildly |
| GR closure in weak field | Extracted $\Phi$, redshift, lensing | Matches standard weak-field phenomenology | Systematic mismatch not removable by universal calibration |
| Finite distinguishability (area law) | Scaling of $S(R)$ with boundary cut | $S(R)$ bounded by area scaling (operationally) | Demonstrated generic violation of area scaling at fixed regime |
| Junction module (γ) | Marginal dependence tests | No measurable marginal dependence in known regimes | Robust marginal dependence without classical side-channel explanation |

### 11.8 Parameter discipline as a falsifier (anti–overfitting rule)
A scientifically conservative stance is to treat **excess functional freedom** as a failure mode. Concretely:
- If matching GR/QFT requires $\beta(x)$ or $f(I)$ to vary by environment beyond a small universal parameter family, the model is overfit.
- If matching requires replacing the stability selection rule with ad hoc, state-dependent exceptions, the “unique projection” postulate is violated.


## 12. Discussion and open problems
Open problems include:

1) **Toy-model realizations of $\Pi$.** Scaling the tensor networks and GPUs to $D^* \approx 3$. Construct explicit toy substrate algebras and states $(\mathcal{A}, \omega)$ for which the stability-selected coarse-graining $\{E_i\}$ can be computed (or approximated) and the induced locality graph and embedding can be visualized.

2) **GR matching as an emergent closure.** Demonstrate in a controllable regime (e.g., weak field) that the metric produced by the embedding + clock mapping satisfies Einstein’s equation to the required approximation, and identify what coarse-grained stress-energy corresponds to the projection-level excitations.

3) **Dynamic cell net / dynamic spacetime (“geometrogenesis”).** The selection principle $E$ is defined over a phase-order window via the weight $w(s)$. In general, the minimizer may depend on the chosen window and (if one allows it) on phase-order location. A “moving-window” or adiabatic selection $E*(s)$ would yield a phase-ordered family of graphs and embeddings, providing a route to dynamically evolving geometry within projection. The present draft treats $\{E_i\}$ as fixed for simplicity; extending to $E_i(s)$ is a natural next step and should be formulated carefully to preserve symmetry-commutation and retention constraints.

4) **Computational tractability.** Global minimization of leakage over all decompositions is intractable at large scale. For a physical theory, this should be interpreted as a variational/thermodynamic principle: the universe’s effective decomposition is expected to be selected by natural relaxation/typicality under the phase-flow, not by an explicit algorithm executed by agents. Toy models can treat the optimization literally.

5) **Quantum sector derivations.** Provide either (i) a derivation of the Born-rule trace form from finite distinguishability + invariance constraints, or (ii) a principled argument for why the projection map yields density operators and POVM statistics as the unique operational description at finite resolution.

6) **Standard Model embedding (optional milestone).** Make the internal algebra program concrete by specifying chiral representations, showing anomaly cancellation, and connecting representation content to stable projection excitations.

7) **Speculative application modules.** Physical-AI, neuromorphic, and junction-based
computing interpretations are outside the core gravity argument. No zero-latency
communication or autonomous learning result is implemented. Any future application
study must be documented separately and obey ordinary no-signaling constraints.

This draft is intended as a stable conceptual reference for these developments: it emphasizes explicit definitions and parameter accounting, and treats open issues (including junction accessibility) as constrained modules rather than as ungrounded assumptions.

## 13. Worked examples (operational extraction of standard observables)

This section specifies proposed future tests for extracting familiar observables
(redshift, bending angles, time delay, expansion rate) from a complete discrete
geometry. Only the finite graph clock-sign diagnostic is currently implemented.

- they do not introduce new postulates;
- they use only objects already defined in Sections 4–8 ($\{E_i\}$, $I_{ij}$, $d_G$, embedding $x_i$, Vietoris-Rips/Delaunay simplicial complex, discrete clock $\Phi_i$); and
- they define explicit comparison targets (Regge-consistent lensing, simplicial volumes).

Throughout, the guiding rule is: **We do not smooth to a continuum.** We compute observables by shooting geodesics through the discrete simplicial complex.

### 13.0 Slicing, gauge, and the Shift Vector via Optimal Transport
To extract dynamic observables (redshift, expansion, gravitational waves) from a sequence of reconstructed discrete geometries, we must rigorously define the "identity" of a point across time. We do not use Procrustes alignment, as there is no ambient Euclidean background to align *in*.

The proposed construction uses **intrinsic Gromov-Wasserstein Optimal Transport** to
map successive phase slices. This transport and the required tangent geometry are not
implemented.

#### 13.0.1 The Shift Vector from Intrinsic Optimal Transport
To rigorously extract the ADM Shift Vector $N^a$ without "Euclidean smuggling" (subtracting coordinates from distinct spaces), we use **discrete differential geometry** to map the transport plan directly into the local tangent bundle.

1.  **Transport Plan:** Compute the Gromov-Wasserstein optimal transport plan $\gamma^*$ between the discrete metric measure spaces $\mathcal{X}_s$ and $\mathcal{X}_{s+\Delta s}$.
2.  **Tangent Space Projection:** For each site $i \in \mathcal{X}_s$, the plan defines a target distribution over $\{j\} \in \mathcal{X}_{s+\Delta s}$. We map each target $j$ into the local tangent space $T_i \mathcal{X}_s$ (defined by the reconstructed metric $h_{ab}$) via the **Discrete Logarithmic Map**:
    $$
    v_{ij}^a \approx \text{Log}_{x_i}(x_j)
    $$
    This vector is constructed relationally: it is the unique vector $v^a \in T_{x_i}$ such that flowing along the geodesic generated by $v^a$ matches the distances $d(i, k)$ to the mapped distances $d(j, k')$ in the target slice, minimizing local distortion.
3.  **The Intrinsic Shift:** The shift vector $N^a$ is the expectation value of these tangent vectors under the transport plan:
    $$
    N^a(x_i) \Delta s := \sum_j \frac{\gamma_{ij}^*}{\mu_i} v_{ij}^a
    $$
    If well defined and stable, this would provide a candidate intrinsic geometric
    flow for tracking evolving correlation structure. Diffeomorphism covariance has
    not been demonstrated.

#### 13.0.2 Discrete Weak-field diagnostic potentials
We define the diagnostic potentials directly on the nodes of the graph.

**Background normalization.**  
Choose a reference region $B$ (far from mass) and define $\Phi_B = \langle \Phi_i \rangle_B$.

**Time potential** $\Phi_t$ at node $i$:
$$
\Phi_t(i) := \Phi_i - \Phi_B
$$
In the current implementation, $\Phi_i$ is a graph-potential diagnostic. A manually
chosen negative source yields a negative well under the stated operator and boundary
conditions. Identifying that source with mass and $\Phi$ with a gravitational
potential requires the source, continuum, and closure tests described above.

The discrete gravitational redshift between an emitter at node $A$ and an observer at node $B$ is then:
$$
1 + z = \exp(\Phi_B - \Phi_A)
$$
This ratio follows exactly from the chosen clock mapping, not from an independently
measured clock. For $\Phi_A < \Phi_B$, it gives $z>0$ consistently with the intended
sign convention. T4 checks this algebra and the finite-graph symmetry; it does not
establish quantitatively correct gravitational redshift.

**Space potential** $\Phi_s$ from discrete conformal factors:
Define the **relational background metric** $h_B := \langle h_{ab}(j) \rangle_{j \in B}$ as the spatial average of the reconstructed metric over the same far-field reference region $B$ used for $\Phi_B$. This is a purely relational quantity derived from the correlation graph — no ambient Euclidean metric is assumed.

For each node $i$, compute the local volume distortion relative to this background:
$$
a_s(i) := \left( \frac{\det(h_{ab}(i))}{\det(h_B)} \right)^{1/6}
$$
Then $\Phi_s(i) := c^2 (1 - a_s(i))$.

**GR closure check:** In the weak field, we require $\Phi_t(i) \approx \Phi_s(i)$ across the graph.

---

### 13.1 Proposed test A — isolated spherical mass (weak-field, static)
**Purpose.** Define a future controlled test of time dilation, lensing, and delay after
a physical source and intrinsic discrete geometry are available.

#### 13.1.1 Setup
- Supply a source region $R_M$ using a candidate that has independently passed linear
  response, conservation, localization, and matching tests. Raw relative entropy does
  not meet that requirement in the tested regime.
- Solve the graph constraint and compare clock and spatial observables against a
  controlled weak-field reference under refinement. A negative diagnostic well alone
  is insufficient.

#### 13.1.2 Discrete Geodesic Shooting
We do not integrate differential equations. We compute **shortest paths on the weighted graph/simplicial complex**.

1.  **Null Geodesics:** A light ray is a path of graph edges $\{e_1, e_2, \dots\}$ that minimizes the **optical length**:
    $$
    L_{opt} = \sum_{edge=(i,j)} \frac{\ell_{ij}}{\beta_{avg}(i,j)}
    $$
    where $\ell_{ij}$ is the proper length from the reconstructed $h_{ab}$, and $\beta_{avg}$ is the clock rate. The factor $1/\beta$ accounts for the effective refractive index $n \approx 1 - 2\Phi/c^2$ of the gravitational field.

2.  **Redshift convention:** For an emitter at node $A$ and observer at node $B$:
    $$
    1+z = \frac{\nu_{\mathrm{emit}}}{\nu_{\mathrm{obs}}}
        = \frac{\beta(B)}{\beta(A)}
        = \frac{e^{\Phi_B}}{e^{\Phi_A}}
    $$
    This follows from the chosen clock mapping; it is not an independently derived
    gravitational observable.

3.  **Lensing Angle:** Shoot two geodesics: one through the potential well (impact parameter $b$), one far away (reference).
    Measure the angular deviation of the final velocity vectors in the asymptotic region using the **discrete parallel transport** defined by the simplicial connection.
    *Check:* Does the deflection vector $\vec{\alpha}$ scale as $4GM/b$?

4.  **Shapiro Delay:** Compare the arrival time $t_{arrival} = \int dS_{act}$ of the ray passing through the potential vs. the reference ray.
    $$
    \Delta \tau \approx \sum_{edges} \ell_{ij} (1 - e^{\Phi_{avg}})
    $$
    (Note: $\Phi < 0$ implies $\beta < 1$, so "light slows down").

---

### 13.2 Worked example B — two-body lensing (binary mass)
**Purpose.** Show discrete superposition.

1.  **Setup:** Two high-density regions.
2.  **Potentials:** The Laplacian is linear. If $\delta\rho \approx \delta\rho_1 + \delta\rho_2$, then $\Phi \approx \Phi_1 + \Phi_2$.
3.  **Ray Tracing:** The optical path minimization naturally probes the sum of the potentials.
    We verify that the discrete deflection field $\vec{\alpha}(\vec{\theta})$ exhibits the shear and magnification of a binary lens without any continuum smoothing.

---

### 13.3 Worked example C — homogeneous cosmology patch (FLRW-like)
**Purpose.** Extract the scale factor $a(t)$ from the **discrete volume** of the simplicial complex.

#### 13.3.1 Discrete Scale Factor
For a phase slice at $t_k$:
1.  Compute the Vietoris-Rips complex $\mathcal{T}_k$ from the correlation metric space (or a Delaunay proxy on the MDS-embedded points $\{x_i(t_k)\}$).
2.  Sum the volumes of all tetrahedra $T \in \mathcal{T}_k$ in a large homogeneous patch $P$:
    $$
    V_P(t_k) := \sum_{T \in P} \text{Vol}(T)
    $$
    where $\text{Vol}(T)$ is calculated using the Cayley-Menger determinant with the edge lengths $l_{ij}$ derived from the metric $h_{ab}$.
3.  **Scale Factor:**
    $$
    a(t_k) := \left( \frac{V_P(t_k)}{V_P(t_0)} \right)^{1/3}
    $$
#### 13.3.2 Hubble Rate
Using the Laplacian-averaged proper time step $\Delta \bar{\tau}_k$ between slices:
$$
H(t_k) \approx \frac{1}{a(t_k)} \frac{a(t_{k+1}) - a(t_k)}{\Delta \bar{\tau}_k}
$$
#### 13.3.3 Failure Modes (Falsification)
- **Volume Collapse:** If the Vietoris-Rips complex degenerates (slivers, zero-volume tetrahedra) despite a growing $V_P$, the geometry is not manifold-like.
- **Anisotropy:** If $a(t)$ differs significantly when measured along different graph axes (using discrete direction-dependent correlators), the emergent space is not FLRW.


---

### 13.4 How these worked examples connect to falsification
These examples define *what to compute* from a candidate instantiation $(\mathcal{A}, \omega)$ and the specified selection rule $E$. The framework is disfavored if, under its own constraints (symmetry commutation, retention, area capacity, and stability), it cannot realize:

- a weak-field isolated-mass regime reproducing redshift and lensing;
- multi-lens weak-field behavior with approximate superposition and shear;
- a homogeneous expanding patch with robust scale-factor diagnostics.

In other words: these examples translate “emergent geometry” from a slogan into a set of explicit computational tests.
