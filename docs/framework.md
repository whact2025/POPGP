# A Phase-Ordered Pre-Geometric Projection Framework
*Conceptual framework draft (v0.10, AI-readable)*  
Author: [Your Name]  
Date: [YYYY-MM-DD]

## Abstract
We propose an atemporal, pre-geometric substrate described purely by relational/algebraic structure. A compact internal SU(2)-like symmetry and a distinguished phase/action ordering generator are taken as primitive. A unique, necessary physical projection map yields an emergent manifold-like spatial structure (with effective dimension `D*≈3` in our regime), objective time-order (as a metric on phase order), and a finite distinguishability bound expressed in terms of a boundary cut-capacity functional (expected to reproduce an area-law in low-distortion manifold regimes). Quantum discreteness is treated as emergent from stable representation content under projection constraints, while gravitational geometry is defined as the output of the projection’s correlation-based embedding and clock-rate reconstruction. In empirically accessed regimes the framework is required to recover standard General Relativity and quantum field theoretic predictions. Possible additional operational access to nonlocal substrate correlations (including any potential signaling via entanglement “junctions”) is formulated as a constrained open module rather than assumed a priori.

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
- **Substrate:** an atemporal, pre-geometric relational/algebraic structure `(A, ω)` with a compact internal symmetry and a preferred **phase/action ordering flow** `σ_s` (order parameter `s`, not time).
- **Projection:** a **unique and necessary** physical map `Π` from substrate to effective descriptions. `Π` is defined operationally by (i) a stability-selected coarse-graining into finite-capacity “cells”, (ii) correlation-defined locality, (iii) an embedding procedure into 3D, and (iv) a phase-order-to-clock-time mapping.
- **Finite distinguishability:** any finite projected region has bounded distinguishable information, scaling with **boundary area** (Planck-area-like scale `ℓ_*^2`), interpreted as a **relational encoding boundary** rather than a geometric boundary in the substrate.
- **Emergence:** discrete quantum “types” arise from stable representation content under projection constraints; effective geometry arises from correlation structure and the clock mapping, not from substrate curvature.

### What this framework does *not* claim (yet)
- It does **not** assume a complete derivation of the Standard Model spectrum, coupling constants, or cosmological parameters in this draft.
- It does **not** assert superluminal signaling. “Junction access” is defined as a constrained open module with parameters required to match existing no-signaling constraints in all empirically accessed regimes.
- It does **not** claim a closed-form fundamental “constitutive law” `g = F(ρ, …)`; the effective metric is defined by the explicit projection construction (graph distances → embedding → reconstructed `h_ab` and `dτ` mapping).

### Scientific status and evaluation
- The framework is **empirically anchored** by matching requirements: in accessible regimes, the induced effective description must reproduce (to stated tolerance) standard GR tests and standard quantum statistics.
- The framework becomes **strictly falsifiable** when it asserts: (i) universal area-law capacity bounds outside their known domain, (ii) saturation/no-singularity behavior in regimes where GR predicts divergence, or (iii) any nonzero operational “junction access” parameter in regimes already constrained by experiment.
- Absent additional predictive commitments, the framework is evaluated by: (a) internal consistency (no hidden time/geometry in substrate), (b) uniqueness/minimality of `Π` under stated selection principles, and (c) whether GR/QFT emerge as stable effective closures with a small, auditable parameter budget.


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
**S1 (Atemporal substrate).** There exists a substrate `S` that is not spacetime and has no intrinsic time parameter.  
**S2 (Pre-geometric substrate).** The substrate carries relational/algebraic structure but no spatial metric, curvature, or geometric topology in the usual sense.  
**S3 (Compact internal symmetry).** The substrate admits a compact internal SU(2)-like symmetry acting on its relational degrees of freedom.  
**S4 (Phase/action ordering generator).** A distinguished generator induces an ordering structure (phase/action order) that is not time but is used by projection to construct time-order.

### 3.2 Projection postulates
**P1 (Unique physical projection).** There exists a unique, necessary projection map `Π` from substrate structure to projection-level effective physics.  
**P2 (Emergent spatial structure; dimension as output).** `Π` yields an emergent metric space with a manifold-like realization of some effective dimension `D*`; in the empirically accessed regime of our universe, `D* ≈ 3` (treated as an output/matching condition rather than an assumed input).  
**P3 (Emergent time-order).** `Π` yields an objective time-order in projection as a metric on phase/action order; non-conscious systems inherit this time-order.  
**P4 (Finite distinguishability; boundary-capacity scaling).** For any finite projected region `R`, physically distinguishable information is bounded by a boundary *cut-capacity* functional (defined on the correlation graph). In manifold-like regimes this scaling is expected to reproduce an area-law.  
**P5 (Bounded intensities).** Projection-level densities and curvatures saturate rather than diverge; classical singularities indicate breakdown of the effective description.

### 3.3 Emergence postulates
**E1 (Quantum discreteness from representations).** Discrete quantum types arise from stable representation content of the internal symmetry under projection constraints, not from fundamental substrate discreteness.  
**E2 (Geometry as projection output).** The effective metric `g_{μν}` is constructed by the projection stages `Π_loc`, `Π_geom`, and `Π_time` (Section 4.4): correlation-derived distances define an emergent locality graph; an embedding procedure reconstructs the spatial metric `h_{ab}`; and phase/action ordering sets the clock mapping via `dτ = β0·exp(Φ) dS_act`, where `Φ` is a smoothed/harmonically extended potential on the correlation graph sourced by an entropy-contrast field `δρ`. Encoding-density proxies are derived scalar summaries of local information/capacity structure and are not primitive geometric variables. Any apparent constitutive form `g ≈ F(·)` is understood as a phenomenological approximation to the reconstructed metric output by `Π`.  
**E4 (Lorentz Protection via Custodial Symmetry).** The strict algebraic commutation of the projection maps with the internal SU(2) symmetry ($E_i \circ \alpha_g = \alpha_g \circ E_i$) enforces a "Custodial Symmetry" on the emergent effective field theory. This symmetry algebraically prohibits the generation of lower-dimensional Lorentz-violating operators (dimension-4 and dimension-5) that typically arise from lattice discreteness, ensuring that Lorentz invariance is preserved as an emergent symmetry to high precision (bounds < $10^{-14}$).
**J (module) (Junction accessibility).** In all known regimes, operations available to agents obey no-signaling in local marginals. Additional junction access is treated as a constrained open module.

## 4. Mathematical primitives

### 4.1 Substrate representation (analog-friendly)
Represent the substrate as a pair:
```text
S = (A, ω)
```
where `A` is an algebra of relational observables and `ω` is a state on `A`. No manifold, metric, or time parameter is assumed at this level.

**Analog-friendly choice.** To support continuous internal symmetry/phase structure while preserving *finite physical realizability* (via projection-level finite distinguishability), take `A` to be a separable operator algebra that is well-approximated by finite matrix algebras at any finite resolution. A convenient template is:
```text
A = A_rel ⊗ A_F
```
- `A_rel` is the “relational bulk” algebra (pre-geometric), chosen so that it can be approximated by an increasing sequence of finite-dimensional matrix algebras:
```text
A_rel ≈ closure( ⋃_n M_{k_n}(C) )
```
This supports the **analog** intuition (continuous symmetry/phase) while remaining compatible with the principle that *only a finite number of states are physically distinguishable in any finite projected region* (Section 6).

- `A_F` is an optional finite “internal” algebra used to encode gauge/representation structure in a purely algebraic way (Section 9). A minimal candidate that naturally contains `U(1)`, `SU(2)`, and `SU(3)`-type unitary structure is:
```text
A_F = C ⊕ H ⊕ M_3(C)
```
with `H` the quaternions.

**Remark (no infinities physically instantiated).** Continuous groups and infinite-dimensional algebras are treated as *idealized descriptions*: the framework’s finiteness claim is that no *projection-level observable* diverges and that any finite region has finite operational capacity (area-law distinguishability), not that the descriptive mathematics cannot employ limits.

### 4.2 Internal symmetry action
Let `α` be an action of the internal symmetry group on `A` as *-automorphisms:
```text
α: SU(2) → Aut(A)
```
Symmetry invariants are quantities unchanged under `α_U` for all `U ∈ SU(2)`.

### 4.3 Phase/action ordering generator
The substrate has no time, but it provides an **ordering structure** that projection uses to construct time-order.

Two equivalent presentations are allowed in this draft:

**(i) Generator form (working placeholder).** Let `H ∈ A` be a distinguished self-adjoint element (`H = H†`) defining a one-parameter family of inner automorphisms:
```text
A ↦ e^{isH} A e^{-isH}
```
The parameter `s` is a phase/action **order parameter**, not substrate time.

**(ii) Canonical-flow form (parameter-reducing option).** Impose that the pair `(A, ω)` determines a canonical one-parameter automorphism flow `σ_s` on `A`. Projection-time is then constructed from monotone “distance” along this flow. In this view, `H` is not chosen freely; it is the generator associated with the canonical flow in a suitable representation.

The framework’s finiteness claim applies to *projection-level observables* and distinguishability (Section 6), not to the mathematical use of continuous `s`.

### 4.4 Projection map (explicit construction template)

In this framework, the projection map `Π` is not treated as an arbitrary “fit function.”  
Instead it is defined as a *constrained construction* that turns substrate relational structure into:

- an emergent 3D locality structure,
- an effective spatial metric,
- an objective time-order (from phase/action order),
- an encoding-density field and finite distinguishability bounds,
- effective subsystem states used for quantum predictions.

Mathematically, it is useful to write `Π` as a composition of four stages:
```text
Π = Π_time ∘ Π_geom ∘ Π_loc ∘ Π_res
```
where:

- `Π_res` implements finite distinguishability (resolution-limited coarse-graining),
- `Π_loc` extracts emergent locality from correlation structure,
- `Π_geom` builds an effective 3D geometry/metric from locality data,
- `Π_time` maps phase/action order into projection time.

The definitions below form a *template*. Each stage is explicit enough to be evaluated, but still leaves room for later refinement.

#### 4.4.1 Stage 0: represent the substrate state (GNS representation)
Given `(A, ω)`, choose the GNS triple `(π_ω, H_ω, |Ω⟩)` such that:
```text
ω(a) = ⟨Ω | π_ω(a) | Ω⟩   for all a ∈ A.
```
All projection-level quantities are ultimately defined from this representation, not from any prior notion of space.

#### 4.4.2 `Π_res`: resolution-limited coarse-graining via split-property funnels
A projection must provide a notion of subsystems/regions, but subsystems are not substrate primitives. Furthermore, strict local algebras in Quantum Field Theory are typically **Type III factors** (entanglement is ubiquitous), meaning they do not admit pure states or density matrices, and cannot be factorized into tensor products `A_R ⊗ A_R^c`.

To resolve this **Type III incompatibility** without abandoning the continuum, we define emergent "cells" explicitly as **split-property funnels** (Type I approximants across algebraic buffer zones).

**The Funnel Definition.**
Instead of a single algebra `A_i`, the projection selects a **funnel structure** for each emergent site `i`:
```text
A_{i, inner} ⊂ N_i ⊂ A_{i, buffer} ⊂ A
```
where `N_i` is a **Type I factor** (isomorphic to `B(H)` or a finite matrix algebra `M_d(C)` in toy models) that acts as a local "distinguishable semantic shell" between the inner core and the buffer zone.

The effective local algebra is identified with the Type I factor `N_i`. This guarantees the existence of well-defined local density matrices, entropies, and particle-like excitations, even if the underlying substrate algebra is Type III.

**Coarse-Graining.**
The coarse-graining maps `E_i` are conditional expectations onto these Type I factors:
```text
{A_i ⊂ π_ω(A)'' }_{i ∈ V},   with   A_i ≅ M_{d_i}(C)   (Type I approximants),
```
together with conditional expectations (coarse-graining maps):
```text
E_i : π_ω(A)'' → A_i
```
that are completely positive, unital, and idempotent (`E_i∘E_i = E_i`).

For any finite set of cells `R ⊂ V`, define the induced effective algebra and reduced state:
```text
A_R  ≈  ⊗_{i∈R} A_i,        (exact in toy models; approximate “split” factorization in the continuum/QFT limit)
E_R  := ⊗_{i∈R} E_i,
ω_R  := ω ∘ E_R .
```

### Finite distinguishability constraint (non-geometric form)
To avoid circularity, the capacity bound is imposed **before** any geometric embedding.

Let `Cap(∂R)` denote a **cut-capacity** functional on the cell net:
```text
Cap(∂R) := Σ_{i∈R, j∉R} κ(I_{ij}),
```
where `I_{ij}` is the mutual information between cells `i` and `j`, and `κ` is a fixed monotone map to dimensionless edge capacities.

**Finite distinguishability constraint (Araki-relative form).**
For Type III compatibility, we eschew the naive von Neumann entropy. The bound is imposed on the **Araki relative entropy** of the region's state `\omega|_R` against the canonical **KMS vacuum baseline** `\omega^{vac}|_R` (defined by the global modular flow):
```text
S_{Araki}( \omega|_R || \omega^{vac}|_R ) ≤ η · Cap(∂R)   for all finite regions R ⊂ V.
```

This is where “Planck-scale resolution” enters operationally: the projection cannot support arbitrarily many independent degrees of freedom in finite regions relative to the vacuum. High-resolution structure is bounded by a cut-capacity rather than a geometric area.

A geometric area scaling `Cap(∂R) ∝ A_geo(∂R)/ℓ_*²` is treated as an emergent behavior in the low-stress embedding regime (§4.4.4), not as an input constraint.

#### 4.4.2a Selection Principle `E`: symmetry-commuting, phase-flow stable cell net

The choice of the effective cell algebras `{A_i}` and coarse-graining maps `{E_i}` is fixed by a *stability principle* tied to the phase/action ordering flow `σ_s` (§4.3), rather than chosen ad hoc. Intuitively: the correct decomposition is the one for which phase/action ordering does not continually “slosh” information across emergent cell boundaries.

Let `α_g` denote the internal SU(2)-like symmetry action on `A` (or on `π_ω(A)''` after the GNS embedding). Let `σ_s` denote the phase/action ordering flow.

**Admissible coarse-grainings.** Define `𝔈_adm` as the set of families `{E_i}` (and their associated cell algebras `{A_i}`) satisfying all of:

1) **Finite-capacity cells.** Each cell algebra is finite-dimensional:
```text
A_i ≅ M_{d_i}(C),
```
with either a fixed common dimension `d_i = d` for all cells (simplest), or an explicit local bound `d_i ≤ d_max` (bounded capacity). In all cases the finite distinguishability constraint in §4.4.2 is required to hold for all finite regions `R`.

2) **Symmetry commutation (internal isotropy constraint).** Each coarse-graining map commutes with the SU(2)-like symmetry:
```text
E_i ∘ α_g = α_g ∘ E_i    for all g ∈ SU(2) and all i.
```
Equivalently (often easier to verify): the subalgebra `A_i` is invariant under `α_g`, and `E_i` is SU(2)-equivariant.

**The "Custodial Symmetry" Mechanism (Lorentz Protection).**
This constraint is the physical reason why the discrete graph does not break Lorentz symmetry at macroscopic scales. The internal SU(2) acts as a **Custodial Symmetry**. By forcing the local algebra of each funnel to transform as a singlet (or specific stable representation) under the internal group, any "lattice artifact" operator that would pick out a preferred direction in the emergent tangent space is algebraically forbidden.
Specifically, dimension-4 and dimension-5 Lorentz-violating operators (e.g., $k^\mu k^\nu \Delta u_\mu u_\nu$) are not singlets under the induced rotation group of the tangent space and are thus projected to zero by the symmetry-commuting map $E_i$. This ensures that the emergent IR Effective Field Theory is locally isotropic to a precision protected by the full strength of the internal symmetry, satisfying the $10^{-14}$ experimental bounds.

3) **Information retention (anti-triviality constraint).** The coarse-graining must not erase essentially all substrate information. Impose a global bound on relative-entropy loss:
```text
D( ω || ω ∘ E ) ≤ ε,
```
where `E = ⊗_i E_i` and `D(·||·)` is the quantum relative entropy. (Other retention constraints are possible, but this one is explicit and auditable.)

**Primary stability functional (phase-flow leakage).** For each admissible `{E_i}`, define the phase-flow leakage:
```text
L_leak(E) := ∫ ds w(s) · Σ_i  || E_i ∘ σ_s  -  σ_s ∘ E_i ||^2,
```
where `w(s) ≥ 0` is a weight over a chosen phase-order interval, and `||·||` is a norm on superoperators (channels). Suitable choices include:
- the diamond norm `||·||_⋄` (operationally strongest),
- or a Hilbert–Schmidt / Frobenius norm in a fixed representation (computationally simpler).

`L_leak` measures how strongly the phase/action ordering flow mixes degrees of freedom across the selected cell boundaries. Exact co-motion corresponds to `L_leak = 0`.

**Optional tie-breaker (local information drift).** Among decompositions that minimize leakage, prefer those for which local cell information varies smoothly (minimally) along phase-order. Define `ω_s := ω ∘ σ_s`, and let `ρ_i(s)` be the density operator corresponding to the reduced state `ω_s ∘ E_i` on the cell Hilbert space `H_i`. For a small step `δ`, define:
```text
L_drift(E) := ∫ ds w(s) · Σ_i  (1/δ^2) · D( ρ_i(s+δ) || ρ_i(s) ).
```
This penalizes rapid change of local reduced information under phase-order advance. It is optional; it is intended as a *tie-breaker* and not the primary selector (to avoid freezing dynamics by choosing overly coarse cells).

**Lexicographic (two-stage) selection rule.** Choose the cell net by:

1) (Primary) minimize leakage over admissible decompositions:
```text
E* ∈ argmin_{E ∈ 𝔈_adm}  L_leak(E).
```

2) (Secondary) among leakage-minimizers, minimize drift:
```text
E* ∈ argmin_{E ∈ argmin L_leak}  L_drift(E).
```

This implements a stability principle in the precise sense requested: emergent subsystems are selected to be as invariant as possible under the phase/action ordering flow (primary), while also exhibiting minimal local information churn along phase-order (optional tie-breaker), all while respecting SU(2)-equivariance and finite distinguishability.











**Variational interpretation (Thermodynamic, not Computational).** The selection rule is a *definition* of the projection (analogous in spirit to least-action principles), not an assumption that any agent or physical subsystem explicitly solves an NP-hard optimization.
Physically, we treat the minimization of $L_{leak}$ as a **thermodynamic relaxation process**. The updates to the coarse-graining maps $E_i$ are restricted to **Lieb-Robinson bounded local unitary operations**. This ensures that the emergent spatial graph evolves causally: no instantaneous global optimization is performed. The "optimal" net is the fixed point of this local relaxation flow, not the output of a magic global solver.

**Implementation notes (default choices for toy models).**
- Weight `w(s)`: choose a finite phase-order window of width `Δs` and set `w(s)=1/Δs` on that window (and `0` outside), or use a Gaussian centered at `s0`. `Δs` should be small enough that the selected cell net is approximately stable across the window.
- Norm `||·||`: use a Hilbert–Schmidt / Frobenius norm in a fixed representation as a computational proxy in toy models; treat the diamond norm `||·||_⋄` as the operationally strongest target definition.
- Retention budget `ε`: define the capacity of a reference region `R0` by `S_cap(R0):=η·Cap(∂R0)` and choose `ε = κ·S_cap(R0)` with `κ≪1` (or impose per-cell bounds). This prevents the trivial “erase everything” minimizer.
- Drift step `δ`: choose `δ ≪ Δs` (e.g., `δ = Δs/N` with large `N`) so that `L_drift` approximates a local drift rate.
- Optimization style: prefer the stated lexicographic minimization over weighted sums to avoid introducing arbitrary weight parameters.

#### 4.4.3 `Π_loc`: locality from correlations (graph metric from mutual information)
Given the effective cell net, define reduced density operators `ρ_R` from `ω_R` in the standard way
(using the finite-dimensional representation of `A_R` on `H_R` in toy models, or split-property approximants
at finite resolution in the continuum/QFT limit).

Define the mutual information between cells `i` and `j`:
```text
I_{ij} = S(ρ_i) + S(ρ_j) - S(ρ_{ij}).
```

Interpret `I_{ij}` as a **relational coupling kernel**: higher mutual information indicates tighter relational dependence.

### Distance kernel
Define an emergent pairwise distance by a fixed monotone decreasing map `f`:
```text
d_{ij} = ℓ_* · f(I_{ij} / I_0),
```
where `I_0` is a reference mutual-information scale (e.g., typical strong-coupling value), and `f` satisfies:
- `f(u) ≥ 0`,
- `f'(u) < 0`,
- `f(u→0) → +∞` (very weak coupling → far apart).

A simple low-parameter choice is:
```text
f(u) = max{0, -log u }.
```

### Weighted graph (avoid disconnection)
Define a **continuous** edge weight kernel (no hard threshold):
```text
w_{ij} := κ(I_{ij})   with κ monotone increasing and κ(0)=0.
```
In principle this defines a fully connected weighted graph. For computational sparsification in toy models, use
a connectivity-preserving rule (e.g., k-nearest neighbors by smallest `d_{ij}` **plus** a minimum-spanning-tree backbone)
rather than a hard `I_{ij}` cutoff, to prevent “islands” created by rapid decay.

### Graph geodesic distance
With edge lengths `d_{ij}` (or equivalently weights `w_{ij}`), define graph-geodesic distance:
```text
d_G(i,j) = inf_{paths i→j} Σ_{(a,b) in path} d_{ab}.
```

This defines a metric space `(V, d_G)` without assuming any background geometry.

#### 4.4.4 `Π_geom`: emergent geometry via complexity-stress minimization
The spatial manifold is not a background container. It is a compression scheme for the correlation graph.

### 1. Dimension Selection (Derived, not assumed)
We do not postulate `D=3`. Instead, the emergent dimension `D*` is the integer that minimizes a **Complexity-Stress Functional** `F(D)` that balances embedding fidelity against topological complexity:
```text
D* := argmin_{D \in \mathbb{N}} [ Stress(D) + \lambda_{dim} \cdot |D - D_S|^2 ]
```
where:
- `Stress(D)` is the standard MDS stress (distortion of graph distances `d_G` vs Euclidean distances in `R^D`).
- `D_S` is the **Spectral Dimension** of the graph (derived from the heat kernel trace `Tr(e^{-t \Delta}) ~ t^{-D_S/2}`), representing the graph's intrinsic diffusive topology.
- `\lambda_{dim}` is a penalty weight ("topological inertia") that prevents "dimensional jitter" under phase flow.

In the physical regime of our universe, we require `D* = 3` as a stable minimum. If `D*` diverges or collapses to 1, the framework predicts a non-geometric phase (see Falsifiers F1).

### 2. Relational Embedding
Given `D*`, the coordinates `{x_i}` are determined by the stress-minimizing configuration in `R^{D*}`. This embedding is unique only up to isometries; physical observables must be relational (diffeomorphism invariant).

### 3. Local Metric Reconstruction with Relational SPD Regularization
The emergent metric `h_{ab}(x)` is recovered by fitting the local embedding to the graph distances. To ensure the metric is strictly **Symmetric Positive-Definite (SPD)** without assuming a Euclidean background `\delta_{ab}`, we define the regularizer relationally.

Solve for `h_{ab}(x_i)`:
```text
min_{h \succ 0} \sum_{j \in N(i)} ( d_G(i,j)^2 - \Delta x^T h \Delta x )^2  +  \lambda_{spd} || h - h_{cov}^{-1} ||_F^2
```
where the reference matrix `h_{cov}^{-1}` is the **inverse local covariance** of the neighbor displacement vectors `{\Delta x_j}` in the tangent space, representing the "natural" geometry of the point cloud distribution. This strictly prevents "smuggling in" a flat background; if the point cloud is highly skewed, the regularizer respects that anisotropy.

### 4. Definition of Geometric Singularities (Horizons)
Singularities are not infinities in curvature (which saturate, see P5), but **algebraic fractures** in the reconstruction map.
A **Geometric Horizon/Singularity** is defined as a region where the condition number `\kappa` of the design matrix `M` for the metric fit diverges:
```text
\kappa(M) \to \infty
```
This indicates an **un-smoothable graph fracture** where the correlation topology cannot be mapped to a manifold of dimension `D*` without tearing.


#### 4.4.5 `Π_time`: unified emergent time via non-local graph potential
Time is not a local scalar. It is a **global potential** determined by the entire network's relational structure.

### 1. The Clock-Rate Potential Equation
We replace all local clock mappings `\beta(\rho)` with a **non-local Clock-Rate Potential** `\Phi` defined on the correlation graph. `\Phi_i` represents the local "gravitational depth" (redshift factor) at site `i`.

`\Phi` is the unique solution to the **Weighted Graph-Laplacian** equation:
```text
(\Delta_w + \mu^2 I) \Phi = \delta\rho
```
where:
- `\Delta_w` is the graph Laplacian on the mutual-information weighted graph: `(\Delta_w \Phi)_i := \sum_j w_{ij} (\Phi_i - \Phi_j)`.
- `\delta\rho` is the source term (defined below).
- `\mu` is a screening mass. We set **`\mu = 0`** (massless) to enforce the standard GR `1/r` long-range behavior.

### 2. The Source Term: Temporal-Filtered Araki Contrast
The source of the potential is not naive entropy, but the **informational contrast** between the local state and the global vacuum.
```text
\delta\rho_i := \frac{1}{s_0} \left[ \bar{S}_{Araki}(\omega_{i} || \omega_{i}^{vac}) \right]
```
where `\bar{S}_{Araki}` is a **temporal average** (over a short phase-order window) of the Araki relative entropy between the local reduced state `\omega_i` and the reference vacuum `\omega_i^{vac}`. This averaging ensures stability and prevents high-frequency phase noise from sourcing gravity.

### 3. The Baseline Vacuum (Solving the Zero-Mode)
For `\mu = 0`, the graph Laplacian has a kernel (zero mode) corresponding to constant shifts. To fix this mode without introducing instantaneous non-local spatial averaging (which would violate causality), we define the baseline vacuum `\omega^{vac}` strictly algebraically:

**Definition:** `\omega^{vac}` is the **KMS (Kubo-Martin-Schwinger) thermal state** with respect to the **global modular automorphism group** of the substrate algebra `\mathcal{A}`.
This provides a canonical, globally invariant reference state that is "always available" algebraically, removing the need for run-time spatial averaging.

### 4. Emergent Proper Time
The local proper time `\tau` along a phase trajectory is then constructed by integrating the **Laplacian-derived clock rate**:
```text
d\tau(i) = \beta_0 \cdot \exp(\Phi_i) \cdot dS_{act}
```
This unifies gravitational redshift, time dilation, and the "flow of time" into a single Laplacian-controlled mechanism.

### Minimal emergent spacetime line element (preferred slicing)
In the preferred foliation induced by phase/action order (a physically distinguished slicing, not assumed to be fundamental time),
define:
```text
ds² = -c² dτ² + h_ab(x) dx^a dx^b,
```
where `a,b = 1..D*` and `D*` is the emergent embedding dimension from §4.4.4 (empirically `D*≈3` in our regime).

**Lorentz covariance status (explicit).**  
A preferred slicing exists at the projection-construction level because `σ_s` defines a distinguished order.
Local Lorentz covariance is therefore treated as an **emergent IR symmetry**: in regimes `L ≫ ℓ_*`, predictions derived from
`g^{(L)}_{μν}` must agree with locally Lorentz-invariant effective field theory to within experimental bounds.
Any detectable Lorentz violation is constrained to be suppressed (e.g., by powers of `ℓ_*/L`) and is part of the falsification program (§11).

#### 4.4.6 Summary: the projection map as output object
With the above stages, the projection map can be summarized as:
```text
Π(A, ω, α) = ( M^{D*}, {x_i}, h_ab(x), τ(x), Φ(x), {ρ_R}, Cap(∂R), … )
```
with:
- `D*` and `{x_i}` from stress-minimizing embedding of the correlation metric space (dimension selected by distortion/complexity),
- `h_ab(x)` from SPD-constrained local metric reconstruction on the embedded point cloud,
- `Cap(∂R)` from the MI-weighted graph cut functional (primitive “boundary size” for capacity bounds),
- `δρ` (entropy-contrast) from local reduced states relative to a vacuum baseline,
- `Φ` from graph-Laplacian smoothing / harmonic extension `(Δ_w + μ² I)Φ = δρ`,
- `τ` from `dτ = β0·exp(Φ) dS_act` (phase/action order converted to clock time).

**Where freedom remains (auditable).**  
The remaining degrees of freedom are explicit and constrained:
- coarse-graining selection tolerances (`ε`, choice of channel norm, phase-window weight `w(s)`, drift step `δ`),
- MI→distance map `f` and MI→weight map `κ` (both fixed monotone families, not fitted per region),
- embedding criterion (`ε_embed` or `λ_dim`) and neighborhood rule `N(i)`,
- SPD regularization strength `λ_spd`,
- smoothing parameter `μ` (default `μ=0`; treated as an optional controlled IR-modification module).

These are the only places “parameter fitting” could enter; the framework requires universality (same choices everywhere) and
empirical matching constraints to keep them from becoming arbitrary.

### 4.5 Symbol glossary
| Symbol | Meaning |
|---|---|
| `S` | Substrate object |
| `A` | Substrate algebra of relational observables |
| `A_rel` | Relational “bulk” algebra (pre-geometric; approximable by finite matrices) |
| `A_F` | Finite internal algebra encoding gauge/representation structure (optional program) |
| `ω` | State on `A` |
| `α` | Internal symmetry action on `A` |
| `H` | Phase/action ordering generator (order parameter; not substrate time) |
| `Π` | Unique physical projection map |
| `M³` | Emergent 3D spatial structure |
| `g_{μν}` | Emergent spacetime metric in projection |
| `t` | Emergent time coordinate/order metric in projection |
| `ρ(x)` | Encoding density field in projection |
| `ℓ_*` | Minimal encoding resolution scale (Planck-like) |
| `η` | Dimensionless coefficient in the area-law capacity bound |
| `γ` | Junction-access gating parameter (open module) |
| `Δ` | Junction-access correction functional (open module) |


### 4.6 Substrate algebra and parameter budget (review-facing)

This section makes explicit (i) what is fixed by structural choice versus (ii) what remains to be calibrated to match empirical reality. The goal is to prevent “free-form projection fitting” by keeping the degrees of freedom auditable.

#### 4.6.1 Algebraic architecture (what the substrate “looks like”)
**Substrate data.** The substrate is specified by:
```text
S = (A, ω, α)
```
where:
- `A = A_rel ⊗ A_F` is a relational algebra with optional finite internal factor,
- `ω` is a state on `A`,
- `α: SU(2) → Aut(A)` is the compact internal symmetry action (Section 4.2).

**Key point.** `A_rel` is not “functions on a space.” It is a non-spatial relational algebra. Any appearance of locality/geometry arises only after projection.

#### 4.6.2 Where continuous structure enters (and why this does not force physical infinities)
- Continuous symmetry (SU(2)-like) and continuous phase/action order are treated as idealized descriptions.
- Physical finiteness is enforced at the projection level via the area-law bound and bounded intensities:
  - finite distinguishability in any finite region (Section 6),
  - saturation rather than divergence for densities/curvatures (Postulate P5).

In other words: *the model forbids unbounded physical observables, not the use of continuous mathematics.*

#### 4.6.3 Parameter budget (what must be fixed or fitted)

| Category | Examples in this framework | Status / how constrained |
|---|---|---|
| **Structural (discrete) choices** | Choice of algebra class for `A_rel`; choice of internal algebra `A_F`; compact internal symmetry group; uniqueness of projection postulate | Architecture decisions (not continuously fitted). Must be defended by conceptual minimality and consistency. |
| **Empirical scale-setting constants** | `ℓ_*` (minimal encoding patch); `η` (area-law coefficient); effective constants matching `(c, ħ, G, Λ)` | Expected to be measured unless derived from a deeper principle; treated like fundamental constants in early drafts. |
| **Projection response functions** | Mapping from encoding density to metric scaling; mapping from phase-order to clock time (e.g., monotone/saturating functions) | Primary risk of “parameter fitting.” Must be restricted to minimal families with clear constraints and matching requirements. |
| **Substrate state selection** | Choice/characterization of `ω` | Must be constrained by a selection principle (symmetry, equilibrium relative to canonical flow, minimal information, etc.) to avoid embedding arbitrary structure by hand. |
| **Open junction module** | `γ(regime)` and `Δ(…)` | Must satisfy `γ ≈ 0` in all tested regimes; left open otherwise, but constrained by normalization/positivity and by compatibility with known no-signaling bounds. |

#### 4.6.4 Anti-overfitting constraints on the projection map `Π`
To keep the framework falsifiable and reviewable, `Π` should be restricted by explicit principles such as:

1) **Universality:** the same mapping rules apply across all regions and epochs (no ad hoc patchwork).  
2) **Locality-from-correlation:** emergent adjacency is defined by relational/correlation structure in `(A, ω)`, not assumed.  
3) **Minimal functional freedom:** response functions are chosen from low-parameter families (monotone, saturating) with their parameters tied to known constants.  
4) **Matching constraints:** `Π` must reproduce GR in weak-field and tested strong-field regimes and reproduce standard quantum statistics in laboratory regimes.

These constraints convert “projection” from a free fitting function into a tightly parameterized construction with a transparent parameter budget.

#### 4.6.5 Optional parameter reduction via a canonical phase-order flow
To reduce arbitrariness in the choice of the phase/action ordering generator, one may impose:

- the phase-order flow is **canonical** given `(A, ω)` (i.e., derived from the substrate state rather than chosen as an additional free input).

This provides a principled origin for the preferred ordering used to construct projection-time while preserving substrate atemporality.

## 5. Projection outputs and emergent time
Time does not exist at the substrate level. Instead, projection constructs an objective time-order by mapping phase/action order to a temporal metric. Conceptually: substrate provides an ordering structure; projection interprets it as time.

### 5.1 Order-to-time mapping
A minimal requirement is a monotone mapping from phase order to time:
```text
t = f(Φ) with f' > 0
```
where `Φ` is a projection-accessible phase/order functional.

## 6. Finite distinguishability and Planck-scale resolution
The framework adopts an analog substrate intuition while enforcing bounded physical realizability in projection. Planck-scale resolution is treated as a limit on physically distinguishable information, not necessarily as a lattice of spacetime points.

### 6.1 Finite distinguishability via algebraic cut-capacity
To completely eliminate geometric circularity, we define the distinguishability bound using a strictly pre-geometric **algebraic cut-capacity** functional. This ensures that "area" is an output of the theory, not an input.

**1. Algebraic Cut-Capacity.**
Let `R ⊂ V` be a finite set of emergent funnels. Define the **cut-capacity** `Cap(∂R)` purely from the correlation graph weights:
```text
Cap(∂R) := Σ_{i∈R, j∉R} κ(I_{ij})
```
where `I_{ij}` is the mutual information between funnels `i` and `j`, and `κ` is the edge-weight kernel. This quantity measures the total "information flux" connecting `R` to its complement, with no reference to spatial geometry.

**2. The Araki Entropy Bound.**
For rigorous compatibility with Type III limits, we replace the naive von Neumann entropy with the **Araki relative entropy** evaluated on the buffer zones. Let `N_R = ⊗_{i∈R} N_i` be the composite Type I factor for region `R`.
The bound requires that the distinguishable information content of `R` (measured relative to the vacuum state `ω_vac` on the buffer) is limited by the cut-capacity:
```text
S_{Araki}( ω |_R || ω_vac |_R ) ≤ η · Cap(∂R)
```
where `η` is a dimensionless constant (`O(1)`).

**3. Theorem of Emergence (Not an Axiom).**
The geometric Area Law is not an axiom. Instead, it is asserted as an **emergent theorem**:
*   *Theorem (conjecture):* If the stability-selected correlation graph admits a low-distortion embedding into `R^3`, then `Cap(∂R)` will scale proportionally to the geometric surface area `A_{geo}(∂R)` defined in that embedding.
    ```text
    Cap(∂R) ∝ A_{geo}(∂R) / ℓ_*^2
    ```
This reverses the standard logic: we do not assume an Area Law to get geometry; we assume a Cut-Capacity Bound, and recover the Area Law only in regimes where a stable geometry emerges.

### 6.2 Finite distinguishability
A useful operational translation is a bound on the effective Hilbert-space dimension associated with region `R`:
```text
dim(H_R) ≲ exp(S(R))
```

## 7. Quantum statistics as projection-limited inference
Quantum probabilities are framed as a consequence of limited distinguishability: projection exposes only coarse-grained effective states for subsystems. Many substrate micro-configurations correspond to the same effective projection state.

### 7.1 Effective states
For a subsystem associated with region `R`, projection yields an effective density operator `ρ_R` acting on `H_R`. Measurement outcomes are computed by the Born-rule form in empirically accessed regimes:
```text
P(i) = Tr(ρ_R E_i)
```

### 7.2 Program toward a derivation
A derivation program (not completed in this draft) is to show that the trace-rule probability assignment is uniquely selected by a combination of:
1) additivity for exclusive outcomes,  
2) invariance under unitary transformations induced by internal symmetry actions, and  
3) finite distinguishability constraints that prevent access to deeper microstructure.

## 8. Emergent geometry and GR matching
The effective spacetime metric `g_{μν}` is treated as a projection output, not a substrate primitive. Geometry is determined by projection-accessible fields including encoding density `ρ(x)`.

### 8.1 Metric definition as projection output (replacing an abstract constitutive law)
In this framework, we do not postulate a fundamental constitutive law of the form $g_{\mu\nu} = F(\rho, ...)$. Instead, the effective spacetime metric is the **output of the projection construction** defined in Section 4.4.

The projection pipeline replaces the role of a constitutive law by constructively generating geometry from correlation.
1.  **Locality:** Defined by the stability-selected cell net and mutual information (Section 4.4.2–4.4.3).
2.  **Spatial Metric:** Defined by the relational embedding and local SPD reconstruction (Section 4.4.4).
3.  **Time & Lapse:** Defined by the Laplacian clock-rate potential $\Phi$ and phase-order integration (Section 4.4.5).

Any apparent "law" connecting density to curvature (like Einstein's equations) is an **effective closure condition** on this reconstructed field, not a primitive input. Section 8.2 defines the rigorous testing of this closure via Discrete Regge Calculus.

### 8.2 GR matching as a discrete closure condition
This framework does not postulate Einstein’s equation as a substrate law. Instead, **GR is imposed as a tested-regime closure condition** on the *reconstructed* effective metric field.

To avoid ambiguity regarding smoothing commutators, we establish the **Discrete Route** as the primary definition of the matching condition.

#### 8.2.1 Primary Definition: Discrete Regge Closure
The comparison is performed natively on the discrete pre-geometric structure, before any continuum limit is taken.

1.  **Discrete Geometry:** Construct the **Delaunay triangulation** `\mathcal{T}` of the embedded point cloud `{x_i} \subset \mathbb{R}^{D^*}`. This provides a canonical simplicial complex.
2.  **Discrete Metric:** Assign edge squared-lengths `l_{ij}^2 = h_{ab}(x_i) \Delta x^a \Delta x^b` consistent with the reconstructed local metric.
3.  **Discrete Curvature (Regge):** Compute the curvature using **Regge Calculus**. The curvature is concentrated on the `(D^*-2)`-dimensional bones (hinges). For `D^*=3`, these are edges. The deficit angle `\epsilon_h` at hinge `h` is:
    ```text
    \epsilon_h = 2\pi - \sum_{cell \supset h} \theta_{cell}(h)
    ```
    where `\theta_{cell}(h)` is the dihedral angle of the tetrahedron at hinge `h`.
    This definition is chosen because **Regge Calculus strictly satisfies the exact discrete Bianchi identities** (conservation of geometry), ensuring that the geometric side of the equation is structurally sound even at finite resolution.

#### 8.2.2 The Closure Mismatch Functional
We define the macroscopic closure mismatch `M_L` by comparing the discrete Regge Einstein tensor `G_{Regge}` against the coarse-grained stress-energy proxy on the graph.

Define the **discrete Regge Einstein tensor** `G_h` on each hinge `h`:
```text
G_h \cdot l_h := \epsilon_h - \Lambda V_h
```
(normalized appropriately by Voronoi dual volumes).

Define the **closure mismatch** `M(L)` over a region `V_L`:
```text
M(L) := \left\| G_{Regge} - 8\pi G \, \Pi_{graph}(T_{\mu\nu}^{eff}) \right\|_L
```
where `\Pi_{graph}` projects the effective field theory stress-energy `T_{\mu\nu}^{eff}` onto the discrete graph structure (integrating over dual volumes).

**Matching Requirement:**
In empirically accessed regimes, `M(L)` must be bounded by experimental precision `\varepsilon_{GR}(L)`. This ensures that the discrete geometry "shadows" a solution to Einstein's equations, converging to it in the continuum limit where applicable.

#### 8.2.3 Weak-field (Newtonian) consistency via discrete diagnostics
A minimal reviewer-facing check is the weak-field limit. This is evaluated directly on the discrete graph structure.

1.  **Discrete Scalar Potential:** The clock-rate potential $\Phi_i$ is already defined on the graph (via the Laplacian). In the weak-field limit, this identifies with the Newtonian potential.
2.  **Effective Mass Density:** We extract an effective mass density $\rho_{mass}$ from the discrete potential using the **discrete Poisson equation** on the Delaunay dual:
    ```text
    4\pi G \rho_i \approx (\Delta_{Delaunay} \Phi)_i
    ```
    where $\Delta_{Delaunay}$ is the cotan-weighted Laplacian on the triangulation.
3.  **Consistency:** The key demand is that the recovered $\rho_i$ matches the coarse-grained source distribution $\delta\rho$ projected onto the nodes.

#### 8.2.4 Parameter calibration versus parameter fitting
Matching GR introduces *scale-setting* constants (analogous to `c`, `G`, `Λ`) through:
- the unit choice `ℓ_*` (minimal encoding patch scale),
- the clock-rate field coupling $\beta_0$ and potential $\Phi_i$ (controlling gravitational time dilation),
- the distance mapping `f(I/I0)` (sets correlation-to-distance conversion).

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
```text
A_F = C ⊕ H ⊕ M_3(C).
```

### How this integrates non-ad-hoc with the projection
A natural integration point is at the cell level: treat each effective cell as carrying an internal factor,
schematically:
```text
A_i  ≈  M_{d_i}(C) ⊗ A_F,
```
(with the same “approximate split” caveats as in §4.4.2 for continuum/QFT limits).

- The compact SU(2)-like structure is naturally supported by the quaternionic component `H`.
- `M_3(C)` supports `SU(3)`-type color structure.
- `C` supports a `U(1)`-type phase.

Gauge structure then appears as **internal automorphisms** (unitaries) acting on the `A_F` factor that
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
```text
P(b | x, y) = P(b | x)   (within experimental bounds)
```

### 10.2 Parameterized junction-access extension
To keep the possibility of additional operational access open without asserting it, introduce a gating parameter `γ(regime) ∈ [0, 1]` and write:
```text
P(b | x, y) = Tr(ρ_B E_b^{(x)}) + γ · Δ_b(x, y; ρ_AB)
```
Constraints: (i) `Σ_b Δ_b = 0` for normalization, (ii) probabilities remain in `[0,1]`, and (iii) `γ ≈ 0` in all regimes tested so far.

## 11. Evaluation, falsification, and test program

### 11.1 What “prove or disprove” means here
This framework is structured as (i) **definitions** of substrate and projection primitives, plus (ii) **matching requirements** to known physics in empirically accessed regimes, plus (iii) optional **extension modules** (e.g., junction access) that may introduce new testable deviations. In the scientific sense, the framework is:
- **disprovable** if it cannot satisfy its own matching requirements without introducing hidden time/geometry, or if it predicts deviations already excluded by experiment;
- **supported** if a small-parameter instantiation reproduces broad classes of GR/QFT phenomena while remaining internally consistent and non-arbitrary.

### 11.2 Non-negotiable constraints (must hold in accessible regimes)
The following are hard requirements for viability:
- **No hidden substrate time:** all ordering must be definable as phase/action order; no background time parameter may be reintroduced implicitly.
- **No hidden substrate geometry:** locality, dimension, and metric must arise from the projection construction (coarse-graining → correlations → embedding).
- **Operational no-signaling (junction module):** in all regimes already probed experimentally, the junction-access gate must satisfy `γ ≈ 0` so that local marginals do not depend on distant choices.
- **Universality:** scale-setting constants and response functions (e.g., `f`, `β`) must be global (not tuned region-by-region).
- **Continuum regime:** at scales `L ≫ ℓ_*`, the reconstructed metric must admit smooth limits sufficient to compute curvature and compare with GR tests.

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
| Emergent locality via mutual information \(I_{ij}\) and graph metric \(d_G\) | Definition | Locality is defined from correlations and multi-hop routing on weighted graphs (no geometric inputs). | Definition is checkable; failure mode is pathological graphs (disconnectedness, non-manifold structure) for physically relevant states. |
| Dimension \(D^*\) from embedding distortion | Definition + matching output | \(D^*\) is selected by a distortion/complexity criterion; empirically we require \(D^*\approx 3\) in the stable regime. | Compute \(D^*\) across toy ensembles; falsified if \(D^*\) is unstable or consistently far from 3 in GR/QFT-like toy states. |
| Spatial metric \(h_{ab}(x)\) from SPD-constrained local fit | Definition | The emergent spatial metric is reconstructed from neighbor distances with explicit positive-definiteness constraints. | Validate numerically (SPD fit succeeds and is stable under smoothing); failure if SPD enforcement destroys the ability to match observables. |
| Clock-rate field \(\beta\) and proper time \(d\tau=\beta\,dS_{\text{act}}\) | Definition | Emergent time is defined as a conversion from phase-order length to clock time, with \(\beta\) sourced by an entropy-contrast potential (graph-Laplacian extension). | Validate weak-field behaviors in worked examples; falsified if exterior/vacuum tails cannot reproduce redshift/Shapiro-like effects without tuning. |
| Spacetime metric \(g_{\mu\nu}\) | Definition | \(g\) is defined by the embedding-derived \(h\) plus the clock mapping \(\tau\) (no separate constitutive law \(g=F(\rho)\)). | Pure definition; key is whether the reconstructed \(g\) admits a continuum regime enabling curvature tests. |
| GR closure mismatch functional \(M(L)\) | Definition + matching condition | At scales \(L\gg\ell_*\), reconstructed \(g^{(L)}\) must satisfy Einstein closure within tolerance. | Falsified if no small-parameter instantiation yields small \(M(L)\) across standard tests. |
| Born-rule statistics in accessible regimes | Matching condition (derivation open) | Outcome statistics must match standard QM in all currently tested regimes. | Falsified by any predicted deviation already excluded; derivation from projection-limited inference is an open program item. |
| Operational no-signaling in accessible regimes | Matching condition | Local marginals must not depend on distant choices (junction gate \(\gamma\approx 0\)). | Falsified by any proposal that enables controllable signaling in ordinary Bell-test regimes. |
| Junction-access deviations \(\gamma(\cdot)>0\) | Open module | Whether additional operational access exists in extreme regimes is left open and parameterized. | Must remain consistent with existing constraints; becomes testable if a concrete activation regime is specified. |
| Local Lorentz covariance | Matching condition | Despite a preferred phase-order foliation in the construction, local Lorentz symmetry must emerge as an excellent IR symmetry. | Constrained by precision Lorentz-violation bounds; falsified if implied violations exceed limits. |


### 11.4 Internal validation tests (toy-model / simulation tests of Π)
These tests do not require new physics; they evaluate whether the projection map is well-defined and non-arbitrary.

**T1 — Geometry recovery from known relational states.**
Choose a toy substrate state known to correspond (in standard physics) to an approximately local system (e.g., a lattice model ground state or thermal state). Apply the stability-selected coarse-graining, build `I_{ij}`, reconstruct distances, and test whether the embedding yields the correct effective dimension (≈3) and locality structure.

**T2 — Stability under phase-flow.**
For the selected `{E_i}`, verify low leakage `L_leak(E*)` and low drift (tie-breaker), demonstrating that the emergent cell net is stable under the phase/action ordering.

**T3 — Robustness to coarse-graining scale.**
Show that reconstructed large-scale geometry is stable when varying the smoothing scale `L` within a reasonable window (`L ≫ ℓ_*`), i.e., that `g^{(L)}` approaches a stable effective metric class.

These tests are “reviewer-proof” because they are algorithmic and can be implemented in controlled toy models.

### 11.5 Empirical compatibility checks (must reproduce known physics)
A minimal set of comparisons that any instantiation must pass:
- **Newtonian limit / weak field:** recover standard gravitational acceleration and redshift behavior from extracted `Φ` diagnostics (§8.2.3).
- **Light bending and time delay:** compute null geodesics of `g^{(L)}` and compare with standard lensing and Shapiro-delay phenomenology in weak field.
- **Consistency of clock mapping:** `β(x)` must reproduce observed gravitational time dilation in weak-field regimes without ad hoc spatial dependence.
- **Quantum statistics:** for laboratory-scale systems, projection must reproduce standard interference and entanglement statistics (Born-rule form at the operational level).
- **No-singularity claims (if asserted):** if the model claims bounded curvature universally, it must not conflict with any observed high-density astrophysical phenomena.

### 11.6 Distinctive empirical commitments and lethal falsifiers
The framework becomes strictly falsifiable when it asserts universal constraints that can be violated by specific mathematical or empirical counterexamples. The following are **severe, lethal falsifiers**:

**F1 — Dimensional Collapse.**
The framework is falsified if ground states of topological lattice toy models globally minimize complexity-stress at $D^* \to \infty$ (expander graphs) or $D^* = 1$ (trees).

**F2 — Volume-Law Saturation.**
Falsified if numerical simulations of split-property funnels show cut-capacity scaling with volume rather than boundary cuts in low-distortion regimes.

**F3 — Lorentz-Violation Bound Breach.**
Discrete spatial graphs inherently break Lorentz symmetry. Require that the internal SU(2) symmetry acts as a 'Custodial Symmetry' that algebraically cancels dimension-4 and dimension-5 Lorentz-violating operators. The framework is falsified if predicted energy-dependent photon speeds exceed the $10^{-14}$ experimental bound.

**F4 — The Two-Potential Disconnect.**
Falsified if matching the Shapiro delay ($\Phi_t$) and spatial lensing ($\Phi_s$) for the exact same source requires fundamentally different parameter calibrations.

**F5 — Junction Accessibility (Open Module).**
If `γ(regime)` is asserted nonzero in any accessible regime, then it predicts detectable signaling via marginal shifts. Conversely, existing no-signaling constraints imply `γ ≈ 0` in all currently accessed regimes; any model variant predicting otherwise is excluded.

### 11.7 A practical “test matrix” (what to measure and what would count as failure)
| Target claim/module | Observable proxy | What must hold | What would disprove it |
|---|---|---|---|
| Stability-selected locality | Leakage/drift metrics | `L_leak` small, drift small at fixed capacity | No stable decomposition exists without collapsing capacity |
| Emergent 3D space | Dimensionality of embedding | Best-fit embedding dimension ≈ 3 across scales | No stable low-dimensional embedding; dimension drifts wildly |
| GR closure in weak field | Extracted `Φ`, redshift, lensing | Matches standard weak-field phenomenology | Systematic mismatch not removable by universal calibration |
| Finite distinguishability (area law) | Scaling of `S(R)` with boundary cut | `S(R)` bounded by area scaling (operationally) | Demonstrated generic violation of area scaling at fixed regime |
| Junction module (γ) | Marginal dependence tests | No measurable marginal dependence in known regimes | Robust marginal dependence without classical side-channel explanation |

### 11.8 Parameter discipline as a falsifier (anti–overfitting rule)
A reviewer-proof stance is to treat **excess functional freedom** as a failure mode. Concretely:
- If matching GR/QFT requires `β(x)` or `f(I)` to vary by environment beyond a small universal parameter family, the model is overfit.
- If matching requires replacing the stability selection rule with ad hoc, state-dependent exceptions, the “unique projection” postulate is violated.


## 12. Discussion and open problems
Open problems include:

1) **Toy-model realizations of `Π`.** Construct explicit toy substrate algebras and states `(A, ω)` for which the stability-selected coarse-graining `{E_i}` can be computed (or approximated) and the induced locality graph and embedding can be visualized.

2) **GR matching as an emergent closure.** Demonstrate in a controllable regime (e.g., weak field) that the metric produced by the embedding + clock mapping satisfies Einstein’s equation to the required approximation, and identify what coarse-grained stress-energy corresponds to the projection-level excitations.

3) **Dynamic cell net / dynamic spacetime (“geometrogenesis”).** The selection principle `E` is defined over a phase-order window via the weight `w(s)`. In general, the minimizer may depend on the chosen window and (if one allows it) on phase-order location. A “moving-window” or adiabatic selection `E*(s)` would yield a phase-ordered family of graphs and embeddings, providing a route to dynamically evolving geometry within projection. The present draft treats `{E_i}` as fixed for simplicity; extending to `E_i(s)` is a natural next step and should be formulated carefully to preserve symmetry-commutation and retention constraints.

4) **Computational tractability.** Global minimization of leakage over all decompositions is intractable at large scale. For a physical theory, this should be interpreted as a variational/thermodynamic principle: the universe’s effective decomposition is expected to be selected by natural relaxation/typicality under the phase-flow, not by an explicit algorithm executed by agents. Toy models can treat the optimization literally.

5) **Quantum sector derivations.** Provide either (i) a derivation of the Born-rule trace form from finite distinguishability + invariance constraints, or (ii) a principled argument for why the projection map yields density operators and POVM statistics as the unique operational description at finite resolution.

6) **Standard Model embedding (optional milestone).** Make the internal algebra program concrete by specifying chiral representations, showing anomaly cancellation, and connecting representation content to stable projection excitations.

This draft is intended as a stable conceptual reference for these developments: it emphasizes explicit definitions and parameter accounting, and treats open issues (including junction accessibility) as constrained modules rather than as ungrounded assumptions.

## 13. Worked examples (operational extraction of standard observables)

This section adds three worked examples that show how the framework is *used* to extract familiar observables (redshift, bending angles, time delay, expansion rate) from the **discrete reconstructed metric**. These examples are intended to be **reviewer-proof** in the following sense:

- they do not introduce new postulates;
- they use only objects already defined in Sections 4–8 (`{E_i}`, `I_{ij}`, `d_G`, embedding `x_i`, Delaunay triangulation, discrete clock $\Phi_i$); and
- they produce explicit comparison targets (Regge-consistent lensing, simplicial volumes).

Throughout, the guiding rule is: **We do not smooth to a continuum.** We compute observables by shooting geodesics through the discrete simplicial complex.

### 13.0 Slicing, gauge, and the Shift Vector via Optimal Transport
To extract dynamic observables (redshift, expansion, gravitational waves) from a sequence of reconstructed discrete geometries, we must rigorously define the "identity" of a point across time. We do not use Procrustes alignment, as there is no ambient Euclidean background to align *in*.

Instead, we use **intrinsic Gromov-Wasserstein Optimal Transport** to construct the map between successive phase slices.

#### 13.0.1 The Shift Vector from Intrinsic Optimal Transport
To rigorously extract the ADM Shift Vector $N^a$ without "Euclidean smuggling" (subtracting coordinates from distinct spaces), we use **discrete differential geometry** to map the transport plan directly into the local tangent bundle.

1.  **Transport Plan:** Compute the Gromov-Wasserstein optimal transport plan $\gamma^*$ between the discrete metric measure spaces $\mathcal{X}_s$ and $\mathcal{X}_{s+\Delta s}$.
2.  **Tangent Space Projection:** For each site $i \in \mathcal{X}_s$, the plan defines a target distribution over $\{j\} \in \mathcal{X}_{s+\Delta s}$. We map each target $j$ into the local tangent space $T_i \mathcal{X}_s$ (defined by the reconstructed metric $h_{ab}$) via the **Discrete Logarithmic Map**:
    ```text
    v_{ij}^a \approx \text{Log}_{x_i}(x_j)
    ```
    This vector is constructed relationally: it is the unique vector $v^a \in T_{x_i}$ such that flowing along the geodesic generated by $v^a$ matches the distances $d(i, k)$ to the mapped distances $d(j, k')$ in the target slice, minimizing local distortion.
3.  **The Intrinsic Shift:** The shift vector $N^a$ is the expectation value of these tangent vectors under the transport plan:
    ```text
    N^a(x_i) \Delta s := \sum_j \frac{\gamma_{ij}^*}{\mu_i} v_{ij}^a
    ```
    This ensures that $N^a$ represents the **intrinsic geometric flow** of the coordinate system required to track the evolving correlation structure, satisfying diffeomorphism invariance without assuming any ambient background.

#### 13.0.2 Discrete Weak-field diagnostic potentials
We define the diagnostic potentials directly on the nodes of the graph.

**Background normalization.**  
Choose a reference region `B` (far from mass) and define $\Phi_B = \langle \Phi_i \rangle_B$.

**Time potential** $\Phi_t$ at node $i$:
```text
\Phi_t(i) := \Phi_i - \Phi_B
```
This is exact; the Laplacian-derived $\Phi_i$ *is* the gravitational potential.

**Space potential** $\Phi_s$ from discrete conformal factors:
For each node $i$, compute the local volume distortion relative to the background reference metric $h_{flat}$:
```text
a_s(i) := \left( \frac{\det(h_{ab}(i))}{\det(h_{flat})} \right)^{1/6}
```
Then $\Phi_s(i) := c^2 (1 - a_s(i))$.

**GR closure check:** In the weak field, we require $\Phi_t(i) \approx \Phi_s(i)$ across the graph.

---

### 13.1 Worked example A — isolated spherical mass (weak-field, static)
**Purpose.** Demonstrate how a localized persistent encoding-density footprint yields gravitational time dilation, lensing, and Shapiro delay using **discrete graph geodesics**.

#### 13.1.1 Setup
- Identify a "source" region $R_M$ where the entropy contrast $\delta\rho_i$ is high.
- The reconstruction produces a densified mesh near $R_M$ with a deep clock-rate potential well $\Phi_i < 0$.

#### 13.1.2 Discrete Geodesic Shooting
We do not integrate differential equations. We compute **shortest paths on the weighted graph/simplicial complex**.

1.  **Null Geodesics:** A light ray is a path of graph edges $\{e_1, e_2, \dots\}$ that minimizes the **optical length**:
    ```text
    L_{opt} = \sum_{edge=(i,j)} \frac{\ell_{ij}}{\beta_{avg}(i,j)}
    ```
    where $\ell_{ij}$ is the proper length from the reconstructed $h_{ab}$, and $\beta_{avg}$ is the clock rate. The factor $1/\beta$ accounts for the effective refractive index $n \approx 1 - 2\Phi/c^2$ of the gravitational field.

2.  **Redshift:** For a signal sent from node $A$ to node $B$:
    ```text
    1+z = \frac{\beta(A)}{\beta(B)} = \frac{e^{\Phi_A}}{e^{\Phi_B}}
    ```
    This is exact on the graph.

3.  **Lensing Angle:** Shoot two geodesics: one through the potential well (impact parameter $b$), one far away (reference).
    Measure the angular deviation of the final velocity vectors in the asymptotic region using the **discrete parallel transport** defined by the Delaunay connection.
    *Check:* Does the deflection vector $\vec{\alpha}$ scale as $4GM/b$?

4.  **Shapiro Delay:** Compare the arrival time $t_{arrival} = \int dS_{act}$ of the ray passing through the potential vs. the reference ray.
    ```text
    \Delta \tau \approx \sum_{edges} \ell_{ij} (1 - e^{\Phi_{avg}})
    ```
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
**Purpose.** Extract the scale factor $a(t)$ from the **discrete volume** of the Delaunay complex.

#### 13.3.1 Discrete Scale Factor
For a phase slice at $t_k$:
1.  Compute the Delaunay triangulation $\mathcal{T}_k$ of the embedded points $\{x_i(t_k)\}$.
2.  Sum the volumes of all tetrahedra $T \in \mathcal{T}_k$ in a large homogeneous patch $P$:
    ```text
    V_P(t_k) := \sum_{T \in P} \text{Vol}(T)
    ```
    where $\text{Vol}(T)$ is calculated using the Cayley-Menger determinant with the edge lengths $l_{ij}$ derived from the metric $h_{ab}$.
3.  **Scale Factor:**
    ```text
    a(t_k) := \left( \frac{V_P(t_k)}{V_P(t_0)} \right)^{1/3}
    ```

#### 13.3.2 Hubble Rate
Using the Laplacian-averaged proper time step $\Delta \bar{\tau}_k$ between slices:
```text
H(t_k) \approx \frac{1}{a(t_k)} \frac{a(t_{k+1}) - a(t_k)}{\Delta \bar{\tau}_k}
```

#### 13.3.3 Failure Modes (Falsification)
- **Volume Collapse:** If the Delaunay complex degenerates (slivers, zero-volume tetrahedra) despite a growing $V_P$, the geometry is not manifold-like.
- **Anisotropy:** If $a(t)$ differs significantly when measured along different graph axes (using discrete direction-dependent correlators), the emergent space is not FLRW.


---

### 13.4 How these worked examples connect to falsification
These examples define *what to compute* from a candidate instantiation `(A, ω)` and the specified selection rule `E`. The framework is disfavored if, under its own constraints (symmetry commutation, retention, area capacity, and stability), it cannot realize:

- a weak-field isolated-mass regime reproducing redshift and lensing;
- multi-lens weak-field behavior with approximate superposition and shear;
- a homogeneous expanding patch with robust scale-factor diagnostics.

In other words: these examples translate “emergent geometry” from a slogan into a set of explicit computational tests.
