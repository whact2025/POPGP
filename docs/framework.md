# A Phase-Ordered Pre-Geometric Projection Framework
*Conceptual framework draft (v0.8, AI-readable)*  
Author: [Your Name]  
Date: [YYYY-MM-DD]

## Abstract
We propose an atemporal, pre-geometric substrate described purely by relational/algebraic structure. A compact internal SU(2)-like symmetry and a distinguished phase/action ordering generator are taken as primitive. A unique, necessary physical projection map yields effective three-dimensional space, objective time-order (as a metric on phase order), and a finite distinguishability bound that scales with boundary area via a relational encoding boundary. Quantum discreteness is treated as emergent from stable representation content under projection constraints, while gravitational geometry is treated as an effective constitutive response of the projected metric to bounded encoding-density distributions. In empirically accessed regimes the framework is required to recover standard General Relativity and quantum field theoretic predictions. Possible additional operational access to nonlocal substrate correlations (including any potential signaling via entanglement “junctions”) is formulated as a constrained open module rather than assumed a priori.

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
**P2 (Emergent 3D space).** `Π` yields an effective three-dimensional spatial structure `M³` with emergent locality.  
**P3 (Emergent time-order).** `Π` yields an objective time-order in projection as a metric on phase/action order; non-conscious systems inherit this time-order.  
**P4 (Area-law finite distinguishability).** For any finite projected region `R`, physically distinguishable information is bounded by an area-law capacity tied to a relational encoding boundary.  
**P5 (Bounded intensities).** Projection-level densities and curvatures saturate rather than diverge; classical singularities indicate breakdown of the effective description.

### 3.3 Emergence postulates
**E1 (Quantum discreteness from representations).** Discrete quantum types arise from stable representation content of the internal symmetry under projection constraints, not from fundamental substrate discreteness.  
**E2 (Geometry as projection output).** The effective metric `g_{μν}` is constructed by the projection stages `Π_loc`, `Π_geom`, and `Π_time` (Section 4.4): correlation-derived distances define an emergent locality graph, an embedding procedure reconstructs the spatial metric `h_{ab}`, and phase/action ordering sets the time scaling `dτ = β(ρ)dS_act`. Encoding density `ρ(x)` is treated as a derived scalar summary of local capacity/coherence rather than a fundamental geometric primitive. Any apparent constitutive form `g ≈ F(ρ, …)` is understood as an emergent phenomenological approximation to the metric produced by the projection construction.
**E3 (GR matching).** In empirically accessed regimes, `g_{μν}` must satisfy Einstein’s equation with effective stress-energy matching known matter to experimental precision.  
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

#### 4.4.2 `Π_res`: resolution-limited coarse-graining into effective local “cells”
A projection must provide a notion of subsystems/regions, but subsystems are not substrate primitives.  
We introduce an *emergent cell decomposition* by selecting a family of finite-dimensional effective subalgebras:
```text
{A_i ⊂ π_ω(A)'' }_{i ∈ V},   with   A_i ≅ M_{d_i}(C),
```
together with conditional expectations (coarse-graining maps):
```text
E_i : π_ω(A)'' → A_i
```
that are completely positive, unital, and idempotent (E_i∘E_i = E_i).

For any finite set of cells `R ⊂ V`, define:
```text
A_R = ⊗_{i∈R} A_i,
E_R = ⊗_{i∈R} E_i,
ω_R = ω ∘ E_R .
```

**Finite distinguishability constraint (imposed here).**  
The choice of `{A_i}` and `{E_i}` is restricted so that for all finite regions `R`:
```text
S(ω_R) ≤ η · A(∂R) / ℓ_*²,
```
where `S(ω_R)` is the von Neumann entropy of the density operator representing `ω_R` on `H_R`,
and `A(∂R)` is a projection-level boundary area measure (defined in §4.4.4 via the locality graph / min-cut).

This is where “Planck-scale resolution” enters: the projection cannot resolve arbitrarily large `d_i` and cannot support arbitrarily fine-grained independent degrees of freedom in finite regions.

#### 4.4.2a Selection Principle `E`: symmetry-commuting, phase-flow stable cell net

The choice of the effective cell algebras `{A_i}` and coarse-graining maps `{E_i}` is fixed by a *stability principle* tied to the phase/action ordering flow `σ_s` (§4.3), rather than chosen ad hoc. Intuitively: the correct decomposition is the one for which phase/action ordering does not continually “slosh” information across emergent cell boundaries.

Let `α_g` denote the internal SU(2)-like symmetry action on `A` (or on `π_ω(A)''` after the GNS embedding). Let `σ_s` denote the phase/action ordering flow.

**Admissible coarse-grainings.** Define `𝔈_adm` as the set of families `{E_i}` (and their associated cell algebras `{A_i}`) satisfying all of:

1) **Finite-capacity cells.** Each cell algebra is finite-dimensional:
```text
A_i ≅ M_{d_i}(C),
```
with either a fixed common dimension `d_i = d` for all cells (simplest), or an explicit local bound `d_i ≤ d_max` (bounded capacity). In all cases the area-law constraint in §4.4.2 is required to hold for all finite regions `R`.

2) **Symmetry commutation (internal isotropy constraint).** Each coarse-graining map commutes with the SU(2)-like symmetry:
```text
E_i ∘ α_g = α_g ∘ E_i    for all g ∈ SU(2) and all i.
```
Equivalently (often easier to verify): the subalgebra `A_i` is invariant under `α_g`, and `E_i` is SU(2)-equivariant. This prevents the projection from introducing spurious preferred directions in the emergent 3D description.

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






**Variational interpretation (no “global computation” assumption).** The selection rule is a *definition* of the projection (analogous in spirit to least-action principles), not an assumption that any agent or physical subsystem explicitly solves an NP-hard optimization. In toy models one can compute or approximate `E*` directly. In a physical reading, the minimizer is interpreted as the stable fixed-point/attractor coarse-graining compatible with (i) phase-flow stability, (ii) SU(2)-equivariance, and (iii) bounded information loss under the finite-distinguishability constraints. The role of the minimization is to remove arbitrariness and expose a small, auditable freedom set (choice of norm, window weight, etc.), not to posit an explicit algorithm executed by nature.

**Implementation notes (default choices for toy models).**
- Weight `w(s)`: choose a finite phase-order window of width `Δs` and set `w(s)=1/Δs` on that window (and `0` outside), or use a Gaussian centered at `s0`. `Δs` should be small enough that the selected cell net is approximately stable across the window.
- Norm `||·||`: use a Hilbert–Schmidt / Frobenius norm in a fixed representation as a computational proxy in toy models; treat the diamond norm `||·||_⋄` as the operationally strongest target definition.
- Retention budget `ε`: define the capacity of a reference region `R0` by `S_cap(R0):=η·A(∂R0)/ℓ_*²` and choose `ε = κ·S_cap(R0)` with `κ≪1` (or impose per-cell bounds). This prevents the trivial “erase everything” minimizer.
- Drift step `δ`: choose `δ ≪ Δs` (e.g., `δ = Δs/N` with large `N`) so that `L_drift` approximates a local drift rate.
- Optimization style: prefer the stated lexicographic minimization over weighted sums to avoid introducing arbitrary weight parameters.

#### 4.4.3 `Π_loc`: locality from correlations (graph metric from mutual information)

Given the effective cell net, define reduced density operators `ρ_R` from `ω_R` in the standard way
(using the finite-dimensional representation of `A_R` on `H_R`).

Define the mutual information between cells `i` and `j`:
```text
I_{ij} = S(ρ_i) + S(ρ_j) - S(ρ_{ij}).
```

Interpret `I_{ij}` as a *relational proximity kernel*: higher mutual information indicates tighter relational coupling.

Define an emergent distance between cells by a monotone decreasing map `f`:
```text
d_{ij} = ℓ_* · f(I_{ij} / I_0),
```
where `I_0` is a reference scale (e.g., typical nearest-neighbor mutual information),
and `f` satisfies:
- `f(u) ≥ 0`,
- `f'(u) < 0`,
- `f(u→0) → +∞` (very weak correlation → far apart).

A simple low-parameter choice (often used because it turns products into sums) is:
```text
f(u) = max{0, -log u }.
```

Now build a weighted graph `G = (V, E, d)`:
- vertices are cells `i ∈ V`,
- edges `E` connect pairs with `I_{ij}` above a threshold (or k-nearest neighbors),
- edge lengths are `d_{ij}`.

Graph geodesic distance:
```text
d_G(i,j) = inf_{paths i→j} Σ_{(a,b) in path} d_{ab}.
```

This defines a metric space `(V, d_G)` without assuming any background geometry.

#### 4.4.4 `Π_geom`: from correlation metric space to an emergent 3D geometry

**Embedding (coordinate realization).**  
Choose coordinates `x_i ∈ R³` by minimizing “stress” between graph distances and Euclidean distances:
```text
min_{ {x_i} ⊂ R³ }  Σ_{i<j} w_{ij} ( ||x_i - x_j|| - d_G(i,j) )²,
```
with weights `w_{ij}` (e.g., larger for near neighbors).

This yields an emergent point cloud in `R³`. In the continuum limit (many cells) this approximates a 3D manifold `M³`
when the stress is small and local neighborhoods are well approximated by 3D charts.

**Local metric reconstruction.**  
For each cell `i`, use a neighborhood `N(i)` (near neighbors) and fit a local metric tensor `h_ab(x_i)` by solving:
```text
d_G(i,j)² ≈ (x_j - x_i)^a h_ab(x_i) (x_j - x_i)^b   for j ∈ N(i).
```
This is a linear least-squares problem for the symmetric matrix `h_ab(x_i)`.

The result is an emergent spatial metric field `h_ab(x)` on `M³`.

**Boundary area (needed for the area law).**  
Given the graph, define the boundary “area” of a region `R ⊂ V` by a cut functional:
```text
A(∂R) := ℓ_*² · Σ_{(i,j) crosses the cut} a_{ij},
```
where `a_{ij}` is a dimensionless weight per edge (often taken proportional to `I_{ij}` or set to 1 for simplicity).
This is a discrete surrogate for boundary area consistent with an area-law capacity bound.

In a continuum limit, this cut definition becomes the area of a surface separating `R` from its complement.

**Encoding density field.**  
Define a local encoding density (one convenient operational choice) as entropy per minimal patch area:
```text
ρ(x_i) := S(ρ_i) / ℓ_*².
```
More refined choices use the derivative of entropy with respect to boundary area:
```text
ρ(x) := dS(R)/dA(∂R)   (evaluated locally in the continuum limit).
```

#### 4.4.5 `Π_time`: time-order from phase/action ordering
Let `σ_s` denote the phase/action ordering flow on `A` (generator form or canonical-flow form in §4.3).
Define a monotone “unwrapped action length” along the flow, `S_act(s)`, so that:
```text
S_act(s2) > S_act(s1)  iff  s2 is later than s1 in phase-order.
```

Projection-time is then defined by a local conversion from action-length to clock time:
```text
dτ(x) = β(ρ(x)) · dS_act,
```
where `β(ρ) > 0` is a monotone response function (interpreted as a local clock-rate mapping).

This is the native mechanism behind time dilation in the framework: different encoding density implies different
conversion between phase-order advance and experienced/clock time.

A minimal projection-level spacetime line element can then be written (in a comoving slicing) as:
```text
ds² = -c² dτ² + h_ab(x) dx^a dx^b.
```

#### 4.4.6 Summary: the projection map as output object
With the above stages, the projection map can be summarized as:
```text
Π(A, ω, α) = ( M³, h_ab(x), τ(x), ρ(x), {ρ_R}, A(∂R), … )
```
with:
- `M³, h_ab` from correlation-defined embedding and local metric reconstruction,
- `ρ(x)` from entropy-per-area (or an equivalent capacity-density definition),
- `τ(x)` from the phase/action-order → time conversion.

**Where freedom remains (auditable).**  
The main “degrees of freedom” that remain in this template are:
- the selection principle for `{A_i, E_i}` (specified in §4.4.2a; remaining explicit tolerances include `ε`, the channel norm choice, and the weight `w(s)`),
- the monotone map `f` from mutual information to distance,
- the clock-rate function `β(ρ)`.

These are exactly the objects that must be constrained by universality, minimal functional freedom, and matching requirements
to avoid parameter fitting (see §4.6.4).


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

### 6.1 Area-law capacity bound
For any finite region `R` in projection:
```text
S(R) ≤ η · A(∂R) / ℓ_*²
```

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
In this framework the effective metric `g_{μν}` is not introduced by a separate primitive function `F`. Instead, `g_{μν}` is *defined* by the output of the explicit projection construction in Section 4.4.

**Definition (metric from embedding + phase-order clock mapping).** The projection defines:

1) **Correlation distances.** From the stability-selected cell net `{A_i, E_i}` (§4.4.2a), compute mutual informations `I_{ij}` and distances `d_{ij}`, and obtain the graph-geodesic distance `d_G(i,j)` (§4.4.3).

2) **Embedding into 3D.** Define an embedding `x: V → R^3` by minimizing the stress functional (§4.4.4a):
```text
x* ∈ argmin_{x_i ∈ R^3}  Σ_{i<j} w_{ij} ( ||x_i - x_j|| - d_G(i,j) )^2 .
```

3) **Spatial metric reconstruction.** Reconstruct the local spatial metric `h_{ab}(x_i)` from neighbor distances (§4.4.4b):
```text
d_G(i,j)^2 ≈ (x_j - x_i)^a · h_{ab}(x_i) · (x_j - x_i)^b    for j ∈ N(i).
```

4) **Encoding density as a derived scalar.** Define encoding density as a derived scalar summary of local capacity/coherence (§4.4.4d), e.g.
```text
ρ(x_i) := S(ρ_i)/ℓ_*^2     (cell-entropy proxy),
```
or, in a continuum limit,
```text
ρ(x) := dS(R)/dA(∂R)       (local capacity density).
```

5) **Emergent proper time.** Construct emergent proper time from phase/action order (§4.4.5):
```text
dτ(x) = β(ρ(x)) · dS_act .
```

Given the preferred slicing induced by phase/action ordering, the emergent spacetime line element in coordinates `(τ, x^a)` is then:
```text
ds^2 = -c^2 dτ^2 + h_{ab}(x) dx^a dx^b .
```

This *is* the constitutive content of the framework: `g_{μν}` is the metric implied by (i) stability-selected correlation geometry (graph → embedding → `h_{ab}`) and (ii) the local phase-order-to-clock-time mapping (`dτ = β(ρ)dS_act`). Any effective low-parameter relationship of the schematic form `g_{μν} ≈ F_{μν}(ρ, ∂ρ, …)` should be understood as a phenomenological approximation to this construction (e.g., weak-field expansion), not as an independent defining postulate.

### 8.2 GR matching as an effective closure on reconstructed metric fields
This framework does not postulate Einstein’s equation as a substrate law. Instead, **GR is imposed as a tested-regime closure condition** on the *reconstructed* effective metric field produced by the projection construction.

#### 8.2.1 Continuum reconstruction and smoothing
The correlation-derived geometry is defined initially on a discrete cell set `V` (a weighted graph with embedded coordinates `{x_i}` and local spatial metric estimates `h_ab(x_i)`). To compare with continuum GR, introduce a coarse-graining/smoothing operator at scale `L ≫ ℓ_*`:

- Construct an interpolated spatial metric field `h_ab^{(L)}(x)` (e.g., via kernel regression or finite-element/triangulation on the point cloud).
- Construct the emergent proper time field `τ^{(L)}(x)` via `dτ = β(ρ^{(L)}(x)) dS_act`, with `ρ^{(L)}` the correspondingly smoothed encoding-density proxy.

This yields an effective 3+1 metric (in the preferred slicing induced by phase/action order):
```text
g^{(L)}_{μν} dx^μ dx^ν := -c^2 dτ^{(L)}(x)^2 + h^{(L)}_{ab}(x) dx^a dx^b .
```

#### 8.2.2 Matching criterion
Let `T^{(L)}_{μν}` denote the effective stress-energy tensor used to describe projection-level excitations at scale `L` (in the simplest matching program this may be taken from standard effective field theory in curved space at that scale). Define the **closure mismatch**:
```text
M(L) := || G_{μν}[g^{(L)}] + Λ g^{(L)}_{μν} - 8πG T^{(L)}_{μν} ||_L
```
where `||·||_L` is a norm appropriate for comparison on the coarse-grained region (e.g., an `L^2` norm over a domain or a maximum norm on selected observables).

**GR matching requirement (tested regime).**
In empirically accessed regimes and at scales `L` where continuum physics is valid,
```text
M(L) ≤ ε_GR(L)
```
for a tolerance `ε_GR(L)` set by the precision of the relevant experimental tests.

This formulation makes the empirical requirement explicit and checkable without introducing a new primitive “constitutive law.”

#### 8.2.3 Weak-field (Newtonian) consistency check
A minimal reviewer-facing check is the weak-field limit. For slowly varying fields, one demands that there exists a scalar potential `Φ(x)` such that, after an appropriate coordinate choice within the preferred slicing,
```text
g_{00} ≈ -(1 + 2Φ/c^2),     h_{ab} ≈ (1 - 2Φ/c^2) δ_{ab},
```
and `Φ` satisfies (to leading order) the Poisson equation with the effective mass density derived from projection-level excitations:
```text
∇^2 Φ ≈ 4πG ρ_mass .
```

In this framework, `Φ` is not fundamental; it is a diagnostic extracted from `g^{(L)}`. The key demand is that the projection construction admits regimes where the reconstructed metric yields the observed Newtonian limit and standard post-Newtonian corrections.

#### 8.2.4 Parameter calibration versus parameter fitting
Matching GR introduces *scale-setting* constants (analogous to `c`, `G`, `Λ`) through:
- the unit choice `ℓ_*` (minimal encoding patch scale),
- the clock mapping `β(ρ)` (controls gravitational time dilation in the preferred slicing),
- the distance mapping `f(I/I0)` (sets correlation-to-distance conversion).

To avoid unconstrained fitting, the framework treats these as:
1) **universal functions/constants** (no region-by-region tuning), and
2) **minimally parameterized** (e.g., monotone/saturating families) with calibration fixed by a small set of benchmark tests (Newtonian limit, gravitational redshift, lensing).

Any additional freedom beyond this limited set should be declared explicitly as a model extension.


## 9. Standard Model compatibility (program)
A full embedding of the Standard Model is not derived in this draft. However, an algebraic route is identified that is compatible with a pre-geometric substrate.

### 9.1 Candidate internal algebra
A commonly studied finite algebra capable of yielding the Standard Model gauge group via its unitary structure is:
```text
A_F = C ⊕ H ⊕ M_3(C)
```
Under appropriate unimodularity conditions on its unitary group action, this can reproduce `SU(3)×SU(2)×U(1)` as an effective gauge symmetry. Anomaly cancellation can be imposed as a projection admissibility constraint on allowed chiral representations.

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

### 11.3 Internal validation tests (toy-model / simulation tests of Π)
These tests do not require new physics; they evaluate whether the projection map is well-defined and non-arbitrary.

**T1 — Geometry recovery from known relational states.**
Choose a toy substrate state known to correspond (in standard physics) to an approximately local system (e.g., a lattice model ground state or thermal state). Apply the stability-selected coarse-graining, build `I_{ij}`, reconstruct distances, and test whether the embedding yields the correct effective dimension (≈3) and locality structure.

**T2 — Stability under phase-flow.**
For the selected `{E_i}`, verify low leakage `L_leak(E*)` and low drift (tie-breaker), demonstrating that the emergent cell net is stable under the phase/action ordering.

**T3 — Robustness to coarse-graining scale.**
Show that reconstructed large-scale geometry is stable when varying the smoothing scale `L` within a reasonable window (`L ≫ ℓ_*`), i.e., that `g^{(L)}` approaches a stable effective metric class.

These tests are “reviewer-proof” because they are algorithmic and can be implemented in controlled toy models.

### 11.4 Empirical compatibility checks (must reproduce known physics)
A minimal set of comparisons that any instantiation must pass:
- **Newtonian limit / weak field:** recover standard gravitational acceleration and redshift behavior from extracted `Φ` diagnostics (§8.2.3).
- **Light bending and time delay:** compute null geodesics of `g^{(L)}` and compare with standard lensing and Shapiro-delay phenomenology in weak field.
- **Consistency of clock mapping:** `β(ρ)` must reproduce observed gravitational time dilation in weak-field regimes without ad hoc spatial dependence.
- **Quantum statistics:** for laboratory-scale systems, projection must reproduce standard interference and entanglement statistics (Born-rule form at the operational level).
- **No-singularity claims (if asserted):** if the model claims bounded curvature universally, it must not conflict with any observed high-density astrophysical phenomena.

### 11.5 Distinctive empirical commitments and falsifiers
The framework becomes more falsifiable as you elevate additional commitments from “interpretation” to “prediction.” Examples:

**C1 — Universal area-law capacity (strong).**
If the area-law distinguishability bound is claimed to apply universally (not only in special gravitational regimes), then any demonstrated **volume-scaling maximum capacity** for generic finite regions would falsify the claim.

**C2 — Bounded encoding density / saturation (moderate).**
If the framework asserts a maximum effective encoding density (hence maximum curvature) rather than true singularities, then evidence requiring unbounded curvature/density (not merely extreme values) would falsify that assertion.

**C3 — Junction accessibility (open module).**
If `γ(regime)` is asserted nonzero in any accessible regime, then it predicts detectable signaling via marginal shifts. Conversely, existing no-signaling constraints imply `γ ≈ 0` in all currently accessed regimes; any model variant predicting otherwise is excluded.

**C4 — Correlation geometry as the unique source of spatial metric (strong).**
If the spatial metric is claimed to be fully determined by correlation distances, then persistent empirical signatures of geometry that cannot be reproduced by any admissible correlation-geometry reconstruction (without breaking the selection principles) would falsify the reconstruction hypothesis.

### 11.6 A practical “test matrix” (what to measure and what would count as failure)
| Target claim/module | Observable proxy | What must hold | What would disprove it |
|---|---|---|---|
| Stability-selected locality | Leakage/drift metrics | `L_leak` small, drift small at fixed capacity | No stable decomposition exists without collapsing capacity |
| Emergent 3D space | Dimensionality of embedding | Best-fit embedding dimension ≈ 3 across scales | No stable low-dimensional embedding; dimension drifts wildly |
| GR closure in weak field | Extracted `Φ`, redshift, lensing | Matches standard weak-field phenomenology | Systematic mismatch not removable by universal calibration |
| Finite distinguishability (area law) | Scaling of `S(R)` with boundary cut | `S(R)` bounded by area scaling (operationally) | Demonstrated generic violation of area scaling at fixed regime |
| Junction module (γ) | Marginal dependence tests | No measurable marginal dependence in known regimes | Robust marginal dependence without classical side-channel explanation |

### 11.7 Parameter discipline as a falsifier (anti–overfitting rule)
A reviewer-proof stance is to treat **excess functional freedom** as a failure mode. Concretely:
- If matching GR/QFT requires `β(ρ)` or `f(I)` to vary by environment beyond a small universal parameter family, the model is overfit.
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

This section adds three worked examples that show how the framework is *used* to extract familiar observables (redshift, bending angles, time delay, expansion rate) from the reconstructed metric. These examples are intended to be **reviewer-proof** in the following sense:

- they do not introduce new postulates;
- they use only objects already defined in Sections 4–8 (`{E_i}`, `I_{ij}`, `d_G`, embedding `x_i`, reconstructed `h_ab`, and clock mapping `β(ρ)`); and
- they produce explicit comparison targets (GR weak-field formulas, lens equation structure, FLRW diagnostics).

Throughout, the guiding rule is: **`g^{(L)}` is whatever the projection construction produces**; GR enters only as the *closure condition* and comparison standard (§8.2).

### 13.0 Slicing, coordinate gauge, and “what is being compared”
To compare reconstructed metrics to standard weak-field and cosmological diagnostics, we fix two kinds of gauge freedom that are inherent in the projection construction:

1) **Preferred slicing (time coordinate).**  
Use the global phase-order parameter as the preferred coordinate time:
```text
t := S_act   (up to an overall constant scale).
```
The local proper-time mapping is:
```text
dτ(x) = β(ρ(x)) dt .
```
So in the preferred slicing the reconstructed coarse-grained metric can be written as:
```text
ds^2 = -c^2 β(ρ(x))^2 dt^2 + h_ab(x) dx^a dx^b .
```
In weak-field language, `β(ρ)` acts as a **lapse-like clock-rate factor**, while `h_ab` is the reconstructed spatial metric produced by correlation geometry.

2) **Embedding gauge (spatial coordinates).**  
The embedding `x: V → R^3` produced by stress minimization is unique only up to global Euclidean isometries (translations, rotations, and reflections). This is not a problem for single-slice observables (distances, geodesics), but it matters when comparing multiple slices (e.g., cosmology). To compare slices at different `t`, fix the gauge by aligning embeddings using a rigid Procrustes transform on a chosen reference subset of cells (e.g., a background patch), so that `x_i(t)` can be compared across `t` without spurious coordinate drift.

### 13.0.1 Weak-field diagnostic potentials extracted from g^(L)
Given a reconstructed and smoothed metric `g^{(L)}` (Section 8.2.1), define weak-field **diagnostic** potentials from `g_00` and the isotropic part of `h_ab`. These are not assumed fundamental; they are extracted from the projection output for comparison to GR.

**Background normalization.**  
Choose a reference region `B` (far from an isolated mass, or an average over a homogeneous patch) and define:
```text
β_B := ⟨β(ρ(x))⟩_B,
h_B := ⟨h_ab(x)⟩_B .
```
Use `h_B` to define the local “flat” reference metric for normalization within the patch (i.e., treat `δ_ab` as the normalized form of `h_B`).

**Time potential** `Φ_t(x)` from `g_00 = -c^2 β(x)^2`:
```text
Φ_t(x) := (c^2/2) ( (β(x)^2 / β_B^2) - 1 )
       ≈ c^2 ( (β(x)/β_B) - 1 )    (weak field).
```

**Space potential** `Φ_s(x)` from the isotropic (conformal) part of `h_ab`:
define a local conformal factor relative to background:
```text
a_s(x) := ( det(h_ab(x)) / det(h_B) )^(1/6) .
```
In a near-isotropic weak-field regime, `a_s(x) ≈ 1 - Φ_s(x)/c^2`, so define:
```text
Φ_s(x) := c^2 ( 1 - a_s(x) )   (weak field diagnostic).
```

**GR closure check.**  
In the GR weak-field limit (in standard gauge), one expects `Φ_t ≈ Φ_s ≡ Φ` up to post-Newtonian corrections and coordinate choices. This framework does not assume that; it is a check implied by the GR matching requirement (§8.2.3). When possible, the most robust test is to compute observables directly from geodesics of `g^{(L)}` (coordinate-invariant), and treat `(Φ_t, Φ_s)` as interpretable diagnostics.

---

### 13.1 Worked example A — isolated spherical mass (weak-field, static)
**Purpose.** Demonstrate how a localized persistent encoding-density footprint (interpreted as “matter” in projection) yields:
- gravitational time dilation (redshift),
- light bending (lensing),
- and Shapiro time delay,
using only the reconstructed metric.

#### 13.1.1 Setup (what you assume, minimally)
- Choose a phase-order window where the system is approximately stationary (a static regime): correlations and `ρ(x)` are approximately time-independent under the preferred slicing.
- Choose a region in the emergent space containing a compact “source” region `R_M` where `ρ(x)` (or the local entropy proxy `S(ρ_i)`) exceeds background.

No spherical symmetry is imposed at the substrate level; “spherical” here is an emergent statement about the reconstructed geometry at scales `L ≫ ℓ_*`.

#### 13.1.2 Reconstruction procedure (algorithmic)
1) Use the stability-selected cell net `{E_i}` (Section 4.4.2a).
2) Compute `I_{ij}` and convert to graph distances `d_G(i,j)` (Section 4.4.3).
3) Embed into `R^3` via stress minimization to obtain coordinates `{x_i}` (Section 4.4.4a).
4) Reconstruct `h_ab(x)` from neighbor distances (Section 4.4.4b), and smooth to `h_ab^{(L)}(x)` (Section 8.2.1).
5) Compute `ρ^{(L)}(x)` (entropy or capacity-density proxy) and thus `β(ρ^{(L)}(x))`.
6) Assemble the coarse-grained metric:
```text
g_00^{(L)}(x) = -c^2 β(ρ^{(L)}(x))^2,     g_ab^{(L)}(x) = h_ab^{(L)}(x).
```

#### 13.1.3 Observable extraction
**(A) Gravitational redshift / time dilation.**  
For two stationary observers at radii `r1, r2` (defined with respect to the reconstructed spatial metric), the predicted redshift is:
```text
1 + z ≈ β(ρ(r_emit)) / β(ρ(r_obs))     (weak field, static).
```
Compare to GR’s weak-field redshift `1+z ≈ 1 + (Φ(r1)-Φ(r2))/c^2`.

**(B) Light bending.**  
Compute null geodesics of `g^{(L)}` numerically, or (in weak field) use the standard lensing integral in terms of diagnostic potentials:
```text
α(b) ≈ (2/c^2) ∫ ∇_⊥ (Φ_t + Φ_s) dz .
```
For GR-like closure (`Φ_t ≈ Φ_s`), this reduces to:
```text
α(b) ≈ (4/c^2) ∫ ∇_⊥ Φ dz .
```
If the reconstructed geometry is approximately Schwarzschild in weak field, the result should approach:
```text
α(b) ≈ 4 G M_eff / (c^2 b) .
```

**Operational definition of M_eff in this framework.**  
Because “mass” is not a primitive here, define an effective enclosed mass from the reconstructed potential:
```text
M_eff(r) := (r^2 / G) dΦ_t/dr    (spherical approximation diagnostic).
```
A reviewer-proof check is that `M_eff(r)` is approximately constant outside the source region.

**(C) Shapiro delay.**  
For a null path passing near the mass, the excess coordinate time relative to the same endpoints in flat space is (weak field):
```text
Δt ≈ (2/c^3) ∫ (Φ_t + Φ_s) dz ,
```
and the corresponding proper-time delay for distant clocks follows from `dτ = β(ρ) dt`.

#### 13.1.4 What would count as failure (specific falsifiers)
- No stable reconstruction yields a near-spherically symmetric `h_ab^{(L)}` and `β(ρ)` profile from any admissible `{E_i}` (violates viability of `Π` for this regime).
- Extracted redshift ratios cannot be made consistent with weak-field gravitational time dilation using any **universal** `β(ρ)` within the allowed minimal family.
- Lensing angle scaling with impact parameter does not match observed `~1/b` behavior in weak field (after universal calibration).

---

### 13.2 Worked example B — two-body lensing (binary mass distribution)
**Purpose.** Show how the framework produces the standard *qualitative* and *quantitative* structure of gravitational lensing by multiple masses: superposition in the weak field, shear, and image multiplicity.

#### 13.2.1 Setup
- Identify two localized high-`ρ` regions `R_M1` and `R_M2` separated by distance `D` in the reconstructed 3D embedding.
- Work in a quasi-static regime (geometry changes slowly compared to light transit across the region), so a “snapshot” metric is meaningful.

#### 13.2.2 Reconstruction procedure
Use the same pipeline as in §13.1 to obtain `g^{(L)}` for the region containing both masses. The only difference is that the reconstructed potentials will exhibit two centers.

Define weak-field diagnostic potentials `Φ_t(x), Φ_s(x)` as in §13.0.1.

#### 13.2.3 Lensing observables (weak field)
**Deflection field.** The deflection angle vector at impact parameter `b` in the lens plane is:
```text
α⃗(θ⃗) ≈ (2/c^2) ∫ ∇_⊥ (Φ_t + Φ_s) dz
```
evaluated along an unperturbed (Born-approximation) path, or by full geodesic integration in `g^{(L)}`.

**Lens equation structure.** For a source angle `β⃗` and observed image angle `θ⃗`, the standard weak-lensing form is:
```text
β⃗ = θ⃗ - (D_LS/D_S) α⃗(θ⃗) .
```
In this framework, `α⃗(θ⃗)` is computed from the reconstructed metric; the distance ratios use reconstructed geometric distances in the smoothed `h_ab^{(L)}`.

**Shear and convergence.** If you define a projected lensing potential `ψ(θ⃗)` from `(Φ_t+Φ_s)`, the convergence `κ` and shear `γ` are derived from second derivatives of `ψ`. Operationally, compute them from the Jacobian of the mapping `β⃗(θ⃗)` derived from the geodesic shooting map.

#### 13.2.4 Reviewer-proof checks (what you demand)
- **Weak-field linearity:** In regimes where GR predicts approximate superposition, the reconstructed deflection field should be approximately the sum of the two single-lens fields:
```text
α⃗_total ≈ α⃗_1 + α⃗_2    (within tolerance).
```
- **Symmetry:** In the equal-mass case, the lensing map should display the expected reflection symmetries in the reconstructed embedding.
- **Scaling:** Deflection should scale approximately with the inferred effective masses `M_eff` extracted by the diagnostic in §13.1.

#### 13.2.5 Failure modes
- Deflection field fails to approximate superposition in the weak field without invoking non-universal tuning of `β(ρ)` or `f(I)`.
- Image multiplicity/shear behavior cannot be reproduced in any admissible reconstruction, indicating that correlation geometry is insufficient to represent the known lensing degrees of freedom.

---

### 13.3 Worked example C — homogeneous cosmology patch (FLRW-like slice)
**Purpose.** Show how an approximately homogeneous and isotropic expanding cosmology can be represented as a phase-ordered family of reconstructed 3D geometries, and how standard cosmological diagnostics (scale factor, Hubble rate) are extracted.

#### 13.3.1 The key idea (dynamic geometry without substrate time)
A cosmology is not a single static metric but a *family* of reconstructed metrics indexed by phase order. The substrate provides the phase/action ordering flow `σ_s`; projection maps it into a time parameter and evolving 3D geometry.

There are two conceptually distinct implementations:

- **Fixed cell net, evolving correlations (preferred in v0.7):** Choose a single stability-selected `{E_i}` over a broad phase-order window and reconstruct geometry from `ω_s := ω∘σ_s` at different `s`.
- **Moving-window cell net (open problem):** Allow `{E_i}` itself to be re-optimized adiabatically as `s` advances, producing `E_i(s)` (see §12.3). This is not required to define cosmological expansion but may become important near “geometrogenesis” transitions.

This worked example uses the **fixed cell net** method to avoid introducing new selection machinery.

#### 13.3.2 Reconstruction as a function of phase order
For a sequence of phase-order values `{s_k}` (or coordinate times `{t_k}`):
1) Define the phase-shifted substrate state:
```text
ω_{s_k} := ω ∘ σ_{s_k}.
```
2) Compute reduced cell states and mutual informations `I_{ij}(s_k)` using the *same* `{E_i}`.
3) Build distances `d_G(s_k)`, embed to `{x_i(s_k)}`, and reconstruct `h_ab^{(L)}(t_k, x)`.

This yields a family:
```text
g^{(L)}(t_k) = -c^2 β(ρ^{(L)}(t_k,x))^2 dt^2 + h_ab^{(L)}(t_k,x) dx^a dx^b .
```

#### 13.3.3 Extracting an FLRW diagnostic (scale factor and Hubble rate)
Choose a large patch `P` where the reconstructed geometry is approximately homogeneous and isotropic. Define a spatial average of the metric (or its volume element) over the patch.

A robust scale-factor diagnostic uses the spatial volume element:
- Let `V_P(t)` be the reconstructed spatial volume of `P` computed from `h_ab^{(L)}(t,x)`:
```text
V_P(t) := ∫_P sqrt(det(h^{(L)}(t,x))) d^3x .
```
- Define the scale factor (relative to a reference time `t0`) as:
```text
a(t) := ( V_P(t) / V_P(t0) )^{1/3} .
```

Convert to proper time using the averaged clock mapping:
```text
dτ̄ := ⟨β(ρ^{(L)}(t,x))⟩_P dt .
```

Then the Hubble rate diagnostic is:
```text
H(τ̄) := (1/a) da/dτ̄ .
```

#### 13.3.4 What to compare to observations (without claiming prediction yet)
Even before deriving a full dynamical law for `a(t)`, the framework can be evaluated by whether it can *represent* the standard cosmological structure in an admissible reconstruction:

- **Hubble law representation:** Whether comoving separations in the reconstructed metric scale approximately with `a(t)` in a homogeneous patch.
- **Redshift mapping:** Whether redshift of photons between emission and observation follows the standard geometric relation:
```text
1+z ≈ a(t_obs)/a(t_emit)    (in a homogeneous regime),
```
with corrections computable from the reconstructed metric.

- **CMB temporal interpretation (structural):** Whether the reconstructed family admits a consistent time-ordering and redshift history for free-streaming radiation.

#### 13.3.5 Failure modes
- The reconstruction cannot produce a stable approximately homogeneous/isotropic patch for any admissible `{E_i}` and realistic `ω_s`.
- The derived `a(t)` is not robust under smoothing scale `L` (violating continuum stability).
- The clock mapping `β(ρ)` must be tuned non-universally across the patch to reproduce consistent redshift/expansion diagnostics, violating parameter discipline.

---

### 13.4 How these worked examples connect to falsification
These examples define *what to compute* from a candidate instantiation `(A, ω)` and the specified selection rule `E`. The framework is disfavored if, under its own constraints (symmetry commutation, retention, area capacity, and stability), it cannot realize:

- a weak-field isolated-mass regime reproducing redshift and lensing;
- multi-lens weak-field behavior with approximate superposition and shear;
- a homogeneous expanding patch with robust scale-factor diagnostics.

In other words: these examples translate “emergent geometry” from a slogan into a set of explicit computational tests.
