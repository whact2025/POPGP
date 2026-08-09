# Theory-to-code gap register

| Component | Required object | Implemented object | Gap and risk |
|---|---|---|---|
| Substrate | Algebra, state, automorphism/modular flow, representation independence | Finite qubits, chosen Heisenberg/Ising Hamiltonian, thermal density matrix | Interaction adjacency is microscopic input; claims of topology-free emergence would be circular. |
| Resolution projection | Conditional expectations selected by a causal local flow and admissibility conditions | Partial traces plus exhaustive equal-block partition search with sampled leakage/drift | No flow, fixed-point theorem, diamond norm, or scale consistency. Global-search success may not survive scaling. |
| Symmetry | Relevant covariant conditional expectations | Sampled global SU(2) equivariance | Spatial rotations do not imply boosts or Lorentz covariance. |
| Capacity | State-independent finite distinguishability principle with a justified scale | MI cut sum and callable inequality check | η is not derived and no refinement law exists. “Area” language is premature. |
| Locality | QCMI/Markov-screened relational neighborhoods | Pairwise MI plus adaptive-gap/MST or k-NN/MST | Pairwise MI can mistake long-range entanglement for proximity. |
| Spectral dimension | Scale plateau and continuum/refinement limit | Peak of finite-graph heat-kernel derivative | A peak on 4–9 nodes is a diagnostic, not a continuum dimension. |
| Embedding | Intrinsic geometry with uncertainty and non-geometric rejection | Graph geodesics and classical MDS | MDS can make many graphs look geometric; control graph tests remain incomplete. |
| Local metric | SPD tangent metric with stable intrinsic neighborhoods | Regularized fit in MDS coordinates with rank/condition/residual diagnostics | Boundary fits can be underdetermined; no intrinsic/refinement convergence evidence. |
| Discrete curvature | Intrinsic simplicial complex, hinge deficits, dual volumes | Integrated 2D embedding-Delaunay proxy with boundary-aware angle deficits; disconnected CUDA experiment | The complex and lengths are embedding proxies; no dual-volume normalization or refinement convergence. |
| Source | Temporally averaged, vacuum-relative quantity with a derived response law and conserved localization | Positive von Neumann placeholder; raw relative-entropy and reduced-modular controls; exact-backend `−βΔ⟨h_i⟩` candidate | Raw relative entropy is quadratic and entropy-confounded; reduced modular energy is blind in the symmetric KMS chain. Affine-mixture unit slopes are analytic identities. A non-affine KMS family has a nonzero first-order susceptibility in the declared β≤3 finite sweep, whereas an isospectral unitary family is quadratic. The source density sums to minus the global modular-energy change; its localization remains graph/decomposition dependent and non-covariant. Temporal averaging, scalable refinement, and physical clock matching are absent. |
| Clock constraint | Well-posed equation, zero modes, sign, normalization, observable clocks | Weighted graph solve with screened/unscreened policies and residual | Numerical solve is sound, but its physical source and coupling are unvalidated. Native `clock.cu` remains identity. |
| Proper time | Operational `dS_act` and clock comparison | Simulation `dt` substituted for `dS_act` | The main observable is not independently constructed. |
| Newtonian limit | Controlled continuum/refinement limit with correct Green function and normalization | Manually injected 3×3 diagnostic source | Two radial shells cannot distinguish a gravitational law. |
| Einstein closure | Regge/continuum geometric tensor matched to a conserved source | A tensor-agnostic mismatch interface only | No validated geometric/source tensors, coupling derivation, or conservation identity; the central claim remains untested. |
| Lorentz recovery | Dispersion, boosts, common limiting speed, anisotropy bounds | None | SU(2) cannot support the claimed conclusion. |
| High curvature | Evolving controlled solution with bounded invariants and causal extendibility | None | Finite capacity alone does not resolve a singularity. |
| Scalable backend | Entangling approximation with controlled errors and MI/QCMI | Pure-product mean-field CUDA evolution; MI explicitly unsupported | The automatic large-N path cannot run the full projection pipeline. |

## Resolved implementation defects in the hardening pass

- The documented grid constructor now selects one-qubit cells and runs directly.
- Hamiltonian family selection no longer silently ignores `ising`.
- Relative entropy preserves its exact support divergence instead of flooring zeros.
- Product-state magnetization is no longer reported as mutual information.
- CUDA edge updates are color-batched by the Python backend to prevent shared-node races.
- MI distance scale defaults cannot collapse the strongest correlated nodes to zero length.
- Reported graph weights contain inferred edges only, not every pair.
- Geometry reports stress separately from its penalized objective and bounds candidate
  dimension by graph size.
- The heat-kernel Laplacian is assembled in float64, including an exact zero mode.
- The clock solve no longer pins an arbitrary node or accepts incompatible unscreened
  sources silently.
- The redshift exponent and negative-well clock sign are consistent in code and tests.

These repairs improve internal validity; they do not close the theoretical gaps above.
