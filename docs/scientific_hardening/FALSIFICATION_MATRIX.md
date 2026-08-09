# Falsification matrix

| Hypothesis | Controlled family and observable | Expected behavior | Failure threshold | Confounders / requirements | Current readiness |
|---|---|---|---|---|---|
| Linear source response | `ρ(ε)=(1-ε)ρ_vac+ερ_exc`; source and Φ amplitude | Physical weak source begins at O(ε) | Fitted slope excludes 1 after numerical/systematic errors | Faithful KMS reference, conserved charges, solver linearity | Pilot complete: relative entropy slope ≈2; modular energy ≈1. Raw RE fails. |
| Equal energy, different entropy | KMS reference; states matched in energy/charges/localization | Far-field monopole insensitive to entropy alone | Source differs beyond matched-energy tolerance without additional physical charge | Exact energy matching, finite-size degeneracy | Analytic finite control complete; raw RE differs. Larger localized control needed. |
| Blind topology | Chains, square/triangular/3D lattices; edge precision/recall and distance distortion | High recovery without ground-truth access | Pre-register by family; current target P,R≥0.9 | Temperature, size, threshold, long-range correlations | Chain/grid pilot and permutation test pass. Other families open. |
| Non-geometric controls | Bell pairs, expanders, random regular and shuffled MI matrices | No stable low-D manifold declaration | False low-D selection or “separable” edge claim across perturbations | MDS can visually flatten arbitrary finite graphs | Uniform and disjoint-Bell controls are non-separable; Petersen is non-geometric at default but exposes λ sensitivity. |
| Parameter robustness | Sweep β, I0 multiplier, gap ratio, λ, probes, seed | Qualitative result survives a predeclared region | Target appears only at isolated/post-selected values | Correlated hyperparameters and phase transitions | Five probe seeds preserve chain partition; systematic sweeps open. |
| Refinement convergence | Same physical geometry at growing N | Observables converge with stated order/error | No stable window or drifting inferred dimension | Boundary conditions and scale matching | Not currently feasible with exact backend alone. |
| Clock/spatial consistency | Independent clock simulation versus solved Φ | `dτ` contrasts match predicted `exp(Φ)` | Sign/magnitude mismatch beyond error | `dt` currently substitutes for action distance | Sign convention tested; independent clock observable absent. |
| Lorentz recovery | Excitation dispersion by direction and boost-like transformations | Common limiting speed and vanishing IR anisotropy | Persistent preferred-frame signal | Finite lattice artifacts, gauge choice | Not implemented. |
| Causal completeness | Controlled high-curvature evolution and causal trajectories | Deterministic extendibility or defined phase transition | Evolution terminates or trajectories cannot extend | Must identify modified singularity-theorem assumption | Not implemented. |
| Closure / conservation | Regge geometric tensor, source tensor, divergence residual | Closure and discrete Bianchi residual converge to zero | Residual fails to decrease under refinement | Boundary terms, dual volumes, gauge | Not implemented. |

Negative results must remain in version control with the same prominence as passes.
Changing the source law, inference rule, or acceptance threshold requires a recorded
decision and a rerun of all relevant controls.
