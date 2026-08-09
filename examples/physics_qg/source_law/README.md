# Source-law linear-response experiment

This controlled experiment compares two vacuum-relative candidates under

`rho(epsilon) = (1-epsilon) sigma + epsilon rho_exc`.

It measures relative entropy, modular-energy variation, entropy change, and the
clock-potential amplitude produced by each candidate on the same three-node graph.

```text
uv run python -m examples.physics_qg.source_law
```

Expected and observed behavior:

- `D(rho(epsilon)||sigma)` begins at order epsilon squared;
- `Delta<K_sigma>` is exactly proportional to epsilon because it is linear in rho;
- the graph solver preserves source scaling by linearity; and
- a KMS qutrit control has equal-energy states with equal modular energy but
  different entropy and relative entropy.

![Source scaling](results/source_scaling.png)

This is a negative result for raw relative entropy as a standalone linear mass
source in the tested regime. The modular-energy and graph-solver unit slopes are
analytic identity regressions for this affine family, not a passed first-order
diagnostic. A non-affine response family, localization, conservation, covariance, and
many-body refinement remain open.
