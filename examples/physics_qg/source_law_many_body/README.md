# Localized many-body KMS source-law experiment

This exact five-site Heisenberg-chain experiment tests whether a localized microscopic
energy-density candidate has a nonzero susceptibility in a genuinely non-affine KMS
family, rather than inheriting unit slope from an affine mixture by construction.

```text
uv run python -m examples.physics_qg.source_law_many_body
```

The response family is
`rho(epsilon) proportional to exp[-beta(H + epsilon V)]`, with localized
`V = -h_center`. It measures:

- global relative entropy, modular energy, entropy change, and physical energy;
- an explicit symmetric site-energy decomposition that sums to the Hamiltonian;
- conservation and spreading of that site-energy profile under exact evolution; and
- the sign of a diagnostic graph potential sourced by negative local energy change.

The quadratic claim is checked by fitting `D/epsilon^2` to a finite intercept on
nested windows. The two intercepts must agree within three combined fit standard
errors; linear-response slopes must contain 1 within five fit standard errors. The
check is repeated for Heisenberg and Ising chains through `beta = 3` and for three
small odd Heisenberg-chain sizes.

Response order is family-specific. An isospectral local-unitary control has
`Delta S = 0` and `D = Delta<K> = beta Delta<E>` with quadratic order. A separate
full local quench supplies the profile-spreading experiment; a commuting Ising control
retains a stationary profile, showing that spreading is dynamics-dependent.

The experiment also preserves an integration-level negative result: the earlier
one-site reduced-state modular source is numerically zero for the symmetric KMS
reference. A distinct exact-backend candidate based on `−β Δ⟨h_i⟩` recovers the
audited local-energy profile and sums to minus the global modular-energy change. This is an
explicit repair path, not a silent replacement or a claim of uniqueness.

The interaction chain is supplied by the Hamiltonian, and the local-energy split is a
declared microscopic convention. A successful result therefore supports only the
feasibility of a localized test object in the declared finite KMS family. It does not
derive a family-independent first-order law, gravitational source, covariance, or a
continuum limit.

![Many-body source-law diagnostics](results/many_body_source.png)
