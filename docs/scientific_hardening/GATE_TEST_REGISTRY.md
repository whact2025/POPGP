# Acceptance-gate mutation registry

Every active acceptance gate and every gate retaining a deliberate negative result
has a stable ID and at least one executable negative control. A passing identity
regression is not an acceptance gate. The table records the statistic change that the
test computes; exact floating-point values remain assertions in the cited test.

| Gate ID | Matrix row | Status | Demonstrated negative control or mutation | Statistic before → after |
|---|---|---|---|---|
| `GATE-SOURCE-KMS-RESPONSE` | Source response under a non-affine KMS family | active | `tests/scientific/test_many_body_source_law.py::test_quadratic_gate_rejects_first_order_negative_control` | quadratic gate `passed=True` for an `epsilon^2` control → `False` after adding a first-order term; slope about 2 → about 1 |
| `GATE-ENTROPY-CONFOUND` | Equal energy, different entropy | negative-result control | `tests/scientific/test_source_law_controls.py::test_equal_energy_states_expose_entropy_source_confound` | matched energy remains equal → raw relative-entropy source differs after entropy is changed |
| `GATE-MODULAR-LOCALIZATION` | Modular-energy localization | active with retained negative controls | `tests/scientific/test_many_body_source_law.py::test_profile_spreading_is_not_automatic_in_commuting_ising_control`; `tests/scientific/test_many_body_source_law.py::test_reduced_modular_source_is_blind_but_kms_energy_density_is_not` | noncommuting profile spreads → commuting profile change is roundoff-only; KMS density nonzero → reduced modular source norm below `1e-12` |
| `GATE-BLIND-TOPOLOGY` | Blind topology | active | `tests/scientific/test_topology_recovery.py::test_nonseparable_correlations_are_not_called_identifiable` | separated grid has precision/recall `1/1` → uniform correlations have `connectivity_separable=False` |
| `GATE-NONGEOMETRIC-CONTROLS` | Non-geometric controls | known negative result | `tests/scientific/test_topology_recovery.py::test_disjoint_bell_pairs_are_marked_nonseparable` | nonseparable Bell control → still receives false `D*=1` geometric-candidate status, preserving the gate failure |
| `GATE-PARAMETER-ROBUSTNESS` | Parameter robustness | active finite sweep | `tests/unit/test_diagnostics.py::test_quadratic_asymptote_enforces_absolute_precision_floor`; `tests/scientific/test_many_body_source_law.py::test_quadratic_gate_rejects_first_order_negative_control` | declared finite sweep passes → sub-floor or first-order response fails the same gate |

For numerical gates, the mutation must cross the stated acceptance boundary. Parameter
sweeps should also report the observed range of the gated statistic when that range is
scientifically meaningful. They are not required to vary by more than the acceptance
tolerance: a robustness sweep is meant to stay within that tolerance, so imposing the
opposite condition would reject the very stability being tested.
