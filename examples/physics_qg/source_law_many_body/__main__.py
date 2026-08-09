"""Test localized modular-energy response in an exact finite KMS chain."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from popgp import Simulator, SimulatorConfig, validation_json
from popgp.backend import ExactBackend
from popgp.diagnostics import fit_power_law, fit_quadratic_asymptote
from popgp.information import (
    finite_gibbs_state,
    modular_energy_delta,
    quantum_relative_entropy,
    von_neumann_entropy,
)


def _expectation_delta(
    state: torch.Tensor,
    reference: torch.Tensor,
    observable: torch.Tensor,
) -> float:
    return float(torch.trace((state - reference) @ observable).real.item())


def _fit_dict(fit) -> dict[str, float]:
    return {
        "slope": fit.slope,
        "intercept": fit.intercept,
        "slope_standard_error": fit.slope_standard_error,
        "r_squared": fit.r_squared,
    }


def _asymptote_dict(fit) -> dict[str, float]:
    return {
        "coefficient": fit.coefficient,
        "coefficient_standard_error": fit.coefficient_standard_error,
        "linear_correction": fit.linear_correction,
        "normalized_rmse": fit.normalized_rmse,
    }


def _order_sensitivity() -> list[dict]:
    """Sweep a non-affine KMS family over the declared finite regime."""
    epsilons = np.logspace(-5, -3, 9)
    sensitivity = []
    cases = [
        (5, family, beta)
        for family in ("heisenberg", "ising")
        for beta in (0.3, 1.0, 2.0, 3.0)
    ]
    cases.extend([(3, "heisenberg", 1.0), (7, "heisenberg", 1.0)])
    for n_sites, family, beta in cases:
        config = SimulatorConfig.for_chain(
            n=n_sites,
            beta=beta,
            boundary="open",
            hamiltonian=family,
        )
        backend = ExactBackend(config)
        hamiltonian = backend.build_hamiltonian()
        reference = backend.prepare_state()
        perturbation = -backend.build_local_energy_operators()[n_sites // 2]
        relative_entropy = []
        modular_energy = []
        for epsilon in epsilons:
            state = finite_gibbs_state(
                hamiltonian + float(epsilon) * perturbation,
                beta,
            )
            relative_entropy.append(quantum_relative_entropy(state, reference))
            modular_energy.append(abs(modular_energy_delta(state, reference)))
        full_asymptote = fit_quadratic_asymptote(
            epsilons, np.asarray(relative_entropy)
        )
        lower_asymptote = fit_quadratic_asymptote(
            epsilons[:6], np.asarray(relative_entropy[:6])
        )
        modular_fit = fit_power_law(epsilons, np.asarray(modular_energy))
        coefficient_difference = abs(
            full_asymptote.coefficient - lower_asymptote.coefficient
        )
        coefficient_tolerance = 3.0 * np.hypot(
            full_asymptote.coefficient_standard_error,
            lower_asymptote.coefficient_standard_error,
        )
        slope_deviation = abs(modular_fit.slope - 1.0)
        slope_tolerance = 5.0 * modular_fit.slope_standard_error
        sensitivity.append(
            {
                "n_sites": n_sites,
                "hamiltonian": family,
                "beta": beta,
                "epsilons": epsilons.tolist(),
                "relative_entropy_full_asymptote": _asymptote_dict(
                    full_asymptote
                ),
                "relative_entropy_lower_asymptote": _asymptote_dict(
                    lower_asymptote
                ),
                "coefficient_difference": coefficient_difference,
                "three_sigma_coefficient_tolerance": coefficient_tolerance,
                "modular_energy_fit": _fit_dict(modular_fit),
                "slope_deviation": slope_deviation,
                "five_sigma_slope_tolerance": slope_tolerance,
                "passed": (
                    full_asymptote.coefficient > 0.0
                    and coefficient_difference <= coefficient_tolerance
                    and slope_deviation <= slope_tolerance
                ),
            }
        )
    return sensitivity


def main() -> None:
    results = Path(__file__).parent / "results"
    results.mkdir(parents=True, exist_ok=True)

    config = SimulatorConfig.for_chain(
        n=5,
        beta=1.3,
        boundary="open",
        hamiltonian="heisenberg",
    )
    backend = ExactBackend(config)
    hamiltonian = backend.build_hamiltonian()
    reference = backend.prepare_state()
    local_energy = backend.build_local_energy_operators()
    pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    quench_generator = backend.site_operator(pauli_x, site=2)
    quench_excitation = quench_generator @ reference @ quench_generator.conj().T
    perturbation = -local_energy[2]

    n_sites = config.substrate.n_qubits
    weights = torch.zeros((n_sites, n_sites), dtype=torch.float64)
    for i, j in backend.build_edges():
        weights[i, j] = weights[j, i] = 1.0

    epsilons = np.logspace(-5, -3, 9)
    diagnostic_epsilon = 0.01
    relative_entropy = []
    modular_energy = []
    entropy_change = []
    total_energy = []
    local_energy_profiles = []
    potential_amplitudes = []

    for epsilon in epsilons:
        state = finite_gibbs_state(
            hamiltonian + float(epsilon) * perturbation,
            config.substrate.beta,
        )
        relative_entropy.append(quantum_relative_entropy(state, reference))
        modular_energy.append(modular_energy_delta(state, reference))
        entropy_change.append(
            von_neumann_entropy(state) - von_neumann_entropy(reference)
        )
        total_energy.append(_expectation_delta(state, reference, hamiltonian))
        profile = np.asarray(
            [_expectation_delta(state, reference, term) for term in local_energy]
        )
        local_energy_profiles.append(profile)
        phi, _, _, residual = Simulator._solve_clock_constraint(
            weights,
            -config.substrate.beta * torch.from_numpy(profile),
            mu=0.1,
            zero_mode_policy="subtract_mean",
            normalize_potential=True,
        )
        if residual > 1e-12:
            raise RuntimeError(f"clock constraint residual too large: {residual}")
        potential_amplitudes.append(float(phi.max() - phi.min()))

    relative_entropy = np.asarray(relative_entropy)
    modular_energy = np.asarray(modular_energy)
    entropy_change = np.asarray(entropy_change)
    total_energy = np.asarray(total_energy)
    local_energy_profiles = np.asarray(local_energy_profiles)
    potential_amplitudes = np.asarray(potential_amplitudes)
    fits = {
        "relative_entropy": fit_power_law(epsilons, relative_entropy),
        "modular_energy": fit_power_law(epsilons, np.abs(modular_energy)),
        "total_energy": fit_power_law(epsilons, np.abs(total_energy)),
        "potential_amplitude": fit_power_law(epsilons, potential_amplitudes),
    }
    relative_asymptote = fit_quadratic_asymptote(epsilons, relative_entropy)
    relative_lower_asymptote = fit_quadratic_asymptote(
        epsilons[:6], relative_entropy[:6]
    )
    coefficient_difference = abs(
        relative_asymptote.coefficient - relative_lower_asymptote.coefficient
    )
    coefficient_tolerance = 3.0 * np.hypot(
        relative_asymptote.coefficient_standard_error,
        relative_lower_asymptote.coefficient_standard_error,
    )
    midpoint_state = finite_gibbs_state(
        hamiltonian + (diagnostic_epsilon / 2.0) * perturbation,
        config.substrate.beta,
    )
    diagnostic_state = finite_gibbs_state(
        hamiltonian + diagnostic_epsilon * perturbation,
        config.substrate.beta,
    )
    affine_midpoint = 0.5 * (reference + diagnostic_state)
    nonaffine_deviation = float(
        torch.linalg.matrix_norm(midpoint_state - affine_midpoint).item()
    )

    evolution_times = np.asarray([0.0, 0.2, 0.5, 1.0, 2.0])
    evolved_profiles = []
    evolved_total_energy = []
    for time in evolution_times:
        state = backend.evolve(quench_excitation, dt=float(time))
        evolved_profiles.append(
            [_expectation_delta(state, reference, term) for term in local_energy]
        )
        evolved_total_energy.append(
            _expectation_delta(state, reference, hamiltonian)
        )
    evolved_profiles = np.asarray(evolved_profiles)
    evolved_total_energy = np.asarray(evolved_total_energy)

    unitary_amplitudes = np.logspace(-4, -2, 9)
    unitary_relative_entropy = []
    unitary_modular_energy = []
    unitary_entropy_change = []
    for amplitude in unitary_amplitudes:
        unitary = torch.linalg.matrix_exp(
            -1j * float(amplitude) * quench_generator
        )
        state = unitary @ reference @ unitary.conj().T
        unitary_relative_entropy.append(
            quantum_relative_entropy(state, reference)
        )
        unitary_modular_energy.append(modular_energy_delta(state, reference))
        unitary_entropy_change.append(
            von_neumann_entropy(state) - von_neumann_entropy(reference)
        )
    unitary_relative_entropy = np.asarray(unitary_relative_entropy)
    unitary_modular_energy = np.asarray(unitary_modular_energy)
    unitary_entropy_change = np.asarray(unitary_entropy_change)
    unitary_relative_fit = fit_power_law(
        unitary_amplitudes, unitary_relative_entropy
    )
    unitary_modular_fit = fit_power_law(
        unitary_amplitudes, unitary_modular_energy
    )
    unitary_identity_error = float(
        np.max(np.abs(unitary_relative_entropy - unitary_modular_energy))
    )
    unitary_entropy_error = float(np.max(np.abs(unitary_entropy_change)))
    unitary_identity_atol = float(
        np.finfo(float).eps * reference.shape[0]
    )

    sensitivity = _order_sensitivity()
    ising_config = SimulatorConfig.for_chain(
        n=5,
        beta=1.0,
        boundary="open",
        hamiltonian="ising",
    )
    ising_backend = ExactBackend(ising_config)
    ising_reference = ising_backend.prepare_state()
    ising_energy = ising_backend.build_local_energy_operators()
    ising_unitary = ising_backend.site_operator(pauli_x, site=2)
    ising_excitation = ising_unitary @ ising_reference @ ising_unitary.conj().T
    ising_evolved = ising_backend.evolve(ising_excitation, dt=1.0)
    ising_initial_profile = np.asarray(
        [
            _expectation_delta(ising_excitation, ising_reference, term)
            for term in ising_energy
        ]
    )
    ising_evolved_profile = np.asarray(
        [
            _expectation_delta(ising_evolved, ising_reference, term)
            for term in ising_energy
        ]
    )
    ising_profile_change = float(
        np.max(np.abs(ising_evolved_profile - ising_initial_profile))
    )

    final_state = diagnostic_state
    final_profile = torch.tensor(
        [
            _expectation_delta(final_state, reference, term)
            for term in local_energy
        ],
        dtype=torch.float64,
    )
    pipeline = Simulator(config)
    cells = [[site] for site in range(n_sites)]
    config.pi_time.source_model = "negative_modular_energy_candidate"
    reduced_modular_source = pipeline._compute_source_term(
        final_state,
        cells,
        config.pi_time,
        reference_state=reference,
    )
    config.pi_time.source_model = "negative_kms_energy_density_candidate"
    kms_energy_density_source = pipeline._compute_source_term(
        final_state,
        cells,
        config.pi_time,
        reference_state=reference,
    )
    expected_kms_density = -config.substrate.beta * final_profile
    kms_density_match_error = float(
        torch.max(torch.abs(kms_energy_density_source - expected_kms_density)).item()
    )
    phi, effective_source, source_background, residual = (
        Simulator._solve_clock_constraint(
            weights,
            kms_energy_density_source,
            mu=0.1,
            zero_mode_policy="subtract_mean",
            normalize_potential=True,
        )
    )
    redshift = Simulator.gravitational_redshift(
        phi_emitter=phi[2], phi_observer=phi[0]
    )
    clock_rate_ratio = float(torch.exp(phi[2] - phi[0]).item())

    kms_identity_error = float(
        np.max(np.abs(modular_energy - config.substrate.beta * total_energy))
    )
    first_law_identity_error = float(
        np.max(np.abs(relative_entropy - (modular_energy - entropy_change)))
    )
    decomposition_error = float(
        np.max(np.abs(local_energy_profiles.sum(axis=1) - total_energy))
    )
    conservation_drift = float(np.ptp(evolved_total_energy))
    initial_outside_fraction = float(
        np.abs(evolved_profiles[0, [0, 4]]).sum()
        / np.abs(evolved_profiles[0]).sum()
    )
    evolved_outside_fraction = float(
        np.abs(evolved_profiles[3, [0, 4]]).sum()
        / np.abs(evolved_profiles[3]).sum()
    )

    print("Localized non-affine KMS source-law diagnostics")
    for name, fit in fits.items():
        print(
            f"  {name}: slope={fit.slope:.6f} +/- "
            f"{fit.slope_standard_error:.6f}, R^2={fit.r_squared:.8f}"
        )
    print(f"  max |Delta<K>-beta Delta<E>|: {kms_identity_error:.3e}")
    print(
        "  quadratic coefficient (full/lower window): "
        f"{relative_asymptote.coefficient:.6e} / "
        f"{relative_lower_asymptote.coefficient:.6e}"
    )
    print(f"  non-affine midpoint deviation: {nonaffine_deviation:.3e}")
    print(
        "  isospectral unitary slopes (D/DeltaK): "
        f"{unitary_relative_fit.slope:.6f} / {unitary_modular_fit.slope:.6f}"
    )
    print(f"  energy-conservation drift: {conservation_drift:.3e}")
    print(f"  evolved endpoint energy fraction (t=1): {evolved_outside_fraction:.6f}")
    print(f"  commuting Ising profile change (t=1): {ising_profile_change:.3e}")
    print(
        "  reduced-state modular / KMS-density source norms: "
        f"{torch.linalg.vector_norm(reduced_modular_source).item():.3e} / "
        f"{torch.linalg.vector_norm(kms_energy_density_source).item():.3e}"
    )
    print(f"  center-to-edge clock-rate ratio: {clock_rate_ratio:.9f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].loglog(epsilons, relative_entropy, "o-", label="D(rho||sigma)")
    axes[0].loglog(epsilons, modular_energy, "s-", label="Delta<K_sigma>")
    axes[0].loglog(epsilons, total_energy, "^-", label="Delta<E>")
    axes[0].set_xlabel("KMS perturbation amplitude epsilon")
    axes[0].set_ylabel("Response magnitude")
    axes[0].set_title("Non-affine KMS response")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0].legend()

    sites = np.arange(n_sites)
    for index, time in enumerate(evolution_times):
        axes[1].plot(sites, evolved_profiles[index], "o-", label=f"t={time:g}")
    axes[1].set_xlabel("Microscopic chain site")
    axes[1].set_ylabel("Local energy change")
    axes[1].set_title("Conserved profile spreading")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend()

    axes[2].plot(sites, phi.numpy(), "o-", color="tab:purple")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Microscopic chain site")
    axes[2].set_ylabel("Normalized Phi")
    axes[2].set_title("Diagnostic potential at epsilon=0.01")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    figure_path = results / "many_body_source.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    linear_response_within_uncertainty = all(
        abs(fits[name].slope - 1.0)
        <= 5.0 * fits[name].slope_standard_error
        for name in ("modular_energy", "total_energy", "potential_amplitude")
    )
    unitary_quadratic_within_uncertainty = all(
        abs(fit.slope - 2.0) <= 5.0 * fit.slope_standard_error
        for fit in (unitary_relative_fit, unitary_modular_fit)
    )

    checks = [
        {
            "name": "nonaffine_kms_response_orders",
            "criterion": (
                "D/epsilon^2 has a positive nested-window limit within three "
                "combined standard errors; linear-observable slopes contain 1 "
                "within five fit standard errors; family is detectably non-affine"
            ),
            "value": {
                "descriptive_power_law_fits": {
                    name: _fit_dict(fit) for name, fit in fits.items()
                },
                "relative_entropy_full_asymptote": _asymptote_dict(
                    relative_asymptote
                ),
                "relative_entropy_lower_asymptote": _asymptote_dict(
                    relative_lower_asymptote
                ),
                "coefficient_difference": coefficient_difference,
                "three_sigma_coefficient_tolerance": coefficient_tolerance,
                "nonaffine_midpoint_deviation": nonaffine_deviation,
            },
            "passed": (
                relative_asymptote.coefficient > 0.0
                and coefficient_difference <= coefficient_tolerance
                and linear_response_within_uncertainty
                and nonaffine_deviation > 1e-9
            ),
        },
        {
            "name": "kms_and_local_decomposition_identities",
            "criterion": "all identity residuals below 5e-13",
            "value": {
                "kms_identity_error": kms_identity_error,
                "first_law_identity_error": first_law_identity_error,
                "local_decomposition_error": decomposition_error,
            },
            "passed": max(
                kms_identity_error,
                first_law_identity_error,
                decomposition_error,
            )
            < 5e-13,
        },
        {
            "name": "localized_energy_is_conserved_and_spreads",
            "criterion": (
                "conservation drift below 1e-12, initial endpoint fraction below "
                "1e-12, and t=1 endpoint fraction above 0.05"
            ),
            "value": {
                "conservation_drift": conservation_drift,
                "initial_endpoint_fraction": initial_outside_fraction,
                "t1_endpoint_fraction": evolved_outside_fraction,
            },
            "passed": (
                conservation_drift < 1e-12
                and initial_outside_fraction < 1e-12
                and evolved_outside_fraction > 0.05
            ),
        },
        {
            "name": "nonaffine_kms_parameter_sensitivity",
            "criterion": (
                "nested-window quadratic coefficients agree within three combined "
                "standard errors and modular slopes contain 1 within five fit "
                "standard errors over the declared beta<=3 finite sweep"
            ),
            "value": sensitivity,
            "passed": all(case["passed"] for case in sensitivity),
        },
        {
            "name": "isospectral_unitary_family_qualifier",
            "criterion": (
                "D and Delta<K> are equal and quadratic while DeltaS=0 for the "
                "declared local unitary family"
            ),
            "value": {
                "relative_entropy_fit": _fit_dict(unitary_relative_fit),
                "modular_energy_fit": _fit_dict(unitary_modular_fit),
                "max_D_minus_modular_energy": unitary_identity_error,
                "max_entropy_change": unitary_entropy_error,
                "dimension_scaled_float64_tolerance": unitary_identity_atol,
            },
            "passed": (
                unitary_quadratic_within_uncertainty
                and unitary_identity_error <= unitary_identity_atol
                and unitary_entropy_error <= unitary_identity_atol
            ),
        },
        {
            "name": "spreading_requires_noncommuting_dynamics",
            "criterion": "commuting Ising control profile change below 1e-12",
            "value": {"maximum_profile_change_at_t1": ising_profile_change},
            "passed": ising_profile_change < 1e-12,
        },
        {
            "name": "pipeline_reduced_modular_blindness_and_density_repair",
            "criterion": (
                "reduced-state source norm below 1e-12, KMS energy-density source "
                "norm above 1e-3, and density implementation error below 1e-14"
            ),
            "value": {
                "reduced_modular_source": reduced_modular_source.tolist(),
                "reduced_modular_source_norm": float(
                    torch.linalg.vector_norm(reduced_modular_source).item()
                ),
                "kms_energy_density_source": kms_energy_density_source.tolist(),
                "kms_energy_density_source_norm": float(
                    torch.linalg.vector_norm(kms_energy_density_source).item()
                ),
                "kms_density_match_error": kms_density_match_error,
            },
            "passed": (
                torch.linalg.vector_norm(reduced_modular_source).item() < 1e-12
                and torch.linalg.vector_norm(kms_energy_density_source).item() > 1e-3
                and kms_density_match_error < 1e-14
            ),
        },
        {
            "name": "negative_energy_candidate_has_slower_source_clock",
            "criterion": "center Phi below edge Phi, clock-rate ratio below 1, redshift positive",
            "value": {
                "phi": phi.tolist(),
                "effective_source": effective_source.tolist(),
                "source_background": source_background,
                "constraint_residual": residual,
                "center_to_edge_clock_rate_ratio": clock_rate_ratio,
                "edge_observed_redshift": redshift,
            },
            "passed": (
                int(torch.argmin(phi).item()) == 2
                and clock_rate_ratio < 1.0
                and redshift > 0.0
                and residual < 1e-12
            ),
        },
    ]

    report = {
        "example": "source_law_many_body",
        "framework_version": "1.0-submission-draft",
        "package_version": "0.1.0",
        "scientific_status": "feasible_candidate_not_validated_physical_law",
        "hypothesis": (
            "For the non-affine family rho(epsilon) proportional to "
            "exp[-beta(H+epsilon V)], a localized finite-chain perturbation has "
            "nonzero first-order modular/energy susceptibility while relative "
            "entropy converges quadratically. The order is family-specific."
        ),
        "config": {
            "n_sites": n_sites,
            "hamiltonian": config.substrate.hamiltonian,
            "topology": config.substrate.topology,
            "interaction_graph_supplied": True,
            "boundary": config.substrate.boundary,
            "beta": config.substrate.beta,
            "declared_sensitivity_beta_max": 3.0,
            "center_site": 2,
            "response_family": "non-affine finite KMS family of H + epsilon V",
            "perturbation": "V = -h_center",
            "dynamics_control": "full local Pauli-X quench",
            "isospectral_control": "exp(-i theta X_center) rho exp(i theta X_center)",
            "local_energy_convention": "split each pair term equally between endpoints",
            "epsilons": epsilons.tolist(),
            "diagnostic_source_epsilon": diagnostic_epsilon,
            "evolution_times": evolution_times.tolist(),
            "clock_mu": 0.1,
            "clock_zero_mode_policy": "subtract_mean",
        },
        "measurements": {
            "relative_entropy": relative_entropy.tolist(),
            "modular_energy": modular_energy.tolist(),
            "entropy_change": entropy_change.tolist(),
            "total_energy_change": total_energy.tolist(),
            "local_energy_profiles": local_energy_profiles.tolist(),
            "potential_amplitudes": potential_amplitudes.tolist(),
            "fits": {name: _fit_dict(fit) for name, fit in fits.items()},
            "relative_entropy_full_asymptote": _asymptote_dict(
                relative_asymptote
            ),
            "relative_entropy_lower_asymptote": _asymptote_dict(
                relative_lower_asymptote
            ),
            "nonaffine_midpoint_deviation": nonaffine_deviation,
            "evolved_local_energy_profiles": evolved_profiles.tolist(),
            "evolved_total_energy": evolved_total_energy.tolist(),
            "order_sensitivity": sensitivity,
            "isospectral_unitary_control": {
                "amplitudes": unitary_amplitudes.tolist(),
                "relative_entropy": unitary_relative_entropy.tolist(),
                "modular_energy": unitary_modular_energy.tolist(),
                "entropy_change": unitary_entropy_change.tolist(),
                "relative_entropy_fit": _fit_dict(unitary_relative_fit),
                "modular_energy_fit": _fit_dict(unitary_modular_fit),
            },
            "commuting_ising_control": {
                "initial_profile": ising_initial_profile.tolist(),
                "t1_profile": ising_evolved_profile.tolist(),
                "maximum_profile_change": ising_profile_change,
            },
            "pipeline_source_comparison": {
                "reduced_modular_source": reduced_modular_source.tolist(),
                "kms_energy_density_source": kms_energy_density_source.tolist(),
                "kms_density_match_error": kms_density_match_error,
            },
        },
        "checks": checks,
        "overall_pass": all(check["passed"] for check in checks),
        "conclusion": {
            "raw_relative_entropy_linear_source": "falsified_in_tested_regime",
            "affine_mixture_linear_response": "analytic_identity_not_a_test",
            "nonaffine_kms_response": "first_order_in_declared_finite_beta_window",
            "isospectral_unitary_response": "quadratic_with_D_equal_to_DeltaK",
            "reduced_state_modular_localization": "blind_in_symmetric_kms_control",
            "kms_energy_density_localization": "feasible_microscopic_candidate",
            "profile_spreading": "dynamics_dependent_not_universal",
            "remaining_requirements": [
                "state-independent localization prescription",
                "covariant conservation law",
                "interaction-family and size robustness",
                "continuum/refinement behavior",
                "independent operational clock observable",
            ],
        },
        "artifacts": [
            "results/many_body_source.png",
            "results/validation.json",
        ],
    }
    validation_path = results / "validation.json"
    validation_path.write_text(validation_json(report) + "\n", encoding="utf-8")
    print(f"Saved: {figure_path}")
    print(f"Saved: {validation_path}")


if __name__ == "__main__":
    main()
