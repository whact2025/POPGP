"""Compare relative-entropy and modular-energy clock-source scaling."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from popgp import Simulator, validation_json
from popgp.diagnostics import fit_power_law
from popgp.information import (
    mix_states,
    modular_energy_delta,
    quantum_relative_entropy,
    von_neumann_entropy,
)


def _diag(*values: float) -> torch.Tensor:
    return torch.diag(torch.tensor(values, dtype=torch.complex128))


results = Path(__file__).parent / "results"
results.mkdir(parents=True, exist_ok=True)

reference = _diag(0.7, 0.3)
excitation = _diag(0.2, 0.8)
epsilons = np.logspace(-5, -2, 16)
weights = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=torch.float64,
)

relative_entropy_values = []
modular_energy_values = []
entropy_changes = []
relative_phi_amplitudes = []
modular_phi_amplitudes = []

for epsilon in epsilons:
    state = mix_states(reference, excitation, float(epsilon))
    relative_entropy = quantum_relative_entropy(state, reference)
    modular_energy = modular_energy_delta(state, reference)
    entropy_change = von_neumann_entropy(state) - von_neumann_entropy(reference)
    relative_entropy_values.append(relative_entropy)
    modular_energy_values.append(modular_energy)
    entropy_changes.append(entropy_change)

    for source_strength, amplitudes in (
        (relative_entropy, relative_phi_amplitudes),
        (abs(modular_energy), modular_phi_amplitudes),
    ):
        source = torch.tensor([0.0, -source_strength, 0.0], dtype=torch.float64)
        phi, _, _, residual = Simulator._solve_clock_constraint(
            weights,
            source,
            mu=0.1,
            zero_mode_policy="subtract_mean",
            normalize_potential=True,
        )
        if residual > 1e-12:
            raise RuntimeError(f"clock constraint residual too large: {residual}")
        amplitudes.append(float(phi.max() - phi.min()))

relative_entropy_values = np.asarray(relative_entropy_values)
modular_energy_values = np.asarray(modular_energy_values)
entropy_changes = np.asarray(entropy_changes)
relative_phi_amplitudes = np.asarray(relative_phi_amplitudes)
modular_phi_amplitudes = np.asarray(modular_phi_amplitudes)

fits = {
    "relative_entropy": fit_power_law(epsilons, relative_entropy_values),
    "modular_energy": fit_power_law(epsilons, np.abs(modular_energy_values)),
    "relative_entropy_phi": fit_power_law(epsilons, relative_phi_amplitudes),
    "modular_energy_phi": fit_power_law(epsilons, modular_phi_amplitudes),
}

hamiltonian = torch.diag(torch.tensor([0.0, 1.0, 2.0], dtype=torch.complex128))
thermal_weights = torch.softmax(
    -torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64), dim=0
)
kms_reference = torch.diag(thermal_weights.to(torch.complex128))
pure_middle = _diag(0.0, 1.0, 0.0)
mixed_extremes = _diag(0.5, 0.0, 0.5)


def _state_control(state: torch.Tensor) -> dict[str, float]:
    return {
        "energy": float(torch.trace(state @ hamiltonian).real.item()),
        "entropy": von_neumann_entropy(state),
        "relative_entropy": quantum_relative_entropy(state, kms_reference),
        "modular_energy": modular_energy_delta(state, kms_reference),
    }


pure_control = _state_control(pure_middle)
mixed_control = _state_control(mixed_extremes)

print("Perturbative source-law fits")
for name, fit in fits.items():
    print(
        f"  {name}: slope={fit.slope:.6f}, residual scale={fit.slope_residual_scale:.6f}, "
        f"R^2={fit.r_squared:.8f}"
    )
print("Equal-energy KMS control")
print(f"  pure:  {pure_control}")
print(f"  mixed: {mixed_control}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].loglog(epsilons, relative_entropy_values, "o-", label="D(rho||sigma)")
axes[0].loglog(
    epsilons,
    np.abs(modular_energy_values),
    "s-",
    label="|Delta<K_sigma>|",
)
axes[0].set_xlabel("Mixture amplitude epsilon")
axes[0].set_ylabel("Candidate source magnitude")
axes[0].set_title("Candidate source scaling")
axes[0].grid(True, which="both", linestyle=":", alpha=0.5)
axes[0].legend()

axes[1].loglog(
    epsilons,
    relative_phi_amplitudes,
    "o-",
    label="Phi amplitude from relative entropy",
)
axes[1].loglog(
    epsilons,
    modular_phi_amplitudes,
    "s-",
    label="Phi amplitude from modular energy",
)
axes[1].set_xlabel("Mixture amplitude epsilon")
axes[1].set_ylabel("max(Phi)-min(Phi)")
axes[1].set_title("Linear graph solver inherits source scaling")
axes[1].grid(True, which="both", linestyle=":", alpha=0.5)
axes[1].legend()
fig.tight_layout()
fig.savefig(results / "source_scaling.png", dpi=150)


def _fit_dict(name: str) -> dict[str, float]:
    fit = fits[name]
    return {
        "slope": fit.slope,
        "intercept": fit.intercept,
        "slope_residual_scale": fit.slope_residual_scale,
        "r_squared": fit.r_squared,
    }


equal_energy = abs(pure_control["energy"] - mixed_control["energy"]) < 1e-12
equal_modular = abs(pure_control["modular_energy"] - mixed_control["modular_energy"]) < 1e-12
different_relative = (
    abs(pure_control["relative_entropy"] - mixed_control["relative_entropy"]) > 0.1
)
modular_coefficient = modular_energy_delta(excitation, reference)
modular_identity_error = float(
    np.max(np.abs(modular_energy_values - epsilons * modular_coefficient))
)
relative_solver_ratios = relative_phi_amplitudes / relative_entropy_values
modular_solver_ratios = modular_phi_amplitudes / np.abs(modular_energy_values)
solver_homogeneity_spread = float(
    max(np.ptp(relative_solver_ratios), np.ptp(modular_solver_ratios))
)

report = {
    "example": "source_law",
    "framework_version": "1.0-submission-draft",
    "package_version": "0.1.0",
    "scientific_status": "candidate_comparison_with_negative_result",
    "config": {
        "epsilons": epsilons.tolist(),
        "reference": torch.diagonal(reference).real.tolist(),
        "excitation": torch.diagonal(excitation).real.tolist(),
        "clock_graph": "three_node_path",
        "mu": 0.1,
    },
    "measurements": {
        "relative_entropy": relative_entropy_values.tolist(),
        "modular_energy": modular_energy_values.tolist(),
        "entropy_change": entropy_changes.tolist(),
        "relative_entropy_phi_amplitude": relative_phi_amplitudes.tolist(),
        "modular_energy_phi_amplitude": modular_phi_amplitudes.tolist(),
        "fits": {name: _fit_dict(name) for name in fits},
        "equal_energy_control": {
            "pure_middle": pure_control,
            "mixed_extremes": mixed_control,
        },
    },
    "checks": [
        {
            "name": "relative_entropy_is_quadratic",
            "criterion": "abs(slope-2) < 0.02",
            "value": _fit_dict("relative_entropy"),
            "passed": abs(fits["relative_entropy"].slope - 2.0) < 0.02,
        },
        {
            "name": "affine_modular_linearity_identity_regression",
            "criterion": "max|DeltaK(epsilon)-epsilon*DeltaK(1)| < 1e-12",
            "value": {
                "fit": _fit_dict("modular_energy"),
                "max_absolute_identity_error": modular_identity_error,
            },
            "passed": modular_identity_error < 1e-12,
        },
        {
            "name": "linear_solver_homogeneity_identity_regression",
            "criterion": (
                "Phi/source amplitude ratio is constant and relative-entropy Phi "
                "retains the measured quadratic order"
            ),
            "value": {
                "relative_entropy_phi": _fit_dict("relative_entropy_phi"),
                "modular_energy_phi": _fit_dict("modular_energy_phi"),
                "max_ratio_spread": solver_homogeneity_spread,
            },
            "passed": (
                abs(fits["relative_entropy_phi"].slope - 2.0) < 0.02
                and solver_homogeneity_spread < 1e-10
            ),
        },
        {
            "name": "equal_energy_entropy_confound",
            "criterion": (
                "equal energy and modular energy, but different relative entropy"
            ),
            "value": {
                "equal_energy": equal_energy,
                "equal_modular_energy": equal_modular,
                "different_relative_entropy": different_relative,
            },
            "passed": equal_energy and equal_modular and different_relative,
        },
    ],
    "conclusion": {
        "raw_relative_entropy_linear_source": "falsified_in_tested_regime",
        "modular_energy": "affine_linearity_identity_not_a_falsification_test",
    },
    "artifacts": ["results/source_scaling.png", "results/validation.json"],
}
report["overall_pass"] = all(check["passed"] for check in report["checks"])
(results / "validation.json").write_text(
    validation_json(report) + "\n", encoding="utf-8"
)
print(f"Saved: {results / 'source_scaling.png'}")
print(f"Saved: {results / 'validation.json'}")
