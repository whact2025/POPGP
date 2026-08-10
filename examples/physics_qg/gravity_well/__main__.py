# Copyright (c) 2026 WHACT. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Clock constraint: localized Green-function sign test.

This example validates the finite-graph constraint solver and the proposed
clock/redshift sign convention. It does not validate a physical source law
or derive a Newtonian limit. On a 3x3 Heisenberg grid we:

1. Run the full quantum pipeline to build the MI-weighted graph Laplacian.
2. Inject a small negative localized diagnostic source at the center cell.
3. Solve (Delta_w + mu^2 I) Phi = delta_rho  with small mu > 0.
4. Check that Phi is most negative at the source, clocks there are slower,
   and the emitter-to-boundary redshift is positive.

The 3D gravity-well visualization requires a larger grid (N >> 9) to
produce a smooth surface.  It will be enabled once the mean-field GPU
backend supports MI-weighted Laplacian construction at scale.

Run:
    uv run python -m examples.physics_qg.gravity_well
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from popgp import Simulator, SimulatorConfig, validation_json
from popgp.config import PiResConfig
from popgp.simulator import Simulator as _Sim

_PKG_DIR = Path(__file__).parent
torch.set_default_dtype(torch.float64)

# ── Configuration ────────────────────────────────────────────────────────

WIDTH, HEIGHT = 3, 3
N = WIDTH * HEIGHT
cfg = SimulatorConfig.for_grid(width=WIDTH, height=HEIGHT, beta=2.0)
cfg.pi_res = PiResConfig(cell_dim=1)
cfg.pi_loc.I_0 = 1.0
cfg.pi_geom.lambda_dim = 0.01

results = _PKG_DIR / "results"
results.mkdir(parents=True, exist_ok=True)

# ── Run the standard pipeline (for MI weights and embedding) ─────────────

sim = Simulator(cfg)
result = sim.run()

print(f"Substrate: {WIDTH}x{HEIGHT} = {N} qubit Heisenberg grid "
      f"(beta={cfg.substrate.beta})")
print(f"D* = {result.pi_geom.D_star}")

coords = result.pi_geom.coords.numpy()
D_star = result.pi_geom.D_star
if D_star < 2:
    coords = _Sim._classical_mds(result.pi_loc.distance_matrix, 2).numpy()

edges = result.pi_loc.edges
w = result.pi_loc.weight_matrix

# ── Graph distances from center (BFS) ───────────────────────────────────

center = (HEIGHT // 2) * WIDTH + (WIDTH // 2)

adj: dict[int, set[int]] = {i: set() for i in range(N)}
for i, j in edges:
    adj[i].add(j)
    adj[j].add(i)

graph_dist = [-1] * N
queue = [center]
graph_dist[center] = 0
while queue:
    node = queue.pop(0)
    for nb in adj[node]:
        if graph_dist[nb] == -1:
            graph_dist[nb] = graph_dist[node] + 1
            queue.append(nb)
graph_dist = np.array(graph_dist, dtype=float)

print(f"\nCenter cell (diagnostic source): {center}")
print(f"Graph distances: {dict(enumerate(graph_dist.astype(int).tolist()))}")

# ── Natural source: Phi from the framework's entropy-contrast delta_rho ──

print("\n=== Test 1: Non-physical pipeline placeholder ===")
if result.pi_time is not None:
    phi_natural = result.pi_time.phi.numpy()
    print(f"Phi range: [{phi_natural.min():.4f}, {phi_natural.max():.4f}]")
else:
    phi_natural = None
    print("Phi_time not computed.")

# ── Localized source: inject a point mass at the center ──────────────────

print("\n=== Test 2: Negative localized Green-function source ===")

delta_rho_point = torch.zeros(N)
POINT_SOURCE_STRENGTH = -0.01
delta_rho_point[center] = POINT_SOURCE_STRENGTH

MU = 0.1
phi_point_tensor, effective_source, source_background, constraint_residual = (
    Simulator._solve_clock_constraint(
        w,
        delta_rho_point,
        mu=MU,
        zero_mode_policy="subtract_mean",
        normalize_potential=True,
    )
)
phi_point = phi_point_tensor.numpy()
effective_source_norm = float(torch.linalg.vector_norm(effective_source).item())
relative_constraint_residual = constraint_residual / effective_source_norm

print(f"mu (mass parameter): {MU}")
print(f"source strength: {POINT_SOURCE_STRENGTH}")
print(f"Phi range: [{phi_point.min():.4f}, {phi_point.max():.4f}]")
print(f"Phi at center (source): {phi_point[center]:.4f}")
print(f"constraint residual: {constraint_residual:.3e}")
print(f"relative constraint residual: {relative_constraint_residual:.3e}")

# ── Radial profile by graph distance ─────────────────────────────────────

print("\n--- Radial Profile (by graph distance) ---")
unique_d = sorted(set(graph_dist))
radial_d, radial_phi, radial_std = [], [], []
for d_val in unique_d:
    mask = graph_dist == d_val
    cells_at_d = np.where(mask)[0].tolist()
    vals = phi_point[mask]
    avg = vals.mean()
    std = vals.std() if len(vals) > 1 else 0.0
    radial_d.append(d_val)
    radial_phi.append(avg)
    radial_std.append(std)
    print(f"  d={int(d_val)}: avg Phi={avg:.4f} +/- {std:.4f}  "
          f"cells={cells_at_d}  values=[{', '.join(f'{v:.4f}' for v in vals)}]")

radial_d = np.array(radial_d)
radial_phi = np.array(radial_phi)
radial_std = np.array(radial_std)

# ── Monotonicity test ────────────────────────────────────────────────────

print("\n--- Monotonicity Test ---")
monotonic = all(radial_phi[i] < radial_phi[i+1]
                for i in range(len(radial_phi) - 1))
print(f"Phi increases monotonically away from the negative well: "
      f"{'PASS' if monotonic else 'FAIL'}")

# ── Symmetry test ────────────────────────────────────────────────────────

print("\n--- Symmetry Test ---")
max_asymmetry = 0.0
for d_val in unique_d:
    mask = graph_dist == d_val
    vals = phi_point[mask]
    if len(vals) > 1:
        scale = max(float(np.max(np.abs(vals))), 1e-12)
        spread = (vals.max() - vals.min()) / scale * 100
        max_asymmetry = max(max_asymmetry, spread)
        print(f"  d={int(d_val)}: spread = {spread:.2f}% of mean")

symmetric = max_asymmetry < 5.0
print(f"Max asymmetry < 5%: {'PASS' if symmetric else 'FAIL'} "
      f"(max = {max_asymmetry:.2f}%)")

# ── Log(r) fit (excluding source) ────────────────────────────────────────

print("\n--- log(r) Fit ---")
fit_mask = radial_d > 0
if fit_mask.sum() >= 2:
    r_fit_data = radial_d[fit_mask]
    phi_fit_data = radial_phi[fit_mask]
    log_r = np.log(r_fit_data)

    A_fit = np.column_stack([log_r, np.ones_like(log_r)])
    coeffs, *_ = np.linalg.lstsq(A_fit, phi_fit_data, rcond=None)
    slope, intercept = coeffs
    phi_pred = A_fit @ coeffs
    ss_res = np.sum((phi_fit_data - phi_pred)**2)
    ss_tot = np.sum((phi_fit_data - phi_fit_data.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"Fit: Phi = {slope:.4f} * log(d) + {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    print("Expected: positive slope (Phi rises away from a negative source)")
    print(f"Note: only {int(fit_mask.sum())} radial shells -- "
          f"log(r) convergence requires N >> 9")
else:
    slope, intercept, r_squared = 0, 0, 0
    print("Not enough radial shells for fit.")

# ── Gravitational redshift ───────────────────────────────────────────────

print("\n--- Gravitational Redshift ---")
phi_at_source = phi_point[center]
phi_at_boundary = radial_phi[-1]
z = Simulator.gravitational_redshift(
    phi_emitter=phi_at_source,
    phi_observer=phi_at_boundary,
)
print(f"Phi(source) = {phi_at_source:.4f},  Phi(boundary) = {phi_at_boundary:.4f}")
print(f"Redshift factor: 1+z = exp(Phi_obs-Phi_emit) = {1+z:.4f}  (z = {z:.4f})")
clock_ratio = np.exp(phi_at_source - phi_at_boundary)
print(f"Source clock rate / boundary clock rate: {clock_ratio:.4f} (slower).")

# ── Summary verdict ──────────────────────────────────────────────────────

diagnostic_pass = monotonic and symmetric
print(f"\n{'='*50}")
print(f"GREEN-FUNCTION DIAGNOSTIC: {'PASS' if diagnostic_pass else 'FAIL'}")
print(f"  Monotonic recovery from well: {'PASS' if monotonic else 'FAIL'}")
print(f"  Grid symmetry:     {'PASS' if symmetric else 'FAIL'}")
print(f"{'='*50}")

# ══════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════

# ── Plot 1: Phi heatmap + radial falloff ─────────────────────────────────

phi_grid = phi_point.reshape(HEIGHT, WIDTH)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax1 = axes[0]
im1 = ax1.imshow(phi_grid, cmap="viridis", origin="lower",
                  interpolation="bilinear")
fig.colorbar(im1, ax=ax1, label="Phi (Clock-Rate Potential)")
ax1.set_title("Clock Potential (Negative Diagnostic Source)")
for iy in range(HEIGHT):
    for ix in range(WIDTH):
        val = phi_grid[iy, ix]
        ax1.text(ix, iy, f"{val:.2f}", ha="center", va="center",
                 fontsize=10, fontweight="bold", color="white")
ax1.scatter([center % WIDTH], [center // WIDTH], marker="*", s=200,
            color="cyan", edgecolors="white", linewidths=1, zorder=5)
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2 = axes[1]
colors_by_d = {0: "red", 1: "steelblue", 2: "orange", 3: "green", 4: "purple"}

for i in range(N):
    d = int(graph_dist[i])
    marker = "*" if d == 0 else "o"
    ms = 15 if d == 0 else 10
    ax2.plot(graph_dist[i], phi_point[i], marker, color=colors_by_d.get(d, "gray"),
             markersize=ms, zorder=5)

ax2.errorbar(radial_d, radial_phi, yerr=radial_std, fmt="k--", linewidth=1.5,
             capsize=4, label="Radial average", zorder=3)

if fit_mask.sum() >= 2:
    r_cont = np.linspace(0.8, radial_d.max() + 0.2, 50)
    ax2.plot(r_cont, slope * np.log(r_cont) + intercept, "r:",
             linewidth=1.5, label=f"log(d) fit (R$^2$={r_squared:.2f})")

for d_val in unique_d:
    label = f"d={int(d_val)}"
    ax2.plot([], [], "o", color=colors_by_d.get(int(d_val), "gray"),
             markersize=8, label=label)

verdict = "DIAGNOSTIC PASS" if diagnostic_pass else "DIAGNOSTIC FAIL"
verdict_color = "green" if diagnostic_pass else "red"
ax2.text(
    0.98, 0.05, verdict, transform=ax2.transAxes,
    fontsize=18, fontweight="bold", color="white",
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", fc=verdict_color, alpha=0.9),
)

ax2.set_xlabel("Graph Distance from Source")
ax2.set_ylabel("Phi (Clock-Rate Potential)")
ax2.set_title("Potential Falloff vs Graph Distance")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, linestyle=":", alpha=0.5)

fig.tight_layout()
fig.savefig(results / "gravity_well.png", dpi=150)
print(f"\nSaved: {results / 'gravity_well.png'}")

# ── Plot 2: Phi on the embedded geometry ─────────────────────────────────

fig2, ax3 = plt.subplots(figsize=(7, 6))

for i, j in edges:
    ax3.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
             "k-", alpha=0.3, linewidth=1)

sc = ax3.scatter(coords[:, 0], coords[:, 1], c=phi_point, cmap="viridis",
                 s=200, zorder=5, edgecolors="black", linewidths=1)
fig2.colorbar(sc, ax=ax3, label="Phi (Clock-Rate Potential)")
ax3.scatter(coords[center, 0], coords[center, 1], marker="*", s=300,
            color="cyan", edgecolors="white", linewidths=1.5, zorder=6)

for i in range(N):
    y_off = 10 if (i % 2 == 0) else -14
    ax3.annotate(f"{phi_point[i]:.2f}", (coords[i, 0], coords[i, 1]),
                 xytext=(6, y_off), textcoords="offset points",
                 fontsize=9, fontweight="bold")

verdict_color_emb = "green" if diagnostic_pass else "red"
ax3.text(
    0.98, 0.02, verdict, transform=ax3.transAxes,
    fontsize=18, fontweight="bold", color="white",
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", fc=verdict_color_emb, alpha=0.9),
)

ax3.set_title("Clock Constraint on Inferred Correlation Geometry")
ax3.axis("equal")
ax3.grid(True, linestyle=":", alpha=0.5)
fig2.tight_layout()
fig2.savefig(results / "gravity_embedding.png", dpi=150)
print(f"Saved: {results / 'gravity_embedding.png'}")

# ── Plot 3: Natural vs point-source comparison ──────────────────────────

if phi_natural is not None:
    fig3, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))

    im4 = ax4.imshow(phi_natural.reshape(HEIGHT, WIDTH), cmap="inferno",
                      origin="lower", interpolation="bilinear")
    fig3.colorbar(im4, ax=ax4, label="Phi")
    ax4.set_title("Pipeline Placeholder (von Neumann entropy)")
    for iy in range(HEIGHT):
        for ix in range(WIDTH):
            val = phi_natural[iy * WIDTH + ix]
            ax4.text(ix, iy, f"{val:.1f}", ha="center", va="center",
                     fontsize=9, fontweight="bold", color="white")

    im5 = ax5.imshow(phi_grid, cmap="viridis", origin="lower",
                      interpolation="bilinear")
    fig3.colorbar(im5, ax=ax5, label="Phi")
    ax5.set_title("Negative Diagnostic Source")
    for iy in range(HEIGHT):
        for ix in range(WIDTH):
            val = phi_grid[iy, ix]
            ax5.text(ix, iy, f"{val:.1f}", ha="center", va="center",
                     fontsize=9, fontweight="bold", color="white")
    ax5.scatter([center % WIDTH], [center // WIDTH], marker="*", s=200,
                color="cyan", edgecolors="white", linewidths=1, zorder=5)

    fig3.suptitle("Clock Potential: Placeholder vs Green-Function Test", fontsize=13)
    fig3.tight_layout()
    fig3.savefig(results / "source_comparison.png", dpi=150)
    print(f"Saved: {results / 'source_comparison.png'}")

# ── Validation Report ─────────────────────────────────────────────────────

report = {
    "example": "gravity_well",
    "scientific_status": "numerical_green_function_diagnostic",
    "framework_version": "1.0-submission-draft",
    "package_version": "0.1.0",
    "config": {
        "width": WIDTH,
        "height": HEIGHT,
        "n_qubits": N,
        "topology": cfg.substrate.topology,
        "beta": cfg.substrate.beta,
        "cell_dim": cfg.pi_res.cell_dim,
        "I_0": cfg.pi_loc.I_0,
        "lambda_dim": cfg.pi_geom.lambda_dim,
        "mu": MU,
        "point_source_strength": POINT_SOURCE_STRENGTH,
        "center_cell": int(center),
        "use_exact_backend": cfg.use_exact_backend,
    },
    "pipeline": {
        "pi_res": {
            "n_cells": len(result.pi_res.cells),
            "cells": result.pi_res.cells,
            "leakage": result.pi_res.leakage,
            "retention_loss": result.pi_res.retention_loss,
            "admissible": result.pi_res.admissible,
            "n_total_partitions": result.pi_res.n_total,
            "n_admissible_partitions": result.pi_res.n_admissible,
        },
        "pi_geom": {
            "D_spectral": float(result.pi_geom.D_spectral),
            "D_star": int(result.pi_geom.D_star),
            "stress": float(result.pi_geom.stress),
            "embedding_status": result.pi_geom.embedding_status,
            "metric_diagnostics": result.pi_geom.metric_diagnostics,
            "complex_status": result.pi_geom.complex_status,
        },
        "pi_time_natural": {
            "phi": phi_natural.tolist() if phi_natural is not None else None,
            "phi_min": float(phi_natural.min()) if phi_natural is not None else None,
            "phi_max": float(phi_natural.max()) if phi_natural is not None else None,
        },
        "gravity_test": {
            "scientific_status": "numerical_green_function_diagnostic",
            "physical_source_law_validated": False,
            "effective_source": effective_source.tolist(),
            "source_background": float(source_background),
            "constraint_residual": float(constraint_residual),
            "relative_constraint_residual": float(relative_constraint_residual),
            "phi_point": phi_point.tolist(),
            "phi_min": float(phi_point.min()),
            "phi_max": float(phi_point.max()),
            "phi_at_source": float(phi_point[center]),
            "graph_distances": graph_dist.astype(int).tolist(),
            "radial_profile": {
                "distances": radial_d.tolist(),
                "phi_avg": radial_phi.tolist(),
                "phi_std": radial_std.tolist(),
            },
            "log_fit": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_squared),
            },
            "redshift": {
                "phi_source": float(phi_at_source),
                "phi_boundary": float(phi_at_boundary),
                "z": float(z),
                "one_plus_z": float(1 + z),
            },
        },
    },
    "checks": [
        {
            "name": "pi_res_admissibility",
            "description": (
                "The singleton resolution must satisfy the configured retention bound"
            ),
            "framework_section": "4.4.2a",
            "criterion": "pi_res.admissible == true",
            "value": {
                "admissible": result.pi_res.admissible,
                "retention_loss": result.pi_res.retention_loss,
                "retention_epsilon": cfg.pi_res.retention_epsilon,
            },
            "passed": result.pi_res.admissible is True,
        },
        {
            "name": "nonzero_source_constraint_residual",
            "description": "The nonzero diagnostic source satisfies the screened constraint",
            "framework_section": "4.4.5",
            "criterion": "relative_constraint_residual < 1e-12",
            "value": float(relative_constraint_residual),
            "passed": relative_constraint_residual < 1e-12,
        },
        {
            "name": "monotonic_falloff",
            "description": "Phi rises monotonically away from the negative diagnostic source",
            "framework_section": "4.4.5 / 5.1",
            "criterion": "Phi(d) < Phi(d+1) for all consecutive shells",
            "value": {f"d={int(d)}": float(p) for d, p in zip(radial_d, radial_phi)},
            "passed": monotonic,
        },
        {
            "name": "grid_symmetry",
            "description": (
                "Cells equidistant from source have equal Phi "
                "(lattice symmetry preserved)"
            ),
            "framework_section": "4.4.5",
            "criterion": "max asymmetry < 5%",
            "value": float(max_asymmetry),
            "threshold": 5.0,
            "passed": symmetric,
        },
        {
            "name": "negative_well_at_source",
            "description": "Clock potential is minimal at the negative source cell",
            "framework_section": "5.1",
            "criterion": "argmin(phi) == center",
            "value": {"argmin": int(np.argmin(phi_point)), "center": int(center)},
            "passed": int(np.argmin(phi_point)) == int(center),
        },
        {
            "name": "redshift_positive",
            "description": "Emitter in the negative well is redshifted at the boundary",
            "framework_section": "5.1",
            "criterion": "z > 0",
            "value": float(z),
            "passed": z > 0,
        },
        {
            "name": "dimension_selection_2d",
            "description": "Complexity-stress selects D* = 2 for the 2D grid",
            "framework_section": "4.4.4",
            "criterion": "D_star == 2",
            "value": int(result.pi_geom.D_star),
            "passed": result.pi_geom.D_star == 2,
        },
    ],
}

report["overall_pass"] = all(c["passed"] for c in report["checks"])
report["artifacts"] = [
    "results/gravity_well.png",
    "results/gravity_embedding.png",
    "results/source_comparison.png",
    "results/validation.json",
]

val_path = results / "validation.json"
val_path.write_text(validation_json(report) + "\n", encoding="utf-8")
print(f"Saved: {val_path}")

print("\nDone.")
