# Copyright (c) 2026 WHACT. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""
2D Grid: Emergent geometry from a scrambled algebra.

Uses the unified Simulator API to demonstrate:
  - Π_loc: Mutual information between all qubit pairs (§4.4.3)
  - Π_geom: Dimension selection (D*=2) and MDS embedding (§4.4.4)
  - Π_time: Clock potential on the 2D grid (§4.4.5)
  - Blind inferred-edge validation against held-out Hamiltonian edges

Run:
    uv run python -m examples.physics_qg.grid_2d
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from popgp import Simulator, SimulatorConfig, validation_json
from popgp.config import PiResConfig
from popgp.diagnostics import edge_recovery_metrics
from popgp.simulator import Simulator as _Sim

_PKG_DIR = Path(__file__).parent
torch.set_default_dtype(torch.float64)

# ── Configuration ────────────────────────────────────────────────────────

WIDTH, HEIGHT = 3, 3
cfg = SimulatorConfig.for_grid(width=WIDTH, height=HEIGHT, beta=2.0)
cfg.pi_res = PiResConfig(cell_dim=1)
cfg.pi_loc.I_0 = 1.0
cfg.pi_geom.lambda_dim = 0.01

results = _PKG_DIR / "results"
results.mkdir(parents=True, exist_ok=True)

# ── Run the full pipeline ────────────────────────────────────────────────

sim = Simulator(cfg)
result = sim.run()

N = cfg.substrate.n_qubits
print(f"Substrate: {WIDTH}x{HEIGHT} = {N} qubit Heisenberg grid "
      f"(beta={cfg.substrate.beta})")
print(f"Cells: {len(result.pi_res.cells)} (1 qubit each)")

# ── MI Matrix ────────────────────────────────────────────────────────────

print("\n--- Mutual Information ---")
mi = result.pi_loc.mi_matrix
print(f"MI range: [{mi[mi > 0].min().item():.4f}, {mi.max().item():.4f}]")

# ── Geometry ─────────────────────────────────────────────────────────────

print("\n--- Geometry Recovery ---")
print(f"D_spectral: {result.pi_geom.D_spectral:.2f}")
print(f"D*: {result.pi_geom.D_star}")
print(f"MDS stress: {result.pi_geom.stress:.4f}")

coords = result.pi_geom.coords.numpy()
D_star = result.pi_geom.D_star
print(f"Coordinates shape: {coords.shape}")

if D_star < 2:
    print(f"WARNING: D* = {D_star} < 2; re-embedding with D=2 for visualization.")
    coords = _Sim._classical_mds(result.pi_loc.distance_matrix, 2).numpy()

edges = result.pi_loc.edges
reference_edges = {tuple(sorted(edge)) for edge in sim.backend.build_edges()}
recovery = edge_recovery_metrics(set(edges), reference_edges)

# ── Topology Validation (compute before plotting) ────────────────────────

print("\n--- Topology Validation ---")
print(f"Blind edge precision: {recovery.precision:.3f}")
print(f"Blind edge recall:    {recovery.recall:.3f}")
print(f"MI gap ratio:         {result.pi_loc.connectivity_gap_ratio:.3f}")
print(f"MI spectrum separable: {result.pi_loc.connectivity_separable}")
center = (HEIGHT // 2) * WIDTH + (WIDTH // 2)
neighbor_set = set()
for i, j in reference_edges:
    if i == center:
        neighbor_set.add(j)
    if j == center:
        neighbor_set.add(i)
other_set = set(range(N)) - neighbor_set - {center}

d_neighbors = [np.linalg.norm(coords[center] - coords[n])
               for n in neighbor_set]
d_others = [np.linalg.norm(coords[center] - coords[o])
            for o in other_set]

avg_neigh = np.mean(d_neighbors) if d_neighbors else float("nan")
avg_other = np.mean(d_others) if d_others else float("nan")
topology_ok = bool(d_neighbors and d_others) and avg_neigh < avg_other
separation = (avg_other - avg_neigh) / avg_other * 100 if avg_other > 0 else 0

print(f"Center node: {center}")
print(f"Avg dist to neighbors:     {avg_neigh:.4f}")
print(f"Avg dist to non-neighbors: {avg_other:.4f}")
print(f"Separation: {separation:.1f}%")
if topology_ok:
    print("SUCCESS: Local structure preserved.")
else:
    print("FAILURE: Geometry distorted.")

# ── Embedding plot ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 7))

for i, j in edges:
    ax.plot(
        [coords[i, 0], coords[j, 0]],
        [coords[i, 1], coords[j, 1]],
        "k-", alpha=0.3, linewidth=1,
    )

colors = ["steelblue"] * N
colors[center] = "orange"
ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=120, zorder=5, edgecolors="black")

for i in range(N):
    ax.annotate(
        str(i), (coords[i, 0], coords[i, 1]),
        xytext=(6, 6), textcoords="offset points", fontsize=11, fontweight="bold",
    )

verdict = "PASS" if topology_ok else "FAIL"
verdict_color = "green" if topology_ok else "red"
ax.text(
    0.98, 0.02, verdict, transform=ax.transAxes,
    fontsize=18, fontweight="bold", color="white",
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", fc=verdict_color, alpha=0.9),
)
ax.set_title(f"Emergent 2D Geometry ({WIDTH}x{HEIGHT} Heisenberg, Sec 4.4.4)")
ax.axis("equal")
ax.grid(True, linestyle=":", alpha=0.6)
fig.tight_layout()
fig.savefig(results / "embedding.png", dpi=150)
print(f"Saved: {results / 'embedding.png'}")

# ── Clock Potential ──────────────────────────────────────────────────────

if result.pi_time is not None:
    print("\n--- Clock Potential ---")
    phi = result.pi_time.phi.numpy()
    print(f"Phi range: [{phi.min():.4f}, {phi.max():.4f}]")

    phi_grid = phi.reshape(HEIGHT, WIDTH)
    phi_range = phi.max() - phi.min()
    phi_mean = phi.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    effective_source_norm = float(
        torch.linalg.vector_norm(result.pi_time.delta_rho).item()
    )
    placeholder_degenerate = effective_source_norm < 1e-12 and phi_range < 1e-12
    display_phi_grid = np.zeros_like(phi_grid) if placeholder_degenerate else phi_grid
    display_limits = {"vmin": -1e-12, "vmax": 1e-12} if placeholder_degenerate else {}
    im = ax.imshow(
        display_phi_grid,
        cmap="inferno",
        origin="lower",
        **display_limits,
    )
    fig.colorbar(im, ax=ax, label="Phi (Clock-Rate Potential)")
    ax.set_title("Placeholder Clock Diagnostic (Constant Source Removed)")
    if placeholder_degenerate:
        ax.text(
            0.5,
            -0.12,
            "SU(2)-invariant one-site marginals make this source zero at round-off",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
        )

    for iy in range(HEIGHT):
        for ix in range(WIDTH):
            ax.text(ix, iy, f"{display_phi_grid[iy, ix]:.1f}",
                    ha="center", va="center", fontsize=8, color="white")

    fig.tight_layout()
    fig.savefig(results / "clock_potential.png", dpi=150)
    print(f"Saved: {results / 'clock_potential.png'}")

# ── Validation Report ─────────────────────────────────────────────────────

phi = result.pi_time.phi.numpy() if result.pi_time is not None else None
phi_range = float(phi.max() - phi.min()) if phi is not None else None
phi_mean = float(phi.mean()) if phi is not None else None
effective_source_norm = (
    float(torch.linalg.vector_norm(result.pi_time.delta_rho).item())
    if result.pi_time is not None
    else None
)
placeholder_degenerate = (
    effective_source_norm is not None
    and effective_source_norm < 1e-12
    and phi_range is not None
    and phi_range < 1e-12
)

report = {
    "example": "grid_2d",
    "scientific_status": "finite_exact_benchmark",
    "framework_version": "1.0-submission-draft",
    "package_version": "0.1.0",
    "config": {
        "width": WIDTH,
        "height": HEIGHT,
        "n_qubits": N,
        "topology": cfg.substrate.topology,
        "beta": cfg.substrate.beta,
        "coupling_J": cfg.substrate.coupling_J,
        "cell_dim": cfg.pi_res.cell_dim,
        "retention_epsilon": cfg.pi_res.retention_epsilon,
        "I_0": cfg.pi_loc.I_0,
        "lambda_dim": cfg.pi_geom.lambda_dim,
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
        "pi_loc": {
            "mi_matrix_shape": list(result.pi_loc.mi_matrix.shape),
            "mi_min_positive": float(mi[mi > 0].min().item()),
            "mi_max": float(mi.max().item()),
            "connectivity_method": result.pi_loc.connectivity_method,
            "connectivity_threshold": result.pi_loc.connectivity_threshold,
            "connectivity_gap_ratio": result.pi_loc.connectivity_gap_ratio,
            "connectivity_separable": result.pi_loc.connectivity_separable,
            "inferred_edges": result.pi_loc.edges,
            "held_out_reference_edges": sorted(reference_edges),
            "edge_precision": recovery.precision,
            "edge_recall": recovery.recall,
        },
        "pi_geom": {
            "D_spectral": float(result.pi_geom.D_spectral),
            "spectral_diagnostic": "finite_graph_peak",
            "spectral_peak_time": result.pi_geom.spectral_peak_time,
            "D_star": int(result.pi_geom.D_star),
            "stress": float(result.pi_geom.stress),
            "objective": float(result.pi_geom.objective),
            "embedding_status": result.pi_geom.embedding_status,
            "selection_margin": result.pi_geom.selection_margin,
            "metric_diagnostics": result.pi_geom.metric_diagnostics,
            "complex_status": result.pi_geom.complex_status,
            "simplices": result.pi_geom.simplices,
            "deficit_angles": result.pi_geom.deficit_angles,
            "coords": coords.tolist(),
        },
        "pi_time": {
            "source_model": result.pi_time.source_model,
            "source_status": result.pi_time.source_status,
            "phi": phi.tolist() if phi is not None else None,
            "phi_min": float(phi.min()) if phi is not None else None,
            "phi_max": float(phi.max()) if phi is not None else None,
            "phi_range": phi_range,
            "phi_mean": phi_mean,
            "constraint_residual": result.pi_time.constraint_residual,
            "effective_source_norm": effective_source_norm,
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
            "name": "blind_edge_recovery",
            "description": "MI-only inference is compared with held-out Hamiltonian edges",
            "framework_section": "4.4.3",
            "criterion": "edge precision == 1 and edge recall == 1",
            "value": {
                "precision": recovery.precision,
                "recall": recovery.recall,
                "false_positives": recovery.false_positives,
                "false_negatives": recovery.false_negatives,
            },
            "passed": recovery.precision == 1.0 and recovery.recall == 1.0,
        },
        {
            "name": "dimension_selection",
            "description": "Complexity-stress functional selects D* = 2 for a 2D grid",
            "framework_section": "4.4.4",
            "criterion": "D_star == 2",
            "value": int(result.pi_geom.D_star),
            "passed": result.pi_geom.D_star == 2,
        },
        {
            "name": "topology_preservation",
            "description": (
                "Held-out graph neighbors of the center are closer in the embedding "
                "than held-out non-neighbors"
            ),
            "framework_section": "4.4.4",
            "criterion": "avg_dist_neighbors < avg_dist_non_neighbors",
            "value": {
                "center_node": center,
                "avg_dist_neighbors": float(avg_neigh),
                "avg_dist_non_neighbors": float(avg_other),
                "separation_pct": float(separation),
            },
            "passed": topology_ok,
        },
        {
            "name": "finite_graph_spectral_peak",
            "description": (
                "The scale-dependent heat-kernel dimension has a finite-size peak; "
                "this is not a continuum spectral-dimension estimate"
            ),
            "framework_section": "4.4.4",
            "criterion": "1.0 <= finite_graph_peak <= 2.0",
            "value": float(result.pi_geom.D_spectral),
            "passed": 1.0 <= result.pi_geom.D_spectral <= 2.0,
        },
        {
            "name": "mds_stress",
            "description": "MDS stress is low, indicating faithful embedding",
            "framework_section": "4.4.4",
            "criterion": "stress < 0.5",
            "value": float(result.pi_geom.stress),
            "passed": result.pi_geom.stress < 0.5,
        },
        {
            "name": "placeholder_source_degeneracy",
            "description": (
                "SU(2)-invariant one-site Gibbs marginals make the entropy "
                "placeholder constant, so zero-mode removal gives Phi=0"
            ),
            "framework_section": "4.4.5",
            "criterion": "effective_source_norm < 1e-12 and phi_range < 1e-12",
            "value": {
                "effective_source_norm": effective_source_norm,
                "phi_range": phi_range,
                "constraint_residual": result.pi_time.constraint_residual,
            },
            "passed": placeholder_degenerate,
        },
    ],
}

report["overall_pass"] = all(c["passed"] for c in report["checks"])
report["artifacts"] = [
    "results/embedding.png",
    "results/clock_potential.png",
    "results/validation.json",
]

val_path = results / "validation.json"
val_path.write_text(validation_json(report) + "\n", encoding="utf-8")
print(f"Saved: {val_path}")

print("\nDone.")
