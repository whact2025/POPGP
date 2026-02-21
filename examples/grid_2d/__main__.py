"""
2D Grid: Emergent geometry from a scrambled algebra.

Uses the unified Simulator API to demonstrate:
  - Π_loc: Mutual information between all qubit pairs (§4.4.3)
  - Π_geom: Dimension selection (D*=2) and MDS embedding (§4.4.4)
  - Π_time: Clock potential on the 2D grid (§4.4.5)
  - Visualization: ground-truth topology overlay on the embedding

Run:
    uv run python -m examples.grid_2d
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from popgp import Simulator, SimulatorConfig, validation_json
from popgp.config import PiResConfig

_PKG_DIR = Path(__file__).parent

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
    from popgp.simulator import Simulator as _Sim
    coords = _Sim._classical_mds(result.pi_loc.distance_matrix, 2).numpy()

edges = sim.backend.build_edges()

# ── Topology Validation (compute before plotting) ────────────────────────

print("\n--- Topology Validation ---")
center = (HEIGHT // 2) * WIDTH + (WIDTH // 2)
neighbor_set = set()
for i, j in edges:
    if i == center:
        neighbor_set.add(j)
    if j == center:
        neighbor_set.add(i)
other_set = set(range(N)) - neighbor_set - {center}

d_neighbors = [np.linalg.norm(coords[center] - coords[n])
               for n in neighbor_set]
d_others = [np.linalg.norm(coords[center] - coords[o])
            for o in other_set]

avg_neigh = np.mean(d_neighbors) if d_neighbors else 0
avg_other = np.mean(d_others) if d_others else 0
topology_ok = avg_neigh < avg_other
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
    im = ax.imshow(phi_grid, cmap="inferno", origin="lower")
    fig.colorbar(im, ax=ax, label="Phi (Clock-Rate Potential)")
    ax.set_title("Emergent Clock Potential (Sec 4.4.5)")

    for iy in range(HEIGHT):
        for ix in range(WIDTH):
            ax.text(ix, iy, f"{phi_grid[iy, ix]:.1f}",
                    ha="center", va="center", fontsize=8, color="white")

    fig.tight_layout()
    fig.savefig(results / "clock_potential.png", dpi=150)
    print(f"Saved: {results / 'clock_potential.png'}")

# ── Validation Report ─────────────────────────────────────────────────────

phi = result.pi_time.phi.numpy() if result.pi_time is not None else None
phi_range = float(phi.max() - phi.min()) if phi is not None else None
phi_mean = float(phi.mean()) if phi is not None else None

report = {
    "example": "grid_2d",
    "framework_version": "0.10",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "config": {
        "width": WIDTH,
        "height": HEIGHT,
        "n_qubits": N,
        "topology": cfg.substrate.topology,
        "beta": cfg.substrate.beta,
        "coupling_J": cfg.substrate.coupling_J,
        "cell_dim": cfg.pi_res.cell_dim,
        "I_0": cfg.pi_loc.I_0,
        "lambda_dim": cfg.pi_geom.lambda_dim,
        "use_exact_backend": cfg.use_exact_backend,
    },
    "pipeline": {
        "pi_res": {
            "n_cells": len(result.pi_res.cells),
            "cells": result.pi_res.cells,
        },
        "pi_loc": {
            "mi_matrix_shape": list(result.pi_loc.mi_matrix.shape),
            "mi_min_positive": float(mi[mi > 0].min().item()),
            "mi_max": float(mi.max().item()),
        },
        "pi_geom": {
            "D_spectral": float(result.pi_geom.D_spectral),
            "D_star": int(result.pi_geom.D_star),
            "stress": float(result.pi_geom.stress),
            "coords": coords.tolist(),
        },
        "pi_time": {
            "phi": phi.tolist() if phi is not None else None,
            "phi_min": float(phi.min()) if phi is not None else None,
            "phi_max": float(phi.max()) if phi is not None else None,
            "phi_range": phi_range,
            "phi_mean": phi_mean,
        },
    },
    "checks": [
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
            "description": "In the MDS embedding, graph neighbors of the center node are closer than non-neighbors",
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
            "name": "spectral_dimension",
            "description": "Spectral dimension D_S is near 2.0 for a 2D lattice (finite-size effects expected at N=9)",
            "framework_section": "4.4.4",
            "criterion": "0.5 <= D_spectral <= 3.0",
            "value": float(result.pi_geom.D_spectral),
            "passed": 0.5 <= result.pi_geom.D_spectral <= 3.0,
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
            "name": "clock_potential_computed",
            "description": "Clock potential Phi was successfully computed",
            "framework_section": "4.4.5",
            "criterion": "pi_time is not None",
            "value": result.pi_time is not None,
            "passed": result.pi_time is not None,
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
val_path.write_text(validation_json(report))
print(f"Saved: {val_path}")

print("\nDone.")
