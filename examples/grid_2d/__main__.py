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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from popgp import Simulator, SimulatorConfig
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

print("\nDone.")
