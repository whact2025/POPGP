"""
1D Chain: Stability selection and geometry recovery.

Uses the unified Simulator API to demonstrate:
  - Π_res: Stability selection — local cells leak less than non-local (§4.4.2a)
  - Π_loc: Mutual information → distance kernel (§4.4.3)
  - Π_geom: MDS embedding recovers 1D geometry (§4.4.4)
  - Π_time: Clock potential from entropy contrast (§4.4.5)

Run:
    uv run python -m examples.chain_1d
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from popgp import Simulator, SimulatorConfig
from popgp.config import PiResConfig

_PKG_DIR = Path(__file__).parent

# ── Configuration ────────────────────────────────────────────────────────

cfg = SimulatorConfig.for_chain(n=8, beta=1.0)
cfg.pi_res = PiResConfig(
    cell_dim=2, phase_window_width=2.0, phase_window_samples=20,
)
cfg.simulation.dt = 0.1
cfg.simulation.n_steps = 20

results = _PKG_DIR / "results"
results.mkdir(parents=True, exist_ok=True)

# ── Run the full pipeline ────────────────────────────────────────────────

sim = Simulator(cfg)
result = sim.run()

n_cells = len(result.pi_res.cells)
print(f"Substrate: {cfg.substrate.n_qubits}-qubit Heisenberg chain "
      f"(beta={cfg.substrate.beta})")
print(f"Cells: {n_cells} blocks of {cfg.pi_res.cell_dim} qubits")
print(f"L_leak: {result.pi_res.leakage:.6f}")

# ── Stability comparison: local vs non-local cells ──────────────────────

print("\n--- Stability Comparison ---")
sim.prepare()

valid_cells = result.pi_res.cells
N = cfg.substrate.n_qubits
k = cfg.pi_res.cell_dim
n_cells = N // k
invalid_cells = [[i, i + n_cells] for i in range(n_cells)]

dt = cfg.simulation.dt
steps = cfg.simulation.n_steps
entropy_valid = torch.zeros((steps, n_cells))
entropy_invalid = torch.zeros((steps, n_cells))

neel_state_0 = torch.tensor([1, 0], dtype=torch.complex128)
neel_state_1 = torch.tensor([0, 1], dtype=torch.complex128)
psi = neel_state_0
for i in range(1, N):
    psi = torch.kron(psi, neel_state_1 if i % 2 else neel_state_0)
rho_neel = torch.outer(psi, psi.conj())

rho_curr = rho_neel.clone()
for t_idx in range(steps):
    if t_idx > 0:
        rho_curr = sim.backend.evolve(rho_curr, dt)
    for i, cell in enumerate(valid_cells):
        rho_red = sim.backend.reduced_state(rho_curr, cell)
        entropy_valid[t_idx, i] = sim.backend.entropy(rho_red)
    for i, cell in enumerate(invalid_cells):
        rho_red = sim.backend.reduced_state(rho_curr, cell)
        entropy_invalid[t_idx, i] = sim.backend.entropy(rho_red)

slope_valid = (entropy_valid[-1, :] - entropy_valid[0, :]).mean().item()
slope_invalid = (entropy_invalid[-1, :] - entropy_invalid[0, :]).mean().item()
print(f"Entropy Increase (Valid/Local):     {slope_valid:.4f}")
print(f"Entropy Increase (Invalid/Non-local): {slope_invalid:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
t_axis = np.arange(steps) * dt
y_valid = entropy_valid.mean(dim=1).numpy()
y_invalid = entropy_invalid.mean(dim=1).numpy()
ax.plot(t_axis, y_valid, "b-", linewidth=2, label="Valid Cells (Local)")
ax.plot(t_axis, y_invalid, "r--", linewidth=2, label="Invalid Cells (Non-local)")

gap = y_invalid - y_valid
stability_ok = (gap[-1] > 0)
ax.fill_between(
    t_axis, y_valid, y_invalid,
    where=(gap > 0), alpha=0.15, color="green", label="Stability gap (red > blue)",
)
ax.fill_between(
    t_axis, y_valid, y_invalid,
    where=(gap < 0), alpha=0.15, color="red",
)

verdict = "PASS" if stability_ok else "FAIL"
verdict_color = "green" if stability_ok else "red"
ax.text(
    0.98, 0.05, verdict, transform=ax.transAxes,
    fontsize=18, fontweight="bold", color="white",
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", fc=verdict_color, alpha=0.9),
)
ax.set_xlabel("Phase Order (s)")
ax.set_ylabel("Avg Cell Entropy")
ax.set_title("Stability Selection: Local vs Non-local Subsystems (Sec 4.4.2a)")
ax.legend(loc="center right")
fig.tight_layout()
fig.savefig(results / "entropy_growth.png", dpi=150)
print(f"Saved: {results / 'entropy_growth.png'}")

# ── Mutual Information & Geometry ────────────────────────────────────────

print("\n--- Geometry Recovery ---")
coords = result.pi_geom.coords.numpy().flatten()

print(f"D_spectral: {result.pi_geom.D_spectral:.2f}")
print(f"D*: {result.pi_geom.D_star}")
print(f"MDS stress: {result.pi_geom.stress:.4f}")
print(f"Recovered coords: {coords}")

sorted_idx = np.argsort(coords)
print(f"Sorted cell order: {sorted_idx}")

sorted_coords = coords[np.argsort(coords)]
diffs = np.diff(sorted_coords)
tol = max(np.ptp(coords) * 1e-3, 1e-10)
rank = np.zeros(n_cells, dtype=int)
current_rank = 0
order_by_rank = np.argsort(coords)
rank[order_by_rank[0]] = 0
for k in range(1, n_cells):
    if diffs[k - 1] > tol:
        current_rank += 1
    rank[order_by_rank[k]] = current_rank

is_monotonic = (
    all(rank[i] <= rank[i + 1] for i in range(n_cells - 1)) or
    all(rank[i] >= rank[i + 1] for i in range(n_cells - 1))
)
print(f"Tolerance-ranked order: {rank.tolist()}  (tol={tol:.2e})")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(coords, np.zeros_like(coords), "o", markersize=12, color="steelblue")
for i in range(n_cells):
    y_off = 18 if i % 2 == 0 else -24
    ax.annotate(
        str(i), (coords[i], 0),
        xytext=(0, y_off), textcoords="offset points",
        ha="center", fontsize=13, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    )

verdict = "PASS" if is_monotonic else "FAIL"
verdict_color = "green" if is_monotonic else "red"
ax.text(
    0.98, 0.95, verdict, transform=ax.transAxes,
    fontsize=18, fontweight="bold", color="white",
    ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.4", fc=verdict_color, alpha=0.9),
)
ax.set_title("Recovered 1D Geometry from Substrate Correlations (Sec 4.4.4)")
ax.set_yticks([])
ax.set_xlabel("Emergent Dimension 1")
fig.tight_layout()
fig.savefig(results / "embedding.png", dpi=150)
print(f"Saved: {results / 'embedding.png'}")

# ── Clock Potential ──────────────────────────────────────────────────────

if result.pi_time is not None:
    print("\n--- Clock Potential ---")
    phi = result.pi_time.phi.numpy()
    print(f"Phi range: [{phi.min():.4f}, {phi.max():.4f}]")

    phi_range = phi.max() - phi.min()
    phi_mean = phi.mean()
    is_flat = phi_range < 0.3 * abs(phi_mean) if abs(phi_mean) > 1e-6 else phi_range < 1e-3

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(n_cells), phi, color="teal", edgecolor="black", linewidth=0.5)
    ax.axhline(phi_mean, color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean = {phi_mean:.2f}")
    ax.set_xlabel("Cell Index")
    ax.set_ylabel("Phi (Clock-Rate Potential)")
    ax.set_title("Emergent Clock Potential (Sec 4.4.5)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results / "clock_potential.png", dpi=150)
    print(f"Saved: {results / 'clock_potential.png'}")

print("\nDone.")
