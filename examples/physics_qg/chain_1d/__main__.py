# Copyright (c) 2026 WHACT. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""
1D Chain: Stability selection and geometry recovery.

Uses the unified Simulator API to demonstrate:
  - Π_res: Stability selection — local cells leak less than non-local (§4.4.2a)
  - Π_loc: Mutual information → distance kernel (§4.4.3)
  - Π_geom: MDS embedding recovers 1D geometry (§4.4.4)
  - Π_time: Clock potential from entropy contrast (§4.4.5)

Run:
    uv run python -m examples.physics_qg.chain_1d
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from popgp import Simulator, SimulatorConfig, validation_json
from popgp.config import PiResConfig

_PKG_DIR = Path(__file__).parent

# ── Configuration ────────────────────────────────────────────────────────

cfg = SimulatorConfig.for_chain(n=8, beta=1.0)
cfg.pi_res = PiResConfig(
    cell_dim=2,
    phase_window_width=2.0,
    phase_window_samples=5,
    retention_epsilon=10.0,  # permissive; framework says set relative to Cap(∂R)
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
print(f"Selected cells: {result.pi_res.cells}")
print(f"L_leak: {result.pi_res.leakage:.6e}")
if result.pi_res.drift is not None:
    print(f"L_drift: {result.pi_res.drift:.6e}")
if result.pi_res.retention_loss is not None:
    print(f"Retention loss: {result.pi_res.retention_loss:.4f}")
if result.pi_res.su2_equivariant is not None:
    print(f"SU(2) equivariant: {result.pi_res.su2_equivariant}")

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
for ri in range(1, n_cells):
    if diffs[ri - 1] > tol:
        current_rank += 1
    rank[order_by_rank[ri]] = current_rank

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
    colors = ["silver" if v == 0.0 else "teal" for v in phi]
    bars = ax.bar(range(n_cells), phi, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(phi_mean, color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean = {phi_mean:.2f}")
    ax.set_xlabel("Cell Index")
    ax.set_ylabel("Phi (Clock-Rate Potential)")
    ax.set_title("Emergent Clock Potential (Sec 4.4.5)")
    ax.set_xticks(range(n_cells))
    for i, v in enumerate(phi):
        ax.text(i, max(v, phi.max() * 0.02), f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend()

    fig.tight_layout()
    fig.savefig(results / "clock_potential.png", dpi=150)
    print(f"Saved: {results / 'clock_potential.png'}")

# ── Validation Report ─────────────────────────────────────────────────────

phi = result.pi_time.phi.numpy() if result.pi_time is not None else None
phi_range = float(phi.max() - phi.min()) if phi is not None else None
phi_mean = float(phi.mean()) if phi is not None else None

report = {
    "example": "chain_1d",
    "framework_version": "0.10",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "config": {
        "n_qubits": cfg.substrate.n_qubits,
        "topology": cfg.substrate.topology,
        "beta": cfg.substrate.beta,
        "coupling_J": cfg.substrate.coupling_J,
        "cell_dim": cfg.pi_res.cell_dim,
        "phase_window_width": cfg.pi_res.phase_window_width,
        "phase_window_samples": cfg.pi_res.phase_window_samples,
        "retention_epsilon": cfg.pi_res.retention_epsilon,
        "dt": cfg.simulation.dt,
        "n_steps": cfg.simulation.n_steps,
        "use_exact_backend": cfg.use_exact_backend,
    },
    "pipeline": {
        "pi_res": {
            "n_cells": n_cells,
            "cell_dim": cfg.pi_res.cell_dim,
            "cells": result.pi_res.cells,
            "leakage": float(result.pi_res.leakage),
            "drift": float(result.pi_res.drift) if result.pi_res.drift is not None else None,
            "retention_loss": float(result.pi_res.retention_loss) if result.pi_res.retention_loss is not None else None,
            "su2_equivariant": result.pi_res.su2_equivariant,
        },
        "pi_loc": {
            "mi_matrix_shape": list(result.pi_loc.mi_matrix.shape),
            "mi_min_positive": float(result.pi_loc.mi_matrix[result.pi_loc.mi_matrix > 0].min().item()),
            "mi_max": float(result.pi_loc.mi_matrix.max().item()),
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
            "name": "stability_selection",
            "description": "Invalid (non-local) cells reach higher entropy than valid (local) cells",
            "framework_section": "4.4.2a",
            "criterion": "entropy_increase_invalid > entropy_increase_valid",
            "value": {"valid": slope_valid, "invalid": slope_invalid},
            "passed": stability_ok,
        },
        {
            "name": "contiguous_cells",
            "description": "Optimizer selects contiguous 2-qubit blocks",
            "framework_section": "4.4.2a",
            "criterion": "cells == [[0,1],[2,3],[4,5],[6,7]]",
            "value": result.pi_res.cells,
            "passed": result.pi_res.cells == [[0, 1], [2, 3], [4, 5], [6, 7]],
        },
        {
            "name": "su2_equivariance",
            "description": "Coarse-graining map commutes with SU(2) action",
            "framework_section": "4.4.2a (E4)",
            "criterion": "su2_equivariant == True",
            "value": result.pi_res.su2_equivariant,
            "passed": result.pi_res.su2_equivariant is True,
        },
        {
            "name": "geometry_1d_ordering",
            "description": "MDS embedding recovers monotonic 1D ordering of chain cells",
            "framework_section": "4.4.4",
            "criterion": "tolerance-ranked coords are monotonic",
            "value": {"coords": coords.tolist(), "rank": rank.tolist(), "tolerance": tol},
            "passed": is_monotonic,
        },
        {
            "name": "dimension_selection",
            "description": "Complexity-stress functional selects D* = 1",
            "framework_section": "4.4.4",
            "criterion": "D_star == 1",
            "value": int(result.pi_geom.D_star),
            "passed": result.pi_geom.D_star == 1,
        },
        {
            "name": "clock_potential_nontrivial",
            "description": "Clock potential Phi has non-trivial structure (not flat)",
            "framework_section": "4.4.5",
            "criterion": "phi_range > 0",
            "value": phi_range,
            "passed": phi_range is not None and phi_range > 0,
        },
    ],
}

report["overall_pass"] = all(c["passed"] for c in report["checks"])
report["artifacts"] = [
    "results/entropy_growth.png",
    "results/embedding.png",
    "results/clock_potential.png",
    "results/validation.json",
]

val_path = results / "validation.json"
val_path.write_text(validation_json(report))
print(f"Saved: {val_path}")

print("\nDone.")
