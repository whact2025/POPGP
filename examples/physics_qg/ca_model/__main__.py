# Copyright (c) 2026 WHACT. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Cellular Automata: Stability selection and radiative cooling.

Phenomenological model (Bloch-sphere cells, not full quantum mechanics)
demonstrating stability selection (§4.4.2a) and emergent persistence.

Does NOT use the unified Simulator — its dynamics are fundamentally
different from the exact density-matrix pipeline.

Demonstrates:
  - Cells as effective subsystems with Bloch-vector state
  - Selection pressure: high-entropy cells die (leakage threshold)
  - Radiative cooling: entropy export restores purity
  - Reproduction with mutation: stable cells replicate

Run:
    uv run python -m examples.physics_qg.ca_model
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from popgp import validation_json

_PKG_DIR = Path(__file__).parent

# ── Configuration ────────────────────────────────────────────────────────
# All parameters labeled per docs/framework.md §4.6.3.

np.random.seed(42)

WIDTH = 10                      # [STRUCTURAL_CHOICE] grid dimension
HEIGHT = 10                     # [STRUCTURAL_CHOICE] grid dimension
STEPS = 50                      # [TUNABLE_HYPERPARAMETER] simulation length
LEAKAGE_THRESHOLD = 0.4         # [TUNABLE_HYPERPARAMETER] entropy survival cutoff
REPLICATION_PROB = 0.05         # [TUNABLE_HYPERPARAMETER] reproduction probability
MUTATION_RATE = 0.02            # [TUNABLE_HYPERPARAMETER] Bloch-vector noise on replication
COOLING_PROB = 0.02             # [TUNABLE_HYPERPARAMETER] probability of radiative cooling per step
INITIAL_DENSITY = 0.4           # [TUNABLE_HYPERPARAMETER] fraction of grid initially occupied
DECAY_RATE = 0.3                # [TUNABLE_HYPERPARAMETER] neighbor-mismatch decay
ALIGN_STRENGTH = 0.1            # [TUNABLE_HYPERPARAMETER] alignment force between neighbors
DT = 0.1                        # [TUNABLE_HYPERPARAMETER] interaction timestep
REPRO_PURITY_THRESHOLD = 0.1    # [TUNABLE_HYPERPARAMETER] max entropy for reproduction eligibility
_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

results = _PKG_DIR / "results"
results.mkdir(parents=True, exist_ok=True)

# ── Cell helpers ─────────────────────────────────────────────────────────


def _random_bloch():
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi)
    return np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)


def _entropy_bloch(rx, ry, rz):
    r = min(np.sqrt(rx**2 + ry**2 + rz**2), 0.9999)
    p1, p2 = (1 + r) / 2, (1 - r) / 2
    return -p1 * np.log(p1) - p2 * np.log(p2)


def _purity_bloch(rx, ry, rz):
    return (1 + rx**2 + ry**2 + rz**2) / 2


def _interact(b1, b2):
    """Pairwise interaction: Bloch-vector decay + alignment."""
    rx1, ry1, rz1 = b1
    rx2, ry2, rz2 = b2
    dot = rx1 * rx2 + ry1 * ry2 + rz1 * rz2
    decay = 1.0 - DECAY_RATE * (1.0 - dot**2) * DT

    rx1, ry1, rz1 = rx1 * decay, ry1 * decay, rz1 * decay
    rx2, ry2, rz2 = rx2 * decay, ry2 * decay, rz2 * decay

    a = ALIGN_STRENGTH * DT
    ax = 0.5 * (rx1 + rx2)
    ay = 0.5 * (ry1 + ry2)
    az = 0.5 * (rz1 + rz2)
    rx1 += a * (ax - rx1)
    ry1 += a * (ay - ry1)
    rz1 += a * (az - rz1)
    rx2 += a * (ax - rx2)
    ry2 += a * (ay - ry2)
    rz2 += a * (az - rz2)

    return (rx1, ry1, rz1), (rx2, ry2, rz2)


# ── Initialize grid ─────────────────────────────────────────────────────

grid: dict[tuple[int, int], tuple[float, float, float]] = {}
for _ in range(int(WIDTH * HEIGHT * INITIAL_DENSITY)):
    x, y = np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)
    if (x, y) not in grid:
        grid[(x, y)] = _random_bloch()

print(f"Initialized {len(grid)} cells on {WIDTH}x{HEIGHT} grid.")

# ── Simulation loop ──────────────────────────────────────────────────────

history_entropy: list[float] = []
history_count: list[int] = []
frames: list[np.ndarray] = []

print("Running simulation...")
for step in range(STEPS):
    visited: set[tuple[int, int, int, int]] = set()
    for (x, y) in list(grid):
        for dx, dy in _NEIGHBORS:
            nx, ny = x + dx, y + dy
            if (nx, ny) in grid and (nx, ny, x, y) not in visited:
                grid[(x, y)], grid[(nx, ny)] = _interact(
                    grid[(x, y)], grid[(nx, ny)],
                )
                visited.add((x, y, nx, ny))

    new_grid: dict[tuple[int, int], tuple[float, float, float]] = {}
    avg_ent = 0.0
    for (x, y), (rx, ry, rz) in grid.items():
        ent = _entropy_bloch(rx, ry, rz)

        if ent > LEAKAGE_THRESHOLD:
            continue

        if np.random.random() < COOLING_PROB:
            norm = np.sqrt(rx**2 + ry**2 + rz**2)
            if norm > 1e-6:
                rx, ry, rz = rx / norm, ry / norm, rz / norm
            ent = 0.0

        if ent < REPRO_PURITY_THRESHOLD and np.random.random() < REPLICATION_PROB:
            candidates = [(x + dx, y + dy) for dx, dy in _NEIGHBORS]
            np.random.shuffle(candidates)
            for cx, cy in candidates:
                if 0 <= cx < WIDTH and 0 <= cy < HEIGHT and (cx, cy) not in new_grid:
                    mrx = rx + np.random.normal(0, MUTATION_RATE)
                    mry = ry + np.random.normal(0, MUTATION_RATE)
                    mrz = rz + np.random.normal(0, MUTATION_RATE)
                    mn = np.sqrt(mrx**2 + mry**2 + mrz**2)
                    new_grid[(cx, cy)] = (mrx / mn, mry / mn, mrz / mn)
                    break

        new_grid[(x, y)] = (rx, ry, rz)
        avg_ent += ent

    grid = new_grid
    count = len(grid)
    history_entropy.append(avg_ent / count if count > 0 else 0)
    history_count.append(count)

    frame = np.zeros((WIDTH, HEIGHT))
    for (x, y), bloch in grid.items():
        frame[x, y] = _purity_bloch(*bloch)
    frames.append(frame)

# ── Visualization ────────────────────────────────────────────────────────

final_pop = history_count[-1] if history_count else 0
initial_pop = history_count[0] if history_count else 0
pop_survived = final_pop > 0
pop_grew = final_pop >= initial_pop
final_entropy = history_entropy[-1] if history_entropy else 0
entropy_below = final_entropy < LEAKAGE_THRESHOLD

fig, ax1 = plt.subplots(figsize=(9, 5))
color_pop = "tab:red"
ax1.set_xlabel("Phase-Order Steps")
ax1.set_ylabel("Population", color=color_pop)
ax1.plot(history_count, color=color_pop, linewidth=2, label="Population")
ax1.tick_params(axis="y", labelcolor=color_pop)
ax1.axhline(0, color="red", linestyle=":", alpha=0.3)

ax2 = ax1.twinx()
color_ent = "tab:blue"
ax2.set_ylabel("Avg Entropy (Leakage)", color=color_ent)
ax2.plot(history_entropy, color=color_ent, linewidth=2, label="Avg Entropy")
ax2.tick_params(axis="y", labelcolor=color_ent)

ax2.axhline(LEAKAGE_THRESHOLD, color="darkblue", linestyle="--", linewidth=1.5,
            alpha=0.7, label=f"Death threshold = {LEAKAGE_THRESHOLD}")
ax2.legend(loc="upper right", fontsize=8)

verdict = "SURVIVES" if pop_grew else "NEGATIVE"
verdict_color = "green" if pop_grew else "darkorange"
ax1.text(
    0.98, 0.5, verdict, transform=ax1.transAxes,
    fontsize=18, fontweight="bold", color="white",
    ha="right", va="center",
    bbox=dict(boxstyle="round,pad=0.4", fc=verdict_color, alpha=0.9),
)

fig.suptitle("Phenomenological CA Run (Cooling Configured, No Control)", fontsize=13)
fig.tight_layout()
fig.savefig(results / "dynamics_cooling.png", dpi=150)
print(f"Saved: {results / 'dynamics_cooling.png'}")

fig_anim, ax_anim = plt.subplots()
im = ax_anim.imshow(frames[0], cmap="inferno", vmin=0, vmax=1)
plt.title("Grid Evolution (Cooling Enabled)")


def update(i):
    if i < len(frames):
        im.set_array(frames[i])
    return [im]


ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), blit=True)
ani.save(results / "evolution_cooling.gif", writer="pillow", fps=10)
print(f"Saved: {results / 'evolution_cooling.gif'}")

# ── Validation Report ─────────────────────────────────────────────────────

peak_pop = max(history_count)
min_pop = min(history_count)
avg_entropy_final_5 = (
    float(np.mean(history_entropy[-5:]))
    if len(history_entropy) >= 5
    else final_entropy
)

report = {
    "example": "ca_model",
    "scientific_status": "phenomenological_analogy_negative_population_result",
    "framework_version": "1.0-submission-draft",
    "package_version": "0.1.0",
    "config": {
        "width": WIDTH,
        "height": HEIGHT,
        "steps": STEPS,
        "leakage_threshold": LEAKAGE_THRESHOLD,
        "replication_prob": REPLICATION_PROB,
        "mutation_rate": MUTATION_RATE,
        "cooling_prob": COOLING_PROB,
        "initial_density": INITIAL_DENSITY,
        "decay_rate": DECAY_RATE,
        "align_strength": ALIGN_STRENGTH,
        "dt": DT,
        "repro_purity_threshold": REPRO_PURITY_THRESHOLD,
        "random_seed": 42,
    },
    "pipeline": {
        "dynamics": {
            "initial_population": initial_pop,
            "final_population": final_pop,
            "peak_population": peak_pop,
            "min_population": min_pop,
            "final_avg_entropy": float(final_entropy),
            "avg_entropy_last_5_steps": avg_entropy_final_5,
            "population_history": history_count,
            "entropy_history": [float(e) for e in history_entropy],
        },
    },
    "checks": [
        {
            "name": "population_survival",
            "description": "Population survives to end of simulation (non-zero)",
            "framework_section": "4.4.2a",
            "criterion": "final_population > 0",
            "value": final_pop,
            "passed": pop_survived,
        },
        {
            "name": "entropy_below_threshold",
            "description": (
                "Average entropy of surviving cells is below the leakage death threshold"
            ),
            "framework_section": "4.4.2a",
            "criterion": f"final_avg_entropy < {LEAKAGE_THRESHOLD}",
            "value": float(final_entropy),
            "threshold": LEAKAGE_THRESHOLD,
            "passed": entropy_below,
        },
        {
            "name": "population_growth",
            "description": "Population at end >= population at start (stable or growing)",
            "framework_section": "4.4.2a",
            "criterion": "final_population >= initial_population",
            "value": {"initial": initial_pop, "final": final_pop},
            "passed": pop_grew,
        },
        {
            "name": "survivor_entropy_filter_regression",
            "description": (
                "Survivor entropy remains below the configured culling threshold by "
                "construction; this does not measure a cooling effect"
            ),
            "framework_section": "4.4.2a",
            "criterion": "entropy_last_5 <= leakage_threshold",
            "value": avg_entropy_final_5,
            "threshold": LEAKAGE_THRESHOLD,
            "severity": "informational",
            "passed": avg_entropy_final_5 < LEAKAGE_THRESHOLD,
        },
    ],
}

report["overall_pass"] = all(
    c["passed"] for c in report["checks"] if c.get("severity") != "informational"
)
report["artifacts"] = [
    "results/dynamics_cooling.png",
    "results/evolution_cooling.gif",
    "results/validation.json",
]

val_path = results / "validation.json"
val_path.write_text(validation_json(report) + "\n", encoding="utf-8")
print(f"Saved: {val_path}")

print("\nDone.")
