"""
Cellular automata stability selection using the CUDA phase-flow kernel.

Native-engine analogue of src/toy/ca_model/ca_model.py.
Demonstrates stability selection (Section 4.4.2a) and emergent persistence
via the compiled CUDA kernel for the interaction phase.  Selection, cooling,
and reproduction remain in Python.  See docs/framework.md.

The CUDA kernel applies the full Heisenberg mean-field interaction
(Sx⊗Sx + Sy⊗Sy + Sz⊗Sz), which exchanges amplitude and evolves
Sz dynamically.  After each kernel step we apply a decoherence channel
(Bloch-vector shrinkage proportional to neighbour misalignment) and
an alignment force on the CPU.  This mirrors the two effects of the
toy model's ``interact()`` function.
"""

import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from popgp.engine import Engine

# ---------------------------------------------------------------------------
# Configuration (fixed for reproducibility)
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

WIDTH = 50
HEIGHT = 50
N = WIDTH * HEIGHT
STEPS = 80
dt = 0.05            # Phase-flow step size (tunable hyperparameter)

# Decoherence parameters (mirroring the toy model's interact())
DECAY_J = 0.3        # Purity-decay coupling (tunable hyperparameter)
ALIGN_J = 0.1        # Alignment coupling (tunable hyperparameter)

LEAKAGE_THRESHOLD = 0.4
REPLICATION_PROB = 0.05
MUTATION_RATE = 0.02
COOLING_PROB = 0.02
INITIAL_DENSITY = 0.4

device = torch.device("cuda")
_results = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bloch_to_ab(rx, ry, rz):
    """Bloch vector -> qubit amplitudes (alpha, beta)."""
    r = np.sqrt(rx * rx + ry * ry + rz * rz)
    r = np.clip(r, 1e-12, None)
    theta = np.arccos(np.clip(rz / r, -1, 1))
    phi = np.arctan2(ry, rx)
    scale = np.sqrt((1 + r) / 2)
    alpha = scale * np.cos(theta / 2)
    beta = scale * np.sin(theta / 2) * np.exp(1j * phi)
    return complex(alpha), complex(beta)


def ab_to_bloch(alpha, beta):
    """Qubit amplitudes -> Bloch vector (rx, ry, rz)."""
    rx = 2.0 * (alpha * beta.conjugate()).real
    ry = 2.0 * (alpha * beta.conjugate()).imag
    rz = abs(alpha) ** 2 - abs(beta) ** 2
    return float(rx), float(ry), float(rz)


def bloch_entropy(rx, ry, rz):
    """Von Neumann entropy from Bloch vector magnitude."""
    r = min(np.sqrt(rx * rx + ry * ry + rz * rz), 0.9999)
    p1 = (1 + r) / 2
    p2 = (1 - r) / 2
    return -p1 * np.log(p1) - p2 * np.log(p2)


def bloch_purity(rx, ry, rz):
    return (1 + rx * rx + ry * ry + rz * rz) / 2.0


# ---------------------------------------------------------------------------
# 1. Initialize grid -- sparse random pure states
# ---------------------------------------------------------------------------
occupied: dict[int, tuple[float, float, float]] = {}

count_init = int(N * INITIAL_DENSITY)
positions = np.random.choice(N, size=count_init, replace=False)
for pos in positions:
    phi_r = np.random.uniform(0, 2 * np.pi)
    theta_r = np.random.uniform(0, np.pi)
    occupied[int(pos)] = (
        float(np.sin(theta_r) * np.cos(phi_r)),
        float(np.sin(theta_r) * np.sin(phi_r)),
        float(np.cos(theta_r)),
    )

print(f"Initialized {len(occupied)} cells on {WIDTH}x{HEIGHT} grid ({device})")

engine = Engine(precision="double")

# ---------------------------------------------------------------------------
# 2. Build grid edge set (two checkerboard colours)
# ---------------------------------------------------------------------------

def build_edges():
    a_s, a_d, b_s, b_d = [], [], [], []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            idx = y * WIDTH + x
            if x + 1 < WIDTH:
                r = y * WIDTH + (x + 1)
                (a_s if (x + y) % 2 == 0 else b_s).append(idx)
                (a_d if (x + y) % 2 == 0 else b_d).append(r)
            if y + 1 < HEIGHT:
                d = (y + 1) * WIDTH + x
                (a_s if (x + y) % 2 == 0 else b_s).append(idx)
                (a_d if (x + y) % 2 == 0 else b_d).append(d)
    return a_s, a_d, b_s, b_d


ea_s, ea_d, eb_s, eb_d = build_edges()

def to_gpu(lst):
    return torch.tensor(lst, dtype=torch.int32, device=device)

a_src, a_dst = to_gpu(ea_s), to_gpu(ea_d)
a_w = torch.ones(len(ea_s), dtype=torch.float64, device=device)
b_src, b_dst = to_gpu(eb_s), to_gpu(eb_d)
b_w = torch.ones(len(eb_s), dtype=torch.float64, device=device)

alphas = torch.zeros(N, dtype=torch.complex128, device=device)
betas = torch.zeros(N, dtype=torch.complex128, device=device)

# ---------------------------------------------------------------------------
# 3. Upload / download helpers
# ---------------------------------------------------------------------------

def upload_state():
    h_a = np.zeros(N, dtype=np.complex128)
    h_b = np.zeros(N, dtype=np.complex128)
    for idx, (rx, ry, rz) in occupied.items():
        a, b = bloch_to_ab(rx, ry, rz)
        h_a[idx] = a
        h_b[idx] = b
    alphas.copy_(torch.from_numpy(h_a).to(device))
    betas.copy_(torch.from_numpy(h_b).to(device))


def download_state():
    h_a = alphas.cpu().numpy()
    h_b = betas.cpu().numpy()
    for idx in list(occupied.keys()):
        rx, ry, rz = ab_to_bloch(h_a[idx], h_b[idx])
        occupied[idx] = (rx, ry, rz)


# ---------------------------------------------------------------------------
# 4. Decoherence -- purity decay + alignment (CPU, mirrors toy interact())
# ---------------------------------------------------------------------------

def apply_decoherence():
    """Shrink Bloch vectors + align neighbours (non-unitary channel)."""
    updates: dict[int, tuple[float, float, float]] = {}

    for idx, (rx, ry, rz) in occupied.items():
        x, y = idx % WIDTH, idx // WIDTH
        neigh_coords = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        acc_rx, acc_ry, acc_rz = rx, ry, rz
        for nx, ny in neigh_coords:
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                nidx = ny * WIDTH + nx
                nb = occupied.get(nidx)
                if nb is None:
                    continue
                nrx, nry, nrz = nb
                dot = rx * nrx + ry * nry + rz * nrz
                decay = DECAY_J * (1.0 - dot * dot) * dt
                factor = max(1.0 - decay, 0.0)
                acc_rx *= factor
                acc_ry *= factor
                acc_rz *= factor
                acc_rx += ALIGN_J * dt * (0.5 * (rx + nrx) - rx)
                acc_ry += ALIGN_J * dt * (0.5 * (ry + nry) - ry)
                acc_rz += ALIGN_J * dt * (0.5 * (rz + nrz) - rz)

        updates[idx] = (acc_rx, acc_ry, acc_rz)

    occupied.update(updates)


# ---------------------------------------------------------------------------
# 5. Main simulation loop
# ---------------------------------------------------------------------------
history_entropy: list[float] = []
history_count: list[int] = []
frames: list[np.ndarray] = []

print(f"Running {STEPS} steps...")
t0 = time.time()

for step in range(STEPS):
    # Coherent interaction (GPU kernel)
    upload_state()
    engine.step(alphas, betas, a_src, a_dst, a_w, dt)
    engine.step(alphas, betas, b_src, b_dst, b_w, dt)
    download_state()

    # Non-unitary decoherence channel (CPU)
    apply_decoherence()

    # Selection / Cooling / Reproduction
    new_occupied: dict[int, tuple[float, float, float]] = {}
    avg_ent = 0.0

    for idx, (rx, ry, rz) in list(occupied.items()):
        ent = bloch_entropy(rx, ry, rz)

        if ent > LEAKAGE_THRESHOLD:
            continue

        if np.random.random() < COOLING_PROB:
            norm = np.sqrt(rx * rx + ry * ry + rz * rz)
            if norm > 1e-6:
                rx, ry, rz = rx / norm, ry / norm, rz / norm
            ent = 0.0

        new_occupied[idx] = (rx, ry, rz)
        avg_ent += ent

        if ent < 0.1 and np.random.random() < REPLICATION_PROB:
            x, y = idx % WIDTH, idx // WIDTH
            neigh = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            np.random.shuffle(neigh)
            for nx, ny in neigh:
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    nidx = ny * WIDTH + nx
                    if nidx not in new_occupied:
                        mrx = rx + np.random.normal(0, MUTATION_RATE)
                        mry = ry + np.random.normal(0, MUTATION_RATE)
                        mrz = rz + np.random.normal(0, MUTATION_RATE)
                        norm = np.sqrt(mrx ** 2 + mry ** 2 + mrz ** 2)
                        new_occupied[nidx] = (mrx / norm, mry / norm, mrz / norm)
                        break

    occupied = new_occupied
    count = len(occupied)
    history_count.append(count)
    history_entropy.append(avg_ent / count if count > 0 else 0.0)

    frame = np.zeros((WIDTH, HEIGHT))
    for idx, (rx, ry, rz) in occupied.items():
        x, y = idx % WIDTH, idx // WIDTH
        frame[x, y] = bloch_purity(rx, ry, rz)
    frames.append(frame)

elapsed = time.time() - t0
print(f"Done in {elapsed:.2f}s  (final pop: {len(occupied)})")

# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------
_results.mkdir(parents=True, exist_ok=True)

fig, ax1 = plt.subplots()
ax1.set_xlabel("Time Steps")
ax1.set_ylabel("Population", color="tab:red")
ax1.plot(history_count, color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.set_ylabel("Avg Entropy (Leakage)", color="tab:blue")
ax2.plot(history_entropy, color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title(f"Emergent Stability ({WIDTH}x{HEIGHT}, CUDA kernel)")
plt.tight_layout()
fig.savefig(_results / "dynamics_cooling.png")
print(f"Saved {_results / 'dynamics_cooling.png'}")

if frames:
    fig_anim, ax_anim = plt.subplots()
    im = ax_anim.imshow(frames[0], cmap="inferno", vmin=0, vmax=1)
    plt.title("Grid Evolution (CUDA Kernel + Cooling)")

    def update(frame_idx):
        if frame_idx >= len(frames):
            return [im]
        im.set_array(frames[frame_idx])
        return [im]

    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), blit=True)
    ani.save(_results / "evolution_cooling.gif", writer="pillow", fps=10)
    print(f"Saved {_results / 'evolution_cooling.gif'}")
