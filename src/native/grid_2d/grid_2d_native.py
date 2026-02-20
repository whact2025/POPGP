"""
2D grid emergent geometry using the CUDA phase-flow kernel.

Native-engine analogue of src/toy/grid_2d/grid_2d.py.
Implements correlation-based locality (Section 4.4.3) and MDS embedding
(Section 4.4.4) of docs/framework.md at GPU scale.

The CUDA kernel implements the full Heisenberg mean-field interaction
which drives dynamic Sz evolution.  Time-averaged Sz correlations between
cells encode graph distance and are used for MDS geometry recovery.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from popgp.engine import Engine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
torch.manual_seed(42)

WIDTH = 30
HEIGHT = 30
N = WIDTH * HEIGHT  # 900 cells (vs 9 in the toy model)
dt = 0.2            # Phase-order step size (tunable hyperparameter)
WARMUP_STEPS = 500
MEASURE_STEPS = 2000
SAMPLE_CELLS = 150

device = torch.device("cuda")
_results = Path(__file__).parent / "results"

print(f"Initializing {WIDTH}x{HEIGHT} = {N} cell grid on {device}")

# ---------------------------------------------------------------------------
# 1. Initialize Cells -- perturbed uniform state (near |0⟩)
# ---------------------------------------------------------------------------
# Perturbation amplitude is a tunable hyperparameter.
PERTURBATION = 0.15

theta = torch.randn(N, device=device) * PERTURBATION
phi = torch.rand(N, device=device) * 2 * np.pi

alphas = torch.cos(theta / 2).to(torch.complex128)
betas = (torch.sin(theta / 2) * torch.exp(1j * phi)).to(torch.complex128)

# ---------------------------------------------------------------------------
# 2. Build 2D Grid Graph with checkerboard coloring
# ---------------------------------------------------------------------------
def flat(x, y):
    return y * WIDTH + x

h_src, h_dst = [], []
for y in range(HEIGHT):
    for x in range(WIDTH - 1):
        h_src.append(flat(x, y))
        h_dst.append(flat(x + 1, y))

v_src, v_dst = [], []
for y in range(HEIGHT - 1):
    for x in range(WIDTH):
        v_src.append(flat(x, y))
        v_dst.append(flat(x, y + 1))

ground_truth_edges = list(zip(h_src + v_src, h_dst + v_dst))

all_src = h_src + v_src
all_dst = h_dst + v_dst

color_a_src, color_a_dst = [], []
color_b_src, color_b_dst = [], []

for s, d in zip(all_src, all_dst):
    sx, sy = s % WIDTH, s // WIDTH
    if (sx + sy) % 2 == 0:
        color_a_src.append(s)
        color_a_dst.append(d)
    else:
        color_b_src.append(s)
        color_b_dst.append(d)

def to_gpu(lst):
    return torch.tensor(lst, dtype=torch.int32, device=device)

a_src = to_gpu(color_a_src)
a_dst = to_gpu(color_a_dst)
a_w = torch.ones(len(color_a_src), dtype=torch.float64, device=device)

b_src = to_gpu(color_b_src)
b_dst = to_gpu(color_b_dst)
b_w = torch.ones(len(color_b_src), dtype=torch.float64, device=device)

print(f"Edges: {a_src.numel()} (color A) + {b_src.numel()} (color B)")

# ---------------------------------------------------------------------------
# 3. Phase-flow evolution + time-averaged correlation accumulation
# ---------------------------------------------------------------------------
engine = Engine(precision="double")

sample_idx = torch.linspace(0, N - 1, SAMPLE_CELLS, device=device).long()

sz_sum = torch.zeros(SAMPLE_CELLS, device=device, dtype=torch.float64)
sz_sq_sum = torch.zeros(SAMPLE_CELLS, SAMPLE_CELLS, device=device, dtype=torch.float64)

steps_total = WARMUP_STEPS + MEASURE_STEPS
print(f"Running {steps_total} phase-flow steps "
      f"(warmup={WARMUP_STEPS}, measure={MEASURE_STEPS})...")
torch.cuda.synchronize()
t0 = time.time()

for s in range(steps_total):
    engine.step(alphas, betas, a_src, a_dst, a_w, dt)
    engine.step(alphas, betas, b_src, b_dst, b_w, dt)

    if s >= WARMUP_STEPS:
        sz = (alphas.abs() ** 2 - betas.abs() ** 2).real
        sz_s = sz[sample_idx]
        sz_sum += sz_s
        sz_sq_sum += torch.outer(sz_s, sz_s)

torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"Done in {elapsed:.2f}s  ({N * steps_total / elapsed:.2e} cell-updates/sec)")

# ---------------------------------------------------------------------------
# 4. Time-averaged Sz correlation matrix
# ---------------------------------------------------------------------------
print("Building time-averaged correlation matrix...")
M = MEASURE_STEPS
sz_mean = sz_sum / M
cov = (sz_sq_sum / M - torch.outer(sz_mean, sz_mean)).cpu().float()

std = cov.diag().clamp(min=1e-12).sqrt()
corr_matrix = cov / (std.unsqueeze(1) * std.unsqueeze(0))
corr_matrix.fill_diagonal_(1.0)

dist_matrix = -torch.log(corr_matrix.abs() + 1e-9)
dist_matrix.fill_diagonal_(0.0)

# ---------------------------------------------------------------------------
# 5. MDS Embedding -- recover 2D geometry
# ---------------------------------------------------------------------------
print("Running MDS...")
D2 = dist_matrix.double() ** 2
n = SAMPLE_CELLS
J = (torch.eye(n) - torch.ones((n, n)) / n).double()
B = -0.5 * J @ D2 @ J

vals, vecs = torch.linalg.eigh(B)
idx_sort = vals.argsort(descending=True)
vals = vals[idx_sort]
vecs = vecs[:, idx_sort]

coords = (vecs[:, :2] @ torch.diag(torch.sqrt(torch.abs(vals[:2])))).real.numpy()

# ---------------------------------------------------------------------------
# 6. Validation -- neighbors should be closer than non-neighbors
# ---------------------------------------------------------------------------
sample_np = sample_idx.cpu().numpy()
sample_xy = np.column_stack([sample_np % WIDTH, sample_np // WIDTH])

gt_neighbor_dists = []
gt_other_dists = []

for i in range(len(sample_np)):
    for j in range(i + 1, len(sample_np)):
        manhattan = abs(sample_xy[i][0] - sample_xy[j][0]) + abs(sample_xy[i][1] - sample_xy[j][1])
        emb_dist = np.linalg.norm(coords[i] - coords[j])
        if manhattan == 1:
            gt_neighbor_dists.append(emb_dist)
        elif manhattan >= 3:
            gt_other_dists.append(emb_dist)

avg_neigh = np.mean(gt_neighbor_dists) if gt_neighbor_dists else float("nan")
avg_other = np.mean(gt_other_dists) if gt_other_dists else float("nan")

print(f"Avg embedded dist (grid-neighbors):     {avg_neigh:.4f}")
print(f"Avg embedded dist (grid-non-neighbors):  {avg_other:.4f}")
if avg_neigh < avg_other:
    print("SUCCESS: Local structure preserved.")
else:
    print("FAILURE: Geometry distorted.")

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
_results.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(coords[:, 0], coords[:, 1], c="blue", s=8, alpha=0.6)

sample_set = set(sample_np.tolist())
sample_to_local = {s: i for i, s in enumerate(sample_np)}
for s, d in ground_truth_edges:
    if s in sample_set and d in sample_set:
        i, j = sample_to_local[s], sample_to_local[d]
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            "k-", alpha=0.15, linewidth=0.5,
        )

ax.set_title(f"Emergent 2D Geometry ({WIDTH}x{HEIGHT}, CUDA Heisenberg kernel)")
ax.set_xlabel("Emergent Dimension 1")
ax.set_ylabel("Emergent Dimension 2")
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.4)
fig.savefig(_results / "embedding.png")
print(f"Saved {_results / 'embedding.png'}")
