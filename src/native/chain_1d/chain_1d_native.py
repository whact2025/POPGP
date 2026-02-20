"""
1D chain stability and geometry recovery using the CUDA phase-flow kernel.

Native-engine analogue of src/toy/chain_1d/chain_1d_stability.py.
Implements stability selection (Section 4.4.2a), correlation-based locality
(Section 4.4.3), and MDS embedding (Section 4.4.4) of docs/framework.md,
but at GPU scale using the compiled popgp kernel.

The CUDA kernel implements the full Heisenberg mean-field interaction
(Sx⊗Sx + Sy⊗Sy + Sz⊗Sz), which exchanges amplitude between cells and
drives dynamic Sz evolution.  Time-averaged Sz correlations encode the
graph topology and are used for MDS geometry recovery.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from popgp.engine import Engine

# ---------------------------------------------------------------------------
# Configuration (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
torch.manual_seed(42)

N = 100            # Number of cells (vs 4 in the toy model)
dt = 0.3           # Phase-order step size (tunable hyperparameter)
WARMUP_STEPS = 1000 # Let dynamics settle before measuring
MEASURE_STEPS = 5000 # Steps over which to accumulate time-averaged correlations
SAMPLE_CELLS = 100 # All cells used (no subsampling at this scale)

device = torch.device("cuda")
_results = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# 1. Initialize Cells -- perturbed uniform state (near |0⟩)
# ---------------------------------------------------------------------------
# A perturbed uniform state breaks the symmetry just enough for spin-wave
# correlations to develop via the Heisenberg interaction.  Perturbation
# amplitude PERTURBATION is a tunable hyperparameter.
PERTURBATION = 0.15

print(f"Initializing {N}-cell 1D chain on {device}...")

theta = torch.randn(N, device=device) * PERTURBATION
phi = torch.rand(N, device=device) * 2 * np.pi

alphas = torch.cos(theta / 2).to(torch.complex128)
betas = (torch.sin(theta / 2) * torch.exp(1j * phi)).to(torch.complex128)

# ---------------------------------------------------------------------------
# 2. Build Graph -- 1D chain with red/black edge coloring
# ---------------------------------------------------------------------------
indices = torch.arange(N - 1, device=device)

red_src = indices[indices % 2 == 0]
red_dst = red_src + 1
red_w = torch.ones_like(red_src, dtype=torch.float64)

black_src = indices[indices % 2 == 1]
black_dst = black_src + 1
black_w = torch.ones_like(black_src, dtype=torch.float64)

print(f"Graph: {red_src.numel()} red + {black_src.numel()} black edges")

engine = Engine(precision="double")

# ---------------------------------------------------------------------------
# 3. Evolution + time-averaged correlation accumulation
# ---------------------------------------------------------------------------
K = 2
num_pairs = N // K
local_a = torch.arange(num_pairs, device=device) * K
local_b = local_a + 1
half = N // 2
n_nonlocal = min(num_pairs, half)
nonlocal_a = torch.arange(n_nonlocal, device=device)
nonlocal_b = nonlocal_a + half

sample_idx = torch.linspace(0, N - 1, SAMPLE_CELLS, device=device).long()

# Accumulators for time-averaged correlation
sz_sum = torch.zeros(SAMPLE_CELLS, device=device, dtype=torch.float64)
sz_sq_sum = torch.zeros(SAMPLE_CELLS, SAMPLE_CELLS, device=device, dtype=torch.float64)

corr_local_ts = torch.zeros(MEASURE_STEPS, device="cpu")
corr_nonlocal_ts = torch.zeros(MEASURE_STEPS, device="cpu")

steps_total = WARMUP_STEPS + MEASURE_STEPS
print(f"Phase-flow evolution: {steps_total} steps "
      f"(warmup={WARMUP_STEPS}, measure={MEASURE_STEPS})...")
torch.cuda.synchronize()
t0 = time.time()

for s in range(steps_total):
    engine.step(alphas, betas, red_src.int(), red_dst.int(), red_w, dt)
    engine.step(alphas, betas, black_src.int(), black_dst.int(), black_w, dt)

    if s >= WARMUP_STEPS:
        mi = s - WARMUP_STEPS
        sz = (alphas.abs() ** 2 - betas.abs() ** 2).real

        # Per-step stability metric
        corr_local_ts[mi] = (sz[local_a] * sz[local_b]).mean().item()
        corr_nonlocal_ts[mi] = (sz[nonlocal_a] * sz[nonlocal_b]).mean().item()

        # Accumulate for time-averaged correlation matrix
        sz_s = sz[sample_idx]
        sz_sum += sz_s
        sz_sq_sum += torch.outer(sz_s, sz_s)

torch.cuda.synchronize()
elapsed = time.time() - t0

print(f"Simulation complete: {elapsed:.2f}s")
print(f"Throughput: {N * steps_total / elapsed:.2e} cell-updates/sec")
print(f"Avg Sz correlation (local pairs):     {corr_local_ts.mean():.6f}")
print(f"Avg Sz correlation (non-local pairs): {corr_nonlocal_ts.mean():.6f}")

# ---------------------------------------------------------------------------
# 4. Time-averaged correlation matrix
# ---------------------------------------------------------------------------
print("Building time-averaged correlation matrix...")
M = MEASURE_STEPS
sz_mean = sz_sum / M
# Cov(i,j) = <Sz_i Sz_j>_t - <Sz_i>_t <Sz_j>_t
cov = (sz_sq_sum / M - torch.outer(sz_mean, sz_mean)).cpu().float()

# Normalise to Pearson correlation
std = cov.diag().clamp(min=1e-12).sqrt()
corr_matrix = cov / (std.unsqueeze(1) * std.unsqueeze(0))
corr_matrix.fill_diagonal_(1.0)

dist_matrix = -torch.log(corr_matrix.abs() + 1e-9)
dist_matrix.fill_diagonal_(0.0)

# ---------------------------------------------------------------------------
# 5. MDS Embedding -- recover 1D geometry
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

coord = (vecs[:, 0] * torch.sqrt(torch.abs(vals[0]))).real.numpy()

sorted_order = np.argsort(coord)
true_positions = sample_idx.cpu().numpy()
rank_corr = np.corrcoef(coord, true_positions)[0, 1]

print(f"Recovered 1D order (first 10): {sorted_order[:10]}")
print(f"Rank correlation (MDS vs true): {rank_corr:.4f}")

# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
_results.mkdir(parents=True, exist_ok=True)

# Stability: Sz correlation over time
fig, ax = plt.subplots()
t_axis = np.arange(MEASURE_STEPS) * dt
ax.plot(t_axis, corr_local_ts.numpy(), "b-", label="Local pairs (contiguous)", alpha=0.7)
ax.plot(t_axis, corr_nonlocal_ts.numpy(), "r--", label="Non-local pairs (scattered)", alpha=0.7)
ax.set_xlabel("Phase Order (s)")
ax.set_ylabel("Mean ⟨Sz_i · Sz_j⟩")
ax.set_title(f"Stability Selection via Sz Correlation (N={N})")
ax.legend()
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
fig.savefig(_results / "stability.png")
print(f"Saved {_results / 'stability.png'}")

# Geometry: MDS coordinate vs true chain position
fig2, ax2 = plt.subplots()
ax2.scatter(coord, true_positions, s=4, alpha=0.6)
ax2.set_xlabel("MDS Coordinate")
ax2.set_ylabel("True Chain Position")
ax2.set_title(f"Geometry Recovery: MDS vs True Position (r={rank_corr:.3f})")
ax2.grid(True, linestyle=":", alpha=0.4)
fig2.savefig(_results / "embedding.png")
print(f"Saved {_results / 'embedding.png'}")
