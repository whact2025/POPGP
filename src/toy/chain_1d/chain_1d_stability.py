"""
1D chain stability and geometry recovery (POPGP toy model).

Implements stability selection (Section 4.4.2a), locality from mutual information
(Section 4.4.3), and MDS embedding (Section 4.4.4) of docs/framework.md.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parameters (fixed for reproducibility)
N = 8  # Number of qubits
dim = 2**N
k = 2  # Block size for cells (N/k cells)
num_cells = N // k

# Tunable hyperparameter: inverse temperature for thermal state
beta = 1.0

device = torch.device('cpu')  # Use CPU for small problem

# --- 1. Define Substrate (Heisenberg Hamiltonian) ---
# H = sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
Sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device) / 2
Sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device) / 2
Sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device) / 2
I = torch.eye(2, dtype=torch.complex128, device=device)

def tensor_op(op, site, N):
    ops = [I] * N
    ops[site] = op
    res = ops[0]
    for i in range(1, N):
        res = torch.kron(res, ops[i])
    return res

H = torch.zeros((dim, dim), dtype=torch.complex128, device=device)
for i in range(N - 1):  # Open boundary conditions
    H += (tensor_op(Sx, i, N) @ tensor_op(Sx, i+1, N) +
          tensor_op(Sy, i, N) @ tensor_op(Sy, i+1, N) +
          tensor_op(Sz, i, N) @ tensor_op(Sz, i+1, N))

# Thermal state (beta set above)
# For stability, diagonalize H
evals, evecs = torch.linalg.eigh(H)
exp_H = torch.diag(torch.exp(-beta * evals)).to(dtype=torch.complex128)
rho_total = evecs @ exp_H @ evecs.conj().T
rho_total /= torch.trace(rho_total)

print(f"Substrate defined: {N}-qubit Heisenberg chain (beta={beta})")

# --- 2. Define Cells ---
def get_reduced_rho(rho, keep_indices, N):
    # Reshape to (2, 2, ..., 2) [2N times]
    shape = [2] * (2 * N)
    rho_tensor = rho.view(shape)
    
    trace_indices = [i for i in range(N) if i not in keep_indices]
    
    # Move kept indices to front
    perm = []
    perm += list(keep_indices)
    perm += list(trace_indices)
    perm += [x + N for x in keep_indices]
    perm += [x + N for x in trace_indices]
    
    rho_perm = rho_tensor.permute(perm)
    
    dim_keep = 2**len(keep_indices)
    dim_trace = 2**(N - len(keep_indices))
    
    # (kept_row, trace_row, kept_col, trace_col)
    rho_reshaped = rho_perm.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
    
    # Trace over middle axes: sum_j rho[i, j, k, j]
    rho_red = torch.einsum('ijkj->ik', rho_reshaped)
    
    return rho_red

valid_cells = []
for i in range(num_cells):
    valid_cells.append(list(range(i*k, (i+1)*k)))

invalid_cells = []
for i in range(num_cells):
    # [0, 4], [1, 5], ...
    invalid_cells.append([i, i + num_cells])

# --- 3. Compute Leakage (Stability) ---
# Use a non-stationary state (Neel state) to test dynamic stability
# |psi> = |01010101>
psi_neel = torch.zeros(dim, dtype=torch.complex128, device=device)
# Index for 01010101 is sum(2^i for i in [0, 2, 4, 6]) = 1+4+16+64 = 85? 
# Wait, binary representation.
# Let's construct it properly via kron.
state_0 = torch.tensor([1, 0], dtype=torch.complex128, device=device)
state_1 = torch.tensor([0, 1], dtype=torch.complex128, device=device)
psi_neel = state_0
for i in range(1, N):
    psi_neel = torch.kron(psi_neel, state_1 if i % 2 == 1 else state_0)

rho_neel = torch.outer(psi_neel, psi_neel.conj())

def entropy(rho):
    evals = torch.linalg.eigvalsh(rho)
    # Clamp to avoid log(0)
    evals = evals[evals > 1e-12]
    if len(evals) == 0: return 0.0
    return -torch.sum(evals * torch.log(evals))

# Tunable hyperparameters: phase-order step and number of steps
dt = 0.1
steps = 20
U_dt = evecs @ torch.diag(torch.exp(-1j * evals * dt).to(dtype=torch.complex128)) @ evecs.conj().T

rho_curr = rho_neel.clone()
entropy_valid = torch.zeros((steps, num_cells))
entropy_invalid = torch.zeros((steps, num_cells))

print("Simulating time evolution (Neel state)...")
for t_idx in range(steps):
    if t_idx > 0:
        rho_curr = U_dt @ rho_curr @ U_dt.conj().T
        # Re-normalize to avoid drift
        rho_curr /= torch.trace(rho_curr)
    
    for i, cell in enumerate(valid_cells):
        rho_red = get_reduced_rho(rho_curr, cell, N)
        entropy_valid[t_idx, i] = entropy(rho_red)
        
    for i, cell in enumerate(invalid_cells):
        rho_red = get_reduced_rho(rho_curr, cell, N)
        entropy_invalid[t_idx, i] = entropy(rho_red)

# Measure rate of entropy production (slope)
# Simple linear fit or just max entropy achieved
slope_valid = torch.mean(entropy_valid[-1, :] - entropy_valid[0, :]).item()
slope_invalid = torch.mean(entropy_invalid[-1, :] - entropy_invalid[0, :]).item()

print(f"Entropy Increase (Valid): {slope_valid:.4f}")
print(f"Entropy Increase (Invalid): {slope_invalid:.4f}")

# Plot Entropy Growth
_results = Path(__file__).parent / "results"
_results.mkdir(parents=True, exist_ok=True)
plt.figure()
t_axis = np.arange(steps) * dt
plt.plot(t_axis, entropy_valid.mean(dim=1).numpy(), 'b-', label='Valid Cells (Local)')
plt.plot(t_axis, entropy_invalid.mean(dim=1).numpy(), 'r--', label='Invalid Cells (Non-local)')
plt.xlabel("Phase Order / Time")
plt.ylabel("Avg Cell Entropy")
plt.title("Stability Selection: Local vs Non-local Subsystems")
plt.legend()
plt.savefig(_results / "entropy_growth.png")
print(f"Entropy plot saved to {_results / 'entropy_growth.png'}")


# --- 4. Mutual Information ---
dist_matrix = torch.zeros((num_cells, num_cells))
rhos_1 = []
for i in range(num_cells):
    rhos_1.append(get_reduced_rho(rho_total, valid_cells[i], N))

for i in range(num_cells):
    for j in range(num_cells):
        if i == j: continue
        
        joint = valid_cells[i] + valid_cells[j]
        rho_12 = get_reduced_rho(rho_total, joint, N)
        
        S1 = entropy(rhos_1[i])
        S2 = entropy(rhos_1[j])
        S12 = entropy(rho_12)
        
        I_12 = S1 + S2 - S12
        I_12 = max(0.0, I_12.item())
        
        dist_matrix[i, j] = -np.log(I_12 + 1e-9)

print("Distance Matrix:\n", dist_matrix)

# --- 5. Simple MDS ---
# B = -1/2 J D^2 J
D2 = dist_matrix ** 2
n = num_cells
J = torch.eye(n) - torch.ones((n, n))/n
B = -0.5 * J @ D2 @ J

# Diagonalize B
vals, vecs = torch.linalg.eigh(B)
# Sort descending
idx = vals.argsort(descending=True)
vals = vals[idx]
vecs = vecs[:, idx]

# Keep top 1 component for 1D
coord = vecs[:, 0] * torch.sqrt(torch.abs(vals[0]))
coord = coord.real

print("Recovered 1D Coords:", coord)
sorted_idx = torch.argsort(coord)
print("Sorted Indices:", sorted_idx)

# Save Plot
plt.figure()
plt.plot(coord.numpy(), np.zeros_like(coord), 'o-')
for i, txt in enumerate(range(num_cells)):
    plt.annotate(txt, (coord[i], 0.01))
plt.title("Recovered Geometry from Substrate Correlations")
plt.yticks([])
plt.xlabel("Emergent Dimension 1")
plt.savefig(_results / "embedding.png")
print(f"Plot saved to {_results / 'embedding.png'}")
