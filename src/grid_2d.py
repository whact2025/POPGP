import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
WIDTH = 3
HEIGHT = 3
N = WIDTH * HEIGHT
dim = 2**N
beta = 2.0  # Lower temp to enhance correlations

device = torch.device('cpu')
print(f"Simulating {WIDTH}x{HEIGHT} = {N} qubit grid on {device}")

# --- 1. Define 2D Heisenberg Hamiltonian ---
# Flat index k = y * WIDTH + x
# Neighbors: (x, y) <-> (x+1, y) and (x, y) <-> (x, y+1)

Sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device) / 2
Sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device) / 2
Sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device) / 2
I = torch.eye(2, dtype=torch.complex128, device=device)

def tensor_op(op, site, N):
    # Construct Operator acting on specific site in N-qubit chain
    # This naive Kronecker product is O(2^N), fine for N=9 (512x512 matrix)
    ops = [I] * N
    ops[site] = op
    res = ops[0]
    for i in range(1, N):
        res = torch.kron(res, ops[i])
    return res

# Precompute single-site operators to save time
ops_Sx = [tensor_op(Sx, i, N) for i in range(N)]
ops_Sy = [tensor_op(Sy, i, N) for i in range(N)]
ops_Sz = [tensor_op(Sz, i, N) for i in range(N)]

H = torch.zeros((dim, dim), dtype=torch.complex128, device=device)

# Add interactions
edges = []
for y in range(HEIGHT):
    for x in range(WIDTH):
        k = y * WIDTH + x
        
        # Right neighbor
        if x + 1 < WIDTH:
            k_right = y * WIDTH + (x + 1)
            edges.append((k, k_right))
        
        # Bottom neighbor
        if y + 1 < HEIGHT:
            k_down = (y + 1) * WIDTH + x
            edges.append((k, k_down))

print(f"Topology: {len(edges)} interactions defined.")

for (i, j) in edges:
    # Heisenberg interaction S_i . S_j
    term = (ops_Sx[i] @ ops_Sx[j] + 
            ops_Sy[i] @ ops_Sy[j] + 
            ops_Sz[i] @ ops_Sz[j])
    H += term

# --- 2. Compute State (Thermal) ---
print("Diagonalizing Hamiltonian...")
evals, evecs = torch.linalg.eigh(H)
# Boltzmann weights
probs = torch.exp(-beta * (evals - evals[0])) # Shift for stability
probs /= torch.sum(probs)

# Construct Thermal Density Matrix
# rho = sum p_k |k><k|
exp_H = torch.diag(probs).to(dtype=torch.complex128)
rho = evecs @ exp_H @ evecs.conj().T

print("State prepared.")

# --- 3. Compute Mutual Information ---
def get_reduced_rho_1site(rho, site, N):
    # Reshape to (2, 2, ..., 2) [2N times]
    shape = [2] * (2 * N)
    rho_tensor = rho.view(shape)
    
    trace_indices = [k for k in range(N) if k != site]
    
    # 1. Permute to bring 'site' to front
    # [site, site + N] + [k, k+N for k in trace_indices]
    perm = [site, site + N]
    for k in trace_indices:
        perm.append(k)
        perm.append(k + N)
    
    rho_perm = rho_tensor.permute(perm)
    
    # 2. Reshape to (2, 2, dim_trace, dim_trace) ? No.
    # The remaining dims are (2, 2, 2, 2...)
    # We want to trace pairs (k, k+N)
    
    # Actually, simpler einsum approach is robust if we group correctly.
    # [site, site+N, rest_in, rest_out] -> then einsum ii
    
    # Let's use the explicit loop over trace indices
    # It's slow but correct. Or recursive trace.
    
    curr = rho_tensor
    # Trace out indices from N-1 down to 0
    for k in reversed(range(N)):
        if k == site:
            continue
        # k is the index in the original list 0..N-1
        # But after tracing some, indices shift.
        # This is messy.
        
        # Better: Permute once.
        pass

    # Use the Einsum method that worked in main.py?
    # No, that was for contiguous blocks.
    
    # Let's retry the permute-and-einsum carefully.
    # We want rho[s, s', k1, k1', k2, k2' ...] -> sum over k1=k1', k2=k2'...
    
    # Permute: [site, site+N, rest_0, rest_0+N, rest_1, rest_1+N ...]
    perm = [site, site + N]
    for k in trace_indices:
        perm.append(k)
        perm.append(k + N)
    rho_perm = rho_tensor.permute(perm)
    
    # Flatten all trace pairs into one dimension? No.
    # Reshape to (2, 2, 4, 4, 4...)
    new_shape = [2, 2] + [4] * (N - 1)
    rho_reshaped = rho_perm.reshape(new_shape)
    
    # Now we want to sum the diagonal elements of the 4x4 matrices?
    # No. The indices (k, k+N) are (0,0), (0,1), (1,0), (1,1).
    # We want trace: sum (0,0) and (1,1). i.e. indices 0 and 3 in the flattened 4-dim.
    
    # This corresponds to a specific contraction vector.
    # Let's perform the trace sequentially on the last dimension.
    
    temp = rho_reshaped
    for _ in range(N - 1):
        # Last dim is 4. Sum index 0 and 3.
        # temp shape: (..., 4)
        t0 = temp[..., 0]
        t3 = temp[..., 3]
        temp = t0 + t3
        
    return temp

def get_reduced_rho_2sites(rho, i, j, N):
    if i > j: i, j = j, i
    
    shape = [2] * (2 * N)
    rho_tensor = rho.view(shape)
    
    trace_indices = [k for k in range(N) if k != i and k != j]
    
    # Permute: [i, i+N, j, j+N, rest...]
    perm = [i, i+N, j, j+N]
    for k in trace_indices:
        perm.append(k)
        perm.append(k + N)
        
    rho_perm = rho_tensor.permute(perm)
    
    # Reshape: (2, 2, 2, 2, 4, 4...)
    new_shape = [2, 2, 2, 2] + [4] * (N - 2)
    rho_reshaped = rho_perm.reshape(new_shape)
    
    temp = rho_reshaped
    for _ in range(N - 2):
        t0 = temp[..., 0]
        t3 = temp[..., 3]
        temp = t0 + t3
        
    # Result is (2, 2, 2, 2) -> (row_i, col_i, row_j, col_j)
    # We want (row_i, row_j; col_i, col_j)
    # Permute to (row_i, row_j, col_i, col_j)
    temp = temp.permute(0, 2, 1, 3)
    return temp.reshape(4, 4)

def entropy(rho):
    vals = torch.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-12]
    return -torch.sum(vals * torch.log(vals))

print("Computing Mutual Information Matrix...")
mi_matrix = torch.zeros((N, N))
entropies = []

# Cache single site entropies
for i in range(N):
    rho_i = get_reduced_rho_1site(rho, i, N)
    entropies.append(entropy(rho_i))

for i in range(N):
    for j in range(i + 1, N):
        rho_ij = get_reduced_rho_2sites(rho, i, j, N)
        S_ij = entropy(rho_ij)
        I_ij = entropies[i] + entropies[j] - S_ij
        mi_matrix[i, j] = I_ij
        mi_matrix[j, i] = I_ij

# --- 4. Geometry Recovery (MDS) ---
# Distance metric
# d_ij = -log(I_ij / I_0)
# Use max(I) as I_0
I_max = torch.max(mi_matrix)
dist_matrix = -torch.log(mi_matrix / I_max + 1e-9)
# Zero out diagonal
dist_matrix.fill_diagonal_(0)

print("Running MDS...")
# Classical MDS
D2 = dist_matrix ** 2
n = N
J_mat = torch.eye(n) - torch.ones((n, n))/n
B = -0.5 * J_mat @ D2 @ J_mat

vals, vecs = torch.linalg.eigh(B)
idx = vals.argsort(descending=True)
vals = vals[idx]
vecs = vecs[:, idx]

# Keep top 2 components for 2D
coords = vecs[:, :2] @ torch.diag(torch.sqrt(torch.abs(vals[:2])))
coords = coords.real

# --- 5. Visualization ---
coords_np = coords.numpy()
plt.figure(figsize=(6, 6))
plt.scatter(coords_np[:, 0], coords_np[:, 1], c='blue', s=100)

# Label points
for i in range(N):
    plt.annotate(str(i), (coords_np[i, 0], coords_np[i, 1]), xytext=(5, 5), textcoords='offset points')

# Draw edges from Hamiltonian to check topology
for (i, j) in edges:
    p1 = coords_np[i]
    p2 = coords_np[j]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.3)

plt.title(f"Emergent 2D Geometry ({WIDTH}x{HEIGHT} Heisenberg)")
plt.axis('equal')
plt.grid(True, linestyle=':', alpha=0.6)

# Highlight edges to verify grid structure
for (i, j) in edges:
    p1 = coords_np[i]
    p2 = coords_np[j]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.3)

plt.savefig("toy_model/2d_grid_embedding.png")
print("Plot saved to toy_model/2d_grid_embedding.png")

print("\nCoordinates:")
print(coords_np)

# Check if grid topology is preserved
# Center point should be index 4 (1,1).
# Neighbors of 4: 1, 3, 5, 7.
# Let's calculate average distance to neighbors vs non-neighbors.
center = 4
neighbors = [1, 3, 5, 7]
others = [0, 2, 6, 8]

d_neighbors = []
for n in neighbors:
    d = np.linalg.norm(coords_np[center] - coords_np[n])
    d_neighbors.append(d)

d_others = []
for o in others:
    d = np.linalg.norm(coords_np[center] - coords_np[o])
    d_others.append(d)
    
avg_neigh = np.mean(d_neighbors)
avg_other = np.mean(d_others)

print(f"\nAvg Dist to Neighbors: {avg_neigh:.4f}")
print(f"Avg Dist to Non-Neighbors: {avg_other:.4f}")
if avg_neigh < avg_other:
    print("SUCCESS: Local structure preserved.")
else:
    print("FAILURE: Geometry distorted.")
