import torch
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add Engine Binding to Path
sys.path.append(os.path.join(os.path.dirname(__file__), "../popgp_engine/kernel/python"))
try:
    from popgp import Engine
    print("SUCCESS: Loaded POPGP C++ Engine")
except ImportError as e:
    print(f"ERROR: Could not load engine: {e}")
    sys.exit(1)

def run_simulation():
    # Configuration
    N = 1000000  # 1 Million Cells! (Demonstrates scaling)
    dt = 0.1
    steps = 50
    device = torch.device("cuda")

    print(f"Initializing {N} cells on {device}...")

    # 1. Initialize State (SoA Layout)
    # Complex numbers in PyTorch are usually c64/c128 (struct of 2 floats).
    # Our kernel expects split arrays (alpha_real, alpha_imag is inside struct, but SoA expects alpha[], beta[])
    # Wait, types.cuh defined `cuDoubleComplex` which is `{x, y}`.
    # PyTorch `complex128` is `{real, imag}`.
    # So `d_alphas` as a `complex128` tensor matches `cuDoubleComplex*`.
    
    # Initial State: |+> (Superposition)
    # alpha = 1/sqrt(2), beta = 1/sqrt(2)
    inv_sq2 = 1.0 / np.sqrt(2)
    
    # Use float64 (double) for precision matches C++ kernel
    alphas = torch.full((N,), inv_sq2, dtype=torch.complex128, device=device)
    betas  = torch.full((N,), inv_sq2, dtype=torch.complex128, device=device)
    
    # 2. Initialize Graph (1D Chain)
    # Edges: 0-1, 1-2, ...
    # We need Red/Black coloring for parallel safety, but the kernel wrapper handles raw edges.
    # The kernel implementation I wrote assumes the USER provides non-overlapping edges per batch if using coloring.
    # OR if we pass ALL edges, we risk race conditions unless the kernel uses atomics or coloring.
    
    # My kernel implementation `phase_flow.cu` does NOT implement internal coloring.
    # It assumes the `edges` array passed is safe to execute in parallel.
    # So we must pass Red edges, then Black edges.
    
    print("Constructing 1D Graph...")
    indices = torch.arange(N-1, device=device)
    
    # Red Edges (0-1, 2-3, ...) -> Even indices
    red_idx = indices[indices % 2 == 0]
    red_src = red_idx
    red_dst = red_idx + 1
    red_w   = torch.ones_like(red_src, dtype=torch.float64)
    
    # Black Edges (1-2, 3-4, ...) -> Odd indices
    black_idx = indices[indices % 2 == 1]
    black_src = black_idx
    black_dst = black_idx + 1
    black_w   = torch.ones_like(black_src, dtype=torch.float64)
    
    # 3. Initialize Engine
    engine = Engine(precision="double")
    
    # 4. Run Loop
    print("Starting Phase Flow...")
    torch.cuda.synchronize()
    t0 = time.time()
    
    for s in range(steps):
        # Step A: Red Layer
        engine.step(alphas, betas, red_src.int(), red_dst.int(), red_w, dt)
        
        # Step B: Black Layer
        engine.step(alphas, betas, black_src.int(), black_dst.int(), black_w, dt)
        
        if s % 10 == 0:
            print(f"Step {s}/{steps} completed.")

    torch.cuda.synchronize()
    t1 = time.time()
    
    print(f"Done! {steps} steps for {N} cells in {t1-t0:.4f}s")
    print(f"Throughput: {steps * N / (t1-t0):.2e} cell-updates/sec")

    # 5. Validation
    # Check if state evolved (Imaginary part should be non-zero due to Z-rotation)
    # Initial state |+> is Real. Evolution e^{-iH} creates Imaginary components.
    sample = betas[0:5].cpu().numpy()
    print("Sample Betas:", sample)
    if np.any(np.imag(sample) != 0):
        print("SUCCESS: Quantum State Evolved.")
    else:
        print("FAILURE: State remained static.")

if __name__ == "__main__":
    try:
        run_simulation()
    except ImportError:
        print("PyTorch with CUDA is required for this demo.")

