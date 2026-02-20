# POPGP Engine Development Tasks

This document outlines the tasks required to build the high-performance CUDA engine (`popgp_engine`) for the POPGP framework. The engine must rigorously adhere to the theoretical definitions in `docs/framework.md`, specifically the "Discrete Route" for geometry and "Intrinsic" definitions for time.

## 1. Substrate & State Management (Kernel)
*   **Data Structures**: Define dense/sparse matrix classes for the Relational Algebra `A`.
    *   *Requirement*: Must support `complex<double>` (or `cuDoubleComplex`) to satisfy precision bounds.
*   **Phase Flow (`\sigma_s`)**: Implement unitary evolution kernel.
    *   *Action*: `U(s) = exp(-i H s)`. Use `cuBLAS` or custom kernel for sparse matrix exponentiation (Taylor series or Padé approximant).
*   **Custodial Symmetry Enforcement**: Implement the projection operator $P_{singlet}$ to enforce the $E_i \circ \alpha_g = \alpha_g \circ E_i$ constraint.
    *   *Action*: Project local operator blocks to $SU(2)$ singlets.

## 2. Information Geometry (`\Pi_{res}`, `\Pi_{loc}`)
*   **Funnel Coarse-Graining**: Implement `E_i` maps to Type I factors.
    *   *Action*: Spectral decomposition of local density matrices using `cuSOLVER` (`cusolverDnZheevd`) to compute reduced states.
*   **Araki Relative Entropy**: Implement the rigorous entropy contrast.
    *   *Formula*: $S(\omega_i || \omega_{vac}) = \text{Tr}[\rho_i (\log \rho_i - \log \rho_{vac})]$.
    *   *Implementation*: Requires matrix logarithm. Use `cuSOLVER` eigenvalues: $U \log(\Lambda) U^\dagger$.
*   **Mutual Information Graph**: Compute $I_{ij}$ for all/neighboring pairs.
    *   *Action*: Block-sparse kernel to compute $S(\rho_i) + S(\rho_j) - S(\rho_{ij})$.
    *   *Optimization*: Use `faiss` (if allowed) or a spatial hash to limit pair computation to candidate neighbors, but the theory is non-local, so initially $O(N^2)$ or random sampling might be needed.

## 3. Geometric Projection (`\Pi_{geom}`)
*   **Complexity-Stress Minimization**: Implement the embedding optimizer.
    *   *Functional*: $F(X) = \sum w_{ij} (|x_i - x_j| - d_{G,ij})^2 + \lambda |D - D_S|^2$.
    *   *Implementation*: Gradient descent (SGD/Adam) on the GPU coordinates $X$.
    *   *Spectral Dimension $D_S$*: Compute via Heat Kernel Trace on the graph Laplacian ($Tr(e^{-t \Delta})$). Requires Graph Laplacian eigen-decomposition or Chebyshev approximation.
*   **Delaunay Triangulation**: Construct the simplicial complex from $X$.
    *   *Constraint*: GPU Delaunay is complex.
    *   *Approach*: Copy $X$ to Host $\to$ Use `CGAL` or `Geogram` (via vcpkg) $\to$ Copy Connectivity to Device.
*   **Discrete Regge Calculus**: Compute curvature on the GPU mesh.
    *   *Action*: Kernel to compute edge lengths $l_{ij}^2$ and deficit angles $\epsilon_h$.

## 4. Unified Time & Gravity (`\Pi_{time}`)
*   **Graph Laplacian Solver**: Solve $(\Delta_w + \mu^2 I)\Phi = \delta\rho$.
    *   *Implementation*: Use `cuSOLVER` (QR/Cholesky) or `AmgX` (Multigrid) for the sparse linear system.
    *   *Zero Mode*: Handle the $\mu=0$ rank-deficiency (fix one node or project out the null space).
*   **Optimal Transport (Shift Vector)**: Compute Gromov-Wasserstein plan $\gamma^*$.
    *   *Implementation*: Sinkhorn-Knopp algorithm kernel.
    *   *Intrinsic Shift*: Map target distribution to tangent space via Discrete Log map.

## 5. Visualization (Renderer)
*   **CUDA-OpenGL Interop**: Map GPU position/color buffers to OpenGL VBOs.
    *   *Action*: Use `cudaGraphicsGLRegisterBuffer`.
*   **Field Rendering**: Visualize $\Phi$ (color) and $N^a$ (vector field lines).

## 6. Testing & Validation
*   **Regge Closure Test**: Compute the closure mismatch $M(L)$.
    *   *Action*: Compare $G_{Regge}$ to projected $T_{\mu\nu}$.
*   **Lorentz Violation Check**: Measure photon dispersion relation on the graph.
