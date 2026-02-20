#include "types.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// --- Device Helpers ---

template <typename T>
struct ComplexType;

template <> struct ComplexType<float> { using type = cuFloatComplex; };
template <> struct ComplexType<double> { using type = cuDoubleComplex; };

__device__ __forceinline__ cuFloatComplex  cmul(cuFloatComplex a, cuFloatComplex b)   { return cuCmulf(a, b); }
__device__ __forceinline__ cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }

__device__ __forceinline__ cuFloatComplex  cadd(cuFloatComplex a, cuFloatComplex b)   { return make_cuFloatComplex(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }

__device__ __forceinline__ cuFloatComplex  cconj(cuFloatComplex a)  { return cuConjf(a); }
__device__ __forceinline__ cuDoubleComplex cconj(cuDoubleComplex a) { return cuConj(a); }

__device__ __forceinline__ float  cabs(cuFloatComplex a)  { return cuCabsf(a); }
__device__ __forceinline__ double cabs(cuDoubleComplex a) { return cuCabs(a); }

__device__ __forceinline__ cuFloatComplex  make_complex(float r, float i)   { return make_cuFloatComplex(r, i); }
__device__ __forceinline__ cuDoubleComplex make_complex(double r, double i) { return make_cuDoubleComplex(r, i); }

// ---------------------------------------------------------------------------
// Full Heisenberg mean-field interaction  (Sx⊗Sx + Sy⊗Sy + Sz⊗Sz)
//
// Rotates |ψ⟩ = (alpha, beta) of one cell around the Bloch vector
// h⃗ = (hx, hy, hz) of the other cell:
//
//   U = exp(-i J dt h⃗·σ⃗ )
//     = cos(θ)·I  -  i sin(θ) (n̂·σ⃗ )
//
// where θ = J·dt·|h⃗| and n̂ = h⃗/|h⃗|.
//
// The Bloch vector components are:
//   hx = 2 Re(α*β),  hy = 2 Im(α*β),  hz = |α|²-|β|²
//
// For normalised states |h⃗| = 1; the general formula handles
// unnormalised states from floating-point drift.
//
// The Ising-only kernel (previous version) applied only the hz (Sz⊗Sz)
// component, which preserves |α|² and |β|² individually and cannot
// generate the amplitude exchange needed for dynamic Sz correlations or
// entropy production via the stability selection principle (Section 4.4.2a
// of docs/framework.md).
// ---------------------------------------------------------------------------

template <typename Real>
__device__ void apply_heisenberg_step(
    typename ComplexType<Real>::type& alpha,
    typename ComplexType<Real>::type& beta,
    typename ComplexType<Real>::type  other_alpha,
    typename ComplexType<Real>::type  other_beta,
    Real J, Real dt
) {
    using Complex = typename ComplexType<Real>::type;

    // Bloch vector of the "other" cell:  p = conj(other_alpha) * other_beta
    Complex p = cmul(cconj(other_alpha), other_beta);

    Real mag_a = cabs(other_alpha);
    Real mag_b = cabs(other_beta);
    Real hz = mag_a * mag_a - mag_b * mag_b;

    // |h⃗|² = 4|p|² + hz²   (equals (|α|²+|β|²)² for normalised states)
    Real p_abs = cabs(p);
    Real h_sq  = Real(4) * p_abs * p_abs + hz * hz;
    Real h_mag = sqrt(h_sq);

    if (h_mag < Real(1e-12)) return;

    Real theta = J * dt * h_mag;
    Real c = cos(theta);
    Real s = sin(theta) / h_mag;

    // 2×2 unitary matrix in the {|0⟩,|1⟩} basis:
    //   U00 = c - i·s·hz
    //   U01 = -2i·s·conj(p)  =  (-2s·Im(p), -2s·Re(p))
    //   U10 = -2i·s·p         =  ( 2s·Im(p), -2s·Re(p))
    //   U11 = c + i·s·hz
    Complex U00 = make_complex(c, -s * hz);
    Complex U01 = make_complex(Real(-2) * s * p.y, Real(-2) * s * p.x);
    Complex U10 = make_complex(Real( 2) * s * p.y, Real(-2) * s * p.x);
    Complex U11 = make_complex(c, s * hz);

    Complex new_alpha = cadd(cmul(U00, alpha), cmul(U01, beta));
    Complex new_beta  = cadd(cmul(U10, alpha), cmul(U11, beta));

    alpha = new_alpha;
    beta  = new_beta;
}

template <typename Real>
__device__ void apply_interaction_T(
    typename ComplexType<Real>::type& c1_alpha,
    typename ComplexType<Real>::type& c1_beta,
    typename ComplexType<Real>::type& c2_alpha,
    typename ComplexType<Real>::type& c2_beta,
    Real J,
    Real dt
) {
    // Rotate cell 1 around cell 2's Bloch vector
    apply_heisenberg_step<Real>(c1_alpha, c1_beta, c2_alpha, c2_beta, J, dt);

    // Rotate cell 2 around cell 1's (updated) Bloch vector
    apply_heisenberg_step<Real>(c2_alpha, c2_beta, c1_alpha, c1_beta, J, dt);
}

template <typename Real>
__global__ void __launch_bounds__(256) phase_flow_kernel_soa(
    typename ComplexType<Real>::type* __restrict__ cell_alphas,
    typename ComplexType<Real>::type* __restrict__ cell_betas,
    const int* __restrict__ edge_src,
    const int* __restrict__ edge_dst,
    const Real* __restrict__ edge_weights,
    int num_active_edges,
    Real dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_active_edges) return;

    int s = edge_src[idx];
    int d = edge_dst[idx];
    Real w = edge_weights[idx];

    #if __CUDA_ARCH__ >= 350
    auto a1 = __ldg(&cell_alphas[s]);
    auto b1 = __ldg(&cell_betas[s]);
    auto a2 = __ldg(&cell_alphas[d]);
    auto b2 = __ldg(&cell_betas[d]);
    #else
    auto a1 = cell_alphas[s];
    auto b1 = cell_betas[s];
    auto a2 = cell_alphas[d];
    auto b2 = cell_betas[d];
    #endif

    apply_interaction_T<Real>(a1, b1, a2, b2, w, dt);

    cell_alphas[s] = a1;
    cell_betas[s] = b1;
    cell_alphas[d] = a2;
    cell_betas[d] = b2;
}

extern "C" void launch_phase_flow_float(
    cuFloatComplex* alphas, cuFloatComplex* betas,
    const int* src, const int* dst, const float* w,
    int n, float dt
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    phase_flow_kernel_soa<float><<<grid, block>>>(alphas, betas, src, dst, w, n, dt);
}

extern "C" void launch_phase_flow_double(
    cuDoubleComplex* alphas, cuDoubleComplex* betas,
    const int* src, const int* dst, const double* w,
    int n, double dt
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    phase_flow_kernel_soa<double><<<grid, block>>>(alphas, betas, src, dst, w, n, dt);
}
