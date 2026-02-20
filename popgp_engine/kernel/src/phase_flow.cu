#include "types.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// --- Device Helpers ---

template <typename T>
struct ComplexType;

template <> struct ComplexType<float> { using type = cuFloatComplex; };
template <> struct ComplexType<double> { using type = cuDoubleComplex; };

__device__ __forceinline__ cuFloatComplex cmul(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }
__device__ __forceinline__ cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
__device__ __forceinline__ float cabs(cuFloatComplex a) { return cuCabsf(a); }
__device__ __forceinline__ double cabs(cuDoubleComplex a) { return cuCabs(a); }
__device__ __forceinline__ cuFloatComplex make_complex(float r, float i) { return make_cuFloatComplex(r, i); }
__device__ __forceinline__ cuDoubleComplex make_complex(double r, double i) { return make_cuDoubleComplex(r, i); }

template <typename Real>
__device__ void apply_interaction_T(
    typename ComplexType<Real>::type& c1_alpha,
    typename ComplexType<Real>::type& c1_beta,
    typename ComplexType<Real>::type& c2_alpha,
    typename ComplexType<Real>::type& c2_beta,
    Real J, 
    Real dt
) {
    using Complex = typename ComplexType<Real>::type;
    Real mag_b2 = cabs(c2_beta);
    Real mag_a2 = cabs(c2_alpha);
    Real z2 = mag_a2*mag_a2 - mag_b2*mag_b2;
    Real angle1 = J * z2 * dt;
    Complex rot1 = make_complex(cos(angle1), -sin(angle1));
    Complex rot1_c = make_complex(cos(angle1), sin(angle1));
    c1_alpha = cmul(c1_alpha, rot1);
    c1_beta  = cmul(c1_beta, rot1_c);
    Real mag_b1 = cabs(c1_beta);
    Real mag_a1 = cabs(c1_alpha);
    Real z1 = mag_a1*mag_a1 - mag_b1*mag_b1;
    Real angle2 = J * z1 * dt;
    Complex rot2 = make_complex(cos(angle2), -sin(angle2));
    Complex rot2_c = make_complex(cos(angle2), sin(angle2));
    c2_alpha = cmul(c2_alpha, rot2);
    c2_beta  = cmul(c2_beta, rot2_c);
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

