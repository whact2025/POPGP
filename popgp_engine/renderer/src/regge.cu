#include "regge.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>

// --- Helper: Vector Math ---
__device__ double3 sub(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ double dot(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ double length(double3 a) {
    return sqrt(dot(a, a));
}

__device__ double3 normalize(double3 a) {
    double l = length(a);
    if (l < 1e-9) return make_double3(0,0,0);
    return make_double3(a.x/l, a.y/l, a.z/l);
}

// --- Kernel 1: Compute Angles per Triangle ---
__global__ void compute_triangle_angles_kernel(
    const double* vertices, // [N*3]
    const int* triangles,   // [M*3]
    double* angle_sums,     // [N] Accumulator
    int num_triangles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_triangles) return;

    // Load Indices
    int i0 = triangles[idx * 3 + 0];
    int i1 = triangles[idx * 3 + 1];
    int i2 = triangles[idx * 3 + 2];

    // Load Vertices
    double3 v0 = make_double3(vertices[i0*3], vertices[i0*3+1], vertices[i0*3+2]);
    double3 v1 = make_double3(vertices[i1*3], vertices[i1*3+1], vertices[i1*3+2]);
    double3 v2 = make_double3(vertices[i2*3], vertices[i2*3+1], vertices[i2*3+2]);

    // Edges
    double3 e01 = sub(v1, v0);
    double3 e12 = sub(v2, v1);
    double3 e20 = sub(v0, v2);

    // Normalize
    double3 u01 = normalize(e01);
    double3 u12 = normalize(e12);
    double3 u20 = normalize(e20);

    // Angles using Dot Product: cos(theta)
    // Angle at v0 is between e01 and -e20 (e02)
    // Note: u20 points V2->V0. -u20 points V0->V2.
    
    double a0 = acos(dot(u01, make_double3(-u20.x, -u20.y, -u20.z)));
    double a1 = acos(dot(make_double3(-u01.x, -u01.y, -u01.z), u12));
    double a2 = acos(dot(make_double3(-u12.x, -u12.y, -u12.z), u20));

    // Atomic Accumulation
    atomicAdd(&angle_sums[i0], a0);
    atomicAdd(&angle_sums[i1], a1);
    atomicAdd(&angle_sums[i2], a2);
}

// --- Kernel 2: Compute Deficit ---
__global__ void finalize_curvature_kernel(
    const double* angle_sums,
    double* curvature,
    int num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;

    // Deficit = 2*PI - Sum(angles)
    curvature[idx] = 2.0 * CUDART_PI - angle_sums[idx];
}

extern "C" void compute_regge_curvature(
    const double* h_vertices, 
    const int* h_triangles, 
    double* h_curvature,
    int num_vertices, 
    int num_triangles
) {
    // 1. Allocate Device Memory
    double *d_v, *d_sums, *d_curv;
    int *d_t;
    
    cudaMalloc(&d_v, num_vertices * 3 * sizeof(double));
    cudaMalloc(&d_t, num_triangles * 3 * sizeof(int));
    cudaMalloc(&d_sums, num_vertices * sizeof(double));
    cudaMalloc(&d_curv, num_vertices * sizeof(double));
    
    // 2. Upload
    cudaMemcpy(d_v, h_vertices, num_vertices * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, h_triangles, num_triangles * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_sums, 0, num_vertices * sizeof(double)); // Zero accumulator

    // 3. Launch Triangle Kernel
    int block = 256;
    int grid_tri = (num_triangles + block - 1) / block;
    compute_triangle_angles_kernel<<<grid_tri, block>>>(d_v, d_t, d_sums, num_triangles);
    cudaDeviceSynchronize();

    // 4. Launch Vertex Kernel
    int grid_v = (num_vertices + block - 1) / block;
    finalize_curvature_kernel<<<grid_v, block>>>(d_sums, d_curv, num_vertices);
    cudaDeviceSynchronize();

    // 5. Download
    cudaMemcpy(h_curvature, d_curv, num_vertices * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_v); cudaFree(d_t); cudaFree(d_sums); cudaFree(d_curv);
}

