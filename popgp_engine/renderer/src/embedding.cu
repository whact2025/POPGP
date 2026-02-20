#include "embedding.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>

// --- Physics Constants ---
#define EPSILON 1e-6

// --- Kernel 1: Calculate Forces ---
__global__ void compute_forces_kernel(
    const double* px, const double* py, const double* pz,
    const int* src, const int* dst, const double* target_dist,
    double* fx, double* fy, double* fz,
    int num_nodes, int num_edges,
    double k_attract, double k_repel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. Repulsion (N^2 Body Problem - Simplified)
    // Each thread handles one node.
    if (idx < num_nodes) {
        double my_x = px[idx];
        double my_y = py[idx];
        double my_z = pz[idx];
        
        double force_x = 0;
        double force_y = 0;
        double force_z = 0;

        // Naive All-to-All Repulsion (Optimize later with Barnes-Hut)
        // Only run for small N, or sample random subset for large N
        int limit = (num_nodes > 1024) ? 1024 : num_nodes; // Performance clamp
        
        for (int j = 0; j < limit; j++) {
            if (idx == j) continue;
            
            double dx = my_x - px[j];
            double dy = my_y - py[j];
            double dz = my_z - pz[j];
            double dist_sq = dx*dx + dy*dy + dz*dz + EPSILON;
            double dist = sqrt(dist_sq);
            
            // Coulomb Repulsion: F = k / r^2
            double f = k_repel / dist_sq;
            
            force_x += f * (dx / dist);
            force_y += f * (dy / dist);
            force_z += f * (dz / dist);
        }
        
        fx[idx] = force_x;
        fy[idx] = force_y;
        fz[idx] = force_z;
    }
    
    // 2. Attraction (Springs) - Run on Edges
    // Need separate kernel or atomicAdd.
    // Let use separate kernel for edges to avoid sync issues.
}

__global__ void compute_spring_forces_kernel(
    const double* px, const double* py, const double* pz,
    const int* src, const int* dst, const double* target_dist,
    double* fx, double* fy, double* fz,
    int num_edges,
    double k_attract
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    
    int s = src[idx];
    int d = dst[idx];
    double rest_len = target_dist[idx];
    
    double dx = px[d] - px[s];
    double dy = py[d] - py[s];
    double dz = pz[d] - pz[s];
    
    double dist = sqrt(dx*dx + dy*dy + dz*dz) + EPSILON;
    
    // Hooke Law: F = k * (r - r0)
    double f = k_attract * (dist - rest_len);
    
    double f_vec_x = f * (dx / dist);
    double f_vec_y = f * (dy / dist);
    double f_vec_z = f * (dz / dist);
    
    // Apply to both (Action-Reaction)
    atomicAdd(&fx[s], f_vec_x);
    atomicAdd(&fy[s], f_vec_y);
    atomicAdd(&fz[s], f_vec_z);
    
    atomicAdd(&fx[d], -f_vec_x);
    atomicAdd(&fy[d], -f_vec_y);
    atomicAdd(&fz[d], -f_vec_z);
}

__global__ void apply_integration_kernel(
    double* px, double* py, double* pz,
    const double* fx, const double* fy, const double* fz,
    int num_nodes, double dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    // Overdamped dynamics (dx = F * dt)
    px[idx] += fx[idx] * dt;
    py[idx] += fy[idx] * dt;
    pz[idx] += fz[idx] * dt;
}

extern "C" void step_force_layout(
    double* h_px, double* h_py, double* h_pz,
    const int* h_src, const int* h_dst, const double* h_target,
    int num_nodes, int num_edges,
    double k_attract, double k_repel,
    double dt, int iterations
) {
    // 1. Allocate
    double *d_px, *d_py, *d_pz;
    double *d_fx, *d_fy, *d_fz;
    double *d_target;
    int *d_src, *d_dst;
    
    cudaMalloc(&d_px, num_nodes * sizeof(double));
    cudaMalloc(&d_py, num_nodes * sizeof(double));
    cudaMalloc(&d_pz, num_nodes * sizeof(double));
    cudaMalloc(&d_fx, num_nodes * sizeof(double));
    cudaMalloc(&d_fy, num_nodes * sizeof(double));
    cudaMalloc(&d_fz, num_nodes * sizeof(double));
    cudaMalloc(&d_src, num_edges * sizeof(int));
    cudaMalloc(&d_dst, num_edges * sizeof(int));
    cudaMalloc(&d_target, num_edges * sizeof(double));
    
    // 2. Upload
    cudaMemcpy(d_px, h_px, num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pz, h_pz, num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src, h_src, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, num_edges * sizeof(double), cudaMemcpyHostToDevice);
    
    int block = 256;
    int grid_nodes = (num_nodes + block - 1) / block;
    int grid_edges = (num_edges + block - 1) / block;
    
    // 3. Loop
    for (int i=0; i<iterations; i++) {
        // Compute Repulsion
        compute_forces_kernel<<<grid_nodes, block>>>(
            d_px, d_py, d_pz, d_src, d_dst, d_target,
            d_fx, d_fy, d_fz, num_nodes, num_edges, k_attract, k_repel
        );
        
        // Compute Attraction (accumulate)
        compute_spring_forces_kernel<<<grid_edges, block>>>(
            d_px, d_py, d_pz, d_src, d_dst, d_target,
            d_fx, d_fy, d_fz, num_edges, k_attract
        );
        
        // Integrate
        apply_integration_kernel<<<grid_nodes, block>>>(
            d_px, d_py, d_pz, d_fx, d_fy, d_fz, num_nodes, dt
        );
    }
    
    // 4. Download
    cudaMemcpy(h_px, d_px, num_nodes * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_py, d_py, num_nodes * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pz, d_pz, num_nodes * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_target);
}

