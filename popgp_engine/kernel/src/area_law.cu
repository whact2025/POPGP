#include "types.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// --- Area Law Logic ---

__global__ void calculate_cut_kernel(
    const int* edge_src,
    const int* edge_dst,
    const double* edge_weights,
    const int* node_active_mask, // 1=Active, 0=Frozen
    double* node_cut_sizes,      // Output: Total weight crossing boundary for this node
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    int s = edge_src[idx];
    int d = edge_dst[idx];
    double w = edge_weights[idx];

    int active_s = node_active_mask[s];
    int active_d = node_active_mask[d];

    // Debug Print
    if (idx < 5) {
        printf("Edge %d: %d->%d (w=%.1f) Active: %d->%d\n", 
               idx, s, d, w, active_s, active_d);
    }

    // Simple implementation: Atomic add weight to both nodes if they differ in state
    if (active_s != active_d) {
        atomicAdd(&node_cut_sizes[s], w);
        atomicAdd(&node_cut_sizes[d], w);
        if (idx < 5) printf("  -> CUT DETECTED! Adding %.1f\n", w);
    }
}

__global__ void prune_bulk_kernel(
    int* node_active_mask,
    const double* node_cut_sizes,
    const double* node_entropies,
    int num_nodes,
    double threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    double area = node_cut_sizes[idx];
    double entropy = node_entropies[idx];
    
    if (node_active_mask[idx] == 1) {
        if (entropy < 0.1 * area) {
            // node_active_mask[idx] = 0; 
        }
    } else {
        if (area > threshold) {
            node_active_mask[idx] = 1;
        }
    }
}

extern "C" void launch_area_law_pruning(
    const int* src, const int* dst, const double* w,
    int* active_mask, double* cut_sizes, double* entropies,
    int num_edges, int num_nodes
) {
    cudaMemset(cut_sizes, 0, num_nodes * sizeof(double));

    int block = 256;
    int grid_edges = (num_edges + block - 1) / block;
    calculate_cut_kernel<<<grid_edges, block>>>(src, dst, w, active_mask, cut_sizes, num_edges);
    
    // Wait for prints
    cudaDeviceSynchronize();

    int grid_nodes = (num_nodes + block - 1) / block;
    prune_bulk_kernel<<<grid_nodes, block>>>(active_mask, cut_sizes, entropies, num_nodes, 0.5);
}

