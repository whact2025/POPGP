#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "../include/types.cuh"

// Area Law Unit Test

TEST(AreaLawTest, BoundaryCutCalculation) {
    int num_nodes = 3;
    int num_edges = 2;
    
    int h_src[] = {0, 1};
    int h_dst[] = {1, 2};
    double h_w[] = {1.0, 5.0}; 
    int h_mask[] = {1, 1, 0};
    
    int *d_src, *d_dst, *d_mask;
    double *d_w, *d_cut, *d_ent;
    
    cudaMalloc(&d_src, num_edges * sizeof(int));
    cudaMalloc(&d_dst, num_edges * sizeof(int));
    cudaMalloc(&d_w, num_edges * sizeof(double));
    cudaMalloc(&d_mask, num_nodes * sizeof(int));
    cudaMalloc(&d_cut, num_nodes * sizeof(double));
    cudaMalloc(&d_ent, num_nodes * sizeof(double));
    
    cudaMemcpy(d_src, h_src, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, num_edges * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    
    // Run Cut Calculation
    launch_area_law_pruning(d_src, d_dst, d_w, d_mask, d_cut, d_ent, num_edges, num_nodes);
    
    // Check Errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Sync Error: %s\n", cudaGetErrorString(err));
    
    // Check Results
    double h_cut[3];
    cudaMemcpy(h_cut, d_cut, num_nodes * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Debug Output
    printf("Cut Sizes: [%.1f, %.1f, %.1f]\n", h_cut[0], h_cut[1], h_cut[2]);

    EXPECT_EQ(h_cut[0], 0.0);
    EXPECT_EQ(h_cut[1], 5.0);
    EXPECT_EQ(h_cut[2], 5.0);
    
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_w);
    cudaFree(d_mask); cudaFree(d_cut); cudaFree(d_ent);
}

