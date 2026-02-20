#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/types.cuh"
#include <math.h>

TEST(PhaseFlowTest, TwoQubitEntanglement) {
    int n = 2;
    float dt = 0.5f; // Larger step to see phase
    float J = 1.0f;
    
    // Initial State: |0>|+>
    // Qubit 0: |0> -> Creates Z-field (Bz = 1)
    // Qubit 1: |+> -> Should rotate around Z-axis (Phase change)
    
    cuFloatComplex* d_alphas;
    cuFloatComplex* d_betas;
    cudaMalloc(&d_alphas, n * sizeof(cuFloatComplex));
    cudaMalloc(&d_betas, n * sizeof(cuFloatComplex));
    
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    cuFloatComplex plus_state = make_cuFloatComplex(inv_sqrt2, 0.0f);
    cuFloatComplex zero_state = make_cuFloatComplex(1.0f, 0.0f);
    
    // Qubit 0 is |0>, Qubit 1 is |+>
    cuFloatComplex h_alphas[] = {zero_state, plus_state};
    cuFloatComplex h_betas[]  = {make_cuFloatComplex(0,0), plus_state};
    
    cudaMemcpy(d_alphas, h_alphas, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_betas, h_betas, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    
    // Graph: 0-1 (Weight J)
    int* d_src; int* d_dst; float* d_w;
    cudaMalloc(&d_src, sizeof(int));
    cudaMalloc(&d_dst, sizeof(int));
    cudaMalloc(&d_w, sizeof(float));
    
    int h_src = 0; int h_dst = 1; float h_w = J;
    cudaMemcpy(d_src, &h_src, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, &h_dst, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, &h_w, sizeof(float), cudaMemcpyHostToDevice);
    
    // 2. Run Kernel
    launch_phase_flow_float(d_alphas, d_betas, d_src, d_dst, d_w, 1, dt);
    cudaDeviceSynchronize();
    
    // 3. Check Result
    cuFloatComplex res_b[2];
    cudaMemcpy(res_b, d_betas, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    
    // Qubit 1 Beta should pick up a phase (Imaginary part)
    float imag_beta = cuCimagf(res_b[1]);
    printf("Qubit 1 Beta Imag: %f\n", imag_beta);
    
    // |imag| > 0
    EXPECT_GT(fabs(imag_beta), 0.001f) << "Kernel did not evolve phase!";
    
    cudaFree(d_alphas); cudaFree(d_betas);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_w);
}

