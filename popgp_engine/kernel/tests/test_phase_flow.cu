#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/types.cuh"

TEST(PhaseFlowTest, TwoQubitEntanglement) {
    int n = 2;
    float dt = 0.1f;
    float J = 1.0f;
    
    // Initial State |00> (alpha=1, beta=0)
    cuFloatComplex* d_alphas;
    cuFloatComplex* d_betas;
    cudaMalloc(&d_alphas, n * sizeof(cuFloatComplex));
    cudaMalloc(&d_betas, n * sizeof(cuFloatComplex));
    
    cuFloatComplex zero = make_cuFloatComplex(0.0f, 0.0f);
    cuFloatComplex one = make_cuFloatComplex(1.0f, 0.0f);
    
    cuFloatComplex h_alphas[] = {one, one};
    cuFloatComplex h_betas[] = {zero, zero};
    
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
    cuFloatComplex res_a[2], res_b[2];
    cudaMemcpy(res_a, d_alphas, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_b, d_betas, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    
    // Logic Check:
    // After interaction, |0> state should mix with |1>.
    // |alpha| should decrease from 1.0. |beta| should increase from 0.0.
    
    float mag_beta = cuCabsf(res_b[0]);
    printf("Beta Magnitude: %f\n", mag_beta);
    
    EXPECT_GT(mag_beta, 0.001f) << "Kernel did not evolve the state!";
    
    // Check Norm Conservation
    float norm0 = cuCabsf(res_a[0])*cuCabsf(res_a[0]) + cuCabsf(res_b[0])*cuCabsf(res_b[0]);
    EXPECT_NEAR(norm0, 1.0f, 1e-5);
    
    cudaFree(d_alphas); cudaFree(d_betas);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_w);
}

