#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/types.cuh"
#include <math.h>

// Phase evolution: |0>|+> should rotate the |+> state's phase.
TEST(PhaseFlowTest, TwoQubitPhaseEvolution) {
    int n = 2;
    float dt = 0.5f;
    float J = 1.0f;

    cuFloatComplex* d_alphas;
    cuFloatComplex* d_betas;
    cudaMalloc(&d_alphas, n * sizeof(cuFloatComplex));
    cudaMalloc(&d_betas, n * sizeof(cuFloatComplex));

    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    cuFloatComplex plus_state = make_cuFloatComplex(inv_sqrt2, 0.0f);
    cuFloatComplex zero_state = make_cuFloatComplex(1.0f, 0.0f);

    cuFloatComplex h_alphas[] = {zero_state, plus_state};
    cuFloatComplex h_betas[]  = {make_cuFloatComplex(0,0), plus_state};

    cudaMemcpy(d_alphas, h_alphas, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_betas, h_betas, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    int* d_src; int* d_dst; float* d_w;
    cudaMalloc(&d_src, sizeof(int));
    cudaMalloc(&d_dst, sizeof(int));
    cudaMalloc(&d_w, sizeof(float));

    int h_src = 0; int h_dst = 1; float h_w = J;
    cudaMemcpy(d_src, &h_src, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, &h_dst, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, &h_w, sizeof(float), cudaMemcpyHostToDevice);

    launch_phase_flow_float(d_alphas, d_betas, d_src, d_dst, d_w, 1, dt);
    cudaDeviceSynchronize();

    cuFloatComplex res_a[2], res_b[2];
    cudaMemcpy(res_a, d_alphas, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_b, d_betas, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    float imag_beta = cuCimagf(res_b[1]);
    printf("Qubit 1 Beta Imag: %f\n", imag_beta);
    EXPECT_GT(fabs(imag_beta), 0.001f) << "Kernel did not evolve phase!";

    cudaFree(d_alphas); cudaFree(d_betas);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_w);
}

// The full Heisenberg interaction must exchange amplitude between cells.
// With the old Ising-only kernel, |alpha|^2 and |beta|^2 never changed;
// now they must change when the cells are not aligned along Z.
TEST(PhaseFlowTest, AmplitudeExchange) {
    int n = 2;
    float dt = 0.3f;
    float J = 1.0f;

    cuFloatComplex* d_alphas;
    cuFloatComplex* d_betas;
    cudaMalloc(&d_alphas, n * sizeof(cuFloatComplex));
    cudaMalloc(&d_betas, n * sizeof(cuFloatComplex));

    // Cell 0: |0> (Sz = +1, pure Z-up)
    // Cell 1: |+> = (|0> + |1>)/sqrt(2)  (Sz = 0, equatorial)
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);

    cuFloatComplex h_alphas[] = {
        make_cuFloatComplex(1.0f, 0.0f),      // cell 0 alpha
        make_cuFloatComplex(inv_sqrt2, 0.0f),  // cell 1 alpha
    };
    cuFloatComplex h_betas[] = {
        make_cuFloatComplex(0.0f, 0.0f),       // cell 0 beta
        make_cuFloatComplex(inv_sqrt2, 0.0f),   // cell 1 beta
    };

    cudaMemcpy(d_alphas, h_alphas, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_betas, h_betas, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    int* d_src; int* d_dst; float* d_w;
    cudaMalloc(&d_src, sizeof(int));
    cudaMalloc(&d_dst, sizeof(int));
    cudaMalloc(&d_w, sizeof(float));

    int h_src = 0; int h_dst = 1; float h_w = J;
    cudaMemcpy(d_src, &h_src, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, &h_dst, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, &h_w, sizeof(float), cudaMemcpyHostToDevice);

    launch_phase_flow_float(d_alphas, d_betas, d_src, d_dst, d_w, 1, dt);
    cudaDeviceSynchronize();

    cuFloatComplex res_a[2], res_b[2];
    cudaMemcpy(res_a, d_alphas, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_b, d_betas, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    // Cell 0 started with |beta|^2 = 0. After full Heisenberg interaction
    // with an equatorial neighbour, amplitude exchange gives |beta_0|^2 > 0.
    float beta0_sq = cuCrealf(res_b[0]) * cuCrealf(res_b[0])
                   + cuCimagf(res_b[0]) * cuCimagf(res_b[0]);
    printf("Cell 0 |beta|^2 after interaction: %f\n", beta0_sq);
    EXPECT_GT(beta0_sq, 0.01f) << "Heisenberg flip-flop did not transfer amplitude!";

    // Verify unitarity: norm is preserved for both cells
    float norm0 = cuCrealf(res_a[0]) * cuCrealf(res_a[0])
                + cuCimagf(res_a[0]) * cuCimagf(res_a[0])
                + beta0_sq;
    float alpha1_sq = cuCrealf(res_a[1]) * cuCrealf(res_a[1])
                    + cuCimagf(res_a[1]) * cuCimagf(res_a[1]);
    float beta1_sq = cuCrealf(res_b[1]) * cuCrealf(res_b[1])
                   + cuCimagf(res_b[1]) * cuCimagf(res_b[1]);
    float norm1 = alpha1_sq + beta1_sq;

    printf("Cell 0 norm: %f, Cell 1 norm: %f\n", norm0, norm1);
    EXPECT_NEAR(norm0, 1.0f, 1e-4f) << "Unitarity violated for cell 0!";
    EXPECT_NEAR(norm1, 1.0f, 1e-4f) << "Unitarity violated for cell 1!";

    cudaFree(d_alphas); cudaFree(d_betas);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_w);
}

// When both cells point the same direction (aligned Z), the interaction
// should reduce to a pure phase rotation with no amplitude exchange.
TEST(PhaseFlowTest, AlignedCellsNoFlipFlop) {
    int n = 2;
    float dt = 0.5f;
    float J = 1.0f;

    cuFloatComplex* d_alphas;
    cuFloatComplex* d_betas;
    cudaMalloc(&d_alphas, n * sizeof(cuFloatComplex));
    cudaMalloc(&d_betas, n * sizeof(cuFloatComplex));

    // Both cells in |0> state (aligned along +Z)
    cuFloatComplex h_alphas[] = {
        make_cuFloatComplex(1.0f, 0.0f),
        make_cuFloatComplex(1.0f, 0.0f),
    };
    cuFloatComplex h_betas[] = {
        make_cuFloatComplex(0.0f, 0.0f),
        make_cuFloatComplex(0.0f, 0.0f),
    };

    cudaMemcpy(d_alphas, h_alphas, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_betas, h_betas, n * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    int* d_src; int* d_dst; float* d_w;
    cudaMalloc(&d_src, sizeof(int));
    cudaMalloc(&d_dst, sizeof(int));
    cudaMalloc(&d_w, sizeof(float));

    int h_src = 0; int h_dst = 1; float h_w = J;
    cudaMemcpy(d_src, &h_src, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, &h_dst, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, &h_w, sizeof(float), cudaMemcpyHostToDevice);

    launch_phase_flow_float(d_alphas, d_betas, d_src, d_dst, d_w, 1, dt);
    cudaDeviceSynchronize();

    cuFloatComplex res_b[2];
    cudaMemcpy(res_b, d_betas, n * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    // Both started with beta=0. With aligned Z states (hx=hy=0), the
    // off-diagonal U01/U10 are zero, so beta must stay zero.
    float beta0_sq = cuCrealf(res_b[0]) * cuCrealf(res_b[0])
                   + cuCimagf(res_b[0]) * cuCimagf(res_b[0]);
    float beta1_sq = cuCrealf(res_b[1]) * cuCrealf(res_b[1])
                   + cuCimagf(res_b[1]) * cuCimagf(res_b[1]);

    printf("Aligned: Cell 0 |beta|^2 = %e, Cell 1 |beta|^2 = %e\n", beta0_sq, beta1_sq);
    EXPECT_LT(beta0_sq, 1e-10f) << "Unexpected amplitude exchange for aligned cells!";
    EXPECT_LT(beta1_sq, 1e-10f) << "Unexpected amplitude exchange for aligned cells!";

    cudaFree(d_alphas); cudaFree(d_betas);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_w);
}
