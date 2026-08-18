#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "../include/types.cuh"

TEST(ClockTest, UnimplementedSolverFailsWithoutWritingOutput) {
    double initial_phi = 123.5;
    double* d_phi = nullptr;
    ASSERT_EQ(cudaMalloc(&d_phi, sizeof(double)), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(d_phi, &initial_phi, sizeof(double), cudaMemcpyHostToDevice),
        cudaSuccess
    );

    int status = solve_clock_potential(
        nullptr, nullptr, nullptr, nullptr, d_phi, 0, 1
    );

    double observed_phi = 0.0;
    ASSERT_EQ(
        cudaMemcpy(&observed_phi, d_phi, sizeof(double), cudaMemcpyDeviceToHost),
        cudaSuccess
    );
    EXPECT_EQ(status, POPGP_STATUS_NOT_IMPLEMENTED);
    EXPECT_DOUBLE_EQ(observed_phi, initial_phi);
    EXPECT_EQ(cudaFree(d_phi), cudaSuccess);
}
