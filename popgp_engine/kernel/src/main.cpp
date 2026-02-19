#include "types.cuh"
#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <fmt/core.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper for SoA Data
struct EdgeSoA {
    std::vector<int> src;
    std::vector<int> dst;
    std::vector<double> w;
    
    void push_back(int s, int d, double weight) {
        src.push_back(s);
        dst.push_back(d);
        w.push_back(weight);
    }
    
    size_t size() const { return src.size(); }
    
    void upload_to_device(int** d_src, int** d_dst, double** d_w) {
        if (src.empty()) return;
        size_t n = src.size();
        CUDA_CHECK(cudaMalloc(d_src, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(d_dst, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(d_w, n * sizeof(double)));
        
        CUDA_CHECK(cudaMemcpy(*d_src, src.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_dst, dst.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_w, w.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    void free_device(int* d_src, int* d_dst, double* d_w) {
        if (d_src) cudaFree(d_src);
        if (d_dst) cudaFree(d_dst);
        if (d_w) cudaFree(d_w);
    }
};

void color_graph_1d(
    int num_cells, 
    EdgeSoA& red, 
    EdgeSoA& black
) {
    for (int i = 0; i < num_cells - 1; i++) {
        // Red: (0,1), (2,3)... Black: (1,2), (3,4)...
        if (i % 2 == 0) {
            red.push_back(i, i + 1, 1.0);
        } else {
            black.push_back(i, i + 1, 1.0);
        }
    }
}

int main() {
    int num_cells = 1000000; // 1 Million Cells
    double dt = 0.01;
    int steps = 100;
    
    fmt::print("Initializing POPGP Kernel (Optimized SoA)...\n");

    // 1. Initialize State (SoA for Cells too!)
    std::vector<cuDoubleComplex> h_alphas(num_cells);
    std::vector<cuDoubleComplex> h_betas(num_cells);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < num_cells; i++) {
        double r = dis(gen);
        double theta = dis(gen) * 2 * 3.14159;
        h_alphas[i] = make_cuDoubleComplex(sqrt(r), 0);
        h_betas[i]  = make_cuDoubleComplex(sqrt(1-r) * cos(theta), sqrt(1-r) * sin(theta));
    }

    // 2. Initialize Graph (SoA)
    EdgeSoA h_red, h_black;
    color_graph_1d(num_cells, h_red, h_black);
    
    fmt::print("Graph: {} Red, {} Black\n", h_red.size(), h_black.size());

    // 3. Allocate Device Memory (SoA for Cells)
    cuDoubleComplex *d_alphas, *d_betas;
    CUDA_CHECK(cudaMalloc(&d_alphas, num_cells * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_betas, num_cells * sizeof(cuDoubleComplex)));
    
    CUDA_CHECK(cudaMemcpy(d_alphas, h_alphas.data(), num_cells * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_betas, h_betas.data(), num_cells * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Allocate Edges
    int *d_r_src=0, *d_r_dst=0; double *d_r_w=0;
    int *d_b_src=0, *d_b_dst=0; double *d_b_w=0;
    
    h_red.upload_to_device(&d_r_src, &d_r_dst, &d_r_w);
    h_black.upload_to_device(&d_b_src, &d_b_dst, &d_b_w);

    // 4. Run Simulation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int s = 0; s < steps; s++) {
        // Red Batch
        launch_phase_flow_double(d_alphas, d_betas, d_r_src, d_r_dst, d_r_w, h_red.size(), dt);
        
        // Black Batch
        launch_phase_flow_double(d_alphas, d_betas, d_b_src, d_b_dst, d_b_w, h_black.size(), dt);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    fmt::print("Simulation Complete. Time: {:.2f} ms\n", milliseconds);
    fmt::print("Updates/sec: {:.2e}\n", (double)(num_cells) * steps / (milliseconds / 1000.0));

    // Cleanup
    cudaFree(d_alphas); cudaFree(d_betas);
    h_red.free_device(d_r_src, d_r_dst, d_r_w);
    h_black.free_device(d_b_src, d_b_dst, d_b_w);
    
    return 0;
}
