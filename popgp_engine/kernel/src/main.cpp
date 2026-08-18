#include "types.cuh"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <random>
#include <vector>

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
        if (d_src) CUDA_CHECK(cudaFree(d_src));
        if (d_dst) CUDA_CHECK(cudaFree(d_dst));
        if (d_w) CUDA_CHECK(cudaFree(d_w));
    }
};

void checksum_bytes(std::uint64_t& checksum, const void* value, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(value);
    for (std::size_t i = 0; i < size; ++i) {
        checksum ^= bytes[i];
        checksum *= UINT64_C(1099511628211);
    }
}

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
    // Warm up context creation, module loading, and both coloured launch paths.
    launch_phase_flow_double(
        d_alphas, d_betas, d_r_src, d_r_dst, d_r_w, h_red.size(), dt
    );
    CUDA_CHECK(cudaGetLastError());
    launch_phase_flow_double(
        d_alphas, d_betas, d_b_src, d_b_dst, d_b_w, h_black.size(), dt
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset the seeded state so the measured checksum always represents exactly
    // `steps` simulation steps, independent of the warmup.
    CUDA_CHECK(cudaMemcpy(
        d_alphas, h_alphas.data(), num_cells * sizeof(cuDoubleComplex),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_betas, h_betas.data(), num_cells * sizeof(cuDoubleComplex),
        cudaMemcpyHostToDevice
    ));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int s = 0; s < steps; s++) {
        // Red Batch
        launch_phase_flow_double(d_alphas, d_betas, d_r_src, d_r_dst, d_r_w, h_red.size(), dt);
        CUDA_CHECK(cudaGetLastError());
        
        // Black Batch
        launch_phase_flow_double(d_alphas, d_betas, d_b_src, d_b_dst, d_b_w, h_black.size(), dt);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    if (!std::isfinite(milliseconds) || milliseconds <= 0.0f) {
        fmt::print(stderr, "Invalid CUDA timing result: {} ms\n", milliseconds);
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMemcpy(
        h_alphas.data(), d_alphas, num_cells * sizeof(cuDoubleComplex),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        h_betas.data(), d_betas, num_cells * sizeof(cuDoubleComplex),
        cudaMemcpyDeviceToHost
    ));

    double max_norm_error = 0.0;
    std::uint64_t checksum = UINT64_C(14695981039346656037);
    bool valid_state = true;
    for (int i = 0; i < num_cells; ++i) {
        const double ar = cuCreal(h_alphas[i]);
        const double ai = cuCimag(h_alphas[i]);
        const double br = cuCreal(h_betas[i]);
        const double bi = cuCimag(h_betas[i]);
        if (!(std::isfinite(ar) && std::isfinite(ai) &&
              std::isfinite(br) && std::isfinite(bi))) {
            valid_state = false;
            break;
        }
        const double norm = ar * ar + ai * ai + br * br + bi * bi;
        max_norm_error = std::fmax(max_norm_error, std::fabs(norm - 1.0));
        checksum_bytes(checksum, &ar, sizeof(ar));
        checksum_bytes(checksum, &ai, sizeof(ai));
        checksum_bytes(checksum, &br, sizeof(br));
        checksum_bytes(checksum, &bi, sizeof(bi));
    }
    if (!valid_state || max_norm_error > 1.0e-12) {
        fmt::print(
            stderr,
            "Invalid simulation state: finite={}, max_norm_error={:.17g}\n",
            valid_state,
            max_norm_error
        );
        return EXIT_FAILURE;
    }

    const double edge_updates =
        static_cast<double>(h_red.size() + h_black.size()) * steps;
    fmt::print("Simulation Complete. Time: {:.2f} ms\n", milliseconds);
    fmt::print("Edge updates/sec: {:.2e}\n", edge_updates / (milliseconds / 1000.0));
    fmt::print("State checksum (FNV-1a-64): {:016x}\n", checksum);
    fmt::print("Maximum norm error: {:.17g}\n", max_norm_error);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_alphas));
    CUDA_CHECK(cudaFree(d_betas));
    h_red.free_device(d_r_src, d_r_dst, d_r_w);
    h_black.free_device(d_b_src, d_b_dst, d_b_w);
    
    return 0;
}
