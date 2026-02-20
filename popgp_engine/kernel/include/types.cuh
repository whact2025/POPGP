#pragma once
#include <cuComplex.h>

// DLL Export/Import Macros
#if defined(_WIN32)
    #if defined(POPGP_KERNEL_EXPORTS)
        #define POPGP_API __declspec(dllexport)
    #else
        #define POPGP_API __declspec(dllimport)
    #endif
#else
    #define POPGP_API __attribute__((visibility("default")))
#endif

// --- SoA Wrappers (C-API for Python) ---

extern "C" POPGP_API void launch_phase_flow_float(
    cuFloatComplex* alphas, cuFloatComplex* betas,
    const int* src, const int* dst, const float* w,
    int n, float dt
);

extern "C" POPGP_API void launch_phase_flow_double(
    cuDoubleComplex* alphas, cuDoubleComplex* betas,
    const int* src, const int* dst, const double* w,
    int n, double dt
);

extern "C" POPGP_API void launch_area_law_pruning(
    const int* src, const int* dst, const double* w,
    int* active_mask, double* cut_sizes, double* entropies,
    int num_edges, int num_nodes
);

extern "C" POPGP_API void solve_clock_potential(
    const int* src, const int* dst, const double* w,
    const double* rho, 
    double* phi,
    int num_edges, int num_nodes
);

// --- Original Structs (Internal / Host Logic) ---

struct Cell {
    cuDoubleComplex alpha;
    cuDoubleComplex beta;
};

struct Edge {
    int src;
    int dst;
    double weight;
};

