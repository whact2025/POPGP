#pragma once

#if defined(_WIN32)
    #if defined(POPGP_RENDER_EXPORTS)
        #define RENDER_API __declspec(dllexport)
    #else
        #define RENDER_API __declspec(dllimport)
    #endif
#else
    #define RENDER_API __attribute__((visibility("default")))
#endif

extern "C" {
    // C-API for Python binding
    // Iteratively updates coordinates to minimize stress (Force-Directed).
    // 
    // pos_x, pos_y, pos_z: [N] Arrays of coordinates (Input/Output)
    // edge_src, edge_dst: [M] Edge indices
    // edge_target_dist: [M] Target distance derived from Mutual Information
    // k_attract, k_repel: Physical constants for the layout simulation
    // iterations: Number of steps to run
    
    RENDER_API void step_force_layout(
        double* pos_x, double* pos_y, double* pos_z,
        const int* edge_src, const int* edge_dst, const double* edge_target_dist,
        int num_nodes, int num_edges,
        double k_attract, double k_repel,
        double dt, int iterations
    );
}

