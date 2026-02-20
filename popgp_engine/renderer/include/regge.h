#pragma once
#include <vector>
#include <Eigen/Dense>

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
    // Computes scalar curvature (deficit angle) at each vertex of a mesh.
    // 
    // vertices: [N * 3] array (x, y, z)
    // triangles: [M * 3] array (indices)
    // curvature: [N] output array
    RENDER_API void compute_regge_curvature(
        const double* vertices, 
        const int* triangles, 
        double* curvature,
        int num_vertices, 
        int num_triangles
    );
}

