# POPGP Engine: 3D Projection Renderer

This directory contains the visualization pipeline that projects the algebraic substrate into a 3D geometric manifold ($\Pi_{geom}$).

## Core Components
1.  **Metric Computer:** Calculates the Mutual Information distance ($d_{ij}$) from the quantum state.
2.  **MDS Embedder:** Solves for 3D coordinates $(x, y, z)$ that best preserve the graph distances (using Multidimensional Scaling or Force-Directed Layout).
3.  **Holographic Shader:** Visualizes local entropy production ($\dot{S}$) as brightness/color temperature (simulating radiation).
4.  **Real-Time Viewer:** OpenGL/Vulkan window to display the evolving simulation.

## Implementation Steps (Phase 4)
- [ ] Implement `compute_distance_matrix` (CPU/GPU).
- [ ] Implement `MDS_embedding` or `ForceDirected_layout` algorithm.
- [ ] Build a simple OpenGL/Vulkan renderer to draw points (Cells) and lines (Edges).
- [ ] Add shader for entropy-based coloring ("Heat Map").
