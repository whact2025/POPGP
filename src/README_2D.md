## 2D Grid Emergence (v0.6)

This experiment extends the framework to 2D geometries.

### Setup
*   **Substrate:** 9-qubit system ($3 \times 3$ grid).
*   **Hamiltonian:** 2D Heisenberg Model.
*   **Scrambling:** The system is treated as a bag of 9 sites with unknown topology.
*   **Projection:** We compute the Mutual Information distance matrix $D_{ij}$ from the thermal state ($\beta=2.0$).

### Results
*   **MDS Embedding:** The algorithm recovers a 2D arrangement where nearest-neighbor qubits (in the Hamiltonian) are spatially closer than non-neighbors.
*   **Quantitative Metric:**
    *   Avg Distance to Neighbors: **0.98**
    *   Avg Distance to Non-Neighbors: **1.32**
    *   **Result:** The local grid topology is successfully recovered from the "scrambled" algebra.

### Artifacts
*   `toy_model/grid_2d.py`: Simulation script.
*   `toy_model/2d_grid_embedding.png`: Visualization of the recovered grid.
