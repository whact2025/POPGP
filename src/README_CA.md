## 2D Cellular Automata with Cooling (v0.7)

This experiment adds a "Radiative Cooling" mechanism to the CA model to stabilize the population.

### Changes
*   **Cooling:** Cells have a 2% chance per step to spontaneously reset to a pure state (shedding entropy). This mimics the emission of radiation by matter.

### Results
*   **Stabilization:** The population no longer crashes after the initial bloom. The cooling mechanism allows dense clusters to survive by exporting their disorder.
*   **Coexistence:** We observe a stable coexistence of "Matter" (Population) and "Space" (Empty cells), with low average entropy.

### Artifacts
*   `toy_model/ca_model.py`: Simulation script (with cooling).
*   `toy_model/ca_dynamics_cooling.png`: Population and Entropy dynamics.
*   `toy_model/ca_evolution_cooling.gif`: Animation of the stabilized grid.
