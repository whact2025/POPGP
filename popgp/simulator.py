# Copyright (c) 2026 WHACT. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Unified POPGP Simulator.

Architecture Decision 1: single simulator, not separate toy/native paths.

This class composes the four projection stages into one pipeline:

    (A, ω, α) → Π_res → Π_loc → Π_geom → Π_time → Observables

Each stage is a method that reads from the canonical SimulatorConfig and
delegates quantum operations to the Backend (exact or GPU, selected
automatically).  The pipeline can be run end-to-end via :meth:`run`, or
stage-by-stage for inspection and debugging.

Implements the projection map Π (§4.4) of docs/framework.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from popgp.backend import Backend, _I2, _SX, _SY, _SZ, create_backend
from popgp.coarse_grain import (
    check_su2_equivariance,
    compute_leakage,
    compute_retention_loss,
    count_partitions,
    optimize_cells,
)
from popgp.config import SimulatorConfig

log = logging.getLogger(__name__)


# ── Pipeline result containers ──────────────────────────────────────────


@dataclass
class PiResResult:
    """Output of the Π_res stage (§4.4.2)."""

    cells: list[list[int]]
    """Selected cell decomposition: cells[i] = list of qubit indices."""

    leakage: float
    """L_leak value for the selected decomposition."""

    drift: float | None = None
    """L_drift value (tie-breaker), or None if not computed."""

    retention_loss: float | None = None
    """D(ω ‖ ω∘E) for the selected decomposition."""

    su2_equivariant: bool | None = None
    """Whether the SU(2) equivariance check passed."""


@dataclass
class PiLocResult:
    """Output of the Π_loc stage (§4.4.3)."""

    mi_matrix: torch.Tensor
    """Mutual information matrix I_ij [n_cells × n_cells]."""

    distance_matrix: torch.Tensor
    """Graph-metric distance matrix d_G(i,j) [n_cells × n_cells]."""

    weight_matrix: torch.Tensor
    """Edge weight matrix w_ij = κ(I_ij) [n_cells × n_cells]."""

    edges: list[tuple[int, int]]
    """Connectivity edges after k-NN + MST enforcement."""


@dataclass
class PiGeomResult:
    """Output of the Π_geom stage (§4.4.4)."""

    D_star: int
    """Selected embedding dimension."""

    D_spectral: float
    """Spectral dimension of the correlation graph."""

    coords: torch.Tensor
    """Embedded coordinates [n_cells × D_star]."""

    stress: float
    """MDS stress of the embedding."""

    h_ab: list[torch.Tensor] | None = None
    """Local metric tensors h_ab(x_i), one per node.  [D_star × D_star]."""

    simplices: Any = None
    """Vietoris-Rips simplicial complex (or Delaunay proxy in toy models, §8.2.1)."""

    deficit_angles: dict | None = None
    """Regge deficit angles at each hinge."""


@dataclass
class PiTimeResult:
    """Output of the Π_time stage (§4.4.5)."""

    phi: torch.Tensor
    """Clock-rate potential Φ_i at each cell."""

    delta_rho: torch.Tensor
    """Source term δρ_i (temporal-averaged Araki contrast)."""

    dtau: torch.Tensor
    """Proper time increment dτ_i = β_0 · exp(Φ_i) · dS_act, where dS_act is
    the incremental trace-distance advanced by the canonical flow σ_s (§4.4.5).
    Current implementation uses dt (phase-order step) as a placeholder."""


@dataclass
class SimulatorResult:
    """Complete output of a full pipeline run."""

    config: SimulatorConfig
    state: object
    pi_res: PiResResult
    pi_loc: PiLocResult
    pi_geom: PiGeomResult
    pi_time: PiTimeResult | None = None
    metadata: dict = field(default_factory=dict)


# ── The Simulator ───────────────────────────────────────────────────────


class Simulator:
    """
    Unified POPGP Simulator.

    Architecture Decision 1: one class, one pipeline, automatic backend
    selection.  No conditional logic between toy and native code paths.

    Usage::

        from popgp import Simulator, SimulatorConfig

        cfg = SimulatorConfig.for_grid(width=3, height=3, beta=2.0)
        sim = Simulator(cfg)
        result = sim.run()

        # Or stage-by-stage:
        state = sim.prepare()
        state = sim.evolve(state, n_steps=50)
        pi_res = sim.run_pi_res(state)
        pi_loc = sim.run_pi_loc(state, pi_res)
        pi_geom = sim.run_pi_geom(pi_loc)
        pi_time = sim.run_pi_time(state, pi_res, pi_loc, pi_geom)
    """

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        self.backend: Backend = create_backend(self.config)
        log.info(
            "Simulator initialized: N=%d, backend=%s",
            self.config.substrate.n_qubits,
            type(self.backend).__name__,
        )

    # ── substrate ────────────────────────────────────────────────────

    def prepare(self) -> object:
        """Build the substrate and prepare the initial state ω (§4.4.1)."""
        torch.manual_seed(self.config.substrate.seed)
        np.random.seed(self.config.substrate.seed)
        return self.backend.prepare_state()

    def evolve(self, state: object, n_steps: int | None = None) -> object:
        """Advance the state by n_steps phase-order steps (§4.3)."""
        steps = n_steps or self.config.simulation.n_steps
        dt = self.config.simulation.dt
        for _ in range(steps):
            state = self.backend.evolve(state, dt)
        return state

    # ── Π_res: resolution-limited coarse-graining ────────────────────

    def run_pi_res(self, state: object) -> PiResResult:
        """
        Execute the Π_res projection stage (§4.4.2, §4.4.2a).

        Selects the cell decomposition that minimizes phase-flow leakage
        L_leak under admissibility constraints (SU(2) equivariance,
        retention bound, finite capacity).

        The framework (v1.0, §4.4.2a) defines E* as the stable fixed-point
        of a causal gradient flow.  At toy scale (exact backend), the
        attractor is located via exhaustive search over all equal-size
        partitions: minimize L_leak (primary), then L_drift (tie-breaker),
        subject to SU(2) equivariance and retention bound constraints.

        At GPU scale: uses heuristic (contiguous blocks along the graph).
        """
        cfg_res = self.config.pi_res
        cfg_sub = self.config.substrate
        N = cfg_sub.n_qubits
        k = cfg_res.cell_dim

        if N % k != 0:
            raise ValueError(
                f"n_qubits ({N}) must be divisible by cell_dim ({k})."
            )

        if k == 1:
            cells = [[i] for i in range(N)]
            result = {
                "cells": cells,
                "leakage": 0.0,
                "drift": None,
                "retention_loss": 0.0,
                "su2_equivariant": True,
            }
        elif self.config.use_exact_backend:
            result = self._pi_res_exact(state)
        else:
            n_cells = N // k
            cells = [list(range(i * k, (i + 1) * k)) for i in range(n_cells)]
            result = {
                "cells": cells,
                "leakage": float("nan"),
                "drift": None,
                "retention_loss": None,
                "su2_equivariant": None,
            }

        log.info(
            "Π_res: %d cells of size %d, L_leak=%.6e",
            len(result["cells"]), k, result["leakage"],
        )
        return PiResResult(
            cells=result["cells"],
            leakage=result["leakage"],
            drift=result.get("drift"),
            retention_loss=result.get("retention_loss"),
            su2_equivariant=result.get("su2_equivariant"),
        )

    def _pi_res_exact(self, state: object) -> dict:
        """Locate the causal flow attractor via combinatorial search (§4.4.2a).

        Delegates to :func:`popgp.coarse_grain.optimize_cells`, which
        enumerates all equal-size partitions, filters by admissibility,
        and minimizes L_leak with L_drift as tie-breaker.
        """
        cfg_res = self.config.pi_res
        cfg_sub = self.config.substrate

        return optimize_cells(
            state=state,
            backend=self.backend,
            n_qubits=cfg_sub.n_qubits,
            cell_dim=cfg_res.cell_dim,
            edges=self.backend.build_edges(),
            coupling_J=cfg_sub.coupling_J,
            phase_window_width=cfg_res.phase_window_width,
            phase_window_samples=cfg_res.phase_window_samples,
            drift_delta=cfg_res.drift_delta,
            retention_epsilon=cfg_res.retention_epsilon,
            leakage_tie_tolerance=1e-8,
        )

    # ── Π_loc: locality from correlations ────────────────────────────

    def run_pi_loc(
        self, state: object, pi_res: PiResResult
    ) -> PiLocResult:
        """
        Execute the Π_loc projection stage (§4.4.3).

        Computes pairwise mutual information between all cells, applies
        the canonical distance kernel, builds the weighted connectivity
        graph, and computes graph-geodesic distances.

        Note: the framework specifies QCMI (Quantum Conditional Mutual
        Information) Markovian Geometric Filtering to screen out long-range
        topological entanglement from true geometric proximity.  This is
        deferred to a future phase; the current implementation uses raw
        pairwise MI as a placeholder.
        """
        cells = pi_res.cells
        n_cells = len(cells)
        cfg_loc = self.config.pi_loc

        mi_matrix = torch.zeros((n_cells, n_cells))
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                mi = self.backend.mutual_information(
                    state, cells[i], cells[j]
                )
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        I_0 = cfg_loc.I_0 if cfg_loc.I_0 is not None else mi_matrix.max().item()
        if I_0 <= 0:
            I_0 = 1.0

        dist_matrix = torch.zeros((n_cells, n_cells))
        weight_matrix = torch.zeros((n_cells, n_cells))
        for i in range(n_cells):
            for j in range(n_cells):
                if i == j:
                    continue
                ratio = mi_matrix[i, j].item() / I_0
                ratio = max(ratio, cfg_loc.mi_epsilon)
                dist_matrix[i, j] = cfg_loc.distance_kernel(ratio)
                weight_matrix[i, j] = cfg_loc.weight_kernel(
                    mi_matrix[i, j].item()
                )

        graph_dist, edges = self._graph_geodesics(
            dist_matrix, weight_matrix, n_cells, cfg_loc.k_nearest
        )

        log.info(
            "Π_loc: MI range [%.4f, %.4f], I_0=%.4f, %d edges",
            mi_matrix[mi_matrix > 0].min().item() if (mi_matrix > 0).any() else 0,
            mi_matrix.max().item(),
            I_0,
            len(edges),
        )
        return PiLocResult(
            mi_matrix=mi_matrix,
            distance_matrix=graph_dist,
            weight_matrix=weight_matrix,
            edges=edges,
        )

    def _graph_geodesics(
        self,
        dist_matrix: torch.Tensor,
        weight_matrix: torch.Tensor,
        n: int,
        k_nearest: int,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Compute shortest-path distances on the k-NN + MST graph (§4.4.3)."""
        adj = torch.full((n, n), float("inf"))
        edges: set[tuple[int, int]] = set()

        for i in range(n):
            dists_i = dist_matrix[i].clone()
            dists_i[i] = float("inf")
            _, neighbors = torch.topk(dists_i, min(k_nearest, n - 1), largest=False)
            for j_idx in neighbors:
                j = j_idx.item()
                adj[i, j] = dist_matrix[i, j]
                adj[j, i] = dist_matrix[j, i]
                edge = (min(i, j), max(i, j))
                edges.add(edge)

        mst_edges = self._minimum_spanning_tree(dist_matrix, n)
        for i, j in mst_edges:
            adj[i, j] = dist_matrix[i, j]
            adj[j, i] = dist_matrix[j, i]
            edges.add((min(i, j), max(i, j)))

        for i in range(n):
            adj[i, i] = 0.0

        graph_dist = self._floyd_warshall(adj, n)
        return graph_dist, sorted(edges)

    @staticmethod
    def _minimum_spanning_tree(
        dist_matrix: torch.Tensor, n: int
    ) -> list[tuple[int, int]]:
        """Prim's MST on the full distance matrix."""
        in_tree = [False] * n
        in_tree[0] = True
        edges_out: list[tuple[int, int]] = []
        min_edge = dist_matrix[0].clone()
        min_from = [0] * n

        for _ in range(n - 1):
            masked = min_edge.clone()
            for i in range(n):
                if in_tree[i]:
                    masked[i] = float("inf")
            j = masked.argmin().item()
            in_tree[j] = True
            edges_out.append((min(min_from[j], j), max(min_from[j], j)))

            for k in range(n):
                if not in_tree[k] and dist_matrix[j, k] < min_edge[k]:
                    min_edge[k] = dist_matrix[j, k]
                    min_from[k] = j

        return edges_out

    @staticmethod
    def _floyd_warshall(adj: torch.Tensor, n: int) -> torch.Tensor:
        """All-pairs shortest paths."""
        dist = adj.clone()
        for via in range(n):
            for i in range(n):
                for j in range(n):
                    candidate = dist[i, via] + dist[via, j]
                    if candidate < dist[i, j]:
                        dist[i, j] = candidate
        return dist

    # ── Π_geom: emergent geometry ────────────────────────────────────

    def run_pi_geom(self, pi_loc: PiLocResult) -> PiGeomResult:
        """
        Execute the Π_geom projection stage (§4.4.4).

        1. Compute spectral dimension D_S from the graph Laplacian.
        2. Select D* via the complexity-stress functional.
        3. Embed via classical MDS into ℝ^{D*}.
        4. (Future phases: local metric, Vietoris-Rips complex, Regge curvature.)
        """
        cfg_geom = self.config.pi_geom
        d_G = pi_loc.distance_matrix
        w = pi_loc.weight_matrix
        n = d_G.shape[0]

        D_S = self._spectral_dimension(w, n)

        best_D, best_stress, best_coords = 1, float("inf"), None
        for D in range(1, cfg_geom.D_max + 1):
            coords = self._classical_mds(d_G, D)
            stress = self._mds_stress(d_G, coords)
            F_D = stress + cfg_geom.lambda_dim * (D - D_S) ** 2
            if F_D < best_stress:
                best_D, best_stress, best_coords = D, F_D, coords

        log.info(
            "Π_geom: D_S=%.2f, D*=%d, stress=%.4f",
            D_S, best_D, best_stress,
        )
        return PiGeomResult(
            D_star=best_D,
            D_spectral=D_S,
            coords=best_coords,
            stress=best_stress,
        )

    def _spectral_dimension(self, weight_matrix: torch.Tensor, n: int) -> float:
        """Estimate D_S from heat kernel trace Tr(e^{−tΔ}) (§4.4.4)."""
        L = torch.diag(weight_matrix.sum(dim=1)) - weight_matrix
        evals = torch.linalg.eigvalsh(L).real
        evals = evals[evals > 1e-10]
        if len(evals) < 2:
            return 1.0

        t_values = torch.logspace(-1, 1, steps=50)
        traces = []
        for t in t_values:
            traces.append(torch.sum(torch.exp(-t * evals)).item())
        traces = torch.tensor(traces)
        log_t = torch.log(t_values)
        log_tr = torch.log(traces.clamp(min=1e-30))

        d_log_tr = (log_tr[1:] - log_tr[:-1]) / (log_t[1:] - log_t[:-1])
        D_S = -2 * d_log_tr.median().item()
        return max(1.0, D_S)

    @staticmethod
    def _classical_mds(d_matrix: torch.Tensor, D: int) -> torch.Tensor:
        """Classical MDS embedding into ℝ^D (§4.4.4 step 2)."""
        n = d_matrix.shape[0]
        D2 = d_matrix**2
        J = torch.eye(n) - torch.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J

        evals, evecs = torch.linalg.eigh(B)
        idx = evals.argsort(descending=True)
        evals = evals[idx]
        evecs = evecs[:, idx]

        D_eff = min(D, n - 1)
        top_evals = evals[:D_eff].clamp(min=0)
        coords = evecs[:, :D_eff] @ torch.diag(torch.sqrt(top_evals))
        return coords.real

    @staticmethod
    def _mds_stress(d_target: torch.Tensor, coords: torch.Tensor) -> float:
        """Kruskal stress-1: normalized residual of distance reproduction."""
        n = coords.shape[0]
        d_embed = torch.cdist(coords.float(), coords.float()).double()
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        residuals = (d_target[mask] - d_embed[mask]) ** 2
        denom = (d_target[mask] ** 2).sum()
        if denom < 1e-15:
            return 0.0
        return torch.sqrt(residuals.sum() / denom).item()

    # ── Π_time: emergent time ────────────────────────────────────────

    def run_pi_time(
        self,
        state: object,
        pi_res: PiResResult,
        pi_loc: PiLocResult,
        pi_geom: PiGeomResult,
    ) -> PiTimeResult:
        """
        Execute the Π_time projection stage (§4.4.5).

        1. Compute the entropy-contrast source term δρ.
        2. Build the weighted graph Laplacian.
        3. Solve (Δ_w + μ²I)Φ = δρ for the clock-rate potential.
           This acts as an elliptic constraint equation on the foliation
           (analogous to the ADM Hamiltonian constraint), not as an
           acausal dynamical propagator.
        4. Compute proper time dτ = β_0 · exp(Φ) · dS_act.
        """
        cfg_time = self.config.pi_time
        cells = pi_res.cells
        n_cells = len(cells)
        w = pi_loc.weight_matrix

        delta_rho = self._compute_source_term(state, cells, cfg_time)

        L = torch.diag(w.sum(dim=1)) - w
        A = L + cfg_time.mu**2 * torch.eye(n_cells)

        if cfg_time.mu == 0.0:
            A[0, :] = 0.0
            A[0, 0] = 1.0
            delta_rho_pinned = delta_rho.clone()
            delta_rho_pinned[0] = 0.0
            phi = torch.linalg.solve(A, delta_rho_pinned)
        else:
            phi = torch.linalg.solve(A, delta_rho)

        dt = self.config.simulation.dt
        dtau = cfg_time.beta_0 * torch.exp(phi) * dt

        log.info(
            "Π_time: Φ range [%.4f, %.4f], μ=%.2f",
            phi.min().item(), phi.max().item(), cfg_time.mu,
        )
        return PiTimeResult(phi=phi, delta_rho=delta_rho, dtau=dtau)

    def _compute_source_term(
        self,
        state: object,
        cells: list[list[int]],
        cfg_time,
    ) -> torch.Tensor:
        """δρ_i = S(ρ_i) as a simple entropy proxy.

        The full framework (§4.4.5, §8.2.2) defines:
            δρ_i := -(1/s₀) S_Araki(ω_i ‖ ω_i^vac)
        with an explicit minus sign ensuring δρ < 0 for all physical
        excitations.  In the GR matching (§8.2.2), δρ acts as the
        effective energy density component T_00 driving the lapse Φ.

        This initial implementation uses von Neumann entropy (always ≥ 0)
        as a placeholder.  The sign inversion and KMS vacuum baseline
        are deferred to Phase 4.  Examples that require δρ < 0 (gravity
        wells) must override the source term manually.
        """
        n_cells = len(cells)
        delta_rho = torch.zeros(n_cells)
        for i, cell in enumerate(cells):
            rho_i = self.backend.reduced_state(state, cell)
            delta_rho[i] = self.backend.entropy(rho_i)
        return delta_rho

    # ── full pipeline ────────────────────────────────────────────────

    def run(self, evolve_steps: int | None = None) -> SimulatorResult:
        """
        Execute the complete projection pipeline Π = Π_time ∘ Π_geom ∘ Π_loc ∘ Π_res.

        This is the primary entry point for end-to-end simulation.
        """
        state = self.prepare()
        if evolve_steps:
            state = self.evolve(state, n_steps=evolve_steps)

        pi_res = self.run_pi_res(state)
        pi_loc = self.run_pi_loc(state, pi_res)
        pi_geom = self.run_pi_geom(pi_loc)

        pi_time = None
        try:
            pi_time = self.run_pi_time(state, pi_res, pi_loc, pi_geom)
        except Exception as exc:
            log.warning("Π_time failed (non-fatal): %s", exc)

        return SimulatorResult(
            config=self.config,
            state=state,
            pi_res=pi_res,
            pi_loc=pi_loc,
            pi_geom=pi_geom,
            pi_time=pi_time,
        )
