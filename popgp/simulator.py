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

from popgp.backend import Backend, ExactBackend, create_backend
from popgp.coarse_grain import (
    compute_retention_loss,
    optimize_cells,
)
from popgp.config import SimulatorConfig
from popgp.geometry import (
    build_delaunay_proxy,
    reconstruct_local_metrics,
    vertex_deficits_2d,
)
from popgp.information import finite_gibbs_state, modular_energy_delta

log = logging.getLogger(__name__)


KMS_REFERENCE_TRACE_DISTANCE_TOLERANCE = 1e-10
"""Maximum trace distance accepted by the runtime KMS-reference check."""


# ── Pipeline result containers ──────────────────────────────────────────


@dataclass
class PiResResult:
    """Output of the Π_res stage (§4.4.2)."""

    cells: list[list[int]]
    """Selected cell decomposition: cells[i] = list of qubit indices."""

    leakage: float | None
    """Unnormalized common-probe leakage ranking, or None if not computed."""

    drift: float | None = None
    """L_drift value (tie-breaker), or None if not computed."""

    retention_loss: float | None = None
    """D(ω ‖ ω∘E) for the selected decomposition."""

    su2_equivariant: bool | None = None
    """Whether the SU(2) equivariance check passed."""

    admissible: bool | None = None
    """Whether the fixed/selected decomposition passed implemented constraints."""

    n_total: int | None = None
    """Number of candidate partitions considered, if known."""

    n_admissible: int | None = None
    """Number of candidates passing implemented admissibility checks."""


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

    connectivity_method: str = "unspecified"
    """Blind inference rule used to select graph edges."""

    connectivity_threshold: float | None = None
    """MI threshold selected by an adaptive rule, if applicable."""

    connectivity_gap_ratio: float | None = None
    """Largest multiplicative separation in the positive MI spectrum."""

    connectivity_separable: bool | None = None
    """Whether the MI spectrum passed the configured gap criterion."""


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

    objective: float
    """Dimension-selection objective: stress plus the spectral penalty."""

    stress_by_dimension: dict[int, float] = field(default_factory=dict)
    """Unpenalized MDS stress for every candidate dimension."""

    objective_by_dimension: dict[int, float] = field(default_factory=dict)
    """Penalized selection objective for every candidate dimension."""

    spectral_peak_time: float | None = None
    """Diffusion scale at which the finite-graph spectral dimension peaks."""

    spectral_dimension_curve: list[float] = field(default_factory=list)
    """Scale-dependent finite-graph spectral-dimension diagnostic."""

    embedding_status: str = "unclassified"
    """Finite diagnostic: geometric_candidate, poor_fit, or non_geometric_dimension."""

    selection_margin: float = 0.0
    """Objective difference between the best and second-best dimensions."""

    h_ab: list[torch.Tensor] | None = None
    """Local metric tensors h_ab(x_i), one per node.  [D_star × D_star]."""

    metric_diagnostics: list[dict[str, float | int | bool]] = field(default_factory=list)
    """Condition, residual, rank, and constraint count for local metric fits."""

    simplices: Any = None
    """Vietoris-Rips simplicial complex (or Delaunay proxy in toy models, §8.2.1)."""

    deficit_angles: dict | None = None
    """Regge deficit angles at each hinge."""

    complex_status: str = "not_constructed"
    """Scientific status of the simplicial-complex/curvature diagnostic."""


@dataclass
class PiTimeResult:
    """Output of the Π_time stage (§4.4.5)."""

    phi: torch.Tensor
    """Clock-rate potential Φ_i at each cell."""

    delta_rho: torch.Tensor
    """Effective source after finite-graph zero-mode handling."""

    delta_rho_raw: torch.Tensor
    """Unmodified source produced by the configured source model."""

    source_model: str
    """Name of the source model used for this diagnostic."""

    source_status: str
    """Scientific status of the source model (placeholder/candidate/validated)."""

    source_background: float
    """Constant source mode removed before solving, if any."""

    constraint_residual: float
    """L2 residual of the graph constraint actually solved."""

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
        steps = self.config.simulation.n_steps if n_steps is None else n_steps
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
            retention = (
                compute_retention_loss(cells, state, self.backend)
                if self.config.use_exact_backend
                else None
            )
            admissible = (
                retention <= cfg_res.retention_epsilon
                if retention is not None
                else None
            )
            result = {
                "cells": cells,
                "leakage": None,
                "drift": None,
                "retention_loss": retention,
                "su2_equivariant": True,
                "admissible": admissible,
                "n_total": 1,
                "n_admissible": int(admissible) if admissible is not None else None,
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
                "admissible": None,
                "n_total": None,
                "n_admissible": None,
            }

        leakage_text = (
            f"{result['leakage']:.6e}"
            if result["leakage"] is not None
            else "not computed"
        )
        log.info(
            "Π_res: %d cells of size %d, leakage proxy=%s",
            len(result["cells"]),
            k,
            leakage_text,
        )
        if result.get("admissible") is False:
            log.warning(
                "Pi_res returned an inadmissible decomposition: retention loss %s "
                "exceeds the configured bound %.6g",
                result.get("retention_loss"),
                cfg_res.retention_epsilon,
            )
        return PiResResult(
            cells=result["cells"],
            leakage=result["leakage"],
            drift=result.get("drift"),
            retention_loss=result.get("retention_loss"),
            su2_equivariant=result.get("su2_equivariant"),
            admissible=result.get("admissible"),
            n_total=result.get("n_total"),
            n_admissible=result.get("n_admissible"),
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
            leakage_tie_tolerance=cfg_res.leakage_tie_tolerance,
            leakage_probe_states=cfg_res.leakage_probe_states,
            drift_probe_states=cfg_res.drift_probe_states,
            probe_seed=cfg_res.probe_seed,
            su2_tolerance=cfg_res.su2_tolerance,
            su2_samples=cfg_res.su2_samples,
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

        mi_matrix = torch.zeros((n_cells, n_cells), dtype=torch.float64)
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                mi = self.backend.mutual_information(
                    state, cells[i], cells[j]
                )
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        max_mi = mi_matrix.max().item()
        if max_mi <= 0:
            raise ValueError(
                "Π_loc cannot construct distances because all mutual information is zero."
            )
        if cfg_loc.I_0 is None:
            if cfg_loc.I_0_multiplier <= 1.0:
                raise ValueError(
                    "I_0_multiplier must be greater than 1 to preserve positive lengths."
                )
            I_0 = cfg_loc.I_0_multiplier * max_mi
        else:
            I_0 = cfg_loc.I_0
            if I_0 <= max_mi:
                raise ValueError(
                    f"I_0 ({I_0:.6g}) must be larger than max(I_ij) ({max_mi:.6g}); "
                    "otherwise the default distance kernel collapses distinct nodes."
                )

        dist_matrix = torch.zeros((n_cells, n_cells), dtype=torch.float64)
        raw_weight_matrix = torch.zeros((n_cells, n_cells), dtype=torch.float64)
        for i in range(n_cells):
            for j in range(n_cells):
                if i == j:
                    continue
                ratio = mi_matrix[i, j].item() / I_0
                ratio = max(ratio, cfg_loc.mi_epsilon)
                dist_matrix[i, j] = cfg_loc.distance_kernel(ratio)
                raw_weight_matrix[i, j] = cfg_loc.weight_kernel(
                    mi_matrix[i, j].item()
                )

        graph_dist, edges, threshold, gap_ratio, separable = self._graph_geodesics(
            dist_matrix,
            raw_weight_matrix,
            n_cells,
            method=cfg_loc.connectivity_method,
            k_nearest=cfg_loc.k_nearest,
            minimum_gap_ratio=cfg_loc.minimum_gap_ratio,
        )

        weight_matrix = torch.zeros_like(raw_weight_matrix)
        for i, j in edges:
            weight_matrix[i, j] = raw_weight_matrix[i, j]
            weight_matrix[j, i] = raw_weight_matrix[j, i]

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
            connectivity_method=cfg_loc.connectivity_method,
            connectivity_threshold=threshold,
            connectivity_gap_ratio=gap_ratio,
            connectivity_separable=separable,
        )

    def _graph_geodesics(
        self,
        dist_matrix: torch.Tensor,
        weight_matrix: torch.Tensor,
        n: int,
        *,
        method: str,
        k_nearest: int,
        minimum_gap_ratio: float,
    ) -> tuple[
        torch.Tensor,
        list[tuple[int, int]],
        float | None,
        float | None,
        bool | None,
    ]:
        """Infer blind connectivity and compute shortest-path distances."""
        adj = torch.full(
            (n, n),
            float("inf"),
            dtype=dist_matrix.dtype,
            device=dist_matrix.device,
        )
        edges: set[tuple[int, int]] = set()
        threshold: float | None = None
        gap_ratio: float | None = None
        separable: bool | None = None

        if method == "adaptive_gap":
            if minimum_gap_ratio <= 1.0:
                raise ValueError("minimum_gap_ratio must be greater than 1")
            pair_weights = torch.tensor(
                [
                    weight_matrix[i, j].item()
                    for i in range(n)
                    for j in range(i + 1, n)
                    if weight_matrix[i, j] > 0
                ],
                dtype=torch.float64,
            )
            if pair_weights.numel() >= 2:
                pair_weights = torch.sort(pair_weights, descending=True).values
                ratios = pair_weights[:-1] / pair_weights[1:]
                gap_index = int(torch.argmax(ratios).item())
                gap_ratio = float(ratios[gap_index].item())
                separable = gap_ratio >= minimum_gap_ratio
                if separable:
                    threshold = float(
                        torch.sqrt(
                            pair_weights[gap_index] * pair_weights[gap_index + 1]
                        ).item()
                    )
                    for i in range(n):
                        for j in range(i + 1, n):
                            if weight_matrix[i, j] >= threshold:
                                edges.add((i, j))
            else:
                separable = False
        elif method == "knn_mst":
            if k_nearest < 1:
                raise ValueError("k_nearest must be positive for knn_mst connectivity")
            for i in range(n):
                dists_i = dist_matrix[i].clone()
                dists_i[i] = float("inf")
                _, neighbors = torch.topk(
                    dists_i, min(k_nearest, n - 1), largest=False
                )
                for j_idx in neighbors:
                    j = j_idx.item()
                    edges.add((min(i, j), max(i, j)))
        else:
            raise ValueError(f"Unknown connectivity method: {method!r}")

        mst_edges = self._minimum_spanning_tree(dist_matrix, n)
        for i, j in mst_edges:
            edges.add((min(i, j), max(i, j)))

        for i, j in edges:
            adj[i, j] = dist_matrix[i, j]
            adj[j, i] = dist_matrix[j, i]

        for i in range(n):
            adj[i, i] = 0.0

        graph_dist = self._floyd_warshall(adj, n)
        return graph_dist, sorted(edges), threshold, gap_ratio, separable

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
        4. Fit regularized local SPD metrics in the selected embedding.
        5. (Future phases: intrinsic complex, Regge curvature, closure.)
        """
        cfg_geom = self.config.pi_geom
        d_G = pi_loc.distance_matrix
        w = pi_loc.weight_matrix
        n = d_G.shape[0]

        D_S, spectral_peak_time, spectral_curve = self._spectral_dimension(w, n)

        best_D, best_objective, best_stress, best_coords = 1, float("inf"), float("inf"), None
        stress_by_dimension: dict[int, float] = {}
        objective_by_dimension: dict[int, float] = {}
        candidate_dimension_max = min(cfg_geom.D_max, max(1, n - 1))
        for D in range(1, candidate_dimension_max + 1):
            coords = self._classical_mds(d_G, D)
            stress = self._mds_stress(d_G, coords)
            F_D = stress + cfg_geom.lambda_dim * (D - D_S) ** 2
            stress_by_dimension[D] = stress
            objective_by_dimension[D] = F_D
            if F_D < best_objective:
                best_D = D
                best_objective = F_D
                best_stress = stress
                best_coords = coords

        assert best_coords is not None

        ordered_objectives = sorted(objective_by_dimension.values())
        selection_margin = (
            ordered_objectives[1] - ordered_objectives[0]
            if len(ordered_objectives) > 1
            else float("inf")
        )
        if best_D > cfg_geom.max_geometric_dimension:
            embedding_status = "non_geometric_dimension"
        elif best_stress > cfg_geom.max_geometric_stress:
            embedding_status = "poor_fit"
        else:
            embedding_status = "geometric_candidate"

        metric_fits = reconstruct_local_metrics(
            best_coords,
            d_G,
            pi_loc.edges,
            regularization=cfg_geom.lambda_spd,
            eigenvalue_floor=cfg_geom.metric_eigenvalue_floor,
        )
        metric_diagnostics = [
            {
                "condition_number": fit.condition_number,
                "relative_residual": fit.relative_residual,
                "design_rank": fit.design_rank,
                "n_constraints": fit.n_constraints,
                "underdetermined": fit.underdetermined,
            }
            for fit in metric_fits
        ]

        simplices = None
        deficit_angles = None
        complex_status = "not_available_for_selected_dimension"
        if best_D == 2 and embedding_status == "geometric_candidate":
            try:
                simplices = build_delaunay_proxy(best_coords)
                deficit_angles, boundary_vertices = vertex_deficits_2d(
                    best_coords, simplices
                )
                complex_status = "embedding_delaunay_proxy"
                for vertex in boundary_vertices:
                    metric_diagnostics[vertex]["is_proxy_boundary"] = 1
            except ValueError as exc:
                complex_status = f"proxy_failed:{exc}"

        log.info(
            "Π_geom: D_S=%.2f, D*=%d, stress=%.4f, objective=%.4f",
            D_S, best_D, best_stress, best_objective,
        )
        return PiGeomResult(
            D_star=best_D,
            D_spectral=D_S,
            coords=best_coords,
            stress=best_stress,
            objective=best_objective,
            stress_by_dimension=stress_by_dimension,
            objective_by_dimension=objective_by_dimension,
            spectral_peak_time=spectral_peak_time,
            spectral_dimension_curve=spectral_curve,
            embedding_status=embedding_status,
            selection_margin=selection_margin,
            h_ab=[fit.metric for fit in metric_fits],
            metric_diagnostics=metric_diagnostics,
            simplices=simplices,
            deficit_angles=deficit_angles,
            complex_status=complex_status,
        )

    def _spectral_dimension(
        self, weight_matrix: torch.Tensor, n: int
    ) -> tuple[float, float | None, list[float]]:
        """Return the peak finite-graph heat-kernel dimension and its curve.

        A finite connected graph has spectral dimension zero in both the UV
        and IR limits. The peak of ``-2 d log Tr(exp(-tL)) / d log(t)`` is
        therefore reported as a finite-size diagnostic, not a continuum limit.
        """
        weights = weight_matrix.to(dtype=torch.float64)
        L = torch.diag(weights.sum(dim=1)) - weights
        evals = torch.linalg.eigvalsh(L).real
        zero_tolerance = max(float(evals.max().item()) * 1e-10, 1e-14)
        evals[evals < zero_tolerance] = 0.0
        positive = evals[evals > 0]
        if len(positive) < 2:
            return 0.0, None, []

        t_min = 0.1 / positive.max()
        t_max = 10.0 / positive.min()
        t_values = torch.logspace(
            torch.log10(t_min), torch.log10(t_max), steps=100, dtype=torch.float64
        )
        traces = torch.stack([torch.sum(torch.exp(-t * evals)) for t in t_values])
        log_t = torch.log(t_values)
        log_tr = torch.log(traces.clamp(min=1e-30))

        d_log_tr = (log_tr[1:] - log_tr[:-1]) / (log_t[1:] - log_t[:-1])
        dimension_curve = -2 * d_log_tr
        peak_index = int(torch.argmax(dimension_curve).item())
        D_S = float(dimension_curve[peak_index].item())
        peak_time = float(
            torch.sqrt(t_values[peak_index] * t_values[peak_index + 1]).item()
        )
        return D_S, peak_time, dimension_curve.tolist()

    @staticmethod
    def _classical_mds(d_matrix: torch.Tensor, D: int) -> torch.Tensor:
        """Classical MDS embedding into ℝ^D (§4.4.4 step 2)."""
        n = d_matrix.shape[0]
        D2 = d_matrix**2
        J = torch.eye(n, dtype=d_matrix.dtype, device=d_matrix.device)
        J -= torch.ones((n, n), dtype=d_matrix.dtype, device=d_matrix.device) / n
        B = -0.5 * J @ D2 @ J

        evals, evecs = torch.linalg.eigh(B)
        idx = evals.argsort(descending=True)
        evals = evals[idx]
        evecs = evecs[:, idx]
        evecs = Simulator._canonicalize_degenerate_eigenvector_blocks(
            evals, evecs
        )

        D_eff = min(D, n - 1)
        top_evals = evals[:D_eff].clamp(min=0)
        coords = (evecs[:, :D_eff] @ torch.diag(torch.sqrt(top_evals))).real
        return Simulator._canonicalize_embedding(coords)

    @staticmethod
    def _canonicalize_degenerate_eigenvector_blocks(
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        """Fix label-ordered bases before a requested dimension cuts a block."""
        canonical = eigenvectors.clone()
        if eigenvalues.numel() == 0:
            return canonical
        scale = max(1.0, float(torch.max(torch.abs(eigenvalues)).item()))
        tolerance = 256.0 * torch.finfo(eigenvalues.dtype).eps * scale
        start = 0
        while start < eigenvalues.numel():
            end = start + 1
            while end < eigenvalues.numel() and abs(
                float(eigenvalues[end] - eigenvalues[start])
            ) <= tolerance:
                end += 1
            if end - start > 1 and eigenvalues[start] > tolerance:
                canonical[:, start:end] = Simulator._canonicalize_embedding(
                    canonical[:, start:end]
                )
            start = end
        return canonical

    @staticmethod
    def _canonicalize_embedding(coords: torch.Tensor) -> torch.Tensor:
        """Fix the arbitrary orthogonal MDS frame using label-ordered anchors.

        Classical MDS coordinates are defined only up to an orthogonal transform.
        A label-stable maximum-volume anchor scan is equivariant under that transform,
        and the SVD polar factor is exactly orthogonal to floating-point precision.
        Applying it yields deterministic coordinates without changing pairwise distance.
        """
        centered = coords - coords.mean(dim=0, keepdim=True)
        dimension = centered.shape[1]
        if dimension == 0:
            return centered

        singular_values = torch.linalg.svdvals(centered)
        rank_tolerance = 1e-8 * float(singular_values[0].item())
        represented_rank = int(
            torch.count_nonzero(singular_values > rank_tolerance).item()
        )
        if represented_rank < dimension:
            return Simulator._canonicalize_rank_deficient_embedding(
                centered, represented_rank
            )

        anchor_matrix = Simulator._select_embedding_anchors(centered, dimension)
        polar_left, _, polar_right_h = torch.linalg.svd(
            anchor_matrix.T, full_matrices=False
        )
        orientation = polar_left @ polar_right_h
        return centered @ orientation

    @staticmethod
    def _select_embedding_anchors(
        centered: torch.Tensor,
        dimension: int,
    ) -> torch.Tensor:
        """Select a stable maximum-volume row basis with label-ordered ties."""
        basis: list[torch.Tensor] = []
        anchors: list[torch.Tensor] = []
        remaining = list(range(centered.shape[0]))
        epsilon = torch.finfo(centered.dtype).eps
        scale = max(1.0, float(torch.linalg.matrix_norm(centered).item()))

        for _ in range(dimension):
            residuals: list[tuple[int, torch.Tensor, float]] = []
            for index in remaining:
                residual = centered[index].clone()
                for vector in basis:
                    residual -= torch.dot(residual, vector) * vector
                residuals.append(
                    (index, residual, float(torch.linalg.vector_norm(residual).item()))
                )
            maximum = max(norm for _, _, norm in residuals)
            if maximum <= epsilon * scale:
                raise RuntimeError("embedding rank and anchor selection disagree")
            tie_tolerance = 256.0 * epsilon * max(1.0, maximum)
            index, residual, _ = next(
                item for item in residuals if item[2] >= maximum - tie_tolerance
            )
            anchors.append(centered[index])
            basis.append(residual / torch.linalg.vector_norm(residual))
            remaining.remove(index)

        return torch.stack(anchors)

    @staticmethod
    def _canonicalize_rank_deficient_embedding(
        coords: torch.Tensor,
        represented_rank: int,
    ) -> torch.Tensor:
        """Fix the full represented frame and pad unrepresented axes with zeros."""
        dimension = coords.shape[1]
        if represented_rank == 0:
            return torch.zeros_like(coords)

        left, singular_values, _ = torch.linalg.svd(coords, full_matrices=False)
        represented = left[:, :represented_rank] * singular_values[:represented_rank]
        canonical = Simulator._canonicalize_embedding(represented)
        if represented_rank == dimension:
            return canonical
        padding = torch.zeros(
            (coords.shape[0], dimension - represented_rank),
            dtype=coords.dtype,
            device=coords.device,
        )
        return torch.cat([canonical, padding], dim=1)

    @staticmethod
    def _mds_stress(d_target: torch.Tensor, coords: torch.Tensor) -> float:
        """Kruskal stress-1: normalized residual of distance reproduction."""
        n = coords.shape[0]
        d_embed = torch.cdist(coords, coords)
        mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=coords.device), diagonal=1
        )
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
        reference_state: object | None = None,
    ) -> PiTimeResult:
        """
        Execute the Π_time projection stage (§4.4.5).

        1. Compute the configured placeholder or experimental source term δρ.
        2. Build the weighted graph Laplacian.
        3. Solve (Δ_w + μ²I)Φ = δρ for the clock-rate potential.
           This is numerically an elliptic graph constraint. Its proposed
           relationship to an ADM constraint remains unvalidated.
        4. Compute proper time dτ = β_0 · exp(Φ) · dS_act.
        """
        cfg_time = self.config.pi_time
        cells = pi_res.cells
        w = pi_loc.weight_matrix

        delta_rho_raw = self._compute_source_term(
            state, cells, cfg_time, reference_state=reference_state
        )
        phi, delta_rho, source_background, residual = self._solve_clock_constraint(
            w,
            delta_rho_raw,
            mu=cfg_time.mu,
            zero_mode_policy=cfg_time.zero_mode_policy,
            normalize_potential=cfg_time.normalize_potential,
        )

        dt = self.config.simulation.dt
        dtau = cfg_time.beta_0 * torch.exp(phi) * dt

        log.info(
            "Π_time: Φ range [%.4f, %.4f], μ=%.2f",
            phi.min().item(), phi.max().item(), cfg_time.mu,
        )
        return PiTimeResult(
            phi=phi,
            delta_rho=delta_rho,
            delta_rho_raw=delta_rho_raw,
            dtau=dtau,
            source_model=cfg_time.source_model,
            source_status=(
                "placeholder"
                if cfg_time.source_model == "von_neumann_placeholder"
                else "candidate"
            ),
            source_background=source_background,
            constraint_residual=residual,
        )

    @staticmethod
    def _solve_clock_constraint(
        weight_matrix: torch.Tensor,
        source: torch.Tensor,
        *,
        mu: float,
        zero_mode_policy: str,
        normalize_potential: bool,
        compatibility_tolerance: float = 1e-10,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Solve a finite-graph Poisson constraint without arbitrary node pinning."""
        w = weight_matrix.to(dtype=torch.float64)
        effective_source = source.to(dtype=torch.float64).clone()
        laplacian = torch.diag(w.sum(dim=1)) - w
        source_background = 0.0

        remove_constant_mode = mu == 0.0 or normalize_potential
        if remove_constant_mode:
            source_background = effective_source.mean().item()
            if zero_mode_policy == "subtract_mean":
                effective_source -= source_background
            elif zero_mode_policy == "require_zero_sum":
                if abs(effective_source.sum().item()) > compatibility_tolerance:
                    raise ValueError(
                        "The unscreened graph Poisson source must sum to zero. "
                        "Use zero_mode_policy='subtract_mean' to add a neutralizing background."
                    )
                source_background = 0.0
            else:
                raise ValueError(f"Unknown zero_mode_policy: {zero_mode_policy!r}")

        n = laplacian.shape[0]
        if mu == 0.0:
            ones = torch.ones((n, 1), dtype=torch.float64)
            augmented = torch.cat(
                [
                    torch.cat([laplacian, ones], dim=1),
                    torch.cat([ones.T, torch.zeros((1, 1), dtype=torch.float64)], dim=1),
                ],
                dim=0,
            )
            rhs = torch.cat([effective_source, torch.zeros(1, dtype=torch.float64)])
            phi = torch.linalg.solve(augmented, rhs)[:n]
            operator = laplacian
        else:
            operator = laplacian + mu**2 * torch.eye(n, dtype=torch.float64)
            phi = torch.linalg.solve(operator, effective_source)

        if normalize_potential:
            phi -= phi.mean()

        residual = torch.linalg.vector_norm(operator @ phi - effective_source).item()
        return phi, effective_source, source_background, residual

    @staticmethod
    def gravitational_redshift(
        phi_emitter: float | torch.Tensor,
        phi_observer: float | torch.Tensor,
    ) -> float:
        """Return z for stationary clocks with dτ = exp(Φ) dt.

        ``1 + z = ν_emit / ν_obs = exp(Φ_observer - Φ_emitter)``.
        """
        delta = torch.as_tensor(
            phi_observer, dtype=torch.float64
        ) - torch.as_tensor(phi_emitter, dtype=torch.float64)
        return float(torch.exp(delta).item() - 1.0)

    def _compute_source_term(
        self,
        state: object,
        cells: list[list[int]],
        cfg_time,
        *,
        reference_state: object | None = None,
    ) -> torch.Tensor:
        """Compute an explicitly classified placeholder or candidate source.

        The default remains von Neumann entropy (always ≥ 0), explicitly
        classified as a pipeline placeholder. Negative relative-entropy and
        reduced-state modular-energy candidates are available only when the caller
        supplies an explicit reference state. The exact backend also exposes a
        microscopic KMS energy-density candidate using the declared Hamiltonian
        decomposition. Availability is not validation of a physical law: raw
        relative entropy fails linear response, reduced-state modular energy is
        blind in the symmetric KMS-chain control, and the microscopic candidate has
        only passed finite-system diagnostics. None is an established local
        stress-energy component.
        """
        n_cells = len(cells)
        delta_rho = torch.zeros(n_cells, dtype=torch.float64)
        if cfg_time.source_model == "negative_kms_energy_density_candidate":
            if reference_state is None:
                raise ValueError(
                    f"Source model {cfg_time.source_model!r} requires a reference_state"
                )
            if not isinstance(self.backend, ExactBackend):
                raise NotImplementedError(
                    "The KMS energy-density candidate requires the exact backend "
                    "and an explicit microscopic Hamiltonian."
                )
            flattened_cells = [site for cell in cells for site in cell]
            expected_sites = list(range(self.config.substrate.n_qubits))
            if any(not cell for cell in cells) or sorted(flattened_cells) != expected_sites:
                raise ValueError(
                    "cells must form a nonempty disjoint partition of all microscopic sites"
                )
            beta_kms = (
                cfg_time.beta_kms
                if cfg_time.beta_kms is not None
                else self.config.substrate.beta
            )
            reference_state = self._validate_kms_reference_state(
                reference_state,
                beta_kms=beta_kms,
            )
            local_energy = self.backend.build_local_energy_operators()
            state_delta = state - reference_state
            for i, cell in enumerate(cells):
                cell_energy = torch.zeros_like(local_energy[0])
                for site in cell:
                    cell_energy += local_energy[site]
                delta_rho[i] = -beta_kms * torch.trace(
                    state_delta @ cell_energy
                ).real.item()
            return cfg_time.source_scale * delta_rho

        for i, cell in enumerate(cells):
            rho_i = self.backend.reduced_state(state, cell)
            if cfg_time.source_model == "von_neumann_placeholder":
                delta_rho[i] = self.backend.entropy(rho_i)
                continue

            if reference_state is None:
                raise ValueError(
                    f"Source model {cfg_time.source_model!r} requires a reference_state"
                )
            sigma_i = self.backend.reduced_state(reference_state, cell)
            if cfg_time.source_model == "negative_relative_entropy_candidate":
                delta_rho[i] = -self.backend.araki_relative_entropy(rho_i, sigma_i)
            elif cfg_time.source_model == "negative_modular_energy_candidate":
                delta_rho[i] = -modular_energy_delta(rho_i, sigma_i)
            else:
                raise ValueError(f"Unknown source model: {cfg_time.source_model!r}")
        return cfg_time.source_scale * delta_rho

    def _validate_kms_reference_state(
        self,
        reference_state: object,
        *,
        beta_kms: float,
    ) -> torch.Tensor:
        """Require the backend Gibbs state at ``beta_kms`` as KMS reference."""
        if not isinstance(reference_state, torch.Tensor):
            raise ValueError(
                "reference_state must be a tensor containing the Gibbs/KMS state"
            )

        expected = finite_gibbs_state(
            self.backend.build_hamiltonian(),
            beta_kms,
        )
        reference = reference_state.to(
            dtype=expected.dtype,
            device=expected.device,
        )
        if reference.shape != expected.shape or not torch.isfinite(reference).all():
            raise ValueError(
                "reference_state must be a finite density matrix with the same "
                "shape as the backend Gibbs/KMS state"
            )

        trace_distance = 0.5 * torch.linalg.matrix_norm(
            reference - expected,
            ord="nuc",
        ).item()
        if trace_distance > KMS_REFERENCE_TRACE_DISTANCE_TOLERANCE:
            raise ValueError(
                "reference_state must be the Gibbs/KMS state of the backend "
                f"Hamiltonian at beta_kms={beta_kms!r}; trace distance "
                f"{trace_distance:.6e} exceeds "
                "KMS_REFERENCE_TRACE_DISTANCE_TOLERANCE="
                f"{KMS_REFERENCE_TRACE_DISTANCE_TOLERANCE:.1e}"
            )
        return reference

    # ── full pipeline ────────────────────────────────────────────────

    def run(
        self,
        evolve_steps: int | None = None,
        *,
        reference_state: object | None = None,
    ) -> SimulatorResult:
        """
        Execute the complete projection pipeline Π = Π_time ∘ Π_geom ∘ Π_loc ∘ Π_res.

        This is the primary entry point for end-to-end simulation. Candidate
        source models that depend on a reference receive it through the explicit
        ``reference_state`` keyword argument.
        """
        state = self.prepare()
        if evolve_steps is not None and evolve_steps > 0:
            state = self.evolve(state, n_steps=evolve_steps)

        pi_res = self.run_pi_res(state)
        pi_loc = self.run_pi_loc(state, pi_res)
        pi_geom = self.run_pi_geom(pi_loc)

        pi_time = self.run_pi_time(
            state,
            pi_res,
            pi_loc,
            pi_geom,
            reference_state=reference_state,
        )

        return SimulatorResult(
            config=self.config,
            state=state,
            pi_res=pi_res,
            pi_loc=pi_loc,
            pi_geom=pi_geom,
            pi_time=pi_time,
        )
