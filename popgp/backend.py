# Copyright (c) 2026 WHACT. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Backend abstraction for the POPGP simulator.

Architecture Decision 2: Python-first for correctness, GPU for scale.

The simulator operates through a uniform Backend interface.  Two concrete
implementations are provided:

* **ExactBackend** — full density-matrix algebra via torch/quimb.
  Used when n_qubits ≤ config.backend.exact_threshold (default 12).
  All quantum operations are exact: diagonalization, partial trace,
  von Neumann entropy, mutual information.

* **GPUBackend** — mean-field per-cell amplitudes via the CUDA engine.
  Used when n_qubits > exact_threshold.  Phase-flow is a Trotterized
  Heisenberg mean-field interaction on the GPU.  Product-state mean field
  cannot represent entanglement, so mutual information is unsupported and
  the end-to-end projection pipeline stops explicitly at Π_loc.

Architecture Decision 1: both backends expose the same interface so the
Simulator class needs no conditional logic.

Implements substrate definition per docs/framework.md §4.1–§4.3.
"""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from popgp.config import SimulatorConfig

from popgp.engine import Engine
from popgp.information import quantum_relative_entropy, von_neumann_entropy

log = logging.getLogger(__name__)

# ── Pauli matrices (shared) ─────────────────────────────────────────────

_SX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128) / 2
_SY = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128) / 2
_SZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128) / 2
_I2 = torch.eye(2, dtype=torch.complex128)


# ── Abstract interface ──────────────────────────────────────────────────


class Backend(abc.ABC):
    """
    Uniform interface consumed by the Simulator.

    Every method operates on opaque *state* objects whose concrete type
    depends on the backend (density matrix for Exact, amplitude arrays
    for GPU).

    Implements the substrate triple (A, ω, σ_s) from §4.1–§4.3.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config

    # ── substrate ────────────────────────────────────────────────────

    @abc.abstractmethod
    def build_hamiltonian(self) -> torch.Tensor:
        """Construct the substrate Hamiltonian H (§4.3)."""

    @abc.abstractmethod
    def prepare_state(self) -> object:
        """Prepare the initial substrate state ω (§4.4.1 GNS)."""

    @abc.abstractmethod
    def build_edges(self) -> list[tuple[int, int]]:
        """Return the interaction graph edges for the lattice topology."""

    # ── phase-flow σ_s ───────────────────────────────────────────────

    @abc.abstractmethod
    def evolve(self, state: object, dt: float) -> object:
        """Advance the state by one phase-order step of size dt (§4.3).
        Implements σ_s: A ↦ e^{isH} A e^{−isH} (or its Trotterized
        mean-field approximation on GPU)."""

    # ── coarse-graining / partial trace ──────────────────────────────

    @abc.abstractmethod
    def reduced_state(
        self, state: object, cell_indices: list[int]
    ) -> torch.Tensor:
        """Return the reduced density matrix for the given qubit indices.
        Implements the conditional expectation E_i (§4.4.2)."""

    # ── information-theoretic quantities ─────────────────────────────

    @abc.abstractmethod
    def entropy(self, rho: torch.Tensor) -> float:
        """Von Neumann entropy S(ρ) = −Tr(ρ log ρ)."""

    @abc.abstractmethod
    def mutual_information(
        self, state: object, cell_i: list[int], cell_j: list[int]
    ) -> float:
        """Mutual information I(i:j) = S_Araki(ω_{i∪j} ‖ ω_i ⊗ ω_j) (§4.4.3).

        For finite-dimensional toy models this reduces to
        S(ρ_i) + S(ρ_j) − S(ρ_{ij})."""

    @abc.abstractmethod
    def araki_relative_entropy(
        self, rho: torch.Tensor, sigma: torch.Tensor
    ) -> float:
        """S_Araki(ρ ‖ σ) = Tr[ρ(log ρ − log σ)] (§6.1)."""


# ── Exact Backend ───────────────────────────────────────────────────────


class ExactBackend(Backend):
    """
    Full density-matrix backend for N ≤ 12 qubits.

    All operations are exact: the state is a 2^N × 2^N density matrix,
    partial traces use einsum, entropies use eigendecomposition.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        super().__init__(config)
        N = config.substrate.n_qubits
        self._N = N
        self._dim = 2**N
        self._H: torch.Tensor | None = None
        self._evals: torch.Tensor | None = None
        self._evecs: torch.Tensor | None = None
        log.info(
            "ExactBackend: N=%d, dim=%d, device=cpu", N, self._dim
        )

    def _ensure_diagonalized(self) -> None:
        if self._evals is None:
            H = self.build_hamiltonian()
            self._H = H
            self._evals, self._evecs = torch.linalg.eigh(H)

    # ── substrate ────────────────────────────────────────────────────

    def build_hamiltonian(self) -> torch.Tensor:
        """Heisenberg Hamiltonian H = Σ_{<ij>} S_i · S_j  (§4.3)."""
        if self._H is not None:
            return self._H

        N = self._N
        dim = self._dim

        def _site_op(op: torch.Tensor, site: int) -> torch.Tensor:
            parts = [_I2] * N
            parts[site] = op
            result = parts[0]
            for p in parts[1:]:
                result = torch.kron(result, p)
            return result

        if self.config.substrate.hamiltonian not in {"heisenberg", "ising"}:
            raise NotImplementedError(
                f"Hamiltonian family {self.config.substrate.hamiltonian!r} is not implemented."
            )

        H = torch.zeros((dim, dim), dtype=torch.complex128)
        J = self.config.substrate.coupling_J
        for i, j in self.build_edges():
            interaction = _site_op(_SZ, i) @ _site_op(_SZ, j)
            if self.config.substrate.hamiltonian == "heisenberg":
                interaction += (
                    _site_op(_SX, i) @ _site_op(_SX, j)
                    + _site_op(_SY, i) @ _site_op(_SY, j)
                )
            H += J * interaction
        self._H = H
        return H

    def prepare_state(self) -> torch.Tensor:
        """Thermal state ρ = exp(−βH)/Z  (§4.4.1 GNS representation)."""
        self._ensure_diagonalized()
        assert self._evals is not None and self._evecs is not None
        beta = self.config.substrate.beta
        weights = torch.exp(-beta * (self._evals - self._evals[0]))
        weights /= weights.sum()
        rho = (
            self._evecs
            @ torch.diag(weights.to(dtype=torch.complex128))
            @ self._evecs.conj().T
        )
        return rho

    def build_edges(self) -> list[tuple[int, int]]:
        cfg = self.config.substrate
        N = self._N
        edges: list[tuple[int, int]] = []

        if cfg.topology == "chain":
            for i in range(N - 1):
                edges.append((i, i + 1))
            if cfg.boundary == "periodic" and N > 2:
                edges.append((N - 1, 0))

        elif cfg.topology == "grid":
            W = cfg.grid_width or int(np.sqrt(N))
            H = cfg.grid_height or (N // W)
            for y in range(H):
                for x in range(W):
                    k = y * W + x
                    if x + 1 < W:
                        edges.append((k, y * W + x + 1))
                    if y + 1 < H:
                        edges.append((k, (y + 1) * W + x))
                    if cfg.boundary == "periodic":
                        if x == W - 1 and W > 2:
                            edges.append((k, y * W))
                        if y == H - 1 and H > 2:
                            edges.append((k, x))
        else:
            raise ValueError(f"Unknown topology: {cfg.topology}")

        return edges

    # ── phase-flow ───────────────────────────────────────────────────

    def evolve(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """σ_s(ρ) = U(dt) ρ U(dt)†  where U = exp(−iHdt)  (§4.3)."""
        self._ensure_diagonalized()
        assert self._evals is not None and self._evecs is not None
        phases = torch.exp(-1j * self._evals * dt).to(dtype=torch.complex128)
        U = self._evecs @ torch.diag(phases) @ self._evecs.conj().T
        rho_new = U @ state @ U.conj().T
        rho_new /= torch.trace(rho_new)
        return rho_new

    # ── partial trace ────────────────────────────────────────────────

    def reduced_state(
        self, state: torch.Tensor, cell_indices: list[int]
    ) -> torch.Tensor:
        """Partial trace over complement of cell_indices (§4.4.2)."""
        N = self._N
        keep = list(cell_indices)
        trace_out = [i for i in range(N) if i not in keep]
        n_keep = len(keep)
        n_trace = len(trace_out)

        shape = [2] * (2 * N)
        rho_t = state.reshape(shape)

        perm = list(keep) + list(trace_out)
        perm += [x + N for x in keep] + [x + N for x in trace_out]
        rho_t = rho_t.permute(perm)

        dim_keep = 2**n_keep
        dim_trace = 2**n_trace
        rho_t = rho_t.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
        rho_red = torch.einsum("ijkj->ik", rho_t)
        return rho_red

    # ── information theory ───────────────────────────────────────────

    def entropy(self, rho: torch.Tensor) -> float:
        """S(ρ) = −Tr(ρ log ρ)  via eigendecomposition."""
        return von_neumann_entropy(rho)

    def mutual_information(
        self, state: torch.Tensor, cell_i: list[int], cell_j: list[int]
    ) -> float:
        """Toy-model reduction of I(i:j) = S_Araki(ω_{i∪j} ‖ ω_i ⊗ ω_j) (§4.4.3)."""
        rho_i = self.reduced_state(state, cell_i)
        rho_j = self.reduced_state(state, cell_j)
        rho_ij = self.reduced_state(state, cell_i + cell_j)
        return max(0.0, self.entropy(rho_i) + self.entropy(rho_j) - self.entropy(rho_ij))

    def araki_relative_entropy(
        self, rho: torch.Tensor, sigma: torch.Tensor
    ) -> float:
        """S(ρ ‖ σ) = Tr[ρ(log ρ − log σ)]  (§6.1).

        Uses eigendecomposition with an exact support-containment check.
        """
        return quantum_relative_entropy(rho, sigma)


# ── GPU Backend ─────────────────────────────────────────────────────────


class GPUBackend(Backend):
    """
    Mean-field GPU backend for N > 12 cells.

    Each cell is a single qubit (α, β) evolved by the CUDA phase-flow
    kernel. Product states contain no entanglement and cannot supply quantum
    mutual information; this is an experimental dynamics path rather than a
    scalable implementation of the full projection.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        super().__init__(config)
        self._N = config.substrate.n_qubits
        self._device = torch.device(config.backend.device)
        self._engine = None
        log.info(
            "GPUBackend: N=%d, device=%s, precision=%s",
            self._N, config.backend.device, config.backend.precision,
        )

    def _get_engine(self):
        if self._device.type != "cuda":
            raise RuntimeError(
                "GPUBackend evolution requires backend.device='cuda' and a built "
                "native CUDA engine."
            )
        if self.config.substrate.hamiltonian != "heisenberg":
            raise NotImplementedError(
                "The native mean-field backend currently implements only "
                "Heisenberg dynamics."
            )
        if self._engine is None:
            self._engine = Engine(precision=self.config.backend.precision)
        return self._engine

    def build_hamiltonian(self) -> torch.Tensor:
        raise NotImplementedError(
            "GPUBackend does not construct an explicit Hamiltonian. "
            "The Heisenberg interaction is applied pairwise by the CUDA kernel."
        )

    def prepare_state(self) -> dict:
        """Initialize per-cell qubit amplitudes near |0⟩ with perturbation."""
        torch.manual_seed(self.config.substrate.seed)
        N = self._N
        perturbation = self.config.backend.initial_perturbation
        complex_dtype = (
            torch.complex64 if self.config.backend.precision == "float" else torch.complex128
        )
        real_dtype = torch.float32 if self.config.backend.precision == "float" else torch.float64
        alphas = torch.ones(N, dtype=complex_dtype, device=self._device)
        betas = torch.zeros(N, dtype=complex_dtype, device=self._device)
        betas += perturbation * torch.randn(
            N, dtype=real_dtype, device=self._device
        )
        norms = torch.sqrt(
            alphas.real**2 + alphas.imag**2 + betas.real**2 + betas.imag**2
        )
        alphas /= norms
        betas /= norms
        return {"alphas": alphas, "betas": betas}

    def build_edges(self) -> list[tuple[int, int]]:
        cfg = self.config.substrate
        N = self._N
        edges: list[tuple[int, int]] = []
        if cfg.topology == "chain":
            for i in range(N - 1):
                edges.append((i, i + 1))
            if cfg.boundary == "periodic" and N > 2:
                edges.append((N - 1, 0))
        elif cfg.topology == "grid":
            W = cfg.grid_width or int(np.sqrt(N))
            H = cfg.grid_height or (N // W)
            for y in range(H):
                for x in range(W):
                    k = y * W + x
                    if x + 1 < W:
                        edges.append((k, y * W + x + 1))
                    if y + 1 < H:
                        edges.append((k, (y + 1) * W + x))
                    if cfg.boundary == "periodic":
                        if x == W - 1 and W > 2:
                            edges.append((k, y * W))
                        if y == H - 1 and H > 2:
                            edges.append((k, x))
        else:
            raise ValueError(f"Unknown topology: {cfg.topology}")
        return edges

    def evolve(self, state: dict, dt: float) -> dict:
        """One edge-colored mean-field Heisenberg step via CUDA.

        Edges in a batch never share a node. Launching all lattice edges in a
        single kernel would race on cell amplitudes and make evolution
        nondeterministic.
        """
        engine = self._get_engine()
        real_dtype = torch.float32 if self.config.backend.precision == "float" else torch.float64
        for edges in self._edge_color_batches(self.build_edges()):
            src = torch.tensor([e[0] for e in edges], dtype=torch.int32, device=self._device)
            dst = torch.tensor([e[1] for e in edges], dtype=torch.int32, device=self._device)
            weights = torch.ones(len(edges), dtype=real_dtype, device=self._device)
            engine.step(state["alphas"], state["betas"], src, dst, weights, dt)
        return state

    @staticmethod
    def _edge_color_batches(
        edges: list[tuple[int, int]],
    ) -> list[list[tuple[int, int]]]:
        """Greedily partition edges into node-disjoint launch batches."""
        batches: list[list[tuple[int, int]]] = []
        occupied: list[set[int]] = []
        for edge in edges:
            nodes = set(edge)
            for batch, used in zip(batches, occupied, strict=True):
                if nodes.isdisjoint(used):
                    batch.append(edge)
                    used.update(nodes)
                    break
            else:
                batches.append([edge])
                occupied.append(set(nodes))
        return batches

    def reduced_state(
        self, state: dict, cell_indices: list[int]
    ) -> torch.Tensor:
        """Single-qubit density matrix from amplitudes."""
        if len(cell_indices) != 1:
            raise NotImplementedError(
                "GPUBackend only supports single-qubit reduced states."
            )
        idx = cell_indices[0]
        a = state["alphas"][idx].cpu()
        b = state["betas"][idx].cpu()
        psi = torch.tensor([a, b], dtype=torch.complex128)
        return torch.outer(psi, psi.conj())

    def entropy(self, rho: torch.Tensor) -> float:
        return von_neumann_entropy(rho)

    def mutual_information(
        self, state: dict, cell_i: list[int], cell_j: list[int]
    ) -> float:
        raise NotImplementedError(
            "GPUBackend evolves a product-state mean field and cannot compute mutual information. "
            "A connected-correlation or tensor-network backend is required for Π_loc."
        )

    def araki_relative_entropy(
        self, rho: torch.Tensor, sigma: torch.Tensor
    ) -> float:
        return quantum_relative_entropy(rho, sigma)


# ── Factory ─────────────────────────────────────────────────────────────


def create_backend(config: SimulatorConfig) -> Backend:
    """
    Auto-select the appropriate backend based on system size.

    Architecture Decision 2: transparent dispatch.  The caller never
    needs to know which backend is in use.
    """
    if config.use_exact_backend:
        return ExactBackend(config)
    return GPUBackend(config)
