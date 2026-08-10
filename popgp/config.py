# Copyright (c) 2026 WHACT. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""
Canonical parameter configuration for the POPGP simulator.

Every free parameter in the framework is declared here with an explicit
classification (UNIVERSAL_CONSTANT or TUNABLE_HYPERPARAMETER) per the
parameter discipline rules in docs/framework.md §4.6.3.

Architecture Decision 5: No magic numbers in function bodies.  All code
reads from a SimulatorConfig instance passed at construction time.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto


class ParamKind(Enum):
    """Classification required by docs/framework.md §4.6.3."""

    UNIVERSAL_CONSTANT = auto()
    TUNABLE_HYPERPARAMETER = auto()
    STRUCTURAL_CHOICE = auto()
    EMPIRICAL_SCALE_SETTING = auto()


# ── helper: default monotone kernels ────────────────────────────────────


def default_distance_kernel(u: float) -> float:
    """f(u) = max{0, −log u}  (framework §4.4.3 default)."""
    if u <= 0:
        return float("inf")
    return max(0.0, -math.log(u))


def default_weight_kernel(I_ij: float) -> float:
    """κ(I) = I  (identity, simplest monotone-increasing with κ(0)=0)."""
    return max(0.0, I_ij)


# ── Substrate configuration ─────────────────────────────────────────────


@dataclass
class SubstrateConfig:
    """Parameters defining the substrate algebra and state (§4.1–§4.3)."""

    # -- structural choices --
    n_qubits: int = 8
    """Number of qubits in the substrate.  [STRUCTURAL_CHOICE]"""

    hamiltonian: str = "heisenberg"
    """Hamiltonian family ('heisenberg', 'ising', 'custom').  [STRUCTURAL_CHOICE]"""

    coupling_J: float = 1.0
    """Nearest-neighbor coupling strength.  [UNIVERSAL_CONSTANT]
    Fixed to 1.0 (sets the energy scale).  Changing this is equivalent
    to rescaling the phase-order parameter s."""

    boundary: str = "open"
    """Boundary conditions ('open', 'periodic').  [STRUCTURAL_CHOICE]"""

    topology: str = "chain"
    """Lattice topology ('chain', 'grid', 'custom').  [STRUCTURAL_CHOICE]"""

    grid_width: int | None = None
    """Width for 'grid' topology.  Derived from n_qubits if None."""

    grid_height: int | None = None
    """Height for 'grid' topology.  Derived from n_qubits if None."""

    # -- state preparation --
    beta: float = 1.0
    """Inverse temperature for the thermal (Boltzmann) state.
    [TUNABLE_HYPERPARAMETER]  Lower β → more mixed; higher β → closer
    to ground state.  Controls correlation length."""

    seed: int = 42
    """Random seed for reproducibility.  [STRUCTURAL_CHOICE]"""


# ── Π_res configuration ────────────────────────────────────────────────


@dataclass
class PiResConfig:
    """Parameters for resolution-limited coarse-graining (§4.4.2, §4.4.2a)."""

    cell_dim: int = 2
    """Qubit count per cell (k).  Cell Hilbert space has dimension 2^k.
    [TUNABLE_HYPERPARAMETER]  Must divide n_qubits evenly."""

    cell_dim_max: int | None = None
    """Maximum cell Hilbert-space dimension d_max.  If None, uses 2^cell_dim.
    [TUNABLE_HYPERPARAMETER]"""

    # -- phase-order window for leakage integral --
    phase_window_center: float = 0.0
    """Center s_0 of the phase-order window for L_leak integration.
    [TUNABLE_HYPERPARAMETER]"""

    phase_window_width: float = 2.0
    """Width Δs of the phase-order window (w(s)=1/Δs inside, 0 outside).
    [TUNABLE_HYPERPARAMETER]  Should be small enough that the cell net
    is approximately stable across the window."""

    phase_window_samples: int = 20
    """Number of quadrature points in the phase-order window.
    [TUNABLE_HYPERPARAMETER]"""

    norm_type: str = "frobenius"
    """Superoperator norm for L_leak ('frobenius', 'diamond').
    [STRUCTURAL_CHOICE]  Frobenius is computationally simpler;
    diamond is operationally strongest (§4.4.2a)."""

    # -- retention constraint --
    retention_epsilon: float = 0.1
    """Maximum allowed relative-entropy loss D(ω ‖ ω∘E).
    [TUNABLE_HYPERPARAMETER]  Prevents trivial 'erase everything'
    minimizer (§4.4.2a)."""

    # -- drift tie-breaker --
    drift_delta: float = 0.1
    """Phase-order step δ for the drift functional L_drift.
    [TUNABLE_HYPERPARAMETER]  Should satisfy δ ≪ phase_window_width."""

    probe_seed: int = 42
    """Seed for common random channel-norm probes. [NUMERICAL_CONTROL]
    Every candidate partition must be evaluated with the same probes."""

    leakage_probe_states: int = 8
    """Number of common Haar probes used for leakage estimation.
    [TUNABLE_HYPERPARAMETER]"""

    drift_probe_states: int = 4
    """Number of common Haar probes used for drift estimation.
    [TUNABLE_HYPERPARAMETER]"""

    leakage_tie_tolerance: float = 1e-8
    """Absolute tolerance for leakage ties. [NUMERICAL_CONTROL]"""

    su2_tolerance: float = 1e-6
    """Numerical tolerance for SU(2)-equivariance checks. [NUMERICAL_CONTROL]"""

    su2_samples: int = 5
    """Number of sampled SU(2) transformations used for verification.
    [NUMERICAL_CONTROL]"""


# ── Π_loc configuration ────────────────────────────────────────────────


@dataclass
class PiLocConfig:
    """Parameters for locality from correlations (§4.4.3)."""

    I_0: float | None = None
    """Reference mutual-information scale. If None, it is derived as
    ``I_0_multiplier * max(I_ij)`` so distinct nodes retain positive length.
    Explicit values must be strictly larger than the observed maximum MI.
    [TUNABLE_HYPERPARAMETER]"""

    I_0_multiplier: float = math.e
    """Multiplier used to derive I_0 when it is not supplied.
    [EMPIRICAL_SCALE_SETTING] The default gives the strongest pair unit length."""

    distance_kernel: Callable[[float], float] = field(
        default_factory=lambda: default_distance_kernel
    )
    """Monotone decreasing map f: I/I_0 → distance.
    [STRUCTURAL_CHOICE]  Default: f(u) = max{0, −log u} (§4.4.3)."""

    weight_kernel: Callable[[float], float] = field(
        default_factory=lambda: default_weight_kernel
    )
    """Monotone increasing map κ: I_ij → edge weight, κ(0)=0.
    [STRUCTURAL_CHOICE]  Default: κ(I) = I (identity)."""

    connectivity_method: str = "adaptive_gap"
    """Blind connectivity rule. ``adaptive_gap`` separates correlation
    scales at the largest multiplicative MI gap and then adds an MST only
    when needed for connectivity. ``knn_mst`` retains the fixed-k baseline.
    [STRUCTURAL_CHOICE]"""

    minimum_gap_ratio: float = 1.5
    """Minimum multiplicative separation required to call an adaptive MI
    gap identifiable. Below this value only the connectivity MST is returned
    and the result is marked non-separable. [TUNABLE_HYPERPARAMETER]"""

    k_nearest: int = 3
    """k for the optional ``knn_mst`` connectivity baseline.
    [TUNABLE_HYPERPARAMETER]"""

    mi_epsilon: float = 1e-12
    """Numerical floor for MI values (avoids log(0)).
    [NUMERICAL_ARTIFACT]  Not a physics parameter."""


# ── Π_geom configuration ───────────────────────────────────────────────


@dataclass
class PiGeomConfig:
    """Parameters for emergent geometry (§4.4.4)."""

    lambda_dim: float = 0.01
    """Penalty weight for dimension selection: topological inertia.
    [TUNABLE_HYPERPARAMETER]  Evaluated in the degeneracy-breaking
    limit (λ_dim → 0⁺); it breaks ties between equally low-stress
    embeddings to prevent high-frequency quantum noise from causing
    macroscopic dimensionality jitter, not to force D=3 (§4.4.4 step 1).
    The small default implements the stated λ_dim → 0⁺ limit."""

    D_max: int = 6
    """Maximum candidate embedding dimension.
    [STRUCTURAL_CHOICE]  D* > 3 flags a non-geometric phase (F1)."""

    max_geometric_dimension: int = 3
    """Largest selected dimension labeled a geometric candidate.
    [STRUCTURAL_CHOICE] Higher values are reported as non-geometric."""

    max_geometric_stress: float = 0.25
    """Maximum MDS stress labeled an acceptable finite geometric candidate.
    [TUNABLE_HYPERPARAMETER] This is a diagnostic threshold, not a theorem."""

    embedding_method: str = "mds"
    """Embedding algorithm ('mds', 'smacof').
    [STRUCTURAL_CHOICE]  Classical MDS is exact for Euclidean metrics;
    SMACOF iteratively minimizes stress."""

    lambda_spd: float = 0.01
    """SPD regularization strength for local metric reconstruction.
    [TUNABLE_HYPERPARAMETER]  Controls how strongly h_ab is pulled
    toward the inverse local covariance (§4.4.4 step 3)."""

    metric_eigenvalue_floor: float = 1e-8
    """Numerical SPD floor for local embedding-metric fits.
    [NUMERICAL_CONTROL]"""


# ── Π_time configuration ───────────────────────────────────────────────


@dataclass
class PiTimeConfig:
    """Parameters for emergent time (§4.4.5)."""

    mu: float = 0.0
    """Screening mass for the clock-rate Laplacian.
    [TUNABLE_HYPERPARAMETER] μ=0 gives an unscreened graph Poisson
    constraint; no continuum 1/r law is implied without a convergence test.
    μ>0 is an optional controlled IR regulator."""

    zero_mode_policy: str = "subtract_mean"
    """Compatibility policy for μ=0 graph Poisson solves.
    ``subtract_mean`` removes the constant source mode; ``require_zero_sum``
    rejects incompatible sources. [STRUCTURAL_CHOICE]"""

    normalize_potential: bool = True
    """Subtract the mean potential after solving so only clock-rate contrasts
    are reported on a finite graph. [STRUCTURAL_CHOICE]"""

    source_model: str = "von_neumann_placeholder"
    """Clock source model. The default is an explicitly non-physical
    placeholder retained for pipeline diagnostics. Explicit experimental
    alternatives are ``negative_relative_entropy_candidate`` and
    ``negative_modular_energy_candidate`` (both reduced-state contrasts), and
    ``negative_kms_energy_density_candidate`` (an exact-backend microscopic
    Hamiltonian decomposition). All candidates require a reference state passed
    to ``run_pi_time`` or ``Simulator.run``. The KMS-labelled candidate additionally
    validates that reference against the backend Gibbs state at ``beta_kms``.
    [STRUCTURAL_CHOICE]"""

    source_scale: float = 1.0
    """Multiplicative scale applied to the configured clock source.
    [EMPIRICAL_SCALE_SETTING]"""

    beta_0: float = 1.0
    """Clock-rate coupling constant: dτ = β_0·exp(Φ)·dS_act.
    [EMPIRICAL_SCALE_SETTING]  Sets the unit relationship between
    phase-order increments and proper time."""

    phase_window_time: float = 1.0
    """Width of the temporal-averaging window for the Araki contrast
    source term δρ.
    [TUNABLE_HYPERPARAMETER]  Ensures stability against high-frequency
    phase noise (§4.4.5 step 2)."""

    beta_kms: float | None = None
    """Inverse temperature for the KMS vacuum baseline.  If None,
    derived from the modular flow of (A, ω).
    [STRUCTURAL_CHOICE]  §4.4.5 step 3."""


# ── Simulation control ──────────────────────────────────────────────────


@dataclass
class SimulationConfig:
    """Runtime parameters for the simulation loop."""

    dt: float = 0.1
    """Phase-order step size for time evolution.
    [TUNABLE_HYPERPARAMETER]"""

    n_steps: int = 100
    """Total number of phase-order steps.
    [TUNABLE_HYPERPARAMETER]"""


# ── Backend / scale control ─────────────────────────────────────────────


@dataclass
class BackendConfig:
    """
    Controls the exact-vs-GPU dispatch (Architecture Decision 2).

    The simulator uses exact methods (full density matrix, partial trace,
    exact diagonalization) when n_qubits ≤ exact_threshold, and switches
    to the mean-field GPU engine otherwise.
    """

    exact_threshold: int = 12
    """Maximum n_qubits for the exact backend.  Above this, the mean-field
    GPU engine is used.  [STRUCTURAL_CHOICE]  12 qubits ≈ 4096×4096
    density matrix, which fits in ~256 MB at complex128."""

    device: str = "cpu"
    """PyTorch device ('cpu', 'cuda', 'cuda:0', ...).
    [STRUCTURAL_CHOICE]  The exact backend ignores this (always CPU).
    The GPU backend requires 'cuda'."""

    precision: str = "double"
    """Floating-point precision ('float', 'double').
    [STRUCTURAL_CHOICE]  'double' is required for accurate entropy
    and eigenvalue computations at toy scale."""

    initial_perturbation: float = 0.15
    """Symmetry-breaking amplitude for the experimental mean-field product
    state. [TUNABLE_HYPERPARAMETER]"""


# ── Top-level configuration ─────────────────────────────────────────────


@dataclass
class SimulatorConfig:
    """
    Root configuration object for the POPGP strict simulator.

    Architecture Decision 5: every free parameter lives here.  Code never
    contains magic numbers — it reads from config fields that carry explicit
    classification labels in their docstrings.

    Usage::

        cfg = SimulatorConfig()                       # all defaults
        cfg = SimulatorConfig(substrate=SubstrateConfig(n_qubits=10))
        cfg = SimulatorConfig.for_chain(n=8, beta=1.0)
        cfg = SimulatorConfig.for_grid(width=3, height=3, beta=2.0)
    """

    substrate: SubstrateConfig = field(default_factory=SubstrateConfig)
    pi_res: PiResConfig = field(default_factory=PiResConfig)
    pi_loc: PiLocConfig = field(default_factory=PiLocConfig)
    pi_geom: PiGeomConfig = field(default_factory=PiGeomConfig)
    pi_time: PiTimeConfig = field(default_factory=PiTimeConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)

    # ── convenience constructors ────────────────────────────────────────

    @classmethod
    def for_chain(cls, n: int = 8, beta: float = 1.0, **kw) -> SimulatorConfig:
        """Pre-configured for a 1D Heisenberg chain."""
        return cls(
            substrate=SubstrateConfig(
                n_qubits=n, topology="chain", beta=beta, **kw
            ),
            pi_res=PiResConfig(cell_dim=2),
        )

    @classmethod
    def for_grid(
        cls, width: int = 3, height: int = 3, beta: float = 2.0, **kw
    ) -> SimulatorConfig:
        """Pre-configured for a 2D Heisenberg grid."""
        return cls(
            substrate=SubstrateConfig(
                n_qubits=width * height,
                topology="grid",
                grid_width=width,
                grid_height=height,
                beta=beta,
                **kw,
            ),
            pi_res=PiResConfig(cell_dim=1),
        )

    @property
    def use_exact_backend(self) -> bool:
        """Whether the exact (full density matrix) backend should be used."""
        return self.substrate.n_qubits <= self.backend.exact_threshold

    @property
    def n_cells(self) -> int:
        """Number of emergent cells given current substrate and Π_res config."""
        return self.substrate.n_qubits // self.pi_res.cell_dim
