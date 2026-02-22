"""
Cell selection optimization for the Π_res projection stage (§4.4.2, §4.4.2a).

The framework (v1.0) defines the selected cell net E* as the stable
fixed-point of a causal gradient flow driven by L_leak (§4.4.2a).
For toy models (N ≤ 12 qubits), we locate this fixed point via
exhaustive combinatorial search — a computational shortcut that
finds the same attractor without running the continuous flow.

Implements:
- Enumeration of all equal-size partitions of N qubits into cells of size k
- Leakage functional L_leak — commutator norm integral (§4.4.2a primary)
- Drift functional L_drift — tie-breaker (§4.4.2a secondary)
- SU(2) equivariance check (E4, §4.4.2a constraint 2)
- Retention bound check (§4.4.2a constraint 3)
- Lexicographic minimization: minimize L_leak, then L_drift
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Generator

import torch

from popgp.backend import Backend, _I2, _SX, _SY, _SZ

log = logging.getLogger(__name__)


# ── Partition Enumeration ────────────────────────────────────────────────


def enumerate_partitions(
    n_items: int, group_size: int
) -> Generator[list[list[int]], None, None]:
    """Yield every partition of {0..n_items-1} into groups of ``group_size``.

    Groups are ordered by their smallest element; elements within each
    group are sorted.  This canonical ordering avoids duplicate partitions.

    Complexity: (n-1)!! / ((k-1)!)^(n/k-1) partitions for n items in
    groups of k.  For n=8, k=2: 105 partitions.

    Implements partition enumeration for the admissible set 𝔈_adm (§4.4.2a).
    """
    if n_items % group_size != 0:
        raise ValueError(
            f"n_items ({n_items}) must be divisible by group_size ({group_size})"
        )
    items = list(range(n_items))
    yield from _partition_recursive(items, group_size)


def _partition_recursive(
    items: list[int], k: int
) -> Generator[list[list[int]], None, None]:
    if len(items) == 0:
        yield []
        return
    first = items[0]
    rest = items[1:]
    for partners in combinations(rest, k - 1):
        group = [first] + list(partners)
        remaining = [x for x in rest if x not in partners]
        for sub in _partition_recursive(remaining, k):
            yield [group] + sub


def count_partitions(n: int, k: int) -> int:
    """Number of equal-size partitions without enumeration (for diagnostics)."""
    if n % k != 0 or n == 0:
        return 0
    from math import comb, factorial
    m = n // k
    numerator = factorial(n)
    denominator = (factorial(k) ** m) * factorial(m)
    return numerator // denominator


# ── Local Hamiltonian Construction ───────────────────────────────────────


def _build_local_hamiltonian(
    cell: list[int],
    edges: list[tuple[int, int]],
    coupling_J: float,
) -> torch.Tensor:
    """Construct the Hamiltonian restricted to qubits within a single cell.

    Only interactions between qubits that are *both* in the cell are
    included.  This is the operator whose unitary generates the
    'trace-then-evolve' branch of the leakage commutator (§4.4.2a).
    """
    k = len(cell)
    d = 2 ** k
    H_local = torch.zeros((d, d), dtype=torch.complex128)
    local_map = {q: i for i, q in enumerate(cell)}

    for qi, qj in edges:
        if qi in local_map and qj in local_map:
            li, lj = local_map[qi], local_map[qj]

            def _op(op: torch.Tensor, site: int) -> torch.Tensor:
                parts = [_I2] * k
                parts[site] = op
                out = parts[0]
                for p in parts[1:]:
                    out = torch.kron(out, p)
                return out

            H_local += coupling_J * (
                _op(_SX, li) @ _op(_SX, lj)
                + _op(_SY, li) @ _op(_SY, lj)
                + _op(_SZ, li) @ _op(_SZ, lj)
            )
    return H_local


def _evolve_local(
    rho_cell: torch.Tensor,
    H_local: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Evolve a reduced density matrix under a local Hamiltonian."""
    evals, evecs = torch.linalg.eigh(H_local)
    phases = torch.exp(-1j * evals * dt).to(dtype=torch.complex128)
    U = evecs @ torch.diag(phases) @ evecs.conj().T
    return U @ rho_cell @ U.conj().T


# ── Leakage Functional ──────────────────────────────────────────────────


def compute_leakage(
    cells: list[list[int]],
    state: torch.Tensor,
    backend: Backend,
    edges: list[tuple[int, int]],
    coupling_J: float,
    phase_window_width: float,
    phase_window_samples: int,
    n_probe_states: int = 8,
    unitaries: list[torch.Tensor] | None = None,
) -> float:
    r"""Compute L_leak(E) = ∫ ds w(s) · Σ_i ‖E_i∘σ_s − σ_s∘E_i‖²_HS (§4.4.2a).

    The framework defines ‖·‖ as a superoperator (channel) norm, not a
    state-dependent quantity.  We approximate the Hilbert-Schmidt channel
    norm by averaging the Frobenius norm of the commutator evaluated on
    random Haar-distributed pure probe states:

        ‖Δ‖²_HS ≈ (d+1) · E_ψ[ ‖Δ(|ψ⟩⟨ψ|)‖²_F ]

    Parameters
    ----------
    unitaries : list[Tensor], optional
        Pre-computed unitaries [U(s_0), U(s_0 + dt), U(s_0 + 2dt), ...].
        If provided, skips recomputing them from the backend.  Use
        :func:`precompute_unitaries` to build this list once and reuse
        across all partition evaluations.
    """
    dt_sample = phase_window_width / phase_window_samples
    d_full = state.shape[0]

    cell_H = {
        i: _build_local_hamiltonian(cell, edges, coupling_J)
        for i, cell in enumerate(cells)
    }

    probe_vecs = _generate_probe_vectors(d_full, n_probe_states)

    if unitaries is None:
        unitaries = precompute_unitaries(backend, phase_window_samples, dt_sample)

    leakage = 0.0
    for step in range(phase_window_samples):
        U_s = unitaries[step]
        U_s_dt = unitaries[step + 1]

        for psi in probe_vecs:
            psi_s = U_s @ psi
            rho_s = torch.outer(psi_s, psi_s.conj())
            psi_s_dt = U_s_dt @ psi
            rho_s_dt = torch.outer(psi_s_dt, psi_s_dt.conj())

            for i, cell in enumerate(cells):
                evolve_then_trace = backend.reduced_state(rho_s_dt, cell)
                rho_cell = backend.reduced_state(rho_s, cell)
                trace_then_evolve = _evolve_local(rho_cell, cell_H[i], dt_sample)

                diff = evolve_then_trace - trace_then_evolve
                leakage += torch.sum(torch.abs(diff) ** 2).item()

    leakage /= n_probe_states
    leakage *= dt_sample / phase_window_width
    return leakage


def precompute_unitaries(
    backend: Backend, n_samples: int, dt_sample: float
) -> list[torch.Tensor]:
    """Pre-compute U(s) = exp(-iHs) for s = 0, dt, 2dt, ..., n_samples*dt.

    Returns n_samples + 1 unitary matrices (one extra for the final evolve step).
    Reuse across all partition evaluations for O(1) cost per partition.
    """
    backend._ensure_diagonalized()
    evals = backend._evals
    evecs = backend._evecs
    d = evals.shape[0]

    unitaries = []
    for step in range(n_samples + 1):
        t = step * dt_sample
        if t == 0.0:
            unitaries.append(torch.eye(d, dtype=torch.complex128))
        else:
            phases = torch.exp(-1j * evals * t).to(dtype=torch.complex128)
            U = evecs @ torch.diag(phases) @ evecs.conj().T
            unitaries.append(U)
    return unitaries


def _generate_probe_vectors(
    dim: int, n_probes: int
) -> list[torch.Tensor]:
    """Generate Haar-random state vectors for channel norm estimation."""
    vecs = []
    for _ in range(n_probes):
        real = torch.randn(dim, dtype=torch.float64)
        imag = torch.randn(dim, dtype=torch.float64)
        psi = torch.complex(real, imag)
        psi /= psi.norm()
        vecs.append(psi)
    return vecs


# ── Drift Functional ─────────────────────────────────────────────────────


def compute_drift(
    cells: list[list[int]],
    state: torch.Tensor,
    backend: Backend,
    phase_window_width: float,
    phase_window_samples: int,
    drift_delta: float,
    n_probe_states: int = 4,
    unitaries: list[torch.Tensor] | None = None,
) -> float:
    r"""Compute L_drift(E) = ∫ ds w(s) · Σ_i (1/δ²) · D(ρ_i(s+δ) ‖ ρ_i(s)) (§4.4.2a).

    Penalizes rapid change of local reduced information under phase-order
    advance.  Uses Araki relative entropy from the backend.  Averaged
    over probe states for state-independence (same rationale as L_leak).
    """
    dt_sample = phase_window_width / phase_window_samples
    inv_delta_sq = 1.0 / (drift_delta ** 2)
    d_full = state.shape[0]
    probe_vecs = _generate_probe_vectors(d_full, n_probe_states)

    if unitaries is None:
        unitaries = precompute_unitaries(backend, phase_window_samples, dt_sample)

    backend._ensure_diagonalized()
    evals, evecs = backend._evals, backend._evecs
    phases_delta = torch.exp(-1j * evals * drift_delta).to(dtype=torch.complex128)
    U_delta = evecs @ torch.diag(phases_delta) @ evecs.conj().T

    drift = 0.0
    for step in range(phase_window_samples):
        U_s = unitaries[step]
        for psi in probe_vecs:
            psi_s = U_s @ psi
            rho_s = torch.outer(psi_s, psi_s.conj())
            rho_advanced = U_delta @ rho_s @ U_delta.conj().T

            for cell in cells:
                rho_i_s = backend.reduced_state(rho_s, cell)
                rho_i_sd = backend.reduced_state(rho_advanced, cell)
                d_re = backend.araki_relative_entropy(rho_i_sd, rho_i_s)
                drift += inv_delta_sq * max(0.0, d_re)

    drift /= n_probe_states
    drift *= dt_sample / phase_window_width
    return drift


# ── Retention Loss ───────────────────────────────────────────────────────


def compute_retention_loss(
    cells: list[list[int]],
    state: torch.Tensor,
    backend: Backend,
) -> float:
    """Compute D(ω ‖ ω∘E) — total correlation lost by the decomposition (§4.4.2a constraint 3).

    For a cell decomposition into cells {C_i}, the product map E = ⊗_i E_i
    produces the product state ⊗_i ρ_i.  The relative entropy is:

        D(ρ ‖ ⊗_i ρ_i) = Σ_i S(ρ_i) − S(ρ)

    This is the multi-information (total correlation).
    """
    S_total = backend.entropy(state)
    S_cells = sum(
        backend.entropy(backend.reduced_state(state, cell))
        for cell in cells
    )
    return max(0.0, S_cells - S_total)


# ── SU(2) Equivariance Check ────────────────────────────────────────────


def _random_su2() -> torch.Tensor:
    """Sample a Haar-random SU(2) element."""
    a = torch.randn(2, dtype=torch.float64)
    b = torch.randn(2, dtype=torch.float64)
    z1 = torch.complex(a[0], a[1])
    z2 = torch.complex(b[0], b[1])
    norm = torch.sqrt(z1.abs() ** 2 + z2.abs() ** 2)
    z1 /= norm
    z2 /= norm
    U = torch.tensor(
        [[z1, -z2.conj()], [z2, z1.conj()]],
        dtype=torch.complex128,
    )
    return U


def _apply_su2_global(
    state: torch.Tensor, g: torch.Tensor, n_qubits: int
) -> torch.Tensor:
    """Apply α_g = g^{⊗N} to a density matrix: α_g(ρ) = U_g ρ U_g†."""
    U_global = g
    for _ in range(n_qubits - 1):
        U_global = torch.kron(U_global, g)
    return U_global @ state @ U_global.conj().T


def _apply_su2_cell(
    rho_cell: torch.Tensor, g: torch.Tensor, n_cell_qubits: int
) -> torch.Tensor:
    """Apply α_g^{cell} = g^{⊗k} to a cell density matrix."""
    U_cell = g
    for _ in range(n_cell_qubits - 1):
        U_cell = torch.kron(U_cell, g)
    return U_cell @ rho_cell @ U_cell.conj().T


def check_su2_equivariance(
    cells: list[list[int]],
    state: torch.Tensor,
    backend: Backend,
    n_qubits: int,
    n_samples: int = 10,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """Check E_i ∘ α_g = α_g ∘ E_i for random SU(2) elements (§4.4.2a constraint 2).

    For tensor-product SU(2) actions and partial-trace coarse-graining,
    this is automatically satisfied (the partial trace commutes with
    local unitaries on the kept subsystem — proven analytically).  The
    numerical check serves as verification of this mathematical fact.

    Returns ``(passes, max_violation)`` where ``max_violation`` is the
    largest Frobenius norm of the commutator across all cells and samples.
    """
    max_violation = 0.0
    k = len(cells[0])

    for _ in range(n_samples):
        g = _random_su2()
        alpha_g_state = _apply_su2_global(state, g, n_qubits)

        for cell in cells:
            trace_then_sym = _apply_su2_cell(
                backend.reduced_state(state, cell), g, k
            )
            sym_then_trace = backend.reduced_state(alpha_g_state, cell)
            diff = sym_then_trace - trace_then_sym
            violation = torch.sqrt(torch.sum(torch.abs(diff) ** 2)).item()
            max_violation = max(max_violation, violation)

    passes = max_violation < tolerance
    return passes, max_violation


# ── Cell Selection (Causal Flow Attractor via Exhaustive Search) ─────────


def optimize_cells(
    state: torch.Tensor,
    backend: Backend,
    n_qubits: int,
    cell_dim: int,
    edges: list[tuple[int, int]],
    coupling_J: float,
    phase_window_width: float = 2.0,
    phase_window_samples: int = 20,
    drift_delta: float = 0.1,
    retention_epsilon: float = 0.1,
    su2_tolerance: float = 1e-6,
    su2_samples: int = 5,
    leakage_tie_tolerance: float = 1e-8,
) -> dict:
    """Locate the causal gradient flow attractor via exhaustive search (§4.4.2a).

    The framework defines E* as the fixed point of a local causal flow
    (§4.4.2a).  For small toy systems this is equivalent to the global
    minimizer, which we find by enumeration:

    1. Enumerate all equal-size partitions of N qubits into cells of size k
    2. Filter by admissibility:
       a. Finite-capacity cells (automatic — all cells have dim 2^k)
       b. SU(2) equivariance
       c. Retention bound D(ω ‖ ω∘E) ≤ ε
    3. Primary: minimize L_leak over admissible set
    4. Secondary: among leakage-minimizers (within tolerance), minimize L_drift

    Returns a dict with keys: cells, leakage, drift, retention_loss,
    su2_equivariant, n_total, n_admissible, all_results.
    """
    n_cells = n_qubits // cell_dim
    n_total = count_partitions(n_qubits, cell_dim)
    log.info(
        "Optimizing cell selection: N=%d, k=%d → %d cells, %d partitions to evaluate",
        n_qubits, cell_dim, n_cells, n_total,
    )

    dt_sample = phase_window_width / phase_window_samples
    unitaries = precompute_unitaries(backend, phase_window_samples, dt_sample)
    log.info("Pre-computed %d unitary matrices for phase window.", len(unitaries))

    results: list[dict] = []
    n_admissible = 0

    for idx, cells in enumerate(enumerate_partitions(n_qubits, cell_dim)):
        su2_ok, su2_max_viol = check_su2_equivariance(
            cells, state, backend, n_qubits,
            n_samples=su2_samples, tolerance=su2_tolerance,
        )
        if not su2_ok:
            log.debug("Partition %d: SU(2) FAIL (max_viol=%.2e)", idx, su2_max_viol)
            continue

        retention = compute_retention_loss(cells, state, backend)
        if retention > retention_epsilon:
            log.debug(
                "Partition %d: retention FAIL (%.4f > %.4f)",
                idx, retention, retention_epsilon,
            )
            continue

        n_admissible += 1
        leakage = compute_leakage(
            cells, state, backend, edges, coupling_J,
            phase_window_width, phase_window_samples,
            unitaries=unitaries,
        )

        results.append({
            "cells": cells,
            "leakage": leakage,
            "retention_loss": retention,
            "su2_equivariant": True,
            "su2_max_violation": su2_max_viol,
        })

        if (idx + 1) % 20 == 0:
            log.info(
                "  ... evaluated %d / %d partitions (%d admissible)",
                idx + 1, n_total, n_admissible,
            )

    if not results:
        raise RuntimeError(
            f"No admissible partitions found for N={n_qubits}, k={cell_dim}. "
            f"All {n_total} partitions were filtered out by SU(2) equivariance "
            f"or retention bound constraints."
        )

    log.info(
        "Admissible partitions: %d / %d. Computing leakage ranking...",
        n_admissible, n_total,
    )

    results.sort(key=lambda r: r["leakage"])
    best_leakage = results[0]["leakage"]

    tied = [
        r for r in results
        if abs(r["leakage"] - best_leakage) < leakage_tie_tolerance
    ]

    if len(tied) > 1:
        log.info(
            "%d partitions tied at L_leak=%.6e. Computing drift tie-breaker...",
            len(tied), best_leakage,
        )
        for r in tied:
            r["drift"] = compute_drift(
                r["cells"], state, backend,
                phase_window_width, phase_window_samples, drift_delta,
            )
        tied.sort(key=lambda r: r["drift"])
    else:
        tied[0]["drift"] = None

    winner = tied[0]
    log.info(
        "Optimal partition: L_leak=%.6e, drift=%s, retention=%.4f, cells=%s",
        winner["leakage"],
        f"{winner['drift']:.6e}" if winner["drift"] is not None else "N/A",
        winner["retention_loss"],
        winner["cells"],
    )

    return {
        "cells": winner["cells"],
        "leakage": winner["leakage"],
        "drift": winner.get("drift"),
        "retention_loss": winner["retention_loss"],
        "su2_equivariant": True,
        "n_total": n_total,
        "n_admissible": n_admissible,
        "all_results": results,
    }
