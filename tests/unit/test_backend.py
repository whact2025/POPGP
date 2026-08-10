import pytest
import torch

from popgp.backend import Backend, ExactBackend, GPUBackend
from popgp.coarse_grain import compute_leakage, optimize_cells
from popgp.config import BackendConfig, SimulatorConfig, SubstrateConfig


def test_ising_and_heisenberg_are_distinct_hamiltonians() -> None:
    heisenberg = ExactBackend(
        SimulatorConfig(substrate=SubstrateConfig(n_qubits=2, hamiltonian="heisenberg"))
    ).build_hamiltonian()
    ising = ExactBackend(
        SimulatorConfig(substrate=SubstrateConfig(n_qubits=2, hamiltonian="ising"))
    ).build_hamiltonian()

    assert not torch.allclose(heisenberg, ising)
    assert torch.linalg.eigvalsh(heisenberg) == pytest.approx(
        torch.tensor([-0.75, 0.25, 0.25, 0.25], dtype=torch.float64)
    )
    assert torch.linalg.eigvalsh(ising) == pytest.approx(
        torch.tensor([-0.25, -0.25, 0.25, 0.25], dtype=torch.float64)
    )


def test_unknown_hamiltonian_fails_explicitly() -> None:
    backend = ExactBackend(
        SimulatorConfig(substrate=SubstrateConfig(n_qubits=2, hamiltonian="custom"))
    )
    with pytest.raises(NotImplementedError, match="not implemented"):
        backend.build_hamiltonian()


def test_local_energy_decomposition_sums_to_hamiltonian() -> None:
    backend = ExactBackend(
        SimulatorConfig(substrate=SubstrateConfig(n_qubits=4, hamiltonian="heisenberg"))
    )

    local_energy = backend.build_local_energy_operators()

    assert len(local_energy) == 4
    assert torch.allclose(torch.stack(local_energy).sum(dim=0), backend.build_hamiltonian())


@pytest.mark.parametrize("family", ["heisenberg", "ising"])
def test_cell_generators_and_intercell_terms_reconstruct_hamiltonian(
    family: str,
) -> None:
    backend = ExactBackend(
        SimulatorConfig(
            substrate=SubstrateConfig(n_qubits=4, hamiltonian=family)
        )
    )
    identity = torch.eye(4, dtype=torch.complex128)
    left = backend.build_cell_hamiltonian([0, 1])
    right = backend.build_cell_hamiltonian([2, 3])
    interaction_terms = dict(backend.build_interaction_terms())

    reconstructed = (
        torch.kron(left, identity)
        + torch.kron(identity, right)
        + interaction_terms[(1, 2)]
    )

    assert torch.allclose(
        reconstructed,
        backend.build_hamiltonian(),
        atol=1e-14,
        rtol=1e-14,
    )


@pytest.mark.parametrize("family", ["heisenberg", "ising"])
def test_two_site_local_energy_operators_split_interaction_equally(
    family: str,
) -> None:
    backend = ExactBackend(
        SimulatorConfig(
            substrate=SubstrateConfig(n_qubits=2, hamiltonian=family)
        )
    )
    interaction = backend.build_interaction_terms()[0][1]

    local_energy = backend.build_local_energy_operators()

    assert torch.allclose(local_energy[0], 0.5 * interaction)
    assert torch.allclose(local_energy[1], 0.5 * interaction)


def test_site_operator_validates_shape_and_site() -> None:
    backend = ExactBackend(SimulatorConfig(substrate=SubstrateConfig(n_qubits=2)))

    with pytest.raises(ValueError, match="2x2"):
        backend.site_operator(torch.eye(3), 0)
    with pytest.raises(IndexError, match="site"):
        backend.site_operator(torch.eye(2), 2)


def test_cell_hamiltonian_validates_indices_and_backend_capability() -> None:
    backend = ExactBackend(SimulatorConfig(substrate=SubstrateConfig(n_qubits=3)))

    with pytest.raises(ValueError, match="nonempty"):
        backend.build_cell_hamiltonian([])
    with pytest.raises(ValueError, match="unique"):
        backend.build_cell_hamiltonian([0, 0])
    with pytest.raises(IndexError, match="must lie"):
        backend.build_cell_hamiltonian([-1])
    with pytest.raises(IndexError, match="must lie"):
        backend.build_cell_hamiltonian([3])
    with pytest.raises(NotImplementedError, match="does not provide"):
        Backend.build_cell_hamiltonian(backend, [0])


def test_backend_reuses_interaction_and_cell_hamiltonian_caches() -> None:
    backend = ExactBackend(SimulatorConfig(substrate=SubstrateConfig(n_qubits=4)))

    interactions = backend.build_interaction_terms()
    cell_hamiltonian = backend.build_cell_hamiltonian([0, 1])

    assert backend.build_interaction_terms() is interactions
    assert backend.build_cell_hamiltonian([0, 1]) is cell_hamiltonian


def test_compute_leakage_requires_backend_edges_and_coupling() -> None:
    backend = ExactBackend(SimulatorConfig(substrate=SubstrateConfig(n_qubits=2)))
    state = backend.prepare_state()

    with pytest.raises(ValueError, match="edges must match"):
        compute_leakage(
            [[0, 1]],
            state,
            backend,
            [],
            backend.config.substrate.coupling_J,
            0.1,
            1,
        )
    with pytest.raises(ValueError, match="coupling_J must match"):
        compute_leakage(
            [[0, 1]],
            state,
            backend,
            backend.build_edges(),
            backend.config.substrate.coupling_J + 1.0,
            0.1,
            1,
        )


def test_optimize_cells_constructs_each_distinct_cell_generator_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ExactBackend(SimulatorConfig(substrate=SubstrateConfig(n_qubits=6)))
    state = backend.prepare_state()
    original = backend.build_cell_hamiltonian
    calls: list[tuple[int, ...]] = []

    def counted(cell: list[int]) -> torch.Tensor:
        calls.append(tuple(cell))
        return original(cell)

    monkeypatch.setattr(backend, "build_cell_hamiltonian", counted)
    optimize_cells(
        state,
        backend,
        n_qubits=6,
        cell_dim=2,
        edges=backend.build_edges(),
        coupling_J=backend.config.substrate.coupling_J,
        phase_window_width=0.1,
        phase_window_samples=1,
        retention_epsilon=float("inf"),
        su2_tolerance=float("inf"),
        su2_samples=1,
        leakage_probe_states=1,
        drift_probe_states=1,
    )

    assert len(calls) == len(set(calls))
    assert len(calls) <= 15


def test_gpu_backend_does_not_mislabel_product_correlations_as_mi() -> None:
    config = SimulatorConfig(
        substrate=SubstrateConfig(n_qubits=13),
        backend=BackendConfig(exact_threshold=12, device="cpu"),
    )
    backend = GPUBackend(config)
    state = {
        "alphas": torch.ones(13, dtype=torch.complex128),
        "betas": torch.zeros(13, dtype=torch.complex128),
    }

    with pytest.raises(NotImplementedError, match="cannot compute mutual information"):
        backend.mutual_information(state, [0], [12])


def test_edge_color_batches_are_node_disjoint() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]
    batches = GPUBackend._edge_color_batches(edges)

    assert sorted(edge for batch in batches for edge in batch) == sorted(edges)
    for batch in batches:
        flattened = [node for edge in batch for node in edge]
        assert len(flattened) == len(set(flattened))


def test_gpu_evolution_requires_cuda_device() -> None:
    config = SimulatorConfig(
        substrate=SubstrateConfig(n_qubits=13),
        backend=BackendConfig(exact_threshold=12, device="cpu"),
    )
    backend = GPUBackend(config)
    state = backend.prepare_state()

    with pytest.raises(RuntimeError, match="requires backend.device='cuda'"):
        backend.evolve(state, dt=0.1)


def test_gpu_periodic_chain_includes_closing_edge() -> None:
    config = SimulatorConfig(
        substrate=SubstrateConfig(n_qubits=13, boundary="periodic"),
        backend=BackendConfig(exact_threshold=12, device="cpu"),
    )

    assert (12, 0) in GPUBackend(config).build_edges()
