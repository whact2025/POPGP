import pytest
import torch

from popgp.backend import ExactBackend, GPUBackend
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


def test_site_operator_validates_shape_and_site() -> None:
    backend = ExactBackend(SimulatorConfig(substrate=SubstrateConfig(n_qubits=2)))

    with pytest.raises(ValueError, match="2x2"):
        backend.site_operator(torch.eye(3), 0)
    with pytest.raises(IndexError, match="site"):
        backend.site_operator(torch.eye(2), 2)


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
