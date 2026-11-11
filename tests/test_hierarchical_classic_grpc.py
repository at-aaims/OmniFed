"""Outer-hop classic gRPC ranks and communicator package wiring."""

from __future__ import annotations

import inspect
import multiprocessing as mp
import socket
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.omnifed.communicator.base import AggregationOp
from src.omnifed.communicator.grpc import GrpcCommunicator
from src.omnifed.hierarchical.outer_grpc import outer_grpc_rank, outer_grpc_world_size

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestOuterGrpcRanks(unittest.TestCase):
    def test_two_facility_proto(self) -> None:
        clients = {1, 2}
        self.assertEqual(outer_grpc_world_size(clients), 3)
        self.assertEqual(outer_grpc_rank(0, clients), 0)
        self.assertEqual(outer_grpc_rank(1, clients), 1)
        self.assertEqual(outer_grpc_rank(2, clients), 2)

    def test_leader_order_is_sorted_slurm_rank(self) -> None:
        clients = {5, 2}
        self.assertEqual(outer_grpc_rank(2, clients), 1)
        self.assertEqual(outer_grpc_rank(5, clients), 2)

    def test_worker_is_not_on_grpc_world(self) -> None:
        with self.assertRaises(ValueError):
            outer_grpc_rank(3, {1, 2})


class TestClassicGrpcContractUnchanged(unittest.TestCase):
    def test_rank0_is_server_in_world_of_three(self) -> None:
        server = GrpcCommunicator(
            rank=0, world_size=3, master_addr="127.0.0.1", master_port=50051
        )
        leader = GrpcCommunicator(
            rank=1, world_size=3, master_addr="127.0.0.1", master_port=50051
        )
        self.assertTrue(server.is_server)
        self.assertFalse(leader.is_server)
        self.assertEqual(server.world_size, 3)

    def test_torch_mpi_lives_in_communicator_package(self) -> None:
        from src.omnifed.communicator.torch_mpi import TorchMPICommunicator
        from src.omnifed.hierarchical.communicator.torch_mpi import (
            TorchMPICommunicator as Reexport,
        )

        self.assertIs(TorchMPICommunicator, Reexport)

    def test_adapter_lives_in_communicator_package(self) -> None:
        from src.omnifed.communicator.torch_mpi_adapter import TorchMPIAdapter
        from src.omnifed.hierarchical.torch_mpi_adapter import (
            TorchMPIAdapter as Reexport,
        )

        self.assertIs(TorchMPIAdapter, Reexport)

    def test_slurm_runner_uses_classic_grpc_not_flora(self) -> None:
        from src.omnifed.hierarchical import slurm_runner

        src = inspect.getsource(slurm_runner)
        self.assertIn("GrpcCommunicator", src)
        self.assertNotIn("global_grpc", src)
        self.assertNotIn("GrpcLeaderCommunicator", src)
        self.assertNotIn("HybridGrpcComm", src)

    def test_slurm_sync_sends_sample_count_on_global_comm(self) -> None:
        from src.omnifed.hierarchical import slurm_sync

        src = inspect.getsource(slurm_sync)
        self.assertNotIn("GrpcLeaderCommunicator", src)
        self.assertIn("global_total_samples", src)

    def test_hierarchical_experiment_yaml_still_composes(self) -> None:
        conf_dir = str(REPO_ROOT / "conf")
        with initialize_config_dir(version_base=None, config_dir=conf_dir):
            cfg = compose(config_name="experiment/hierarchical_cifar10_resnet18_fedavg")
        self.assertEqual(str(cfg.topology._target_).rsplit(".", 1)[-1], "HierarchicalTopology")
        self.assertEqual(int(cfg.slurm.nodes), 7)
        self.assertEqual(int(cfg.topology.num_clients), 6)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _outer_grpc_sum_worker(
    rank: int, world: int, port: int, q: mp.Queue, barrier: mp.Barrier
) -> None:
    comm = GrpcCommunicator(
        rank=rank,
        world_size=world,
        master_addr="127.0.0.1",
        master_port=port,
    )
    comm.setup()
    out = comm.aggregate(
        torch.tensor([float(rank)], dtype=torch.float32),
        AggregationOp.SUM,
    )
    q.put((rank, float(out.item())))
    barrier.wait(timeout=30)
    comm.close()


class TestOuterGrpcThreeRankAggregate(unittest.TestCase):
    def test_server_and_two_leaders_all_submit(self) -> None:
        port = _free_port()
        ctx = mp.get_context("spawn")
        q: mp.Queue = ctx.Queue()
        barrier = ctx.Barrier(3)
        procs = [
            ctx.Process(
                target=_outer_grpc_sum_worker, args=(rank, 3, port, q, barrier)
            )
            for rank in range(3)
        ]
        procs[0].start()
        procs[0].join(timeout=0.5)
        for p in procs[1:]:
            p.start()
        for p in procs:
            p.join(timeout=60)
            self.assertEqual(p.exitcode, 0, f"rank process failed: {p.exitcode}")
        got = {q.get(timeout=5) for _ in range(3)}
        self.assertEqual({rank for rank, _ in got}, {0, 1, 2})
        self.assertTrue(all(abs(val - 3.0) < 1e-5 for _, val in got))


if __name__ == "__main__":
    unittest.main()
