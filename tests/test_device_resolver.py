"""Classic Slurm device resolver: model vs agg, gRPC server vs trainer."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.omnifed.device_resolver import (
    is_classic_grpc_server,
    resolve_slurm_devices,
)
from src.omnifed.slurm_worker import _resolve_slurm_device


def _grpc_comm(*, is_server: bool, backend: str = "nccl"):
    return SimpleNamespace(is_server=is_server, backend=backend)


def _torchdist_comm(*, backend: str = "nccl"):
    return SimpleNamespace(backend=backend)


class TestIsClassicGrpcServer(unittest.TestCase):
    def test_grpc_rank0(self) -> None:
        self.assertTrue(is_classic_grpc_server(_grpc_comm(is_server=True)))

    def test_grpc_client(self) -> None:
        self.assertFalse(is_classic_grpc_server(_grpc_comm(is_server=False)))

    def test_torchdist_has_no_is_server(self) -> None:
        self.assertFalse(is_classic_grpc_server(_torchdist_comm()))


class TestResolveSlurmDevices(unittest.TestCase):
    def test_explicit_cpu_hint_both_devices(self) -> None:
        node = SimpleNamespace(device_hint="cpu")
        d = resolve_slurm_devices(
            node, rank=0, local_rank=0, local_comm=_grpc_comm(is_server=True)
        )
        self.assertEqual(d.model_device.type, "cpu")
        self.assertEqual(d.agg_device.type, "cpu")

    def test_explicit_cuda_hint(self) -> None:
        node = SimpleNamespace(device_hint="cuda:1")
        d = resolve_slurm_devices(
            node, rank=1, local_rank=0, local_comm=_grpc_comm(is_server=False)
        )
        self.assertEqual(str(d.model_device), "cuda:1")
        self.assertEqual(str(d.agg_device), "cuda:1")

    @patch("src.omnifed.device_resolver.torch.cuda.device_count", return_value=8)
    @patch("src.omnifed.device_resolver.torch.cuda.is_available", return_value=True)
    def test_grpc_server_auto_model_cpu_agg_gpu(self, *_mocks) -> None:
        node = SimpleNamespace(device_hint="auto")
        d = resolve_slurm_devices(
            node, rank=0, local_rank=0, local_comm=_grpc_comm(is_server=True)
        )
        self.assertEqual(d.model_device.type, "cpu")
        self.assertEqual(str(d.agg_device), "cuda:0")

    @patch("src.omnifed.device_resolver.torch.cuda.device_count", return_value=1)
    @patch("src.omnifed.device_resolver.torch.cuda.is_available", return_value=True)
    def test_grpc_client_stays_gpu(self, *_mocks) -> None:
        node = SimpleNamespace(device_hint="auto")
        d = resolve_slurm_devices(
            node, rank=3, local_rank=0, local_comm=_grpc_comm(is_server=False)
        )
        self.assertEqual(str(d.model_device), "cuda:0")
        self.assertEqual(str(d.agg_device), "cuda:0")

    @patch("src.omnifed.device_resolver.torch.cuda.device_count", return_value=8)
    @patch("src.omnifed.device_resolver.torch.cuda.is_available", return_value=True)
    def test_multi_gpu_node_uses_local_rank(self, *_mocks) -> None:
        node = SimpleNamespace(device_hint="auto")
        d = resolve_slurm_devices(
            node, rank=7, local_rank=7, local_comm=_grpc_comm(is_server=False)
        )
        self.assertEqual(str(d.model_device), "cuda:7")

    @patch("src.omnifed.device_resolver.torch.cuda.device_count", return_value=1)
    @patch("src.omnifed.device_resolver.torch.cuda.is_available", return_value=True)
    def test_torchdist_rank0_stays_gpu(self, *_mocks) -> None:
        node = SimpleNamespace(device_hint="auto")
        d = resolve_slurm_devices(
            node, rank=0, local_rank=0, local_comm=_torchdist_comm()
        )
        self.assertEqual(str(d.model_device), "cuda:0")
        self.assertEqual(str(d.agg_device), "cuda:0")

    @patch("src.omnifed.device_resolver.torch.cuda.is_available", return_value=False)
    def test_no_cuda_all_cpu(self, *_mocks) -> None:
        node = SimpleNamespace(device_hint="auto")
        d = resolve_slurm_devices(
            node, rank=1, local_rank=0, local_comm=_grpc_comm(is_server=False)
        )
        self.assertEqual(d.model_device.type, "cpu")
        self.assertEqual(d.agg_device.type, "cpu")

    def test_gloo_trainer_stays_cpu(self) -> None:
        node = SimpleNamespace(device_hint="auto")
        d = resolve_slurm_devices(
            node,
            rank=1,
            local_rank=0,
            local_comm=_grpc_comm(is_server=False, backend="gloo"),
        )
        self.assertEqual(d.model_device.type, "cpu")

    def test_slurm_worker_wrapper_returns_model_device(self) -> None:
        node = MagicMock()
        node.device_hint = "cpu"
        comm = MagicMock()
        comm.is_server = True
        comm.backend = "nccl"
        dev = _resolve_slurm_device(node, rank=0, local_rank=0, local_comm=comm)
        self.assertEqual(dev.type, "cpu")


if __name__ == "__main__":
    unittest.main()
