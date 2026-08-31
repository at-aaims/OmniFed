"""Slurm multi-node gRPC startup: Lustre ready-marker helpers."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from src.omnifed.slurm_worker import (
    _grpc_server_ready_marker,
    _resolve_slurm_device,
    _wait_for_grpc_server_ready_marker,
    _write_grpc_server_ready_marker,
)


class TestSlurmGrpcReadyMarker(unittest.TestCase):
    def test_marker_path_under_engine(self) -> None:
        path = _grpc_server_ready_marker("/tmp/run")
        self.assertTrue(path.endswith(os.path.join("engine", ".grpc_server_ready")))

    def test_wait_returns_when_marker_appears(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = _grpc_server_ready_marker(tmp)

            def _writer() -> None:
                time.sleep(0.3)
                _write_grpc_server_ready_marker(
                    tmp, master_addr="host1", master_port="50051"
                )

            t = threading.Thread(target=_writer, daemon=True)
            t.start()
            _wait_for_grpc_server_ready_marker(
                tmp, rank=3, timeout_s=5.0, poll_s=0.05
            )
            self.assertTrue(os.path.isfile(marker))
            t.join(timeout=2.0)


class TestResolveSlurmDevice(unittest.TestCase):
    def _grpc_server_comm(self) -> MagicMock:
        comm = MagicMock()
        comm.is_server = True
        comm.backend = "nccl"
        comm.master_port = 50051
        return comm

    def test_grpc_server_rank0_explicit_cpu_hint(self) -> None:
        node = MagicMock()
        node.device_hint = "cpu"
        dev = _resolve_slurm_device(
            node, rank=0, local_rank=0, local_comm=self._grpc_server_comm()
        )
        self.assertEqual(dev.type, "cpu")

    def test_explicit_device_hint_cuda(self) -> None:
        node = MagicMock()
        node.device_hint = "cuda:1"
        dev = _resolve_slurm_device(
            node, rank=1, local_rank=0, local_comm=self._grpc_server_comm()
        )
        self.assertEqual(str(dev), "cuda:1")

    def test_grpc_server_auto_model_stays_cpu(self) -> None:
        node = MagicMock()
        node.device_hint = "auto"
        comm = self._grpc_server_comm()
        comm.is_server = True
        comm.backend = "nccl"
        with patch(
            "src.omnifed.device_resolver.torch.cuda.is_available", return_value=True
        ), patch(
            "src.omnifed.device_resolver.torch.cuda.device_count", return_value=1
        ):
            dev = _resolve_slurm_device(
                node, rank=0, local_rank=0, local_comm=comm
            )
        self.assertEqual(dev.type, "cpu")


if __name__ == "__main__":
    unittest.main()
