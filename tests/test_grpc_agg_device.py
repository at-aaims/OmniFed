"""gRPC agg_device: GPU SUM/decompress with CPU fallback."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch

from src.omnifed.communicator.base import AggregationOp
from src.omnifed.communicator.compression.sparsification import TopKCompression
from src.omnifed.communicator.grpc_server import GrpcServer
from src.omnifed.communicator.utils import (
    compress_message_tensors,
    compressor_proto_name,
    proto_to_tensordict_extended,
    tensordict_to_proto,
)
from src.omnifed.device_resolver import is_cuda_oom, mapping_to_device


class TestGrpcAggDevice(unittest.TestCase):
    def test_default_sum_stays_cpu(self) -> None:
        server = GrpcServer(world_size=2, communicate_params=False)
        sid = server.current_aggregation_session
        st = server.aggregation_state[sid]
        st["reduction_type"] = AggregationOp.SUM.value
        server._accumulate_into_session(st, "1", {"w": torch.tensor([1.0, 2.0])})
        server._accumulate_into_session(st, "2", {"w": torch.tensor([3.0, 4.0])})
        self.assertTrue(server.perform_aggregation_if_ready(st, sid))
        result = st["result"]["w"]
        self.assertEqual(result.device.type, "cpu")
        torch.testing.assert_close(result, torch.tensor([4.0, 6.0]))

    def test_proto_decode_default_is_cpu(self) -> None:
        compressor = TopKCompression(device="cpu", compress_ratio=0.5)
        grad = torch.tensor([1.0, 0.0, -2.0, 3.0])
        compressed = compress_message_tensors({"w": grad}, compressor, "grad")
        proto = tensordict_to_proto(compressed, compressor_proto_name(compressor))
        decoded, _ = proto_to_tensordict_extended(proto, overlay_base=None)
        self.assertEqual(decoded["w"].device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_sum_on_cuda_agg_device(self) -> None:
        server = GrpcServer(
            world_size=2, communicate_params=False, agg_device="cuda:0"
        )
        sid = server.current_aggregation_session
        st = server.aggregation_state[sid]
        st["reduction_type"] = AggregationOp.SUM.value
        server._accumulate_into_session(st, "1", {"w": torch.tensor([1.0, 2.0])})
        server._accumulate_into_session(st, "2", {"w": torch.tensor([3.0, 4.0])})
        self.assertTrue(server.perform_aggregation_if_ready(st, sid))
        result = st["result"]["w"]
        self.assertEqual(result.device.type, "cuda")
        torch.testing.assert_close(result.cpu(), torch.tensor([4.0, 6.0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_proto_decompress_on_cuda(self) -> None:
        compressor = TopKCompression(device="cpu", compress_ratio=0.5)
        grad = torch.tensor([1.0, 0.0, -2.0, 3.0])
        compressed = compress_message_tensors({"w": grad}, compressor, "grad")
        proto = tensordict_to_proto(compressed, compressor_proto_name(compressor))
        decoded, _ = proto_to_tensordict_extended(
            proto, overlay_base=None, compute_device=torch.device("cuda:0")
        )
        self.assertEqual(decoded["w"].device.type, "cuda")

    def test_oom_fallback_moves_accum_to_cpu(self) -> None:
        server = GrpcServer(world_size=2, communicate_params=False, agg_device="cpu")
        server._compute_device = torch.device("cuda", 0)
        sid = server.current_aggregation_session
        st = server.aggregation_state[sid]
        st["reduction_type"] = AggregationOp.SUM.value
        st["accum"] = {"w": torch.tensor([1.0])}
        st["participants"].add("1")

        def flaky_map(data, device):
            wanted = torch.device(device)
            if wanted.type == "cuda":
                raise RuntimeError("CUDA out of memory")
            return mapping_to_device(data, device)

        with patch(
            "src.omnifed.communicator.grpc_server.mapping_to_device",
            side_effect=flaky_map,
        ):
            server._accumulate_into_session(st, "2", {"w": torch.tensor([2.0])})
        self.assertEqual(server._compute_device.type, "cpu")
        torch.testing.assert_close(st["accum"]["w"], torch.tensor([3.0]))

    def test_is_cuda_oom_detects_hip(self) -> None:
        self.assertTrue(is_cuda_oom(RuntimeError("HIP out of memory")))
        self.assertFalse(is_cuda_oom(RuntimeError("file not found")))

    def test_communicator_set_agg_device_updates_servicer(self) -> None:
        from src.omnifed.communicator.grpc import GrpcCommunicator

        comm = GrpcCommunicator.__new__(GrpcCommunicator)
        comm._agg_device = torch.device("cpu")
        comm._client = None
        servicer = MagicMock()
        comm._servicer = servicer
        comm.set_agg_device("cpu")
        servicer.set_agg_device.assert_called_once()


if __name__ == "__main__":
    unittest.main()
