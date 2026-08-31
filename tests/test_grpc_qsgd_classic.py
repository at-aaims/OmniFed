"""Classic gRPC QSGD roundtrip for sync grad aggregation."""

from __future__ import annotations

import unittest

import torch

from src.omnifed.communicator.base import AggregationOp
from src.omnifed.communicator.compression.quantization import QSGDQuantCompression
from src.omnifed.communicator.grpc_server import GrpcServer
from src.omnifed.communicator.utils import (
    compress_message_tensors,
    compressor_proto_name,
    proto_to_tensordict_extended,
    tensordict_to_proto,
)


class TestGrpcQsgdClassicSyncGrad(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.compressor = QSGDQuantCompression(device="cpu", bit_width=4)

    def test_proto_roundtrip_per_tensor(self) -> None:
        grad = torch.tensor([1.0, -2.0, 0.5, 3.0])
        compressed = compress_message_tensors(
            {"w": grad}, self.compressor, "grad"
        )
        proto = tensordict_to_proto(compressed, compressor_proto_name(self.compressor))
        decoded, _ = proto_to_tensordict_extended(proto, overlay_base=None)
        self.assertEqual(decoded["w"].shape, grad.shape)
        self.assertTrue(torch.isfinite(decoded["w"]).all())

    def test_uplink_proto_roundtrip_then_server_sum(self) -> None:
        g1 = torch.tensor([1.0, 0.0, -2.0, 3.0])
        g2 = torch.tensor([0.5, 1.5, 2.0, -1.0])
        dense_clients = []
        for grad in (g1, g2):
            compressed = compress_message_tensors(
                {"w": grad}, self.compressor, "grad"
            )
            proto = tensordict_to_proto(
                compressed, compressor_proto_name(self.compressor)
            )
            decoded, _ = proto_to_tensordict_extended(proto, overlay_base=None)
            dense_clients.append(decoded["w"])

        summed = dense_clients[0] + dense_clients[1]
        server = GrpcServer(world_size=3, communicate_params=False)
        session_id = server.current_aggregation_session
        session_state = server.aggregation_state[session_id]
        session_state["reduction_type"] = AggregationOp.SUM.value
        session_state["data"] = {
            "server": {"w": torch.zeros_like(g1)},
            "1": {"w": dense_clients[0]},
            "2": {"w": dense_clients[1]},
        }
        self.assertTrue(
            server.perform_aggregation_if_ready(session_state, session_id)
        )
        torch.testing.assert_close(session_state["result"]["w"], summed)

    def test_dense_payload_when_compressor_none(self) -> None:
        bn = torch.tensor([0.1, 0.2, 0.3, 0.4])
        payload = {"running_mean": bn}
        dense = compress_message_tensors(payload, None, "grad")
        self.assertIs(dense["running_mean"], bn)

    def test_decompress_quantized_inverse(self) -> None:
        grad = torch.randn(8)
        signed, norm, width, levels = self.compressor.compress(grad.clone(), "w")
        self.assertNotEqual(width, -1)
        restored = QSGDQuantCompression.decompress_quantized(
            signed, norm, levels, grad.shape
        )
        self.assertEqual(restored.shape, grad.shape)
        self.assertTrue(torch.isfinite(restored).all())


if __name__ == "__main__":
    unittest.main()
