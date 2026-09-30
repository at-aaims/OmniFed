"""Classic gRPC Top-K roundtrip for sync grad aggregation."""

from __future__ import annotations

import unittest

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


class TestGrpcTopkClassicSyncGrad(unittest.TestCase):
    def setUp(self) -> None:
        self.compressor = TopKCompression(device="cpu", compress_ratio=0.25)

    def test_zero_base_decode_not_local_grad_overlay(self) -> None:
        local_grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
        (values, indices), ctx = self.compressor.compress(local_grad.clone(), "w")
        compressed = {
            "w": {
                "values": values,
                "indices": indices,
                "original_shape": local_grad.shape,
                "ctx": ctx,
            }
        }
        proto = tensordict_to_proto(compressed, compressor_proto_name(self.compressor))

        decoded, _ = proto_to_tensordict_extended(proto, overlay_base=None)
        wrong, _ = proto_to_tensordict_extended(proto, overlay_base={"w": local_grad})

        expected = self.compressor.decompress((values, indices), ctx)
        torch.testing.assert_close(decoded["w"], expected)
        self.assertFalse(torch.allclose(decoded["w"], wrong["w"]))

    def test_uplink_proto_roundtrip_then_server_sum(self) -> None:
        g1 = torch.tensor([1.0, 0.0, -2.0, 3.0])
        g2 = torch.tensor([0.5, 1.5, 2.0, -1.0])
        grads = {"w": g1}, {"w": g2}

        dense_clients = []
        for grad in grads:
            compressed = compress_message_tensors(grad, self.compressor, "grad")
            proto = tensordict_to_proto(compressed, compressor_proto_name(self.compressor))
            decoded, _ = proto_to_tensordict_extended(proto, overlay_base=None)
            dense_clients.append(decoded["w"])

        summed = dense_clients[0] + dense_clients[1]
        self.assertEqual(summed.shape, g1.shape)

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

    def test_downlink_proto_roundtrip(self) -> None:
        aggregated = torch.tensor([1.5, 1.5, 0.0, 2.0])
        compressed = compress_message_tensors(
            {"w": aggregated}, self.compressor, "grad"
        )
        proto = tensordict_to_proto(compressed, compressor_proto_name(self.compressor))
        decoded, _ = proto_to_tensordict_extended(proto, overlay_base=None)
        self.assertEqual(decoded["w"].shape, aggregated.shape)
        nnz = (decoded["w"] != 0).sum().item()
        self.assertGreater(nnz, 0)
        self.assertLess(nnz, aggregated.numel())

    def test_dense_payload_when_compressor_none(self) -> None:
        """BN / sample-count aggs pass compressor=None (num_samples=0 on wire)."""
        bn = torch.tensor([0.1, 0.2, 0.3, 0.4])
        payload = {"running_mean": bn}
        dense = compress_message_tensors(payload, None, "grad")
        self.assertIs(dense["running_mean"], bn)

    def test_topk_stays_on_input_tensor_device(self) -> None:
        """yaml device=cpu must not move a GPU (or CPU) tensor before torch.topk."""
        cpu_grad = torch.tensor([1.0, 0.0, -2.0, 3.0])
        (values, indices), ctx = self.compressor.compress(cpu_grad.clone(), "w_cpu")
        self.assertEqual(values.device.type, "cpu")
        restored = self.compressor.decompress((values, indices), ctx)
        self.assertEqual(restored.device.type, "cpu")

        if not torch.cuda.is_available():
            return
        gpu_grad = cpu_grad.cuda()
        (gvals, gix), gctx = self.compressor.compress(gpu_grad.clone(), "w_gpu")
        self.assertEqual(gvals.device.type, "cuda")
        restored_gpu = self.compressor.decompress((gvals, gix), gctx)
        self.assertEqual(restored_gpu.device.type, "cuda")
        proto = tensordict_to_proto(
            {
                "w": {
                    "values": gvals,
                    "indices": gix,
                    "original_shape": gpu_grad.shape,
                    "ctx": gctx,
                }
            },
            compressor_proto_name(self.compressor),
        )
        decoded, _ = proto_to_tensordict_extended(proto, overlay_base=None)
        self.assertEqual(decoded["w"].device.type, "cpu")

    def test_compress_cuda_oom_retries_on_cpu(self) -> None:
        if not torch.cuda.is_available():
            return
        compressor = TopKCompression(device="cpu", compress_ratio=0.5)
        gpu_grad = torch.tensor([1.0, 0.0, -2.0, 3.0], device="cuda")
        calls = {"n": 0}

        def flaky_topk(tensor, compress_ratio):
            del compress_ratio
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            k = max(1, int(tensor.numel() * 0.5))
            _, indices = torch.topk(tensor.abs().flatten(), k, sorted=False)
            values = torch.gather(tensor.flatten(), 0, indices)
            return values, indices

        from unittest import mock

        with mock.patch(
            "src.omnifed.communicator.compression.sparsification.topk_sparse",
            side_effect=flaky_topk,
        ):
            (values, indices), ctx = compressor.compress(gpu_grad.clone(), "w_oom")
        self.assertEqual(values.device.type, "cpu")
        restored = compressor.decompress((values, indices), ctx)
        self.assertEqual(restored.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
