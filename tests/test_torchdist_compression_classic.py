"""TorchDist Top-K / QSGD aggregation (classic sync grad)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from src.omnifed.communicator.base import AggregationOp
from src.omnifed.communicator.compression.quantization import QSGDQuantCompression
from src.omnifed.communicator.compression.sparsification import TopKCompression
from src.omnifed.communicator.torchdist import TorchDistCommunicator
from src.omnifed.communicator.compression.torchdist_collectives import (
    aggregate_qsgd_tensor,
    aggregate_topk_tensor,
    all_gather_sparse_sum,
)


class TestAllGatherSparseSum(unittest.TestCase):
    def test_sums_two_rank_payloads(self) -> None:
        calls: list[int] = []

        def fake_all_gather(out_list, inp):
            step = len(calls)
            calls.append(step)
            if inp.numel() == 1:
                out_list[0].copy_(torch.tensor([2], dtype=torch.long))
                out_list[1].copy_(torch.tensor([2], dtype=torch.long))
                return
            if inp.dtype == torch.long:
                out_list[0].copy_(torch.tensor([0, 2], dtype=torch.long))
                out_list[1].copy_(torch.tensor([1, 0], dtype=torch.long))
            else:
                out_list[0].copy_(torch.tensor([1.0, 3.0]))
                out_list[1].copy_(torch.tensor([2.0, 4.0]))

        with mock.patch(
            "src.omnifed.communicator.compression.torchdist_collectives.dist.all_gather",
            side_effect=fake_all_gather,
        ):
            out = all_gather_sparse_sum(
                torch.tensor([1.0, 3.0]),
                torch.tensor([0, 2], dtype=torch.long),
                numel=4,
                world_size=2,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        expected = torch.tensor([5.0, 2.0, 3.0, 0.0])
        torch.testing.assert_close(out, expected)


class TestTorchDistCompressionAggregate(unittest.TestCase):
    def test_topk_skips_when_num_samples_zero(self) -> None:
        comm = TorchDistCommunicator(
            rank=0,
            world_size=2,
            master_port=29610,
            communicate_params=False,
            compressor=TopKCompression(device="cpu", compress_ratio=0.5),
        )
        comm.set_aggregation_num_samples(0)
        t = torch.tensor([1.0, 2.0])
        with mock.patch(
            "src.omnifed.communicator.torchdist.dist.all_reduce",
            side_effect=lambda tensor, **kw: tensor.mul_(2.0),
        ) as mock_reduce:
            comm.aggregate(t, AggregationOp.SUM)
            self.assertEqual(mock_reduce.call_count, 1)
            torch.testing.assert_close(t, torch.tensor([2.0, 4.0]))

    def test_topk_compresses_when_num_samples_positive(self) -> None:
        comm = TorchDistCommunicator(
            rank=0,
            world_size=2,
            master_port=29611,
            communicate_params=False,
            compressor=TopKCompression(device="cpu", compress_ratio=0.5),
        )
        comm.set_aggregation_num_samples(8)
        model = nn.Linear(4, 1, bias=False)
        model.weight.grad = torch.tensor([[4.0, 0.0, -1.0, 2.0]])

        with mock.patch(
            "src.omnifed.communicator.torchdist.aggregate_topk_tensor",
            return_value=torch.ones(1, 4),
        ) as mock_topk:
            comm.aggregate(model, AggregationOp.SUM)
            mock_topk.assert_called_once()
            torch.testing.assert_close(model.weight.grad, torch.ones(1, 4))

    def test_topk_compresses_params_when_communicate_params_true(self) -> None:
        comm = TorchDistCommunicator(
            rank=0,
            world_size=2,
            master_port=29612,
            communicate_params=True,
            compressor=TopKCompression(device="cpu", compress_ratio=0.5),
        )
        comm.set_aggregation_num_samples(8)
        model = nn.Linear(4, 1, bias=False)
        model.weight.data.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

        with mock.patch(
            "src.omnifed.communicator.torchdist.aggregate_topk_tensor",
            return_value=torch.full((1, 4), 0.5),
        ) as mock_topk:
            comm.aggregate(model, AggregationOp.SUM)
            mock_topk.assert_called_once()
            torch.testing.assert_close(model.weight.data, torch.full((1, 4), 0.5))

    def test_qsgd_decompress_then_all_reduce(self) -> None:
        torch.manual_seed(0)
        compressor = QSGDQuantCompression(device="cpu", bit_width=4)
        grad = torch.tensor([1.0, -2.0, 0.5, 3.0])
        signed, norm, width, levels = compressor.compress(grad.clone(), "w")
        local = QSGDQuantCompression.decompress_quantized(
            signed, norm, levels, grad.shape
        )

        def fake_all_reduce(tensor, op=None, group=None, async_op=False):
            del op, group, async_op
            tensor.copy_(local * 2.0)

        with mock.patch(
            "src.omnifed.communicator.compression.torchdist_collectives.dist.all_reduce",
            side_effect=fake_all_reduce,
        ):
            out = aggregate_qsgd_tensor(
                compressor,
                grad.clone(),
                name="w",
                op=mock.Mock(),
                logger=None,
            )
        torch.testing.assert_close(out, local * 2.0)

    def test_compress_decompress_timings_recorded(self) -> None:
        compressor = TopKCompression(device="cpu", compress_ratio=0.5)
        logger = SimpleNamespace(_summary_iter_comm={})
        grad = torch.tensor([1.0, -2.0, 0.0, 4.0])

        with mock.patch(
            "src.omnifed.communicator.compression.torchdist_collectives.dist.all_gather",
            side_effect=lambda out_list, inp: [
                o.copy_(inp if i == 0 else inp) for i, o in enumerate(out_list)
            ],
        ):
            aggregate_topk_tensor(
                compressor,
                grad,
                name="w",
                world_size=1,
                logger=logger,
            )

        self.assertIn("grpc_compress_s", logger._summary_iter_comm)
        self.assertIn("grpc_decompress_s", logger._summary_iter_comm)
        self.assertGreater(logger._summary_iter_comm["grpc_compress_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
