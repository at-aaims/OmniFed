"""TorchDistCommunicator grad-mode aggregation for classic sync grad."""

from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn as nn

from src.omnifed.communicator.base import AggregationOp
from src.omnifed.communicator.torchdist import TorchDistCommunicator


class _TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)


class TestTorchDistGradSync(unittest.TestCase):
    def test_grad_mode_all_reduces_grads_not_params(self) -> None:
        comm = TorchDistCommunicator(
            rank=0,
            world_size=2,
            master_port=29500,
            communicate_params=False,
        )
        model = _TinyLinear()
        model.lin.weight.grad = torch.ones_like(model.lin.weight)
        model.lin.bias.grad = torch.full_like(model.lin.bias, 2.0)
        weight_before = model.lin.weight.detach().clone()
        bias_before = model.lin.bias.detach().clone()

        reduced: list[torch.Tensor] = []

        def fake_all_reduce(tensor, op=None, group=None, async_op=False):
            del op, group, async_op
            reduced.append(tensor.detach().clone())
            tensor.mul_(2.0)

        with mock.patch("src.omnifed.communicator.torchdist.dist.all_reduce", fake_all_reduce):
            comm.aggregate(model, AggregationOp.SUM)

        self.assertEqual(len(reduced), 2)
        torch.testing.assert_close(model.lin.weight, weight_before)
        torch.testing.assert_close(model.lin.bias, bias_before)
        torch.testing.assert_close(model.lin.weight.grad, torch.full_like(model.lin.weight, 2.0))
        torch.testing.assert_close(model.lin.bias.grad, torch.full_like(model.lin.bias, 4.0))

    def test_param_mode_all_reduces_weights_not_grads(self) -> None:
        comm = TorchDistCommunicator(
            rank=0,
            world_size=2,
            master_port=29501,
            communicate_params=True,
        )
        model = _TinyLinear()
        model.lin.weight.data.fill_(1.0)
        model.lin.bias.data.fill_(2.0)

        with mock.patch("src.omnifed.communicator.torchdist.dist.all_reduce") as mock_reduce:
            comm.aggregate(model, AggregationOp.SUM)

        reduced_tensors = [call.args[0] for call in mock_reduce.call_args_list]
        self.assertEqual(len(reduced_tensors), 2)
        reduced_shapes = {tuple(t.shape) for t in reduced_tensors}
        self.assertEqual(
            reduced_shapes,
            {tuple(model.lin.weight.shape), tuple(model.lin.bias.shape)},
        )

    def test_set_aggregation_num_samples_api(self) -> None:
        comm = TorchDistCommunicator(rank=0, world_size=1, master_port=29502)
        comm.set_aggregation_num_samples(128)
        self.assertEqual(comm._aggregation_num_samples, 128)


if __name__ == "__main__":
    unittest.main()
