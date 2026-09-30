"""Grad-mode gRPC must not aggregate BN buffers with parameter gradients."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from src.omnifed.communicator.grpc import GrpcCommunicator


class _TinyBn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.lin(x))


class TestGrpcGradBufferExclusion(unittest.TestCase):
    def test_extract_skips_buffers_when_communicating_grads(self) -> None:
        comm = GrpcCommunicator(
            rank=1,
            world_size=2,
            master_port=50051,
            communicate_params=False,
        )
        model = _TinyBn()
        model.lin.weight.grad = torch.ones_like(model.lin.weight)

        extracted = comm._extract_tensordict_from_msg(model)

        self.assertIn("lin.weight", extracted)
        self.assertNotIn("bn.running_mean", extracted)
        self.assertNotIn("bn.running_var", extracted)

    def test_extract_includes_buffers_for_param_fedavg(self) -> None:
        comm = GrpcCommunicator(
            rank=1,
            world_size=2,
            master_port=50052,
            communicate_params=True,
        )
        model = _TinyBn()

        extracted = comm._extract_tensordict_from_msg(model)

        self.assertIn("lin.weight", extracted)
        self.assertIn("bn.running_mean", extracted)
        self.assertIn("bn.running_var", extracted)


if __name__ == "__main__":
    unittest.main()
