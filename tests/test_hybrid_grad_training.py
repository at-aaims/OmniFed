"""Unit tests for hybrid gradient training helpers (Phase 3)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from src.omnifed.hybrid.hybrid_grad_training import (
    apply_optimizer_grads,
    clear_model_grads,
    grad_l2_norm,
    require_model_grads,
)


class TestHybridGradTraining(unittest.TestCase):
    def test_apply_optimizer_grads_updates_weights(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.ones(1, 2)
        model(x).sum().backward()
        w_before = model.weight.detach().clone()
        apply_optimizer_grads(model, opt)
        self.assertFalse(torch.allclose(model.weight, w_before))

    def test_require_model_grads_raises_when_empty(self) -> None:
        model = nn.Linear(2, 1)
        with self.assertRaises(RuntimeError):
            require_model_grads(model)

    def test_grad_l2_norm_after_backward(self) -> None:
        model = nn.Linear(3, 2)
        model(torch.randn(4, 3)).sum().backward()
        self.assertGreater(grad_l2_norm(model), 0.0)

    def test_clear_model_grads(self) -> None:
        model = nn.Linear(2, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.randn(1, 2)).sum().backward()
        clear_model_grads(model, optimizer=opt)
        self.assertIsNone(model.weight.grad)


if __name__ == "__main__":
    unittest.main()
