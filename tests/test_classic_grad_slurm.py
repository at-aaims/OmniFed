"""Regression: classic synchronous grad hooks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.omnifed.classic.classic_grad_slurm import install_classic_grad_slurm_sync


class TestClassicGradSlurmHooks(unittest.TestCase):
    def test_round_start_calls_base_once(self) -> None:
        calls: list[str] = []

        class Algo(SimpleNamespace):
            def _base_round_start(self) -> None:
                calls.append("base")

        algo = Algo()
        algo._round_start = algo._base_round_start
        algo._BaseAlgorithm__local_optimizer = mock.Mock()
        algo.local_comm = mock.Mock()

        install_classic_grad_slurm_sync(algo, local_comm=algo.local_comm)
        algo._round_start()

        self.assertEqual(calls, ["base"])
        algo._BaseAlgorithm__local_optimizer.zero_grad.assert_called_once()


if __name__ == "__main__":
    unittest.main()
