"""Classic grad/param slurm filenames are gone; one worker helper remains."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.omnifed.execution.slurm.worker_helpers import install_round_end_eval


class TestRoundEndEvalHelper(unittest.TestCase):
    def test_install_wraps_round_end_eval(self) -> None:
        calls: list[str] = []

        class Algo(SimpleNamespace):
            progress_info_str = "t"
            datamodule = SimpleNamespace(eval=object())
            local_model = object()

            def _round_end(self) -> None:
                calls.append("base")

            def _BaseAlgorithm__eval_epoch(self, model) -> None:
                calls.append("eval")

        algo = Algo()
        install_round_end_eval(algo, local_comm=mock.Mock())
        algo._round_end()
        self.assertEqual(calls, ["eval", "base"])


if __name__ == "__main__":
    unittest.main()
