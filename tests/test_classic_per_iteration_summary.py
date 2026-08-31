"""Unit tests for classic per-iteration CSV recorder."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from src.omnifed.summary.per_iteration import (
    ClassicIterationRecorder,
    accumulate_iter_comm,
    install_classic_iteration_recorder,
)


class TestClassicPerIterationSummary(unittest.TestCase):
    def test_accumulate_iter_comm(self) -> None:
        algo = SimpleNamespace(_summary_iter_comm={"grpc_upstream_s": 1.0})
        accumulate_iter_comm(algo, "grpc_upstream_s", 0.5)
        self.assertAlmostEqual(algo._summary_iter_comm["grpc_upstream_s"], 1.5)

    def test_accumulate_iter_comm_legacy_bucket(self) -> None:
        algo = SimpleNamespace(_classic_iter_comm={"grpc_upstream_s": 1.0})
        accumulate_iter_comm(algo, "grpc_upstream_s", 0.5)
        self.assertAlmostEqual(algo._classic_iter_comm["grpc_upstream_s"], 1.5)

    def test_record_after_batch_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = ClassicIterationRecorder(rank=2, log_dir=tmp)
            algo = SimpleNamespace(
                round_idx=0,
                epoch_idx=1,
                batch_idx=3,
                experiment_batch_idx=42,
                _summary_iter_batch_t0=None,
                _summary_iter_train={"loss": 1.25, "grad_norm": 0.5},
                _summary_iter_sync={
                    "grpc_agg_sample_time": 0.01,
                    "grpc_agg_grad_time": 1.2,
                    "grpc_agg_bn_time": 0.3,
                    "grad_apply_time": 0.001,
                },
                _summary_iter_comm={
                    "grpc_upstream_s": 0.8,
                    "grpc_downstream_s": 0.7,
                    "grpc_compress_s": 0.05,
                    "grpc_decompress_s": 0.02,
                },
            )
            algo.peek_metric = mock.Mock(
                side_effect=lambda key, agg_context=None: {
                    ("batch_time_data", "train"): 0.01,
                    ("batch_time_compute", "train"): 0.4,
                }.get((key, agg_context))
            )

            with mock.patch(
                "src.omnifed.summary.per_iteration_classic.get_gpu_memory_snapshot_mb",
                return_value={
                    "gpu_allocated_mb": 100.0,
                    "gpu_reserved_mb": 128.0,
                    "gpu_max_allocated_mb": 150.0,
                    "gpu_max_reserved_mb": 160.0,
                    "gpu_device_used_mb": 8192.0,
                    "gpu_device_total_mb": 65536.0,
                    "gpu_device_util_pct": 12.5,
                },
            ):
                recorder.record_after_batch(algo)

            csv_path = os.path.join(tmp, "rank2_classic_per_iteration_summary.csv")
            self.assertTrue(os.path.isfile(csv_path))
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["rank"], "2")
            self.assertEqual(row["batch_idx"], "3")
            self.assertEqual(row["train_loss"], "1.250000")
            self.assertEqual(row["grpc_agg_grad_s"], "1.200000")
            self.assertEqual(row["grpc_upstream_s"], "0.800000")
            self.assertAlmostEqual(float(row["sync_time_total_s"]), 1.511)
            self.assertEqual(row["gpu_device_used_mb"], "8192.000000")
            self.assertEqual(row["gpu_device_total_mb"], "65536.000000")
            self.assertEqual(row["gpu_device_util_pct"], "12.500000")

    def test_install_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            class Algo(SimpleNamespace):
                def _train_batch_start(self) -> None:
                    pass

                def _train_batch_end(self) -> None:
                    pass

            algo = Algo()
            install_classic_iteration_recorder(algo, rank=0, log_dir=tmp)
            self.assertTrue(hasattr(algo, "_summary_iteration_recorder"))
            algo._train_batch_start()
            self.assertIn("grpc_upstream_s", algo._summary_iter_comm)


if __name__ == "__main__":
    unittest.main()
