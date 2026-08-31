"""Unit tests for classic centralized per-round summary (``summary`` package)."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.omnifed.summary.slurm_per_round import write_classic_slurm_per_round_summary


class TestClassicRunSummary(unittest.TestCase):
    def test_build_and_write_round_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hydra = Path(tmp)
            nr = hydra / "engine" / "node_results"
            nr.mkdir(parents=True)

            (nr / "node_000_results.json").write_text(
                json.dumps({"role": "grpc_server", "rank": 0}),
                encoding="utf-8",
            )

            for gr in range(1, 4):
                node_dir = hydra / f"Node0.{gr}"
                node_dir.mkdir(parents=True)
                full = node_dir / "metrics_full.csv"
                with open(full, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "global_step",
                            "round_idx",
                            "epoch_idx",
                            "batch_idx",
                            "agg_ctx",
                            "metric_key",
                            "metric_val",
                        ]
                    )
                    w.writerow([100, 0, 0, 0, "sync", "local_agg_time", 0.05 + gr * 0.01])
                    w.writerow([101, 0, 0, 1, "sync", "local_agg_time", 0.03 + gr * 0.01])
                    w.writerow([200, 0, 0, 0, "sync", "grad_apply_time", 0.002])
                    w.writerow([300, 0, 0, 0, "eval", "eval/loss", 2.5 - gr * 0.1])
                    w.writerow([400, 0, 3, 10, "train", "train/loss", 1.2 - gr * 0.05])

                (nr / f"node_{gr:03d}_results.json").write_text(
                    json.dumps({"rank": gr}),
                    encoding="utf-8",
                )

            out = write_classic_slurm_per_round_summary(
                str(hydra),
                world_size=4,
                rpc_server_rank=0,
                rank_writer=1,
            )
            self.assertIsNotNone(out)

            csv_path = hydra / "classic_per_round_summary.csv"
            self.assertTrue(csv_path.is_file())
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["round_idx"], "0")
            self.assertEqual(rows[0]["grpc_agg_max_ms"], "80.00")
            self.assertEqual(rows[0]["n_eval_trainers"], "3")
            self.assertAlmostEqual(float(rows[0]["eval_loss_avg"]), 2.3, places=4)
            self.assertAlmostEqual(float(rows[0]["train_loss_avg"]), 1.1, places=4)


if __name__ == "__main__":
    unittest.main()
