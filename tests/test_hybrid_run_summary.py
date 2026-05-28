"""Unit tests for ``hybrid_run_summary`` (aggregated table from JSON node_results)."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from src.omnifed.hybrid.hybrid_run_summary import (
    _sync_metric_seconds,
    write_hybrid_slurm_per_round_summary,
)
from src.omnifed.hybrid.topology_builder import build_hybrid_topology


class TestHybridRunSummary(unittest.TestCase):
    def test_sync_metric_seconds_accepts_metriclogger_keys(self) -> None:
        row = {"round_idx": 0, "sync/global_agg_time": 1.234, "noise": 0}
        self.assertAlmostEqual(_sync_metric_seconds(row, "global_agg_time"), 1.234)

    def test_sync_metric_fallback_bare_key(self) -> None:
        row = {"global_agg_time": 0.5}
        self.assertAlmostEqual(_sync_metric_seconds(row, "global_agg_time"), 0.5)

    def test_build_and_write_round_summary(self) -> None:
        topo = OmegaConf.create(
            build_hybrid_topology(num_facilities=2, mpi_ranks_per_facility=3)
        )

        with tempfile.TemporaryDirectory() as tmp:
            hydra = Path(tmp)
            nr = hydra / "engine" / "node_results"
            nr.mkdir(parents=True)

            stub = {"role": "hybrid_grpc_server", "rank": 0}
            (nr / "node_000_results.json").write_text(
                json.dumps(stub), encoding="utf-8"
            )

            # Topology 2×3: fac1 members [1,3,4] leader 1; fac2 members [2,5,6] leader 2
            def trainer_payload(rank: int) -> dict:
                la = {1: 0.001, 3: 0.003, 4: 0.002, 2: 0.01, 5: 0.02, 6: 0.015}
                lb = {1: 0.004, 3: 0.005, 4: 0.004, 2: 0.008, 5: 0.009, 6: 0.007}
                ga = {1: 0.07, 2: 0.09}
                # Match Slurm JSON: durations under ``sync/...`` keys
                tm = dict(
                    local_agg_time=la[rank],
                    local_bcast_time=lb[rank],
                )
                if rank in ga:
                    tm["global_agg_time"] = ga[rank]
                sync_row = {"round_idx": 0.0}
                sync_row.update({f"sync/{k}": v for k, v in tm.items()})
                sync = [sync_row]
                ev = [{"round_idx": 0.0, "accuracy": 0.5 + rank * 0.01, "eval/loss": 0.42}]
                return {"rank": rank, "sync": sync, "eval": ev}

            for r in range(1, 7):
                (nr / f"node_{r:03d}_results.json").write_text(
                    json.dumps(trainer_payload(r)), encoding="utf-8"
                )

            os.environ["OMNIFED_HYBRID_SUMMARY_POLL_SEC"] = "2"
            os.environ["OMNIFED_HYBRID_SUMMARY_POLL_GAP_SEC"] = "0.01"

            out = write_hybrid_slurm_per_round_summary(
                str(hydra),
                topo=topo,
                world_size=7,
                rpc_server_rank=0,
                rank_writer=1,
            )
            self.assertIsNotNone(out)
            root_csv = hydra / "hybrid_per_round_summary.csv"
            eng_csv = hydra / "engine" / "hybrid_per_round_summary.csv"
            self.assertTrue(root_csv.is_file())
            self.assertEqual(
                root_csv.read_text(encoding="utf-8"),
                eng_csv.read_text(encoding="utf-8"),
            )
            txt = (hydra / "engine" / "hybrid_per_round_summary.txt").read_text(encoding="utf-8")
            self.assertIn("gRPC_F1_ms", txt)
            self.assertIn("| 0 |", txt)
            # 70 ms / 90 ms leaders
            self.assertIn("70.00", txt)
            self.assertIn("90.00", txt)
            # max local_agg: F1 max(1,3,2) ms = 3ms; F2 max(10,20,15)=20ms
            self.assertIn("3.00", txt)
            self.assertIn("20.00", txt)
            # max local_bcast: F1 max(4,5,4)=5ms; F2 max(8,9,7)=9ms
            self.assertIn("5.00", txt)
            self.assertIn("9.00", txt)
            # accuracy mean across 6 trainers
            av = sum(0.5 + r * 0.01 for r in range(1, 7)) / 6.0
            self.assertIn(f"{av:.4f}", txt)
            self.assertIn("eval_loss_avg", txt)
            self.assertIn("0.420000", txt)
