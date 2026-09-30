"""Slurm frozen config path (per-run, not shared outputs/engine_frozen.json)."""

from __future__ import annotations

import os
import unittest

from src.omnifed.slurm_launcher import resolve_slurm_frozen_cfg_path


class TestResolveSlurmFrozenCfgPath(unittest.TestCase):
    def test_under_hydra_run_dir_not_shared_outputs_root(self) -> None:
        run_dir = "/tmp/outputs/2026-08-06_14-54-05/test_pi_centralized_sync_param_grpc"
        path = resolve_slurm_frozen_cfg_path(run_dir)
        self.assertEqual(path, os.path.join(run_dir, "engine_frozen.json"))
        self.assertNotEqual(os.path.dirname(path), os.path.dirname(os.path.dirname(run_dir)))

    def test_two_runs_get_distinct_paths(self) -> None:
        a = resolve_slurm_frozen_cfg_path("/tmp/outputs/2026-08-06_14-54-05/run_a")
        b = resolve_slurm_frozen_cfg_path("/tmp/outputs/2026-08-06_14-56-28/run_b")
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
